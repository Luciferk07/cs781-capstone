"""sum_model.py — Phase-6 model."""
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from phase1_loader import load_phase1

BLOCKS = {"prefixlm": "blocks", "encdec": "encoder", "arch4": "blocks", "biscale": "blocks"}
FINAL_NORM = {"prefixlm": "ln_f", "encdec": "enc_ln", "arch4": "ln_f", "biscale": "norm"}
DEFAULT_TARGETS = {
    "prefixlm": ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"],
    "encdec": ["attn.qkv", "attn.proj", "cross.q", "cross.kv", "cross.proj", "mlp.fc1", "mlp.fc2"],
    "arch4": ["msr.proj", "msr.out", "mlp.fc1", "mlp.fc2"],
    "biscale": ["attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.o_proj", "ffn.gate", "ffn.up", "ffn.down"],
}
ADAPTER_SITES = {
    "prefixlm": ("attn", "mlp"), "encdec": ("attn", "mlp"),
    "arch4": ("msr", "mlp"), "biscale": ("attn", "ffn"),
}


def _first_attr(obj, names):
    for n in names:
        if getattr(obj, n, None) is not None:
            return getattr(obj, n)
    raise AttributeError(f"{type(obj).__name__} has none of {names}")


def _out_splits(backbone_name, path, lin, d_model):
    if path.endswith("attn.qkv"):
        return [lin.out_features // 3] * 3
    if path.endswith("cross.kv"):
        return [lin.out_features // 2] * 2
    if backbone_name == "arch4" and path.endswith("msr.proj"):
        vd = (lin.out_features - 2 * d_model) // 2
        return [d_model, d_model, vd, vd]
    return [lin.out_features]


class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha, dropout):
        super().__init__()
        self.base = base
        self.scale = alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(self.drop(x), self.A), self.B) * self.scale


class LoRMALinear(nn.Module):
    def __init__(self, base, r, alpha, splits, variant):
        super().__init__()
        self.base, self.splits, self.variant = base, list(splits), variant
        self.scale = alpha / r
        self.A = nn.ParameterList()
        self.B = nn.ParameterList()
        for d in self.splits:
            if variant == "plus":
                A = torch.empty(r, d)
                nn.init.kaiming_uniform_(A, a=math.sqrt(5))
                B = torch.zeros(d, r)
            elif variant == "pi":
                A = torch.zeros(r, d)
                A[0, 0] = 1.0
                B = torch.randn(d, r) / math.sqrt(r)
                B[:, 0] = 1.0
            else:
                raise ValueError(f"lorma_variant must be 'plus' or 'pi', got {variant!r}")
            self.A.append(nn.Parameter(A))
            self.B.append(nn.Parameter(B))

    def forward(self, x):
        y = F.linear(x, self.base.weight)
        outs = []
        for i, part in enumerate(y.split(self.splits, dim=-1)):
            A, B = self.A[i], self.B[i]
            if self.variant == "plus":
                outs.append(part + F.linear(F.linear(part, A.to(part.dtype)), B.to(part.dtype)) * self.scale)
            else:
                M = B @ A
                d = M.shape[0]
                idx = torch.arange(d, device=M.device)
                R = torch.gather(M, 1, (idx[None, :] - idx[:, None]) % d)
                outs.append(F.linear(part, R.to(part.dtype)))
        y = torch.cat(outs, dim=-1) if len(outs) > 1 else outs[0]
        if self.base.bias is not None:
            y = y + self.base.bias.to(y.dtype)
        return y


class Adapter(nn.Module):
    def __init__(self, d, m):
        super().__init__()
        self.down = nn.Linear(d, m)
        self.up = nn.Linear(m, d)
        nn.init.normal_(self.down.weight, 0.0, 0.02)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, h):
        return h + self.up(F.gelu(self.down(h)))


class WithAdapter(nn.Module):
    def __init__(self, inner, d, m):
        super().__init__()
        self.inner = inner
        self.adapter = Adapter(d, m)

    def _adapt(self, out):
        if isinstance(out, tuple):
            return (self.adapter(out[0]),) + tuple(out[1:])
        return self.adapter(out)

    def forward(self, *args, **kwargs):
        return self._adapt(self.inner(*args, **kwargs))

    def step(self, *args, **kwargs):
        return self._adapt(self.inner.step(*args, **kwargs))

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._modules["inner"], name)


class AttnContext:
    def __init__(self):
        self.key_mask = None
        self.capture = False


class KVStore:
    def __init__(self, n_layers, max_len):
        self.k, self.v = [None] * n_layers, [None] * n_layers
        self.max_len, self.pos = max_len, 0


class PrefixLMAttention(nn.Module):
    def __init__(self, orig, ctx, rope_fn):
        super().__init__()
        self.orig, self.ctx, self.rope_fn = orig, ctx, rope_fn
        self.pk = self.pv = None

    def add_prefix(self, n):
        o = self.orig
        self.pk = nn.Parameter(torch.zeros(o.h, n, o.hd))
        self.pv = nn.Parameter(torch.zeros(o.h, n, o.hd))

    def forward(self, x, cos, sin, mask, kv=None, start=0):
        o = self.orig
        B, T, _ = x.shape
        q, k, v = o.qkv(x).view(B, T, 3, o.h, o.hd).permute(2, 0, 3, 1, 4)
        if o.rope:
            q, k = self.rope_fn(q, cos.to(q.dtype), sin.to(q.dtype)), self.rope_fn(k, cos.to(q.dtype), sin.to(q.dtype))
        if self.pk is not None and self.ctx.capture:
            self.pk.data.copy_(k[0].detach().float())
            self.pv.data.copy_(v[0].detach().float())
        if kv is not None:
            store, li = kv
            if store.k[li] is None:
                shape = (B, o.h, store.max_len, o.hd)
                store.k[li] = torch.empty(shape, device=k.device, dtype=k.dtype)
                store.v[li] = torch.empty(shape, device=v.device, dtype=v.dtype)
            store.k[li][:, :, start:start + T] = k
            store.v[li][:, :, start:start + T] = v
            k, v = store.k[li][:, :, :start + T], store.v[li][:, :, :start + T]
        if self.pk is not None and not self.ctx.capture:
            n = self.pk.shape[1]
            k = torch.cat([self.pk.to(k.dtype)[None].expand(B, -1, -1, -1), k], dim=2)
            v = torch.cat([self.pv.to(v.dtype)[None].expand(B, -1, -1, -1), v], dim=2)
            if mask is not None:
                mask = torch.cat([mask.new_ones(mask.shape[0], 1, T, n), mask.expand(-1, -1, T, -1)], dim=-1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                           dropout_p=o.p_drop if self.training else 0.0)
        return o.proj(y.transpose(1, 2).reshape(B, T, o.d))


class EncoderAttention(nn.Module):
    def __init__(self, orig, ctx, rope_fn):
        super().__init__()
        self.orig, self.ctx, self.rope_fn = orig, ctx, rope_fn
        self.pk = self.pv = None

    def add_prefix(self, n):
        o = self.orig
        self.pk = nn.Parameter(torch.zeros(o.h, n, o.hd))
        self.pv = nn.Parameter(torch.zeros(o.h, n, o.hd))

    def forward(self, x, cos, sin, causal=False, cache=None, li=None, start=0):
        o = self.orig
        B, T, _ = x.shape
        q, k, v = o.qkv(x).view(B, T, 3, o.h, o.hd).permute(2, 0, 3, 1, 4)
        q, k = self.rope_fn(q, cos, sin), self.rope_fn(k, cos, sin)
        km = self.ctx.key_mask
        if self.pk is not None and self.ctx.capture:
            self.pk.data.copy_(k[0].detach().float())
            self.pv.data.copy_(v[0].detach().float())
        elif self.pk is not None:
            n = self.pk.shape[1]
            k = torch.cat([self.pk.to(k.dtype)[None].expand(B, -1, -1, -1), k], dim=2)
            v = torch.cat([self.pv.to(v.dtype)[None].expand(B, -1, -1, -1), v], dim=2)
            if km is not None:
                km = torch.cat([km.new_ones(B, n), km], dim=1)
        mask = None if km is None else km[:, None, None, :]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                           dropout_p=o.p if self.training else 0.0)
        return o.proj(y.transpose(1, 2).reshape(B, T, o.d))


class SumModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.name = cfg.get("phase1", "prefixlm")
        if self.name not in BLOCKS:
            raise ValueError(f"phase1 must be one of {list(BLOCKS)}, got {self.name!r}")
        self.method = cfg.get("method", "full")
        backbone, self.info = load_phase1(self.name)
        self.backbone = backbone
        self.p1 = sys.modules[type(backbone).__module__]
        self.ctx = AttnContext()
        self.use_decoder = self.name == "encdec"
        self.grad_ckpt = bool(cfg.get("grad_ckpt", False))

        self.emb_name = next(n for n in ("tok_emb", "embed_tokens", "embed", "wte")
                             if isinstance(getattr(backbone, n, None), nn.Embedding))
        self.d = self.emb.embedding_dim
        self.vocab = self.emb.num_embeddings
        self.head_name = next((n for n, mod in backbone.named_children()
                               if isinstance(mod, nn.Linear) and mod.out_features == self.vocab), None)
        self._make_student()
        if cfg.get("backbone_dropout") is not None:
            for mod in backbone.modules():
                if isinstance(mod, nn.Dropout):
                    mod.p = float(cfg["backbone_dropout"])
        if self.name == "encdec":
            self._wrap_encoder_attention()
            self.backbone.grad_ckpt = self.grad_ckpt
        if self.name == "prefixlm":
            self._wrap_prefixlm_attention()

        self.n_prompt = 0
        self.prompt = None
        self._apply_method()

    @property
    def emb(self):
        return getattr(self.backbone, self.emb_name)

    def lm_head(self, h):
        if self.head_name is not None:
            return getattr(self.backbone, self.head_name)(h)
        return F.linear(h, self.emb.weight)

    def _blocks(self):
        return getattr(self.backbone, BLOCKS[self.name])

    def _stacks(self):
        s = [self._blocks()]
        if self.use_decoder:
            s.append(self.backbone.decoder)
        return s

    def _make_student(self):
        k = self.cfg.get("keep_layers")
        kd = self.cfg.get("keep_dec_layers")
        mode = self.cfg.get("keep_mode", "first")

        def pick(n, k):
            if mode == "spread":
                return sorted({round(i * (n - 1) / max(1, k - 1)) for i in range(k)})
            return list(range(k))

        if k:
            blocks = self._blocks()
            idx = pick(len(blocks), int(k))
            setattr(self.backbone, BLOCKS[self.name], nn.ModuleList([blocks[i] for i in idx]))
            self.kept_layers = idx
        if kd and self.name == "encdec":
            dec = self.backbone.decoder
            idx = pick(len(dec), int(kd))
            self.backbone.decoder = nn.ModuleList([dec[i] for i in idx])
            self.kept_dec_layers = idx

    def _wrap_encoder_attention(self):
        for blk in self.backbone.encoder:
            blk.attn = EncoderAttention(blk.attn, self.ctx, self.p1.apply_rope)

    def _wrap_prefixlm_attention(self):
        for blk in self.backbone.blocks:
            blk.attn = PrefixLMAttention(blk.attn, self.ctx, self.p1.apply_rope)

    def _apply_method(self):
        cfg, m = self.cfg, self.method
        bb = self.backbone
        if m == "full":
            for p in bb.parameters():
                p.requires_grad = True
            return
        for p in bb.parameters():
            p.requires_grad = False
        if m == "partial":
            n_top = int(cfg.get("unfreeze_layers", 4))
            for stack in self._stacks():
                for blk in list(stack)[-n_top:]:
                    for p in blk.parameters():
                        p.requires_grad = True
            for p in getattr(bb, FINAL_NORM[self.name]).parameters():
                p.requires_grad = True
            if self.use_decoder:
                for p in bb.dec_ln.parameters():
                    p.requires_grad = True
            if cfg.get("unfreeze_embeddings", False):
                self.emb.weight.requires_grad = True
        elif m in ("lora", "lorma"):
            self._wrap_linears(m)
        elif m == "adapter":
            self._add_adapters()
        elif m == "prompt":
            self._add_prompt()
        elif m == "prefix":
            self._add_prefix()
        else:
            raise ValueError(f"method must be lora | lorma | adapter | prompt | prefix (or full | partial), got {m!r}")
        if cfg.get("train_norms", False):
            for mod in bb.modules():
                if isinstance(mod, (nn.LayerNorm, nn.GroupNorm)) or type(mod).__name__.lower().endswith("rmsnorm"):
                    for p in mod.parameters(recurse=False):
                        p.requires_grad = True

    def _wrap_linears(self, kind):
        cfg = self.cfg
        targets = cfg.get("targets") or DEFAULT_TARGETS[self.name]
        r = int(cfg.get("rank", 8))
        alpha = float(cfg.get("alpha", 2 * r))
        stack_names = [BLOCKS[self.name]] + (["decoder"] if self.use_decoder else [])
        replaced = []
        for sname in stack_names:
            for bi, blk in enumerate(getattr(self.backbone, sname)):
                for path, mod in list(blk.named_modules()):
                    clean = path.replace("orig.", "").replace("inner.", "")
                    if not isinstance(mod, nn.Linear) or not any(clean == t or clean.endswith("." + t)
                                                                 for t in targets):
                        continue
                    parent_path, _, leaf = path.rpartition(".")
                    parent = blk.get_submodule(parent_path) if parent_path else blk
                    if kind == "lora":
                        new = LoRALinear(mod, r, alpha, float(cfg.get("lora_dropout", 0.05)))
                    else:
                        splits = _out_splits(self.name, clean, mod, self.d) if cfg.get("lorma_split", True) \
                            else [mod.out_features]
                        new = LoRMALinear(mod, r, alpha, splits, cfg.get("lorma_variant", "plus"))
                    setattr(parent, leaf, new)
                    replaced.append(f"{sname}.{bi}.{path}")
        if not replaced:
            raise ValueError(f"no linear matched targets {targets}")
        self.adapted = replaced

    def _add_adapters(self):
        m = int(self.cfg.get("adapter_size", 64))
        style = self.cfg.get("adapter_style", "houlsby")
        attn_site, mlp_site = ADAPTER_SITES[self.name]
        sites = [attn_site, mlp_site] if style == "houlsby" else [mlp_site]
        for stack in self._stacks():
            for blk in stack:
                for s in sites:
                    setattr(blk, s, WithAdapter(getattr(blk, s), self.d, m))

    def _add_prompt(self):
        n = int(self.cfg.get("n_prompt", 20))
        g = torch.Generator().manual_seed(int(self.cfg.get("init_seed", 0)))
        ids = torch.randint(1000, self.vocab, (n,), generator=g)
        self.prompt = nn.Parameter(self.emb.weight.detach()[ids].clone().float())
        self.n_prompt = n

    def _add_prefix(self):
        if self.name not in ("prefixlm", "encdec"):
            raise ValueError("prefix tuning is implemented for the attention models prefixlm and encdec only")
        n = int(self.cfg.get("n_prefix", 16))
        for blk in self._blocks():
            blk.attn.add_prefix(n)
        g = torch.Generator().manual_seed(int(self.cfg.get("init_seed", 0)))
        ids = torch.randint(1000, self.vocab, (1, n), generator=g)
        dev = self.emb.weight.device
        self.ctx.capture = True
        was = self.training
        self.eval()
        with torch.no_grad():
            x = self._embed(ids.to(dev))
            if self.name == "encdec":
                self._encode(x)
            else:
                self._states_prefixlm(x, n)
        self.ctx.capture = False
        self.train(was)

    def backbone_max_len(self):
        bb = self.backbone
        for n in ("max_seq_len", "max_len"):
            if isinstance(getattr(bb, n, None), int):
                return min(getattr(bb, n), 1024)
        c = getattr(bb, "c", None) or getattr(bb, "cfg", None) or {}
        if isinstance(c, dict) and c.get("max_seq_len"):
            return int(c["max_seq_len"])
        return 1024

    def _run(self, blk, *args):
        if self.grad_ckpt and self.training:
            return checkpoint(blk, *args, use_reentrant=False)
        return blk(*args)

    def _embed(self, ids):
        x = self.emb(ids)
        if self.prompt is not None:
            x = torch.cat([self.prompt.to(x.dtype)[None].expand(ids.shape[0], -1, -1), x], dim=1)
        return x

    def _check_len(self, n, what):
        lim = self.backbone_max_len()
        if n > lim:
            raise ValueError(f"{what} = {n} tokens exceeds the Phase-1 context {lim}")

    def _states_prefixlm(self, x, prefix_len):
        bb = self.backbone
        T = x.shape[1]
        if bb.pos_emb is not None:
            x = x + bb.pos_emb(torch.arange(T, device=x.device))
        x = bb.drop(x)
        cos = bb.rope_cos[:T] if bb.rope_cos is not None else None
        sin = bb.rope_sin[:T] if bb.rope_sin is not None else None
        plen = torch.full((x.shape[0],), prefix_len, dtype=torch.long, device=x.device)
        mask = self.p1.prefix_lm_mask(T, plen, x.device)
        for blk in bb.blocks:
            x = self._run(blk, x, cos, sin, mask)
        return bb.ln_f(x)

    def _states_arch4(self, x):
        bb = self.backbone
        T = x.shape[1]
        t = bb.tables(x.device)
        x = bb.drop(x)
        cos, sin, mask = t["cos"][:T], t["sin"][:T], t["mask"][:, :T, :T]
        for blk in bb.blocks:
            x = self._run(blk, x, cos, sin, mask)
        return bb.ln_f(x)

    def _states_biscale(self, x):
        bb = self.backbone
        drop = getattr(bb, "drop", None)
        if drop is not None:
            x = drop(x)
        for blk in bb.blocks:
            x = self._run(blk, x)
        return _first_attr(bb, ["norm", "ln_f", "final_norm"])(x)

    def _encode(self, x):
        bb = self.backbone
        T = x.shape[1]
        x = bb.drop(x)
        cos, sin = bb.rope_cos[:T], bb.rope_sin[:T]
        for blk in bb.encoder:
            x = self._run(blk, x, cos, sin)
        return bb.enc_ln(x)

    def logits(self, src, tgt_in):
        B, S = src.shape
        T = tgt_in.shape[1]
        if self.name == "encdec":
            self._check_len(self.n_prompt + S, "prompt + source")
            self._check_len(T, "summary")
            mem = self._encode(self._embed(src))
            return self.backbone.decode(tgt_in, mem)
        P = self.n_prompt
        self._check_len(P + S + T, "prompt + source + summary")
        ids = torch.cat([src, tgt_in], dim=1)
        if self.name == "biscale" and self.prompt is None:
            return self.backbone(ids, torch.full((B,), S, dtype=torch.long, device=ids.device))[:, S:S + T]
        x = self._embed(ids)
        if self.name == "prefixlm":
            h = self._states_prefixlm(x, P + S)
        elif self.name == "arch4":
            h = self._states_arch4(x)
        else:
            h = self._states_biscale(x)
        return self.lm_head(h[:, P + S:P + S + T])

    def forward(self, src, tgt_in):
        return self.logits(src, tgt_in)

    @torch.no_grad()
    def prefill(self, src, tgt_in):
        if self.name == "biscale":
            raise NotImplementedError("Bi-Scale has no cached decoding; the harness recomputes logits() each step")
        bb = self.backbone
        if self.name == "encdec":
            self._check_len(self.n_prompt + src.shape[1], "prompt + source")
            mem = self._encode(self._embed(src))
            cache = self.p1.EncDecCache(len(bb.decoder), bb.max_len)
            for li, blk in enumerate(bb.decoder):
                cache.ck[li], cache.cv[li] = blk.cross.memory_kv(mem)
            return bb._decode_cached(tgt_in, cache), cache
        P, S = self.n_prompt, src.shape[1]
        ids = torch.cat([src, tgt_in], dim=1)
        x = self._embed(ids)
        T0 = x.shape[1]
        self._check_len(T0, "prompt + source + start token")
        if self.name == "arch4":
            t = bb.tables(x.device)
            cos, sin, mask = t["cos"][:T0], t["sin"][:T0], t["mask"][:, :T0, :T0]
            w = t["powers"][:, :T0].flip(-1)
            states = []
            for blk in bb.blocks:
                x, st = blk.forward_state(x, cos, sin, mask, w)
                states.append(st)
            return self.lm_head(bb.ln_f(x[:, -1])), {"states": states, "pos": T0}
        if bb.pos_emb is not None:
            x = x + bb.pos_emb(torch.arange(T0, device=x.device))
        cos = bb.rope_cos[:T0] if bb.rope_cos is not None else None
        sin = bb.rope_sin[:T0] if bb.rope_sin is not None else None
        plen = torch.full((x.shape[0],), P + S, dtype=torch.long, device=x.device)
        mask = self.p1.prefix_lm_mask(T0, plen, x.device)
        store = KVStore(len(bb.blocks), self.backbone_max_len())
        for li, blk in enumerate(bb.blocks):
            x = blk(x, cos, sin, mask, (store, li), 0)
        store.pos = T0
        return self.lm_head(bb.ln_f(x[:, -1])), store

    @torch.no_grad()
    def decode_step(self, next_ids, cache):
        bb = self.backbone
        if self.name == "encdec":
            return bb._decode_cached(next_ids, cache), cache
        if self.name == "arch4":
            return bb.decode_step(next_ids, cache)
        n = cache.pos
        self._check_len(n + 1, "generated sequence")
        x = self.emb(next_ids)
        if bb.pos_emb is not None:
            x = x + bb.pos_emb(torch.arange(n, n + 1, device=x.device))
        cos = bb.rope_cos[n:n + 1] if bb.rope_cos is not None else None
        sin = bb.rope_sin[n:n + 1] if bb.rope_sin is not None else None
        for li, blk in enumerate(bb.blocks):
            x = blk(x, cos, sin, None, (cache, li), n)
        cache.pos = n + 1
        return self.lm_head(bb.ln_f(x[:, -1])), cache

    def param_groups(self, base_lr):
        new_mult = float(self.cfg.get("peft_lr_mult", 1.0))
        bb_mult = float(self.cfg.get("backbone_lr_mult", 1.0))
        wd = self.cfg.get("weight_decay")
        groups = {"new": [], "new_nd": [], "bb": [], "bb_nd": []}
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            nd = p.dim() < 2 or n == "prompt" or n.endswith((".pk", ".pv"))
            new = (n == "prompt" or n.endswith((".pk", ".pv", ".A", ".B")) or ".A." in n or ".B." in n
                   or ".adapter." in n)
            groups[("new" if new else "bb") + ("_nd" if nd else "")].append(p)
        mult = {"new": new_mult, "bb": bb_mult}
        out = []
        for k, ps in groups.items():
            if not ps:
                continue
            g = {"params": ps, "lr": base_lr * mult[k.split("_")[0]]}
            if k.endswith("_nd"):
                g["weight_decay"] = 0.0
            elif wd is not None:
                g["weight_decay"] = float(wd)
            out.append(g)
        return out


def build_sum_model(model_cfg: dict) -> nn.Module:
    return SumModel(dict(model_cfg))


@torch.no_grad()
def check_init_equivalence(model_cfg: dict, src_len: int = 256, tgt_len: int = 32, batch: int = 2, seed: int = 0):
    adapted = SumModel(dict(model_cfg)).eval()
    plain = SumModel(dict(model_cfg, method="full")).eval()
    dev = adapted.emb.weight.device
    plain.to(dev)
    g = torch.Generator().manual_seed(seed)
    src = torch.randint(1000, adapted.vocab, (batch, src_len), generator=g).to(dev)
    tgt = torch.randint(1000, adapted.vocab, (batch, tgt_len), generator=g).to(dev)
    a, p = adapted.logits(src, tgt).float(), plain.logits(src, tgt).float()
    diff = (a - p).abs().max().item()
    n_all = sum(q.numel() for q in adapted.parameters())
    n_tr = sum(q.numel() for q in adapted.parameters() if q.requires_grad)
    print(f"{model_cfg.get('phase1')}/{model_cfg.get('method')}: max |logits(adapted) - logits(pretrained)| = "
          f"{diff:.2e} (expect ~0 except prompt/prefix) | trainable {n_tr/1e6:.3f}M of {n_all/1e6:.2f}M "
          f"({100*n_tr/n_all:.2f}%)")
    if plain.name in ("prefixlm", "arch4", "biscale") and not model_cfg.get("keep_layers"):
        ids = torch.cat([src, tgt], 1)
        ref = plain.backbone(ids, torch.full((batch,), src_len, dtype=torch.long, device=dev))[:, src_len:].float()
        print(f"  plain logits() vs Phase-1 forward: max diff {(p - ref).abs().max().item():.2e} (expect ~0)")
    if hasattr(adapted, "prefill") and adapted.name != "biscale":
        last, cache = adapted.prefill(src, tgt[:, :1])
        errs = [(last.float() - a[:, 0]).abs().max().item()]
        for t in range(1, tgt_len):
            last, cache = adapted.decode_step(tgt[:, t:t + 1], cache)
            errs.append((last.float() - a[:, t]).abs().max().item())
        print(f"  cache vs logits max diff {max(errs):.2e} over {tgt_len} steps (should be < 1e-2)")
    return diff