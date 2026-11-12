"""cj_model.py — Phase-4/5 model."""
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
    "encdec": ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"],
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

    def forward(self, *args, **kwargs):
        out = self.inner(*args, **kwargs)
        if isinstance(out, tuple):
            return (self.adapter(out[0]),) + tuple(out[1:])
        return self.adapter(out)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._modules["inner"], name)


class AttnContext:
    def __init__(self):
        self.key_mask = None
        self.capture = False


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
        elif self.pk is not None:
            n = self.pk.shape[1]
            k = torch.cat([self.pk.to(k.dtype)[None].expand(B, -1, -1, -1), k], dim=2)
            v = torch.cat([self.pv.to(v.dtype)[None].expand(B, -1, -1, -1), v], dim=2)
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


class CaseHead(nn.Module):
    def __init__(self, d, num_labels, pool, agg, hidden, dropout, use_position, out_prior=None):
        super().__init__()
        if pool not in ("mean", "last", "first", "attn", "mean_last"):
            raise ValueError(f"pool must be mean | last | first | attn | mean_last, got {pool!r}")
        if agg not in ("mean", "max", "attn", "gru", "chunk_max", "chunk_lme"):
            raise ValueError(f"agg must be mean | max | attn | gru | chunk_max | chunk_lme, got {agg!r}")
        self.pool, self.agg = pool, agg
        self.tok_scorer = nn.Linear(d, 1) if pool == "attn" else None
        d_in = 2 * d if pool == "mean_last" else d
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU())
        self.pos = None
        if use_position:
            self.pos = nn.Linear(2, hidden)
            nn.init.zeros_(self.pos.weight)
            nn.init.zeros_(self.pos.bias)
        self.rnn = nn.GRU(hidden, hidden // 2, batch_first=True, bidirectional=True) if agg == "gru" else None
        self.chunk_scorer = nn.Linear(hidden, 1) if agg in ("attn", "gru") else None
        self.out = nn.Linear(hidden, num_labels)
        nn.init.normal_(self.out.weight, 0.0, 0.02)
        nn.init.zeros_(self.out.bias)
        if out_prior is not None:
            p = torch.as_tensor(out_prior, dtype=torch.float32).clamp(1e-4, 1 - 1e-4).expand(num_labels)
            with torch.no_grad():
                self.out.bias.copy_(torch.log(p / (1 - p)))

    def pool_tokens(self, h, mask):
        N, T, d = h.shape
        h = h.float()
        m = mask.float()
        cnt = m.sum(1).clamp(min=1.0)
        vecs = []
        if self.pool in ("mean", "mean_last"):
            vecs.append((h * m[..., None]).sum(1) / cnt[:, None])
        if self.pool in ("last", "mean_last"):
            pos = torch.arange(T, device=h.device, dtype=torch.float32)[None, :]
            idx = (pos * m).amax(1).long()
            vecs.append(h[torch.arange(N, device=h.device), idx])
        if self.pool == "first":
            pos = torch.arange(T, device=h.device, dtype=torch.float32)[None, :]
            idx = torch.where(m > 0, pos, torch.full_like(pos, float(T))).amin(1).clamp(max=T - 1).long()
            vecs.append(h[torch.arange(N, device=h.device), idx])
        if self.pool == "attn":
            score = self.tok_scorer(h).squeeze(-1).masked_fill(~mask, -1e4)
            w = torch.softmax(score, dim=-1)
            vecs.append(torch.einsum("nt,ntd->nd", w, h))
        return torch.cat(vecs, dim=-1)

    def forward(self, h, mask, doc_ids, n_docs, chunk_pos, chunk_rel):
        with torch.autocast(device_type=h.device.type, enabled=False):
            v = self.pool_tokens(h, mask)
            z = self.proj(self.drop(v))
            N = z.shape[0]
            counts = torch.bincount(doc_ids, minlength=n_docs)
            if self.pos is not None:
                rel = chunk_rel.float() if chunk_rel is not None else torch.zeros(N, device=z.device)
                first = torch.cumsum(counts, 0) - counts
                slot_f = (torch.arange(N, device=z.device) - first[doc_ids]).float()
                frac = slot_f / (counts[doc_ids].float() - 1.0).clamp(min=1.0)
                z = z + self.pos(torch.stack([rel, frac], dim=-1))
            if self.agg in ("chunk_max", "chunk_lme"):
                logits = self.out(self.drop(z))
                member = doc_ids[None, :] == torch.arange(n_docs, device=z.device)[:, None]
                masked = torch.where(member[..., None], logits[None], torch.full_like(logits[None], -1e4))
                if self.agg == "chunk_max":
                    return masked.amax(1)
                return torch.logsumexp(masked, dim=1) - torch.log(counts.clamp(min=1).float())[:, None]
            if self.agg == "mean":
                case = torch.zeros(n_docs, z.shape[1], device=z.device).index_add_(0, doc_ids, z)
                case = case / counts.clamp(min=1)[:, None].float()
            elif self.agg == "max":
                member = doc_ids[None, :] == torch.arange(n_docs, device=z.device)[:, None]
                case = torch.where(member[..., None], z[None], torch.full_like(z[None], -1e4)).amax(1)
            else:
                first = torch.cumsum(counts, 0) - counts
                slot = torch.arange(N, device=z.device) - first[doc_ids]
                M = int(counts.max().item())
                seq = z.new_zeros(n_docs, M, z.shape[1])
                seq[doc_ids, slot] = z
                valid = torch.zeros(n_docs, M, dtype=torch.bool, device=z.device)
                valid[doc_ids, slot] = True
                if self.rnn is not None:
                    packed = nn.utils.rnn.pack_padded_sequence(seq, counts.clamp(min=1).cpu(), batch_first=True,
                                                               enforce_sorted=False)
                    out, _ = self.rnn(packed)
                    seq, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=M)
                score = self.chunk_scorer(seq).squeeze(-1).masked_fill(~valid, -1e4)
                w = torch.softmax(score, dim=-1) * valid.float()
                self.last_chunk_weights = w.detach()
                case = torch.einsum("bm,bmd->bd", w, seq)
            return self.out(self.drop(case))


class CJModel(nn.Module):
    def __init__(self, cfg, num_labels):
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
        self.use_decoder = self.name == "encdec" and bool(cfg.get("encdec_decoder", False))
        self.grad_ckpt = bool(cfg.get("grad_ckpt", False))

        self.emb_name = next(n for n in ("tok_emb", "embed_tokens", "embed", "wte")
                             if isinstance(getattr(backbone, n, None), nn.Embedding))
        self.d = self.emb.embedding_dim
        self._strip_lm_head()
        self._make_student()
        if cfg.get("backbone_dropout") is not None:
            for mod in backbone.modules():
                if isinstance(mod, nn.Dropout):
                    mod.p = float(cfg["backbone_dropout"])
        if self.name == "encdec":
            self._wrap_encoder_attention()
        if self.name == "prefixlm":
            self._wrap_prefixlm_attention()

        self.n_prompt = 0
        self.prompt = None
        self._apply_method()

        default_pool = "mean_last" if self.name in ("arch4", "biscale") else "mean"
        self.head = CaseHead(self.d, num_labels, cfg.get("pool", default_pool), cfg.get("agg", "gru"),
                             int(cfg.get("head_hidden", 256)), float(cfg.get("head_dropout", 0.1)),
                             bool(cfg.get("chunk_position", True)), cfg.get("out_prior"))

    @property
    def emb(self):
        return getattr(self.backbone, self.emb_name)

    def _blocks(self):
        return getattr(self.backbone, BLOCKS[self.name])

    def _stacks(self):
        s = [self._blocks()]
        if self.use_decoder:
            s.append(self.backbone.decoder)
        return s

    def _strip_lm_head(self):
        bb = self.backbone
        V = self.emb.num_embeddings
        for n, mod in list(bb.named_children()):
            if isinstance(mod, nn.Linear) and mod.out_features == V:
                setattr(bb, n, None)
        if self.name == "encdec" and not self.use_decoder:
            bb.decoder = None
            bb.dec_ln = None

    def _make_student(self):
        k = self.cfg.get("keep_layers")
        if not k:
            return
        blocks = self._blocks()
        n = len(blocks)
        k = int(k)
        if self.cfg.get("keep_mode", "first") == "spread":
            idx = sorted({round(i * (n - 1) / max(1, k - 1)) for i in range(k)})
        else:
            idx = list(range(k))
        setattr(self.backbone, BLOCKS[self.name], nn.ModuleList([blocks[i] for i in idx]))
        self.kept_layers = idx

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
        elif m == "bitfit":
            for n, p in bb.named_parameters():
                if n.endswith(".bias"):
                    p.requires_grad = True
        elif m == "lorma":
            self._wrap_linears(m)
        elif m == "adapter":
            self._add_adapters()
        elif m == "prompt":
            self._add_prompt()
        elif m == "prefix":
            self._add_prefix()
        else:
            raise ValueError(f"method must be full | partial | bitfit | lorma | adapter | prompt | prefix, got {m!r}")
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
        ids = torch.randint(1000, self.emb.num_embeddings, (n,), generator=g)
        self.prompt = nn.Parameter(self.emb.weight.detach()[ids].clone().float())
        self.n_prompt = n

    def _add_prefix(self):
        if self.name not in ("prefixlm", "encdec"):
            raise ValueError("prefix tuning is implemented for the attention models prefixlm and encdec only")
        n = int(self.cfg.get("n_prefix", 16))
        for blk in self._blocks():
            blk.attn.add_prefix(n)
        g = torch.Generator().manual_seed(int(self.cfg.get("init_seed", 0)))
        ids = torch.randint(1000, self.emb.num_embeddings, (1, n), generator=g)
        dev = self.emb.weight.device
        self.ctx.capture = True
        was = self.training
        self.eval()
        with torch.no_grad():
            self.hidden(ids.to(dev), torch.ones_like(ids, dtype=torch.bool).to(dev))
        self.ctx.capture = False
        self.train(was)

    def hidden(self, input_ids, attention_mask):
        bb = self.backbone
        B, W = input_ids.shape
        x = self.emb(input_ids)
        am = attention_mask.bool()
        if self.prompt is not None:
            x = torch.cat([self.prompt.to(x.dtype)[None].expand(B, -1, -1), x], dim=1)
            am = torch.cat([am.new_ones(B, self.n_prompt), am], dim=1)
        T = x.shape[1]
        if T > self.backbone_max_len():
            raise ValueError(f"window + prompt = {T} exceeds the Phase-1 context {self.backbone_max_len()}")
        h = getattr(self, f"_hidden_{self.name}")(x, am, T)
        return h[:, T - W:]

    def backbone_max_len(self):
        bb = self.backbone
        for n in ("max_len", "max_seq_len"):
            if isinstance(getattr(bb, n, None), int):
                return getattr(bb, n)
        c = getattr(bb, "c", None) or getattr(bb, "cfg", None) or {}
        if isinstance(c, dict) and c.get("max_seq_len"):
            return int(c["max_seq_len"])
        return 1024

    def _run(self, blk, *args):
        if self.grad_ckpt and self.training:
            return checkpoint(blk, *args, use_reentrant=False)
        return blk(*args)

    def _hidden_prefixlm(self, x, am, T):
        bb = self.backbone
        if bb.pos_emb is not None:
            x = x + bb.pos_emb(torch.arange(T, device=x.device))
        x = bb.drop(x)
        cos = bb.rope_cos[:T] if bb.rope_cos is not None else None
        sin = bb.rope_sin[:T] if bb.rope_sin is not None else None
        mask = self.p1.prefix_lm_mask(T, am.sum(1), x.device)
        for blk in bb.blocks:
            x = self._run(blk, x, cos, sin, mask)
        return bb.ln_f(x)

    def _hidden_encdec(self, x, am, T):
        bb = self.backbone
        self.ctx.key_mask = None if bool(am.all()) else am
        try:
            emb = x
            x = bb.drop(x)
            cos, sin = bb.rope_cos[:T], bb.rope_sin[:T]
            for blk in bb.encoder:
                x = self._run(blk, x, cos, sin)
            mem = bb.enc_ln(x)
            if not self.use_decoder:
                return mem
            y = bb.drop(emb)
            for blk in bb.decoder:
                y = self._run(blk.train_forward, y, cos, sin, mem)
            return bb.dec_ln(y) + mem
        finally:
            self.ctx.key_mask = None

    def _hidden_arch4(self, x, am, T):
        bb = self.backbone
        t = bb.tables(x.device)
        x = bb.drop(x)
        cos, sin, mask = t["cos"][:T], t["sin"][:T], t["mask"][:, :T, :T]
        for blk in bb.blocks:
            x = self._run(blk, x, cos, sin, mask)
        return bb.ln_f(x)

    def _hidden_biscale(self, x, am, T):
        bb = self.backbone
        drop = getattr(bb, "drop", None)
        if drop is not None:
            x = drop(x)
        for blk in bb.blocks:
            x = self._run(blk, x)
        return _first_attr(bb, ["norm", "ln_f", "final_norm"])(x)

    def forward(self, input_ids, attention_mask, doc_ids, n_docs, chunk_pos=None, chunk_rel=None):
        am = attention_mask.bool()
        h = self.hidden(input_ids, am)
        return self.head(h, am, doc_ids.long(), int(n_docs), chunk_pos, chunk_rel)

    def param_groups(self, base_lr):
        head_mult = float(self.cfg.get("head_lr_mult", 1.0))
        new_mult = float(self.cfg.get("peft_lr_mult", 1.0))
        bb_mult = float(self.cfg.get("backbone_lr_mult", 1.0))
        wd = self.cfg.get("weight_decay")
        groups = {"head": [], "head_nd": [], "new": [], "new_nd": [], "bb": [], "bb_nd": []}
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            nd = p.dim() < 2 or n == "prompt" or n.endswith(".pk") or n.endswith(".pv")
            if n.startswith("head."):
                key = "head"
            elif n == "prompt" or n.endswith((".pk", ".pv")) or ".A." in n or ".B." in n or ".adapter." in n:
                key = "new"
            else:
                key = "bb"
            groups[key + ("_nd" if nd else "")].append(p)
        mult = {"head": head_mult, "new": new_mult, "bb": bb_mult}
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


def build_cj_model(model_cfg: dict, num_labels: int) -> nn.Module:
    return CJModel(dict(model_cfg), int(num_labels))


@torch.no_grad()
def check_init_equivalence(model_cfg: dict, num_labels: int, window: int = 256, batch: int = 2, seed: int = 0):
    adapted = CJModel(dict(model_cfg), num_labels).eval()
    plain = CJModel(dict(model_cfg, method="full", keep_layers=model_cfg.get("keep_layers")), num_labels).eval()
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(1000, adapted.emb.num_embeddings, (batch, window), generator=g)
    dev = adapted.emb.weight.device
    plain.to(dev)
    am = torch.ones_like(ids, dtype=torch.bool).to(dev)
    diff = (adapted.hidden(ids.to(dev), am).float() - plain.hidden(ids.to(dev), am).float()).abs().max().item()
    n_all = sum(p.numel() for p in adapted.parameters())
    n_tr = sum(p.numel() for p in adapted.parameters() if p.requires_grad)
    print(f"{model_cfg.get('phase1')}/{model_cfg.get('method')}: max |hidden(adapted) - hidden(pretrained)| = "
          f"{diff:.2e} | trainable {n_tr/1e6:.3f}M of {n_all/1e6:.2f}M")
    return diff
