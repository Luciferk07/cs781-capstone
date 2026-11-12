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


class EncoderAttention(nn.Module):
    def __init__(self, orig, ctx, rope_fn):
        super().__init__()
        self.orig, self.ctx, self.rope_fn = orig, ctx, rope_fn

    def forward(self, x, cos, sin, causal=False, cache=None, li=None, start=0):
        o = self.orig
        B, T, _ = x.shape
        q, k, v = o.qkv(x).view(B, T, 3, o.h, o.hd).permute(2, 0, 3, 1, 4)
        q, k = self.rope_fn(q, cos, sin), self.rope_fn(k, cos, sin)
        km = self.ctx.key_mask
        mask = None if km is None else km[:, None, None, :]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                           dropout_p=o.p if self.training else 0.0)
        return o.proj(y.transpose(1, 2).reshape(B, T, o.d))


class NERHead(nn.Module):
    def __init__(self, d, num_labels, kind, hidden, dropout, lookahead):
        super().__init__()
        self.kind = kind
        self.drop = nn.Dropout(dropout)
        self.context = None
        if lookahead:
            self.context = nn.Conv1d(d, d, kernel_size=2 * lookahead + 1, padding=lookahead)
            nn.init.zeros_(self.context.weight)
            nn.init.zeros_(self.context.bias)
        if kind == "linear":
            self.body, out_d = None, d
        elif kind == "mlp":
            self.body, out_d = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout)), hidden
        else:
            raise ValueError(f"head must be linear | mlp, got {kind!r}")
        self.out = nn.Linear(out_d, num_labels)
        nn.init.normal_(self.out.weight, 0.0, 0.02)
        nn.init.zeros_(self.out.bias)

    def forward(self, h, attention_mask):
        h = h * attention_mask[..., None].to(h.dtype)
        if self.context is not None:
            h = h + self.context(h.transpose(1, 2)).transpose(1, 2)
        h = self.drop(h)
        if self.kind == "mlp":
            h = self.body(h)
        return self.out(h)


class NERModel(nn.Module):
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

        self.n_prompt = 0
        self.prompt = None
        self._apply_method()

        self.head = NERHead(self.d, num_labels, cfg.get("head", "linear"), int(cfg.get("head_hidden", 512)),
                            float(cfg.get("head_dropout", 0.1)), int(cfg.get("lookahead", 0)))

    @property
    def emb(self):
        return getattr(self.backbone, self.emb_name)

    def _blocks(self):
        return getattr(self.backbone, BLOCKS[self.name])

    def _stacks(self):
        return [self._blocks()]

    def _strip_lm_head(self):
        bb = self.backbone
        V = self.emb.num_embeddings
        for n, mod in list(bb.named_children()):
            if isinstance(mod, nn.Linear) and mod.out_features == V:
                setattr(bb, n, None)
        if self.name == "encdec":
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
            if cfg.get("unfreeze_embeddings", False):
                self.emb.weight.requires_grad = True
        elif m == "bitfit":
            for n, p in bb.named_parameters():
                if n.endswith(".bias"):
                    p.requires_grad = True
        elif m in ("lora", "lorma"):
            self._wrap_linears(m)
        elif m == "adapter":
            self._add_adapters()
        elif m == "prompt":
            self._add_prompt()
        else:
            raise ValueError(f"unknown method {m!r}")
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
        stack_names = [BLOCKS[self.name]]
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
        ids = torch.randint(1000, self.emb.num_embeddings, (n,), generator=g)
        self.prompt = nn.Parameter(self.emb.weight.detach()[ids].clone().float())
        self.n_prompt = n

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
            x = bb.drop(x)
            cos, sin = bb.rope_cos[:T], bb.rope_sin[:T]
            for blk in bb.encoder:
                x = self._run(blk, x, cos, sin)
            return bb.enc_ln(x)
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

    def forward(self, input_ids, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        h = self.hidden(input_ids, attention_mask)
        return self.head(h, attention_mask.bool())

    def param_groups(self, base_lr):
        head_mult = float(self.cfg.get("head_lr_mult", 1.0))
        new_mult = float(self.cfg.get("peft_lr_mult", 1.0))
        bb_mult = float(self.cfg.get("backbone_lr_mult", 1.0))
        wd = self.cfg.get("weight_decay")
        groups = {"head": [], "head_nd": [], "new": [], "new_nd": [], "bb": [], "bb_nd": []}
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            nd = p.dim() < 2 or n == "prompt"
            if n.startswith("head."):
                key = "head"
            elif n == "prompt" or n.endswith((".A", ".B")) or ".A." in n or ".B." in n or ".adapter." in n:
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


def build_ner_model(model_cfg: dict, num_labels: int) -> nn.Module:
    return NERModel(dict(model_cfg), int(num_labels))


@torch.no_grad()
def check_init_equivalence(model_cfg: dict, num_labels: int, window: int = 256, batch: int = 2, seed: int = 0):
    adapted = NERModel(dict(model_cfg), num_labels).eval()
    plain = NERModel(dict(model_cfg, method="full", keep_layers=model_cfg.get("keep_layers")), num_labels).eval()
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
