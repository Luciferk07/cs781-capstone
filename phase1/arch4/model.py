import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

DEFAULTS = dict(
    d_model=768, n_layers=9, n_heads=6, d_ff=3072, value_factor=2,
    gamma_start_exp=5.0, rope_base=10000.0, dropout=0.0, tie_embeddings=True,
)


def rope_tables(dim, max_len, base):
    inv = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
    f = torch.outer(torch.arange(max_len, dtype=torch.float64), inv)
    f = torch.cat([f, f], dim=-1)
    return f.cos().float(), f.sin().float()


def apply_rope(x, cos, sin):
    h = x.shape[-1] // 2
    rot = torch.cat([-x[..., h:], x[..., :h]], dim=-1)
    return x * cos.to(x.dtype) + rot * sin.to(x.dtype)


class MultiScaleRetention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.d, self.h = c["d_model"], c["n_heads"]
        self.dk = self.d // self.h
        self.dv = self.dk * c["value_factor"]
        self.vd = self.h * self.dv
        self.proj = nn.Linear(self.d, 2 * self.d + 2 * self.vd, bias=False)
        self.out = nn.Linear(self.vd, self.d, bias=False)
        self.gn = nn.GroupNorm(self.h, self.vd, eps=1e-5)
        self.scale = self.dk ** -0.5

    def _qkvg(self, x, cos, sin):
        B, T, _ = x.shape
        q, k, v, g = self.proj(x).split([self.d, self.d, self.vd, self.vd], dim=-1)
        q = q.view(B, T, self.h, self.dk).transpose(1, 2)
        k = k.view(B, T, self.h, self.dk).transpose(1, 2)
        v = v.view(B, T, self.h, self.dv).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin) * self.scale
        return q, k, v, g

    def _finish(self, o, g):
        B, H, T, dv = o.shape
        o = self.gn(o.transpose(1, 2).reshape(B * T, H * dv)).view(B, T, H * dv)
        return self.out(F.silu(g) * o)

    def forward(self, x, cos, sin, mask, w=None):
        q, k, v, g = self._qkvg(x, cos, sin)
        r = (q @ k.transpose(-1, -2)) * mask.to(q.dtype)


        den = r.detach().float().sum(-1, keepdim=True).abs().clamp(min=1.0)
        o = (r / den.to(r.dtype)) @ v
        y = self._finish(o, g)
        if w is None:
            return y
        with torch.autocast(device_type=x.device.type, enabled=False):
            kf, vf = k.float(), v.float()
            S = torch.einsum("bhld,bhle,hl->bhde", kf, vf, w)
            z = torch.einsum("bhld,hl->bhd", kf, w)
        return y, (S, z)

    def step(self, x, cos, sin, state, gamma, rs):
        q, k, v, g = self._qkvg(x, cos, sin)
        S, z = state
        with torch.autocast(device_type=x.device.type, enabled=False):
            q, k, v = q.float()[:, :, 0], k.float()[:, :, 0], v.float()[:, :, 0]
            S = gamma[None, :, None, None] * S + k[..., :, None] * v[..., None, :]
            z = gamma[None, :, None] * z + k
            inv = rs.rsqrt()[None, :, None]
            o = torch.einsum("bhd,bhde->bhe", q, S) * inv
            s = ((q * z).sum(-1, keepdim=True) * inv).abs().clamp(min=1.0)
            o = (o / s)[:, :, None]
        return self._finish(o, g), (S, z)


class MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.fc1 = nn.Linear(c["d_model"], c["d_ff"])
        self.fc2 = nn.Linear(c["d_ff"], c["d_model"])

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class RetNetBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln1 = nn.LayerNorm(c["d_model"])
        self.msr = MultiScaleRetention(c)
        self.ln2 = nn.LayerNorm(c["d_model"])
        self.mlp = MLP(c)
        self.drop = nn.Dropout(c["dropout"])

    def forward(self, x, cos, sin, mask):
        x = x + self.drop(self.msr(self.ln1(x), cos, sin, mask))
        return x + self.drop(self.mlp(self.ln2(x)))

    def forward_state(self, x, cos, sin, mask, w):
        a, st = self.msr(self.ln1(x), cos, sin, mask, w)
        x = x + a
        return x + self.mlp(self.ln2(x)), st

    def step(self, x, cos, sin, state, gamma, rs):
        a, st = self.msr.step(self.ln1(x), cos, sin, state, gamma, rs)
        x = x + a
        return x + self.mlp(self.ln2(x)), st


class RetentionLM(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        V, D, L, H = c["vocab_size"], c["d_model"], c["n_layers"], c["n_heads"]
        self.max_len = c["max_seq_len"]
        self.tok_emb = nn.Embedding(V, D)
        self.drop = nn.Dropout(c["dropout"])
        self.blocks = nn.ModuleList([RetNetBlock(c) for _ in range(L)])
        self.ln_f = nn.LayerNorm(D)
        self.head = nn.Linear(D, V, bias=False)

        gamma = 1.0 - 2.0 ** (-c["gamma_start_exp"] - torch.arange(H, dtype=torch.float64))
        log_gamma = gamma.log()
        n = torch.arange(2 * self.max_len, dtype=torch.float64)
        powers = torch.exp(log_gamma[:, None] * n[None, :])
        diff = n[:self.max_len, None] - n[None, :self.max_len]
        decay = torch.exp(log_gamma[:, None, None] * diff.clamp(min=0)) * (diff >= 0)
        decay = decay / decay.sum(-1, keepdim=True).sqrt()
        rs = (1.0 - powers * gamma[:, None]) / (1.0 - gamma[:, None])
        cos, sin = rope_tables(D // H, 2 * self.max_len, c["rope_base"])
        self._cpu = dict(mask=decay.float(), powers=powers.float(), rs=rs.float(), gamma=gamma.float(),
                         cos=cos, sin=sin)
        self._dev = {}
        self.grad_ckpt = False

        self.apply(self._init)
        for name, p in self.named_parameters():
            if name.endswith("msr.out.weight") or name.endswith("mlp.fc2.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * L))
        if c["tie_embeddings"]:
            self.head.weight = self.tok_emb.weight

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def tables(self, device):
        key = str(device)
        if key not in self._dev:
            self._dev[key] = {k: v.to(device) for k, v in self._cpu.items()}
        return self._dev[key]

    def set_gradient_checkpointing(self, flag):
        self.grad_ckpt = bool(flag)

    def forward(self, input_ids, prefix_lens=None):
        T = input_ids.shape[1]
        t = self.tables(input_ids.device)
        x = self.drop(self.tok_emb(input_ids))
        cos, sin, mask = t["cos"][:T], t["sin"][:T], t["mask"][:, :T, :T]
        for blk in self.blocks:
            if self.grad_ckpt and self.training:
                x = checkpoint(blk, x, cos, sin, mask, use_reentrant=False)
            else:
                x = blk(x, cos, sin, mask)
        return self.head(self.ln_f(x))

    @torch.no_grad()
    def prefill(self, input_ids, prefix_lens=None):
        T = input_ids.shape[1]
        t = self.tables(input_ids.device)
        x = self.tok_emb(input_ids)
        cos, sin, mask = t["cos"][:T], t["sin"][:T], t["mask"][:, :T, :T]
        w = t["powers"][:, :T].flip(-1)
        states = []
        for blk in self.blocks:
            x, st = blk.forward_state(x, cos, sin, mask, w)
            states.append(st)
        return self.head(self.ln_f(x[:, -1])), {"states": states, "pos": T}

    @torch.no_grad()
    def decode_step(self, next_ids, cache):
        n = cache["pos"]
        t = self.tables(next_ids.device)
        x = self.tok_emb(next_ids)
        cos, sin, rs = t["cos"][n:n + 1], t["sin"][n:n + 1], t["rs"][:, n]
        for i, blk in enumerate(self.blocks):
            x, cache["states"][i] = blk.step(x, cos, sin, cache["states"][i], t["gamma"], rs)
        cache["pos"] = n + 1
        return self.head(self.ln_f(x[:, -1])), cache


def build_model(model_cfg):
    return RetentionLM({**DEFAULTS, **model_cfg})