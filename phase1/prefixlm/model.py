"""model.py — Prefix-LM transformer (arch: "prefixlm")."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

DEFAULTS = dict(
    d_model=768, n_layers=12, n_heads=12, d_ff=3072,
    pos="rope", rope_base=10000.0,
    dropout=0.0, bias=True, tie_embeddings=True,
)


# ----------------------------------------------------------------------------- mask
def prefix_lm_mask(T, prefix_lens, device, start=0):
    i = torch.arange(T, device=device)
    causal = i[None, :] <= i[:, None]
    in_prefix = (i + start)[None, None, :] < prefix_lens[:, None, None]
    return (causal[None] | in_prefix)[:, None]


# ----------------------------------------------------------------------------- rotary embedding
def rope_tables(head_dim, max_len, base):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    f = torch.outer(torch.arange(max_len, dtype=torch.float32), inv)
    f = torch.cat([f, f], dim=-1)
    return f.cos(), f.sin()


def apply_rope(x, cos, sin):
    h = x.shape[-1] // 2
    rot = torch.cat([-x[..., h:], x[..., :h]], dim=-1)
    return x * cos + rot * sin


# ----------------------------------------------------------------------------- layers
class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.h, self.d = c["n_heads"], c["d_model"]
        assert self.d % self.h == 0
        self.hd = self.d // self.h
        self.qkv = nn.Linear(self.d, 3 * self.d, bias=c["bias"])
        self.proj = nn.Linear(self.d, self.d, bias=c["bias"])
        self.p_drop = c["dropout"]
        self.rope = c["pos"] == "rope"

    def forward(self, x, cos, sin, mask, kv=None, start=0):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)
        if self.rope:
            cos, sin = cos.to(q.dtype), sin.to(q.dtype)
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if kv is not None:
            cache, li = kv
            if cache.k[li] is None:
                shape = (B, self.h, cache.max_len, self.hd)
                cache.k[li] = torch.empty(shape, device=k.device, dtype=k.dtype)
                cache.v[li] = torch.empty(shape, device=v.device, dtype=v.dtype)
            kbuf, vbuf = cache.k[li], cache.v[li]
            kbuf[:, :, start:start + T] = k
            vbuf[:, :, start:start + T] = v
            k, v = kbuf[:, :, :start + T], vbuf[:, :, :start + T]
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=False,
            dropout_p=self.p_drop if self.training else 0.0)
        return self.proj(y.transpose(1, 2).reshape(B, T, self.d))


class MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.fc1 = nn.Linear(c["d_model"], c["d_ff"], bias=c["bias"])
        self.fc2 = nn.Linear(c["d_ff"], c["d_model"], bias=c["bias"])
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln1 = nn.LayerNorm(c["d_model"])
        self.attn = Attention(c)
        self.ln2 = nn.LayerNorm(c["d_model"])
        self.mlp = MLP(c)
        self.drop = nn.Dropout(c["dropout"])

    def forward(self, x, cos, sin, mask, kv=None, start=0):
        x = x + self.drop(self.attn(self.ln1(x), cos, sin, mask, kv, start))
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x


class KVCache:
    def __init__(self, n_layers, max_len):
        self.k, self.v = [None] * n_layers, [None] * n_layers
        self.max_len, self.pos = max_len, 0


# ----------------------------------------------------------------------------- model
class PrefixLM(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        V, D, L = c["vocab_size"], c["d_model"], c["n_layers"]
        self.max_len = c["max_seq_len"]
        self.tok_emb = nn.Embedding(V, D)
        self.pos_emb = nn.Embedding(self.max_len, D) if c["pos"] == "learned" else None
        self.drop = nn.Dropout(c["dropout"])
        self.blocks = nn.ModuleList([Block(c) for _ in range(L)])
        self.ln_f = nn.LayerNorm(D)
        self.head = nn.Linear(D, V, bias=False)
        if c["pos"] == "rope":
            cos, sin = rope_tables(D // c["n_heads"], self.max_len, c["rope_base"])
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
        else:
            self.rope_cos = self.rope_sin = None
        self.grad_ckpt = False

        self.apply(self._init)
        for n, p in self.named_parameters():
            if n.endswith("attn.proj.weight") or n.endswith("mlp.fc2.weight"):
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

    def set_gradient_checkpointing(self, flag: bool):
        self.grad_ckpt = bool(flag)


    def _hidden(self, input_ids, mask, cache=None, start=0):
        B, T = input_ids.shape
        assert start + T <= self.max_len, f"sequence {start + T} > max_seq_len {self.max_len}"
        x = self.tok_emb(input_ids)
        if self.pos_emb is not None:
            x = x + self.pos_emb(torch.arange(start, start + T, device=input_ids.device))
        x = self.drop(x)
        cos = self.rope_cos[start:start + T] if self.rope_cos is not None else None
        sin = self.rope_sin[start:start + T] if self.rope_sin is not None else None
        for li, blk in enumerate(self.blocks):
            kv = None if cache is None else (cache, li)
            if self.grad_ckpt and self.training and cache is None:
                x = checkpoint(blk, x, cos, sin, mask, use_reentrant=False)
            else:
                x = blk(x, cos, sin, mask, kv, start)
        return self.ln_f(x)

    def forward(self, input_ids, prefix_lens=None):
        B, T = input_ids.shape
        if prefix_lens is None:
            prefix_lens = torch.zeros(B, dtype=torch.long, device=input_ids.device)
        mask = prefix_lm_mask(T, prefix_lens.to(input_ids.device), input_ids.device)
        return self.head(self._hidden(input_ids, mask))

    # ------------------------------------------------------------------ KV-cached generation
    @torch.no_grad()
    def prefill(self, input_ids, prefix_lens):
        B, T = input_ids.shape
        dev = input_ids.device
        cache = KVCache(len(self.blocks), self.max_len)
        mask = prefix_lm_mask(T, prefix_lens.to(dev), dev)
        h = self._hidden(input_ids, mask, cache, 0)
        cache.pos = T
        return self.head(h[:, -1]), cache

    @torch.no_grad()
    def decode_step(self, next_ids, cache):


        h = self._hidden(next_ids, None, cache, cache.pos)
        cache.pos += next_ids.shape[1]
        return self.head(h[:, -1]), cache


def build_model(model_cfg: dict) -> nn.Module:
    c = {**DEFAULTS, **model_cfg}
    assert c["pos"] in ("rope", "learned"), c["pos"]
    return PrefixLM(c)