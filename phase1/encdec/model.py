import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

DEFAULTS = dict(
    d_model=768, n_heads=12, d_ff=3072, n_enc_layers=6, n_dec_layers=5,
    rope_base=10000.0, dropout=0.0, bias=True, tie_embeddings=True,
)


def rope_tables(head_dim, max_len, base):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    f = torch.outer(torch.arange(max_len, dtype=torch.float32), inv)
    f = torch.cat([f, f], dim=-1)
    return f.cos(), f.sin()


def apply_rope(x, cos, sin):
    h = x.shape[-1] // 2
    rot = torch.cat([-x[..., h:], x[..., :h]], dim=-1)
    return x * cos.to(x.dtype) + rot * sin.to(x.dtype)


class SelfAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.h, self.d = c["n_heads"], c["d_model"]
        self.hd = self.d // self.h
        self.qkv = nn.Linear(self.d, 3 * self.d, bias=c["bias"])
        self.proj = nn.Linear(self.d, self.d, bias=c["bias"])
        self.p = c["dropout"]

    def forward(self, x, cos, sin, causal, cache=None, li=None, start=0):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None:
            if cache.sk[li] is None:
                shape = (B, self.h, cache.max_dec, self.hd)
                cache.sk[li] = torch.empty(shape, device=k.device, dtype=k.dtype)
                cache.sv[li] = torch.empty(shape, device=v.device, dtype=v.dtype)
            cache.sk[li][:, :, start:start + T] = k
            cache.sv[li][:, :, start:start + T] = v
            k, v = cache.sk[li][:, :, :start + T], cache.sv[li][:, :, :start + T]
            causal = causal and T > 1
        y = F.scaled_dot_product_attention(q, k, v, is_causal=causal,
                                           dropout_p=self.p if self.training else 0.0)
        return self.proj(y.transpose(1, 2).reshape(B, T, self.d))


class CrossAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.h, self.d = c["n_heads"], c["d_model"]
        self.hd = self.d // self.h
        self.q = nn.Linear(self.d, self.d, bias=c["bias"])
        self.kv = nn.Linear(self.d, 2 * self.d, bias=c["bias"])
        self.proj = nn.Linear(self.d, self.d, bias=c["bias"])
        self.p = c["dropout"]

    def memory_kv(self, mem):
        B, P, _ = mem.shape
        k, v = self.kv(mem).view(B, P, 2, self.h, self.hd).permute(2, 0, 3, 1, 4)
        return k, v

    def forward(self, x, k, v):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.h, self.hd).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k.to(q.dtype), v.to(q.dtype), is_causal=False,
                                           dropout_p=self.p if self.training else 0.0)
        return self.proj(y.transpose(1, 2).reshape(B, T, self.d))


class MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.fc1 = nn.Linear(c["d_model"], c["d_ff"], bias=c["bias"])
        self.fc2 = nn.Linear(c["d_ff"], c["d_model"], bias=c["bias"])

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class EncoderBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln1 = nn.LayerNorm(c["d_model"])
        self.attn = SelfAttention(c)
        self.ln2 = nn.LayerNorm(c["d_model"])
        self.mlp = MLP(c)
        self.drop = nn.Dropout(c["dropout"])

    def forward(self, x, cos, sin):
        x = x + self.drop(self.attn(self.ln1(x), cos, sin, False))
        return x + self.drop(self.mlp(self.ln2(x)))


class DecoderBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln1 = nn.LayerNorm(c["d_model"])
        self.attn = SelfAttention(c)
        self.ln2 = nn.LayerNorm(c["d_model"])
        self.cross = CrossAttention(c)
        self.ln3 = nn.LayerNorm(c["d_model"])
        self.mlp = MLP(c)
        self.drop = nn.Dropout(c["dropout"])

    def forward(self, x, cos, sin, mem_k, mem_v, cache=None, li=None, start=0):
        x = x + self.drop(self.attn(self.ln1(x), cos, sin, True, cache, li, start))
        x = x + self.drop(self.cross(self.ln2(x), mem_k, mem_v))
        return x + self.drop(self.mlp(self.ln3(x)))

    def train_forward(self, x, cos, sin, mem):
        k, v = self.cross.memory_kv(mem)
        return self.forward(x, cos, sin, k, v)


class EncDecCache:
    def __init__(self, n_layers, max_dec):
        self.max_dec = max_dec
        self.sk, self.sv = [None] * n_layers, [None] * n_layers
        self.ck, self.cv = [None] * n_layers, [None] * n_layers
        self.pos = 0


class EncoderDecoderLM(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        V, D = c["vocab_size"], c["d_model"]
        self.max_len = c["max_seq_len"] + 1
        self.tok_emb = nn.Embedding(V, D)
        self.drop = nn.Dropout(c["dropout"])
        self.encoder = nn.ModuleList([EncoderBlock(c) for _ in range(c["n_enc_layers"])])
        self.enc_ln = nn.LayerNorm(D)
        self.decoder = nn.ModuleList([DecoderBlock(c) for _ in range(c["n_dec_layers"])])
        self.dec_ln = nn.LayerNorm(D)
        self.head = nn.Linear(D, V, bias=False)
        cos, sin = rope_tables(D // c["n_heads"], self.max_len, c["rope_base"])
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.grad_ckpt = False
        self.apply(self._init)
        ne, nd = c["n_enc_layers"], c["n_dec_layers"]
        for n, p in self.named_parameters():
            if n.endswith("proj.weight") or n.endswith("fc2.weight"):
                depth = 2 * ne if n.startswith("encoder.") else 3 * nd
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(depth))
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

    def set_gradient_checkpointing(self, flag):
        self.grad_ckpt = bool(flag)

    def _split(self, input_ids, prefix_lens):
        T = input_ids.shape[1]
        if prefix_lens is None:
            return 1
        P = int(prefix_lens[0])
        assert bool((prefix_lens == P).all()), "encoder-decoder needs one split per batch"
        return max(1, min(P, T))

    def _ckpt(self, fn, *args):
        if self.grad_ckpt and self.training:
            return checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    def encode(self, src):
        P = src.shape[1]
        x = self.drop(self.tok_emb(src))
        cos, sin = self.rope_cos[:P], self.rope_sin[:P]
        for blk in self.encoder:
            x = self._ckpt(blk, x, cos, sin)
        return self.enc_ln(x)

    def decode(self, tgt, mem):
        S = tgt.shape[1]
        x = self.drop(self.tok_emb(tgt))
        cos, sin = self.rope_cos[:S], self.rope_sin[:S]
        for blk in self.decoder:
            x = self._ckpt(blk.train_forward, x, cos, sin, mem)
        return self.head(self.dec_ln(x))

    def forward(self, input_ids, prefix_lens=None):
        P = self._split(input_ids, prefix_lens)
        mem = self.encode(input_ids[:, :P])
        return self.decode(input_ids[:, P - 1:], mem)

    def _decode_cached(self, ids, cache):
        S = ids.shape[1]
        start = cache.pos
        x = self.tok_emb(ids)
        cos, sin = self.rope_cos[start:start + S], self.rope_sin[start:start + S]
        for li, blk in enumerate(self.decoder):
            x = blk(x, cos, sin, cache.ck[li], cache.cv[li], cache, li, start)
        cache.pos += S
        return self.head(self.dec_ln(x[:, -1]))

    @torch.no_grad()
    def prefill(self, input_ids, prefix_lens):
        P = self._split(input_ids, prefix_lens)
        mem = self.encode(input_ids[:, :P])
        cache = EncDecCache(len(self.decoder), self.max_len)
        for li, blk in enumerate(self.decoder):
            cache.ck[li], cache.cv[li] = blk.cross.memory_kv(mem)
        return self._decode_cached(input_ids[:, P - 1:], cache), cache

    @torch.no_grad()
    def decode_step(self, next_ids, cache):
        return self._decode_cached(next_ids, cache), cache


def build_model(model_cfg):
    return EncoderDecoderLM({**DEFAULTS, **model_cfg})