"""model.py for the Bi-Scale generation notebook."""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================== copied from the Bi-Scale training notebook ==============================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq=2048, theta=10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq)

    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def forward(self, seq_len):
        if seq_len > self.cos_cached.shape[2]:
            self._build_cache(seq_len)
        return self.cos_cached[:, :, :seq_len, :], self.sin_cached[:, :, :seq_len, :]

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat([-x2, x1], dim=-1)

def apply_rotary(x, cos, sin):
    return x * cos + rotate_half(x) * sin

class BiScaleGQA(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, dropout, ctx_len, rope_theta, is_local=False, window_size=256):
        super().__init__()
        self.n_heads    = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups   = n_heads // n_kv_heads
        self.d_head     = d_model // n_heads

        self.q_proj = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.o_proj = nn.Linear(n_heads * self.d_head, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.d_head, max_seq=ctx_len, theta=rope_theta)

        causal = torch.tril(torch.ones(ctx_len, ctx_len))
        if is_local:
            causal = causal * torch.triu(torch.ones(ctx_len, ctx_len), diagonal=-window_size)
        self.register_buffer("mask", causal.view(1, 1, ctx_len, ctx_len))


        for proj in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            nn.init.normal_(proj.weight, mean=0.0, std=0.02)

    def forward(self, x):
        B, T, C = x.shape
        dh = self.d_head

        q = self.q_proj(x).view(B, T, self.n_heads, dh).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, dh).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, dh).transpose(1, 2)

        cos, sin = self.rope(T)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        k = k.repeat_interleave(self.n_groups, dim=1)
        v = v.repeat_interleave(self.n_groups, dim=1)

        scale = dh ** -0.5
        att = (q @ k.transpose(-2, -1)) * scale
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = self.attn_drop(F.softmax(att, dim=-1))

        out = (att @ v).transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up   = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)
        for proj in [self.gate, self.up, self.down]:
            nn.init.normal_(proj.weight, mean=0.0, std=0.02)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))

class LearnedGatedResidual(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Linear(dim, dim, bias=True)

        nn.init.normal_(self.gate.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.gate.bias, 2.0)

    def forward(self, x, sublayer):
        g = torch.sigmoid(self.gate(x))
        return x + g * sublayer

class BiScaleLegalBlock(nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()
        is_local = (layer_idx < (cfg["n_layers"] // 2))
        self.norm1 = RMSNorm(cfg["d_model"])
        self.attn  = BiScaleGQA(
            cfg["d_model"], cfg["n_heads"], cfg["n_kv_heads"],
            cfg["dropout"], cfg["ctx_len"], cfg["rope_theta"],
            is_local=is_local, window_size=cfg["local_window"]
        )
        self.res1 = LearnedGatedResidual(cfg["d_model"])
        self.norm2 = RMSNorm(cfg["d_model"])
        self.ffn   = SwiGLU(cfg["d_model"], cfg["d_ff"])
        self.res2  = LearnedGatedResidual(cfg["d_model"])
        self.drop  = nn.Dropout(cfg["dropout"])

    def forward(self, x):
        x = self.res1(x, self.drop(self.attn(self.norm1(x))))
        x = self.res2(x, self.drop(self.ffn(self.norm2(x))))
        return x

class BiScaleLegalLLaMA(nn.Module):
    def __init__(self, cfg, vocab_size):
        super().__init__()
        self.ctx     = cfg["ctx_len"]
        self.tok_emb = nn.Embedding(vocab_size, cfg["d_model"])
        self.drop    = nn.Dropout(cfg["dropout"])
        self.blocks  = nn.ModuleList([BiScaleLegalBlock(cfg, i) for i in range(cfg["n_layers"])])
        self.norm    = RMSNorm(cfg["d_model"])
        self.lm_head = nn.Linear(cfg["d_model"], vocab_size, bias=False)


        self.lm_head.weight = self.tok_emb.weight


        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)

    def forward(self, ids, targets=None):
        B, T = ids.shape
        x = self.drop(self.tok_emb(ids))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
# ======================================= end of copied code =======================================


_ORIGINAL_ATTN_FORWARD = BiScaleGQA.forward


def _sdpa_attn_forward(self, x):
    B, T, C = x.shape
    dh = self.d_head
    q = self.q_proj(x).view(B, T, self.n_heads, dh).transpose(1, 2)
    k = self.k_proj(x).view(B, T, self.n_kv_heads, dh).transpose(1, 2)
    v = self.v_proj(x).view(B, T, self.n_kv_heads, dh).transpose(1, 2)
    cos, sin = self.rope(T)
    q = apply_rotary(q, cos.to(q.dtype), sin.to(q.dtype))
    k = apply_rotary(k, cos.to(k.dtype), sin.to(k.dtype))
    k = k.repeat_interleave(self.n_groups, dim=1)
    v = v.repeat_interleave(self.n_groups, dim=1)
    y = F.scaled_dot_product_attention(q, k, v, attn_mask=self.mask[:, :, :T, :T].bool(),
                                       dropout_p=0.0, scale=dh ** -0.5)
    return self.o_proj(y.transpose(1, 2).contiguous().view(B, T, -1))


def use_sdpa(flag=True):
    BiScaleGQA.forward = _sdpa_attn_forward if flag else _ORIGINAL_ATTN_FORWARD


class BiScaleForHarness(BiScaleLegalLLaMA):
    def forward(self, input_ids, prefix_lens=None):
        return super().forward(input_ids)[0]


def build_model(model_cfg):
    use_sdpa(model_cfg.get("use_sdpa", True))
    return BiScaleForHarness(model_cfg, model_cfg["vocab_size"])
