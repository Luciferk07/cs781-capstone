"""rr_dummy.py — PIPELINE TEST ONLY (smoke test): mean of token embeddings per sentence slot. Never submit it."""
import torch
import torch.nn as nn


class _Dummy(nn.Module):
    def __init__(self, vocab, n):
        super().__init__()
        self.emb = nn.Embedding(vocab, 64)
        self.out = nn.Linear(64, n)

    def forward(self, input_ids, attention_mask, sent_ids, n_slots):
        h = self.emb(input_ids)
        onehot = (sent_ids[..., None] == torch.arange(n_slots, device=input_ids.device)).to(h.dtype)
        pooled = onehot.transpose(1, 2) @ h / onehot.sum(1)[..., None].clamp(min=1)
        return self.out(pooled)


def build_rr_model(model_cfg, num_labels):
    return _Dummy(int(model_cfg.get("vocab_size", 32000)), num_labels)