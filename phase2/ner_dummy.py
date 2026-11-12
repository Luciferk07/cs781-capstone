"""ner_dummy.py — PIPELINE TEST ONLY (smoke test): each token's tag from its own embedding. Never submit it."""
import torch.nn as nn


class _Dummy(nn.Module):
    def __init__(self, vocab, n):
        super().__init__()
        self.emb = nn.Embedding(vocab, 64)
        self.out = nn.Linear(64, n)

    def forward(self, input_ids, attention_mask):
        return self.out(self.emb(input_ids))


def build_ner_model(model_cfg, num_labels):
    return _Dummy(int(model_cfg.get("vocab_size", 32000)), num_labels)