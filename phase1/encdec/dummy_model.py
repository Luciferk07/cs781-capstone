"""dummy_model.py — PIPELINE TEST ONLY (used when RUN["smoke_test"] = True)."""
import torch.nn as nn


class _Dummy(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, 64)
        self.out = nn.Linear(64, vocab_size)

    def forward(self, input_ids, prefix_lens):
        return self.out(self.emb(input_ids))


def build_model(model_cfg):
    return _Dummy(model_cfg["vocab_size"])