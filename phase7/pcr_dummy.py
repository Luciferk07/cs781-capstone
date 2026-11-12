"""Smoke-test model only (NOT a submission model): tiny bag-of-embeddings encoder that satisfies the pcr_train contract."""
import torch
import torch.nn as nn


class Dummy(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = int(cfg.get("d", 64))
        self.emb = nn.Embedding(int(cfg.get("vocab", 32000)), d)
        self.proj = nn.Linear(d, d)
        self.keep = int(cfg.get("keep_dims", d))

    def embed(self, ids):
        return self.proj(self.emb(ids).mean(1))[:, :self.keep]

    def param_groups(self, base_lr):
        return [{"params": [p for p in self.parameters() if p.requires_grad], "lr": base_lr}]


def build_pcr_model(model_cfg):
    return Dummy(model_cfg)