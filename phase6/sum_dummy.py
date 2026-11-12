"""Smoke-test model only (NOT a submission model): bag-of-embeddings source + embedding of the previous token."""
import torch
import torch.nn as nn


class Dummy(nn.Module):
    def __init__(self, vocab_size, cached):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, 32)
        self.out = nn.Linear(32, vocab_size)
        self.cached = cached

    def logits(self, src, tgt_in):
        ctx = self.emb(src).mean(1, keepdim=True)
        return self.out(ctx + self.emb(tgt_in))

    def _prefill(self, src, tgt_in):
        ctx = self.emb(src).mean(1)
        return self.out(ctx + self.emb(tgt_in[:, -1])), {"ctx": ctx}

    def _decode_step(self, next_ids, cache):
        return self.out(cache["ctx"] + self.emb(next_ids[:, -1])), cache

    def __getattr__(self, name):
        if name in ("prefill", "decode_step"):
            if self.__dict__.get("cached", False):
                return object.__getattribute__(self, "_" + name)
            raise AttributeError(name)
        return super().__getattr__(name)


def build_sum_model(model_cfg):
    return Dummy(int(model_cfg["vocab_size"]), bool(model_cfg.get("cached", False)))