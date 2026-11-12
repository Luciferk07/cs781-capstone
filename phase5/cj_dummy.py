"""Smoke-test model only (NOT a submission model): bag of embeddings per chunk, mean over a case's chunks."""
import torch
import torch.nn as nn


class Dummy(nn.Module):
    def __init__(self, vocab_size, num_labels):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, 32)
        self.out = nn.Linear(32, num_labels)

    def forward(self, input_ids, attention_mask, doc_ids, n_docs, chunk_pos=None, chunk_rel=None):
        m = attention_mask.float()[..., None]
        chunk = (self.emb(input_ids) * m).sum(1) / m.sum(1).clamp(min=1)
        doc = torch.zeros(n_docs, chunk.shape[1], device=chunk.device, dtype=chunk.dtype).index_add_(0, doc_ids, chunk)
        cnt = torch.zeros(n_docs, device=chunk.device, dtype=chunk.dtype).index_add_(0, doc_ids, torch.ones_like(doc_ids, dtype=chunk.dtype))
        return self.out(doc / cnt.clamp(min=1)[:, None])


def build_cj_model(model_cfg, num_labels):
    return Dummy(int(model_cfg["vocab_size"]), num_labels)