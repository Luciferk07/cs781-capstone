"""pcr_data.py — Phase-7 prior case retrieval data handling."""
import hashlib, json, math, os, pickle
from collections import Counter

import numpy as np

ID_KEYS = ("id", "query_id", "case_id", "doc_id", "candidate_id", "qid", "cid", "name")
TEXT_KEYS = ("text", "sentences", "document", "content", "body", "judgment", "judgement")
REL_KEYS = ("relevant_candidates", "relevant", "relevant_cases", "positives", "labels", "gold")


def _get(rec, keys):
    for k in keys:
        if k in rec:
            return rec[k]
    return None


def _text(v):
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        return " ".join(str(x).strip() for x in v if str(x).strip())
    return str(v)


def _records(obj):
    if isinstance(obj, dict) and not any(k in obj for k in ID_KEYS):
        for k in ("data", "queries", "candidates", "cases", "examples"):
            if isinstance(obj.get(k), (list, dict)):
                return _records(obj[k])
        out = []
        for k, v in obj.items():
            if isinstance(v, dict):
                out.append(dict(v, **({} if _get(v, ID_KEYS) is not None else {"id": k})))
            else:
                out.append({"id": k, "text": v})
        return out
    return list(obj)


def _sid(x):
    return x if isinstance(x, str) else str(x)


def load_queries(path):
    out = {}
    for r in _records(json.load(open(path, encoding="utf-8"))):
        qid = _sid(_get(r, ID_KEYS))
        text = _text(_get(r, TEXT_KEYS))
        assert text is not None, f"query {qid}: no text field among {TEXT_KEYS}; keys {list(r)[:8]}"
        rel = _get(r, REL_KEYS)
        out[qid] = dict(id=qid, text=text, relevant=None if rel is None else [_sid(c) for c in rel])
    return out


def load_pool(path):
    out = {}
    for r in _records(json.load(open(path, encoding="utf-8"))):
        cid = _sid(_get(r, ID_KEYS))
        text = _text(_get(r, TEXT_KEYS))
        assert text is not None, f"candidate {cid}: no text field among {TEXT_KEYS}; keys {list(r)[:8]}"
        out[cid] = dict(id=cid, text=text)
    return out


def tokenize_texts(texts, encode_batch, cache_dir=None, tag=""):
    h = hashlib.md5((tag + "|".join(f"{k}:{len(v)}" for k, v in texts.items())).encode()).hexdigest()[:16]
    path = os.path.join(cache_dir, f"pcr_tok_{h}.pkl") if cache_dir else None
    if path and os.path.exists(path):
        return pickle.load(open(path, "rb"))
    keys = list(texts)
    out = {}
    for i in range(0, len(keys), 256):
        ks = keys[i:i + 256]
        for k, ids in zip(ks, encode_batch([texts[k] for k in ks])):
            out[k] = np.asarray(ids, dtype=np.uint16)
    if path:
        os.makedirs(cache_dir, exist_ok=True)
        tmp = path + f".tmp{os.getpid()}"
        pickle.dump(out, open(tmp, "wb"), protocol=4)
        os.replace(tmp, path)
    return out


def chunk_ids(ids, chunk_len, n_chunks, mode="first", stride=None):
    n, L = len(ids), int(chunk_len)
    if n == 0:
        return [np.zeros(1, np.uint16)]
    if n <= L:
        return [ids]
    k = max(1, int(n_chunks))
    st = int(stride or L)
    starts_all = list(range(0, n - L + 1, st))
    if starts_all[-1] != n - L and mode != "first":
        starts_all.append(n - L)
    if mode == "first":
        starts = starts_all[:k]
    elif mode == "spread":
        if k >= len(starts_all):
            starts = starts_all
        else:
            starts = sorted({starts_all[round(i * (len(starts_all) - 1) / max(1, k - 1))] for i in range(k)}) if k > 1 else [0]
    elif mode == "head_tail":
        h = (k + 1) // 2
        starts = sorted(set(starts_all[:h] + (starts_all[-(k - h):] if k - h else [])))
    else:
        raise ValueError(f"chunk_mode must be first | spread | head_tail, got {mode!r}")
    return [ids[s:s + L] for s in starts]


class BM25:

    def __init__(self, docs, k1=1.2, b=0.75, max_terms=None, vocab=65536):
        from scipy import sparse
        self.keys = list(docs)
        rows, cols, vals, lens = [], [], [], []
        for i, k in enumerate(self.keys):
            ids = docs[k] if max_terms is None else docs[k][:max_terms]
            t, c = np.unique(np.asarray(ids, np.int64), return_counts=True)
            rows.append(np.full(len(t), i)); cols.append(t); vals.append(c.astype(float)); lens.append(c.sum())
        N = len(self.keys)
        rows, cols, vals = (np.concatenate(x) if x else np.zeros(0) for x in (rows, cols, vals))
        tf = sparse.csr_matrix((vals, (rows.astype(np.int64), cols.astype(np.int64))), shape=(N, vocab))
        lens = np.asarray(lens, float)
        df = np.bincount(cols.astype(np.int64), minlength=vocab)
        self.idf = np.log(1 + (N - df + 0.5) / (df + 0.5))
        norm = k1 * (1 - b + b * lens / max(1e-9, lens.mean() if N else 1.0))
        tf = tf.tocoo()
        w = tf.data * (k1 + 1) / (tf.data + norm[tf.row]) * self.idf[tf.col]
        self.W = sparse.csr_matrix((w, (tf.row, tf.col)), shape=(N, vocab)).T.tocsr()

    def scores(self, q_ids, max_terms=None):
        q = np.unique(np.asarray(q_ids if max_terms is None else q_ids[:max_terms], np.int64))
        return np.asarray(self.W[q].sum(0)).ravel()


def rank_pool(scores, keys, exclude=(), k=None):
    order = np.argsort(-np.asarray(scores), kind="stable")
    ex = set(exclude)
    out = [keys[i] for i in order if keys[i] not in ex]
    return out if k is None else out[:k]


def micro_at_k(ranked, gold, k=7):
    tp = n_pred = n_gold = 0
    for q, g in gold.items():
        g = set(g)
        r = list(ranked.get(q, []))[:k]
        tp += len(set(r) & g)
        n_pred += k
        n_gold += len(g)
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    return dict(precision=p, recall=r, f1=2 * p * r / (p + r) if p + r else 0.0, tp=tp, n_queries=len(gold))


def recall_at(ranked, gold, k):
    tp = sum(len(set(list(ranked.get(q, []))[:k]) & set(g)) for q, g in gold.items())
    return tp / max(1, sum(len(g) for g in gold.values()))


def validate_prediction(pred, queries, pool, k=7):
    assert isinstance(pred, dict), "prediction.json must be a JSON object {query_id: [candidate ids]}"
    assert set(pred) == set(queries), f"query ids differ: missing {len(set(queries) - set(pred))}, extra {len(set(pred) - set(queries))}"
    for q, r in pred.items():
        assert isinstance(r, list) and len(r) == k, f"{q}: need exactly {k} candidate ids, got {len(r) if isinstance(r, list) else type(r)}"
        assert len(set(r)) == k, f"{q}: candidate ids must be unique"
        assert all(isinstance(c, str) for c in r), f"{q}: candidate ids must be strings"
        bad = [c for c in r if c not in pool]
        assert not bad, f"{q}: ids not in the candidate pool, e.g. {bad[:3]}"
    return len(pred)


def length_stats(toks):
    n = sorted(len(v) for v in toks.values())
    q = lambda f: int(n[min(len(n) - 1, int(len(n) * f))]) if n else 0
    return dict(n=len(n), p10=q(.1), p50=q(.5), p90=q(.9), p99=q(.99), max=int(n[-1]) if n else 0)