"""sum_data.py — Phase-6 abstractive summarization data handling."""
import hashlib, json, os, pickle, random, re
from collections import Counter

import numpy as np

ID_KEYS = ("id", "doc_id", "case_id", "uid", "name")
TEXT_KEYS = ("document", "text", "judgment", "judgement", "source", "doc")
SUM_KEYS = ("summary", "abstract", "headnote", "target", "reference")


def _get(rec, keys):
    for k in keys:
        if k in rec:
            return rec[k]
    return None


def _join(v):
    return "\n".join(str(x) for x in v) if isinstance(v, (list, tuple)) else (None if v is None else str(v))


def load_cases(path):
    obj = json.load(open(path, encoding="utf-8"))
    if isinstance(obj, dict) and not any(k in obj for k in ID_KEYS):
        for k in ("data", "cases", "examples", "instances"):
            if isinstance(obj.get(k), list):
                obj = obj[k]
                break
    recs = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            recs.append(dict(id=k, document=v) if isinstance(v, str) else dict(v, id=v.get("id", k)))
    else:
        recs = list(obj)
    out, dup = {}, 0
    for i, r in enumerate(recs):
        cid = _get(r, ID_KEYS)
        cid = i if cid is None else cid
        text = _join(_get(r, TEXT_KEYS))
        assert text is not None, f"record {cid}: no document field among {TEXT_KEYS}; keys {list(r)[:8]}"
        key = str(cid)
        if key in out:
            dup += 1
            key = f"{key}#{dup}"
        out[key] = dict(id=cid, text=text, summary=_join(_get(r, SUM_KEYS)))
    return out


def split_train_val(keys, val_cases, seed=0):
    keys = sorted(keys)
    rng = random.Random(seed)
    val = set(rng.sample(keys, min(len(keys) // 2, int(val_cases)))) if val_cases else set()
    return [k for k in keys if k not in val], [k for k in keys if k in val]


def tokenize_texts(texts, encode_batch, cache_dir=None, tag=""):
    h = hashlib.md5((tag + "|".join(f"{k}:{len(v)}" for k, v in texts.items())).encode()).hexdigest()[:16]
    path = os.path.join(cache_dir, f"sum_tok_{h}.pkl") if cache_dir else None
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


def select_src(ids, max_src, mode="first", head_frac=0.5):
    n = len(ids)
    if n <= max_src:
        return ids
    if mode == "first":
        return ids[:max_src]
    if mode == "last":
        return ids[n - max_src:]
    if mode == "head_tail":
        h = int(round(max_src * float(head_frac)))
        return np.concatenate([ids[:h], ids[n - (max_src - h):]]) if max_src - h else ids[:h]
    raise ValueError(f"src_select must be first | last | head_tail, got {mode!r}")


_W = re.compile(r"[a-z0-9]+")


def _words(t):
    return _W.findall(t.lower())


def _ngrams(w, n):
    return Counter(tuple(w[i:i + n]) for i in range(len(w) - n + 1))


def _f(overlap, p_tot, r_tot):
    if not overlap or not p_tot or not r_tot:
        return 0.0
    p, r = overlap / p_tot, overlap / r_tot
    return 2 * p * r / (p + r)


def _lcs(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def rouge_pair(pred, ref):
    p, r = _words(pred), _words(ref)
    out = {}
    for n in (1, 2):
        pn, rn = _ngrams(p, n), _ngrams(r, n)
        out[f"rouge{n}"] = _f(sum((pn & rn).values()), sum(pn.values()), sum(rn.values()))
    out["rougeL"] = _f(_lcs(p, r), len(p), len(r))
    return out


def rouge(preds, refs):
    rows = [rouge_pair(p, r) for p, r in zip(preds, refs)]
    return {k: float(np.mean([x[k] for x in rows])) if rows else 0.0 for k in ("rouge1", "rouge2", "rougeL")}


def validate_prediction(pred_list, test_cases):
    assert isinstance(pred_list, list), "prediction.json must be a JSON list"
    ids = [str(r["id"]) for r in pred_list]
    assert len(ids) == len(set(ids)), "duplicate ids in prediction"
    want = {str(v["id"]) for v in test_cases.values()}
    assert set(ids) == want, f"ids differ from test.json: missing {len(want - set(ids))}, extra {len(set(ids) - want)}"
    for r in pred_list:
        assert set(r) == {"id", "summary"}, f"each entry must be exactly {{id, summary}}, got {sorted(r)}"
        assert isinstance(r["summary"], str), f"summary must be a string for {r['id']}"
    empty = sum(1 for r in pred_list if not r["summary"].strip())
    return len(pred_list), empty


def length_stats(toks):
    n = sorted(len(v) for v in toks.values())
    q = lambda f: int(n[min(len(n) - 1, int(len(n) * f))]) if n else 0
    return dict(n=len(n), p10=q(.1), p50=q(.5), p90=q(.9), p99=q(.99), max=int(n[-1]) if n else 0)