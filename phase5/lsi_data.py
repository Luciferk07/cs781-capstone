"""lsi_data.py — Phase-5 Legal Statute Identification data handling."""
import hashlib, json, os, pickle
from collections import Counter

import numpy as np

ID_KEYS = ("id", "case_id", "uid", "doc_id", "name")
TEXT_KEYS = ("text", "document", "doc", "judgment", "judgement", "case_text", "content", "facts")
LABEL_KEYS = ("labels", "label", "statutes", "statute_ids", "statute", "targets")
DUPS = Counter()


def _get(rec, keys):
    for k in keys:
        if k in rec:
            return rec[k]
    return None


def _norm_labels(v):
    if v is None:
        return None
    if not isinstance(v, (list, tuple)):
        v = [v]
    return sorted({int(float(x)) for x in v})


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
            if isinstance(v, str):
                recs.append(dict(id=k, text=v))
            else:
                r = dict(v); r.setdefault("id", k); recs.append(r)
    else:
        recs = list(obj)
    out = {}
    for i, r in enumerate(recs):
        cid = _get(r, ID_KEYS)
        cid = i if cid is None else cid
        text = _get(r, TEXT_KEYS)
        assert text is not None, f"record {cid}: no text field among {TEXT_KEYS}; keys {list(r)[:8]}"
        if isinstance(text, (list, tuple)):
            text = "\n".join(str(t) for t in text)
        key, text, lab = str(cid), str(text), _norm_labels(_get(r, LABEL_KEYS))
        if key in out:
            if out[key]["text"] == text and out[key]["labels"] == lab:
                DUPS["identical_dropped"] += 1
                continue
            j = 2
            while f"{key}#{j}" in out:
                j += 1
            key = f"{key}#{j}"
            DUPS["different_kept"] += 1
        out[key] = dict(id=cid, text=text, labels=lab)
    return out


def lowercase_share(cases):
    texts = [v["text"] for v in cases.values()]
    return sum(1 for t in texts if not any(c.isupper() for c in t[:5000])) / max(1, len(texts))


def auto_lowercase(train_cases, test_cases):
    return lowercase_share(test_cases) >= 0.9 and lowercase_share(train_cases) < 0.5


def tokenize_cases(cases, encode_batch, cache_dir=None, tag=""):
    h = hashlib.md5((tag + "|".join(f"{k}:{len(v['text'])}" for k, v in cases.items())).encode()).hexdigest()[:16]
    path = os.path.join(cache_dir, f"lsi_tok_{h}.pkl") if cache_dir else None
    if path and os.path.exists(path):
        return pickle.load(open(path, "rb"))
    keys = list(cases)
    out = {}
    for i in range(0, len(keys), 512):
        ks = keys[i:i + 512]
        for k, ids in zip(ks, encode_batch([cases[k]["text"] for k in ks])):
            out[k] = np.asarray(ids, dtype=np.uint16)
    if path:
        os.makedirs(cache_dir, exist_ok=True)
        tmp = path + f".tmp{os.getpid()}"
        pickle.dump(out, open(tmp, "wb"), protocol=4)
        os.replace(tmp, path)
    return out


def chunk_doc(n_tokens, chunk_len, stride=None, max_chunks=None, select="first"):
    stride = int(stride or chunk_len)
    assert 0 < stride <= chunk_len, "need 0 < stride <= chunk_len"
    if n_tokens <= chunk_len:
        spans = [(0, max(1, n_tokens))]
    else:
        starts = list(range(0, n_tokens - chunk_len + 1, stride))
        if starts[-1] + chunk_len < n_tokens:
            starts.append(n_tokens - chunk_len)
        spans = [(s, s + chunk_len) for s in starts]
    k = len(spans) if not max_chunks else min(int(max_chunks), len(spans))
    if select == "all" or k == len(spans):
        return spans
    if select == "first":
        return spans[:k]
    if select == "last":
        return spans[-k:]
    if select == "head_tail":
        h = (k + 1) // 2
        return spans[:h] + spans[len(spans) - (k - h):] if k - h else spans[:h]
    if select == "spread":
        idx = sorted({round(i * (len(spans) - 1) / max(1, k - 1)) for i in range(k)}) if k > 1 else [0]
        return [spans[i] for i in idx]
    raise ValueError(f"select must be first | last | head_tail | spread | all, got {select!r}")


def _prf(tp, fp, fn):
    p = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0)
    r = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) > 0)
    f = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    return p, r, f


def scores(Pb, Y, labels=None):
    Pb, Y = np.asarray(Pb, bool), np.asarray(Y, bool)
    tp = (Pb & Y).sum(0).astype(float); fp = (Pb & ~Y).sum(0).astype(float); fn = (~Pb & Y).sum(0).astype(float)
    p, r, f = _prf(tp, fp, fn)
    sup = Y.sum(0)
    present = sup > 0
    mp, mr, mf = _prf(np.array([tp.sum()]), np.array([fp.sum()]), np.array([fn.sum()]))
    out = dict(macro_f1=float(f.mean()), macro_f1_present=float(f[present].mean()) if present.any() else 0.0,
               macro_p_present=float(p[present].mean()) if present.any() else 0.0,
               macro_r_present=float(r[present].mean()) if present.any() else 0.0,
               micro_f1=float(mf[0]), micro_p=float(mp[0]), micro_r=float(mr[0]),
               subset_acc=float((Pb == Y).all(1).mean()), n=int(len(Y)), labels_present=int(present.sum()),
               avg_pred_labels=float(Pb.sum(1).mean()), avg_gold_labels=float(Y.sum(1).mean()),
               empty_pred_docs=int((Pb.sum(1) == 0).sum()))
    if labels is not None:
        out["per_label"] = {int(l): dict(f1=float(f[i]), p=float(p[i]), r=float(r[i]), support=int(sup[i]),
                                         predicted=int(tp[i] + fp[i])) for i, l in enumerate(labels)}
    return out


def format_scores(sc, title=""):
    return (f"{title}macro-F1 (labels in gold) {sc['macro_f1_present']:.4f} [{sc['labels_present']} labels] | "
            f"macro-F1 (all labels) {sc['macro_f1']:.4f} | micro-F1 {sc['micro_f1']:.4f} (P {sc['micro_p']:.3f} R {sc['micro_r']:.3f}) | "
            f"subset acc {sc['subset_acc']:.4f} | labels/doc pred {sc['avg_pred_labels']:.2f} gold {sc['avg_gold_labels']:.2f} | "
            f"docs with no prediction {sc['empty_pred_docs']} | n {sc['n']}")


GRID = np.round(np.arange(0.02, 0.981, 0.01), 2)


def decide(P, thr, min_labels=0):
    Pb = P >= np.asarray(thr)
    if min_labels:
        top = np.argsort(-P, axis=1)[:, :int(min_labels)]
        rows = np.arange(len(P))[:, None]
        need = Pb.sum(1) < int(min_labels)
        Pb[rows[need], top[need]] = True
    return Pb


def tune_global(P, Y, metric="macro_f1_present", min_labels=0):
    best_t, best = 0.5, -1.0
    for t in GRID:
        s = scores(decide(P, t, min_labels), Y)[metric]
        if s > best + 1e-12:
            best_t, best = float(t), s
    return best_t, best


def tune_per_label(P, Y, fallback):
    Y = np.asarray(Y, bool)
    thr = np.full(P.shape[1], float(fallback))
    for j in range(P.shape[1]):
        if not Y[:, j].any():
            continue
        pb = P[:, [j]] >= GRID[None, :]
        tp = (pb & Y[:, [j]]).sum(0); fp = (pb & ~Y[:, [j]]).sum(0); fn = (~pb & Y[:, [j]]).sum(0)
        f = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros(len(GRID)), where=(2 * tp + fp + fn) > 0)
        thr[j] = float(GRID[int(f.argmax())])
    return thr


def validate_prediction(pred_list, test_cases, valid_labels):
    assert isinstance(pred_list, list), "prediction.json must be a JSON list"
    ids = [str(r["id"]) for r in pred_list]
    assert len(ids) == len(set(ids)), "duplicate ids in prediction"
    want = {str(v["id"]) for v in test_cases.values()}
    assert set(ids) == want, f"ids differ from test.json: missing {len(want - set(ids))}, extra {len(set(ids) - want)}"
    ok = set(int(l) for l in valid_labels)
    for r in pred_list:
        assert set(r) == {"id", "pred"}, f"each entry must be exactly {{id, pred}}, got {sorted(r)}"
        p = r["pred"]
        assert isinstance(p, list) and all(type(x) is int for x in p), f"pred must be a list of ints: {p!r}"
        assert len(p) == len(set(p)), f"duplicate statute ids in {r['id']}"
        assert set(p) <= ok, f"unknown statute ids {sorted(set(p) - ok)[:5]} in {r['id']}"
    return len(pred_list)


def dataset_stats(cases, toks):
    n = sorted(len(toks[k]) for k in cases)
    q = lambda f: n[min(len(n) - 1, int(len(n) * f))] if n else 0
    labs = [v["labels"] for v in cases.values() if v["labels"] is not None]
    cnt = Counter(l for ls in labs for l in ls)
    per = Counter(len(ls) for ls in labs)
    freq = sorted(cnt.values())
    return dict(cases=len(cases), tokens_p50=q(.5), tokens_p90=q(.9), tokens_max=n[-1] if n else 0,
                labelled=len(labs), distinct_labels=len(cnt),
                labels_per_doc={str(k): v for k, v in sorted(per.items())[:12]},
                label_freq_min=freq[0] if freq else 0, label_freq_p50=freq[len(freq) // 2] if freq else 0,
                label_freq_max=freq[-1] if freq else 0, labels_under_10_docs=sum(1 for c in freq if c < 10))