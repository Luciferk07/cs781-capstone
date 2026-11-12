"""cj_data.py — Phase-4 Court Judgment Prediction data handling."""
import hashlib, json, os, pickle
from collections import Counter

import numpy as np

ID_KEYS = ("id", "case_id", "uid", "doc_id", "name")
TEXT_KEYS = ("text", "document", "doc", "judgment", "judgement", "case_text", "content")
LABEL_KEYS = ("label", "labels", "outcome", "decision", "y", "target")


def _get(rec, keys):
    for k in keys:
        if k in rec:
            return rec[k]
    return None


def _norm_label(v):
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        assert len(v) == 1, f"expected one label, got {v}"
        v = v[0]
    v = int(float(v))
    assert v in (0, 1), f"label {v} is not 0/1"
    return v


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
                recs.append(dict(id=k, text=v, label=None))
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
        key, text, lab = str(cid), str(text), _norm_label(_get(r, LABEL_KEYS))
        if key in out:
            if out[key]["text"] == text and out[key]["label"] == lab:
                DUPS["identical_dropped"] += 1
                continue
            j = 2
            while f"{key}#{j}" in out:
                j += 1
            key = f"{key}#{j}"
            DUPS["different_kept"] += 1
        out[key] = dict(id=cid, text=text, label=lab)
    return out


DUPS = Counter()


def tokenize_cases(cases, encode_batch, cache_dir=None, tag=""):
    h = hashlib.md5((tag + "|".join(f"{k}:{len(v['text'])}" for k, v in cases.items())).encode()).hexdigest()[:16]
    path = os.path.join(cache_dir, f"cj_tok_{h}.pkl") if cache_dir else None
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


def binary_scores(pred, gold):
    keys = list(gold)
    tp, fp, fn = Counter(), Counter(), Counter()
    conf = [[0, 0], [0, 0]]
    for k in keys:
        p, g = int(pred[k]), int(gold[k])
        conf[g][p] += 1
        if p == g:
            tp[p] += 1
        else:
            fp[p] += 1; fn[g] += 1
    per = {}
    for c in (0, 1):
        P = tp[c] / (tp[c] + fp[c]) if tp[c] + fp[c] else 0.0
        R = tp[c] / (tp[c] + fn[c]) if tp[c] + fn[c] else 0.0
        per[c] = dict(p=P, r=R, f1=2 * P * R / (P + R) if P + R else 0.0, support=tp[c] + fn[c], predicted=tp[c] + fp[c])
    m = lambda key: (per[0][key] + per[1][key]) / 2
    return dict(macro_f1=m("f1"), macro_p=m("p"), macro_r=m("r"), accuracy=(tp[0] + tp[1]) / max(1, len(keys)),
                per_class=per, confusion=conf, n=len(keys))


def format_scores(sc, title=""):
    c = sc["confusion"]
    s = (f"{title}macro-F1 {sc['macro_f1']:.4f} | macro-P {sc['macro_p']:.4f} | macro-R {sc['macro_r']:.4f} | "
         f"acc {sc['accuracy']:.4f} | n {sc['n']}\n")
    for k in (0, 1):
        v = sc["per_class"][k]
        s += f"  class {k}: F1 {v['f1']:.3f}  P {v['p']:.3f}  R {v['r']:.3f}  (support {v['support']}, predicted {v['predicted']})\n"
    return s + f"  confusion [gold][pred]: {c}"


def best_threshold(p1, gold):
    keys = list(gold)
    cands = sorted(set(round(float(p1[k]), 4) for k in keys))
    best_t, best_f = 0.5, -1.0
    for t in [0.5] + cands:
        f = binary_scores({k: int(p1[k] >= t) for k in keys}, gold)["macro_f1"]
        if f > best_f + 1e-12:
            best_t, best_f = t, f
    return best_t, best_f


def validate_prediction(pred_list, test_cases):
    assert isinstance(pred_list, list), "prediction.json must be a JSON list"
    ids = [r["id"] for r in pred_list]
    assert len(ids) == len(set(map(str, ids))), "duplicate ids in prediction"
    want = {str(v["id"]) for v in test_cases.values()}
    got = set(map(str, ids))
    assert got == want, f"ids differ from test.json: missing {len(want - got)}, extra {len(got - want)}"
    for r in pred_list:
        assert set(r) == {"id", "pred"}, f"each entry must be exactly {{id, pred}}, got {sorted(r)}"
        assert type(r["pred"]) is int and r["pred"] in (0, 1), f"pred must be int 0/1, got {r['pred']!r}"
    return len(pred_list)


def dataset_stats(cases, toks):
    n = sorted(len(toks[k]) for k in cases)
    q = lambda f: n[min(len(n) - 1, int(len(n) * f))] if n else 0
    lab = Counter(v["label"] for v in cases.values())
    return dict(cases=len(cases), labels={str(k): c for k, c in sorted(lab.items(), key=lambda kv: str(kv[0]))},
                tokens_p10=q(.1), tokens_p50=q(.5), tokens_p90=q(.9), tokens_max=n[-1] if n else 0,
                tokens_total=int(sum(n)))