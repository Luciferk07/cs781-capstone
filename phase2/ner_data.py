"""ner_data.py — Phase-2 L-NER data handling."""
import json
import zlib
from collections import Counter, defaultdict

LABELS_DEFAULT = ["APP", "RESP", "A.COUNSEL", "R.COUNSEL", "JUDGE", "WIT", "AUTH", "COURT",
                  "STAT", "PREC", "DATE", "CASENO"]


# ------------------------------------------------------------------ loading
def read_labels(path):
    labs = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            labs += [x.strip() for x in line.replace(",", " ").split() if x.strip()]
    return labs


def _as_text_map(obj):
    if isinstance(obj, dict):
        return {str(k): (v if isinstance(v, str) else v.get("text", "")) for k, v in obj.items()}
    out = {}
    for r in obj:
        out[str(r.get("id", r.get("doc_id")))] = r["text"]
    return out


def _parse_ent(e):
    if isinstance(e, (list, tuple)):
        s, en, lab = int(e[0]), int(e[1]), str(e[2])
        txt = e[3] if len(e) > 3 else None
    else:
        s = int(e.get("start", e.get("start_char", e.get("begin"))))
        en = int(e.get("end", e.get("end_char")))
        lab = str(e.get("label", e.get("type", e.get("entity"))))
        txt = e.get("text", e.get("entity_text"))
    return s, en, lab, txt


def _as_gold_map(obj):
    if isinstance(obj, dict):
        items = obj.items()
    else:
        items = [(r.get("id", r.get("doc_id")), r.get("entities", r.get("spans", []))) for r in obj]
    return {str(k): [_parse_ent(e) for e in v] for k, v in items}


def load_split(text_path, gold_path=None):
    texts = _as_text_map(json.load(open(text_path, encoding="utf-8")))
    gold = _as_gold_map(json.load(open(gold_path, encoding="utf-8"))) if gold_path else None
    return texts, gold


def detect_end_convention(texts, gold):
    exc = inc = neither = total = 0
    examples = []
    for d, ents in gold.items():
        t = texts.get(d, "")
        for s, e, lab, txt in ents:
            if txt is None:
                continue
            total += 1
            a, b = t[s:e] == txt, t[s:e + 1] == txt
            exc += a; inc += (b and not a)
            if not a and not b:
                neither += 1
                if len(examples) < 5:
                    examples.append((d, s, e, lab, txt[:60], t[s:e + 1][:60]))
    inclusive = inc > exc
    return dict(inclusive=inclusive, exclusive_matches=exc, inclusive_matches=inc, neither=neither,
                with_text=total, examples=examples)


def to_internal(gold, inclusive):
    return {d: [(s, e + 1 if inclusive else e, lab) for s, e, lab, _ in ents] for d, ents in gold.items()}


def from_internal(spans, inclusive):
    return {d: [[s, e - 1 if inclusive else e, lab] for s, e, lab in v] for d, v in spans.items()}


def split_docs(doc_ids, val_frac, seed=0):
    val = [d for d in doc_ids if (zlib.crc32(f"{seed}:{d}".encode()) % 10000) < val_frac * 10000]
    vs = set(val)
    return [d for d in doc_ids if d not in vs], val


# ------------------------------------------------------------------ tokenization
def trim_offsets(text, offsets):
    out = []
    for s, e in offsets:
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        out.append((s, e))
    return out


def tokenize_doc(text, encode):
    ids, offs = encode(text)
    return list(ids), trim_offsets(text, offs)


# ------------------------------------------------------------------ labels
def label_names(labels, scheme):
    if scheme == "IO":
        return ["O"] + [f"I-{l}" for l in labels]
    if scheme == "BIO":
        return ["O"] + [f"{p}-{l}" for l in labels for p in ("B", "I")]
    if scheme == "BIOES":
        return ["O"] + [f"{p}-{l}" for l in labels for p in ("B", "I", "E", "S")]
    raise ValueError(scheme)


def token_labels(offsets, spans, labels, scheme):
    names = label_names(labels, scheme)
    idx = {n: i for i, n in enumerate(names)}
    tags = [0] * len(offsets)
    owner = [None] * len(offsets)
    skipped = 0
    starts = [s for s, _ in offsets]
    import bisect
    for (S, E, lab) in sorted(spans, key=lambda x: (x[0], -(x[1] - x[0]))):
        if lab not in labels:
            continue
        lo = max(0, bisect.bisect_left(starts, S) - 1)
        toks = []
        j = lo
        while j < len(offsets) and offsets[j][0] < E:
            s, e = offsets[j]
            if e > s and s < E and e > S:
                toks.append(j)
            j += 1
        if not toks:
            continue
        if any(owner[t] is not None for t in toks):
            skipped += 1
            continue

        full = list(range(toks[0], toks[-1] + 1))
        for k, t in enumerate(full):
            owner[t] = (S, E, lab)
            if scheme == "IO":
                name = f"I-{lab}"
            elif scheme == "BIO":
                name = f"{'B' if k == 0 else 'I'}-{lab}"
            else:
                if len(full) == 1:
                    name = f"S-{lab}"
                else:
                    name = f"{'B' if k == 0 else ('E' if k == len(full) - 1 else 'I')}-{lab}"
            tags[t] = idx[name]
    return tags, skipped


def decode_spans(tag_ids, offsets, labels, scheme, mode="lenient"):
    names = label_names(labels, scheme)
    if scheme == "IO":
        mode = "lenient"
    out = []
    cur = None

    def close():
        nonlocal cur
        if cur is not None and cur[2] > cur[1]:
            out.append((cur[1], cur[2], cur[0]))
        cur = None

    prev = None
    for t, tid in enumerate(tag_ids):
        s, e = offsets[t]
        name = names[tid]


        same_char = prev is not None and (s, e) == prev and e > s
        prev = (s, e)
        if same_char and cur is not None:
            continue
        if name == "O":
            close(); continue
        p, lab = name.split("-", 1)
        if e <= s:
            if cur is not None and cur[0] != lab:
                close()
            continue
        if p in ("B", "S") or cur is None or cur[0] != lab:
            if p in ("I", "E") and mode == "strict" and (cur is None or cur[0] != lab):
                close(); continue
            close()
            cur = [lab, s, e]
        else:
            cur[2] = e
        if p in ("E", "S"):
            close()
    close()
    seen, uniq = set(), []
    for x in out:
        if x not in seen:
            seen.add(x); uniq.append(x)
    return uniq


# ------------------------------------------------------------------ windows
def make_windows(n_tokens, window, stride):
    if n_tokens <= window:
        return [0]
    starts = list(range(0, n_tokens - window + 1, stride))
    if starts[-1] + window < n_tokens:
        starts.append(n_tokens - window)
    return starts


# ------------------------------------------------------------------ scoring
def strict_scores(pred, gold, labels):
    tp, fp, fn = Counter(), Counter(), Counter()
    for d in set(gold) | set(pred):
        g = set(map(tuple, gold.get(d, [])))
        p = set(map(tuple, pred.get(d, [])))
        for x in p & g: tp[x[2]] += 1
        for x in p - g: fp[x[2]] += 1
        for x in g - p: fn[x[2]] += 1
    per = {}
    for l in labels:
        P = tp[l] / (tp[l] + fp[l]) if tp[l] + fp[l] else 0.0
        R = tp[l] / (tp[l] + fn[l]) if tp[l] + fn[l] else 0.0
        F = 2 * P * R / (P + R) if P + R else 0.0
        per[l] = dict(p=P, r=R, f1=F, tp=tp[l], fp=fp[l], fn=fn[l], support=tp[l] + fn[l])
    present = [l for l in labels if per[l]["support"] or per[l]["fp"]]
    macro = sum(per[l]["f1"] for l in labels) / len(labels)
    macro_present = sum(per[l]["f1"] for l in present) / max(1, len(present))
    mp = sum(per[l]["p"] for l in labels) / len(labels)
    mr = sum(per[l]["r"] for l in labels) / len(labels)
    return dict(macro_f1=macro, macro_f1_present=macro_present, macro_p=mp, macro_r=mr, per_label=per)


def format_scores(sc):
    lines = [f"strict macro-F1 {sc['macro_f1']:.4f} (over labels present: {sc['macro_f1_present']:.4f}) | "
             f"macro-P {sc['macro_p']:.4f} | macro-R {sc['macro_r']:.4f}"]
    for l, v in sc["per_label"].items():
        lines.append(f"  {l:10s} F1 {v['f1']:.3f}  P {v['p']:.3f}  R {v['r']:.3f}  "
                     f"(tp {v['tp']}, fp {v['fp']}, fn {v['fn']}, support {v['support']})")
    return "\n".join(lines)


# ------------------------------------------------------------------ checks + stats
def oracle_check(texts, spans, encode, labels, scheme, mode="lenient"):
    pred, skipped, mid_token = {}, 0, Counter()
    for d, sp in spans.items():
        ids, offs = tokenize_doc(texts[d], encode)
        tags, sk = token_labels(offs, sp, labels, scheme)
        skipped += sk
        pred[d] = decode_spans(tags, offs, labels, scheme, mode)
        starts = {s for s, e in offs if e > s}
        ends = {e for s, e in offs if e > s}
        for s, e, lab in sp:
            if s not in starts or e not in ends:
                mid_token[lab] += 1
    sc = strict_scores(pred, spans, labels)
    return sc, skipped, mid_token


def dataset_stats(texts, spans, encode, labels):
    lens, n_tok = [], 0
    for d, t in texts.items():
        ids, _ = encode(t)
        lens.append(len(ids)); n_tok += len(ids)
    lens.sort()
    cnt = Counter(lab for v in spans.values() for _, _, lab in v) if spans else Counter()
    overl = 0
    if spans:
        for v in spans.values():
            v = sorted(v)
            for a, b in zip(v, v[1:]):
                if b[0] < a[1]:
                    overl += 1
    unknown = sorted({lab for v in (spans or {}).values() for _, _, lab in v} - set(labels))
    return dict(docs=len(texts), tokens=n_tok,
                doc_tokens_p50=lens[len(lens) // 2] if lens else 0,
                doc_tokens_p90=lens[int(len(lens) * 0.9)] if lens else 0,
                doc_tokens_max=lens[-1] if lens else 0,
                entities=sum(cnt.values()), per_label=dict(cnt.most_common()),
                overlapping_pairs=overl, unknown_labels=unknown)


# ------------------------------------------------------------------ submission
def validate_prediction(pred, texts, labels):
    missing = set(texts) - set(pred)
    extra = set(pred) - set(texts)
    assert not missing, f"{len(missing)} test docs missing, e.g. {list(missing)[:3]}"
    assert not extra, f"{len(extra)} unknown doc ids, e.g. {list(extra)[:3]}"
    n = 0
    for d, v in pred.items():
        assert isinstance(v, list), d
        seen = set()
        for x in v:
            assert isinstance(x, list) and len(x) in (3, 4), (d, x)
            s, e, lab = x[0], x[1], x[2]
            assert isinstance(s, int) and isinstance(e, int) and 0 <= s <= e <= len(texts[d]), (d, x)
            assert lab in labels, (d, x)
            assert (s, e, lab) not in seen, f"duplicate {x} in {d}"
            seen.add((s, e, lab)); n += 1
    return n