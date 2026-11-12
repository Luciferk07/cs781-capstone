"""lsi_train.py — Phase-5 Legal Statute Identification harness."""
import importlib, json, math, os, random, sys, time, zipfile
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

import lsi_data as ld

CFG = json.load(open(sys.argv[1]))
D, TR, MODEL_CFG = CFG["data"], CFG["train"], CFG["model"]
KD = CFG.get("kd") or {}
OUT, NAME = CFG["out_dir"], CFG["run_name"]
os.makedirs(OUT, exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = int(TR.get("seed", 0))
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ data
from tokenizers import ByteLevelBPETokenizer
TOK = ByteLevelBPETokenizer(vocab=os.path.join(D["tokenizer_dir"], "vocab.json"),
                            merges=os.path.join(D["tokenizer_dir"], "merges.txt"))
PAD_ID = TOK.token_to_id("<pad>") or 0
L, STRIDE = int(D["chunk_len"]), int(D.get("stride") or D["chunk_len"])
MAXC, SEL = D.get("max_chunks"), D.get("select", "first")
EMAXC, ESEL = D.get("eval_max_chunks", MAXC), D.get("eval_select", SEL)
assert MAXC and EMAXC, "set max_chunks and eval_max_chunks to numbers (very long documents would not fit in memory)"


def enc_batch(texts):
    return [e.ids for e in TOK.encode_batch(texts)]


PARTS, TOKS = {}, {}
RAW = {split: ld.load_cases(D[split]) for split in ("train", "dev", "test") if D.get(split)}
LC = D.get("lowercase", False)
if LC == "auto":
    LC = ld.auto_lowercase(RAW["train"], RAW.get("test", RAW.get("dev", {})))
log(f"lowercase: {LC} (setting {D.get('lowercase', False)!r}; all-lowercase docs: "
    + ", ".join(f"{s} {ld.lowercase_share(r):.0%}" for s, r in RAW.items()) + ")")
for split, raw in RAW.items():
    if LC:
        for v in raw.values():
            v["text"] = v["text"].lower()
    t = ld.tokenize_cases(raw, enc_batch, D.get("cache_dir", "lsi_cache"), tag=split + ("-lc" if LC else ""))
    PARTS[split] = {f"{split}::{k}": v for k, v in raw.items()}
    TOKS.update({f"{split}::{k}": v for k, v in t.items()})
TRAIN, DEVS, TEST = PARTS.get("train", {}), PARTS.get("dev", {}), PARTS.get("test", {})
assert TRAIN and all(v["labels"] is not None for v in TRAIN.values()), "train missing or has cases without labels"
LABELS = sorted({l for part in (TRAIN, DEVS) for v in part.values() for l in (v["labels"] or [])})
L2I = {l: i for i, l in enumerate(LABELS)}
C = len(LABELS)
if D.get("train_on_dev"):
    TRAIN = dict(TRAIN, **DEVS)
    DEVS = {}
if TR.get("train_subset"):
    _ks = random.Random(SEED).sample(sorted(TRAIN), min(len(TRAIN), int(TR["train_subset"])))
    TRAIN = {k: TRAIN[k] for k in _ks}


def multihot(keys, part):
    Y = np.zeros((len(keys), C), np.float32)
    for i, k in enumerate(keys):
        for l in part[k]["labels"] or []:
            if l in L2I:
                Y[i, L2I[l]] = 1.0
    return Y


def spans(key, train):
    return ld.chunk_doc(len(TOKS[key]), L, STRIDE, MAXC if train else EMAXC, SEL if train else ESEL)


TEACH = None
if KD.get("teacher_logits"):
    z = np.load(KD["teacher_logits"])
    meta = json.loads(str(z["__meta__"]))
    assert meta["labels"] == LABELS, "teacher used a different statute vocabulary"
    TEACH = {k[2:]: z[k].astype(np.float32) for k in z.files if k.startswith("d:")}
    covered = {k: v for k, v in TRAIN.items() if k in TEACH}
    assert covered, "teacher logits cover none of the train cases"
    if len(covered) < len(TRAIN):
        log(f"KD: teacher covers {len(covered)} of {len(TRAIN)} train cases -> the student trains on those {len(covered)}")
        TRAIN = covered
    log(f"KD: teacher logits for {len(TEACH)} cases | T={KD['temperature']} alpha={KD['alpha']}")

pos = multihot(list(TRAIN), TRAIN).sum(0)
log(f"train {len(TRAIN)} | dev {len(DEVS)} | test {len(TEST)} | statutes {C} | positives per statute: min {int(pos.min())} "
    f"median {int(np.median(pos))} max {int(pos.max())} | statutes with 0 train positives {int((pos == 0).sum())} | "
    f"chunk_len {L} stride {STRIDE} | train max_chunks {MAXC} {SEL} | eval max_chunks {EMAXC} {ESEL}")


def collate(keys, part, train):
    rows, did, cpos, crel = [], [], [], []
    for j, k in enumerate(keys):
        ids = TOKS[k]
        n = max(1, len(ids))
        for p, (a, b) in enumerate(spans(k, train)):
            seg = ids[a:b] if len(ids) else np.asarray([PAD_ID], np.uint16)
            rows.append(seg); did.append(j); cpos.append(p); crel.append(a / n)
    T = max(len(r) for r in rows)
    x = torch.full((len(rows), T), PAD_ID, dtype=torch.long)
    m = torch.zeros((len(rows), T), dtype=torch.bool)
    for i, r in enumerate(rows):
        x[i, :len(r)] = torch.from_numpy(r.astype(np.int64)); m[i, :len(r)] = True
    y = torch.from_numpy(multihot(keys, part)) if all(part[k]["labels"] is not None for k in keys) else None
    tch = torch.tensor(np.stack([TEACH[k] for k in keys])) if (train and TEACH is not None) else None
    return (x, m, torch.tensor(did), len(keys), torch.tensor(cpos), torch.tensor(crel, dtype=torch.float32)), y, tch


# ------------------------------------------------------------------ model
model_lib = importlib.import_module(CFG.get("model_module", "cj_model"))
build = getattr(model_lib, "build_lsi_model", None) or getattr(model_lib, "build_cj_model")
model = build(dict(MODEL_CFG), C).to(DEV)
n_total = sum(p.numel() for p in model.parameters())
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
log(f"parameters: total {n_total/1e6:.2f}M | trainable {n_train/1e6:.3f}M ({100*n_train/max(1,n_total):.2f}%)")
assert n_train > 0, "no trainable parameters"

if CFG.get("load_trained"):
    sd = torch.load(CFG["load_trained"], map_location="cpu", weights_only=False)["trained"]
    cur = dict(model.named_parameters())
    miss = [k for k in sd if k not in cur]
    assert not miss, f"load_trained: names not in this model, e.g. {miss[:3]}"
    with torch.no_grad():
        for k, v in sd.items():
            cur[k].copy_(v.to(cur[k].device, cur[k].dtype))
    log(f"loaded {len(sd)} trained tensors from {CFG['load_trained']}")


def run_model(inp):
    x, m, did, nd, cpos, crel = inp
    with torch.autocast(DEV.type, dtype=torch.float16, enabled=DEV.type == "cuda"):
        lg = model(x.to(DEV), m.to(DEV), did.to(DEV), nd, chunk_pos=cpos.to(DEV), chunk_rel=crel.to(DEV))
    assert tuple(lg.shape) == (nd, C), f"model returned {tuple(lg.shape)}, expected [{nd}, {C}]"
    return lg.float()


@torch.no_grad()
def predict_logits(part):
    model.eval()
    keys = sorted(part, key=lambda k: -len(spans(k, False)))
    budget = int(TR.get("eval_chunk_budget", max(1, int(TR["batch_size"])) * int(EMAXC)))
    out, i = {}, 0
    while i < len(keys):
        b, used = [], 0
        while i < len(keys) and (not b or used + len(spans(keys[i], False)) <= budget):
            used += len(spans(keys[i], False)); b.append(keys[i]); i += 1
        inp, _, _ = collate(b, part, train=False)
        for k, v in zip(b, run_model(inp).cpu().numpy()):
            out[k] = v
    model.train()
    return out


def probs(lg, keys):
    return 1 / (1 + np.exp(-np.stack([lg[k] for k in keys])))


DEV_KEYS = list(DEVS)
Y_DEV = multihot(DEV_KEYS, DEVS).astype(bool) if DEVS else None
MINL = int(TR.get("min_labels", 0))

# ------------------------------------------------------------------ training
bs, acc = int(TR["batch_size"]), int(TR.get("grad_accum", 1))
epochs = int(TR["epochs"])
keys_all = list(TRAIN)
per_epoch = min(len(keys_all), int(TR.get("epoch_cases") or len(keys_all)))
steps_per_epoch = math.ceil(per_epoch / (bs * acc))
total_steps = steps_per_epoch * epochs
warm = int(float(TR.get("warmup_frac", 0)) * total_steps)
base_lr = float(TR["lr"])
groups = model.param_groups(base_lr) if hasattr(model, "param_groups") else \
    [{"params": [p for p in model.parameters() if p.requires_grad]}]
opt = torch.optim.AdamW(groups, lr=base_lr, weight_decay=float(TR.get("weight_decay", 0.0)))
for g in opt.param_groups:
    g.setdefault("initial_lr", g["lr"])
scaler = torch.amp.GradScaler("cuda", enabled=DEV.type == "cuda")

pw = TR.get("pos_weight")
if pw == "balanced":
    neg = len(keys_all) - pos
    pw = np.clip(neg / np.maximum(pos, 1), 1.0, float(TR.get("pos_weight_max", 100.0))).tolist()
elif isinstance(pw, (int, float)):
    pw = [float(pw)] * C
pw_t = torch.tensor(pw, dtype=torch.float32, device=DEV) if pw is not None else None
LOSS = TR.get("loss", "bce")
GAMMA = float(TR.get("focal_gamma", 2.0))
log(f"loss {LOSS} | pos_weight {'none' if pw is None else f'min {min(pw):.2f} max {max(pw):.2f}'} | cases/epoch {per_epoch} | "
    f"steps/epoch {steps_per_epoch} | epochs {epochs}")


def task_loss(lg, y):
    if LOSS == "focal":
        bce = F.binary_cross_entropy_with_logits(lg, y, pos_weight=pw_t, reduction="none")
        p = torch.sigmoid(lg)
        pt = p * y + (1 - p) * (1 - y)
        return ((1 - pt) ** GAMMA * bce).mean()
    return F.binary_cross_entropy_with_logits(lg, y, pos_weight=pw_t)


def lr_mult(step):
    if step < warm:
        return (step + 1) / max(1, warm)
    t = (step - warm) / max(1, total_steps - warm)
    sch = TR.get("schedule", "linear")
    return 1.0 if sch == "constant" else max(0.0, 1 - t) if sch == "linear" else 0.5 * (1 + math.cos(math.pi * t))


SELECT = {"macro_f1": "dev_macro_f1", "tuned_macro_f1": "dev_tuned_macro_f1", "micro_f1": "dev_micro_f1"}[TR.get("select_metric", "macro_f1")]
best, best_state, history = -1.0, None, []
if epochs == 0:
    best_state = {k: v.detach().cpu().clone() for k, v in model.named_parameters() if v.requires_grad}
step, t0 = 0, time.time()
max_min = float(TR.get("max_minutes", 1e9))
if DEV.type == "cuda":
    torch.cuda.reset_peak_memory_stats()
order = []
model.train()
for ep in range(1, epochs + 1):
    if len(order) < per_epoch:
        rest = keys_all[:]; random.shuffle(rest); order += rest
    ep_keys, order = order[:per_epoch], order[per_epoch:]
    run_loss, nb = 0.0, 0
    for i in range(0, len(ep_keys), bs):
        inp, y, tch = collate(ep_keys[i:i + bs], TRAIN, train=True)
        lg = run_model(inp)
        y = y.to(DEV)
        loss = task_loss(lg, y)
        if tch is not None:
            Tk, a = float(KD["temperature"]), float(KD["alpha"])
            soft = torch.sigmoid(tch.to(DEV) / Tk)
            kd = F.binary_cross_entropy_with_logits(lg / Tk, soft)
            loss = (1 - a) * loss + a * Tk * Tk * kd
        scaler.scale(loss / acc).backward()
        nb += 1; run_loss += loss.item()
        if nb % acc == 0:
            for g in opt.param_groups:
                g["lr"] = g["initial_lr"] * lr_mult(step)
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], float(TR.get("grad_clip", 1.0)))
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); step += 1
        if (time.time() - t0) / 60 > max_min:
            break
    rec = dict(epoch=ep, train_loss=run_loss / max(1, nb), minutes=(time.time() - t0) / 60)
    if DEVS:
        Pd = probs(predict_logits(DEVS), DEV_KEYS)
        s5 = ld.scores(ld.decide(Pd, 0.5, MINL), Y_DEV)
        tt, tf = ld.tune_global(Pd, Y_DEV, min_labels=MINL)
        rec.update(dev_macro_f1=s5["macro_f1_present"], dev_micro_f1=s5["micro_f1"], dev_tuned_macro_f1=tf,
                   dev_tuned_t=tt, dev_labels_per_doc=s5["avg_pred_labels"])
        score = rec[SELECT]
    else:
        score = ep
    history.append(rec)
    log(" | ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in rec.items()))
    if score >= best:
        best = score
        best_state = {k: v.detach().cpu().clone() for k, v in model.named_parameters() if v.requires_grad}
        log(f"  best so far (epoch {ep})")
    if (time.time() - t0) / 60 > max_min:
        log(f"time limit {max_min} min reached"); break
train_min = (time.time() - t0) / 60
peak_gb = torch.cuda.max_memory_allocated() / 1e9 if DEV.type == "cuda" else 0.0

cur = dict(model.named_parameters())
with torch.no_grad():
    for k, v in best_state.items():
        cur[k].copy_(v.to(DEV))
torch.save(dict(trained=best_state, cfg=CFG, labels=LABELS), os.path.join(OUT, f"{NAME}_trained_params.pt"))

# ------------------------------------------------------------------ thresholds (fixed 0.5 | global | per_label), on dev
THR = float(TR.get("threshold_value", 0.5))
mode = TR.get("threshold", "fixed")
result = dict(run=NAME, statutes=C, chunk_len=L, stride=STRIDE, max_chunks=MAXC, select=SEL, eval_max_chunks=EMAXC,
              eval_select=ESEL, params_total=n_total, params_trainable=n_train, train_minutes=train_min, peak_mem_gb=peak_gb,
              lowercase=bool(LC), loss=LOSS, pos_weight="none" if pw is None else TR.get("pos_weight"), min_labels=MINL, history=history)
if DEVS:
    t1 = time.time()
    dlg = predict_logits(DEVS)
    result["dev_infer_seconds"] = time.time() - t1
    Pd = probs(dlg, DEV_KEYS)
    s_fixed = ld.scores(ld.decide(Pd, THR, MINL), Y_DEV, LABELS)
    log(ld.format_scores(s_fixed, f"BEST on dev (threshold {THR}) "))
    result["dev_fixed"] = s_fixed
    if mode in ("global", "per_label"):
        tg, _ = ld.tune_global(Pd, Y_DEV, min_labels=MINL)
        THR = tg
        sg = ld.scores(ld.decide(Pd, THR, MINL), Y_DEV, LABELS)
        log(ld.format_scores(sg, f"BEST on dev (global threshold {tg:.2f}, tuned on dev) "))
        result["dev_global"] = sg
        if mode == "per_label":
            THR = ld.tune_per_label(Pd, Y_DEV, tg)
            sp = ld.scores(ld.decide(Pd, THR, MINL), Y_DEV, LABELS)
            log(ld.format_scores(sp, "BEST on dev (per-statute thresholds, tuned on dev) "))
            result["dev_per_label"] = sp
    final = ld.scores(ld.decide(Pd, THR, MINL), Y_DEV, LABELS)
    result.update(dev=final, threshold_mode=mode,
                  thresholds=THR.tolist() if isinstance(THR, np.ndarray) else THR)
    worst = sorted(((v["f1"], l, v["support"]) for l, v in final["per_label"].items() if v["support"]), key=lambda t: t[0])[:8]
    log("  lowest-F1 statutes on dev (F1, id, support): " + ", ".join(f"({f:.2f}, {l}, {s})" for f, l, s in worst))
    Pb = ld.decide(Pd, THR, MINL)
    json.dump([dict(id=DEVS[k]["id"], pred=[LABELS[j] for j in np.flatnonzero(Pb[i])], gold=DEVS[k]["labels"])
               for i, k in enumerate(DEV_KEYS)], open(os.path.join(OUT, f"{NAME}_dev_prediction.json"), "w"))
    np.savez_compressed(os.path.join(OUT, f"{NAME}_dev_logits.npz"), __meta__=json.dumps(dict(labels=LABELS)),
                        **{f"d:{k}": v.astype(np.float16) for k, v in dlg.items()})

if CFG.get("dump_teacher_logits"):
    tr_keys = sorted(PARTS.get("train", {}))
    if CFG.get("teacher_cases"):
        tr_keys = random.Random(SEED).sample(tr_keys, min(len(tr_keys), int(CFG["teacher_cases"])))
    every = {k: PARTS["train"][k] for k in tr_keys}; every.update(PARTS.get("dev", {}))
    t1 = time.time()
    tl = predict_logits(every)
    np.savez_compressed(os.path.join(OUT, f"{NAME}_teacher_logits.npz"), __meta__=json.dumps(dict(labels=LABELS, run=NAME)),
                        **{f"d:{k}": v.astype(np.float32) for k, v in tl.items()})
    log(f"teacher logits written for {len(tl)} cases ({(time.time() - t1) / 60:.1f} min)")

if TEST:
    t1 = time.time()
    tlg = predict_logits(TEST)
    tkeys = list(TEST)
    Pt = probs(tlg, tkeys)
    Pb = ld.decide(Pt, THR, MINL)
    pred = [dict(id=TEST[k]["id"], pred=[int(LABELS[j]) for j in np.flatnonzero(Pb[i])]) for i, k in enumerate(tkeys)]
    n = ld.validate_prediction(pred, TEST, LABELS)
    pj = os.path.join(OUT, f"{NAME}_prediction.json")
    json.dump(pred, open(pj, "w", encoding="utf-8"), ensure_ascii=False)
    zp = os.path.join(OUT, f"{NAME}_submission.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(pj, arcname="prediction.json")
    np.savez_compressed(os.path.join(OUT, f"{NAME}_test_logits.npz"), **{f"d:{k}": v.astype(np.float16) for k, v in tlg.items()})
    lpd = [len(r["pred"]) for r in pred]
    result.update(test_cases=n, test_infer_seconds=time.time() - t1, test_labels_per_doc=float(np.mean(lpd)),
                  test_empty_docs=int(sum(1 for x in lpd if x == 0)))
    log(f"test: {n} docs | statutes per doc {np.mean(lpd):.2f} | docs with none {sum(1 for x in lpd if x == 0)} | infer {time.time() - t1:.0f}s")
    log(f"wrote {zp}")
json.dump(result, open(os.path.join(OUT, f"{NAME}_result.json"), "w"), indent=1, default=str)
log(f"done: {NAME} | best dev {SELECT} {best:.4f} | train {train_min:.1f} min | peak {peak_gb:.1f} GB | "
    f"trainable {n_train/1e6:.3f}M of {n_total/1e6:.2f}M")