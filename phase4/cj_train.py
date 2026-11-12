"""cj_train.py — Phase-4 Court Judgment Prediction harness."""
import importlib, json, math, os, random, sys, time, zipfile
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

import cj_data as cd

CFG = json.load(open(sys.argv[1]))
D, TR, MODEL_CFG = CFG["data"], CFG["train"], CFG["model"]
KD = CFG.get("kd") or {}
OUT, NAME = CFG["out_dir"], CFG["run_name"]
os.makedirs(OUT, exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = int(TR.get("seed", 0))
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
C = 2


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


def enc_batch(texts):
    return [e.ids for e in TOK.encode_batch(texts)]


PARTS, TOKS = {}, {}
for split in ("train", "dev", "test"):
    if D.get(split):
        raw = cd.load_cases(D[split])
        t = cd.tokenize_cases(raw, enc_batch, D.get("cache_dir", "cj_cache"), tag=split)
        PARTS[split] = {f"{split}::{k}": v for k, v in raw.items()}
        TOKS.update({f"{split}::{k}": v for k, v in t.items()})
TRAIN, DEVS, TEST = PARTS.get("train", {}), PARTS.get("dev", {}), PARTS.get("test", {})
if D.get("train_on_dev"):
    TRAIN = dict(TRAIN, **DEVS)
    DEVS = {}
assert TRAIN and all(v["label"] is not None for v in TRAIN.values()), "train missing or has cases without a label"
tr_lab = Counter(v["label"] for v in TRAIN.values())
log(f"train {len(TRAIN)} cases {dict(tr_lab)} | dev {len(DEVS)} | test {len(TEST)} | chunk_len {L} stride {STRIDE} "
    f"| train max_chunks {MAXC} select {SEL} | eval max_chunks {EMAXC} select {ESEL}")


def spans(key, train):
    n = len(TOKS[key])
    return cd.chunk_doc(n, L, STRIDE, MAXC if train else EMAXC, SEL if train else ESEL)


TEACH = None
if KD.get("teacher_logits"):
    z = np.load(KD["teacher_logits"])
    TEACH = {k[2:]: z[k].astype(np.float32) for k in z.files if k.startswith("d:")}
    covered = {k: v for k, v in TRAIN.items() if k in TEACH}
    assert covered, "teacher logits cover none of the train cases (different data or split names?)"
    if len(covered) < len(TRAIN):
        log(f"KD: teacher covers {len(covered)} of {len(TRAIN)} train cases -> the student trains on those {len(covered)}")
        TRAIN = covered
        tr_lab = Counter(v["label"] for v in TRAIN.values())
    log(f"KD: teacher logits for {len(TEACH)} cases | T={KD['temperature']} alpha={KD['alpha']}")


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
    y = torch.tensor([part[k]["label"] if part[k]["label"] is not None else -100 for k in keys], dtype=torch.long)
    tch = torch.tensor(np.stack([TEACH[k] for k in keys])) if (train and TEACH is not None) else None
    return (x, m, torch.tensor(did), len(keys), torch.tensor(cpos), torch.tensor(crel, dtype=torch.float32)), y, tch


# ------------------------------------------------------------------ model
model_lib = importlib.import_module(CFG.get("model_module", "cj_model"))
model = model_lib.build_cj_model(dict(MODEL_CFG), C).to(DEV)
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
    budget = int(TR.get("eval_chunk_budget", max(1, int(TR["batch_size"])) * int(EMAXC or MAXC or 8)))
    out, i = {}, 0
    while i < len(keys):
        b, used = [], 0
        while i < len(keys) and (not b or used + len(spans(keys[i], False)) <= budget):
            used += len(spans(keys[i], False)); b.append(keys[i]); i += 1
        inp, _, _ = collate(b, part, train=False)
        lg = run_model(inp).cpu().numpy()
        for k, v in zip(b, lg):
            out[k] = v
    model.train()
    return out


THRESH = None


def preds_from(lg):
    if THRESH is None:
        return {k: int(v.argmax()) for k, v in lg.items()}
    return {k: int(torch.softmax(torch.from_numpy(v), -1)[1].item() >= THRESH) for k, v in lg.items()}


def dev_score(lg):
    pred = preds_from(lg)
    return cd.binary_scores(pred, {k: v["label"] for k, v in DEVS.items()}), pred


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
cw = TR.get("class_weights")
if cw == "balanced":
    cw = [len(keys_all) / (2 * max(1, tr_lab[c])) for c in (0, 1)]
cw_t = torch.tensor(cw, dtype=torch.float32, device=DEV) if cw is not None else None
log(f"class weights {cw} | cases/epoch {per_epoch} | steps/epoch {steps_per_epoch} | epochs {epochs}")


def lr_mult(step):
    if step < warm:
        return (step + 1) / max(1, warm)
    t = (step - warm) / max(1, total_steps - warm)
    sch = TR.get("schedule", "linear")
    return 1.0 if sch == "constant" else max(0.0, 1 - t) if sch == "linear" else 0.5 * (1 + math.cos(math.pi * t))


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
        loss = F.cross_entropy(lg, y, weight=cw_t)
        if tch is not None:
            Tk, a = float(KD["temperature"]), float(KD["alpha"])
            kl = F.kl_div(F.log_softmax(lg / Tk, -1), F.softmax(tch.to(DEV) / Tk, -1), reduction="batchmean")
            loss = (1 - a) * loss + a * Tk * Tk * kl
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
        sc, _ = dev_score(predict_logits(DEVS))
        rec.update(dev_macro_f1=sc["macro_f1"], dev_acc=sc["accuracy"],
                   dev_pred1=sc["per_class"][1]["predicted"] / max(1, sc["n"]))
        score = sc["macro_f1"]
    else:
        score = ep
    history.append(rec)
    log(" | ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in rec.items()))
    if score > best:
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
torch.save(dict(trained=best_state, cfg=CFG), os.path.join(OUT, f"{NAME}_trained_params.pt"))

result = dict(run=NAME, chunk_len=L, stride=STRIDE, max_chunks=MAXC, select=SEL, eval_max_chunks=EMAXC, eval_select=ESEL,
              params_total=n_total, params_trainable=n_train, train_minutes=train_min, peak_mem_gb=peak_gb,
              class_weights=cw, history=history)
if DEVS:
    t1 = time.time()
    dlg = predict_logits(DEVS)
    result["dev_infer_seconds"] = time.time() - t1
    sc, _ = dev_score(dlg)
    result["dev_argmax"] = sc
    log(cd.format_scores(sc, "BEST on dev (argmax) "))
    if TR.get("threshold") == "dev_tuned":
        p1 = {k: float(torch.softmax(torch.from_numpy(v), -1)[1]) for k, v in dlg.items()}
        THRESH, f = cd.best_threshold(p1, {k: v["label"] for k, v in DEVS.items()})
        sc, _ = dev_score(dlg)
        log(cd.format_scores(sc, f"BEST on dev (threshold P(1) >= {THRESH:.4f}, tuned on dev) "))
    result.update(dev=sc, threshold=THRESH)
    json.dump([dict(id=DEVS[k]["id"], pred=v, gold=DEVS[k]["label"]) for k, v in preds_from(dlg).items()],
              open(os.path.join(OUT, f"{NAME}_dev_prediction.json"), "w"))
    np.savez_compressed(os.path.join(OUT, f"{NAME}_dev_logits.npz"), **{f"d:{k}": v for k, v in dlg.items()})

if CFG.get("dump_teacher_logits"):
    tr_keys = sorted(PARTS.get("train", {}))
    if CFG.get("teacher_cases"):
        tr_keys = random.Random(SEED).sample(tr_keys, min(len(tr_keys), int(CFG["teacher_cases"])))
    every = {k: PARTS["train"][k] for k in tr_keys}; every.update(PARTS.get("dev", {}))
    t1 = time.time()
    tl = predict_logits(every)
    np.savez_compressed(os.path.join(OUT, f"{NAME}_teacher_logits.npz"), **{f"d:{k}": v for k, v in tl.items()})
    log(f"teacher logits written for {len(tl)} cases ({(time.time() - t1) / 60:.1f} min)")

if TEST:
    t1 = time.time()
    tlg = predict_logits(TEST)
    np.savez_compressed(os.path.join(OUT, f"{NAME}_test_logits.npz"), **{f"d:{k}": v for k, v in tlg.items()})
    pk = preds_from(tlg)
    pred = [dict(id=TEST[k]["id"], pred=int(pk[k])) for k in TEST]
    n = cd.validate_prediction(pred, TEST)
    pj = os.path.join(OUT, f"{NAME}_prediction.json")
    json.dump(pred, open(pj, "w", encoding="utf-8"), ensure_ascii=False)
    zp = os.path.join(OUT, f"{NAME}_submission.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(pj, arcname="prediction.json")
    result.update(test_cases=n, test_infer_seconds=time.time() - t1, test_pred_counts=dict(Counter(r["pred"] for r in pred)))
    log(f"test: {n} cases | predicted {dict(Counter(r['pred'] for r in pred))} | infer {time.time() - t1:.0f}s")
    log(f"wrote {zp}")
json.dump(result, open(os.path.join(OUT, f"{NAME}_result.json"), "w"), indent=1, default=str)
log(f"done: {NAME} | best dev macro-F1 {best:.4f} | train {train_min:.1f} min | peak {peak_gb:.1f} GB | "
    f"trainable {n_train/1e6:.3f}M of {n_total/1e6:.2f}M")