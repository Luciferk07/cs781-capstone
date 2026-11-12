"""ner_train.py — Phase-2 L-NER fine-tuning harness."""
import importlib, json, math, os, random, sys, time, zipfile

import numpy as np
import torch
import torch.nn.functional as F

import ner_data as nd

CFG = json.load(open(sys.argv[1]))
D, TR, MODEL_CFG = CFG["data"], CFG["train"], CFG["model"]
KD = CFG.get("kd") or {}
OUT = CFG["out_dir"]; NAME = CFG["run_name"]
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
PAD_ID = TOK.token_to_id("<pad>")
PAD_ID = 0 if PAD_ID is None else PAD_ID


def encode(text):
    e = TOK.encode(text)
    return e.ids, e.offsets


LABELS = nd.read_labels(D["labels"]) if D.get("labels") else nd.LABELS_DEFAULT
SCHEME, MODE = D["scheme"], D.get("decode", "lenient")
TAGS = nd.label_names(LABELS, SCHEME)
C = len(TAGS)
W, STRIDE = int(D["window"]), int(D["stride"])

texts, gold_raw = {}, {}
for tp, gp in D["train_files"]:
    t, g = nd.load_split(tp, gp)
    texts.update(t); gold_raw.update(g)
conv = nd.detect_end_convention(texts, gold_raw)
INCL = conv["inclusive"]
gold = nd.to_internal(gold_raw, INCL)
test_texts = nd.load_split(D["test_file"])[0] if D.get("test_file") else {}
log(f"labels {len(LABELS)} | scheme {SCHEME} -> {C} tags | gold end is {'INCLUSIVE' if INCL else 'exclusive'} "
    f"({conv['exclusive_matches']} exclusive / {conv['inclusive_matches']} inclusive / {conv['neither']} neither)")

doc_ids = sorted(texts)
if float(D.get("val_frac", 0)) > 0:
    tr_ids, va_ids = nd.split_docs(doc_ids, float(D["val_frac"]), int(D.get("split_seed", 0)))
else:
    tr_ids, va_ids = doc_ids, []
log(f"train docs {len(tr_ids)} | val docs {len(va_ids)} | test docs {len(test_texts)}")

TOKD = {}


def tok(d, src):
    key = (id(src), d)
    if key not in TOKD:
        TOKD[key] = nd.tokenize_doc(src[d], encode)
    return TOKD[key]


# ---- teacher logits for KD (per doc, per token)
TEACH = None
if KD.get("teacher_logits"):
    z = np.load(KD["teacher_logits"])
    meta = json.loads(str(z["__meta__"]))
    assert meta["tags"] == TAGS, "teacher was trained with a different label scheme"
    TEACH = {k[2:]: z[k] for k in z.files if k.startswith("d:")}
    log(f"KD: teacher logits for {len(TEACH)} docs, T={KD['temperature']} alpha={KD['alpha']}")


def build_windows(ids_list, with_tags):
    out, skipped = [], 0
    for d in ids_list:
        ids, offs = tok(d, texts if with_tags else test_texts)
        tags = None
        if with_tags:
            tags, sk = nd.token_labels(offs, gold.get(d, []), LABELS, SCHEME)
            skipped += sk
        for s in nd.make_windows(len(ids), W, STRIDE):
            out.append((d, s, ids[s:s + W], tags[s:s + W] if tags else None))
    return out, skipped


def collate(batch, with_tags=True, with_teacher=False):
    B = len(batch)
    x = torch.full((B, W), PAD_ID, dtype=torch.long)
    m = torch.zeros((B, W), dtype=torch.bool)
    y = torch.full((B, W), -100, dtype=torch.long)
    tch = torch.zeros((B, W, C), dtype=torch.float32) if (with_teacher and TEACH is not None) else None
    for i, (d, s, ids, tags) in enumerate(batch):
        n = len(ids)
        x[i, :n] = torch.tensor(ids); m[i, :n] = True
        if with_tags and tags is not None:
            y[i, :n] = torch.tensor(tags)
        if tch is not None:
            tch[i, :n] = torch.from_numpy(TEACH[d][s:s + n].astype(np.float32))
    return x, m, y, tch


# ------------------------------------------------------------------ model
model_lib = importlib.import_module(CFG.get("model_module", "ner_model"))
model = model_lib.build_ner_model(dict(MODEL_CFG), C).to(DEV)
n_total = sum(p.numel() for p in model.parameters())
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
log(f"parameters: total {n_total/1e6:.2f}M | trainable {n_train/1e6:.3f}M ({100*n_train/max(1,n_total):.2f}%)")
assert n_train > 0, "no trainable parameters"


@torch.no_grad()
def predict_logits(doc_list, src):
    model.eval()
    out = {}
    wins = []
    for d in doc_list:
        ids, _ = tok(d, src)
        out[d] = (np.zeros((len(ids), C), np.float32), np.zeros(len(ids), np.float32))
        for s in nd.make_windows(len(ids), W, STRIDE):
            wins.append((d, s, ids[s:s + W], None))
    bs = int(TR.get("eval_batch_size", TR["batch_size"]))
    for i in range(0, len(wins), bs):
        b = wins[i:i + bs]
        x, m, _, _ = collate(b, with_tags=False)
        with torch.autocast(DEV.type, dtype=torch.float16, enabled=DEV.type == "cuda"):
            lg = model(x.to(DEV), m.to(DEV)).float().cpu().numpy()
        for j, (d, s, ids, _) in enumerate(b):
            n = len(ids)
            out[d][0][s:s + n] += lg[j, :n]; out[d][1][s:s + n] += 1
    model.train()
    return {d: a / np.maximum(c, 1)[:, None] for d, (a, c) in out.items()}


def spans_from_logits(lg, src):
    res = {}
    for d, L in lg.items():
        _, offs = tok(d, src)
        res[d] = nd.decode_spans(L.argmax(-1).tolist(), offs, LABELS, SCHEME, MODE)
    return res


# ------------------------------------------------------------------ training
train_w, skipped = build_windows(tr_ids, True)
log(f"train windows {len(train_w)} (window {W}, stride {STRIDE}) | overlapping gold entities not representable: {skipped}")
bs, acc = int(TR["batch_size"]), int(TR.get("grad_accum", 1))
epochs = int(TR["epochs"])
steps_per_epoch = math.ceil(len(train_w) / (bs * acc))
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
cw = torch.tensor(cw, dtype=torch.float32, device=DEV) if cw is not None else None
if cw is not None:
    assert len(cw) == C, f"class_weights needs {C} values, one per tag {TAGS}"


def lr_mult(step):
    if step < warm:
        return (step + 1) / max(1, warm)
    sch = TR.get("schedule", "linear")
    t = (step - warm) / max(1, total_steps - warm)
    return 1.0 if sch == "constant" else max(0.0, 1 - t) if sch == "linear" else 0.5 * (1 + math.cos(math.pi * t))


best, best_state, history = -1.0, None, []
step, t0 = 0, time.time()
max_min = float(TR.get("max_minutes", 1e9))
if DEV.type == "cuda":
    torch.cuda.reset_peak_memory_stats()
model.train()
for ep in range(1, epochs + 1):
    random.shuffle(train_w)
    run_loss, nb = 0.0, 0
    for i in range(0, len(train_w), bs):
        x, m, y, tch = collate(train_w[i:i + bs], with_teacher=True)
        x, m, y = x.to(DEV), m.to(DEV), y.to(DEV)
        with torch.autocast(DEV.type, dtype=torch.float16, enabled=DEV.type == "cuda"):
            lg = model(x, m)
        lg = lg.float()
        loss = F.cross_entropy(lg.reshape(-1, C), y.reshape(-1), weight=cw, ignore_index=-100)
        if tch is not None:
            T, a = float(KD["temperature"]), float(KD["alpha"])
            v = (y != -100) if KD.get("only_labelled", True) else m
            kl = F.kl_div(F.log_softmax(lg[v] / T, -1), F.softmax(tch.to(DEV)[v] / T, -1), reduction="batchmean")
            loss = (1 - a) * loss + a * T * T * kl
        scaler.scale(loss / acc).backward()
        nb += 1; run_loss += loss.item()
        if nb % acc == 0:
            for g in opt.param_groups:
                g["lr"] = g["initial_lr"] * lr_mult(step)
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                           float(TR.get("grad_clip", 1.0)))
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); step += 1
        if (time.time() - t0) / 60 > max_min:
            break
    rec = dict(epoch=ep, train_loss=run_loss / max(1, nb), minutes=(time.time() - t0) / 60)
    if va_ids:
        vs = nd.strict_scores(spans_from_logits(predict_logits(va_ids, texts), texts),
                              {d: gold.get(d, []) for d in va_ids}, LABELS)
        rec.update(val_macro_f1=vs["macro_f1"], val_macro_p=vs["macro_p"], val_macro_r=vs["macro_r"])
        score = vs["macro_f1"]
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
torch.save(dict(trained=best_state, cfg=CFG, tags=TAGS), os.path.join(OUT, f"{NAME}_trained_params.pt"))

result = dict(run=NAME, tags=TAGS, scheme=SCHEME, decode=MODE, window=W, stride=STRIDE,
              params_total=n_total, params_trainable=n_train, train_minutes=train_min, peak_mem_gb=peak_gb,
              history=history, gold_end_inclusive=INCL)
if va_ids:
    t1 = time.time()
    vl = predict_logits(va_ids, texts)
    vsp = spans_from_logits(vl, texts)
    vs = nd.strict_scores(vsp, {d: gold.get(d, []) for d in va_ids}, LABELS)
    result.update(val=vs, val_infer_seconds=time.time() - t1)
    log("BEST checkpoint on validation:\n" + nd.format_scores(vs))
    json.dump(nd.from_internal(vsp, INCL), open(os.path.join(OUT, f"{NAME}_val_prediction.json"), "w"))

# ---- teacher dump for knowledge distillation (all training docs, best weights)
if CFG.get("dump_teacher_logits"):
    tl = predict_logits(doc_ids, texts)
    np.savez_compressed(os.path.join(OUT, f"{NAME}_teacher_logits.npz"),
                        __meta__=json.dumps(dict(tags=TAGS, run=NAME, window=W, stride=STRIDE)),
                        **{f"d:{d}": v.astype(np.float16) for d, v in tl.items()})
    log(f"teacher logits written for {len(tl)} docs")

# ---- test set -> prediction.json + submission zip
if test_texts:
    t1 = time.time()
    tsp = spans_from_logits(predict_logits(sorted(test_texts), test_texts), test_texts)
    pred = nd.from_internal(tsp, INCL)
    n_ent = nd.validate_prediction(pred, test_texts, LABELS)
    pj = os.path.join(OUT, f"{NAME}_prediction.json")
    json.dump(pred, open(pj, "w", encoding="utf-8"), ensure_ascii=False)
    zp = os.path.join(OUT, f"{NAME}_submission.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(pj, arcname="prediction.json")
    result.update(test_entities=n_ent, test_infer_seconds=time.time() - t1)
    from collections import Counter
    log(f"test: {n_ent} entities in {len(pred)} docs | per label {dict(Counter(x[2] for v in pred.values() for x in v))}")
    log(f"wrote {zp}")
json.dump(result, open(os.path.join(OUT, f"{NAME}_result.json"), "w"), indent=1)
log(f"done: {NAME} | best val macro-F1 {best:.4f} | train {train_min:.1f} min | peak {peak_gb:.1f} GB | "
    f"trainable {n_train/1e6:.3f}M of {n_total/1e6:.2f}M")