"""sum_train.py — Phase-6 abstractive summarization harness."""
import importlib, json, math, os, random, re, sys, time, zipfile

import numpy as np
import torch
import torch.nn.functional as F

import sum_data as sd

CFG = json.load(open(sys.argv[1]))
D, TR, G, MODEL_CFG = CFG["data"], CFG["train"], CFG.get("gen", {}), CFG["model"]
KD = CFG.get("kd") or {}
OUT, NAME = CFG["out_dir"], CFG["run_name"]
os.makedirs(OUT, exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = int(TR.get("seed", 0))
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ tokenizer + data
from tokenizers import ByteLevelBPETokenizer
TOK = ByteLevelBPETokenizer(vocab=os.path.join(D["tokenizer_dir"], "vocab.json"),
                            merges=os.path.join(D["tokenizer_dir"], "merges.txt"))
V = TOK.get_vocab_size()
PAD = TOK.token_to_id("<pad>") or 0
EOS = TOK.token_to_id(D.get("end_token", "</s>"))
START = TOK.token_to_id(D.get("start_token", "</s>"))
assert EOS is not None and START is not None, "start/end token not in the vocabulary"
SEP_IDS = np.asarray(TOK.encode(D["sep_text"]).ids, np.uint16) if D.get("sep_text") else np.zeros(0, np.uint16)
MAXS, MAXT = int(D["max_src_tokens"]), int(D["max_tgt_tokens"])
SEL, HFRAC = D.get("src_select", "first"), float(D.get("head_frac", 0.5))
BAN = [i for i in (TOK.token_to_id(t) for t in ("<s>", "<unk>", "<pad>", "<mask>")) if i is not None and i != EOS]


def enc_batch(texts):
    return [e.ids for e in TOK.encode_batch(texts)]


ALL = sd.load_cases(D["train"])
TEST = sd.load_cases(D["test"]) if D.get("test") else {}
_nosum = [k for k, v in ALL.items() if not (v["summary"] or "").strip()]
for _k in _nosum:
    del ALL[_k]
assert ALL, "no training documents with a summary"
TR_KEYS, VA_KEYS = sd.split_train_val(list(ALL), int(D.get("val_cases", 0)), int(D.get("val_seed", 0)))
cache = D.get("cache_dir", "sum_cache")
DOC = sd.tokenize_texts({k: v["text"] for k, v in ALL.items()}, enc_batch, cache, "train-doc")
SUMT = sd.tokenize_texts({k: v["summary"] for k, v in ALL.items()}, enc_batch, cache, "train-sum")
if TEST:
    DOC.update({f"test::{k}": v for k, v in sd.tokenize_texts({k: v["text"] for k, v in TEST.items()}, enc_batch, cache, "test-doc").items()})

TARGET = {k: SUMT[k] for k in TR_KEYS}
if KD.get("teacher_summaries"):
    tsum = json.load(open(KD["teacher_summaries"], encoding="utf-8"))["summaries"]
    tsum = {k: v for k, v in tsum.items() if k in ALL and k not in VA_KEYS}
    assert tsum, "teacher summaries cover none of the training documents"
    tt = {k: np.asarray(ids, np.uint16) for k, ids in zip(tsum, enc_batch(list(tsum.values())))}
    if KD.get("mode", "replace") == "replace":
        TR_KEYS = [k for k in TR_KEYS if k in tt]
        TARGET = {k: tt[k] for k in TR_KEYS}
    else:
        for k, v in tt.items():
            TARGET[f"kd::{k}"] = v
        TR_KEYS = TR_KEYS + [f"kd::{k}" for k in tt]
    log(f"KD: {len(tt)} teacher summaries ({KD.get('mode', 'replace')}) -> {len(TR_KEYS)} training pairs")
if TR.get("train_subset"):
    TR_KEYS = random.Random(SEED).sample(TR_KEYS, min(len(TR_KEYS), int(TR["train_subset"])))


def doc_key(k):
    return k[4:] if k.startswith("kd::") else k


def src_of(k):
    body = sd.select_src(DOC[doc_key(k)], MAXS, SEL, HFRAC)
    if not len(body):
        body = np.asarray([START], np.uint16)
    return np.concatenate([body, SEP_IDS])


def tgt_of(ids):
    return np.concatenate([ids[:MAXT], np.asarray([EOS], np.uint16)])


ls = lambda d: sd.length_stats(d)
log(f"dropped {len(_nosum)} training documents with an empty summary: {_nosum[:5]}")
log(f"train pairs {len(TR_KEYS)} | val {len(VA_KEYS)} | test {len(TEST)} | doc tokens {ls({k: DOC[k] for k in ALL})} | "
    f"summary tokens {ls(SUMT)} | max_src {MAXS} ({SEL}) + sep {len(SEP_IDS)} | max_tgt {MAXT} | "
    f"summaries longer than max_tgt: {sum(1 for v in SUMT.values() if len(v) > MAXT) / len(SUMT):.1%}")


def batch_src(keys):
    srcs = [src_of(k) for k in keys]
    m = min(len(s) for s in srcs)
    rows = []
    for s in srcs:
        body = s[:len(s) - len(SEP_IDS)]
        keep = m - len(SEP_IDS)
        body = body[len(body) - keep:] if SEL == "last" else body[:keep]
        rows.append(np.concatenate([body, SEP_IDS]))
    return torch.tensor(np.stack(rows).astype(np.int64))


def batch_tgt(tgts):
    T = max(len(t) for t in tgts)
    tin = torch.full((len(tgts), T), PAD, dtype=torch.long)
    lab = torch.full((len(tgts), T), -100, dtype=torch.long)
    for i, t in enumerate(tgts):
        t = torch.tensor(t.astype(np.int64))
        tin[i, 0] = START
        tin[i, 1:len(t)] = t[:-1]
        lab[i, :len(t)] = t
    return tin, lab


def by_src_len(keys):
    return sorted(keys, key=lambda k: len(src_of(k)))


# ------------------------------------------------------------------ model
model_lib = importlib.import_module(CFG.get("model_module", "sum_model"))
model = model_lib.build_sum_model(dict(MODEL_CFG)).to(DEV)
n_total = sum(p.numel() for p in model.parameters())
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
log(f"parameters: total {n_total/1e6:.2f}M | trainable {n_train/1e6:.3f}M ({100*n_train/max(1,n_total):.2f}%)")
assert n_train > 0, "no trainable parameters"
HAS_CACHE = hasattr(model, "prefill") and hasattr(model, "decode_step")
log("generation: " + ("cached (prefill/decode_step)" if HAS_CACHE else "no cache: logits() recomputed each step (slower)"))
if hasattr(model, "max_total_len"):
    assert model.max_total_len(MAXS + len(SEP_IDS), MAXT + 1) is not False, "max_src_tokens + max_tgt_tokens do not fit the model"

if CFG.get("load_trained"):
    sd_ = torch.load(CFG["load_trained"], map_location="cpu", weights_only=False)["trained"]
    cur = dict(model.named_parameters())
    miss = [k for k in sd_ if k not in cur]
    assert not miss, f"load_trained: names not in this model, e.g. {miss[:3]}"
    with torch.no_grad():
        for k, v in sd_.items():
            cur[k].copy_(v.to(cur[k].device, cur[k].dtype))
    log(f"loaded {len(sd_)} trained tensors from {CFG['load_trained']}")


def step_loss(keys, tgt_map):
    src = batch_src(keys).to(DEV)
    tin, lab = batch_tgt([tgt_of(tgt_map[k]) for k in keys])
    tin, lab = tin.to(DEV), lab.to(DEV)
    with torch.autocast(DEV.type, dtype=torch.float16, enabled=DEV.type == "cuda"):
        lg = model.logits(src, tin)
    assert tuple(lg.shape) == (len(keys), tin.shape[1], V), f"logits {tuple(lg.shape)}, expected {(len(keys), tin.shape[1], V)}"
    loss = F.cross_entropy(lg.float().reshape(-1, V), lab.reshape(-1), ignore_index=-100,
                           label_smoothing=float(TR.get("label_smoothing", 0.0)))
    return loss, int((lab != -100).sum())


@torch.no_grad()
def val_loss(keys):
    model.eval()
    tot, n = 0.0, 0
    ks = by_src_len(keys)
    bs = int(TR.get("eval_batch_size", TR["batch_size"]))
    for i in range(0, len(ks), bs):
        l, c = step_loss(ks[i:i + bs], SUMT)
        tot += l.item() * c; n += c
    model.train()
    return tot / max(1, n)


# ------------------------------------------------------------------ generation
class NgramBlocker:
    def __init__(self, n):
        self.n, self.seen, self.hist = n, {}, []

    def banned(self):
        if not self.n or len(self.hist) < self.n - 1:
            return ()
        return self.seen.get(tuple(self.hist[-(self.n - 1):]), ())

    def push(self, t):
        if self.n and len(self.hist) >= self.n - 1:
            self.seen.setdefault(tuple(self.hist[-(self.n - 1):]), set()).add(t)
        self.hist.append(t)


def pick_next(logits, gen, blockers, step):
    logits = logits.float()
    if BAN:
        logits[:, BAN] = -float("inf")
    if step < int(G.get("min_new_tokens", 0)):
        logits[:, EOS] = -float("inf")
    rp = float(G.get("rep_penalty", 1.0) or 1.0)
    if rp != 1.0 and gen.numel():
        sc = torch.gather(logits, 1, gen)
        logits.scatter_(1, gen, torch.where(sc < 0, sc * rp, sc / rp))
    for b, bl in enumerate(blockers):
        bad = bl.banned()
        if bad:
            logits[b, list(bad)] = -float("inf")
    T = float(G.get("temperature", 0.0) or 0.0)
    if not T:
        return logits.argmax(-1)
    logits = logits / T
    k = int(G.get("top_k", 0) or 0)
    if k:
        kth = torch.topk(logits, k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, -float("inf"))
    p = float(G.get("top_p", 1.0) or 1.0)
    probs = F.softmax(logits, -1)
    if p < 1.0:
        sp, si = probs.sort(-1, descending=True)
        sp[(sp.cumsum(-1) - sp) > p] = 0.0
        probs = torch.zeros_like(probs).scatter_(1, si, sp)
    return torch.multinomial(probs / probs.sum(-1, keepdim=True), 1).squeeze(1)


def clean(text):
    return re.sub(r"\s+", " ", text).strip()


@torch.no_grad()
def generate(keys, src_fn):
    model.eval()
    out = {}
    ks = sorted(keys, key=lambda k: len(src_fn(k)))
    bs = int(G.get("batch_size", 8))
    max_new = int(G.get("max_new_tokens", MAXT))
    for i in range(0, len(ks), bs):
        b = ks[i:i + bs]
        srcs = [src_fn(k) for k in b]
        m = min(len(s) for s in srcs)
        rows = []
        for s in srcs:
            body, keep = s[:len(s) - len(SEP_IDS)], m - len(SEP_IDS)
            body = body[len(body) - keep:] if SEL == "last" else body[:keep]
            rows.append(np.concatenate([body, SEP_IDS]))
        src = torch.tensor(np.stack(rows).astype(np.int64), device=DEV)
        B = len(b)
        tin = torch.full((B, 1), START, dtype=torch.long, device=DEV)
        gen = torch.empty((B, 0), dtype=torch.long, device=DEV)
        done = torch.zeros(B, dtype=torch.bool, device=DEV)
        blockers = [NgramBlocker(int(G.get("no_repeat_ngram", 0) or 0)) for _ in range(B)]
        use_cache = HAS_CACHE
        with torch.autocast(DEV.type, dtype=torch.float16, enabled=DEV.type == "cuda"):
            if use_cache:
                try:
                    nl, cachev = model.prefill(src, tin)
                except NotImplementedError:
                    use_cache = False
            if not use_cache:
                nl = model.logits(src, tin)[:, -1]
        for step in range(max_new):
            nxt = pick_next(nl, gen, blockers, step)
            nxt = torch.where(done, torch.full_like(nxt, EOS), nxt)
            gen = torch.cat([gen, nxt[:, None]], 1)
            for bb, t in enumerate(nxt.tolist()):
                blockers[bb].push(t)
            done |= nxt == EOS
            if bool(done.all()) or step == max_new - 1:
                break
            with torch.autocast(DEV.type, dtype=torch.float16, enabled=DEV.type == "cuda"):
                if use_cache:
                    nl, cachev = model.decode_step(nxt[:, None], cachev)
                else:
                    nl = model.logits(src, torch.cat([tin, gen], 1))[:, -1]
        for bb, k in enumerate(b):
            ids = gen[bb].tolist()
            ids = ids[:ids.index(EOS)] if EOS in ids else ids
            out[k] = clean(TOK.decode(ids))
    model.train()
    return out


def gen_scores(keys, tag):
    t1 = time.time()
    pred = generate(keys, src_of)
    refs = [ALL[k]["summary"] for k in keys]
    sc = sd.rouge([pred[k] for k in keys], refs)
    sc.update(words_pred=float(np.mean([len(pred[k].split()) for k in keys])) if keys else 0.0,
              words_ref=float(np.mean([len(r.split()) for r in refs])) if keys else 0.0,
              gen_seconds=time.time() - t1, n=len(keys))
    return sc, pred


# ------------------------------------------------------------------ training
bs, acc = int(TR["batch_size"]), int(TR.get("grad_accum", 1))
epochs = int(TR["epochs"])
per_epoch = min(len(TR_KEYS), int(TR.get("epoch_cases") or len(TR_KEYS)))
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
SELECT = TR.get("select_metric", "val_loss")
VGEN = random.Random(SEED + 3).sample(VA_KEYS, min(len(VA_KEYS), int(TR.get("val_gen_cases", 0) or 0)))
assert SELECT == "val_loss" or VGEN, "select_metric on ROUGE needs val_gen_cases > 0"
log(f"cases/epoch {per_epoch} | steps/epoch {steps_per_epoch} | epochs {epochs} | select by {SELECT} | "
    f"per-epoch generation on {len(VGEN)} val docs")


def lr_mult(step):
    if step < warm:
        return (step + 1) / max(1, warm)
    t = (step - warm) / max(1, total_steps - warm)
    sch = TR.get("schedule", "linear")
    return 1.0 if sch == "constant" else max(0.0, 1 - t) if sch == "linear" else 0.5 * (1 + math.cos(math.pi * t))


best, best_state, history = None, None, []
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
        rest = TR_KEYS[:]; random.shuffle(rest); order += rest
    ep_keys, order = order[:per_epoch], order[per_epoch:]

    ep_keys = by_src_len(ep_keys)
    batches = [ep_keys[i:i + bs] for i in range(0, len(ep_keys), bs)]
    random.shuffle(batches)
    run_loss, nb = 0.0, 0
    for b in batches:
        loss, _ = step_loss(b, TARGET)
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
    if VA_KEYS:
        rec["val_loss"] = val_loss(VA_KEYS)
    if VGEN:
        sc, _ = gen_scores(VGEN, "epoch")
        rec.update(val_rouge1=sc["rouge1"], val_rougeL=sc["rougeL"], val_words=sc["words_pred"])
    history.append(rec)
    log(" | ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in rec.items()))
    score = rec.get(SELECT, ep if SELECT != "val_loss" else -ep)
    better = best is None or (score <= best if SELECT == "val_loss" else score >= best)
    if better:
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

result = dict(run=NAME, max_src_tokens=MAXS, src_select=SEL, max_tgt_tokens=MAXT, sep_text=D.get("sep_text"),
              params_total=n_total, params_trainable=n_train, train_minutes=train_min, peak_mem_gb=peak_gb,
              gen=G, history=history, kd=KD or None)

# ------------------------------------------------------------------ final validation generation (+ optional BERTScore)
FV = random.Random(SEED + 5).sample(VA_KEYS, min(len(VA_KEYS), int(G.get("final_val_cases", 100) or 0)))
if FV:
    sc, vpred = gen_scores(FV, "final")
    if G.get("bertscore"):
        try:
            from bert_score import score as bscore
            P_, R_, F_ = bscore([vpred[k] for k in FV], [ALL[k]["summary"] for k in FV], lang="en",
                                model_type=G.get("bertscore_model"), batch_size=16, verbose=False)
            sc.update(bertscore_p=float(P_.mean()), bertscore_r=float(R_.mean()), bertscore_f1=float(F_.mean()))
        except Exception as e:
            log(f"BERTScore skipped: {type(e).__name__}: {e}")
    result["val"] = sc
    log("VAL generation: " + " | ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in sc.items()))
    json.dump([dict(id=ALL[k]["id"], summary=vpred[k], reference=ALL[k]["summary"]) for k in FV],
              open(os.path.join(OUT, f"{NAME}_val_predictions.json"), "w", encoding="utf-8"), ensure_ascii=False)
    ex = FV[0]
    log(f"  example [{ALL[ex]['id']}]: {vpred[ex][:400]}")

# ------------------------------------------------------------------ teacher summaries for KD (training documents)
if CFG.get("dump_teacher_summaries"):
    pool = [k for k in sorted(ALL) if k not in set(VA_KEYS)]
    tk = random.Random(SEED + 11).sample(pool, min(len(pool), int(CFG["dump_teacher_summaries"])))
    t1 = time.time()
    tp = generate(tk, src_of)
    json.dump(dict(run=NAME, summaries=tp), open(os.path.join(OUT, f"{NAME}_teacher_summaries.json"), "w", encoding="utf-8"),
              ensure_ascii=False)
    log(f"teacher summaries written for {len(tp)} training documents ({(time.time() - t1) / 60:.1f} min)")

# ------------------------------------------------------------------ test
if TEST:
    t1 = time.time()
    tpred = generate([f"test::{k}" for k in TEST], src_of)
    pred = [dict(id=TEST[k]["id"], summary=tpred[f"test::{k}"]) for k in TEST]
    n, empty = sd.validate_prediction(pred, TEST)
    pj = os.path.join(OUT, f"{NAME}_prediction.json")
    json.dump(pred, open(pj, "w", encoding="utf-8"), ensure_ascii=False)
    zp = os.path.join(OUT, f"{NAME}_submission.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(pj, arcname="prediction.json")
    w = [len(r["summary"].split()) for r in pred]
    result.update(test_docs=n, test_empty=empty, test_words_mean=float(np.mean(w)), test_gen_seconds=time.time() - t1)
    log(f"test: {n} docs | words per summary mean {np.mean(w):.0f} (min {min(w)}, max {max(w)}) | empty {empty} | "
        f"generation {(time.time() - t1) / 60:.1f} min")
    log(f"wrote {zp}")
json.dump(result, open(os.path.join(OUT, f"{NAME}_result.json"), "w"), indent=1, default=str)
log(f"done: {NAME} | best {SELECT} {best if best is not None else float('nan'):.4f} | train {train_min:.1f} min | "
    f"peak {peak_gb:.1f} GB | trainable {n_train/1e6:.3f}M of {n_total/1e6:.2f}M")