"""pcr_train.py — Phase-7 prior case retrieval harness."""
import importlib, json, math, os, random, sys, time, zipfile

import numpy as np
import torch
import torch.nn.functional as F

import pcr_data as pdat

CFG = json.load(open(sys.argv[1]))
D, TR, MODEL_CFG = CFG["data"], CFG["train"], CFG["model"]
KD = CFG.get("kd") or {}
OUT, NAME = CFG["out_dir"], CFG["run_name"]
os.makedirs(OUT, exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = int(TR.get("seed", 0))
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
K = 7


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ data
from tokenizers import ByteLevelBPETokenizer
TOK = ByteLevelBPETokenizer(vocab=os.path.join(D["tokenizer_dir"], "vocab.json"),
                            merges=os.path.join(D["tokenizer_dir"], "merges.txt"))


def enc_batch(texts):
    return [e.ids for e in TOK.encode_batch(texts)]


cache = D.get("cache_dir", "pcr_cache")
SPL = {}
for s in ("train", "dev", "test"):
    if not D.get(f"{s}_q"):
        continue
    q, p = pdat.load_queries(D[f"{s}_q"]), pdat.load_pool(D[f"{s}_pool"])
    qt = pdat.tokenize_texts({k: v["text"] for k, v in q.items()}, enc_batch, cache, f"{s}-q")
    pt = pdat.tokenize_texts({k: v["text"] for k, v in p.items()}, enc_batch, cache, f"{s}-pool")
    gold = {k: [c for c in v["relevant"] if c in p and c != k] for k, v in q.items() if v["relevant"] is not None}
    SPL[s] = dict(q=q, pool=p, qt=qt, pt=pt, gold={k: g for k, g in gold.items() if g}, pkeys=list(p))
    miss = sum(len(v["relevant"]) for v in q.values() if v["relevant"]) - sum(len(g) for g in gold.values())
    log(f"{s}: {len(q)} queries | pool {len(p)} | queries with >=1 relevant in pool {len(SPL[s]['gold'])} | "
        f"relevant per query mean {np.mean([len(g) for g in SPL[s]['gold'].values()]) if SPL[s]['gold'] else 0:.2f} | "
        f"relevant ids missing from the pool {miss} | query tokens {pdat.length_stats(qt)} | pool tokens {pdat.length_stats(pt)}")

L_CH, N_CH = int(D["chunk_len"]), int(D["n_chunks"])
MODE, STRIDE = D.get("chunk_mode", "first"), D.get("stride")
TRAIN_CH = int(D.get("train_chunks") or N_CH)
PICK = D.get("train_chunk_pick", "first")
AGG, SIM = D.get("agg", "mean"), D.get("sim", "cos")
assert AGG in ("mean", "max", "first") and SIM in ("cos", "dot"), "agg: mean|max|first, sim: cos|dot"


def chunks_of(ids, train=False):
    ch = pdat.chunk_ids(ids, L_CH, N_CH, MODE, STRIDE)
    if train and len(ch) > TRAIN_CH:
        ch = random.sample(ch, TRAIN_CH) if PICK == "random" else ch[:TRAIN_CH]
    return ch


# ------------------------------------------------------------------ model
model_lib = importlib.import_module(CFG.get("model_module", "pcr_model"))
model = model_lib.build_pcr_model(dict(MODEL_CFG)).to(DEV)
n_total = sum(p.numel() for p in model.parameters())
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
log(f"parameters: total {n_total/1e6:.2f}M | trainable {n_train/1e6:.3f}M ({100*n_train/max(1,n_total):.2f}%)")
if CFG.get("load_trained"):
    sd_ = torch.load(CFG["load_trained"], map_location="cpu", weights_only=False)["trained"]
    cur = dict(model.named_parameters())
    miss = [k for k in sd_ if k not in cur]
    assert not miss, f"load_trained: names not in this model, e.g. {miss[:3]}"
    with torch.no_grad():
        for k, v in sd_.items():
            cur[k].copy_(v.to(cur[k].device, cur[k].dtype))
    log(f"loaded {len(sd_)} trained tensors from {CFG['load_trained']}")
assert n_train > 0 or int(TR.get("epochs", 0)) == 0, "no trainable parameters"
CH_BS = int(TR.get("chunk_batch", 64))


def embed_chunks(chunks):
    groups = {}
    for i, c in enumerate(chunks):
        groups.setdefault(len(c), []).append(i)
    out, idx = [], []
    for L, ii in groups.items():
        for j in range(0, len(ii), CH_BS):
            sub = ii[j:j + CH_BS]
            x = torch.tensor(np.stack([chunks[i] for i in sub]).astype(np.int64), device=DEV)
            with torch.autocast(DEV.type, dtype=torch.float16, enabled=DEV.type == "cuda"):
                e = model.embed(x)
            assert e.dim() == 2 and e.shape[0] == len(sub), f"embed must return [B, D], got {tuple(e.shape)}"
            out.append(e.float()); idx += sub
    E = torch.cat(out, 0)
    inv = torch.empty(len(idx), dtype=torch.long, device=E.device)
    inv[torch.tensor(idx, device=E.device)] = torch.arange(len(idx), device=E.device)
    return E[inv]


def doc_vectors(id_lists, train=False):
    chs, owner = [], []
    for d, ids in enumerate(id_lists):
        c = chunks_of(ids, train)
        chs += c; owner += [d] * len(c)
    E = embed_chunks(chs)
    owner = torch.tensor(owner, device=E.device)
    n = len(id_lists)
    if AGG == "mean":
        V = torch.zeros(n, E.shape[1], device=E.device).index_add_(0, owner, E)
        V = V / torch.bincount(owner, minlength=n).clamp(min=1)[:, None].float()
    elif AGG == "max":
        V = torch.full((n, E.shape[1]), -1e4, device=E.device).scatter_reduce(0, owner[:, None].expand_as(E), E, "amax")
    else:
        cnt = torch.bincount(owner, minlength=n)
        V = E[torch.cumsum(cnt, 0) - cnt]
    return V


def sim(Q, C):
    if SIM == "cos":
        Q, C = F.normalize(Q, dim=-1), F.normalize(C, dim=-1)
    return Q @ C.T


@torch.no_grad()
def encode_all(tok_map, keys, bs_docs=None):
    model.eval()
    bs_docs = bs_docs or max(1, CH_BS // max(1, N_CH))
    order = sorted(range(len(keys)), key=lambda i: len(tok_map[keys[i]]))
    vecs = torch.zeros(len(keys), 1)
    out = [None] * len(keys)
    for i in range(0, len(order), bs_docs):
        b = order[i:i + bs_docs]
        V = doc_vectors([tok_map[keys[j]] for j in b])
        for j, v in zip(b, V):
            out[j] = v.cpu()
    model.train()
    return torch.stack(out) if out else vecs


@torch.no_grad()
def rank_split(s, depth=100):
    S = SPL[s]
    qk, pk = list(S["q"]), S["pkeys"]
    Qv, Pv = encode_all(S["qt"], qk), encode_all(S["pt"], pk)
    sc = sim(Qv.to(DEV), Pv.to(DEV)).float().cpu().numpy()
    ranked = {q: pdat.rank_pool(sc[i], pk, exclude=[q], k=depth) for i, q in enumerate(qk)}
    return ranked, Qv, Pv


def dev_scores(ranked, gold):
    m = pdat.micro_at_k(ranked, gold, K)
    return dict(f1=m["f1"], p=m["precision"], r=m["recall"], r50=pdat.recall_at(ranked, gold, 50),
                r100=pdat.recall_at(ranked, gold, 100))


# ------------------------------------------------------------------ negatives
TRS = SPL.get("train")
N_NEG = int(TR.get("n_neg", 0))
NEG_SRC = TR.get("neg_source", "random")
HARD_POOL = int(TR.get("hard_pool", 50))
HARD = {}
T_EMB = None
if KD.get("teacher_emb") or NEG_SRC == "teacher":
    path = KD.get("teacher_emb") or TR.get("teacher_emb")
    assert path, "neg_source='teacher' needs kd.teacher_emb (or train.teacher_emb)"
    te = torch.load(path, map_location="cpu", weights_only=False)
    T_EMB = dict(q={k: i for i, k in enumerate(te["q_keys"])}, p={k: i for i, k in enumerate(te["p_keys"])},
                 Q=te["Q"].float(), P=te["P"].float(), sim=te["sim"])
    log(f"teacher vectors: {len(te['q_keys'])} queries, {len(te['p_keys'])} candidates (run {te['run']}, sim {te['sim']})")


def t_sim(Q, C):
    if T_EMB["sim"] == "cos":
        Q, C = F.normalize(Q, dim=-1), F.normalize(C, dim=-1)
    return Q @ C.T


if TRS and int(TR.get("epochs", 0)) > 0 and N_NEG and NEG_SRC in ("bm25", "teacher", "mix"):
    t1 = time.time()
    pk = TRS["pkeys"]
    use_teacher = NEG_SRC == "teacher" or (NEG_SRC == "mix" and T_EMB is not None and TR.get("mix_hard") == "teacher")
    if use_teacher:
        qk = [q for q in TRS["gold"] if q in T_EMB["q"]]
        Qt = T_EMB["Q"][[T_EMB["q"][q] for q in qk]]
        Pt = T_EMB["P"][[T_EMB["p"][c] for c in pk]]
        sc = t_sim(Qt, Pt).numpy()
        for i, q in enumerate(qk):
            HARD[q] = pdat.rank_pool(sc[i], pk, exclude=[q] + TRS["gold"][q], k=HARD_POOL)
    else:
        bm = pdat.BM25({c: TRS["pt"][c] for c in pk}, max_terms=TR.get("bm25_max_terms"))
        for q in TRS["gold"]:
            HARD[q] = pdat.rank_pool(bm.scores(TRS["qt"][q], TR.get("bm25_max_terms")), bm.keys,
                                     exclude=[q] + TRS["gold"][q], k=HARD_POOL)
    log(f"hard negatives ({'teacher' if use_teacher else 'bm25'}): top {HARD_POOL} non-relevant for {len(HARD)} queries "
        f"({time.time() - t1:.0f}s)")


def sample_negs(q, n):
    if not n:
        return []
    bad = set(TRS["gold"][q]) | {q}
    out = []
    if NEG_SRC in ("bm25", "teacher", "mix") and HARD.get(q):
        nh = n if NEG_SRC != "mix" else (n + 1) // 2
        out = random.sample(HARD[q], min(nh, len(HARD[q])))
    tries = 0
    while len(out) < n and tries < 50 * n:
        c = random.choice(TRS["pkeys"]); tries += 1
        if c not in bad and c not in out:
            out.append(c)
    while len(out) < n:
        out.append(random.choice([c for c in TRS["pkeys"] if c not in bad] or TRS["pkeys"]))
    return out


# ------------------------------------------------------------------ loss
TAU = float(TR.get("tau", 0.05))
LOSS = TR.get("loss", "infonce")
IN_BATCH = bool(TR.get("in_batch", True))
ALPHA, KD_TAU = float(KD.get("alpha", 0.0)), float(KD.get("kd_tau", 1.0))


def step_loss(qs):
    pos = [random.choice(TRS["gold"][q]) for q in qs]
    negs = [sample_negs(q, N_NEG) for q in qs]
    cols = pos + [c for ns in negs for c in ns]
    Qv = doc_vectors([TRS["qt"][q] for q in qs], train=True)
    Cv = doc_vectors([TRS["pt"][c] for c in cols], train=True)
    S = sim(Qv, Cv)
    B = len(qs)
    own = torch.zeros_like(S, dtype=torch.bool)
    for i in range(B):
        own[i, i] = True
        own[i, B + i * N_NEG: B + (i + 1) * N_NEG] = True
    allowed = own.clone() if not IN_BATCH else torch.ones_like(own)
    for i, q in enumerate(qs):
        g = set(TRS["gold"][q])
        for j, c in enumerate(cols):
            if j != i and c in g:
                allowed[i, j] = False
    neg_inf = -1e4
    if LOSS == "infonce":
        task = F.cross_entropy((S / TAU).masked_fill(~allowed, neg_inf), torch.arange(B, device=S.device))
    elif LOSS == "pairwise":
        m = float(TR.get("margin", 0.2))
        pos_s = S[torch.arange(B), torch.arange(B)][:, None]
        negm = allowed.clone(); negm[torch.arange(B), torch.arange(B)] = False
        task = (F.relu(m - pos_s + S) * negm).sum() / negm.sum().clamp(min=1)
    else:
        raise ValueError("loss must be infonce | pairwise")
    if ALPHA and T_EMB is not None:
        ok = all(q in T_EMB["q"] for q in qs) and all(c in T_EMB["p"] for c in cols)
        if ok:
            Qt = T_EMB["Q"][[T_EMB["q"][q] for q in qs]].to(S.device)
            Ct = T_EMB["P"][[T_EMB["p"][c] for c in cols]].to(S.device)
            St = t_sim(Qt, Ct)
            ls = F.log_softmax((S / KD_TAU).masked_fill(~allowed, neg_inf), -1)
            pt = F.softmax((St / KD_TAU).masked_fill(~allowed, neg_inf), -1)
            kd = (pt * (torch.log(pt.clamp(min=1e-12)) - ls)).masked_fill(~allowed, 0).sum(-1).mean()
            return (1 - ALPHA) * task + ALPHA * kd, float(task), float(kd)
    return task, float(task), 0.0


# ------------------------------------------------------------------ training
epochs = int(TR.get("epochs", 0))
TQ = list(TRS["gold"]) if TRS else []
if TR.get("train_subset"):
    TQ = random.Random(SEED).sample(TQ, min(len(TQ), int(TR["train_subset"])))
bs, acc = int(TR.get("batch_size", 4)), int(TR.get("grad_accum", 1))
per_epoch = min(len(TQ), int(TR.get("epoch_queries") or len(TQ))) if TQ else 0
steps_per_epoch = math.ceil(per_epoch / (bs * acc)) if per_epoch else 0
total_steps = max(1, steps_per_epoch * epochs)
warm = int(float(TR.get("warmup_frac", 0)) * total_steps)
base_lr = float(TR.get("lr", 1e-4))
groups = model.param_groups(base_lr) if hasattr(model, "param_groups") else \
    [{"params": [p for p in model.parameters() if p.requires_grad]}]
opt = torch.optim.AdamW(groups, lr=base_lr, weight_decay=float(TR.get("weight_decay", 0.0))) if n_train else None
if opt:
    for g in opt.param_groups:
        g.setdefault("initial_lr", g["lr"])
scaler = torch.amp.GradScaler("cuda", enabled=DEV.type == "cuda")
SELECT = TR.get("select_metric", "dev_f1")
EVAL_EVERY = int(TR.get("eval_every", 1))
log(f"chunks: {N_CH} x {L_CH} ({MODE}, stride {STRIDE or L_CH}) at inference, {TRAIN_CH} ({PICK}) in training | agg {AGG} | "
    f"sim {SIM} | loss {LOSS} tau {TAU} | n_neg {N_NEG} ({NEG_SRC}) | in_batch {IN_BATCH} | KD alpha {ALPHA}")
log(f"train queries/epoch {per_epoch} | steps/epoch {steps_per_epoch} | epochs {epochs} | select by {SELECT}")


def lr_mult(step):
    if step < warm:
        return (step + 1) / max(1, warm)
    t = (step - warm) / max(1, total_steps - warm)
    sch = TR.get("schedule", "linear")
    return 1.0 if sch == "constant" else max(0.0, 1 - t) if sch == "linear" else 0.5 * (1 + math.cos(math.pi * t))


def snap():
    return {k: v.detach().cpu().clone() for k, v in model.named_parameters() if v.requires_grad}


history, best, best_state = [], None, snap()
if "dev" in SPL and SPL["dev"]["gold"] and TR.get("eval_before", True):
    t1 = time.time()
    rk, _, _ = rank_split("dev")
    sc0 = dev_scores(rk, SPL["dev"]["gold"])
    history.append(dict(epoch=0, **{f"dev_{k}": v for k, v in sc0.items()}))
    best = sc0[SELECT[4:]]
    log("epoch 0 (no training) | " + " | ".join(f"dev_{k} {v:.4f}" for k, v in sc0.items()) + f" | eval {time.time()-t1:.0f}s")
step, t0 = 0, time.time()
max_min = float(TR.get("max_minutes", 1e9))
if DEV.type == "cuda":
    torch.cuda.reset_peak_memory_stats()
order = []
model.train()
stop = False
for ep in range(1, epochs + 1):
    if len(order) < per_epoch:
        rest = TQ[:]; random.shuffle(rest); order += rest
    ep_q, order = order[:per_epoch], order[per_epoch:]
    run, run_t, run_k, nb = 0.0, 0.0, 0.0, 0
    for i in range(0, len(ep_q), bs):
        loss, lt, lk = step_loss(ep_q[i:i + bs])
        scaler.scale(loss / acc).backward()
        nb += 1; run += loss.item(); run_t += lt; run_k += lk
        if nb % acc == 0:
            for g in opt.param_groups:
                g["lr"] = g["initial_lr"] * lr_mult(step)
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], float(TR.get("grad_clip", 1.0)))
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); step += 1
        if (time.time() - t0) / 60 > max_min:
            stop = True; break
    rec = dict(epoch=ep, train_loss=run / max(1, nb), task_loss=run_t / max(1, nb), minutes=(time.time() - t0) / 60)
    if ALPHA:
        rec["kd_loss"] = run_k / max(1, nb)
    if "dev" in SPL and SPL["dev"]["gold"] and (ep % EVAL_EVERY == 0 or ep == epochs or stop):
        rk, _, _ = rank_split("dev")
        rec.update({f"dev_{k}": v for k, v in dev_scores(rk, SPL["dev"]["gold"]).items()})
    history.append(rec)
    log(" | ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in rec.items()))
    if SELECT not in rec and not ("dev" in SPL and SPL["dev"]["gold"]):
        best_state = snap()
    if SELECT in rec and (best is None or rec[SELECT] >= best):
        best = rec[SELECT]; best_state = snap()
        log(f"  best so far (epoch {ep})")
    if stop:
        log(f"time limit {max_min} min reached"); break
train_min = (time.time() - t0) / 60
peak_gb = torch.cuda.max_memory_allocated() / 1e9 if DEV.type == "cuda" else 0.0
cur = dict(model.named_parameters())
with torch.no_grad():
    for k, v in best_state.items():
        cur[k].copy_(v.to(DEV))
torch.save(dict(trained=best_state, cfg=CFG), os.path.join(OUT, f"{NAME}_trained_params.pt"))
result = dict(run=NAME, data={k: v for k, v in D.items() if not k.endswith(("_q", "_pool"))}, train=TR, kd=KD or None,
              params_total=n_total, params_trainable=n_train, train_minutes=train_min, peak_mem_gb=peak_gb, history=history)

# ------------------------------------------------------------------ final dev evaluation (best weights)
if "dev" in SPL and SPL["dev"]["gold"]:
    t1 = time.time()
    rk, _, _ = rank_split("dev")
    sc = dev_scores(rk, SPL["dev"]["gold"])
    result["dev"] = sc; result["dev_eval_seconds"] = time.time() - t1
    log("DEV (best weights): " + " | ".join(f"{k} {v:.4f}" for k, v in sc.items()) + f" | {time.time()-t1:.0f}s")
    json.dump({q: dict(ranked=rk[q][:50], gold=SPL["dev"]["gold"][q]) for q in SPL["dev"]["gold"]},
              open(os.path.join(OUT, f"{NAME}_dev_rankings.json"), "w"))

# ------------------------------------------------------------------ teacher vectors for KD (train split)
if CFG.get("dump_teacher") and TRS:
    t1 = time.time()
    qk, pk = list(TRS["q"]), TRS["pkeys"]
    torch.save(dict(run=NAME, sim=SIM, q_keys=qk, p_keys=pk, Q=encode_all(TRS["qt"], qk).half(),
                    P=encode_all(TRS["pt"], pk).half()), os.path.join(OUT, f"{NAME}_teacher_emb.pt"))
    log(f"teacher vectors written for {len(qk)} train queries + {len(pk)} train candidates ({(time.time()-t1)/60:.1f} min)")

# ------------------------------------------------------------------ test
if "test" in SPL:
    t1 = time.time()
    rk, _, _ = rank_split("test", depth=K)
    pred = {q: rk[q][:K] for q in SPL["test"]["q"]}
    n = pdat.validate_prediction(pred, SPL["test"]["q"], SPL["test"]["pool"], K)
    pj = os.path.join(OUT, f"{NAME}_prediction.json")
    json.dump(pred, open(pj, "w"), indent=1)
    zp = os.path.join(OUT, f"{NAME}_submission.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(pj, arcname="prediction.json")
    result.update(test_queries=n, test_seconds=time.time() - t1)
    log(f"test: {n} queries x {K} ids | {(time.time()-t1)/60:.1f} min | wrote {zp}")
json.dump(result, open(os.path.join(OUT, f"{NAME}_result.json"), "w"), indent=1, default=str)
log(f"done: {NAME} | best {SELECT} {best if best is not None else float('nan'):.4f} | train {train_min:.1f} min | "
    f"peak {peak_gb:.1f} GB | trainable {n_train/1e6:.3f}M of {n_total/1e6:.2f}M")