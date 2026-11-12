import contextlib, glob, json, math, os, sys, time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

CFG = json.load(open(sys.argv[1]))
RUN, MODEL_CFG, TR = CFG["run"], CFG["model"], CFG["train"]

if RUN.get("smoke_test"):
    import dummy_model as model_lib
else:
    import model as model_lib


# ----------------------------------------------------------------------------- setup
def setup_dist():
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        dist.init_process_group("nccl")
        rank, world, local = dist.get_rank(), dist.get_world_size(), int(os.environ["LOCAL_RANK"])
    else:
        rank, world, local = 0, 1, 0
    torch.cuda.set_device(local)
    return rank, world, torch.device("cuda", local)


RANK, WORLD, DEV = setup_dist()
IS_MAIN = RANK == 0
DDP_ON = WORLD > 1


def log(*a):
    if IS_MAIN:
        print(*a, flush=True)


def bcast_float(x):
    t = torch.tensor([float(x)], device=DEV)
    if DDP_ON:
        dist.broadcast(t, 0)
    return t.item()


def allreduce_sum(*vals):
    t = torch.tensor([float(v) for v in vals], device=DEV, dtype=torch.float64)
    if DDP_ON:
        dist.all_reduce(t)
    return t.tolist()


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# ----------------------------------------------------------------------------- time budget
SESSION_START = RUN["session_start"]
SMOKE = bool(RUN.get("smoke_test"))
if SMOKE:
    TRAIN_DEADLINE = time.time() + RUN.get("smoke_train_minutes", 3) * 60
else:
    TRAIN_DEADLINE = (SESSION_START + RUN["session_hours"] * 3600
                      - RUN["reserve_minutes_after_train"] * 60)
log(f"Train deadline in {(TRAIN_DEADLINE - time.time())/3600:.2f} h")

# ----------------------------------------------------------------------------- data
META = json.load(open(os.path.join(RUN["token_dir"], "meta.json")))
CTX = int(TR["ctx_len"])


class WindowSampler:

    def __init__(self, split, mix, seed):
        files = sorted(glob.glob(os.path.join(RUN["token_dir"], f"{split}_*.bin")))
        assert files, f"no {split}_*.bin files in {RUN['token_dir']}"
        self.names, self.arrs = [], []
        for f in files:
            a = np.memmap(f, dtype=np.uint16, mode="r")
            if a.size > CTX + 2:
                self.names.append(os.path.basename(f)[len(split) + 1:-4])
                self.arrs.append(a)
        if mix == "proportional":
            w = np.array([a.size for a in self.arrs], dtype=np.float64)
        else:
            w = np.array([float(mix.get(n, 0.0)) for n in self.names])
        assert w.sum() > 0, f"mix gives zero weight to all of {self.names}"
        self.w = w / w.sum()
        self.rng = np.random.default_rng(seed)

    def window(self, i):
        a = self.arrs[i]
        s = int(self.rng.integers(0, a.size - CTX - 1))
        return np.asarray(a[s:s + CTX + 1], dtype=np.int64)

    def batch(self, B):
        idx = self.rng.choice(len(self.arrs), size=B, p=self.w)
        return torch.from_numpy(np.stack([self.window(i) for i in idx]))


def prefix_lengths(B, mode, fmin, fmax, rng):
    if mode == "none":
        return np.zeros(B, dtype=np.int64)
    lo = int(round(fmin * CTX))
    if mode == "fixed":
        return np.full(B, lo, dtype=np.int64)
    if mode == "random":
        hi = int(round(fmax * CTX))
        return rng.integers(lo, hi + 1, size=B).astype(np.int64)
    if mode == "batch_random":
        hi = int(round(fmax * CTX))
        return np.full(B, int(rng.integers(lo, hi + 1)), dtype=np.int64)
    raise ValueError(f"prefix_mode must be none|fixed|random|batch_random, got {mode!r}")


def make_xy(chunk, plen):
    x, y = chunk[:, :-1].clone(), chunk[:, 1:].clone()
    pos = torch.arange(CTX)[None, :]
    y[pos < (torch.from_numpy(plen)[:, None] - 1)] = -100
    return x, y, torch.from_numpy(plen)


train_sampler = WindowSampler("train", TR["mix"], seed=1234 + RANK * 7919)
train_prefix_rng = np.random.default_rng(99 + RANK)
log(f"Sources: {dict(zip(train_sampler.names, np.round(train_sampler.w, 3)))}")


VAL_PER_SOURCE = 8 if SMOKE else int(RUN.get("val_windows_per_source", 48))
val_sampler = WindowSampler("val", "proportional", seed=4321)
VAL = []
vp_rng = np.random.default_rng(7)
for si, name in enumerate(val_sampler.names):
    for _ in range(VAL_PER_SOURCE):
        c = torch.from_numpy(val_sampler.window(si))[None]
        pl = prefix_lengths(1, TR["val_prefix_mode"], TR["val_prefix_frac"], TR["val_prefix_frac"], vp_rng)
        x, y, p = make_xy(c, pl)
        VAL.append((name, x, y, p))
VAL_SHARD = VAL[RANK::WORLD]

# ----------------------------------------------------------------------------- model
MODEL_CFG.setdefault("vocab_size", META["vocab_size"])
MODEL_CFG.setdefault("max_seq_len", CTX)
raw_model = model_lib.build_model(MODEL_CFG).to(DEV)
n_params = sum(p.numel() for p in raw_model.parameters())
log(f"Parameters: {n_params/1e6:.2f}M")
if not SMOKE:
    assert 50e6 <= n_params <= 300e6, "Phase-1 rule: model must be 50M-300M parameters"
if TR.get("grad_ckpt") and hasattr(raw_model, "set_gradient_checkpointing"):
    raw_model.set_gradient_checkpointing(True)

decay, no_decay = [], []
for n, p in raw_model.named_parameters():
    if p.requires_grad:
        (decay if (p.dim() >= 2 or not TR["wd_exclude_1d"]) else no_decay).append(p)
optimizer = torch.optim.AdamW(
    [{"params": decay, "weight_decay": TR["weight_decay"]},
     {"params": no_decay, "weight_decay": 0.0}],
    lr=TR["lr_max"], betas=(TR["beta1"], TR["beta2"]), eps=1e-8, fused=True)
scaler = torch.amp.GradScaler("cuda", enabled=TR.get("amp_dtype", "fp16") == "fp16")
AMP_DTYPE = torch.float16 if TR.get("amp_dtype", "fp16") == "fp16" else torch.bfloat16

model = raw_model
if TR.get("compile"):
    model = torch.compile(model)
if DDP_ON:
    model = DDP(model, device_ids=[DEV.index], find_unused_parameters=bool(TR.get("find_unused", False)))


def unwrap():
    return raw_model


# ----------------------------------------------------------------------------- checkpoints
OUT = RUN["out_dir"]
os.makedirs(OUT, exist_ok=True)
NAME = RUN["run_name"]


def ckpt_files(d):
    return sorted(glob.glob(os.path.join(d, f"{NAME}_step*.pt")))


def save_ckpt(state, path):
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def save_training_ckpt(step, tokens, total_steps, best_val):
    if not IS_MAIN:
        return
    path = os.path.join(OUT, f"{NAME}_step{step:07d}.pt")
    save_ckpt(dict(model=unwrap().state_dict(), optimizer=optimizer.state_dict(),
                   scaler=scaler.state_dict(), step=step, tokens=tokens,
                   total_steps=total_steps, best_val=best_val, cfg=CFG), path)
    for old in ckpt_files(OUT)[:-2]:
        os.remove(old)
    log(f"  saved {os.path.basename(path)}")


def save_model_only(tag, step, tokens, val):
    if IS_MAIN:
        save_ckpt(dict(model=unwrap().state_dict(), step=step, tokens=tokens, val_loss=val,
                       model_cfg=MODEL_CFG, n_params=n_params),
                  os.path.join(OUT, f"{NAME}_{tag}.pt"))


step, tokens, total_steps, best_val = 0, 0, None, float("inf")
RESUMED = False
cands = []
for d in [OUT] + list(RUN.get("resume_dirs", [])):
    cands += ckpt_files(d)
if cands:
    latest = max(cands, key=lambda p: int(p.split("_step")[-1][:-3]))
    ck = torch.load(latest, map_location=DEV, weights_only=False)
    unwrap().load_state_dict(ck["model"])
    optimizer.load_state_dict(ck["optimizer"])
    scaler.load_state_dict(ck["scaler"])
    step, tokens, total_steps, best_val = ck["step"], ck["tokens"], ck["total_steps"], ck["best_val"]
    log(f"Resumed from {latest} (step {step}, {tokens/1e6:.1f}M tokens)")
    RESUMED = True
    del ck

    train_sampler.rng = np.random.default_rng(1234 + RANK * 7919 + 1_000_003 * step)
    train_prefix_rng = np.random.default_rng(99 + RANK + 1_000_003 * step)
    log("Data sampler re-seeded for resume")

# ----------------------------------------------------------------------------- schedule
def lr_at(s, total):
    lr_max, lr_min, warm = TR["lr_max"], TR["lr_min"], TR["warmup_steps"]
    if s < warm:
        return lr_max * (s + 1) / warm
    if total is None:
        return lr_max
    t = min(1.0, (s - warm) / max(1, total - warm))
    kind = TR["schedule"]
    if kind == "cosine":
        return lr_min + (lr_max - lr_min) * 0.5 * (1 + math.cos(math.pi * t))
    if kind == "linear":
        return lr_max + (lr_min - lr_max) * t
    if kind == "constant":
        return lr_max
    if kind == "wsd":
        d = TR["decay_frac"]
        return lr_max if t < 1 - d else lr_max + (lr_min - lr_max) * (t - (1 - d)) / d
    raise ValueError(f"unknown schedule {kind!r}")


def align_targets(logits, y):
    L, T = logits.size(1), y.size(1)
    if L == T:
        return y
    if L > T or bool((y[:, :T - L] != -100).any()):
        raise ValueError(f"model returned {L} logit positions for {T} targets, but some dropped "
                         f"targets are scored: return at least T - prefix_len + 1 positions")
    return y[:, T - L:]


# ----------------------------------------------------------------------------- eval
@torch.no_grad()
def evaluate():
    model.eval()
    sums = {}
    for name, x, y, p in VAL_SHARD:
        with torch.autocast("cuda", dtype=AMP_DTYPE):
            logits = unwrap()(x.to(DEV), p.to(DEV))
        y = align_targets(logits, y.to(DEV))
        l = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1),
                            ignore_index=-100, reduction="sum").item()
        n = (y != -100).sum().item()
        s = sums.setdefault(name, [0.0, 0])
        s[0] += l; s[1] += n
    names = val_sampler.names
    flat = []
    for n in names:
        flat += sums.get(n, [0.0, 0])
    flat = allreduce_sum(*flat)
    per = {n: flat[2 * i] / max(1, flat[2 * i + 1]) for i, n in enumerate(names)}
    tot = sum(flat[0::2]) / max(1, sum(flat[1::2]))
    model.train()
    return tot, per


# ----------------------------------------------------------------------------- train loop
MB, ACC = int(TR["micro_batch"]), int(TR["grad_accum"])
TOK_PER_STEP = MB * CTX * ACC * WORLD
log(f"Tokens/step: {TOK_PER_STEP:,}  (micro {MB} x ctx {CTX} x accum {ACC} x gpus {WORLD})")

csv_path = os.path.join(OUT, f"{NAME}_log.csv")
if IS_MAIN and not os.path.exists(csv_path):
    open(csv_path, "w").write("step,tokens,train_loss,lr,tok_per_s,elapsed_h,val_loss,val_per_source,"
                               "grad_norm_max,amp_scale,skipped_steps\n")


def csv(row):
    if IS_MAIN:
        open(csv_path, "a").write(row + "\n")


CALIB_AFTER = 10
CALIB_STEPS = 30
calib_t0 = None
run_t0 = time.time()
last_ckpt = time.time()
last_val = time.time()
win_t0, win_tok = time.time(), 0
model.train()
optimizer.zero_grad(set_to_none=True)
steps_this_run = 0


gn_max_win = torch.zeros((), device=DEV)
skipped_total, skipped_win = 0, 0

while True:

    loss_acc = torch.zeros((), device=DEV)
    for micro in range(ACC):
        chunk = train_sampler.batch(MB)
        pl = prefix_lengths(MB, TR["prefix_mode"], TR.get("prefix_frac_min", 0.0),
                            TR.get("prefix_frac_max", 0.0), train_prefix_rng)
        x, y, p = make_xy(chunk, pl)
        x, y, p = x.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True), p.to(DEV, non_blocking=True)
        sync_ctx = model.no_sync() if (DDP_ON and micro < ACC - 1) else contextlib.nullcontext()
        with sync_ctx:
            with torch.autocast("cuda", dtype=AMP_DTYPE):
                logits = model(x, p)
                yy = align_targets(logits, y)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), yy.reshape(-1),
                                       ignore_index=-100)
            scaler.scale(loss / ACC).backward()
        loss_acc += loss.detach() / ACC

    lr = lr_at(step, total_steps)
    for g in optimizer.param_groups:
        g["lr"] = lr
    scaler.unscale_(optimizer)
    gn = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), TR["grad_clip"])
    gn_max_win = torch.maximum(gn_max_win, torch.nan_to_num(gn.detach().float(), nan=float("inf")))
    scale_before = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    if scaler.is_enabled() and scaler.get_scale() < scale_before:
        skipped_total += 1; skipped_win += 1
    optimizer.zero_grad(set_to_none=True)
    step += 1
    steps_this_run += 1
    tokens += TOK_PER_STEP
    win_tok += TOK_PER_STEP


    if steps_this_run == CALIB_AFTER:
        torch.cuda.synchronize(); calib_t0 = time.time()
    if steps_this_run == CALIB_AFTER + CALIB_STEPS and (total_steps is None or RESUMED):
        torch.cuda.synchronize()
        sec_per_step = (time.time() - calib_t0) / CALIB_STEPS
        overhead = 1.0 - RUN.get("eval_ckpt_overhead_frac", 0.04)
        est = step + int((TRAIN_DEADLINE - time.time()) / sec_per_step * overhead)
        if TR.get("max_tokens"):
            est = min(est, int(TR["max_tokens"]) // TOK_PER_STEP)
        est = int(bcast_float(est))
        if RESUMED and total_steps is not None:
            if est < total_steps:
                log(f"Resume: only time for {est} of {total_steps} scheduled steps -> schedule shortened so LR still decays")
                total_steps = est
        else:
            total_steps = est
        log(f"Measured {sec_per_step:.2f}s/step ({TOK_PER_STEP/sec_per_step/1e3:.1f}K tok/s). "
            f"Schedule sized to {total_steps} steps = {total_steps*TOK_PER_STEP/1e6:.0f}M tokens.")

    now = time.time()
    if step % TR.get("log_every_steps", 20) == 0:
        l = allreduce_sum(loss_acc.item())[0] / WORLD
        tps = win_tok / max(1e-9, now - win_t0)
        gmax = gn_max_win.item()
        sc = scaler.get_scale() if scaler.is_enabled() else 1.0
        log(f"step {step:6d} | loss {l:.4f} | lr {lr:.2e} | {tokens/1e6:8.1f}M tok | "
            f"{tps/1e3:5.1f}K tok/s | gnorm {gmax:6.2f} | scale {sc:.0f} | skip {skipped_win}/{skipped_total} | "
            f"left {(TRAIN_DEADLINE-now)/3600:.2f}h")
        csv(f"{step},{tokens},{l:.5f},{lr:.3e},{tps:.0f},{(now-run_t0)/3600:.3f},,,{gmax:.4f},{sc:.0f},{skipped_total}")
        win_t0, win_tok = now, 0
        gn_max_win.zero_(); skipped_win = 0


    flags = torch.tensor([
        float((now >= TRAIN_DEADLINE) or (total_steps is not None and step >= total_steps)),
        float(now - last_val >= TR.get("val_every_min", 30) * 60),
        float(now - last_ckpt >= TR.get("ckpt_every_min", 20) * 60)], device=DEV)
    if DDP_ON:
        dist.broadcast(flags, 0)
    stop, do_val, do_ckpt = (bool(f) for f in flags.tolist())

    if stop or do_val:
        v, per = evaluate()
        log(f"  [val] loss {v:.4f} ppl {math.exp(min(v, 20)):.2f} | " +
            " ".join(f"{k}={x:.3f}" for k, x in per.items()))
        csv(f"{step},{tokens},,{lr:.3e},,{(now-run_t0)/3600:.3f},{v:.5f},"
            + ";".join(f"{k}={x:.4f}" for k, x in per.items()) + ",,,")
        if v < best_val:
            best_val = v
            save_model_only("best", step, tokens, v)
            log(f"  new best -> {NAME}_best.pt")
        last_val = time.time()

    if stop or do_ckpt:
        save_training_ckpt(step, tokens, total_steps, best_val)
        last_ckpt = time.time()

    if stop:
        break

save_model_only("final", step, tokens, best_val)
log(f"Training finished: {step} steps, {tokens/1e6:.1f}M tokens, best val {best_val:.4f}")
if DDP_ON:
    dist.barrier()
    dist.destroy_process_group()