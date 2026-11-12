"""test_arch4_model.py — correctness checks for YOUR architecture-4 model.py (test code only, no architecture)."""
import json, sys

import torch
import torch.nn.functional as F

import model as model_lib

cfg = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else {"model": {}, "train": {"ctx_len": 256}}
mcfg = dict(cfg["model"])
mcfg.setdefault("vocab_size", 32000)
CTX = int(cfg["train"]["ctx_len"])
mcfg.setdefault("max_seq_len", CTX)
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.manual_seed(0)

m = model_lib.build_model(mcfg).to(dev).float().eval()
V = mcfg["vocab_size"]
T = min(96, CTX)
P = (3 * T) // 4
B = 3
TOL = 1e-4
ok = True


def report(name, passed, detail=""):
    global ok
    ok &= bool(passed)
    print(f"[{'PASS' if passed else 'FAIL'}] {name} {detail}", flush=True)


def run(ids, p):
    with torch.no_grad():
        out = m(ids.to(dev), torch.full((ids.shape[0],), p, device=dev)).float()
    return out, ids.shape[1] - out.shape[1]


ids = torch.randint(5, V, (B, T))

# 1 — shape
out, off = run(ids, P)
okshape = tuple(out.shape) in [(B, T, V), (B, T - P + 1, V)]
report("1 output shape", okshape, f"got {tuple(out.shape)}, want {(B, T, V)} or {(B, T - P + 1, V)}")
if not okshape:
    sys.exit(1)
first = (P - 1) - off

# 2 — no leak from later continuation tokens
leaks = []
for j in [P, P + 5, T - 1]:
    ids2 = ids.clone(); ids2[:, j] = (ids2[:, j] + 7) % V
    o2, _ = run(ids2, P)
    upto = j - off
    if upto > 0:
        d = (o2[:, :upto] - out[:, :upto]).abs().max().item()
        if d > TOL:
            leaks.append((j, round(d, 6)))
report("2 no leak from later tokens", not leaks, f"leaks (pos, max diff): {leaks}" if leaks else "")

# 3 — the prompt matters
ids3 = ids.clone(); ids3[:, P - 10] = (ids3[:, P - 10] + 7) % V
o3, _ = run(ids3, P)
d3 = (o3[:, first] - out[:, first]).abs().max().item()
report("3 prediction depends on the prompt", d3 > TOL, f"(changed by {d3:.2e})")

# 4 — rows independent
diffs = []
for b in range(B):
    ob, _ = run(ids[b:b + 1], P)
    diffs.append((ob[0] - out[b]).abs().max().item())
report("4 rows independent", max(diffs) < 1e-3, f"(max diff vs single-row runs {max(diffs):.2e})")

# 5 — other split points
bad = []
for p in [0, T // 4, T // 2, T - 8]:
    try:
        o5, _ = run(ids, p)
        okp = tuple(o5.shape) in [(B, T, V), (B, T - max(p, 1) + 1, V)] and bool(torch.isfinite(o5).all())
        if not okp:
            bad.append((p, tuple(o5.shape)))
    except Exception as e:
        bad.append((p, repr(e)[:100]))
report("5 other split points", not bad, f"problems: {bad}" if bad else "")

# 6 — parameter budget
n = sum(q.numel() for q in m.parameters())
report("6 parameters within 50M-300M", 50e6 <= n <= 300e6, f"({n/1e6:.2f}M)")

# 7 — recurrent decoding near the end of the window
if hasattr(m, "prefill") and hasattr(m, "decode_step"):
    Bd, STEPS = 2, 32
    Tl = CTX if dev == "cuda" else min(CTX, 256)
    L = Tl - STEPS
    idl = torch.randint(5, V, (Bd, Tl))
    ref, offl = run(idl, L)
    errs, scale = [], ref.abs().max().item()
    with torch.no_grad():
        last, cache = m.prefill(idl[:, :L].to(dev), torch.full((Bd,), L, device=dev))
        errs.append((last.float() - ref[:, L - 1 - offl]).abs().max().item())
        for t in range(L, Tl - 1):
            last, cache = m.decode_step(idl[:, t:t + 1].to(dev), cache)
            errs.append((last.float() - ref[:, t - offl]).abs().max().item())
    worst = max(errs)
    report("7 recurrent decode matches forward", worst < 2e-3 * max(1.0, scale),
           f"(max diff {worst:.2e}, logit scale {scale:.2f}, {len(errs)} positions up to {Tl - 1})")
else:
    print("[SKIP] 7 recurrent decode (no prefill/decode_step): generation will recompute the whole "
          "sequence every token - much slower", flush=True)

# 8 — fp16 training step at full length (GPU)
if dev == "cuda":
    Bf = 2
    x = torch.randint(5, V, (Bf, CTX + 1), device=dev)
    Pf = (3 * CTX) // 4
    pl = torch.full((Bf,), Pf, device=dev)
    m.train()
    m.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16):
        lg = m(x[:, :-1], pl)
        y = x[:, 1:].clone(); y[:, :Pf - 1] = -100
        y = y[:, y.size(1) - lg.size(1):]
        loss = F.cross_entropy(lg.float().reshape(-1, V), y.reshape(-1), ignore_index=-100)
    loss.backward()
    bad_grads = [nme for nme, q in m.named_parameters() if q.grad is not None and not torch.isfinite(q.grad).all()]
    no_grad = [nme for nme, q in m.named_parameters() if q.requires_grad and q.grad is None]
    finite = bool(torch.isfinite(lg).all()) and bool(torch.isfinite(loss))
    report("8 fp16 step finite at full ctx_len", finite and not bad_grads,
           f"(loss {loss.item():.3f}; non-finite grads: {bad_grads[:5]})" if bad_grads or not finite
           else f"(loss {loss.item():.3f} ~ ln V = {torch.log(torch.tensor(float(V))).item():.3f} at init)")
    if no_grad:
        print(f"  note: {len(no_grad)} parameters got no gradient (unused?): {no_grad[:5]}"
              " -> they are dead weight, and DDP needs find_unused=True", flush=True)
    m.eval(); m.zero_grad(set_to_none=True)
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.float16):
            l16 = m(x[:, :-1], pl).float()
        l32 = m(x[:, :-1], pl).float()
    print(f"  info: |fp16 - fp32| logits max {(l16 - l32).abs().max().item():.3e} "
          f"(logit scale {l32.abs().max().item():.2f})", flush=True)
else:
    print("[SKIP] 8 fp16 step (no GPU)", flush=True)

print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED — fix model.py before training")
sys.exit(0 if ok else 1)