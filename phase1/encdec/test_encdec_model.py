"""test_encdec_model.py — correctness checks for YOUR encoder-decoder model.py (test code only, no architecture)."""
import json, sys

import torch

import model as model_lib

cfg = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else {"model": {}, "train": {"ctx_len": 256}}
mcfg = dict(cfg["model"])
mcfg.setdefault("vocab_size", 32000)
mcfg.setdefault("max_seq_len", int(cfg["train"]["ctx_len"]))
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

m = model_lib.build_model(mcfg).to(dev).float().eval()
V = mcfg["vocab_size"]
T = min(96, mcfg["max_seq_len"])
P = (3 * T) // 4
B = 3
TOL = 1e-4
ok = True


def report(name, passed, detail=""):
    global ok
    ok &= bool(passed)
    print(f"[{'PASS' if passed else 'FAIL'}] {name} {detail}")


def run(ids, p):
    with torch.no_grad():
        out = m(ids.to(dev), torch.full((ids.shape[0],), p, device=dev)).float()
    off = ids.shape[1] - out.shape[1]
    return out, off


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

# 3 — the prompt reaches the decoder (encoder output is used)
ids3 = ids.clone(); ids3[:, 2] = (ids3[:, 2] + 7) % V
o3, _ = run(ids3, P)
d3 = (o3[:, first] - out[:, first]).abs().max().item()
report("3 decoder reads the prompt", d3 > TOL, f"(first prediction changed by {d3:.2e}; ~0 = cross-attention unused)")

# 4 — rows independent
diffs = []
for b in range(B):
    ob, _ = run(ids[b:b + 1], P)
    diffs.append((ob[0] - out[b]).abs().max().item())
report("4 rows independent", max(diffs) < 1e-3, f"(max diff vs single-row runs {max(diffs):.2e})")

# 5 — other split points
bad = []
for p in [T // 4, T // 2, T - 8]:
    try:
        o5, off5 = run(ids, p)
        if tuple(o5.shape) not in [(B, T, V), (B, T - p + 1, V)] or not torch.isfinite(o5).all():
            bad.append((p, tuple(o5.shape)))
    except Exception as e:
        bad.append((p, repr(e)[:80]))
report("5 other split points", not bad, f"problems: {bad}" if bad else "")

# 6 — parameter budget
n = sum(q.numel() for q in m.parameters())
report("6 parameters within 50M-300M", 50e6 <= n <= 300e6, f"({n/1e6:.2f}M)")

# 7 — KV cache
if hasattr(m, "prefill") and hasattr(m, "decode_step"):
    L = P
    with torch.no_grad():
        last, cache = m.prefill(ids[:, :L].to(dev), torch.full((B,), L, device=dev))
        full, offf = run(ids[:, :L], L)
        errs = [(last.float() - full[:, -1]).abs().max().item()]
        for t in range(L, min(L + 12, T)):
            last, cache = m.decode_step(ids[:, t:t + 1].to(dev), cache)
            ref, _ = run(ids[:, :t + 1], L)
            errs.append((last.float() - ref[:, -1]).abs().max().item())
    report("7 KV cache matches forward", max(errs) < 1e-3, f"(max diff {max(errs):.2e} over {len(errs)} steps)")
else:
    print("[SKIP] 7 KV cache (prefill/decode_step not implemented)")

print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED — fix model.py before training")
sys.exit(0 if ok else 1)