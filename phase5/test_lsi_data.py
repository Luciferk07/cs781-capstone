"""Synthetic tests for lsi_data.py (python test_lsi_data.py)."""
import json, tempfile

import numpy as np

import lsi_data as ld

ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")


for n in [0, 1, 513, 5000]:
    sp = ld.chunk_doc(n, 512, 412, None, "all")
    cov = set()
    for a, b in sp:
        cov.update(range(a, b))
    check(f"1 n={n}: whole doc covered", n == 0 or cov == set(range(n)))
check("1 head_tail", ld.chunk_doc(10000, 512, 512, 3, "head_tail") == ld.chunk_doc(10000, 512, 512, None, "all")[:2] + ld.chunk_doc(10000, 512, 512, None, "all")[-1:])


Y = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 0]], bool)
P = np.array([[1, 0, 0], [0, 1, 1], [1, 0, 0]], bool)
sc = ld.scores(P, Y)

check("2 macro-F1", abs(sc["macro_f1"] - 2 / 3) < 1e-9 and abs(sc["macro_f1_present"] - 2 / 3) < 1e-9)
check("2 micro-F1 + subset acc", abs(sc["micro_f1"] - 2 * 3 / (2 * 3 + 1 + 1)) < 1e-9 and abs(sc["subset_acc"] - 1 / 3) < 1e-9)
Y2 = np.array([[1, 0], [1, 0]], bool); P2 = np.array([[1, 0], [1, 0]], bool)
s2 = ld.scores(P2, Y2)
check("2 labels without gold excluded from 'present' macro", s2["macro_f1_present"] == 1.0 and s2["macro_f1"] == 0.5)


rng = np.random.default_rng(0)
Yr = rng.random((200, 5)) < 0.2
Pr = np.clip(Yr * 0.3 + rng.random((200, 5)) * 0.6, 0, 1)
t, f = ld.tune_global(Pr, Yr)
check("3 global threshold beats 0.5", f >= ld.scores(ld.decide(Pr, 0.5), Yr)["macro_f1_present"], f"(t={t}, f={f:.3f})")
thr = ld.tune_per_label(Pr, Yr, t)
check("3 per-label >= global on dev", ld.scores(ld.decide(Pr, thr), Yr)["macro_f1_present"] >= f - 1e-9)
pb = ld.decide(np.array([[0.1, 0.2, 0.05]]), 0.9, min_labels=1)
check("3 min_labels forces top label", pb.tolist() == [[False, True, False]])


tmp = tempfile.mkdtemp()
rows = [dict(id=f"c{i}", text="x " * i, labels=[i % 3, (i + 1) % 3]) for i in range(6)]
json.dump(rows + [rows[0]], open(f"{tmp}/a.json", "w"))
a = ld.load_cases(f"{tmp}/a.json")
check("4 labels sorted unique ints, identical repeat dropped", len(a) == 6 and a["c1"]["labels"] == [1, 2])
json.dump([dict(id="t1", text="a"), dict(id="t2", text="b")], open(f"{tmp}/t.json", "w"))
t = ld.load_cases(f"{tmp}/t.json")
check("4 test without labels", all(v["labels"] is None for v in t.values()))
json.dump([dict(id="z", text="a", labels="7")], open(f"{tmp}/s.json", "w"))
check("4 scalar/str label -> [7]", ld.load_cases(f"{tmp}/s.json")["z"]["labels"] == [7])


good = [dict(id="t1", pred=[1, 7]), dict(id="t2", pred=[])]
check("5 accepts (empty list allowed)", ld.validate_prediction(good, t, [1, 7, 9]) == 2)
for bad, why in [([dict(id="t1", pred=[1, 1]), dict(id="t2", pred=[])], "duplicate ids in pred"),
                 ([dict(id="t1", pred=[3]), dict(id="t2", pred=[])], "unknown label"),
                 ([dict(id="t1", pred=[1.0]), dict(id="t2", pred=[])], "float label"),
                 ([dict(id="t1", pred=[1])], "missing doc")]:
    try:
        ld.validate_prediction(bad, t, [1, 7, 9]); check(f"5 rejects {why}", False)
    except AssertionError:
        check(f"5 rejects {why}", True)

print("\nALL TESTS PASSED" if ok else "\nSOME TESTS FAILED")