"""Synthetic tests for cj_data.py (python test_cj_data.py)."""
import json, random, tempfile

import cj_data as cd

ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")


random.seed(0)

for n in [0, 1, 50, 512, 513, 1000, 5000, 73001]:
    for L, S in [(512, 512), (512, 384), (256, 128)]:
        sp = cd.chunk_doc(n, L, S, None, "all")
        cov = set()
        for a, b in sp:
            cov.update(range(a, b))
        good = all(0 <= a < b <= max(1, n) and b - a <= L for a, b in sp)
        full = n == 0 or cov == set(range(n))
        check(f"1 n={n} L={L} S={S}: spans valid + whole doc covered", good and full, f"({len(sp)} chunks)")


sp = cd.chunk_doc(10000, 512, 512, None, "all")
for sel in ("first", "last", "head_tail", "spread"):
    got = cd.chunk_doc(10000, 512, 512, 4, sel)
    check(f"2 select {sel}: 4 chunks, subset, ordered", len(got) == 4 and all(g in sp for g in got) and got == sorted(got))
check("2 first/last are ends", cd.chunk_doc(10000, 512, 512, 3, "first") == sp[:3] and cd.chunk_doc(10000, 512, 512, 3, "last") == sp[-3:])
check("2 head_tail splits", cd.chunk_doc(10000, 512, 512, 3, "head_tail") == sp[:2] + sp[-1:])
check("2 short doc unaffected", cd.chunk_doc(300, 512, 512, 4, "last") == [(0, 300)])


gold = {"a": 1, "b": 1, "c": 0, "d": 0}
pred = {"a": 1, "b": 0, "c": 0, "d": 0}
sc = cd.binary_scores(pred, gold)
f0, f1 = 2 * (2 / 3) * 1 / (2 / 3 + 1), 2 * 1 * .5 / 1.5
check("3 macro-F1", abs(sc["macro_f1"] - (f0 + f1) / 2) < 1e-9 and sc["accuracy"] == .75 and sc["confusion"] == [[2, 0], [1, 1]])
t, f = cd.best_threshold({"a": .9, "b": .4, "c": .3, "d": .1}, gold)
check("3 threshold search finds perfect split", f == 1.0 and .3 < t <= .4, f"(t={t})")


tmp = tempfile.mkdtemp()
rows = [dict(id=f"case_{i}", text="x " * i, label=i % 2) for i in range(10)]
json.dump(rows, open(f"{tmp}/a.json", "w"))
json.dump({r["id"]: dict(text=r["text"], label=str(r["label"])) for r in rows}, open(f"{tmp}/b.json", "w"))
json.dump({"data": [dict(case_id=r["id"], text=[r["text"]], label=[r["label"]]) for r in rows]}, open(f"{tmp}/c.json", "w"))
json.dump([dict(id=r["id"], text=r["text"]) for r in rows], open(f"{tmp}/t.json", "w"))
a, b, c, t = (cd.load_cases(f"{tmp}/{x}.json") for x in "abct")
same = lambda u, v: all(u[k]["label"] == v[k]["label"] and u[k]["text"].strip() == v[k]["text"].strip() for k in u)
check("4 loaders agree", set(a) == set(b) == set(c) and same(a, b) and same(a, c))
check("4 test file has no labels", all(v["label"] is None for v in t.values()))
json.dump([dict(id=7, text="a", label=1)], open(f"{tmp}/n.json", "w"))
check("4 integer id kept as int", cd.load_cases(f"{tmp}/n.json")["7"]["id"] == 7)

json.dump([dict(id="x", text="a", label=1), dict(id="x", text="a", label=1), dict(id="x", text="b", label=0)],
          open(f"{tmp}/d.json", "w"))
dd = cd.load_cases(f"{tmp}/d.json")
check("4 repeated ids: identical dropped, different kept", sorted(dd) == ["x", "x#2"] and dd["x#2"]["label"] == 0)


pl = [dict(id=v["id"], pred=i % 2) for i, v in enumerate(t.values())]
check("5 validator accepts", cd.validate_prediction(pl, t) == 10)
for bad, why in [(pl[:-1], "missing id"), (pl + [pl[0]], "duplicate"), ([dict(r, pred=float(r["pred"])) for r in pl], "float pred"),
                 ([dict(r, extra=1) for r in pl], "extra key")]:
    try:
        cd.validate_prediction(bad, t); check(f"5 validator rejects {why}", False)
    except AssertionError:
        check(f"5 validator rejects {why}", True)

print("\nALL TESTS PASSED" if ok else "\nSOME TESTS FAILED")