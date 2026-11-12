"""Synthetic tests for sum_data.py (python test_sum_data.py)."""
import json, tempfile

import numpy as np

import sum_data as sd

ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")


tmp = tempfile.mkdtemp()
rows = [dict(id=f"d{i}", document=f"doc text {i}", summary=f"sum {i}") for i in range(10)]
json.dump(rows, open(f"{tmp}/a.json", "w"))
json.dump({r["id"]: dict(text=r["document"], summary=r["summary"]) for r in rows}, open(f"{tmp}/b.json", "w"))
json.dump([dict(id=r["id"], document=r["document"]) for r in rows], open(f"{tmp}/t.json", "w"))
a, b, t = (sd.load_cases(f"{tmp}/{x}.json") for x in "abt")
check("1 loaders agree", all(a[k]["text"] == b[k]["text"] and a[k]["summary"] == b[k]["summary"] for k in a))
check("1 test has no summary", all(v["summary"] is None for v in t.values()))


tr, va = sd.split_train_val(list(a), 3, seed=1)
tr2, va2 = sd.split_train_val(list(a), 3, seed=1)
check("2 split disjoint + deterministic", len(va) == 3 and not set(tr) & set(va) and va == va2 and len(tr) == 7)


ids = np.arange(100)
check("3 first/last", list(sd.select_src(ids, 10, "first")) == list(range(10)) and list(sd.select_src(ids, 10, "last")) == list(range(90, 100)))
check("3 head_tail", list(sd.select_src(ids, 10, "head_tail", 0.3)) == [0, 1, 2] + list(range(93, 100)))
check("3 short doc unchanged", len(sd.select_src(np.arange(5), 10, "last")) == 5)


r = sd.rouge_pair("the cat sat on the mat", "the cat sat on a mat")

check("4 rouge1", abs(r["rouge1"] - 5 / 6) < 1e-9)
check("4 rougeL", abs(r["rougeL"] - 5 / 6) < 1e-9)
check("4 rouge2", abs(r["rouge2"] - 3 / 5) < 1e-9)
check("4 identical = 1, disjoint = 0", sd.rouge_pair("a b c", "a b c")["rougeL"] == 1.0 and sd.rouge_pair("a b", "c d")["rouge1"] == 0.0)
check("4 case/punctuation ignored", sd.rouge_pair("The Court, held.", "the court held")["rouge1"] == 1.0)


good = [dict(id=v["id"], summary="x") for v in t.values()]
check("5 accepts", sd.validate_prediction(good, t) == (10, 0))
for bad, why in [(good[:-1], "missing id"), (good + [good[0]], "duplicate"), ([dict(r, extra=1) for r in good], "extra key"),
                 ([dict(id=r["id"], summary=None) for r in good], "non-string")]:
    try:
        sd.validate_prediction(bad, t); check(f"5 rejects {why}", False)
    except AssertionError:
        check(f"5 rejects {why}", True)

print("\nALL TESTS PASSED" if ok else "\nSOME TESTS FAILED")