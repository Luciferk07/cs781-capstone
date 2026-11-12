"""Synthetic tests for pcr_data.py (python test_pcr_data.py)."""
import json, tempfile

import numpy as np

import pcr_data as pd_

ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")


tmp = tempfile.mkdtemp()

q = [dict(id="0000021652", text=["First sentence.", "Second one."], relevant_candidates=["0000590455", "0000000987"]),
     dict(id="0000030721", text=["Another query."], relevant_candidates=["0000004354"])]
json.dump(q, open(f"{tmp}/q.json", "w"))
json.dump({r["id"]: dict(text=r["text"], relevant_candidates=r["relevant_candidates"]) for r in q}, open(f"{tmp}/q2.json", "w"))
json.dump([dict(id=r["id"], text=r["text"]) for r in q], open(f"{tmp}/qt.json", "w"))
pool = [dict(id=f"{i:010d}", text=[f"cand {i}"]) for i in range(20)] + [dict(id="0000590455", text="x y"), dict(id="0000004354", text=["z"])]
json.dump(pool, open(f"{tmp}/p.json", "w"))
json.dump({r["id"]: r["text"] for r in pool}, open(f"{tmp}/p2.json", "w"))
a, b, t = pd_.load_queries(f"{tmp}/q.json"), pd_.load_queries(f"{tmp}/q2.json"), pd_.load_queries(f"{tmp}/qt.json")
check("1 query loaders agree + sentences joined", a == b and a["0000021652"]["text"] == "First sentence. Second one.")
check("1 ids kept as zero-padded strings", "0000021652" in a and a["0000021652"]["relevant"] == ["0000590455", "0000000987"])
check("1 test queries have no labels", all(v["relevant"] is None for v in t.values()))
P, P2 = pd_.load_pool(f"{tmp}/p.json"), pd_.load_pool(f"{tmp}/p2.json")
check("1 pool loaders agree", P == P2 and len(P) == 22)


ids = np.arange(1000, dtype=np.uint16)
c = pd_.chunk_ids(ids, 256, 3, "first")
check("2 first", len(c) == 3 and c[0][0] == 0 and c[1][0] == 256 and all(len(x) == 256 for x in c))
c = pd_.chunk_ids(ids, 256, 2, "head_tail")
check("2 head_tail", len(c) == 2 and c[0][0] == 0 and c[-1][-1] == 999)
c = pd_.chunk_ids(ids, 256, 3, "spread")
check("2 spread covers start and end", len(c) == 3 and c[0][0] == 0 and c[-1][-1] == 999)
check("2 short doc = one chunk", len(pd_.chunk_ids(ids[:100], 256, 4)) == 1 and len(pd_.chunk_ids(ids[:100], 256, 4)[0]) == 100)
check("2 stride overlap", pd_.chunk_ids(ids, 256, 3, "first", stride=128)[1][0] == 128)
check("2 empty doc -> one token", len(pd_.chunk_ids(np.zeros(0, np.uint16), 256, 2)[0]) == 1)


ranked = {"q1": ["a", "b", "c", "d", "e", "f", "g"], "q2": ["x", "y", "z", "u", "v", "w", "s"]}
gold = {"q1": ["a", "c", "h"], "q2": ["s", "t"]}
m = pd_.micro_at_k(ranked, gold, 7)

check("3 micro P/R", abs(m["precision"] - 3 / 14) < 1e-12 and abs(m["recall"] - 3 / 5) < 1e-12)
check("3 micro F1", abs(m["f1"] - 2 * (3 / 14) * (3 / 5) / (3 / 14 + 3 / 5)) < 1e-12)
check("3 only top-7 count", pd_.micro_at_k({"q": list("abcdefgh")}, {"q": ["h"]})["tp"] == 0)


docs = {"d0": np.array([1, 2, 3, 4]), "d1": np.array([5, 6, 7, 8]), "d2": np.array([1, 1, 9, 9])}
bm = pd_.BM25(docs)
s = bm.scores(np.array([9, 1]))
check("4 bm25 ranking", pd_.rank_pool(s, bm.keys)[0] == "d2" and s[1] == 0)
check("4 exclude", pd_.rank_pool(s, bm.keys, exclude=["d2"])[0] == "d0")


good = {qid: [f"{i:010d}" for i in range(7)] for qid in t}
check("5 accepts", pd_.validate_prediction(good, t, P) == 2)
bads = [({k: v[:6] for k, v in good.items()}, "6 ids"), ({k: v[:6] + v[:1] for k, v in good.items()}, "duplicate"),
        ({k: v[:6] + ["9999999999"] for k, v in good.items()}, "not in pool"), (dict(list(good.items())[:1]), "missing query")]
for bad, why in bads:
    try:
        pd_.validate_prediction(bad, t, P); check(f"5 rejects {why}", False)
    except AssertionError:
        check(f"5 rejects {why}", True)

print("\nALL TESTS PASSED" if ok else "\nSOME TESTS FAILED")