"""Synthetic tests for rr_data.py (python test_rr_data.py)."""
import json, os, random, tempfile

import rr_data as rd

ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")


random.seed(0)
LAB = ["Fact", "Issue", "ArgumentPetitioner", "ArgumentRespondent", "Statute", "Dissent", "PrecedentReliedUpon",
       "PrecedentNotReliedUpon", "PrecedentOverruled", "RulingByLowerCourt", "RatioOfTheDecision",
       "RulingByPresentCourt", "None"]
enc = lambda s: [hash(w) % 30000 + 5 for w in s.split()]

docs = {}
for d in range(30):
    n = random.randint(1, 80)
    sents = [" ".join(f"w{random.randint(0, 999)}" for _ in range(random.choice([0, 1, 3, 8, 20, 60, 400]))) for _ in range(n)]
    docs[f"doc{d}"] = {"text": sents, "labels": [random.choice(LAB) for _ in range(n)]}
tok = rd.tokenize_sentences(docs, enc, prefix_space=True)

# 1-3 windows: coverage, length, order, slot consistency
for mode in ("sentence", "context"):
    for W, sep in [(64, None), (256, 1), (1024, 1), (30, 1)]:
        wins = rd.build_windows(tok, mode, W, sep_id=sep, target_frac=0.5)
        missing, dup = rd.coverage_check(wins, tok)
        good_len = all(len(w["ids"]) <= W and len(w["ids"]) == len(w["slots"]) for w in wins)
        good_slots = all(max(w["slots"]) == len(w["sent_idx"]) - 1 and min(s for s in w["slots"] if s >= 0) == 0 for w in wins)
        ordered = all(w["sent_idx"] == sorted(w["sent_idx"]) and
                      w["sent_idx"] == list(range(w["sent_idx"][0], w["sent_idx"][0] + len(w["sent_idx"]))) for w in wins)
        tgt_contig = all((lambda t: t == list(range(t[0], t[0] + len(t))))([s for s, f in zip(w["sent_idx"], w["target"]) if f])
                         for w in wins)
        check(f"1 {mode} W={W} sep={sep}: every sentence scored exactly once", not missing and not dup,
              f"(missing {len(missing)}, dup {len(dup)}, windows {len(wins)})")
        check(f"2 {mode} W={W}: lengths <= window, slots consistent", good_len and good_slots)
        check(f"3 {mode} W={W}: sentence order kept, targets contiguous", ordered and tgt_contig)


wins = rd.build_windows(tok, "context", 1024, sep_id=1, target_frac=0.3)
check("4 context windows contain non-target sentences", any(not all(w["target"]) for w in wins))


gold = {"a": ["Fact", "Fact", "Issue"], "b": ["None"]}
pred = {"a": ["Fact", "Issue", "Issue"], "b": ["None"]}
sc = rd.macro_scores(pred, gold, LAB)
check("5 scorer", abs(sc["per_label"]["Fact"]["f1"] - 2 / 3) < 1e-9 and sc["per_label"]["Issue"]["f1"] == 2 / 3
      and sc["accuracy"] == 0.75 and abs(sc["macro_f1_present"] - (2 / 3 + 2 / 3 + 1) / 3) < 1e-9)


tmp = tempfile.mkdtemp()
json.dump([{"id": k, "text": v["text"], "labels": v["labels"]} for k, v in docs.items()], open(f"{tmp}/a.json", "w"))
json.dump({k: {"text": v["text"], "labels": v["labels"]} for k, v in docs.items()}, open(f"{tmp}/b.json", "w"))
json.dump({k: v["text"] for k, v in docs.items()}, open(f"{tmp}/c.json", "w"))
a, b, c = rd.load_docs(f"{tmp}/a.json"), rd.load_docs(f"{tmp}/b.json"), rd.load_docs(f"{tmp}/c.json")
check("6 loaders agree", a == b and all(c[k]["text"] == a[k]["text"] and c[k]["labels"] is None for k in a))
json.dump(LAB, open(f"{tmp}/v1.json", "w")); json.dump({l: i for i, l in enumerate(LAB)}, open(f"{tmp}/v2.json", "w"))
json.dump({"labels": LAB}, open(f"{tmp}/v3.json", "w"))
check("6 label vocab layouts", rd.read_label_vocab(f"{tmp}/v1.json") == rd.read_label_vocab(f"{tmp}/v2.json")
      == rd.read_label_vocab(f"{tmp}/v3.json") == LAB)


p = {k: v["labels"] for k, v in docs.items()}
check("7 validator accepts", rd.validate_prediction(p, docs, LAB) == sum(len(v["text"]) for v in docs.values()))
bad = dict(p); bad["doc0"] = bad["doc0"][:-1] if len(bad["doc0"]) > 1 else ["Fact", "Fact"]
try:
    rd.validate_prediction(bad, docs, LAB); check("7 validator rejects wrong count", False)
except AssertionError:
    check("7 validator rejects wrong count", True)

print("\nALL TESTS PASSED" if ok else "\nSOME TESTS FAILED")