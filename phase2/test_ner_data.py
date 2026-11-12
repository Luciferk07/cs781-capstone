"""Synthetic tests for ner_data.py (run: python test_ner_data.py). A mock tokenizer imitates ByteLevel BPE:"""
import json, os, random, re, tempfile

import ner_data as nd

PIECE = re.compile(r" ?[A-Za-zÀ-ɏ]+| ?[0-9]+| ?[^\sA-Za-z0-9À-ɏ]+|\s+(?!\S)|\s+")


def mock_encode(text, piece=3):
    ids, offs = [], []
    for m in PIECE.finditer(text):
        s, e = m.span()
        tok = text[s:e]
        lead = 1 if tok.startswith(" ") and len(tok) > 1 and not tok.isspace() else 0
        body_s = s + lead
        if tok.isspace() or (e - body_s) <= piece:
            ids.append(hash(tok) % 1000); offs.append((s, e)); continue
        first = True
        k = body_s
        while k < e:
            ke = min(e, k + piece)
            ids.append(hash(text[k:ke]) % 1000)
            offs.append((s if first else k, ke)); first = False
            k = ke
    return ids, offs


ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")


LAB = nd.LABELS_DEFAULT
doc = ("IN THE SUPREME COURT OF INDIA\n\nCivil Appeal No. 1234 of 2019\n\nRamesh  Kumar Sharma ...Appellant\n"
       "versus\nState of Uttar Pradesh ...Respondent\n\nJudgment dated 12.03.2019 by Justice A.K. Sikri, "
       "under Section 302 of the Indian Penal Code, 1860, relying on (1997) 6 SCC 241. Mr. José Ávila, Advocate, "
       "for the appellant. ₹ 5,00,000 compensation.")


def span(sub, lab, nth=0):
    i = -1
    for _ in range(nth + 1):
        i = doc.index(sub, i + 1)
    return (i, i + len(sub), lab)


ents = [span("SUPREME COURT OF INDIA", "COURT"), span("1234 of 2019", "CASENO"),
        span("Ramesh  Kumar Sharma", "APP"), span("State of Uttar Pradesh", "RESP"),
        span("12.03.2019", "DATE"), span("A.K. Sikri", "JUDGE"),
        span("Section 302 of the Indian Penal Code, 1860", "STAT"), span("(1997) 6 SCC 241", "PREC"),
        span("José Ávila", "A.COUNSEL")]
for s, e, l in ents:
    assert doc[s:e].strip() == doc[s:e]


ids, offs = nd.tokenize_doc(doc, mock_encode)
check("1 trimmed offsets have no edge spaces", all(e == s or (not doc[s].isspace() and not doc[e - 1].isspace())
                                                   for s, e in offs))
raw = mock_encode(doc)[1]
check("  (mock really has leading-space raw offsets)", any(doc[s] == " " and e - s > 1 for s, e in raw))


for scheme in ("IO", "BIO", "BIOES"):
    for mode in ("lenient", "strict"):
        sc, sk, mid = nd.oracle_check({"d": doc}, {"d": ents}, mock_encode, LAB, scheme, mode)
        exp = 1.0
        check(f"2 oracle {scheme}/{mode}", abs(sc["macro_f1_present"] - exp) < 1e-9,
              f"macro_present {sc['macro_f1_present']:.3f} skipped {sk} mid-token {dict(mid)}")


adj = [(doc.index("Kumar"), doc.index("Kumar") + 5, "APP"), (doc.index("Sharma"), doc.index("Sharma") + 6, "APP")]
sc_io = nd.oracle_check({"d": doc}, {"d": adj}, mock_encode, LAB, "IO")[0]
sc_bio = nd.oracle_check({"d": doc}, {"d": adj}, mock_encode, LAB, "BIO")[0]
check("3 adjacent same-label: BIO exact, IO merges", sc_bio["per_label"]["APP"]["f1"] == 1.0 and
      sc_io["per_label"]["APP"]["f1"] < 1.0, f"(BIO {sc_bio['per_label']['APP']['f1']:.2f}, IO {sc_io['per_label']['APP']['f1']:.2f})")


i = doc.index("Sharma")
midg = [(i, i + 2, "APP")]
sc, sk, mid = nd.oracle_check({"d": doc}, {"d": midg}, mock_encode, LAB, "BIO")
check("4 mid-token boundary detected", mid["APP"] == 1 and sc["per_label"]["APP"]["f1"] == 0.0)


ov = [span("State of Uttar Pradesh", "RESP"), span("Uttar Pradesh", "AUTH")]
tags, sk = nd.token_labels(offs, ov, LAB, "BIO")
check("5 overlapping entity skipped + counted", sk == 1)


g_exc = {"d": [(s, e, l, doc[s:e]) for s, e, l in ents]}
g_inc = {"d": [(s, e - 1, l, doc[s:e]) for s, e, l in ents]}
c1, c2 = nd.detect_end_convention({"d": doc}, g_exc), nd.detect_end_convention({"d": doc}, g_inc)
check("6 exclusive detected", c1["inclusive"] is False and c1["neither"] == 0)
check("6 inclusive detected", c2["inclusive"] is True and c2["neither"] == 0)
back = nd.from_internal(nd.to_internal(g_inc, True), True)
check("6 inclusive round trip", [tuple(x) for x in back["d"]] == [(s, e, l) for s, e, l, _ in g_inc["d"]])


good = True
for n in [1, 5, 99, 100, 101, 250, 1000, 1023]:
    for Wn, St in [(100, 50), (100, 100), (64, 48)]:
        st = nd.make_windows(n, Wn, St)
        cov = set()
        for s in st:
            cov.update(range(s, min(n, s + Wn)))
        good &= cov == set(range(n)) and all(s + Wn <= max(n, Wn) for s in st)
check("7 windows cover all tokens", good)


gold = {"a": [(0, 5, "DATE"), (10, 20, "COURT")], "b": [(3, 8, "DATE")]}
pred = {"a": [(0, 5, "DATE"), (10, 19, "COURT")], "b": [(3, 8, "COURT")]}
sc = nd.strict_scores(pred, gold, LAB)
check("8 scorer exact-match", sc["per_label"]["DATE"]["tp"] == 1 and sc["per_label"]["DATE"]["fn"] == 1 and
      sc["per_label"]["COURT"]["fp"] == 2 and sc["per_label"]["COURT"]["tp"] == 0)


random.seed(0)
bad = 0
for _ in range(300):
    names = nd.label_names(LAB, "BIOES")
    tg = [random.randrange(len(names)) if random.random() < 0.4 else 0 for _ in offs]
    for mode in ("lenient", "strict"):
        for s, e, l in nd.decode_spans(tg, offs, LAB, "BIOES", mode):
            if not (0 <= s < e <= len(doc)) or doc[s].isspace() or doc[e - 1].isspace():
                bad += 1
check("9 decoded spans always clean", bad == 0, f"(bad {bad})")


tmp = tempfile.mkdtemp()
json.dump({"d": doc}, open(os.path.join(tmp, "t.json"), "w"))
json.dump({"d": [[s, e, l, doc[s:e]] for s, e, l in ents]}, open(os.path.join(tmp, "g1.json"), "w"))
json.dump([{"id": "d", "entities": [{"start": s, "end": e, "label": l, "text": doc[s:e]} for s, e, l in ents]}],
          open(os.path.join(tmp, "g2.json"), "w"))
t1, g1 = nd.load_split(os.path.join(tmp, "t.json"), os.path.join(tmp, "g1.json"))
t2, g2 = nd.load_split(os.path.join(tmp, "t.json"), os.path.join(tmp, "g2.json"))
check("10 both gold layouts load identically", g1 == g2 and len(g1["d"]) == len(ents))


p = nd.from_internal({"d": [tuple(x) for x in ents]}, False)
check("11 validator accepts good file", nd.validate_prediction(p, {"d": doc}, LAB) == len(ents))
try:
    nd.validate_prediction({"d": [[0, 5, "FOO"]]}, {"d": doc}, LAB); check("11 validator rejects bad label", False)
except AssertionError:
    check("11 validator rejects bad label", True)


ids_ = [str(i) for i in range(200)]
a1, b1 = nd.split_docs(ids_, 0.15, 0); a2, b2 = nd.split_docs(ids_, 0.15, 0)
check("12 split deterministic + disjoint", a1 == a2 and b1 == b2 and not set(a1) & set(b1) and 10 < len(b1) < 50,
      f"(val {len(b1)} of 200)")


txt = "Judge \u00c9mile spoke"
offs13 = [(0, 5), (6, 7), (6, 7), (7, 11), (12, 17)]
names = nd.label_names(LAB, "BIO"); B_J, I_J = names.index("B-JUDGE"), names.index("I-JUDGE")
for tags13, label in [([0, B_J, B_J, I_J, 0], "B,B,I"), ([0, B_J, B_J, 0, 0], "B,B"), ([0, B_J, I_J, I_J, 0], "B,I,I")]:
    out = nd.decode_spans(tags13, offs13, LAB, "BIO", "lenient")
    check(f"13 same-char tokens ({label}) -> one span, no duplicates", len(out) == 1 and len(set(out)) == len(out), str(out))
tg, _ = nd.token_labels(offs13, [(6, 11, "JUDGE")], LAB, "BIO")
check("13 gold over a split character round-trips", nd.decode_spans(tg, offs13, LAB, "BIO") == [(6, 11, "JUDGE")])

print("\nALL TESTS PASSED" if ok else "\nSOME TESTS FAILED")