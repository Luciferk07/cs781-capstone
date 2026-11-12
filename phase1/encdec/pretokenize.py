"""pretokenize.py — one-time data prep for all Phase-1 runs (ancillary code, not model code)."""
import argparse, json, os, random, time, zlib

import numpy as np


def parse_kv(items, cast=str):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        out[k.strip()] = cast(v.replace("_", "")) if cast is int else cast(v)
    return out


def is_heldout(text, permille):

    return zlib.crc32(text[:2000].encode("utf-8", "ignore")) % 1000 < permille


def load_tokenizer(tok_dir):
    from tokenizers import ByteLevelBPETokenizer
    tok = ByteLevelBPETokenizer(
        vocab=os.path.join(tok_dir, "vocab.json"),
        merges=os.path.join(tok_dir, "merges.txt"),
    )
    return tok


class BinWriter:

    def __init__(self, path):
        self.f = open(path, "wb")
        self.n = 0

    def write(self, ids_list):
        if not ids_list:
            return
        arr = np.fromiter((t for ids in ids_list for t in ids), dtype=np.uint16)
        arr.tofile(self.f)
        self.n += arr.size

    def close(self):
        self.f.close()


def process_source(name, path, budget, tok, eos_id, args, local_pairs, rng):
    import ijson

    tr = BinWriter(os.path.join(args.out, f"train_{name}.bin"))
    va = BinWriter(os.path.join(args.out, f"val_{name}.bin"))
    stats = dict(docs_seen=0, docs_train=0, docs_val=0, docs_short=0, docs_skipped_stride=0)

    batch_tr, batch_va = [], []
    t0 = time.time()

    def flush():
        for batch, writer in ((batch_tr, tr), (batch_va, va)):
            if batch:
                enc = tok.encode_batch(batch)
                writer.write([e.ids + [eos_id] for e in enc])
                batch.clear()

    with open(path, "rb") as f:
        for i, rec in enumerate(ijson.items(f, "item")):
            text = rec.get(args.text_key, "") if isinstance(rec, dict) else ""
            if not isinstance(text, str) or len(text) < args.min_chars:
                stats["docs_short"] += 1
                continue
            if args.keep_every > 1 and i % args.keep_every != 0:
                stats["docs_skipped_stride"] += 1
                continue
            stats["docs_seen"] += 1

            if is_heldout(text, args.val_permille):
                stats["docs_val"] += 1
                if va.n < args.val_budget_per_source:
                    batch_va.append(text)
                words = text.split()
                need = args.prompt_words + args.cont_words
                if len(words) >= need and len(local_pairs[name]) < args.local_pairs_per_source:
                    s = rng.randint(0, len(words) - need)
                    local_pairs[name].append((
                        " ".join(words[s:s + args.prompt_words]),
                        " ".join(words[s + args.prompt_words:s + need]),
                    ))
            else:
                stats["docs_train"] += 1
                batch_tr.append(text)

            if len(batch_tr) >= args.batch_docs or len(batch_va) >= args.batch_docs:
                flush()
                if stats["docs_seen"] % (args.batch_docs * 20) == 0:
                    rate = tr.n / max(1e-9, time.time() - t0)
                    print(f"  [{name}] train={tr.n/1e6:8.1f}M tok  val={va.n/1e6:6.2f}M  "
                          f"docs={stats['docs_seen']:,}  {rate/1e3:.0f}K tok/s", flush=True)
            if tr.n >= budget:
                break
    flush()
    tr.close(); va.close()
    stats.update(train_tokens=tr.n, val_tokens=va.n, seconds=round(time.time() - t0, 1))
    print(f"  [{name}] DONE train={tr.n/1e6:.1f}M val={va.n/1e6:.2f}M "
          f"in {stats['seconds']/60:.1f} min", flush=True)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok_dir", required=True, help="folder with vocab.json + merges.txt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--file", action="append", required=True, help="name=/path/to/file.json")
    ap.add_argument("--budget", action="append", required=True,
                    help="name=max_train_tokens (your data-mix choice)")
    ap.add_argument("--text_key", default="text")
    ap.add_argument("--min_chars", type=int, default=100)
    ap.add_argument("--keep_every", type=int, default=1,
                    help="keep every k-th doc to spread sampling across a file (1 = all)")
    ap.add_argument("--val_permille", type=int, default=5, help="held-out docs per 1000")
    ap.add_argument("--val_budget_per_source", type=int, default=2_000_000)
    ap.add_argument("--local_pairs_per_source", type=int, default=100)
    ap.add_argument("--prompt_words", type=int, default=512)
    ap.add_argument("--cont_words", type=int, default=128)
    ap.add_argument("--batch_docs", type=int, default=256)
    ap.add_argument("--eos_token", default="</s>")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    files = parse_kv(args.file)
    budgets = parse_kv(args.budget, int)
    missing = set(files) - set(budgets)
    if missing:
        raise SystemExit(f"No --budget given for: {sorted(missing)}")
    os.makedirs(args.out, exist_ok=True)

    tok = load_tokenizer(args.tok_dir)
    vocab = tok.get_vocab_size()
    if vocab > 65535:
        raise SystemExit(f"vocab {vocab} does not fit uint16")
    eos_id = tok.token_to_id(args.eos_token)
    if eos_id is None:
        raise SystemExit(f"EOS token {args.eos_token!r} not in vocab")

    rng = random.Random(args.seed)
    local_pairs = {n: [] for n in files}
    meta = dict(settings=vars(args), vocab_size=vocab, eos_id=eos_id, dtype="uint16", sources={})

    for name, path in files.items():
        print(f"Processing {name}: {path} (budget {budgets[name]/1e6:.0f}M train tokens)")
        meta["sources"][name] = process_source(name, path, budgets[name], tok, eos_id,
                                               args, local_pairs, rng)

    test, ref, origin = {}, {}, {}
    k = 1
    for name, pairs in local_pairs.items():
        for p, r in pairs:
            test[str(k)] = {"text": p}
            ref[str(k)] = {"text": r}
            origin[str(k)] = name
            k += 1
    for fn, obj in (("local_test.json", test), ("local_ref.json", ref), ("local_origin.json", origin)):
        with open(os.path.join(args.out, fn), "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)
    meta["local_pairs"] = {n: len(p) for n, p in local_pairs.items()}
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    tot = sum(s["train_tokens"] for s in meta["sources"].values())
    print(f"\nAll done. Train tokens: {tot/1e6:.1f}M | local test pairs: {len(test)}")
    print(f"Files in {args.out}: train_*.bin, val_*.bin, local_test.json, local_ref.json, meta.json")


if __name__ == "__main__":
    main()