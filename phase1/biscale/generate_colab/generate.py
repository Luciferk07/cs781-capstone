import glob, json, os, re, sys, time, zipfile

import torch
import torch.distributed as dist
import torch.nn.functional as F

CFG = json.load(open(sys.argv[1]))
IN_PATH, OUT_PREFIX = sys.argv[2], sys.argv[3]
RUN, MODEL_CFG, GEN = CFG["run"], CFG["model"], CFG["gen"]
TARGET_WORDS = 128

if RUN.get("smoke_test"):
    import dummy_model as model_lib
else:
    import model as model_lib

if int(os.environ.get("WORLD_SIZE", "1")) > 1:
    dist.init_process_group("nccl")
    RANK, WORLD, LOCAL = dist.get_rank(), dist.get_world_size(), int(os.environ["LOCAL_RANK"])
else:
    RANK, WORLD, LOCAL = 0, 1, 0
torch.cuda.set_device(LOCAL)
DEV = torch.device("cuda", LOCAL)
torch.manual_seed(int(GEN.get("seed", 0)) + RANK)


def log(*a):
    if RANK == 0:
        print(*a, flush=True)


# ----------------------------------------------------------------------------- tokenizer + model
from tokenizers import ByteLevelBPETokenizer

tok = ByteLevelBPETokenizer(vocab=os.path.join(RUN["tokenizer_dir"], "vocab.json"),
                            merges=os.path.join(RUN["tokenizer_dir"], "merges.txt"))
CTX = int(CFG["train"]["ctx_len"])
MODEL_CFG.setdefault("vocab_size", tok.get_vocab_size())
MODEL_CFG.setdefault("max_seq_len", CTX)

ck_choice = GEN["checkpoint"]
ck_path = ck_choice if os.path.exists(ck_choice) else os.path.join(RUN["out_dir"], f"{RUN['run_name']}_{ck_choice}.pt")
model = model_lib.build_model(MODEL_CFG).to(DEV)
ck = torch.load(ck_path, map_location=DEV, weights_only=False)
model.load_state_dict(ck["model"])
model.eval()
log(f"Loaded {ck_path} (step {ck.get('step')}, val {ck.get('val_loss')})")
del ck
HAS_CACHE = hasattr(model, "prefill") and hasattr(model, "decode_step")
log("Decoding with KV cache (prefill/decode_step)" if HAS_CACHE else
    "No prefill/decode_step on the model: recomputing the full sequence each step (slower)")

# ----------------------------------------------------------------------------- sampling utils
MAX_NEW = int(GEN["max_new_tokens"])
PROMPT_MAX = CTX - MAX_NEW
assert PROMPT_MAX > 64, "ctx_len too small for max_new_tokens"
BAN = [tok.token_to_id(t) for t in ("<s>", "</s>", "<unk>", "<pad>", "<mask>") if tok.token_to_id(t) is not None]


class NgramBlocker:

    def __init__(self, n, prompt_ids, scope):
        self.n, self.seen = n, {}
        self.hist = list(prompt_ids[-(n - 1):]) if (n and scope != "all") else []
        if n and scope == "all":
            for i in range(len(prompt_ids) - n + 1):
                self.seen.setdefault(tuple(prompt_ids[i:i + n - 1]), set()).add(prompt_ids[i + n - 1])
            self.hist = list(prompt_ids[-(n - 1):])
        self.scope = scope
        self.gen_count = 0

    def banned(self):
        if not self.n or len(self.hist) < self.n - 1:
            return ()
        return self.seen.get(tuple(self.hist[-(self.n - 1):]), ())

    def push(self, t):
        if self.n:
            key = tuple(self.hist[-(self.n - 1):])

            if len(key) == self.n - 1 and (self.scope == "all" or self.gen_count >= self.n - 1):
                self.seen.setdefault(key, set()).add(t)
            self.hist.append(t)
            self.gen_count += 1


def pick_next(logits, gen_tokens, blockers):
    if BAN:
        logits[:, BAN] = -float("inf")
    rp = GEN["rep_penalty"]
    if rp and rp != 1.0 and gen_tokens.numel():
        sc = torch.gather(logits, 1, gen_tokens)
        logits.scatter_(1, gen_tokens, torch.where(sc < 0, sc * rp, sc / rp))
    for b, bl in enumerate(blockers):
        bad = bl.banned()
        if bad:
            logits[b, list(bad)] = -float("inf")
    T = GEN["temperature"]
    if not T:
        return logits.argmax(-1)
    logits = logits / T
    k = GEN.get("top_k") or 0
    if k:
        kth = torch.topk(logits, k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, -float("inf"))
    p = GEN.get("top_p") or 1.0
    probs = F.softmax(logits, -1)
    if p < 1.0:
        sp, si = probs.sort(-1, descending=True)
        sp[(sp.cumsum(-1) - sp) > p] = 0.0
        probs = torch.zeros_like(probs).scatter_(1, si, sp)
    probs = probs / probs.sum(-1, keepdim=True)
    return torch.multinomial(probs, 1).squeeze(1)


def clean(text):
    return re.sub(r"\s+", " ", text).strip()


@torch.inference_mode()
def generate_batch(prompts_ids):
    B, L = len(prompts_ids), len(prompts_ids[0])
    ids = torch.tensor(prompts_ids, dtype=torch.long, device=DEV)
    plen = torch.full((B,), L, dtype=torch.long, device=DEV)
    blockers = [NgramBlocker(GEN.get("no_repeat_ngram") or 0, p, GEN.get("ngram_scope", "generated"))
                for p in prompts_ids]
    gen = torch.empty((B, 0), dtype=torch.long, device=DEV)
    done = torch.zeros(B, dtype=torch.bool, device=DEV)
    with torch.autocast("cuda", dtype=torch.float16):
        if HAS_CACHE:
            logits, cache = model.prefill(ids, plen)
        else:
            logits = model(ids, plen)[:, -1]
    for step in range(MAX_NEW):
        nxt = pick_next(logits.float(), gen, blockers)
        gen = torch.cat([gen, nxt[:, None]], 1)
        nl = nxt.tolist()
        for b, bl in enumerate(blockers):
            bl.push(nl[b])
        if (step + 1) % 16 == 0:
            for b in range(B):
                if not done[b] and len(tok.decode(gen[b].tolist()).split()) > TARGET_WORDS:
                    done[b] = True
            if bool(done.all()):
                break
        if step == MAX_NEW - 1:
            break
        with torch.autocast("cuda", dtype=torch.float16):
            if HAS_CACHE:
                logits, cache = model.decode_step(nxt[:, None], cache)
            else:
                logits = model(torch.cat([ids, gen], 1), plen)[:, -1]
    return [clean(tok.decode(g.tolist())) for g in gen]


# ----------------------------------------------------------------------------- run
data = json.load(open(IN_PATH, encoding="utf-8"))
all_ids = list(data.keys())
if RUN.get("smoke_test"):
    all_ids = all_ids[:16]
mine = all_ids[RANK::WORLD]
enc = {k: tok.encode(data[k]["text"]).ids[-PROMPT_MAX:] for k in mine}
order = sorted(mine, key=lambda k: len(enc[k]))
BS = int(GEN["batch_size"])
out, dropped = {}, 0
t0 = time.time()
for i in range(0, len(order), BS):
    keys = order[i:i + BS]
    m = min(len(enc[k]) for k in keys)
    dropped += sum(len(enc[k]) - m for k in keys)
    texts = generate_batch([enc[k][-m:] for k in keys])
    for k, t in zip(keys, texts):
        out[k] = t
    if RANK == 0 and (i // BS) % 5 == 0:
        el = time.time() - t0
        print(f"  rank0 {i+len(keys)}/{len(order)} prompts | {el/60:.1f} min | "
              f"eta {el/(i+len(keys))*(len(order)-i-len(keys))/60:.1f} min", flush=True)

part = f"{OUT_PREFIX}.part{RANK}.json"
json.dump(out, open(part, "w", encoding="utf-8"), ensure_ascii=False)
log(f"Context tokens trimmed for batching (rank0): {dropped} total")
if WORLD > 1:
    dist.barrier()

if RANK == 0:
    merged = {}
    for pf in sorted(glob.glob(f"{OUT_PREFIX}.part*.json")):
        merged.update(json.load(open(pf, encoding="utf-8")))
        os.remove(pf)
    preds, short = {}, 0
    for k in all_ids:
        w = merged.get(k, "").split()
        if len(w) < TARGET_WORDS:
            short += 1
        preds[k] = {"text": " ".join(w[:TARGET_WORDS])}
    pred_path = f"{OUT_PREFIX}_prediction.json"
    json.dump(preds, open(pred_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    zip_path = f"{OUT_PREFIX}_submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(pred_path, arcname="prediction.json")
    print(f"Wrote {pred_path} and {zip_path} | {len(preds)} ids | {short} under {TARGET_WORDS} words | "
          f"{(time.time()-t0)/60:.1f} min", flush=True)
if WORLD > 1:
    dist.destroy_process_group()