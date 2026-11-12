# CS781 Capstone — Indian Legal NLP from a from-scratch language model

Kartik Agrawal (240526) · Codabench `k_240526` · IIT Kanpur, CS781 (2026)

Repository: https://github.com/Luciferk07/cs781-capstone

Everything here starts from language models trained **from scratch** on the competition corpus (no pretrained weights).
Phase 1 trains four backbones that share one 32k byte-level BPE tokenizer; Phases 2–7 adapt them to six legal tasks.
Model weights and checkpoints are **not** included (course rule); every folder has the code, the notebook that ran it,
and for Phases 2–7 each run's `config.json`, `train.log` and `*_result.json`.

| Phase | Task | Metric | Best Codabench score | Best submission |
|---|---|---|---|---|
| 1 | Legal LM pretraining (128-word continuation of a 512-word prompt) | BERTScore F1 | 0.8347 | retention model (`arch4`) |
| 2 | Legal NER (12 entity types, character spans) | strict macro-F1 | 0.5126 | `prefixlm_lorma_viterbi` |
| 3 | Rhetorical roles (13 roles, CL + IT) | macro-F1 | 0.4941 | `r1_lorma_prefixlm` |
| 4 | Judgment prediction (CJPE) | macro-F1 | 0.6224 | `r6_kd_student` |
| 5 | Statute identification (100 statutes) | macro-F1 | 0.3582 | `r8_full_prefixlm` |
| 6 | Abstractive summarization | BERTScore F1 | 0.8333 | `r2_lora_prefixlm` |
| 7 | Prior case retrieval | Micro-F1@7 | 0.2593 | `p1_full` |

The full report (architecture, data analysis, all runs, error analysis) is `report/CS781-CP-Kartik-Agrawal-240526-report.pdf`
in the submission zip.

## Repository layout

```
phase1/  prefixlm/ encdec/ arch4/ biscale/   Phase-1 models: model.py, train.py, generate.py, pretokenize.py, notebook
phase2/  ner_*.py      L-NER  (BIO tagging, Viterbi decoding)
phase3/  rr_*.py       rhetorical roles (sentence classification with context)
phase4/  cj_*.py       judgment prediction (chunked long-document classifier)
phase5/  lsi_*.py + cj_model.py   statute identification (multi-label)
phase6/  sum_*.py      abstractive summarization (generation from each backbone)
phase7/  pcr_*.py      prior case retrieval (bi-encoder)
```

Common files in Phases 2–7:

| File | Role |
|---|---|
| `phase1_loader.py`, `p1_biscale.py` | load the four Phase-1 checkpoints (the Bi-Scale classes are copied unchanged from its training notebook) |
| `<task>_data.py` + `test_<task>_data.py` | data loading, splits, task metric, prediction validator (+ unit tests) |
| `<task>_train.py` | training harness: one run = one `config.json`; trains, keeps the best epoch, evaluates on dev, writes `prediction.json` and the submission zip |
| `<task>_model.py` (`ner_model.py`, `rr_model.py`, `cj_model.py`, `sum_model.py`, `pcr_model.py`) | the task model: loads a Phase-1 backbone and applies one adaptation method |
| `*_dummy.py` | tiny model used only by the notebook's smoke test |
| `phase*_run.ipynb` | the Kaggle notebook that ran every experiment of the phase (the plan cell lists each run's settings) |
| `runs/<run>/` | `config.json`, `train.log`, `<run>_result.json` (per-epoch history, dev scores) |

## Phase 1 — four foundation models

| Model | Params | Layers | d | Heads | FFN | Notes |
|---|---|---|---|---|---|---|
| Prefix-LM | 109.6M | 12 | 768 | 12 | 3072 | RoPE, tied embeddings, bidirectional random-length prefix (30–80% of the window), causal continuation |
| Encoder–decoder | 114.4M | 6 + 5 | 768 | 12 | 3072 | RoPE, tied embeddings |
| Retention (`arch4`) | 109.6M | 9 | 768 | 6 | 3072 | multi-scale retention, value factor 2, recurrent decoding |
| Bi-Scale | 144.2M | 16 | 768 | 12 q / 4 kv (GQA) | 2048 SwiGLU | RMSNorm, local 256-token attention in layers 0–7; two stages: 71M Wikipedia tokens (allowed by the instructor), then 283M legal tokens |

Training (prefix-LM, encoder–decoder, retention): data mix Supreme Court 40% / High Courts 30% / Acts 20% / Tribunals 10%,
context 1,024, 131,072 tokens per step, AdamW (β 0.9/0.95, wd 0.1), LR 3e-4 → 3e-5 cosine with 1,000 warmup steps, fp16, 2× T4.

Run (Kaggle, GPU T4 ×2, Internet on, `HF_TOKEN` secret): open `phase1/<model>/<model>_phase1_run.ipynb` and *Run All*.
The notebook tokenizes the corpus (`pretokenize.py`), trains (`train.py`, resumes from a checkpoint if one is attached),
evaluates, chooses decoding settings and writes the submission (`generate.py`).

## Phases 2–7 — how to run

1. Kaggle notebook, GPU T4 ×2, Internet on, secret `HF_TOKEN`.
2. Add the inputs: the Phase-1 checkpoints and the tokenizer (`vocab.json`, `merges.txt`).
3. Open `phaseN/phaseN_*_run.ipynb` and *Run All* (or *Save & Run All*). The notebook downloads the data from
   `Exploration-Lab/CS-781-Capstone`, runs the unit tests, a data check and a smoke test with a dummy model, checks every
   planned model, then trains all runs in two GPU queues and prints a results table with download links.
4. Each run writes `p*runs/<run>/<run>_submission.zip` containing only `prediction.json`.

A single run can also be started directly: `python <task>_train.py runs/<run>/config.json` (paths in the config point to Kaggle).

## Phases 2–7 — methods

Every run = one Phase-1 backbone + one adaptation method + a task head.
Methods implemented in the model files: full fine-tuning, partial fine-tuning (top-k blocks), BitFit, LoRA, rsLoRA,
LoRMA (multiplicative, `W' = (I + BA) W`), Houlsby and Pfeiffer adapters, prefix tuning, prompt tuning, and knowledge
distillation (a student keeping evenly spaced layers; logit KD with T = 2, α = 0.5 for Phases 2–5; sequence-level KD for
Phase 6; similarity-distribution KD for Phase 7).

| Phase | Task head and decoding | Input handling |
|---|---|---|
| 2 NER | linear BIO tagger; greedy or Viterbi decoding; tags mapped back to character spans | 512-token windows, stride 384 |
| 3 RR | sentence classifier from the pooled sentence inside its context | 512-token context window |
| 4 CJPE | chunk pooling (mean / mean+last) → GRU → binary head; threshold tuned on dev; class-weighted loss | last 4 chunks of 512, stride 412 |
| 5 LSI | attention pooling over chunks → 100 sigmoid outputs; balanced positive weights; per-statute thresholds tuned on dev | first+last chunks, 4 × 512 |
| 6 Summ. | the backbone generates the summary (encoder–decoder decodes; other models continue the source prefix); greedy, repetition penalty 1.2, no repeated 4-grams | head (60%) + tail (40%) of the source, 1,019 or 520 tokens |
| 7 PCR | bi-encoder: pooled chunk vectors → max over chunks → cosine ranking; InfoNCE (τ 0.05) with BM25-mined, random and in-batch negatives | 4 × 512 tokens per case (2 head, 2 tail) |

All runs of each phase, their settings, local dev scores and Codabench scores are tabulated in the report.

## Requirements

Python 3.10+, PyTorch 2.x (CUDA), `tokenizers`, `huggingface_hub`, `numpy`, `scipy` (Phase 7 BM25), `pandas`, `matplotlib`.
Phase-6 local evaluation optionally uses `bert-score`.

## LLM use

see the LLM declaration in the report.
