# RiskBrain

A hybrid, inline risk engine for payment disputes and fraud, built for the
Razorpay AI Buildathon (AI Risk Manager track).

Status: Phase 0 and Phase 1 complete. See CLAUDE.md for the full build
order and design rules. This README will be filled out fully in Phase 6
with the thesis, architecture, inline-vs-async split, hybrid rationale,
commodity-vs-built breakdown, and honest metric boundaries.

## Phase 0: base model serving

Model: `Qwen/Qwen3-4B-Instruct-2507` (Apache-2.0, dense, text-only, 4B
params), loaded in 4-bit via bitsandbytes, served through vLLM.

`serving/vllm_server.py` is the real FastAPI deployment path for later
phases and requires a CUDA GPU (there is no CPU fallback for vLLM +
bitsandbytes). For a machine without a GPU, `tests/test_model_load_cpu.py`
is a slow, CPU-only plumbing check (plain transformers, no quantization)
that only confirms the model resolves and produces a token.

`notebooks/colab_runbook.ipynb` is the Google Colab path: runtime check,
Drive mount with the HuggingFace cache pointed at Drive, dependency
install (working around Colab's preinstalled torch), a direct vLLM 4-bit
load-and-generate verification cell, and Phase 1 below. Colab's T4 GPU
verification is a functional check only, not a representative inline
latency number. The quantization format for the real inline path
(bitsandbytes 4-bit vs fp16 vs AWQ) is an open decision, to be settled by
latency benchmarking on a real (non-T4) GPU in Phase 4.

## Phase 1: tabular model + fusion skeleton

`data/schema.py` defines `Transaction` and `Dispute` schemas, grounded in
real Visa/Mastercard reason codes (fraud, authorization, processing
error, consumer dispute categories) with representment deadlines and
per-code evidence requirements.

`data/generate.py` generates a synthetic, labeled transaction dataset
(amount, velocity, device novelty, geo mismatch, card-present, hour of
day, MCC risk bucket) with a hand-picked, directionally-plausible risk
function. This is architectural scaffolding, not a fraud model: it
exists to prove the tabular pipeline runs end to end. Any accuracy
numbers computed on it are directional only.

`tabular/train.py` trains an XGBoost classifier on that data and saves
the model. `fusion/fuse.py` and `fusion/calibrate.py` produce a
calibrated probability (isotonic regression) and save a reliability
diagram. The fusion model is a small logistic regression over
`[tabular_score, *semantic_features]`; semantic features are a stub
(zero-width) until the served model and retrieval exist in Phase 2-3, so
fusion currently degenerates to a monotonic transform of the tabular
score alone.

`tests/test_phase1_smoke.py` runs the whole Phase 1 path end to end.
Artifacts (model, validation scores, reliability diagram) are written to
`$RISKBRAIN_ARTIFACTS_DIR` (defaults to `./artifacts`; set to a Drive
path in Colab for persistence).

## Environments

- **Local, no GPU:** repo structure, Phase 1 (pure CPU), and syntax
  checks all work. Phase 0/serving does not (no CUDA).
- **Rented GPU box / Colab:** everything, including `serving/
  vllm_server.py` and the Phase 0 vLLM verification.
- `requirements.txt` pins `xgboost==3.4.1`, which requires Python
  >=3.12. On an older Python (Colab's runtime varies), the notebook
  auto-detects the interpreter version and overrides to the newest
  compatible xgboost (3.2.0 for 3.10/3.11, 2.1.4 for 3.9 and below) for
  that session only, and prints which override was applied.
