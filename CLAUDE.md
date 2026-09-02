# CLAUDE.md: RiskBrain Build Instructions

This file is persistent context for Claude Code. Read it fully before doing anything. Follow the build order. Do not skip ahead. Confirm each phase runs before starting the next.

## What we are building

RiskBrain: a hybrid, inline risk engine for payment disputes and fraud, targeting the Razorpay AI Buildathon (AI Risk Manager track). Two surfaces:

1. **Inline scoring path (real-time, sub-100ms):** a `/score` endpoint that fuses a tabular fraud score with fast semantic features from a self-served small model. Short forward pass only, NO text generation in this path.
2. **Dispute reasoning agent (async):** a multi-step agent that classifies a dispute, retrieves policy and precedent, decides fight/accept/escalate, and drafts a representment package. Generation lives here, where latency does not matter.

The core thesis is latency and cost at transaction scale, not privacy. A self-hosted inline model is the only option with zero network hop and zero per-call cost, which is why it beats any hosted model in the auth path.

## Critical design rules (do not violate)

- **Never conflate the two paths.** Inline scoring is a forward pass with short/classification output. Rebuttal generation is async. Keep them in separate modules with separate latency budgets. The demo and README must state this split explicitly.
- **The system is hybrid by design.** A gradient-boosted tabular model (XGBoost or LightGBM) owns the numeric fraud score. The language model owns reasoning over unstructured text. A fusion layer combines them. Do NOT make the LLM do tabular fraud scoring.
- **Synthetic data proves systems, not accuracy.** All accuracy numbers are directional. The real, defensible metrics are latency, cost per decision, throughput, calibration mechanics, and routing accuracy. Label them honestly in output and README.
- **Use commodity infra for commodity parts.** Use vLLM for serving, PEFT for training, FAISS for retrieval. Do NOT hand-roll a serving engine or write a custom CUDA/Triton kernel. Those are out of scope for this build.
- **Ground synthetic disputes in real reason codes.** Use actual Visa and Mastercard reason-code categories (fraud, authorization, processing errors, consumer disputes) with representment deadlines and per-code evidence rules. No generic "customer unhappy" data.

## Version policy

Do not assume versions from memory. At the start of Phase 0, check the current stable versions of vllm, transformers, peft, bitsandbytes, xgboost, faiss-cpu, sentence-transformers, and fastapi (pip index versions or the package pages), pin them in requirements.txt, and note the chosen model ID for the current small Qwen3 instruct release. Confirm the model license before committing to it.

## Tech stack

- Language model: current small Qwen3 instruct release (target ~4B), 4-bit via bitsandbytes.
- Tabular model: XGBoost or LightGBM.
- Fine-tuning: PEFT + LoRA (QLoRA, rank 8 to 16). Unsloth or Axolotl optional for speed.
- Serving: vLLM with multi-LoRA.
- Retrieval: FAISS (IVF-PQ or HNSW) + sentence-transformers, with an INT8 quantization pass on the embedding matrix.
- Calibration: scikit-learn isotonic or Platt scaling.
- Fusion: a small logistic or gradient model over [tabular_score, semantic_features].
- Gateway: FastAPI.
- Agent orchestration: LangGraph, or a plain Python state machine (prefer the simpler one that is easy to explain).
- Observability: structured logging to SQLite plus a simple metrics dashboard. Prometheus/Grafana optional.
- Synthetic data: generation scripts using an available frontier API. Keep the scripts in the repo.
- Payments: Razorpay Disputes/Payments sandbox if accessible; otherwise a clean simulated webhook module with the same interface, clearly marked as simulated.
- Frontend (optional, last): a minimal dashboard whose centerpiece is a live side-by-side latency comparison (inline score vs a hosted-API call).

## Repository structure

```
riskbrain/
  README.md
  requirements.txt
  CLAUDE.md
  data/
    generate.py            # synthetic data generator, grounded in reason codes
    schema.py              # dispute/transaction schemas
    corpus/                # RAG grounding docs (reason-code playbooks, evidence rules)
  tabular/
    train.py               # XGBoost/LightGBM training
    features.py            # structured feature extraction
  llm/
    quantize.py            # 4-bit load config
    adapters/              # LoRA training scripts + trained adapters
    train_adapter.py
  retrieval/
    index.py               # FAISS build with INT8 embeddings
    embed.py
    query.py
  serving/
    vllm_server.py         # vLLM multi-LoRA launch
    gateway.py             # FastAPI: /score (inline) and /dispute (async agent)
  fusion/
    fuse.py                # combine tabular + semantic features
    calibrate.py           # calibration + reliability diagram
  agent/
    loop.py                # classify -> retrieve -> decide -> act -> log
    router.py              # adapter routing
    escalation.py          # confidence-threshold human-in-the-loop
  eval/
    baselines.py           # rules-only and hosted-API-only baselines
    harness.py             # runs comparison, emits latency/cost/calibration
    adversarial.py         # fabricated-narrative robustness set
  observability/
    logging.py
    dashboard.py
  tests/
```

## Build order (one phase at a time, verify before proceeding)

### Phase 0: Setup and base serving
- Create the repo structure, requirements.txt with pinned current versions, README stub.
- Load the small Qwen3 model in 4-bit, serve via vLLM, expose a health check, confirm an HTTP request returns a completion.
- STOP and confirm this runs before continuing.

### Phase 1: Tabular model + fusion skeleton
- Build data/schema.py and a minimal synthetic structured dataset.
- Train the XGBoost fraud scorer in tabular/train.py.
- Build fusion/fuse.py and fusion/calibrate.py so a calibrated numeric score comes out. Emit a reliability diagram.
- STOP and confirm a calibrated score is produced.

### Phase 2: Retrieval grounding
- Build the reason-code-grounded corpus in data/generate.py and data/corpus/.
- Build retrieval/embed.py with an INT8 embedding quantization pass, retrieval/index.py (FAISS), retrieval/query.py.
- Wire naive RAG: retrieve then generate.
- STOP and confirm grounded answers.

### Phase 3: Adapters + routing
- Generate per-domain training data (classification, rebuttal drafting, recovery messaging).
- Train three LoRA adapters (QLoRA) in llm/.
- Wire multi-LoRA serving and agent/router.py; log routing decisions.
- STOP and confirm the model specializes and routes.

### Phase 4: Agent loop + inline endpoint + payments
- Build agent/loop.py (the five steps) and agent/escalation.py.
- Build serving/gateway.py with TWO endpoints: /score (inline, forward pass only, target sub-100ms) and /dispute (async, full agent with generation).
- Integrate the Razorpay sandbox or the simulated webhook module.
- STOP and confirm the full end-to-end system, with the inline path measurably fast.

### Phase 5: Eval + observability
- Build eval/baselines.py (rules-only, hosted-API-only) and eval/harness.py.
- Capture latency, cost per decision, calibration curve, routing accuracy. Label systems metrics vs directional accuracy metrics.
- Build observability/dashboard.py with the side-by-side latency centerpiece.
- STOP and confirm real numbers are produced.

### Phase 6: Adversarial + polish
- Build eval/adversarial.py with fabricated-narrative disputes; show the agent resists them.
- Write the full README: thesis, architecture, the inline-vs-async split, what was built vs commodity, honest metric boundaries, and how to run.

## Coding conventions

- Python 3.11+. Type hints on public functions. Docstrings that state intent, not mechanics.
- Every module runnable and testable in isolation. Add a smoke test per phase in tests/.
- No em dashes in generated text, comments, or docs.
- Keep functions small and named for what they do. Prefer clarity over cleverness.
- Log every decision with: inputs, tabular score, semantic features, fused score, adapter used, confidence, action, latency. This log is what produces the eval numbers later.
- Mark any simulated component (e.g. a stand-in for the Razorpay API) explicitly in code and README.

## README must include (this is what the panel reads)

- The thesis in one paragraph: inline latency and cost at scale, not privacy.
- The inline-vs-async split, stated explicitly.
- The hybrid rationale: tabular for numbers, LLM for text.
- A "what I built vs what is commodity" section.
- Honest metric boundaries: which numbers are real (systems) and which are directional (synthetic accuracy).
- Run instructions.
