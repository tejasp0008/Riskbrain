# RiskBrain: A Hybrid, Inline Risk Engine for Payment Disputes and Fraud

**Target:** Razorpay AI Buildathon 2026 (AI Risk Manager track, Revenue Recovery as a secondary surface)
**Author:** Tejas P
**One-line thesis:** Fraud and dispute decisions happen inline, in milliseconds, on millions of transactions. You physically cannot put a frontier API in that path. You build the thing that fits: a hybrid engine where a gradient-boosted model owns the numeric fraud score and a small, quantized, self-served model owns the reasoning over unstructured dispute text, fused into one sub-100ms decision. Every frontier-API team in the room is disqualified from this problem by the laws of latency and cost. You are the only one who can solve it.

---

## 0. Read This First: The Positioning That Makes It Win

The naive version of this project leads with privacy ("data stays in your VPC"). Do not lead with that. A cloud-native fintech using Bedrock or Vertex with a no-training data agreement already keeps data in its VPC and gets a frontier-quality model, so privacy alone loses the argument.

Lead instead with the constraint no frontier API can beat: **inline latency and cost at transaction scale.** Payment authorization decisions run in tens of milliseconds across millions of transactions. A frontier API call is hundreds of milliseconds to seconds, plus per-call cost that is absurd at that volume. A tiny fine-tuned model, quantized and served locally, is the only architecture that can run inline at auth time. That is not a preference, it is physics and economics, and it is exactly the world Razorpay's risk team lives in.

Privacy then becomes a strong secondary bullet, not the headline.

---

## 1. The Idea

Razorpay's AI Risk Manager handles chargebacks and disputes: it verifies proof, assembles evidence, submits representment, and analyzes return-to-origin (RTO) patterns to flag preventable returns. Every student in the room will rebuild a version of this by calling a frontier API.

RiskBrain is different in two ways. First, it is **hybrid**: it uses the right tool for each signal instead of forcing an LLM to do everything. Second, its language reasoning core is a small model you have quantized to 4-bit, fine-tuned with domain LoRA adapters, and served yourself, so it can run inline where a frontier API cannot.

The system does two jobs:
- **Inline risk scoring** on incoming transactions and disputes, in sub-100ms, fusing a tabular fraud score with semantic features from the served model.
- **Dispute reasoning and evidence assembly**: a multi-step agent that classifies a dispute, retrieves policy and precedent, decides fight/accept/escalate, and drafts a submission-ready representment package.

---

## 2. The Hybrid Architecture Is Non-Negotiable

This is the single most important design decision, and the one a risk engineer will attack first.

Fraud detection on structured transaction features (amount, velocity, device fingerprint, geo, BIN, time-of-day) is owned by gradient-boosted trees. XGBoost or LightGBM beats an LLM on tabular fraud, trains in minutes, and costs almost nothing to serve. Do not claim your LLM does this. You will lose.

So the architecture splits by signal type:

- **Tabular model (XGBoost / LightGBM):** owns the numeric fraud score from structured transaction features. Fast, calibrated, cheap.
- **Served small model (Qwen3-4B, 4-bit, LoRA adapters):** owns what tabular models cannot touch. Reasoning over unstructured dispute narratives, reading policy, assembling evidence, drafting rebuttals, extracting semantic risk signals from free text (merchant descriptions, customer complaint text, chat transcripts).
- **Fusion layer:** combines the tabular score with the model's semantic features into one calibrated decision.

The one-sentence defense: "Numbers are a tabular problem, text and reasoning are a language-model problem, and I fused them instead of pretending an LLM is good at both."

---

## 3. What It Does (User-Facing Behavior)

**Surface A: Inline scoring endpoint.**
A `/score` endpoint takes a transaction or dispute and returns a calibrated risk probability in sub-100ms. This is the piece that is structurally impossible for a frontier-API team, and it is the heart of the pitch. Benchmark it head-to-head against a frontier API on the same inputs and show the latency and cost gap.

**Surface B: Dispute reasoning agent.**
A dispute or chargeback event arrives (Razorpay Disputes API sandbox or a simulated webhook). The agent runs five steps:
1. **Classify** the dispute against its network reason code (Visa/Mastercard taxonomy) and transaction metadata.
2. **Retrieve** the reason-code playbook, evidence requirements, similar past disputes and their outcomes, and merchant history.
3. **Decide** fight / accept / request-more-info, with a win-probability estimate grounded in retrieved precedent.
4. **Act:** if fighting, assemble the representment package (rebuttal narrative plus itemized proof), formatted for the Disputes API. On the recovery side, draft a dunning message and smart retry schedule for a failed subscription payment.
5. **Log** the full reasoning trace, the adapter routed to at each step, and a confidence score, for audit.

**Surface C: Confidence-thresholded escalation.**
High-confidence decisions auto-act. Low-confidence ones route to a human review queue. This shows risk and product maturity and answers the inevitable "what happens when it is wrong" question before it is asked.

---

## 4. Architecture Diagram

```
   Transaction / dispute event
   (Razorpay API webhook or simulated)
              |
              v
   +--------------------------+
   |   Feature Extraction     |
   |  structured  |  text     |
   +--------------------------+
        |                |
        v                v
  +-----------+   +--------------------+
  | Tabular   |   |  Served Model      |  Qwen3-4B (4-bit)
  | XGBoost / |   |  vLLM + multi-LoRA |  resident base +
  | LightGBM  |   |  adapters A / B / C|  domain adapters
  | numeric   |   |  semantic features |
  | fraud     |   +--------------------+
  | score     |            ^
  +-----------+            | grounding context
        |                  |
        |          +--------------------+
        |          |  Retrieval Layer   |  FAISS / Qdrant
        |          |  quantized embeds  |  reason-code playbooks,
        |          |  policy, precedent |  evidence rules, merchant hist
        |          +--------------------+
        v                  v
   +--------------------------------+
   |        Fusion + Calibration    |  calibrated risk probability
   +--------------------------------+
              |
              v
   +--------------------------------+
   |   Agent Orchestrator           |  classify -> retrieve -> decide
   |   (LangGraph / state machine)  |  -> act -> log
   +--------------------------------+
        |                    |
        v                    v
   confidence >= T?     confidence < T?
   auto-act             human review queue
        |
        v
   Representment package -> Razorpay Disputes API
        |
        v
   +--------------------------------+
   |   Observability + Eval         |  latency, cost-vs-frontier,
   |   Prometheus / Grafana         |  calibration curve, routing acc,
   |                                |  baseline comparison
   +--------------------------------+
```

---

## 5. Tech Stack

| Layer | Tool | Why this choice |
|---|---|---|
| Tabular fraud model | XGBoost or LightGBM | Owns numeric fraud scoring; the correct tool, fast and calibrated. |
| Base language model | Qwen3-4B-Instruct | Widest 2026 ecosystem for quantization and LoRA, multilingual, small enough to serve inline, big enough for meaningful adapters. |
| Quantization | bitsandbytes 4-bit (or AWQ/GPTQ) | Standard 2026 QLoRA recipe; reuses your INT4 background. |
| Fine-tuning | PEFT + LoRA via Unsloth or Axolotl | One-command QLoRA, single-GPU, rank 8 to 16. |
| Serving | vLLM (multi-LoRA) | Continuous batching and paged KV cache out of the box. Do not hand-roll under time pressure. |
| Calibration | scikit-learn (isotonic / Platt) | Turns raw scores into calibrated probabilities; risk teams care about this deeply. |
| Fusion | Small logistic/gradient model over [tabular score, semantic features] | The piece you actually built; keep it simple and defensible. |
| Gateway | FastAPI | You already use it; thin routing over the serving engine. |
| Agent orchestration | LangGraph or plain Python state machine | Legible multi-step control; a hand-rolled machine is fine and more transparent in the pitch. |
| Retrieval | FAISS (IVF-PQ / HNSW) or Qdrant | Fast ANN; FAISS lets you show quantized-embedding depth rather than a wrapper. |
| Embeddings | sentence-transformers + your own INT8 quantization pass | Ties retrieval to your compression wedge. |
| Synthetic data | Frontier model (Claude / GPT via provided stack) | Generate disputes grounded in real reason-code taxonomies; 500 to 2000 curated examples per adapter. |
| Cloud | RunPod or Lambda Labs GPU spot + Docker | Affordable GPU time; Docker you already run. Two nodes if you want a real distributed claim. |
| Observability | Prometheus + Grafana (or structured logs + dashboard) | Produces the benchmark numbers that make the project defensible. |
| Payments integration | Razorpay Disputes / Payments API (sandbox) | A real working system against their stack, not a mockup. |
| Frontend (optional) | React / Next.js dashboard | Shows the agent working live during the pitch. |
| GPU-opt companion (stretch) | Custom Triton INT4 kernel | Standalone "I wrote my own kernel" artifact; benchmark vs naive path. Roadmap, not a build target. |

---

## 6. Data (Grounded, Not Generic)

Generate a synthetic corpus with a frontier model, but ground it in the real chargeback world so it survives a domain expert:

- **Reason-code realism:** structure disputes around actual Visa and Mastercard reason-code categories (fraud, authorization, processing errors, consumer disputes) with correct representment deadlines and per-code evidence requirements. Generic "customer unhappy" text gets spotted instantly.
- **Dispute events:** a few hundred synthetic disputes across reason codes, each with transaction metadata, merchant context, and a ground-truth correct action.
- **Policy corpus:** reason-code playbooks and evidence rules (the RAG grounding documents).
- **Precedent set:** past disputes with won/lost outcomes so the win-probability step has something to retrieve.
- **Adversarial set:** disputes with fabricated or manipulative narratives, because real disputes include people gaming the system. Your agent should resist these (see feature 5 below).
- **Per-adapter training sets:** 500 to 2000 curated examples each. Quality beats quantity; hand-check the generated examples.

Keep the generation scripts in the repo. "I built the dataset generator, grounded in real reason codes" is a signal most candidates skip.

---

## 7. The Features That Raise Selection Odds (Priority Ordered)

1. **Inline scoring endpoint (highest impact).** The sub-100ms `/score` path with the frontier-API latency and cost benchmark. This is what makes the project unbeatable rather than merely nice.
2. **Calibrated probabilities, not vibes.** Output an actual probability with a reliability diagram. A score of 0.8 must mean 80%. Frontier models hand you text; you hand them calibration. This signals risk-engineer thinking.
3. **Baseline evaluation harness (your unfair advantage).** Compare RiskBrain against a rules-only baseline and a frontier-API-only baseline on the same set. Show you match on decision quality while winning decisively on latency and cost. You already do paired-model comparison in your DataAnnotation work, so this is home turf.
4. **Confidence-thresholded human-in-the-loop escalation.** Auto-act on high confidence, escalate low. Shows product and risk maturity.
5. **Adversarial robustness on dispute narratives (your sleeper edge).** Show the agent resists fabricated or manipulative dispute text. This ties directly to your DigitalXplore prompt-injection and OWASP LLM Top 10 work, so you can defend it more deeply than anyone in the room.

Optional if time allows: an RTO (return-to-origin) prediction model as a second purely predictive piece, hitting the "predictive modelling" line. Cut it without guilt if scope tightens.

Do not build: the custom Triton kernel or a hand-rolled serving engine. Both stay as roadmap. Three additions done well beats eight half-finished.

---

## 8. What You Personally Built (Have This Answer Ready)

vLLM, PEFT, and FAISS are commodity infrastructure. When the panel asks what is actually yours, name it without hesitation:
- The tabular-plus-LLM fusion and calibration layer.
- The adapter routing logic.
- The evaluation harness with real baselines.
- The agent control flow and the confidence-escalation policy.
- The grounded synthetic-data generator.
- The quantized-embedding retrieval pass.

Own that the serving engine and trainer are commodity. Pointing precisely at your own non-trivial work reads as honest and senior. Implying you built the commodity parts gets you caught.

---

## 9. Honesty Boundaries (State These Before They Are Raised)

- **Synthetic data proves systems, not accuracy.** Lead with latency, cost, throughput, calibration mechanics, and routing accuracy, which synthetic data genuinely demonstrates. Say out loud that decision-quality numbers are directional only because the data is synthetic. Volunteering this reads as maturity.
- **The 4B model is not GPT-class on reasoning.** That is fine, because it is not competing on raw reasoning, it is competing on inline latency and cost where GPT cannot play. Say so plainly.
- **Calibration on synthetic data is partly circular.** Acknowledge that real calibration needs real outcome labels; show the mechanism works and note what real deployment would require.

---

## 10. Anticipated Panel Objections and Your Answers

**"Why an LLM for fraud when XGBoost wins on tabular?"**
It does not do tabular fraud. The tabular model owns the numeric score. The LLM owns reasoning over unstructured dispute text, which trees cannot read. They are fused.

**"Your numbers are synthetic, so they are meaningless."**
The accuracy numbers are directional, and I say so. The systems numbers (latency, cost, throughput, calibration mechanics) are real and are the point, because the thesis is an architectural and economic one, not an accuracy claim.

**"Why not just use Bedrock with a data agreement for privacy?"**
Privacy is not my headline. Inline latency at transaction scale is. You cannot put a frontier call in a sub-100ms auth path at millions of transactions per day. That constraint is why this architecture exists.

**"Did you build anything, or wire libraries together?"**
The serving engine and trainer are commodity, and I used them deliberately instead of wasting time reinventing them. What I built is the fusion, calibration, routing, eval harness, agent control flow, and grounded data generator.

**"Do you understand real chargebacks?"**
Yes. The corpus is grounded in Visa and Mastercard reason-code categories with representment deadlines and per-code evidence rules, and the agent assembles representment accordingly.

**"Why Qwen3-4B specifically?"**
Widest ecosystem for quantization and LoRA, multilingual for Indic dispute text, small enough to serve inline, large enough for meaningful adapters. A 1.7B underperforms on reasoning; an 8B breaks the inline latency budget.

---

## 11. How It Will Be Done (Phased Build)

Build in phases so you always have a working system to show. Do not build the fancy parts until the boring end-to-end path works.

**Phase 0 (Days 1-2): Base serving.** Qwen3-4B in 4-bit through vLLM on one cloud GPU, reachable over HTTP. Goal: cloud serving works.

**Phase 1 (Days 3-4): Tabular model + fusion skeleton.** Train the XGBoost fraud scorer on synthetic structured data; stand up the fusion and calibration layer. Goal: a calibrated numeric score exists.

**Phase 2 (Days 5-6): Retrieval grounding.** Build the reason-code-grounded corpus, embed with INT8-quantized embeddings, index in FAISS, wire naive RAG. Goal: grounded reasoning.

**Phase 3 (Days 7-9): Adapters + routing.** Train the three LoRA adapters (QLoRA, rank 8-16), wire multi-LoRA serving with a router, log routing. Goal: the model specializes and routes.

**Phase 4 (Days 10-11): Agent loop + inline endpoint + Razorpay API.** Build the five-step state machine, the sub-100ms `/score` endpoint, and the confidence escalation; connect to the Disputes sandbox. Goal: full end-to-end system with the inline path live.

**Phase 5 (Days 12-13): Eval + observability.** Run the baseline comparison (rules-only, frontier-only, RiskBrain), capture latency, cost, calibration curve, routing accuracy. Build the dashboard. Goal: benchmarked, defensible numbers.

**Phase 6 (Day 14): Adversarial hardening + pitch.** Run the adversarial dispute set, tighten the demo, record a fallback, write the eval README. Goal: portfolio-grade finish with a demo that cannot fail live.

---

## 12. Scope Tiers (Never Show a Broken Demo)

**Must-have (this is the submission):**
- Qwen3-4B served in 4-bit on cloud GPU.
- XGBoost tabular scorer + fusion + calibration.
- RAG grounding on the reason-code corpus.
- At least one LoRA adapter.
- The five-step agent loop producing a representment package.
- The inline `/score` endpoint.
- Razorpay sandbox integration.
- A working demo surface.

**Should-have (the differentiators):**
- All three adapters + routing.
- Baseline evaluation harness with the cost-and-latency comparison.
- Calibration curve.
- Confidence escalation.

**Stretch (roadmap if unfinished):**
- Adversarial robustness demo.
- RTO prediction model.
- Second GPU node for a genuine distributed claim.
- Custom Triton INT4 kernel.

Under time pressure, use vLLM multi-LoRA, present the custom engine and kernel as roadmap, and protect the end-to-end path first.

---

## 13. The Pitch (5 Minutes)

1. **The constraint (30s):** Fraud and dispute decisions run inline in milliseconds on millions of transactions. You cannot put a frontier API there. Set the trap the frontier-API teams cannot escape.
2. **The demo (2m):** A live transaction scored inline in sub-100ms, then a dispute flowing through classify, retrieve, decide, act, with the representment package submitted to the sandbox.
3. **The architecture (1m):** The diagram. Emphasize hybrid (tabular for numbers, served LLM for text) and inline serving.
4. **The numbers (1m):** Latency and cost vs frontier, calibration curve, baseline comparison. Say clearly which numbers are real (systems) and which are directional (synthetic accuracy).
5. **The close (30s):** "Everyone here built an agent that calls GPT. I built the engine that runs where GPT structurally cannot: inline, in milliseconds, at transaction scale, with the numbers to prove it."

---

## 14. Risks and Caveats

- **Cost:** Cloud GPU spot time is real money. Budget 50 to 150 USD, kill instances when idle. If too much, run on one rented GPU and simulate distribution with two containers, but only claim "distributed" if you truly run multi-node.
- **Scope creep:** The kernel and hand-rolled engine are time sinks. Roadmap, not build targets.
- **Data realism:** Ground in real reason codes and inject adversarial disputes, or the numbers mean nothing.
- **Demo failure:** Record a fallback and have a local degraded path in case the GPU hiccups live.
- **Defend every choice:** Interns are judged on reasoning and ownership. Know your tradeoffs cold; depth in a few areas beats breadth you cannot defend.
- **Verify the deliverable:** Confirm on the application page whether it is a submitted repo plus pitch or a live build day. Applications close 5 September; lock the format this week.
