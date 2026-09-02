"""Loads Qwen3-4B-Instruct-2507 in 4-bit via bitsandbytes and serves it
through vLLM behind a minimal FastAPI app.

This is Phase 0 only: base serving plus a health check and a single
/generate endpoint to confirm the model produces completions. It is NOT
the inline /score path (that is a forward pass with no generation, built
in Phase 4) and it is NOT the multi-LoRA dispute-agent server (Phase 3).
Keep this file simple; it exists to prove the base model loads and serves.

Requires a CUDA GPU. vLLM's model executor and bitsandbytes 4-bit
quantization both require CUDA; there is no CPU path for this script.
If you are on a machine without a GPU, do not run this file directly,
use it only to prepare code to run on the rented GPU box. See
tests/test_model_load_cpu.py for a CPU-only plumbing smoke test that
checks tokenizer/model download and a raw forward pass without vLLM or
quantization (slow, not representative of inline latency).

OPEN DECISION: the inline path's quantization format (bitsandbytes 4-bit
vs fp16 vs AWQ) is not settled. 4-bit here is a Phase 0 starting point to
prove the load-and-serve path works, not a claim that it is the fastest
option for the sub-100ms inline path. Phase 4 must benchmark 4-bit vs
fp16 vs AWQ latency on a real (non-T4) GPU before the inline scoring
path commits to one.
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("riskbrain.vllm_server")

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"

_engine = None
_sampling_params_cls = None


def _check_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU detected. vLLM and bitsandbytes 4-bit quantization "
            "require a CUDA GPU. Run this on the rented GPU box, not "
            "locally. See tests/test_model_load_cpu.py for a CPU-only "
            "plumbing check that does not use vLLM or quantization."
        )


def load_engine():
    """Loads the base model in 4-bit via vLLM's bitsandbytes load path."""
    from vllm import LLM, SamplingParams

    global _sampling_params_cls
    _sampling_params_cls = SamplingParams

    logger.info("Loading %s in 4-bit (bitsandbytes) via vLLM...", MODEL_ID)
    start = time.perf_counter()
    llm = LLM(
        model=MODEL_ID,
        quantization="bitsandbytes",
        load_format="bitsandbytes",
        dtype="bfloat16",
        max_model_len=8192,
        gpu_memory_utilization=0.85,
        enforce_eager=False,
    )
    logger.info("Model loaded in %.1fs", time.perf_counter() - start)
    return llm


@asynccontextmanager
async def lifespan(app: FastAPI):
    _check_cuda()
    global _engine
    _engine = load_engine()
    yield
    _engine = None


app = FastAPI(title="RiskBrain Phase 0 base server", lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.7


class GenerateResponse(BaseModel):
    text: str
    latency_ms: float


@app.get("/health")
def health():
    if _engine is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ok", "model": MODEL_ID}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if _engine is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    sampling_params = _sampling_params_cls(
        temperature=req.temperature,
        max_tokens=req.max_tokens,
    )
    start = time.perf_counter()
    outputs = _engine.generate([req.prompt], sampling_params)
    latency_ms = (time.perf_counter() - start) * 1000

    text = outputs[0].outputs[0].text
    return GenerateResponse(text=text, latency_ms=latency_ms)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
