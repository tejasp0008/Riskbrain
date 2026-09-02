"""FastAPI gateway exposing /score (inline, forward pass only, sub-100ms
target) and /dispute (async agent, generation allowed). These two paths
must never share a latency budget or a code path.

OPEN DECISION: the quantization format backing /score (bitsandbytes
4-bit vs fp16 vs AWQ) is not settled until Phase 4 benchmarks each on a
real GPU. See serving/vllm_server.py.
"""
