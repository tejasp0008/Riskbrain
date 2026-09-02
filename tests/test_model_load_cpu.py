"""CPU-only plumbing smoke test for Phase 0.

This does NOT use vLLM or bitsandbytes quantization (both require CUDA).
It only confirms the model ID resolves, downloads, and produces a token
via plain transformers on CPU. It is slow (expect one to several minutes
for a few tokens on a laptop CPU) and is not representative of inline
latency or of the quantized serving path. Run manually, not in CI:

    python tests/test_model_load_cpu.py
"""

def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "Qwen/Qwen3-4B-Instruct-2507"
    print(f"Loading {model_id} on CPU (fp32, no quantization)...")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32, device_map="cpu"
    )

    prompt = "In one sentence, what is a chargeback?"
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=32)
    print(tokenizer.decode(output[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
