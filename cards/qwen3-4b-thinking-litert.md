---
license: apache-2.0
base_model: Qwen/Qwen3-4B-Thinking-2507
tags:
  - litert
  - litert-lm
  - litertlm
  - on-device
  - edge
  - qwen3
  - reasoning
  - thinking
pipeline_tag: text-generation
library_name: litert-lm
---

# Qwen3-4B-Thinking-2507 — LiteRT-LM (blockwise int4)

[Qwen/Qwen3-4B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507) converted to
the **LiteRT-LM** (`.litertlm`) format for on-device inference with Google's
[LiteRT-LM](https://github.com/google-ai-edge/litert-lm) runtime (the engine behind the official
`litert-community/*` models).

Qwen3-4B-Thinking-2507 is a dense 4B **reasoning model** (`Qwen3ForCausalLM`, 36 layers) that
operates exclusively in thinking mode — it emits a `<think>…</think>` chain before its answer —
so it rides the existing Qwen3 converter and runtime directly.

| | |
|---|---|
| **File** | `model.litertlm` — int4 **block 128** (~2.3 GB) |
| **Quantization** | int4 weights (symmetric) + **OCTAV** optimal-clipping; embeddings INT8 (externalized section) |
| **Compute** | integer |
| **Context (KV cache)** | 4096 |
| **Base model** | Qwen/Qwen3-4B-Thinking-2507 |
| **Decode speed** | ~14 tok/s (iPhone 17 Pro, GPU) · ~67 tok/s (Mac M-series, GPU) |

## ⚠️ It's a reasoning model — give it room to think

This model generates a `<think>…</think>` reasoning chain, then the answer. **Run it with
`max_tokens` ≥ 2048** — at a short limit it gets cut off mid-thought and never reaches the answer.
(All quality numbers below were measured at 2048.)

## Quality — GSM8K parity

Measured on GSM8K (n=100, greedy, 0-shot chain-of-thought, **max_tokens 2048**, identical prompt
and answer-extraction for every row).

| Configuration | GSM8K |
|---|---|
| bf16 (reference) | 90.0% |
| **LiteRT int4 — block 128** | **86.0%** (−4 pt) |

int4 is at parity (−4 pt). Note: evaluating a reasoning model at a short token budget badly
understates int4 — the longer int4 reasoning chains get truncated before the answer; benchmark
reasoning models with `max_tokens` ≥ 2048.

**Why block 128 (and not block 32)?** For this reasoning model the block-32 build degraded more
(−9 pt) and produced corrupted output under the iPhone GPU delegate, while **block 128 is robust
on every backend, ~40 % faster to decode** (¼ the dequant scales — which matters when generating
long `<think>` chains), and stays at −4 pt parity. So only the block-128 build is published.

## Usage

```bash
# build litert-lm from https://github.com/google-ai-edge/litert-lm, then:
litert_lm_main \
  --model_path model.litertlm \
  --backend gpu \
  --input_prompt "A bat and a ball cost \$1.10. The bat costs \$1.00 more than the ball. How much is the ball?"
```

The `.litertlm` bundle carries the tokenizer and prompt template (Qwen3 ChatML —
`<|im_start|>role\n…<|im_end|>`, stop token `<|im_end|>`), so no separate tokenizer files are
needed. The model produces a `<think>…</think>` block followed by its answer.

## Run on Android

> **Update (July 2026):** [Google AI Edge Gallery](https://github.com/google-ai-edge/gallery) **v1.0.16+** can import litert-lm models **directly from Hugging Face** inside the app (tap **+**) — no computer or `adb` needed. The manual steps below are only required on older builds or for sideloading a local file.

The official **[Google AI Edge Gallery](https://github.com/google-ai-edge/gallery)** app runs
`.litertlm` models on-device:

1. Install a **recent** Gallery (package `com.google.ai.edge.gallery`, 1.0.15+ supports `.litertlm`).
2. Download `model.litertlm` and push it: `adb push model.litertlm /sdcard/Download/`
3. In the app tap **+**, pick the file, choose the **GPU** backend, and raise the max-tokens setting (≥2048).
4. Chat — the bundle already carries the tokenizer and Qwen3 chat template.

A 4B int4 build needs ~2.5 GB free RAM; reboot the phone first if memory is tight.

## Run on iPhone

Verified on **iPhone 17 Pro** (LiteRT-LM Swift runtime): loads and generates at ~14 tok/s.

## Conversion

Converted with the **official** [`litert-torch`](https://github.com/google-ai-edge/litert-torch)
converter — a standard `Qwen3ForCausalLM`, so it uses the existing Qwen3 path with no custom graph
code. Recipe: **blockwise-128 int4 + OCTAV** (INT4 weights, block 128, symmetric, OCTAV
optimal-clipping), embeddings INT8, KV cache 4096.

```python
from litert_torch.generative.export_hf.export import export
export(
    model="Qwen/Qwen3-4B-Thinking-2507",
    output_dir="out",
    quantization_recipe="qwen3_int4_block128_octav.json",  # blockwise-128 int4 + OCTAV, int8 embeddings
    cache_length=4096,
    externalize_embedder=True,
)
```

## License

Apache-2.0, inherited from the base model
[Qwen/Qwen3-4B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507).
