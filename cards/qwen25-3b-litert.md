---
license: other
license_name: qwen-research
license_link: LICENSE
base_model: Qwen/Qwen2.5-3B-Instruct
tags:
  - litert
  - litert-lm
  - litertlm
  - on-device
  - edge
  - qwen2
  - gptq
pipeline_tag: text-generation
library_name: litert-lm
---

# Qwen2.5-3B-Instruct — LiteRT-LM (GPTQ-calibrated int4, block 128)

**Built with Qwen.**

[Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) converted to the
**LiteRT-LM** (`.litertlm`) format for on-device inference with Google's
[LiteRT-LM](https://github.com/google-ai-edge/litert-lm) runtime (the engine behind the
`litert-community/*` models).

What makes this build different: the int4 weights are **not re-quantized from scratch** — they carry
**Qwen's official GPTQ calibration**
([Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4)),
transported **losslessly** into the LiteRT bundle via `ai-edge-quantizer`'s
`dequantized_weight_recovery` (blockwise support, nightly ≥ 0.8.0.dev20260703). You get
calibrated-int4 quality at block-128 speed, with no calibration step in the conversion.

| | |
|---|---|
| **File** | `model.litertlm` — int4 **block 128** (~1.75 GB) |
| **Quantization** | int4 weights (symmetric, blockwise-128) on **Qwen's official GPTQ grid**; embeddings + lm_head INT8 |
| **Compute** | integer (dynamic int8 activations) |
| **Context (KV cache)** | 4096 |
| **Base model** | Qwen/Qwen2.5-3B-Instruct (36 layers, `Qwen2ForCausalLM`) |
| **Decode speed** | ~74 tok/s (Mac M-series, GPU) |

## Quality — GSM8K parity

Measured on GSM8K (n=100, greedy, 0-shot chain-of-thought, max_tokens 512, identical prompt and
answer-extraction for every row).

| Configuration | GSM8K |
|---|---|
| bf16 (reference) | 81.0% |
| Qwen official GPTQ-Int4, dequantized in PyTorch (n=50) | 82.0% |
| **LiteRT int4 — block 128 (this file)** | **75.0%** (−6 pt vs bf16) |

The official GPTQ calibration itself is lossless on GSM8K (82.0 vs 81.0 = noise), so the −6 pt is
the cost of the on-device execution format (integer compute), not of the 4-bit weights. The 8-question
smoke gate reads **8/8** (arithmetic, factual, translation — all correct, terse clean answers, no
degeneration).

## Usage

```bash
# build litert-lm from https://github.com/google-ai-edge/litert-lm, then:
litert_lm_main \
  --model_path model.litertlm \
  --backend gpu \
  --input_prompt "Natalia sold clips to 48 friends in April, and half as many in May. How many altogether?"
```

The `.litertlm` bundle carries the tokenizer and prompt template (Qwen2 ChatML —
`<|im_start|>role\n…<|im_end|>`), so no separate tokenizer files are needed.

## Run on Android

> **Update (July 2026):** [Google AI Edge Gallery](https://github.com/google-ai-edge/gallery)
> **v1.0.16+** can import litert-lm models **directly from Hugging Face** inside the app (tap **+**)
> — no computer or `adb` needed. The manual steps below are only required on older builds or for
> sideloading a local file.

The official **[Google AI Edge Gallery](https://github.com/google-ai-edge/gallery)** app runs
`.litertlm` models on-device:

1. Install a **recent** Gallery (package `com.google.ai.edge.gallery`, 1.0.15+ supports `.litertlm`).
2. Download `model.litertlm` and push it: `adb push model.litertlm /sdcard/Download/`
3. In the app tap **+**, pick the file, and choose the **GPU** backend.
4. Chat — the bundle already carries the tokenizer and Qwen2 chat template.

## Conversion — GPTQ grid pass-through

Converted with the official [`litert-torch`](https://github.com/google-ai-edge/litert-torch)
converter. Instead of a data-free int4 recipe, the quantization stage uses
`ai-edge-quantizer`'s **`dequantized_weight_recovery`** algorithm (blockwise support landed
2026-06-11, nightly-only at the time of conversion): the official GPTQ checkpoint is dequantized to
fp32 (exact — fp16 scale × int4 is exactly representable in fp32), and recovery re-derives the
per-block scales bit-exactly, so the deployed int4 grid **is** Qwen's calibrated grid.

```json
[
  {"regex": ".*", "operation": "*",
   "algorithm_key": "dequantized_weight_recovery",
   "op_config": {"weight_tensor_config": {"num_bits": 4, "symmetric": true,
                 "granularity": "BLOCKWISE_128", "dtype": "INT"},
                 "compute_precision": "INTEGER"}},
  {"regex": ".*", "operation": "EMBEDDING_LOOKUP",
   "algorithm_key": "min_max_uniform_quantize",
   "op_config": {"weight_tensor_config": {"num_bits": 8, "symmetric": true,
                 "granularity": "CHANNELWISE", "dtype": "INT"}}},
  {"regex": ".*(logits_output|Linear_lm_head).*", "operation": "FULLY_CONNECTED",
   "algorithm_key": "min_max_uniform_quantize",
   "op_config": {"weight_tensor_config": {"num_bits": 8, "symmetric": true,
                 "granularity": "CHANNELWISE", "dtype": "INT"}}}
]
```

(The embedding / tied lm_head is not GPTQ-quantized upstream, so it goes to INT8. KV cache 4096.)

## License

**Qwen Research License** (see `LICENSE`), inherited from the base model
[Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct). **Non-commercial
(research/evaluation) use only** — for commercial use, request a license from Alibaba Cloud.

This repository is a **modified distribution** of the Qwen materials: the model weights were
quantized (official GPTQ int4 grid, transported via `dequantized_weight_recovery`) and repackaged
into the LiteRT-LM `.litertlm` format as described in the Conversion section above. Attribution
notice is in `NOTICE`. Built with Qwen.
