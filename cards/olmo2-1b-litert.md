---
license: apache-2.0
base_model: allenai/OLMo-2-0425-1B-Instruct
tags:
  - litert
  - litert-lm
  - litertlm
  - on-device
  - edge
  - olmo2
pipeline_tag: text-generation
library_name: litert-lm
---

# OLMo-2-1B-Instruct — LiteRT-LM (blockwise int4)

[allenai/OLMo-2-0425-1B-Instruct](https://huggingface.co/allenai/OLMo-2-0425-1B-Instruct)
converted to the **LiteRT-LM** (`.litertlm`) format for on-device inference with
Google's [LiteRT-LM](https://github.com/google-ai-edge/litert-lm) runtime (the
engine behind the official `litert-community/*` models).

OLMo-2 is AllenAI's **fully-open** model family (Apache-2.0; open weights, data,
and training code). This 1B variant is small enough to run on a phone — verified on
iPhone 17 Pro. Converted with the **official** upstream `litert-torch` — no fork.

| | |
|---|---|
| **File** | `model.litertlm` (~0.93 GB) |
| **Quantization** | int4 weights — **blockwise (block 32) + OCTAV** optimal-clipping, symmetric; embedding INT8 |
| **Compute** | integer |
| **Context (KV cache)** | 4096 |
| **Base model** | allenai/OLMo-2-0425-1B-Instruct |
| **Decode speed** | ~24 tok/s (iPhone 17 Pro; loads 5.2 s, ~1.2 GB footprint) · ~138 tok/s (Mac M-series, Metal GPU) |

## Usage

Run with the LiteRT-LM runtime:

```bash
litert_lm_main \
  --model_path model.litertlm \
  --backend gpu \
  --input_prompt "Explain on-device AI in one sentence."
```

The `.litertlm` bundle carries the tokenizer and the prompt template (OLMo-2's
native Tülu format — `<|user|>` / `<|assistant|>`, stop token `<|endoftext|>`),
so no separate tokenizer files are needed.

## Run on Android

> **Update (July 2026):** [Google AI Edge Gallery](https://github.com/google-ai-edge/gallery) **v1.0.16+** can import litert-lm models **directly from Hugging Face** inside the app (tap **+**) — no computer or `adb` needed. The manual steps below are only required on older builds or for sideloading a local file.

The easiest way to try this model on a phone is the official
**[Google AI Edge Gallery](https://github.com/google-ai-edge/gallery)** app:

1. Install a **recent** Gallery (package `com.google.ai.edge.gallery`, APK from the repo's
   [releases](https://github.com/google-ai-edge/gallery/releases) — 1.0.15+ supports `.litertlm`).
2. Download `model.litertlm` and push it to the device:
   ```bash
   adb push model.litertlm /sdcard/Download/
   ```
3. In the app, tap **+** (bottom-right), pick the file, and choose CPU or GPU. At
   ~0.93 GB this 1B fits comfortably on an 8 GB phone.
4. Chat — the bundle already carries the tokenizer and OLMo-2 prompt template.

See the Gallery
[Importing Local Models](https://github.com/google-ai-edge/gallery/wiki/6.-Importing-Local-Models-(optional))
guide for details. To embed it in **your own** Android app, use the LiteRT-LM Kotlin API
(`com.google.ai.edge.litertlm:litertlm-android`).

## Quality — GSM8K

Measured on GSM8K (n=100, greedy, 0-shot chain-of-thought, identical prompt and
answer-extraction for every row).

| Configuration | GSM8K |
|---|---|
| bf16 (reference) | 72.0% |
| **This model — LiteRT int4 (BOCTAV4)** | **63.0%** |

63 % is a strong, coherent, non-degenerate score for a 1B (the `\boxed{}`-style answers
terminate cleanly at `<|endoftext|>`). At 1B, 4-bit quantization costs ~9 pt vs bf16 —
a small model has less redundancy to absorb int4 rounding than a 3B+ (where the same
recipe is at parity). An int8 build recovers only ~2 pt (65 %) for +60 % size, so int4
is shipped as the best size/quality trade-off for on-device.

## Conversion

Converted with the **official** upstream [`litert-torch`](https://github.com/google-ai-edge/litert)
`export_hf` (clean `git worktree` at `upstream/main`, dev-fork patches excluded).
`Olmo2ForCausalLM` rides the stock converter with no custom code: QK-norm and OLMo-2's
reordered post-norm lower to generic ops. The int4 recipe is **blockwise (block 32) +
OCTAV** with the embedding at INT8.

## License

Apache-2.0, inherited from the base model
[allenai/OLMo-2-0425-1B-Instruct](https://huggingface.co/allenai/OLMo-2-0425-1B-Instruct).
