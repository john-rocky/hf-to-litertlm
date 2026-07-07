---
license: mit
base_model: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
tags:
  - litert
  - litert-lm
  - litertlm
  - on-device
  - edge
  - reasoning
  - deepseek-r1
pipeline_tag: text-generation
library_name: litert-lm
---

# DeepSeek-R1-Distill-Qwen-1.5B — LiteRT-LM (blockwise int4)

[deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)
converted to the **LiteRT-LM** (`.litertlm`) format for on-device inference with
Google's [LiteRT-LM](https://github.com/google-ai-edge/litert-lm) runtime.

A **mobile-size reasoning** model: it emits a `<think> … </think>` chain before
the answer, and at ~1 GB it runs on a phone. MIT-licensed (Apache-2.0 Qwen2.5
base). Converted with the **official** upstream `litert-torch` — no fork.

| | |
|---|---|
| **File** | `model.litertlm` (~1.0 GB) |
| **Quantization** | int4 weights — **blockwise (block 32) + OCTAV** optimal-clipping, symmetric; embedding INT8 |
| **Compute** | integer |
| **Context (KV cache)** | 4096 |
| **Base model** | deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B |
| **Decode speed** | ~116 tok/s (Mac M-series, Metal GPU, greedy); runs on 8 GB phones (iPhone / Android) |

## Run on Android

> **Update (July 2026):** [Google AI Edge Gallery](https://github.com/google-ai-edge/gallery) **v1.0.16+** can import litert-lm models **directly from Hugging Face** inside the app (tap **+**) — no computer or `adb` needed. The manual steps below are only required on older builds or for sideloading a local file.

The easiest way to try this on a phone is the official
**[Google AI Edge Gallery](https://github.com/google-ai-edge/gallery)** app:

1. Install a recent Gallery (`com.google.ai.edge.gallery`, 1.0.15+ supports `.litertlm`).
2. `adb push model.litertlm /sdcard/Download/`
3. In the app: **+** → pick the file → CPU or GPU. At ~1 GB this fits comfortably.
4. Chat — the bundle carries the tokenizer and DeepSeek prompt template
   (`<｜User｜>` / `<｜Assistant｜>`, stop `<｜end▁of▁sentence｜>`). The model opens a
   `<think>` block, reasons, then answers.

To embed it in your own app, use the LiteRT-LM Kotlin API
(`com.google.ai.edge.litertlm:litertlm-android`).

## Quality — GSM8K

GSM8K (n=100, greedy, 0-shot, identical prompt + extraction; `max_new_tokens=2048`).

| Configuration | GSM8K |
|---|---|
| bf16 (reference) | 81.0% |
| **This model — LiteRT int4 (BOCTAV4)** | **73.0%** |

73 % is a strong, coherent, non-degenerate score for a **1.5B reasoning model that
fits on a phone**; the `<think>` reasoning is preserved through 4-bit. At 1.5B,
int4 costs ~8 pt vs bf16 (small-model 4-bit sensitivity — a 1.5B has less
redundancy than the 7B sibling, which is at −1 pt parity). Shipped as int4 for the
best on-device size/speed.

## Conversion

Official upstream [`litert-torch`](https://github.com/google-ai-edge/litert)
`export_hf` (clean worktree at `upstream/main`, no fork). `Qwen2ForCausalLM`, no
custom code. int4 = blockwise-32 + OCTAV, INT8 embedding, KV cache 4096.

## License

MIT (model weights); Qwen2.5 base is Apache-2.0. Commercial use and derivatives permitted.
