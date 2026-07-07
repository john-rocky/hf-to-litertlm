---
license: apache-2.0
base_model: HuggingFaceTB/SmolLM3-3B
tags:
  - litert
  - litert-lm
  - litertlm
  - on-device
  - edge
  - smollm3
pipeline_tag: text-generation
library_name: litert-lm
---

# SmolLM3-3B — LiteRT-LM (blockwise int4)

[HuggingFaceTB/SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B)
converted to the **LiteRT-LM** (`.litertlm`) format for on-device inference with
Google's [LiteRT-LM](https://github.com/google-ai-edge/litert-lm) runtime (the
engine behind the official `litert-community/*` models).

SmolLM3 is a fully-open 3B decoder (Apache-2.0) with GQA, a NoPE attention schedule,
multilingual support, and long-context training — a strong small reasoner.

| | |
|---|---|
| **File** | `model.litertlm` (~1.9 GB) |
| **Quantization** | int4 weights — **blockwise (block 32) + OCTAV** optimal-clipping, symmetric; embedding INT8 |
| **Compute** | integer |
| **Context (KV cache)** | 4096 |
| **Base model** | HuggingFaceTB/SmolLM3-3B |
| **Decode speed** | ~22.5 tok/s (iPhone 17 Pro, Metal GPU; loads 7.7 s, ~1.24 GB footprint) · ~93 tok/s (Mac M-series, LiteRT-LM, Metal GPU, greedy) |

## Usage

Run with the LiteRT-LM runtime:

```bash
# build litert-lm from https://github.com/google-ai-edge/litert-lm, then:
litert_lm_main \
  --model_path model.litertlm \
  --backend gpu \
  --input_prompt "Explain on-device AI in one sentence."
```

The `.litertlm` bundle carries the tokenizer and the prompt template (ChatML —
`<|im_start|>role` / `<|im_end|>`, stop token `<|im_end|>`), so no separate
tokenizer files are needed.

## Run on Android

> **Update (July 2026):** [Google AI Edge Gallery](https://github.com/google-ai-edge/gallery) **v1.0.16+** can import litert-lm models **directly from Hugging Face** inside the app (tap **+**) — no computer or `adb` needed. The manual steps below are only required on older builds or for sideloading a local file.

The easiest way to try this model on a phone is the official
**[Google AI Edge Gallery](https://github.com/google-ai-edge/gallery)** app — it
runs `.litertlm` models fully on-device and can import your own:

1. Install a **recent** Gallery (package `com.google.ai.edge.gallery`, APK from the repo's
   [releases](https://github.com/google-ai-edge/gallery/releases) — 1.0.15+ supports
   `.litertlm`). Older 1.0.x builds (package `com.google.aiedge.gallery`) only accept the
   legacy MediaPipe `.task` format and reject `.litertlm`.
2. Download `model.litertlm` from this repo and push it to the device:
   ```bash
   adb push model.litertlm /sdcard/Download/
   ```
3. In the app, tap the **+** button (bottom-right), pick the file, and choose the
   **GPU** backend (CPU also works).
4. Chat. Nothing else to configure — the `.litertlm` bundle already carries the
   tokenizer and ChatML prompt template.

See the Gallery
[Importing Local Models](https://github.com/google-ai-edge/gallery/wiki/6.-Importing-Local-Models-(optional))
guide for details. To embed the model in **your own** Android app instead, use the
LiteRT-LM Kotlin API (Gradle artifact `com.google.ai.edge.litertlm:litertlm-android`,
[getting started](https://github.com/google-ai-edge/LiteRT-LM/blob/main/docs/api/kotlin/getting_started.md)).

## Quality — GSM8K parity

Measured on GSM8K (n=100, greedy, 0-shot chain-of-thought asking for `#### <n>`,
identical prompt and answer-extraction for both rows — only the quantization differs).

| Configuration | GSM8K |
|---|---|
| bf16 (reference) | 81.0% |
| **This model — LiteRT int4 (BOCTAV4)** | **81.0%** |

LiteRT int4 is **fully at parity — 0.0 pt** vs the bf16 reference. The blockwise-32 +
OCTAV recipe with a 4096 KV cache preserves reasoning accuracy exactly at n=100. The
model produces visible step-by-step chain-of-thought in the answer body and
terminates cleanly at `<|im_end|>` (no rambling).

## Conversion

Converted with [`litert-torch`](https://github.com/google-ai-edge/litert) via its
generic `export_hf` path. `SmolLM3ForCausalLM` rides the existing converter with no
custom code: the **NoPE** attention schedule (rotary disabled on every 4th layer,
`no_rope_layer_interval=4`) lowers to generic ops with no custom kernel. The int4
recipe is **blockwise (block 32) + OCTAV** optimal-clipping with the embedding kept
at INT8; the embedding is externalized into its own bundle section so the main
weights section stays under the iOS ~2 GiB single-mmap limit. Blockwise (not
channelwise) int4 plus OCTAV is what holds reasoning accuracy at parity.

## License

Apache-2.0, inherited from the base model
[HuggingFaceTB/SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B).
