---
license: apache-2.0
base_model: allenai/OLMo-2-1124-7B-Instruct
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

# OLMo-2-7B-Instruct — LiteRT-LM (blockwise int4)

[allenai/OLMo-2-1124-7B-Instruct](https://huggingface.co/allenai/OLMo-2-1124-7B-Instruct)
converted to the **LiteRT-LM** (`.litertlm`) format for on-device inference with
Google's [LiteRT-LM](https://github.com/google-ai-edge/litert-lm) runtime (the
engine behind the official `litert-community/*` models).

OLMo-2 is AllenAI's **fully-open** 7B dense model (Apache-2.0; open weights, data,
and training code). Architecturally it uses QK-norm and a reordered post-norm.
Converted with the **official** upstream `litert-torch` — no fork, no custom code.

| | |
|---|---|
| **File** | `model.litertlm` (~4.0 GB) |
| **Quantization** | int4 weights — **blockwise (block 32) + OCTAV** optimal-clipping, symmetric; embedding INT8 |
| **Compute** | integer |
| **Context (KV cache)** | 4096 |
| **Base model** | allenai/OLMo-2-1124-7B-Instruct |
| **Decode speed** | ~35 tok/s (Mac M-series, LiteRT-LM, Metal GPU, greedy) |
| **Platforms** | Mac (desktop) ✓ · Android CPU ✓ · **iPhone ✗** (7B exceeds the iOS ~2 GiB single-section `mmap` limit — see below) |

## Usage

Run with the LiteRT-LM runtime:

```bash
# build litert-lm from https://github.com/google-ai-edge/litert-lm, then:
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
**[Google AI Edge Gallery](https://github.com/google-ai-edge/gallery)** app — it
runs `.litertlm` models fully on-device and can import your own:

1. Install a **recent** Gallery (package `com.google.ai.edge.gallery`, APK from the repo's
   [releases](https://github.com/google-ai-edge/gallery/releases) — 1.0.15+ supports
   `.litertlm`).
2. Download `model.litertlm` from this repo and push it to the device:
   ```bash
   adb push model.litertlm /sdcard/Download/
   ```
3. In the app, tap the **+** button (bottom-right), pick the file, and choose the
   backend. This is a ~4 GB 7B model — on 8 GB phones it exceeds the GPU/ML-Drift
   budget, so use the **CPU** backend there; higher-memory devices can use GPU.
4. Chat. Nothing else to configure — the `.litertlm` bundle already carries the
   tokenizer and OLMo-2 prompt template.

See the Gallery
[Importing Local Models](https://github.com/google-ai-edge/gallery/wiki/6.-Importing-Local-Models-(optional))
guide for details. To embed the model in **your own** Android app instead, use the
LiteRT-LM Kotlin API (Gradle artifact `com.google.ai.edge.litertlm:litertlm-android`,
[getting started](https://github.com/google-ai-edge/LiteRT-LM/blob/main/docs/api/kotlin/getting_started.md)).

## Quality — GSM8K parity

Measured on GSM8K (n=100, greedy, 0-shot chain-of-thought, identical prompt and
answer-extraction for both rows — only the quantization differs).

| Configuration | GSM8K |
|---|---|
| bf16 (reference) | 81.0% |
| **This model — LiteRT int4 (BOCTAV4)** | **80.0%** |

LiteRT int4 is **at parity — −1.0 pt** vs the bf16 reference (within n=100 noise).
The blockwise-32 + OCTAV recipe with a 4096 KV cache preserves reasoning accuracy.
The model produces step-by-step chain-of-thought and terminates cleanly at
`<|endoftext|>`.

## Conversion

Converted with the **official** upstream [`litert-torch`](https://github.com/google-ai-edge/litert)
`export_hf` (a clean `git worktree` at `upstream/main`, with the dev fork's patches
excluded). `Olmo2ForCausalLM` rides the stock converter with no custom code: QK-norm
and OLMo-2's reordered post-norm lower to generic ops with no flex / no custom kernel.
The int4 recipe is **blockwise (block 32) + OCTAV** optimal-clipping with the embedding
kept at INT8 and externalized into its own bundle section. Blockwise (not channelwise)
int4 plus OCTAV is what holds reasoning accuracy at parity.

**Platform note.** At 7B the int4 main-weights section is ~3.61 GiB, which exceeds the
iOS single-section `mmap` limit (~2 GiB) — loading on iPhone aborts with `std::bad_alloc`
(verified on iPhone 17 Pro). `externalize_embedder=True` moves the 394 MiB embedder out
but cannot bring the main section under 2 GiB at this size. The bundle therefore targets
**desktop (Mac, verified ~35 tok/s) and Android (CPU)**; it does not run on iPhone. For
on-device iOS, use a ≤3B model (e.g. the 3B LiteRT bundles), which stay under the limit.

## License

Apache-2.0, inherited from the base model
[allenai/OLMo-2-1124-7B-Instruct](https://huggingface.co/allenai/OLMo-2-1124-7B-Instruct).
