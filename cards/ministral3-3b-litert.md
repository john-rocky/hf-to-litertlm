---
license: apache-2.0
base_model: mistralai/Ministral-3-3B-Instruct-2512
tags:
  - litert
  - litert-lm
  - litertlm
  - on-device
  - edge
  - ministral
  - mistral
pipeline_tag: text-generation
library_name: litert-lm
---

# Ministral-3-3B-Instruct-2512 — LiteRT-LM (blockwise int4)

[mistralai/Ministral-3-3B-Instruct-2512](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512)
converted to the **LiteRT-LM** (`.litertlm`) format for on-device inference with
Google's [LiteRT-LM](https://github.com/google-ai-edge/litert-lm) runtime (the
engine behind the official `litert-community/*` models).

Text-only conversion (the Ministral-3 text decoder; the Pixtral vision tower is dropped).

| | |
|---|---|
| **File** | `model.litertlm` (~2.3 GB; embedding externalized so every section is <2 GiB → loads on iOS) |
| **Quantization** | int4 weights — **blockwise (block 32)**, symmetric, OCTAV clipping; tied embedding/lm_head at INT8 |
| **Compute** | integer |
| **Context (KV cache)** | 4096 |
| **Base model** | mistralai/Ministral-3-3B-Instruct-2512 (Apache-2.0) |
| **Decode speed** | ~17.6 tok/s (iPhone 17 Pro, Metal GPU; loads 7.6 s) · ~80 tok/s (Mac M4 Max, greedy) |

## Usage

Run with the LiteRT-LM runtime:

```bash
# build litert-lm from https://github.com/google-ai-edge/litert-lm, then:
litert_lm_main \
  --model_path model.litertlm \
  --backend gpu \
  --input_prompt "Explain on-device AI in one sentence."
```

The `.litertlm` bundle carries the tokenizer and the prompt template (Ministral's
native Mistral `[INST] … [/INST]` format, stop token `</s>`), so no separate
tokenizer files are needed. This is a direct-answering instruct model (no `<think>`
block) and terminates cleanly at `</s>`.

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
   tokenizer and prompt template, so the model uses its native Mistral `[INST]`
   chat format automatically.

**Device RAM (important — this is a ~2.7 GB / 3B model):** GPU on Android needs roughly
**2×** the model size (the weights *plus* the ML Drift GPU weight cache), so GPU is only
offered on ~**12 GB+** devices. On an 8 GB phone (e.g. Pixel 8a) only **CPU** is selectable,
and you must free RAM first (close apps / reboot → ~4 GB free) or the app is OOM-killed on
load. For smaller phones, prefer a **1–2B** model (e.g. a Qwen3-1.7B `.litertlm`), which runs
comfortably and can use the GPU.

See the Gallery
[Importing Local Models](https://github.com/google-ai-edge/gallery/wiki/6.-Importing-Local-Models-(optional))
guide for details. To embed the model in **your own** Android app instead, use the
LiteRT-LM Kotlin API (Gradle artifact `com.google.ai.edge.litertlm:litertlm-android`,
[getting started](https://github.com/google-ai-edge/LiteRT-LM/blob/main/docs/api/kotlin/getting_started.md)).

## Quality — GSM8K parity

Measured on GSM8K (n=100, greedy, 0-shot chain-of-thought asking for `#### <n>`,
identical prompt and answer-extraction for both rows so the only variable is the
on-device quantization:

| Configuration | GSM8K |
|---|---|
| bf16 (reference) | 89.0% |
| **This model — LiteRT int4** | **85.0%** |

LiteRT int4 is at parity: **−4 pt vs bf16** with no reasoning collapse. The model also
passes the local quality gate **8/8** (non-degenerate, clean stop at `</s>`). 85% is a
strong on-device GSM8K for a 3B and far above a naive min-max int4 of the same model
(blockwise-32 + OCTAV optimal-clipping is what preserves the accuracy).

## Conversion

Converted with [`litert-torch`](https://github.com/google-ai-edge/litert) using a
**blockwise int4** recipe (INT4 weights, block size 32, symmetric, OCTAV
optimal-clipping) with the tied embedding/lm_head kept at INT8, KV cache 4096, and
Ministral's native Mistral chat template. Ministral-3 is a standard dense decoder
(`Ministral3ForCausalLM`, YaRN RoPE), so it rides the existing converter and runtime
with no custom graph code; only the text decoder is exported (the vision tower is
dropped first).

**`externalize_embedder=True` (required for iPhone).** This 3B's weights would otherwise
be a single ~2.55 GiB TFLite section, which exceeds the ~2 GiB single-section `mmap` limit
on iOS — engine creation fails with *"Failed to map section: Cannot allocate memory"*.
Externalizing the (tied) embedding into its own section drops the main weights section to
~1.8 GiB (and dedups the tied matrix, ~2.74 GB → 2.34 GB total), so the model loads on
**iPhone (Metal GPU)** as well as Android/desktop. Same weights, so GSM8K parity is
unchanged. Verified on-device: **iPhone 17 Pro loads in ~7.6 s and decodes at ~17.6 tok/s on the
Metal GPU (prefill 21.9 tok/s, TTFT 0.70 s, ~1.4 GB footprint)** — the previously-failing
">2 GiB section / Cannot allocate memory" mmap error no longer occurs.

**Template note (important for any Mistral/Ministral):** the model must be exported
with its native Mistral `[INST] … [/INST]` template and real EOS `</s>` — **not**
ChatML. Mistral's tekken tokenizer has no `<|im_end|>` token, so under a ChatML
template the int4 model never hits a registered stop token and runs away after the
correct answer. With the Mistral template it stops cleanly.

## Reproduce

Built with `litert-torch` and a blockwise-32 + OCTAV int4 recipe, forcing the simple
Mistral `[INST]` chat template (the model's full jinja template doesn't render in the
runtime's minimal jinja engine, so the structured `[INST]` prefixes are extracted
instead):

```bash
EXTERNALIZE_EMBEDDER=1 CACHE=4096 python scripts/export_simple_template.py \
    src_models/ministral3-3b-text \
    out/ministral3-3b-boctav4 \
    templates/mistral_simple.jinja \
    BOCTAV4   # blockwise-32 int4 + OCTAV, int8 embeddings
```

The equivalent ai_edge_quantizer recipe is included as
`ministral3_int4_block32_octav.json`. The text decoder is extracted from the
multimodal checkpoint with `scripts/extract_ministral3_text.py` (drops the vision
tower; loads with missing=0/unexpected=0).

## License

Apache-2.0, inherited from the base model
[mistralai/Ministral-3-3B-Instruct-2512](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512).
