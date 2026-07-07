---
license: apache-2.0
base_model: mistralai/Ministral-3-3B-Reasoning-2512
tags:
  - litert
  - litert-lm
  - litertlm
  - on-device
  - edge
  - reasoning
  - ministral
  - mistral
pipeline_tag: text-generation
library_name: litert-lm
---

# Ministral-3-3B-Reasoning-2512 — LiteRT-LM (blockwise int4)

[mistralai/Ministral-3-3B-Reasoning-2512](https://huggingface.co/mistralai/Ministral-3-3B-Reasoning-2512)
converted to the **LiteRT-LM** (`.litertlm`) format for on-device inference with
Google's [LiteRT-LM](https://github.com/google-ai-edge/litert-lm) runtime (the
engine behind the official `litert-community/*` models).

Text-only conversion (the Ministral-3 text decoder; the Pixtral vision tower is dropped).

| | |
|---|---|
| **File** | `model.litertlm` (~2.2 GB; embedding externalized so every section is <2 GiB → loads on iOS) |
| **Quantization** | int4 weights — **blockwise (block 32)**, symmetric, OCTAV clipping; tied embedding/lm_head at INT8 |
| **Compute** | integer |
| **Context (KV cache)** | 4096 |
| **Base model** | mistralai/Ministral-3-3B-Reasoning-2512 (Apache-2.0) |
| **Decode speed** | ~91 tok/s (Mac M4 Max, Metal GPU, greedy) |

## Usage

Run with the LiteRT-LM runtime:

```bash
# build litert-lm from https://github.com/google-ai-edge/litert-lm, then:
litert_lm_main \
  --model_path model.litertlm \
  --backend gpu \
  --input_prompt "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total?"
```

The `.litertlm` bundle carries the tokenizer and the prompt template (Ministral's
native Mistral `[INST] … [/INST]` format, stop token `</s>`), so no separate
tokenizer files are needed. This is a **reasoning** model: it works the problem
step-by-step before giving the final answer (best evaluated with a generous token
budget — see below), and terminates cleanly at `</s>`.

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
   **GPU** backend (CPU also works).
4. Chat. The bundle already carries the tokenizer and the native Mistral `[INST]`
   chat template, so nothing else needs configuring.

**Device RAM (important — this is a ~2.2 GB / 3B model):** GPU on Android needs roughly
**2×** the model size (weights *plus* the ML Drift GPU weight cache), so GPU is only
offered on ~**12 GB+** devices. On an 8 GB phone only **CPU** is selectable; free RAM
first (close apps / reboot) or the app is OOM-killed on load. Because this is a reasoning
model that emits long chains of thought, give it a high max-tokens.

## Quality — GSM8K

Measured on GSM8K (n=150, greedy, 0-shot chain-of-thought, **max-tokens 2048** — a
reasoning model needs the budget to finish its work; scoring it at 512 tokens
falsely penalises it):

| Configuration | GSM8K |
|---|---|
| **This model — LiteRT int4 (block32 + OCTAV)** | **90.7%** |

90.7% is a strong on-device GSM8K for a 3B, and *above* the same-family
[Ministral-3-3B-Instruct LiteRT build](https://huggingface.co/litert-community/Ministral-3-3B-Instruct-2512)
(85%) — the reasoning traces help. No reasoning collapse, non-degenerate; the model
also passes the local quality gate **8/8** with a clean stop at `</s>`. Blockwise-32 +
OCTAV optimal-clipping (data-free) is what preserves the accuracy versus a naive
min-max int4.

## Conversion

Converted with [`litert-torch`](https://github.com/google-ai-edge/litert) using a
**blockwise int4** recipe (INT4 weights, block size 32, symmetric, OCTAV
optimal-clipping) with the tied embedding/lm_head kept at INT8, KV cache 4096, and
Ministral's native Mistral chat template. Ministral-3 is a standard dense decoder
(`Ministral3ForCausalLM`), so it rides the existing converter and runtime with no
custom graph code; only the text decoder is exported (the vision tower is dropped
first, with a strict `missing=0 / unexpected=0` weight check).

**`externalize_embedder=True` (required for iPhone).** This 3B's weights would otherwise
be a single >2 GiB TFLite section, which exceeds the ~2 GiB single-section `mmap` limit
on iOS — engine creation fails with *"Failed to map section: Cannot allocate memory."*
Externalizing the (tied) embedding into its own section drops the main weights section
under 2 GiB (total ~2.2 GB), so the model loads on **iPhone (Metal GPU)** as well as
Android/desktop. Same weights, so GSM8K is unchanged.

**Template note (important for any Mistral/Ministral):** the model must be exported
with its native Mistral `[INST] … [/INST]` template and real EOS `</s>` — **not**
ChatML. Mistral's tokenizer has no `<|im_end|>` token, so under a ChatML template the
int4 model never hits a registered stop token and runs away after the correct answer.
With the Mistral template it stops cleanly.

## Reproduce

```bash
# 1. extract the dense text decoder from the multimodal checkpoint (missing=0/unexpected=0)
python scripts/extract_text_backbone.py \
    mistralai/Ministral-3-3B-Reasoning-2512 \
    src_models/ministral-3-3b-reasoning-text

# 2. convert: blockwise-32 int4 + OCTAV, int8 embeddings, embedding externalized for iOS
EXTERNALIZE_EMBEDDER=1 FORCE_SPM=1 CACHE=4096 python scripts/export_simple_template.py \
    src_models/ministral-3-3b-reasoning-text \
    out/ministral-3-3b-reasoning-ext \
    templates/mistral_simple.jinja \
    BOCTAV4
```

## License

Apache-2.0, inherited from the base model
[mistralai/Ministral-3-3B-Reasoning-2512](https://huggingface.co/mistralai/Ministral-3-3B-Reasoning-2512).
