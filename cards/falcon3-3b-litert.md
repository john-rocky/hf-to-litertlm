---
license: other
license_name: falcon-llm-license
license_link: https://falconllm.tii.ae/falcon-terms-and-conditions.html
base_model: tiiuae/Falcon3-3B-Instruct
tags:
  - litert
  - litert-lm
  - litertlm
  - on-device
  - edge
  - falcon3
pipeline_tag: text-generation
library_name: litert-lm
---

# Falcon3-3B-Instruct — LiteRT-LM (blockwise int4)

[tiiuae/Falcon3-3B-Instruct](https://huggingface.co/tiiuae/Falcon3-3B-Instruct)
converted to the **LiteRT-LM** (`.litertlm`) format for on-device inference with
Google's [LiteRT-LM](https://github.com/google-ai-edge/litert-lm) runtime (the
engine behind the official `litert-community/*` models).

Text-only conversion (the Falcon3 decoder; no vision/audio towers).

| | |
|---|---|
| **File** | `model.litertlm` (~1.74 GB) |
| **Quantization** | int4 weights — **blockwise (block 128)**, symmetric; embeddings INT8 |
| **Compute** | integer |
| **Context (KV cache)** | 2048 |
| **Base model** | tiiuae/Falcon3-3B-Instruct |
| **Decode speed** | ~27 tok/s (iPhone 17 Pro, Metal GPU) · ~89 tok/s (Mac M4 Max, LiteRT-LM, greedy) |

## Usage

Run with the LiteRT-LM runtime:

```bash
# build litert-lm from https://github.com/google-ai-edge/litert-lm, then:
litert_lm_main \
  --model_path model.litertlm \
  --backend gpu \
  --input_prompt "Explain on-device AI in one sentence."
```

The `.litertlm` bundle carries the tokenizer and the prompt template (Falcon3's
native `<|user|>` / `<|assistant|>` format, stop token `<|endoftext|>`), so no
separate tokenizer files are needed.

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
   tokenizer and prompt template, so the model uses its native Falcon3 chat format
   automatically.

See the Gallery
[Importing Local Models](https://github.com/google-ai-edge/gallery/wiki/6.-Importing-Local-Models-(optional))
guide for details. To embed the model in **your own** Android app instead, use the
LiteRT-LM Kotlin API (Gradle artifact `com.google.ai.edge.litertlm:litertlm-android`,
[getting started](https://github.com/google-ai-edge/LiteRT-LM/blob/main/docs/api/kotlin/getting_started.md)).

## Quality — GSM8K parity

Measured on GSM8K (n=100, greedy, 0-shot chain-of-thought asking for `#### <n>`,
identical prompt and answer-extraction for every row). The 4-bit MLX build is the
known-good 4-bit control:

| Configuration | GSM8K |
|---|---|
| bf16 (reference) | 75% |
| MLX 4-bit (control) | 76% |
| **This model — LiteRT int4** | **77%** |

LiteRT int4 is fully at parity — it matches or slightly exceeds both the 4-bit
control and bf16 here (the small spread is sampling noise at n=100). This is a
direct-answering instruct model (no `<think>` block) and terminates cleanly at
`<|endoftext|>`.

## Conversion

Converted with [`litert-torch`](https://github.com/google-ai-edge/litert) using a
**blockwise int4** recipe (INT4 weights, block size 128, symmetric) with embeddings
kept at INT8, KV cache 2048, and Falcon3's native chat template. Falcon3-3B is a
standard `LlamaForCausalLM` architecture, so it rides the existing converter and
runtime with no custom code. Blockwise (not channelwise) int4 is what preserves
reasoning accuracy.

## Reproduce (official tools only)

Built with **stock `litert-torch`** — no custom code, no graph patches. The only
non-default choice is the int4 recipe: the tool's default named int4 is
*channelwise* (which degrades small models), so this uses **blockwise-128** (the
scheme the official models ship), passed as a recipe file to the standard export:

```python
from litert_torch.generative.export_hf.export import export
export(
    model="tiiuae/Falcon3-3B-Instruct",
    output_dir="out",
    quantization_recipe="falcon_int4_block128.json",  # included in this repo
    cache_length=2048,
    trust_remote_code=True,
)
```

`falcon_int4_block128.json` is included in this repo. (If the export errors with a
missing `ai_edge_quantizer/recipes/` directory, create it empty — a packaging gap
in some releases that trips the `.json`-recipe path.)

## License

Falcon LLM License (TII), inherited from the base model
[tiiuae/Falcon3-3B-Instruct](https://huggingface.co/tiiuae/Falcon3-3B-Instruct).
See https://falconllm.tii.ae/falcon-terms-and-conditions.html
