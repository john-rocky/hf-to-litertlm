---
license: apache-2.0
base_model: Qwen/Qwen3-1.7B
tags:
  - litert
  - litert-lm
  - litertlm
  - on-device
  - edge
  - qwen3
pipeline_tag: text-generation
library_name: litert-lm
---

# Qwen3-1.7B — LiteRT-LM (blockwise int4)

[Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) converted to the
**LiteRT-LM** (`.litertlm`) format for on-device inference with Google's
[LiteRT-LM](https://github.com/google-ai-edge/litert-lm) runtime (the engine
behind the official `litert-community/*` models).

| | |
|---|---|
| **File** | `model.litertlm` (~1.6 GB) |
| **Quantization** | int4 weights — **blockwise (block 32)**, symmetric, OCTAV clipping; tied embedding/lm_head at INT8 |
| **Compute** | integer |
| **Context (KV cache)** | 4096 |
| **Base model** | Qwen/Qwen3-1.7B (Apache-2.0) |
| **Decode speed** | ~117 tok/s (Apple Silicon, CPU, LiteRT-LM, greedy) |

## Usage

Run with the LiteRT-LM runtime:

```bash
# build litert-lm from https://github.com/google-ai-edge/litert-lm, then:
litert_lm_main \
  --model_path model.litertlm \
  --backend cpu \
  --input_prompt "What is the capital of Japan?"
```

The `.litertlm` bundle carries the tokenizer and the prompt template, so no
separate tokenizer files are needed. Qwen3 is a reasoning model: it emits a
`<think>…</think>` block before the answer — give it enough generation budget for
reasoning-heavy queries.

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
   **CPU** backend (the speed above was measured on CPU; GPU also works).
4. Chat. Nothing else to configure — the `.litertlm` bundle already carries the
   tokenizer and prompt template, so the model uses its native Qwen3 chat format
   automatically. As a reasoning model it emits a `<think>…</think>` block first, so
   set a generous max-tokens limit in the app.

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
| bf16 (reference, n=50) | 72% |
| MLX 4-bit (control) | 73% |
| **This model — LiteRT int4** | **72%** |

LiteRT int4 is at parity with both the 4-bit control and bf16 — no meaningful
reasoning loss from the on-device quantization.

## Conversion

Converted with [`litert-torch`](https://github.com/google-ai-edge/litert) using a
**blockwise int4** recipe (INT4 weights, block size 32, symmetric, OCTAV
optimal-clipping) with the tied embedding/lm_head kept at INT8, KV cache 4096, and
Qwen3's native chat template. Blockwise (not channelwise) int4 is what makes small
models retain their reasoning accuracy — for this 1.7B, block size **32** (not 128)
is needed to hold parity (block128 drops GSM8K to ~56%).

## Reproduce (official tools only)

Built with **stock `litert-torch`** — no custom code, no graph patches. The only
non-default choice is the int4 recipe: the tool's default named int4 is
*channelwise* (which collapses small models), so this uses **blockwise-32 + OCTAV**,
passed as a recipe file to the standard export:

```python
from litert_torch.generative.export_hf.export import export
export(
    model="Qwen/Qwen3-1.7B",
    output_dir="out",
    quantization_recipe="qwen3_int4_block32_octav.json",  # included in this repo
    cache_length=4096,
    trust_remote_code=True,
)
```

`qwen3_int4_block32_octav.json` is included in this repo. (If the export errors
with a missing `ai_edge_quantizer/recipes/` directory, create it empty — a
packaging gap in some releases that trips the `.json`-recipe path.)

## License

Apache-2.0, inherited from the base model
[Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B).
