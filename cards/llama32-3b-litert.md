---
license: llama3.2
base_model: meta-llama/Llama-3.2-3B-Instruct
tags:
  - litert
  - litert-lm
  - litertlm
  - on-device
  - edge
  - llama
pipeline_tag: text-generation
library_name: litert-lm
extra_gated_button_content: Submit
---

# Llama-3.2-3B-Instruct — LiteRT-LM (blockwise int4)

Built with Llama. [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
converted to the **LiteRT-LM** (`.litertlm`) format for on-device inference with
Google's [LiteRT-LM](https://github.com/google-ai-edge/litert-lm) runtime (the
engine behind the official `litert-community/*` models).

| | |
|---|---|
| **File** | `model.litertlm` (~2.1 GB) |
| **Quantization** | int4 weights — **blockwise (block 32)**, symmetric; embeddings INT8 (externalized section) |
| **Compute** | integer |
| **Context (KV cache)** | 4096 |
| **Base model** | meta-llama/Llama-3.2-3B-Instruct |
| **Decode speed** | ~18.5 tok/s (iPhone 17 Pro, Metal GPU, ttft 0.64 s) · ~87 tok/s (Mac M4 Max, greedy) |

## Usage

Run with the LiteRT-LM runtime:

```bash
# build litert-lm from https://github.com/google-ai-edge/litert-lm, then:
litert_lm_main \
  --model_path model.litertlm \
  --backend gpu \
  --input_prompt "Explain on-device AI in one sentence."
```

The `.litertlm` bundle carries the tokenizer and the prompt template (Llama-3's
native `<|start_header_id|>role<|end_header_id|>` format, start token
`<|begin_of_text|>`, stop tokens `<|eot_id|>` / `<|end_of_text|>`), so no separate
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
   tokenizer and prompt template, so the model uses its native Llama-3 chat format
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
| bf16 (reference) | 78.0% |
| MLX 4-bit (control) | 73.3%¹ |
| **This model — LiteRT int4** | **73.0%** |

LiteRT int4 is at parity: **−5 pt vs bf16** and **equal to the MLX 4-bit control**
(73.3% vs 73.3% on the common subset¹). The model also passes the local quality gate
**8/8** (no degeneracy). bf16's 78.0% matches Llama's published 8-shot GSM8K (~77.7%),
confirming the harness is calibrated. This is a direct-answering instruct model (no
`<think>` block); it terminates cleanly at `<|eot_id|>`.

¹ The MLX control hit a reproducible Metal out-of-memory abort at one question on the
test machine, so bf16 / LiteRT-int4 / MLX are compared on the common 45-question subset
(77.8 / 73.3 / 73.3); the LiteRT-int4 headline (73.0%) is the full n=100.

## Conversion

Converted with the **official** [`litert-torch`](https://github.com/google-ai-edge/litert-torch)
converter (upstream `main`), no custom graph code. Llama-3.2-3B is a standard
`LlamaForCausalLM` architecture, so it rides the existing converter and runtime
directly. The recipe is **blockwise int4** (INT4 weights, block size 32, symmetric)
with embeddings kept at INT8 and KV cache 4096. Blockwise (not the tool's default
*channelwise*) int4 is what preserves reasoning accuracy.

```python
from litert_torch.generative.export_hf.export import export
export(
    model="meta-llama/Llama-3.2-3B-Instruct",
    output_dir="out",
    quantization_recipe="llama_int4_block32.json",  # blockwise-32 int4, int8 embeddings
    cache_length=4096,
    externalize_embedder=True,  # embedding → its own section (see iOS note)
)
```

**`externalize_embedder=True` (required for iOS).** This 28-layer 3B's weights are a
single ~2.4 GB TFLite section, which exceeds the ~2 GiB single-section `mmap` limit on
iOS — engine creation fails with *"Failed to map section: Cannot allocate memory"*.
Externalizing the (tied) embedding into its own section drops the main weights section
below 2 GiB (and dedups the tied matrix, ~2.4 GB → 2.1 GB total), so the model loads on
iPhone. Verified on **iPhone 17 Pro** (loads in 8.8 s, ~18.5 tok/s, coherent). This is
the generic equivalent of Gemma's per-layer-embedding mmap. (Mac/desktop load >2 GB
sections fine, so this only matters for iOS.)

A **block-128** variant is also available (slightly smaller, ~+5% decode on Apple GPU,
quality gate 7/8) for latency-sensitive deployments.

## License

Llama 3.2 Community License, inherited from the base model
[meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct).
Built with Llama. See https://www.llama.com/llama3_2/license/
