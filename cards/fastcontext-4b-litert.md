---
license: mit
base_model: microsoft/FastContext-1.0-4B-SFT
tags:
  - litert
  - litert-lm
  - litertlm
  - on-device
  - edge
  - qwen3
  - agent
  - tool-calling
pipeline_tag: text-generation
library_name: litert-lm
---

# FastContext-1.0-4B-SFT — LiteRT-LM (blockwise int4)

[microsoft/FastContext-1.0-4B-SFT](https://huggingface.co/microsoft/FastContext-1.0-4B-SFT)
converted to the **LiteRT-LM** (`.litertlm`) format for on-device inference with
Google's [LiteRT-LM](https://github.com/google-ai-edge/litert-lm) runtime (the engine
behind the official `litert-community/*` models).

FastContext is a lightweight **repository-exploration sub-agent** for coding agents —
it specializes in repo discovery and evidence gathering via parallel tool calls
(READ / GLOB / GREP). The 4B backbone is **Qwen3-4B-Instruct** (`Qwen3ForCausalLM`),
so it rides the existing Qwen3 converter and runtime directly.

| | |
|---|---|
| **Files** | `model.litertlm` — int4 **block 32** (best quality, recommended) · `model_block128.litertlm` — int4 **block 128** (faster decode) |
| **Quantization** | int4 weights (symmetric) + **OCTAV** optimal-clipping; embeddings INT8 (externalized section) |
| **Compute** | integer |
| **Context (KV cache)** | 4096 |
| **Base model** | microsoft/FastContext-1.0-4B-SFT (Qwen3-4B-Instruct) |

## Which file?

| File | int4 granularity | GSM8K | iPhone 17 Pro | Mac (M-series, GPU) |
|---|---|---|---|---|
| **`model.litertlm`** | block 32 | **88.0%** (best) | 10 tok/s | 64–73 tok/s |
| **`model_block128.litertlm`** | block 128 | 81.0% | **14 tok/s** (+40%) | 66–68 tok/s |

**Use `model.litertlm` (block 32)** unless decode latency dominates — it is full parity
with bf16 (see below) and loads on iPhone, Android and desktop alike. The block-128 build
trades −6 pt accuracy for ~40% faster decode (block 128 stores ¼ the scales → lighter GPU
dequant; this is the granularity Google's official Gemma block-128 bundles use).

## Usage

Run with the LiteRT-LM runtime:

```bash
# build litert-lm from https://github.com/google-ai-edge/litert-lm, then:
litert_lm_main \
  --model_path model.litertlm \
  --backend gpu \
  --input_prompt "List the files you would read to understand a Python project's entry point."
```

The `.litertlm` bundle carries the tokenizer and prompt template (Qwen3 ChatML —
`<|im_start|>role\n…<|im_end|>`, stop token `<|im_end|>`), so no separate tokenizer
files are needed.

## Run on Android

> **Update (July 2026):** [Google AI Edge Gallery](https://github.com/google-ai-edge/gallery) **v1.0.16+** can import litert-lm models **directly from Hugging Face** inside the app (tap **+**) — no computer or `adb` needed. The manual steps below are only required on older builds or for sideloading a local file.

The easiest way to try this on a phone is the official
**[Google AI Edge Gallery](https://github.com/google-ai-edge/gallery)** app:

1. Install a **recent** Gallery (package `com.google.ai.edge.gallery`, APK from the repo's
   [releases](https://github.com/google-ai-edge/gallery/releases) — 1.0.15+ supports `.litertlm`).
2. Download `model.litertlm` and push it:
   ```bash
   adb push model.litertlm /sdcard/Download/
   ```
3. In the app tap **+** (bottom-right), pick the file, choose the **GPU** backend.
4. Chat — the bundle already carries the tokenizer and Qwen3 chat template.

A 4B int4 build needs ~2.5 GB free RAM; reboot the phone first if memory is tight.

## Run on iPhone

Verified on **iPhone 17 Pro** with the LiteRT-LM Swift runtime
([swift-litert-lm](https://github.com/google-ai-edge/LiteRT-LM)): both files load and
generate on-device (block 32 ~10 tok/s, block 128 ~14 tok/s). Note: this 4B's main
weights section is ~2.11 GiB for block 32 / ~1.94 GiB for block 128 — **both load on
iPhone 17 Pro**, so externalizing the embedder (below) is sufficient; no further size
reduction is required to fit iOS.

## Quality — GSM8K parity

Measured on GSM8K (n=100, greedy, 0-shot chain-of-thought, identical prompt and
answer-extraction for every row).

| Configuration | GSM8K |
|---|---|
| bf16 (reference) | 87.0% |
| **LiteRT int4 — block 32** | **88.0%** |
| LiteRT int4 — block 128 | 81.0% |

The block-32 build is at **full parity (−0 pt vs bf16)** — the OCTAV + blockwise-32
recipe leaves int4 indistinguishable from bf16 here. (FastContext is a tool-calling
agent, not a math model, so GSM8K is an *indirect* capability measure; the bf16-vs-int4
delta is nonetheless the correct test for "did int4 quantization degrade the model" —
and at block 32 it did not.) Passes the local quality gate 8/8 (no degeneracy).

> **Identity:** asked "what is your name?", the model answers "I am Qwen…". FastContext
> is fine-tuned from Qwen3-4B-Instruct and the SFT does not override the base identity —
> this is inherited from the base model (the bf16 original behaves identically), not a
> conversion artifact.

## Conversion

Converted with the **official** [`litert-torch`](https://github.com/google-ai-edge/litert-torch)
converter — FastContext is a standard `Qwen3ForCausalLM`, so it uses the existing Qwen3
path with no custom graph code. Recipe: **blockwise int4 + OCTAV** (INT4 weights, block 32
or 128, symmetric, OCTAV optimal-clipping) with embeddings kept at INT8, KV cache 4096.
Blockwise (not the tool's default *channelwise*) int4 is what preserves accuracy.

```python
from litert_torch.generative.export_hf.export import export
export(
    model="microsoft/FastContext-1.0-4B-SFT",
    output_dir="out",
    quantization_recipe="qwen3_int4_block32_octav.json",  # blockwise-32 int4 + OCTAV, int8 embeddings
    cache_length=4096,
    externalize_embedder=True,  # embedding → its own section (dedups tied matrix)
)
```

`externalize_embedder=True` writes the (tied) embedding as its own `.litertlm` section
and dedups the tied matrix, shrinking the main weights section — the generic equivalent
of Gemma's per-layer-embedding mmap.

## License

MIT, inherited from the base model
[microsoft/FastContext-1.0-4B-SFT](https://huggingface.co/microsoft/FastContext-1.0-4B-SFT)
(itself built on Qwen3-4B-Instruct).
