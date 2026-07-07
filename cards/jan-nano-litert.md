---
license: apache-2.0
base_model: Menlo/Jan-nano
tags:
  - litert
  - litert-lm
  - litertlm
  - on-device
  - edge
  - qwen3
  - agent
  - reasoning
  - deep-research
pipeline_tag: text-generation
library_name: litert-lm
---

# Jan-nano — LiteRT-LM (blockwise int4)

[Menlo/Jan-nano](https://huggingface.co/Menlo/Jan-nano) converted to the **LiteRT-LM**
(`.litertlm`) format for on-device inference with Google's
[LiteRT-LM](https://github.com/google-ai-edge/litert-lm) runtime (the engine behind the
official `litert-community/*` models).

Jan-nano is a 4B **deep-research agent** fine-tuned from **Qwen3-4B** (`Qwen3ForCausalLM`)
with a multi-stage RLVR recipe, optimized for tool use via the Model Context Protocol (MCP).
It is a **reasoning model** — it emits a `<think>…</think>` chain before its answer — so it
rides the existing Qwen3 converter and runtime directly.

| | |
|---|---|
| **Files** | `model.litertlm` — int4 **block 128** (recommended, on-device) · `model_block32.litertlm` — int4 **block 32** (finer-grain, desktop/Android) |
| **Quantization** | int4 weights (symmetric) + **OCTAV** optimal-clipping; embeddings INT8 (externalized section) |
| **Compute** | integer |
| **Context (KV cache)** | 4096 |
| **Base model** | Menlo/Jan-nano (Qwen3-4B) |

## ⚠️ It's a reasoning model — give it room to think

Jan-nano generates a `<think>…</think>` reasoning chain, then the answer. **Run it with
`max_tokens` ≥ 2048** — at a short limit it gets cut off mid-thought and never reaches the
answer. (All quality numbers below were measured at 2048.)

## Which file?

| File | int4 granularity | GSM8K (max_tokens 2048) | iPhone 17 Pro | Mac (M-series, GPU) |
|---|---|---|---|---|
| **`model.litertlm`** | block 128 | **88.0%** | **~14 tok/s, loads** | ~67 tok/s |
| `model_block32.litertlm` | block 32 | 85.0% | 2.11 GiB section — near the iOS memory ceiling, may not load | ~67 tok/s |

**Use `model.litertlm` (block 128)** — for a reasoning model that emits long `<think>` chains,
faster decode matters, and block 128 (¼ the scales → lighter GPU dequant) is ~40% faster while
matching block 32 on accuracy here. It is also the build that loads reliably on iPhone (the
block-32 build's larger section sits at the device memory edge). block 32 is provided for
desktop/Android where the extra granularity is free.

## Quality — GSM8K parity

Measured on GSM8K (n=100, greedy, 0-shot chain-of-thought, **max_tokens 2048**, identical
prompt and answer-extraction for every row).

| Configuration | GSM8K |
|---|---|
| bf16 (reference) | 92.0% |
| **LiteRT int4 — block 128** | **88.0%** (−4 pt) |
| LiteRT int4 — block 32 | 85.0% (−7 pt) |

int4 is at parity (−4 pt for the recommended block-128 build). Note: evaluating a reasoning
model at a short token budget badly understates int4 — at `max_tokens 1024` the same block-32
build scored only 63% purely because the longer int4 reasoning chains were truncated before the
answer; at 2048 it recovers to 85%. Always benchmark reasoning models with enough headroom.

## Usage

```bash
# build litert-lm from https://github.com/google-ai-edge/litert-lm, then:
litert_lm_main \
  --model_path model.litertlm \
  --backend gpu \
  --input_prompt "Plan how to find where HTTP retries are configured in a Python repo."
```

The `.litertlm` bundle carries the tokenizer and prompt template (Qwen3 ChatML —
`<|im_start|>role\n…<|im_end|>`, stop token `<|im_end|>`), so no separate tokenizer files are
needed. The model will produce a `<think>…</think>` block followed by its answer.

## Run on Android

> **Update (July 2026):** [Google AI Edge Gallery](https://github.com/google-ai-edge/gallery) **v1.0.16+** can import litert-lm models **directly from Hugging Face** inside the app (tap **+**) — no computer or `adb` needed. The manual steps below are only required on older builds or for sideloading a local file.

The official **[Google AI Edge Gallery](https://github.com/google-ai-edge/gallery)** app runs
`.litertlm` models on-device:

1. Install a **recent** Gallery (package `com.google.ai.edge.gallery`, 1.0.15+ supports `.litertlm`).
2. Download `model.litertlm` and push it: `adb push model.litertlm /sdcard/Download/`
3. In the app tap **+**, pick the file, choose the **GPU** backend, and raise the max-tokens setting.
4. Chat — the bundle already carries the tokenizer and Qwen3 chat template.

A 4B int4 build needs ~2.5 GB free RAM; reboot the phone first if memory is tight.

## Run on iPhone

Verified on **iPhone 17 Pro** (LiteRT-LM Swift runtime): `model.litertlm` (block 128, 1.94 GiB
section) loads and generates at ~14 tok/s. The block-32 build's section (2.11 GiB) sits at the
device memory ceiling and may fail to load — prefer block 128 on iPhone.

## Conversion

Converted with the **official** [`litert-torch`](https://github.com/google-ai-edge/litert-torch)
converter — Jan-nano is a standard `Qwen3ForCausalLM`, so it uses the existing Qwen3 path with
no custom graph code. Recipe: **blockwise int4 + OCTAV** (INT4 weights, block 128 or 32,
symmetric, OCTAV optimal-clipping), embeddings INT8, KV cache 4096.

```python
from litert_torch.generative.export_hf.export import export
export(
    model="Menlo/Jan-nano",
    output_dir="out",
    quantization_recipe="qwen3_int4_block128_octav.json",  # blockwise-128 int4 + OCTAV, int8 embeddings
    cache_length=4096,
    externalize_embedder=True,
)
```

## License

Apache-2.0, inherited from the base model [Menlo/Jan-nano](https://huggingface.co/Menlo/Jan-nano)
(itself fine-tuned from [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B), also Apache-2.0).
