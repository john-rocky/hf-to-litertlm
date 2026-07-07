---
license: apache-2.0
base_model: POLARIS-Project/Polaris-4B-Preview
tags:
  - litert
  - litert-lm
  - litertlm
  - on-device
  - edge
  - reasoning
  - math
pipeline_tag: text-generation
library_name: litert-lm
---

# Polaris-4B-Preview — LiteRT-LM (blockwise int4)

[POLARIS-Project/Polaris-4B-Preview](https://huggingface.co/POLARIS-Project/Polaris-4B-Preview)
converted to the **LiteRT-LM** (`.litertlm`) format for on-device inference with Google's
[LiteRT-LM](https://github.com/google-ai-edge/litert-lm) runtime (the engine behind the
official `litert-community/*` models).

Polaris-4B is an **RL post-trained reasoning model** built on Qwen3-4B (standard dense
`qwen3`, Apache-2.0). It is tuned for hard competition math and works the problem inside a
`<think>…</think>` chain before answering — a **SOTA-for-size math reasoner** that runs
fully on a phone.

| | |
|---|---|
| **File** | `model.litertlm` (~2.3 GB; embedding externalized so every section is <2 GiB → loads on iOS) |
| **Quantization** | int4 weights — **blockwise (block 128) + OCTAV** optimal-clipping, symmetric; embedding INT8 |
| **Compute** | integer |
| **Context (KV cache)** | 4096 |
| **Base model** | POLARIS-Project/Polaris-4B-Preview (Apache-2.0) |
| **Decode speed** | ~69 tok/s (Mac M4 Max, Metal GPU, greedy) |

## What it's good at — hard math (AIME)

Polaris-4B's headline is competition math. Per the base model card, at **~4B params** it
reports **AIME24 81.2 / AIME25 79.4**, in the range of far larger frontier reasoners. It is
optimized for long-chain hard-problem reasoning rather than grade-school arithmetic — give
it a **generous token budget** (it thinks at length).

## Usage

```bash
litert_lm_main \
  --model_path model.litertlm \
  --backend gpu \
  --input_prompt "Find the number of ordered pairs (a,b) of integers with 1<=a,b<=100 such that a*b is a perfect square."
```

The `.litertlm` bundle carries the tokenizer and a ChatML prompt template
(`<|im_start|>role\n … <|im_end|>`). It emits a `<think>…</think>` chain then the final
answer, and stops cleanly at `<|im_end|>`. **Set a high max-tokens** (≥2048) — a reasoning
model truncated mid-thought produces no answer.

## Run on Android

Install a recent [Google AI Edge Gallery](https://github.com/google-ai-edge/gallery)
(1.0.16+ imports `.litertlm` directly from Hugging Face), import this repo (or push
`model.litertlm`), pick the **GPU** backend, and chat. It's a ~2.3 GB / 4B model — GPU
needs a ~12 GB+ device; free RAM first on smaller phones.

## Quality — GSM8K (on-device int4 parity)

Measured on GSM8K (n=50, greedy, 0-shot chain-of-thought, **max-tokens 2048**):

| Configuration | GSM8K |
|---|---|
| **This model — LiteRT int4 (block128 + OCTAV)** | **82.0%** |

Non-degenerate, passes the local quality gate **8/8** with a clean stop at `<|im_end|>`.
GSM8K undersells this model — it is tuned for AIME-level problems, and on easy arithmetic its
long exploratory reasoning is not where its edge shows. **block128** is used (rather than
block32) because a 4B reasoning model's block32 weights can corrupt on the iPhone Metal GPU;
block128 loads and runs stably across iPhone / Android / desktop.

## Conversion

Converted with [`litert-torch`](https://github.com/google-ai-edge/litert): blockwise int4
(block 128) + OCTAV optimal-clipping, embedding INT8, KV cache 4096, ChatML template.
Polaris-4B is a standard dense `Qwen3ForCausalLM` (with `rope_scaling: yarn`, exported with a
cache within `original_max_position_embeddings` so base RoPE is exact), so it rides the
existing Qwen3 converter with no custom graph code. `externalize_embedder=True` keeps every
`.litertlm` section under the iOS ~2 GiB single-section `mmap` limit so it loads on iPhone.

## License

Apache-2.0, inherited from the base model
[POLARIS-Project/Polaris-4B-Preview](https://huggingface.co/POLARIS-Project/Polaris-4B-Preview).
