---
license: apache-2.0
base_model: Nanbeige/Nanbeige4.2-3B
pipeline_tag: text-generation
tags:
  - litert-lm
  - on-device
  - reasoning
---

# Nanbeige4.2-3B — LiteRT-LM (.litertlm)

On-device build of [Nanbeige/Nanbeige4.2-3B](https://huggingface.co/Nanbeige/Nanbeige4.2-3B) for Google's [LiteRT-LM](https://github.com/google-ai-edge/litert-lm) runtime — an Apache-2.0 3B-class **looped-transformer** reasoning model (its 22 layers run twice per token, `num_loops=2`).

| | |
|---|---|
| **File** | `model.litertlm` (~2.4 GB) |
| **Quantization** | int4 weights — **blockwise (block 32) + OCTAV** optimal-clipping, symmetric; embedding INT8 |
| **Context (KV cache)** | 4096 (44 KV slots: 22 layers × 2 loops) |
| **Base model** | Nanbeige/Nanbeige4.2-3B (Apache-2.0) |
| **Decode speed** | ~3.2 tok/s (iPhone 17 Pro, CPU; loads ~4.4 s, ~1.8–2.1 GB peak) · ~10 tok/s (Mac M-series, CPU engine) |

Note on speed: the loop makes this model **~2× the per-token compute of a normal 3B** (that's where its quality comes from) — expect roughly 8B-class decode rates on CPU.

## What's in the box

- `model.litertlm` — int4 (blockwise-32, OCTAV clipping) + int8 embedding, 4096-token KV cache, ~2.4 GB.
- The loop is unrolled at export: 44 layer executions over 22 shared-weight layers, one KV-cache slot per (loop, layer) — 44 slots. Weights are stored once; only compute and KV double.
- The chat template auto-opens the model's `<think>` block (this is how the official template works — the model starts already inside its reasoning).

## Quality (GSM8K, n=50, 0-shot CoT, official sampling)

| build | GSM8K |
|---|---|
| bf16 (transformers 4.51, reference) | 94.0% (47/50) |
| this int4 .litertlm (LiteRT-LM CPU) | 90.0% (45/50) |

Local gate: 8/8 basic-quality questions, clean `<think>…</think>` closure, ChatML stop.

## ⚠ Usage notes

- **Sampling is required.** The model collapses under greedy decoding (official recipe: `temperature 0.6, top_k 20, top_p 0.95` — see its `generation_config.json`). Configure the runtime session accordingly; do not run at temperature 0.
- **CPU backend.** The current GPU delegate produces incorrect output on this 44-layer unrolled graph; run on CPU.
- Reasoning outputs open inside `<think>`; the visible answer follows `</think>`.

## Run it

Android: push the file and import it in [Google AI Edge Gallery](https://github.com/google-ai-edge/gallery) (`+` → import, keep CPU).
CLI / other platforms: see the LiteRT-LM repo for `litert-lm` runtime usage.

## Reproduce

Converted with [hf-to-litertlm](https://github.com/john-rocky/hf-to-litertlm):

```bash
bash scripts/reproduce_llm.sh nanbeige4.2-3b
```

The conversion script (`scripts/convert_nanbeige42.py`) documents the two traps this model hits: transformers-5.x zeroes the modeling's init-computed rotary `inv_freq` buffer (rope must be recomputed from config), and the looped architecture needs a 44-slot KV cache registered through the export pipeline's `cache_implementation` hook.
