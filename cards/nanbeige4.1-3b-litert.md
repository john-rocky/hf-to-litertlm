---
license: apache-2.0
base_model: Nanbeige/Nanbeige4.1-3B
tags:
  - litert
  - litert-lm
  - litertlm
  - on-device
  - edge
  - reasoning
pipeline_tag: text-generation
library_name: litert-lm
---

# Nanbeige4.1-3B — LiteRT-LM (blockwise int4)

[Nanbeige/Nanbeige4.1-3B](https://huggingface.co/Nanbeige/Nanbeige4.1-3B) converted to the
**LiteRT-LM** (`.litertlm`) format for on-device inference with Google's
[LiteRT-LM](https://github.com/google-ai-edge/litert-lm) runtime (the engine behind the
official `litert-community/*` models).

Nanbeige4.1-3B is a fresh (Dec 2025) **phone-size reasoning** model on a plain dense
Llama architecture (Apache-2.0), reported to be competitive with much larger models. It
works the problem inside a `<think>…</think>` block before giving the final answer.

| | |
|---|---|
| **File** | `model.litertlm` (~2.2 GB; embedding externalized so every section is <2 GiB → loads on iOS) |
| **Quantization** | int4 weights — **blockwise (block 32) + OCTAV** optimal-clipping, symmetric; embedding INT8 |
| **Compute** | integer |
| **Context (KV cache)** | 4096 |
| **Base model** | Nanbeige/Nanbeige4.1-3B (Apache-2.0) |
| **Decode speed** | ~89 tok/s (Mac M4 Max, Metal GPU, greedy) |

## Usage

```bash
# build litert-lm from https://github.com/google-ai-edge/litert-lm, then:
litert_lm_main \
  --model_path model.litertlm \
  --backend gpu \
  --input_prompt "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts total?"
```

The `.litertlm` bundle carries the tokenizer and the prompt template (ChatML —
`<|im_start|>role\n … <|im_end|>`), so no separate tokenizer files are needed. This is a
**reasoning** model: it emits a `<think>…</think>` chain then the final answer (best
evaluated with a generous token budget), and stops cleanly at `<|im_end|>`.

## Run on Android

Install a recent [Google AI Edge Gallery](https://github.com/google-ai-edge/gallery)
(1.0.16+ can import `.litertlm` directly from Hugging Face), download `model.litertlm`
(or import this repo in-app), pick the **GPU** backend (CPU also works), and chat. Give it
a high max-tokens — it's a reasoning model with long chains of thought.

## Quality — GSM8K

Measured on GSM8K (n=50, greedy, 0-shot chain-of-thought, **max-tokens 2048** — a
reasoning model needs the budget to finish; scoring it at 512 tokens falsely penalises it):

| Configuration | GSM8K |
|---|---|
| **This model — LiteRT int4 (block32 + OCTAV)** | **84.0%** |

84% is a strong on-device GSM8K for a 3B, non-degenerate; the model also passes the local
quality gate **8/8** with a clean stop at `<|im_end|>`. Blockwise-32 + OCTAV optimal-clipping
(data-free) preserves the accuracy versus a naive min-max int4.

## Conversion

Converted with [`litert-torch`](https://github.com/google-ai-edge/litert) using a
**blockwise int4** recipe (INT4 weights, block size 32, symmetric, OCTAV optimal-clipping)
with the embedding at INT8, KV cache 4096, and a ChatML prompt template. Nanbeige4.1 is a
standard dense `LlamaForCausalLM`, so it rides the existing converter and runtime with no
custom graph code.

**`externalize_embedder=True` (required for iPhone).** The large 166k-token vocab makes the
weights a >2 GiB single TFLite section, which exceeds the ~2 GiB single-section `mmap` limit
on iOS. Externalizing the embedding drops the main section under 2 GiB so the model loads on
**iPhone (Metal GPU)** as well as Android/desktop. Same weights, so GSM8K is unchanged.

**Added-tokens tokenizer fix.** Nanbeige's 10 special tokens (`<|im_start|>`, `<|im_end|>`,
`<think>`, `</think>`, `<tool_call>`, …) live at vocab ids 166100–166109, above the base
SentencePiece vocab (166100). The base SP conversion drops them, so the reasoning model would
generate `<think>` (id 166103) and the runtime would crash with *"Token id out of range."* The
converted tokenizer here appends those added tokens as USER_DEFINED SentencePiece pieces at
their exact ids (padded to the model vocab), so `<think>` and friends decode correctly.

## License

Apache-2.0, inherited from the base model
[Nanbeige/Nanbeige4.1-3B](https://huggingface.co/Nanbeige/Nanbeige4.1-3B).
