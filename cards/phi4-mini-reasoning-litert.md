---
license: mit
base_model: microsoft/Phi-4-mini-reasoning
tags:
  - litert
  - litert-lm
  - litertlm
  - on-device
  - edge
  - phi
  - reasoning
  - math
pipeline_tag: text-generation
library_name: litert-lm
---

# Phi-4-mini-reasoning — LiteRT-LM (blockwise int4)

[microsoft/Phi-4-mini-reasoning](https://huggingface.co/microsoft/Phi-4-mini-reasoning) converted to
the **LiteRT-LM** (`.litertlm`) format for on-device inference with Google's
[LiteRT-LM](https://github.com/google-ai-edge/litert-lm) runtime (the engine behind the official
`litert-community/*` models).

Phi-4-mini-reasoning is a dense 3.8B **math/reasoning model** from Microsoft (implemented as
`Phi3ForCausalLM`, 32 layers) — it solves problems with a `<think>…</think>` chain-of-thought, then
the answer.

| | |
|---|---|
| **File** | `model.litertlm` — int4 **block 32** (~2.6 GB) |
| **Quantization** | int4 weights (symmetric) + **OCTAV** optimal-clipping; embeddings INT8 (externalized section) |
| **Compute** | integer |
| **Context (KV cache)** | 4096 |
| **Base model** | microsoft/Phi-4-mini-reasoning |
| **Decode speed** | ~84 tok/s (Mac M-series, GPU) |

## ⚠️ It's a reasoning model — give it room to think

This model emits a `<think>…</think>` chain-of-thought, then a `\boxed{}` answer. **Run it with
`max_tokens` ≥ 2048** — at a short limit it gets cut off before the answer. (All quality numbers below
were measured at 2048.)

## Quality — GSM8K parity

Measured on GSM8K (n=100, greedy, 0-shot chain-of-thought, **max_tokens 2048**, identical prompt and
answer-extraction for every row).

| Configuration | GSM8K |
|---|---|
| bf16 (reference) | 89.0% |
| **LiteRT int4 — block 32** | **81.0%** (−8 pt) |

int4 (block 32) is at parity (−8 pt). **Why block 32 (not block 128)?** This is a precision-sensitive
math model: the coarser block-128 int4 dropped to 74% (−15 pt) and degenerated on some prompts, while
block 32 holds at 81%. So only the block-32 build is published.

## Usage

```bash
# build litert-lm from https://github.com/google-ai-edge/litert-lm, then:
litert_lm_main \
  --model_path model.litertlm \
  --backend gpu \
  --input_prompt "A bat and a ball cost \$1.10. The bat costs \$1.00 more than the ball. How much is the ball?"
```

The `.litertlm` bundle carries the tokenizer and prompt template (Phi format — `<|user|>…<|end|><|assistant|>`),
so no separate tokenizer files are needed.

## Run on Android

> **Update (July 2026):** [Google AI Edge Gallery](https://github.com/google-ai-edge/gallery) **v1.0.16+** can import litert-lm models **directly from Hugging Face** inside the app (tap **+**) — no computer or `adb` needed. The manual steps below are only required on older builds or for sideloading a local file.

The official **[Google AI Edge Gallery](https://github.com/google-ai-edge/gallery)** app runs
`.litertlm` models on-device:

1. Install a **recent** Gallery (package `com.google.ai.edge.gallery`, 1.0.15+ supports `.litertlm`).
2. Download `model.litertlm` and push it: `adb push model.litertlm /sdcard/Download/`
3. In the app tap **+**, pick the file, choose the **GPU** backend, and raise the max-tokens setting (≥2048).

## Run on iPhone

Verified on **iPhone 17 Pro** (LiteRT-LM Swift runtime): loads and generates correct answers. This is a
~2.6 GB bundle (Phi's 200K-token vocab makes a large externalized embedder), so it sits near the iOS
memory ceiling — if you hit *"embedding lookup model is not initialized"* (a low-memory symptom), reboot
the phone to free RAM and reload.

## Conversion

Converted with the **official** [`litert-torch`](https://github.com/google-ai-edge/litert-torch)
converter. Phi-4-mini uses the `Phi3ForCausalLM` arch with **LongRoPE** + a (nominal) sliding window;
two export-time adjustments are needed for current litert-torch:
1. **LongRoPE:** replace `Phi3RotaryEmbedding.forward` with a static version (the `@dynamic_rope_update`
   seq-len branch is data-dependent under torch.export; for cache ≤ original_max=4096 the short factor
   is always correct).
2. **Sliding window:** set `config.sliding_window=None` (it is 262144 ≫ context, i.e. full-causal) so
   the standard causal mask path is used.

Recipe: **blockwise-32 int4 + OCTAV**, embeddings INT8, KV cache 4096, `externalize_embedder=True`.

## License

MIT, inherited from the base model
[microsoft/Phi-4-mini-reasoning](https://huggingface.co/microsoft/Phi-4-mini-reasoning).
