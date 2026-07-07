---
license: mit
base_model: WeiboAI/VibeThinker-3B
tags:
  - litert
  - litert-lm
  - litertlm
  - on-device
  - edge
  - qwen2
  - reasoning
  - math
pipeline_tag: text-generation
library_name: litert-lm
---

# VibeThinker-3B — LiteRT-LM (blockwise int4)

[WeiboAI/VibeThinker-3B](https://huggingface.co/WeiboAI/VibeThinker-3B) converted to the
**LiteRT-LM** (`.litertlm`) format for on-device inference with Google's
[LiteRT-LM](https://github.com/google-ai-edge/litert-lm) runtime (the engine behind the official
`litert-community/*` models).

VibeThinker-3B is a dense 3B **math/reasoning model** (`Qwen2ForCausalLM`, 36 layers) — it solves
problems with an inline chain-of-thought and is strong at arithmetic and math word problems. Standard
Qwen2 architecture, so it rides the existing converter and runtime directly.

| | |
|---|---|
| **File** | `model.litertlm` — int4 **block 32** (~1.9 GB) |
| **Quantization** | int4 weights (symmetric) + **OCTAV** optimal-clipping; embeddings INT8 (externalized section) |
| **Compute** | integer |
| **Context (KV cache)** | 4096 |
| **Base model** | WeiboAI/VibeThinker-3B |
| **Decode speed** | iPhone 17 Pro (GPU) · ~87–93 tok/s (Mac M-series, GPU) |

## ⚠️ It's a reasoning model — give it room to think

VibeThinker solves with a step-by-step chain-of-thought, then a `\boxed{}` answer. **Run it with
`max_tokens` ≥ 2048** — at a short limit it gets cut off before the answer. (All quality numbers
below were measured at 2048.)

## Quality — GSM8K parity

Measured on GSM8K (n=100, greedy, 0-shot chain-of-thought, **max_tokens 2048**, identical prompt
and answer-extraction for every row).

| Configuration | GSM8K |
|---|---|
| bf16 (reference) | 97.0% |
| **LiteRT int4 — block 32** | **90.0%** (−7 pt) |

int4 (block 32) is at parity (−7 pt) and still 90% — strong for an on-device math model. bf16's 97%
reflects this model's math specialization.

**Why block 32 (not block 128)?** This is a precision-sensitive math model: the coarser block-128
int4 (¼ the dequant scales) collapsed to 64% (−33 pt) on GSM8K, while block 32 holds at 90%. So only
the block-32 build is published. (Note: for general-purpose 4B reasoning models the opposite holds —
block 128 is fine and faster — but exact arithmetic needs the finer block-32 grid.)

## Usage

```bash
# build litert-lm from https://github.com/google-ai-edge/litert-lm, then:
litert_lm_main \
  --model_path model.litertlm \
  --backend gpu \
  --input_prompt "A bat and a ball cost \$1.10. The bat costs \$1.00 more than the ball. How much is the ball?"
```

The `.litertlm` bundle carries the tokenizer and prompt template (Qwen2 ChatML —
`<|im_start|>role\n…<|im_end|>`), so no separate tokenizer files are needed.

## Run on Android

> **Update (July 2026):** [Google AI Edge Gallery](https://github.com/google-ai-edge/gallery) **v1.0.16+** can import litert-lm models **directly from Hugging Face** inside the app (tap **+**) — no computer or `adb` needed. The manual steps below are only required on older builds or for sideloading a local file.

The official **[Google AI Edge Gallery](https://github.com/google-ai-edge/gallery)** app runs
`.litertlm` models on-device:

1. Install a **recent** Gallery (package `com.google.ai.edge.gallery`, 1.0.15+ supports `.litertlm`).
2. Download `model.litertlm` and push it: `adb push model.litertlm /sdcard/Download/`
3. In the app tap **+**, pick the file, choose the **GPU** backend, and raise the max-tokens setting (≥2048).
4. Chat — the bundle already carries the tokenizer and Qwen2 chat template.

## Run on iPhone

Verified on **iPhone 17 Pro** (LiteRT-LM Swift runtime): the block-32 build (1.62 GiB section, under
the iOS limit) loads and generates correct answers.

## Conversion

Converted with the **official** [`litert-torch`](https://github.com/google-ai-edge/litert-torch)
converter — a standard `Qwen2ForCausalLM`, no custom graph code. Recipe: **blockwise-32 int4 + OCTAV**
(INT4 weights, block 32, symmetric, OCTAV optimal-clipping), embeddings INT8, KV cache 4096.

```python
from litert_torch.generative.export_hf.export import export
export(
    model="WeiboAI/VibeThinker-3B",
    output_dir="out",
    quantization_recipe="qwen3_int4_block32_octav.json",  # blockwise-32 int4 + OCTAV, int8 embeddings
    cache_length=4096,
    externalize_embedder=True,
)
```

## License

MIT, inherited from the base model [WeiboAI/VibeThinker-3B](https://huggingface.co/WeiboAI/VibeThinker-3B).
