---
license: mit
base_model: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
tags:
  - litert
  - litert-lm
  - litertlm
  - on-device
  - edge
  - reasoning
  - deepseek-r1
pipeline_tag: text-generation
library_name: litert-lm
---

# DeepSeek-R1-Distill-Qwen-7B — LiteRT-LM (blockwise int4)

[deepseek-ai/DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)
converted to the **LiteRT-LM** (`.litertlm`) format for on-device inference with
Google's [LiteRT-LM](https://github.com/google-ai-edge/litert-lm) runtime (the
engine behind the official `litert-community/*` models).

A **reasoning** model: it emits a `<think> … </think>` chain before the answer.
MIT-licensed (distilled onto an Apache-2.0 Qwen2.5 base). Converted with the
**official** upstream `litert-torch` — no fork, no custom code.

| | |
|---|---|
| **File** | `model.litertlm` (~4.2 GB) |
| **Quantization** | int4 weights — **blockwise (block 32) + OCTAV** optimal-clipping, symmetric; embedding INT8 |
| **Compute** | integer |
| **Context (KV cache)** | 4096 |
| **Base model** | deepseek-ai/DeepSeek-R1-Distill-Qwen-7B |
| **Decode speed** | ~67 tok/s (Mac M-series, LiteRT-LM, Metal GPU, greedy) |
| **Platforms** | Desktop (Mac) ✓ · high-RAM (12 GB+) Android ✓ · **iPhone / 8 GB phones ✗** (4 GB exceeds the budget) |

## Usage

```bash
litert_lm_main --model_path model.litertlm --backend gpu \
  --input_prompt "If a train travels 60 km in 45 minutes, what is its speed in km/h?"
```

The `.litertlm` bundle carries the tokenizer and the DeepSeek prompt template
(`<｜User｜>` / `<｜Assistant｜>`, stop token `<｜end▁of▁sentence｜>`). The assistant
opens a `<think>` block, reasons step by step, then gives the final answer
(commonly in `\boxed{}`).

## Quality — GSM8K parity

GSM8K (n=100, greedy, 0-shot, identical prompt + answer-extraction; `max_new_tokens=2048`
to fit the reasoning chain).

| Configuration | GSM8K |
|---|---|
| bf16 (reference) | 88.0% |
| **This model — LiteRT int4 (BOCTAV4)** | **87.0%** |

LiteRT int4 is **at parity — −1.0 pt** vs bf16. The reasoning behavior is fully
preserved through 4-bit quantization; the shallow-wide Qwen2 (28 layers) absorbs
int4 rounding cleanly.

## Conversion

Converted with the **official** upstream [`litert-torch`](https://github.com/google-ai-edge/litert)
`export_hf` (clean `git worktree` at `upstream/main`, dev-fork patches excluded).
`Qwen2ForCausalLM` rides the stock converter with no custom code. int4 recipe =
**blockwise (block 32) + OCTAV** with INT8 embedding (externalized into its own
bundle section); KV cache 4096.

## License

MIT (model weights), inherited from
[deepseek-ai/DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B);
the Qwen2.5 base is Apache-2.0. Commercial use and derivatives permitted.
