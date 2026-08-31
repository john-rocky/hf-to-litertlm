---
license: apache-2.0
base_model: ibm-granite/granite-4.2-3b
tags:
  - litert
  - litert-lm
  - litertlm
  - on-device
  - edge
  - granite
  - reasoning
pipeline_tag: text-generation
library_name: litert-lm
---

# granite-4.2-3b — LiteRT-LM

[ibm-granite/granite-4.2-3b](https://huggingface.co/ibm-granite/granite-4.2-3b) converted to the **LiteRT-LM** (`.litertlm`) format for on-device inference with Google's [LiteRT-LM](https://github.com/google-ai-edge/litert-lm) runtime. **Requires litert-lm ≥ 0.16.**

Granite 4.2-3B is IBM's compact **reasoning** model (3.66B dense, 40 layers, hidden 2560, GQA 40:8, untied embeddings, 100k vocab), released August 2026. It works problems inside `<think>…</think>` before answering, and these bundles carry that machinery: the generation prompt pre-fills the think opener the way IBM's own chat template does, and a `thought` channel is declared in the bundle metadata so the runtime separates reasoning from the answer and honors a thinking budget.

| File | Recipe | Size |
|---|---|---|
| `granite-4.2-3b_int4.litertlm` | int4 blockwise-32 + OCTAV on linears, int8 embedding | 2.19 GB |
| `granite-4.2-3b_int8.litertlm` | int8 dynamic on linears + embedding | 3.76 GB |

The **int4 file is the one to use on a phone**: on Apple hardware it is smaller *and* decodes faster than int8 (see below). int8 is the desktop / quality-reference build.

## Correctness

Both files score **8/8 on an 8-question sanity gate on both CPU and GPU** (Apple M4 Max, litert-lm 0.16.0 runtime lineage), with every answer arriving as a clean final answer through the thought channel — no reasoning leakage, no degeneration. The bf16 PyTorch reference scores 8/8 on the same questions under the same template. Phone-side results (iPhone / Android) will be added to this card as they are measured; until then the table below is the measured record.

## Usage

```bash
litert-lm run ./granite-4.2-3b_int4.litertlm --prompt "What is the capital of France? Answer in one word."

# GPU
litert-lm run ./granite-4.2-3b_int4.litertlm --backend gpu --cache no --prompt "..."
```

Notes for a reasoning model:

- **Give it a generous output budget (≥ 2048 tokens).** The model thinks before it answers; truncated mid-thought it produces no final answer at all.
- The bundle declares the `thought` channel (`<think>`…`</think>`), so runtimes that expose `ThinkingConfig` / a thinking budget can cap or read the reasoning separately; the streamed answer contains only the final response.
- The bundle uses ChatML role markers (`<|im_start|>…<|im_end|>`) with a 4096-token KV budget and six prefill signatures (1024, 256, 64, 16, 4, 1) — six rather than eleven because every exported signature is charged engine memory whether or not it is called, a lesson measured on this model family's 4.1 ship, where an eleven-signature build was killed by iOS during Metal engine initialisation.

## Performance

`litert-lm benchmark` (litert-lm 0.16.0), Apple M4 Max, `-p 256 -d 256 --runs 3 --cache no --max-num-tokens 1024`, quiet machine, serialized, ≥300 s rest before each GPU reading:

| File | Backend | Prefill (256) | Decode | TTFT | Init |
|---|---|---|---|---|---|
| int4 | **GPU (Metal)** | **1240 tok/s** | **85.5 tok/s** | 0.24 s | 4.4 s |
| int4 | CPU | 106 tok/s | 21.5 tok/s | 2.94 s | 6.9 s |
| int8 | GPU (Metal) | 1208 tok/s | 70.7 tok/s | 0.24 s | 3.5 s |
| int8 | CPU | 264 tok/s | 21.1 tok/s | 2.61 s | 23.0 s |

Two patterns worth knowing, both consistent with the granite-4.1-3b conversion of the same shape: **int4 beats int8 on the GPU** (+21% decode on top of the 1.7× size reduction), while **int8 wins CPU prefill** (2.5×) — XNNPACK's large matmuls outrun the blockwise dequant.

## Conversion notes

Converted with [`litert-torch`](https://github.com/google-ai-edge/litert-torch) 0.9.3 / litert-converter 0.4.0 / ai-edge-quantizer 0.9.0 / litert-lm-builder 0.16.1, from a pristine released stack — no patched checkout (reproduction script: [hf-to-litertlm](https://github.com/john-rocky/hf-to-litertlm)).

- **The think opener is pre-filled by the bundle's prompt template.** IBM's chat template ends the generation prompt with `<|im_start|>assistant\n<think>\n`, and the bundle's assistant prefix reproduces that exactly. This matters at export time: the converter derives the assistant prefix from an assistant *history* message, so a template that only opens `<think>` in its generation branch silently ships without the scaffold — the reasoning discipline is then left to the quantized model's own choice.
- **A `thought` channel is declared in the metadata** (`<think>` / `</think>`). Without it the runtime streams raw reasoning into the answer and silently ignores any thinking budget. The exporter does not emit it on this path; it was added post-export with the weights verified byte-identical.
- **No start token in the metadata.** granite-4.2's tokenizer declares `<s>` as BOS but never prepends it (the tokenizer's post-processor adds nothing), so the converter's unconditional `start_token` write was suppressed. The 4.1 lesson, different mechanism: a prepended token the model never saw in that position degrades answers.
- **Embedder externalised** so the 100352×2560 input-embedding table sits in its own section, clear of the ~2 GiB single-section mmap ceiling on iOS. (4.2's embeddings are untied; the lm_head stays in the main graph.)
- **Tokenizer embedded as the upstream `tokenizer.json`** (HF tokenizer section), not a SentencePiece conversion — byte-level BPE survives intact.
- **Quantization**: int4 is blockwise-32 with OCTAV clipping on the linears and int8 on the embedding; int8 is dynamic per-channel on linears and embedding.

## License and changes

Distributed under **Apache-2.0** (inherited from the base model). **Changes from the original work:** weights converted from safetensors bf16 to LiteRT flatbuffers and quantized as described above; tokenizer and chat template repackaged into the `.litertlm` bundle (thinking pre-fill and thought-channel metadata as described). This repository is a community conversion and is not affiliated with IBM.
