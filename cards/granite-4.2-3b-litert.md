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

Both files score **8/8 on an 8-question sanity gate on both CPU and GPU** (Apple M4 Max, litert-lm 0.16.0 runtime lineage), with every answer arriving as a clean final answer through the thought channel — no reasoning leakage, no degeneration. The bf16 PyTorch reference scores 8/8 on the same questions under the same template.

On a **Galaxy S26 (SM-S942Q, Snapdragon SM8850, Adreno)** both files generate correctly on GPU and CPU, with **full OpenCL delegation — 1783/1783 nodes on every prefill signature and decode, zero rejected ops** — and the runtime separates the reasoning on-device exactly as intended: the answer arrives as a labeled thought block followed by the clean final answer.

On an **iPhone 17 Pro** the int4 file passes the same gate on both backends (on-device byte count verified against the source file): **Metal GPU 7/8** (engine created at the bundle's full 4096-token budget, init 13.0 s; the one miss is the rhyme line at the end of the composite 8-question prompt — the model answers that question correctly when asked on its own, on every other platform) and **CPU 8/8** (init 5.5 s). Available process memory never dropped below ~4.2 GB during either leg.

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

Galaxy S26 (SM-S942Q, Snapdragon SM8850, Adreno; `litert_lm_advanced_main` from the litert-lm v0.16.0 release kit, 205-token prompt with `--benchmark`, 2 runs per cell, ranges shown; a reasoning model decodes its own full response, so decode-turn lengths vary):

| File | Backend | Prefill (205) | Decode | TTFT | Init |
|---|---|---|---|---|---|
| int4 | **GPU (OpenCL)** | **235–241 tok/s** | **12.3–15.6 tok/s** | 0.93–0.94 s | 13.6–15.7 s |
| int4 | CPU | 22–42 tok/s | 9.5–9.8 tok/s | 5.0–9.3 s | 3.8–8.4 s |
| int8 | GPU (OpenCL) | 239–241 tok/s | 9.2 tok/s | 0.96–0.97 s | 5.4–5.8 s |
| int8 | CPU | 51–95 tok/s | 7.0 tok/s | 2.3–4.2 s | 0.6 s |

The int4 GPU decode range's low end is the second back-to-back run (thermal); its high end is the rested reading. The shape matches the other platforms: **int4 wins GPU decode 1.7× over int8**, GPU wins TTFT ~5–10× over CPU, and peak process RSS stayed at 1.4 GB (int4 GPU) / 4.3 GB (int8 CPU) with no OOM on the 12 GB device.

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
