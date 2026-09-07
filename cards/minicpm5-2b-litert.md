---
license: apache-2.0
base_model: openbmb/MiniCPM5-2B
base_model_relation: quantized
tags:
  - litert
  - litert-lm
  - litertlm
  - on-device
  - edge
  - minicpm
  - minicpm5
  - reasoning
pipeline_tag: text-generation
library_name: litert-lm
---

# MiniCPM5-2B — LiteRT-LM

[openbmb/MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B) converted to the **LiteRT-LM** (`.litertlm`) format for on-device inference with Google's [LiteRT-LM](https://github.com/google-ai-edge/litert-lm) runtime. **Requires litert-lm ≥ 0.16** (thought channel + `ThinkingConfig`); measured here on 0.17.0.

MiniCPM5-2B is OpenBMB's 2.5B-parameter dense **hybrid-reasoning** model (42 layers, hidden 2048, GQA 16:2, untied 130k-vocab embeddings, 131k-token native context, released September 2026): one checkpoint answers directly or works the problem inside `<think>…</think>` first, chosen by an `enable_thinking` switch in its chat template. These bundles carry that machinery unchanged: the model's own `chat_template.jinja` verbatim, a declared `thought` channel, and the `enable_thinking` knob reachable from the runtime.

| File | Recipe | Size |
|---|---|---|
| `MiniCPM5-2B_int4.litertlm` | int4 blockwise-32 + OCTAV on linears, int8 embedding | 1.55 GB |
| `MiniCPM5-2B_int8.litertlm` | int8 dynamic on linears + embedding; **fp32 activations declared** (see notes) | 2.60 GB |

The **int4 file is the phone file** (smaller, fastest GPU decode on every platform measured) — best used for direct answers or short reasoning; see the thinking-mode note below. **int8 is the file for reasoning that has to complete**: its thinking chains are ~3–4× shorter than int4's on the same questions and terminate where int4 runs into the token budget. int8's main weight section is 2.33 GB, above the single-section mmap ceiling of default-entitlement iOS apps, so it is a desktop / Android build.

## Correctness

Both files score **8/8 on an 8-question sanity gate on both CPU and GPU** (Apple M4 Max), with the reasoning arriving on the thought channel and only the final answer in the streamed text. The bf16 PyTorch model scores 8/8 on the same gate with thinking on or left to the model, and **6/8 with thinking forced off** ("opposite of hot" → "Cool.", the rhyme line → "Green.") — read any thinking-off result against that, not against 8/8.

**GSM8K** (first 100 test questions, greedy, 0-shot chain-of-thought prompt, **thinking off** — the protocol OpenBMB's own MiniCPM5 cards use, max 2048 new tokens, identical prompt and extraction on every row):

| Configuration | GSM8K |
|---|---|
| bf16 PyTorch (MPS), upstream template | **92 %** |
| int8, CPU | **91 %** |
| int4, CPU | 86 % |
| int4, GPU (Metal) | 87 % |

int8 is at parity (7 of its 9 misses are the bf16 model's own). int4 costs about five points on this 42-layer model; the GPU's default fp16 activations cost nothing measurable in no-think mode.

**Thinking mode is where int4 shows its damage.** On ten GSM8K questions with thinking on and a 3584-token budget, the bf16 model closes its reasoning on 9/10 with ~3,000-character chains and int8 reproduces that question for question (9/10, median ~3,200 characters on CPU), while **int4 closes 0/10 on CPU** (median ~13,700 characters — it keeps re-checking and runs into the budget; the answer is usually right inside the thought text but never gets emitted). On the Metal GPU with the runtime's default fp16 activations the same int4 file happened to close 10/10 with ~8,000-character chains, but that is fp16 rounding steering the trajectory, not a property to rely on. If your use needs the reasoning to finish, use int8 or turn thinking off on int4.

On a **Galaxy S26 (SM-S942Q, Snapdragon SM8850, Adreno)** both files generate correctly on GPU and CPU with **full OpenCL delegation — 1873/1873 nodes on every prefill signature and decode, zero rejected ops** — and the runtime separates the reasoning on-device (`[thought] … [/thought]`, then the answer).

On an **iPhone 17 Pro** the int4 file passes the same 8-question gate on both backends (on-device byte count verified against the source file): **Metal GPU 7/8** (init 5.7 s) and **CPU 7/8** (init 2.2 s); the one miss on each leg is the rhyme line inside the 8-question composite prompt, answered "green" — the same answer the bf16 model gives with thinking off, so a prompt-format artifact rather than conversion damage.

Multi-turn: three-turn conversations (introduce a name and a city, an arithmetic question, then "which city do I live in?") hold on both files under all three thinking modes with the name and city recalled — the template's history rendering stays consistent with what the runtime already streamed.

## Usage

```bash
pip install litert-lm   # or: uv tool install litert-lm
litert-lm run --from-huggingface-repo=mlboydaisuke/MiniCPM5-2B-LiteRT MiniCPM5-2B_int4.litertlm \
  --prompt "What is the capital of France? Answer in one word."

# local file, GPU
litert-lm run ./MiniCPM5-2B_int4.litertlm --backend gpu --cache no --prompt "..."
```

On Android the files import into the [Google AI Edge Gallery](https://github.com/google-ai-edge/gallery) app ("Import from HF" with this repo's file URL, or a local file).

Thinking is the model's **default**: with no `ThinkingConfig` it decides for itself and, in practice, reasons before every answer (100–700 characters on trivial questions, thousands on math). To control it:

- **Give it a generous output budget (≥ 2048 tokens; 4096 for math).** Truncated mid-thought it produces no final answer at all.
- The bundle declares the `thought` channel (`<think>` / `</think>`), so the streamed text contains only the answer and runtimes that expose `ThinkingConfig` can cap or read the reasoning separately.
- `enable_thinking=false` (via `ThinkingConfig` or the conversation's extra context — both reach the template) switches the model to **direct answers**: two- to seven-token replies on the gate questions, ~10× faster turns, and the GSM8K numbers above. `enable_thinking=true` pre-fills the think opener explicitly.
- Sampling: OpenBMB recommends `temperature 1.0, top_p 0.95`; the gates above are greedy.
- Prompt format is ChatML (`<|im_start|>role\n…<|im_end|>\n`), 4096-token KV budget, six prefill signatures (1024, 256, 64, 16, 4, 1).

## Performance

`litert-lm benchmark` (litert-lm 0.17.0), Apple M4 Max, `-p 256 -d 256 --runs 3 --cache no --max-num-tokens 1024`, quiet machine, serialized, ≥300 s rest before each GPU reading; each backend confirmed to generate real text before its number was recorded:

| File | Backend | Prefill (256) | Decode | TTFT | Init |
|---|---|---|---|---|---|
| int4 | **GPU (Metal)** | **1699 tok/s** | **92.8 tok/s** | 0.16 s | 3.7 s |
| int4 | CPU | 149 tok/s | 31.1 tok/s | 1.76 s | 4.5 s |
| int8 (fp32 activations) | GPU (Metal) | 1405 tok/s | 74.7 tok/s | 0.20 s | 3.0 s |
| int8 (fp32 activations) | CPU | 161 tok/s | 30.0 tok/s | 1.62 s | 15.0 s |

Galaxy S26 (SM-S942Q, Snapdragon SM8850, Adreno; `litert_lm_advanced_main` from the litert-lm v0.16.0 release kit, 205-token prompt with `--benchmark`, 2 runs per cell, ranges shown; a reasoning model decodes its own full response, so decode-turn lengths vary):

| File | Backend | Prefill (205) | Decode | TTFT | Init | Peak RSS |
|---|---|---|---|---|---|---|
| int4 | **GPU (OpenCL)** | **401–411 tok/s** | **16.1–18.6 tok/s** | 0.56 s | 11.2–13.1 s | 1.14 GB |
| int4 | CPU | 39–72 tok/s | 15.6–15.8 tok/s | 2.9–5.3 s | 3.1–5.7 s | 2.12 GB |
| int8 (fp32 activations) | GPU (OpenCL) | 150–160 tok/s | 10.9–12.8 tok/s | 1.4 s | 4.0–6.3 s | 1.10 GB |
| int8 (fp32 activations) | CPU | 103–157 tok/s | 11.7 tok/s | 1.4–2.1 s | 0.3 s | 2.90 GB |

GPU wins prefill (5–10× on the phone, 11× on the Mac) and time-to-first-token everywhere; on Adreno the int4 GPU decode edge over the same-device CPU is modest (~1.1×), while the int8 file's fp32 activations bring its GPU decode level with its CPU.

## Conversion notes

Converted with [`litert-torch`](https://github.com/google-ai-edge/litert-torch) 0.9.3 / litert-converter 0.4.0 / ai-edge-quantizer 0.9.0 / litert-lm-builder 0.16.1 from a pristine released stack (reproduction: [hf-to-litertlm](https://github.com/john-rocky/hf-to-litertlm), `bash scripts/reproduce_llm.sh minicpm5-2b`).

- **The chat template is the checkpoint's `chat_template.jinja`, byte for byte**, embedded on the runtime's jinja path — the same packaging as litert-community/MiniCPM5-1B. That is what keeps `enable_thinking` and the tool-calling format available to the app; the `thought` channel is declared alongside it.
- **Start token `<s>`** is correct for this family: the template's own `{{ bos_token }}` renders empty at runtime and the engine prepends the metadata start token, so the model sees exactly one `<s>` as it does upstream. Stops are the model's `</s>` and `<|im_end|>`.
- **int4 needed a zero-scale fix.** Decoder layer 0's MLP contains 13 all-zero rows; blockwise quantization emits a zero scale for each of their blocks, which the CPU (XNNPACK) path refuses to load while the GPU path silently accepts. The scales were replaced in place by a tiny positive epsilon (the quantized values in those blocks are zero, so the dequantized weights are unchanged) — 3,328 bytes of a 1.55 GB file.
- **int8 declares fp32 activations in-bundle.** With the runtime's default fp16 GPU activations, the int8 model's reasoning on one gate question ran 2000+ tokens without closing `</think>` (it fails the 8-question gate); with fp32 declared it closes in ~450 tokens and passes 8/8, at a cost of ~14 % GPU decode speed. On a ten-question thinking-on GSM8K subset the two dtypes were closer (fp16 finished 10/10, fp32 7/10), so this is a measured trade for the gate, not a cure. The int4 file keeps the fp16 default: it passes the gate there, and declaring fp32 only reproduces the CPU reference's non-terminating chains (see Correctness).
- **Embedder externalised** so the 130560×2048 input-embedding table sits in its own section; tokenizer embedded as the upstream `tokenizer.json` (byte-level BPE survives intact).
- **Quantization**: int4 is blockwise-32 with OCTAV clipping on the linears and int8 on the embedding; int8 is dynamic per-channel on linears and embedding. A block-128 int4 variant was built and rejected: it loses 8 GSM8K points (79 vs 87, thinking off), fails the 8-question gate on CPU, and does not shorten the thinking chains.

## Machine-readable manifest

`litertlm_manifest.json` in this repo describes both files (sha256, size, sections, context length, verified backends, the measured rows above with their conditions, and known issues) for tooling that picks a file per device.

## License and changes

Distributed under **Apache-2.0** (inherited from the base model). **Changes from the original work:** weights converted from safetensors bf16 to LiteRT flatbuffers and quantized as described above; 13 all-zero weight rows' quantization scales set to an epsilon (no numeric change); tokenizer and chat template repackaged into the `.litertlm` bundle with runtime metadata (thought channel, stop tokens, activation-dtype preference). No fine-tuning.
