---
license: other
license_name: s1-mini-license
license_link: LICENSE
base_model: superwhisper/s1-mini
base_model_relation: quantized
tags:
  - litert
  - litert-lm
  - litertlm
  - on-device
  - edge
  - asr
  - automatic-speech-recognition
  - text-normalization
  - inverse-text-normalization
  - punctuation
  - truecasing
  - speech-to-text
  - dictation
  - post-processing
  - qwen3
language:
  - en
pipeline_tag: text-generation
library_name: litert-lm
---

# S1-mini — LiteRT-LM

**"S1-mini" by "Superwhisper"** — [superwhisper/s1-mini](https://huggingface.co/superwhisper/s1-mini) converted to the **LiteRT-LM** (`.litertlm`) format for on-device inference with Google's [LiteRT-LM](https://github.com/google-ai-edge/litert-lm) runtime. The model's license requires that it keep being identified by that name; this repository does, and so should anything built on it.

S1-mini takes a raw speech-to-text transcript and rewrites it as clean written text: fillers removed, false starts and self-corrections resolved to the value the speaker landed on, punctuation and capitalization applied, and spoken numbers, dates, times, currency and email addresses rendered in written form. It is a 0.6B Qwen3 finetune that does one job — **it is not a chat model, and it will normalize your question instead of answering it.**

| File | Recipe | Size |
|---|---|---|
| `S1-mini_int8.litertlm` | int8 dynamic on linears + embedding (`dynamic_wi8_afp32`) | 688 MB |

int8 only, and that is a measurement rather than a default. An int4 blockwise-32 build was made and rejected: it drops the comma before "and", drops discourse words like "hmm", and decodes *slower* than int8 at this size, on the same Mac and the same harness. Punctuation fidelity is this model's entire product.

## Usage

The required system prompt is **baked into the bundle's chat template**, so you send only the control line and the transcript:

```bash
litert-lm run ./S1-mini_int8.litertlm --prompt \
  "[Styling: semi-formal] [Structure: prose] [Context: general]
so um i need to like send the the report by uh friday no wait make that thursday"
# I need to send the report by Thursday.
```

Every input is a control line, a newline, then one raw transcript:

```
[Styling: <casual|semi-casual|semi-formal|formal>] [Structure: <prose|lists>] [Context: <general|email>]
<raw transcript>
```

The three axes are independent and every combination was trained. `Styling` sets the register, `Structure: lists` permits Markdown bullets for genuinely enumerable content, and `Context: email` produces greeting / body / sign-off layout. Values outside those sets are out of contract — the upstream card warns they cause hallucinated or garbled output. When the input is nothing but filler, the correct output is an empty string, and that is what you get.

English only. Keep inputs to roughly 1,000 tokens and chunk longer transcripts. Greedy decoding is the trained behaviour (the upstream `generation_config.json` sets `do_sample: false`); in apps that expose sampling, set top-k 1 / temperature 0.

## Correctness

The conversion is checked by feeding **the identical rendered prompt** to this bundle and to the HF checkpoint in fp32, then comparing strings exactly:

- **Bundle output == HF fp32 greedy, byte-for-byte, 10/10 cases, on CPU and on GPU** (macOS, litert-lm 0.16.0). The cases cover all four registers, list structuring, email layout, number/time/currency rendering, and the pure-filler empty-string case.
- **Pixel 8a CPU: 12/12 byte-identical** to the same bundle's macOS output across the twelve examples published on the upstream model card.
- A BOS A/B check makes no difference to outputs, so the bundle's start token is a non-issue here.

Two honest caveats, both measured rather than assumed:

- **Pixel 8a on the GPU scores 9 of those 12.** Two differences are harmless near-ties ("Hm," for "Hmm,", and prose where the reference chose bullets — the same choice the bf16 PyTorch model makes). One is real: the number-dense example loses a date separator, `"…due on March 3rd, 2026."` becoming `"…due on March 32026."`, deterministically on repeat runs. Use the CPU backend on this phone when exactness matters, and the GPU when time-to-first-token matters.
- **The released checkpoint itself drifts from its own card's example table** — it keeps discourse markers ("Hmm,", "So") that the card's expected outputs drop. This conversion reproduces the checkpoint, not the card; that is why the parity claim above is stated against HF fp32 rather than against the published examples.

## Performance

`litert-lm benchmark` (litert-lm 0.16.0), Apple M4 Max, `-p 256 -d 256 --runs 3 --cache no`, quiet machine:

| Backend | Prefill (256) | Decode | TTFT | Init |
|---|---|---|---|---|
| GPU (Metal) | 3777 tok/s | 146.7 tok/s | 0.075 s | 2.5 s |
| CPU | 466 tok/s | 34.6 tok/s | 0.58 s | 5.8 s |

Pixel 8a (Tensor G3), `litert_lm_advanced_main` from the v0.16.0 kit, 205-token prompt, `--benchmark`:

| Backend | Prefill (205) | Decode | TTFT |
|---|---|---|---|
| GPU (OpenCL, LiteRT `CompiledModel` / `LITERT_CL`) | 434 tok/s | 12.3 tok/s | 0.67 s |
| CPU (XNNPACK) | 40 tok/s | 5.8 tok/s | 6.5 s |

The GPU takes all 1247 decode nodes on that phone (full delegation, no XNNPACK fallback).

iPhone 17 Pro, single 180-token prompt, cold start:

| Backend | Prefill | Decode | TTFT | Peak memory |
|---|---|---|---|---|
| GPU (Metal) | 768–801 tok/s | 32.0 tok/s | 0.31 s | 1.7 GB |
| CPU | 281 tok/s | 15.5 tok/s | 0.74 s | — |

The Metal row is a two-run retake on a cooled phone (both runs agreed to 0.1 tok/s). The CPU row was measured with the device already reporting a `serious` thermal state, so treat it as a floor.

In practice a short dictation turn — one or two sentences in, one or two out — completes in about 3 seconds on a Pixel 8a CPU.

## Running it on Android

Import the file into Google's [AI Edge Gallery](https://github.com/google-ai-edge/gallery) (Models → Import model → local file) and set **TopK 1 / temperature 0.00** in the import dialog: the defaults are sampling values (TopK 64, temperature 1.0), which are wrong for a greedy single-task model.

Note that Gallery builds current as of 2026-08 offer **no accelerator choice for imported models** and run them on the CPU. That is a limitation of the app's import path, not of this file — the same bundle delegates fully to the GPU through the LiteRT-LM CLI on the same phone, at the speeds in the table above.

## Conversion notes

Converted with stock [`litert-torch`](https://github.com/google-ai-edge/litert-torch) 0.9.3 — no patches to the exporter. Reproduction script: [hf-to-litertlm `s1mini_work/`](https://github.com/john-rocky/hf-to-litertlm). Two things were resolved at conversion time, and both matter to anyone converting this model themselves:

- **The chat template is replaced, because the model requires a flag no runtime can pass.** S1-mini inherits Qwen3's template, whose default renders thinking mode; the upstream card requires `enable_thinking=False` and warns that the thinking render usually produces no usable output at all. A LiteRT-LM engine applies the embedded template with no such argument, so shipping the vendor template verbatim would ship a broken model. The bundled template hardcodes the non-thinking render (the assistant turn opens with an empty `<think>` block) and bakes in the exact system prompt the model was trained with, so clients that cannot send a system message still get the trained format. Before export, the replacement was verified to render byte-identically to the vendor template under `enable_thinking=False` across all twelve card examples, with and without an explicit system message; a second turn's render is a string-extension of the first turn plus its reply, which is what LiteRT-LM's incremental conversation rendering requires.
- **The exporter's composite stop strings are removed, because they eat the final punctuation.** litert-torch expands the template's turn suffix into punctuation-prefixed variants (`".<|im_end|>\n"`, `"?<|im_end|>\n"`, …) to work around SentencePiece greedy merging. Qwen3 is BPE, that merge never happens, but the runtime still matches the multi-token stop and strips the *whole* match — so a reply ending `…Thursday."` comes back as `…Thursday`. Measured cleanly: 7 of 7 test cases whose reference ends in "." lost exactly that period, and rewriting the metadata to keep only the two real stop token ids flipped all 7 back. On a punctuation-restoration model this is fatal, and a keyword-matching quality gate cannot see it.

Also worth knowing: this checkpoint ties its embedding and `lm_head`, so an int4 recipe that asks for int4 on the linears and int8 on the embedding describes one tensor two ways, and the quantizer resolves that by copying the vocabulary table per prefill signature — an "int4" file that came out *larger* than the int8 built the same way (656 MB against 613 MB) until the embedder was externalized.

Exported with a 4096-token cache and prefill signatures from 1 to 1024, so the runtime can chunk the ~1,000-token inputs the upstream card recommends.

## License and changes

Distributed under **Apache-2.0 plus the upstream naming term**: any use, distribution, or integration of this model, modified or not, must continue to identify it as **"S1-mini" by "Superwhisper"**, with that exact capitalization, regardless of any other name a product using it is marketed under. See `LICENSE`.

**Changes from the original work:** weights converted from safetensors bf16 to LiteRT flatbuffers and quantized to int8 as described above; tokenizer repackaged unmodified; the chat template replaced with the non-thinking, system-prompt-baked template described above; the exporter's composite stop-token strings dropped in favour of the two real stop token ids.

This repository is a community conversion. It is not affiliated with Superwhisper.
