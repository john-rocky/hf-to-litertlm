---
license: apache-2.0
base_model: ibm-granite/granite-docling-258M
tags:
  - litert
  - litert-lm
  - litertlm
  - on-device
  - edge
  - vlm
  - multimodal
  - ocr
  - document-parsing
  - table-recognition
  - docling
  - doctags
pipeline_tag: image-text-to-text
library_name: litert-lm
---

# granite-docling-258M — LiteRT-LM (on-device document conversion, DocTags)

[ibm-granite/granite-docling-258M](https://huggingface.co/ibm-granite/granite-docling-258M) converted to the **LiteRT-LM** (`.litertlm`) format for **on-device document conversion** with Google's [LiteRT-LM](https://github.com/google-ai-edge/litert-lm) runtime.

granite-docling is IBM's document-conversion VLM — the model behind [Docling](https://github.com/docling-project/docling). Give it a page image and the prompt `Convert this page to docling.`; it emits **DocTags**, a structured markup with layout elements, OTSL table structure, code and formulas, which [docling-core](https://github.com/docling-project/docling-core) converts losslessly to Markdown, HTML or JSON. At 258M parameters (~338 MB here) it is one of the smallest document-AI models that captures table *structure*, not just text.

| | |
|---|---|
| **File** | `granite-docling-258M.litertlm` (~338 MB) |
| **Vision** | SigLIP-base p16 (512×512, 1024 patches) + pixel-shuffle ×4 + Linear connector, **int8** → **64 image tokens** |
| **Decoder** | Llama-architecture granite decoder (576-dim, 30 layers, GQA 9/3, 100k vocab), **int8 weights with float compute** |
| **Context (KV cache)** | 4096 |
| **Image input** | **must be pre-resized to 512×512 BILINEAR** (see contract below) |
| **Backend** | CPU |
| **Base model** | ibm-granite/granite-docling-258M (Apache-2.0) |

## ⚠️ Input contract: pre-resize to 512×512 BILINEAR

Resize the page to **exactly 512×512 with BILINEAR resampling in your app** before sending it. Do not hand the runtime a larger image: its internal resampler downscales with a different filter, and the model then produces a hallucinated page instead of the real one.

This is a base-model trait, not conversion damage: granite-docling is trained with image splitting (tiles + a global view), and this bundle runs the single-global-512 path — sharp on a 512-legible page, but sensitive to the exact resampling filter. The same full-precision model run the same way in `transformers` flips between perfect and degenerate on a BILINEAR↔LANCZOS change. Pre-sized 512 BILINEAR input is the verified path.

**Scope.** Best for pages whose text is legible at 512²: report-style pages, tables, layout extraction. Dense multi-column pages and small formulas degrade in single-512 mode — also in the original model run this way. For those, tile the page app-side and send crops.

## Quality — measured, this exact bundle

Structure gate (synthetic report page with a 5×6 table), greedy decoding:

- **macOS CPU** (litert-lm 0.15.0): exact title, complete 5×6 OTSL grid, **25/25 data cells correct**, clean stop — **5.2 s/page** (~240 output tokens, Apple M4 Max).
- **Galaxy S26 CPU** (SM8850, LiteRT-LM v0.16.1 CLI): same page — exact title, complete grid, **25/25 cells**, clean stop — **35.4 s/page** (1025 output tokens). Two runs byte-identical.
- Conversion parity: the vision tower converts with corr 1.000 vs the original (fp32; shipped int8 corr 0.98, structurally identical output). Full-precision and int8-weight decoders produce the same table structure; **int4 was tested and rejected** — integer-compute quantization corrupts DocTags on this decoder, so the bundle ships int8 weights with float compute.

## Performance (measured)

Text path, `litert-lm benchmark -p 256 -d 256 --runs 3 --cache no`; backend gated on a real docling generation first:

| Device | Backend | Prefill (tok/s) | Decode (tok/s) | TTFT |
|---|---|---|---|---|
| Apple M4 Max (litert-lm 0.15.0) | CPU | 912 | 64.0 | 0.57 s |

The vision encoder runs once per image (~0.5 s on the M4 Max CPU) and is not included in the table.

**GPU:** the shipped bundle does not create a GPU engine — the GPU delegate rejects a quantized 576×576 projection at kernel init (`Shape mismatch: {576,1,1,576} vs {1,1,576,576}`, same signature on macOS and Android OpenCL). An fp16-decoder variant runs correctly on Android OpenCL but ~4× slower than CPU, so it is not shipped. CPU is the intended backend for this model.

## Run on Android — Google AI Edge Gallery

Recent [Google AI Edge Gallery](https://github.com/google-ai-edge/gallery) builds import litert-lm models directly from Hugging Face (tap **+**), or sideload:

1. `adb push granite-docling-258M.litertlm /sdcard/Download/`
2. Tap **+**, pick the file, and in the Import dialog **check "Support image"**, set max tokens ≥ 2048, pick **CPU**.
3. Open **Ask Image**, attach a **512×512 pre-resized** page, and prompt `Convert this page to docling.`

The Gallery renders DocTags as plain text; convert it with docling-core for Markdown/HTML.

## Run on iPhone / macOS

Use the LiteRT-LM Swift runtime ([swift-litert-lm](https://github.com/google-ai-edge/litert-lm)). Load the bundle with the **vision tower enabled** (`Modality.textImage` — vision-only bundle, no audio tower), attach the pre-resized page, and send `Convert this page to docling.` One page per conversation; start a new one for the next page.

## Conversion notes

- LiteRT-LM `fast_vlm` bundle: VISION_ENCODER (`[1,512,512,3]`→`[1,1024,768]`, SigLIP static-position path, normalization baked in) + VISION_ADAPTER (pixel-shuffle ×4 + Linear 12288→576 → `[1,64,576]`) + single-token EMBEDDER + PREFILL_DECODE. Single global 512² view, no tiling → fixed 64 soft tokens.
- The idefics3 vision tower is architecturally identical to SmolVLM2's and converts with the same scripts, unchanged (corr 1.000, no FLEX/custom ops).
- Decoder: the granite 100k-vocab Llama-architecture decoder, int8 weights + float compute, `prefer_activation_type=fp32_fp16` declared in-bundle. Integer-compute int8/int4 corrupt DocTags structure — document-parsing decoders at this size are quantization-sensitive (same behavior family as PaddleOCR-VL's ERNIE decoder).
- Tokenizer: the HF `tokenizer.json` (GPT-2 byte-BPE) with all DocTags added tokens at their exact ids. No BOS token (`add_bos_token=false`); stop token `<|end_of_text|>`.
- Template: granite chat format. The newline after `<|end_of_text|>` is part of the format — this model is format-exact, and dropping that single token turns output into a hallucinated blank page. If you re-template, protect the `\n`.

## License

Apache-2.0, inherited from the base model [ibm-granite/granite-docling-258M](https://huggingface.co/ibm-granite/granite-docling-258M).
