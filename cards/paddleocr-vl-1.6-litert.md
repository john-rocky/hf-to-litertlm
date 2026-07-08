---
license: apache-2.0
base_model: PaddlePaddle/PaddleOCR-VL-1.6
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
pipeline_tag: image-text-to-text
library_name: litert-lm
---

# PaddleOCR-VL-1.6 — LiteRT-LM (on-device document-parsing / OCR VLM)

[PaddlePaddle/PaddleOCR-VL-1.6](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6) converted to the **LiteRT-LM** (`.litertlm`) format for **on-device document AI** with Google's [LiteRT-LM](https://github.com/google-ai-edge/litert-lm) runtime — the first document-parsing / OCR-specialist VLM in this format.

PaddleOCR-VL is the **SOTA open document-parsing model** (OmniDocBench v1.6 **96.33%**, ahead of both open and closed alternatives at any size) in a phone-sized package: a NaViT-style dynamic-resolution SigLIP vision encoder + the tiny **ERNIE-4.5-0.3B** decoder, ~0.9B parameters total, supporting **109 languages**. It is a *task-prompted* model — you select what it does with the text prompt:

| Prompt | Task |
|---|---|
| `OCR:` | text recognition (page / paragraph / line) |
| `Table Recognition:` | table → structured cells (`<fcel>`/`<nl>` format) |
| `Formula Recognition:` | formula → LaTeX |
| `Chart Recognition:` | chart parsing |
| `Spotting:` | text + `<|LOC_*|>` box tokens |
| `Seal Recognition:` | seal/stamp text |

| | |
|---|---|
| **File** | `PaddleOCR-VL-1.6.litertlm` (~1.39 GB) |
| **Vision** | SigLIP-so400m-class NaViT encoder (27L, hidden 1152) made **static 560×560** → 1600 patches → 2×2 merge → **400 image tokens**, int8 weights |
| **Adapter** | LN → 2×2 spatial merge → MLP (4608→1024), int8 |
| **Decoder** | ERNIE-4.5-0.3B (18L, hidden 1024, GQA kv2, head_dim 128), **fp16 weights** (+ fp16 externalized embedder) |
| **Context (KV cache)** | 4096 |
| **Image input** | resized by the runtime to 560×560 (mean/std-0.5 normalization baked into the encoder) |
| **Base model** | PaddlePaddle/PaddleOCR-VL-1.6 (Apache-2.0) |

## Quality

**Device-verified on a Pixel 8a** (Google AI Edge Gallery 1.0.15, CPU backend, this exact bundle):

- **Page OCR** (`OCR:` on a synthetic report page): **perfect transcription** in **22 s** — every figure, the e-mail address and the phone number come out exactly ("Revenue increased by 18.4% to $2,315 million …").
- **Table recognition** (`Table Recognition:`): **every cell correct** including the Total row, in **17 s** (`Product / Units / Revenue … Total 24,510 $1,207,000`).

Also validated on the desktop LiteRT-LM runtime (macOS CPU): table output is **byte-identical** to the full-precision eager reference, including the structured `<fcel>`/`<nl>` cell tokens (the bundle's SentencePiece vocabulary carries all 1 019 added tokens at their exact ids, so they detokenize correctly on-device).
- Vision tower: tflite vs the reference implementation corr **1.0** (fp32) / **0.9975** (shipped int8), **zero FLEX/CUSTOM ops** (GPU-clean).
- Decoder: extracted as a standalone Llama-layout model — **bit-exact** fp32 logits vs the original; shipped **fp16** weights are teacher-forced-parity **corr 1.0000, top-1 10/10**. This 0.36 B decoder is unusually quantization-sensitive (int4 *and* integer-compute int8 measurably corrupt transcription), so the bundle spends ~460 MB extra on fp16 exactness — for OCR, exactness wins.

> **M-RoPE note.** The base decoder uses Qwen2-VL-style 3-D M-RoPE. The LiteRT-LM `fast_vlm` contract supplies plain sequential positions — for text tokens this is mathematically identical, and for image tokens an A/B eager test (true M-RoPE vs 1-D) showed **no quality loss** on OCR/table tasks (raster reading order survives 1-D positions). This is what makes the ride possible.

> **Scope.** This is the *recognition* component of the PaddleOCR 3.x pipeline. The official server pipeline puts a layout-detection model (PP-DocLayout) in front for full-page multi-element parsing; standalone, this bundle handles pages, paragraphs, text lines, tables, formulas and charts directly — best on single elements or simple pages, exactly like the upstream `transformers` usage.

> **Aspect ratio.** The runtime resizes input to a fixed square, so extreme aspect ratios (very wide single lines) get distorted. For best results feed roughly page/paragraph-shaped crops; the deployed 560² contract transcribed a 16:9 test page perfectly.

> **One image per chat.** Like the other fast_vlm bundles, send each document in a fresh conversation — a second image in the same chat degrades (context bleed from the first turn was observed on CPU).

## Run on iPhone / macOS

Use the LiteRT-LM Swift runtime ([swift-litert-lm](https://github.com/google-ai-edge/litert-lm)). Load `PaddleOCR-VL-1.6.litertlm` with the **vision tower enabled** (`Modality.textImage`), attach a document photo, and send one of the task prompts above (e.g. `OCR:`).

> Vision-only bundle (no audio tower): bring the engine up with the vision modality only — requesting `.all` fails at session creation on bundles without an audio section.

## Run on Android — Google AI Edge Gallery

Install a recent [Google AI Edge Gallery](https://github.com/google-ai-edge/gallery), download `PaddleOCR-VL-1.6.litertlm`, import it (tap **+**), attach a document image and prompt `OCR:`. The bundle carries the tokenizer, template and both towers.

## Conversion notes

- LiteRT-LM `fast_vlm` bundle: VISION_ENCODER (`[1,560,560,3]`→`[1,1600,1152]`) + VISION_ADAPTER (`[1,1600,1152]`→`[1,400,1024]`) + single-token EMBEDDER + PREFILL_DECODE (embeddings-input).
- **Static NaViT rewrite:** PaddleOCR-VL's encoder is dynamic-resolution (packed patches, interpolated position embeddings, 2-D rotary over h/w ids) and does not `torch.export`. The LM calls it with **full attention** (`window_size=-1`), so the static graph is: whole-image patch Conv2d (the processor packs patches in **pure raster order** — no gather needed), precomputed bilinear-interpolated position embedding, precomputed 2-D rope cos/sin for the fixed 40×40 grid.
- The projector's 2×2 spatial merge is done GPU-safe with 4 strided slices + concat (all tensors ≤4D) instead of the literal 6-D rearrange.
- Decoder: the ERNIE-4.5-0.3B inside the VLM is **layout-identical to Llama** — re-hosted as a standalone `LlamaForCausalLM` (state-dict 1:1, bit-exact logits) and exported with the standard litert-torch path, cache 4096.
- Tokenizer: the base SP model lacks the 1 019 added tokens (`<|IMAGE_START|>`, `<|LOC_0|>`…`<|LOC_1000|>`, `<fcel>`/`<nl>` table tokens…). They are appended as `USER_DEFINED` SentencePiece pieces at their exact ids (padded to vocab 103 424) so table/spotting output detokenizes correctly on device.
- Prompt template (baked into the bundle): `<|begin_of_sentence|>User: <image>PROMPT\nAssistant:\n`, stop token `</s>`.

## License

Apache-2.0, inherited from the base model [PaddlePaddle/PaddleOCR-VL-1.6](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6).
