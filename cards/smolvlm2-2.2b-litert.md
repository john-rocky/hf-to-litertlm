---
license: apache-2.0
base_model: HuggingFaceTB/SmolVLM2-2.2B-Instruct
tags:
  - litert
  - litert-lm
  - litertlm
  - on-device
  - edge
  - vlm
  - multimodal
  - smolvlm
  - image-text-to-text
pipeline_tag: image-text-to-text
library_name: litert-lm
---

# SmolVLM2-2.2B — LiteRT-LM (on-device Vision-Language Model)

[HuggingFaceTB/SmolVLM2-2.2B-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct)
(image path) converted to the **LiteRT-LM** (`.litertlm`) format for **on-device image+text**
inference with Google's [LiteRT-LM](https://github.com/google-ai-edge/litert-lm) runtime.

SmolVLM2-2.2B is the largest / most capable of Hugging Face's SmolVLM2 family: a **SigLIP** vision
encoder + pixel-shuffle connector feeding a **SmolLM2-1.7B (Llama-architecture)** language decoder.
Give it an image and a question, get a grounded answer, fully offline.

| | |
|---|---|
| **File** | `SmolVLM2-2.2B.litertlm` |
| **Vision** | SigLIP encoder (384×384, patch 14 → 729 patches, no CLS) + pixel-shuffle ×3 + Linear connector, **int8** → **81 image tokens** |
| **Decoder** | SmolLM2-1.7B (Llama, 2048-dim, 24 layers), **int4** weights (blockwise-32 + OCTAV); tied embedding INT8 (externalized) |
| **Compute** | integer |
| **Context (KV cache)** | 2048 |
| **Image input** | resized to 384×384 ((x−0.5)/0.5 normalization baked into the vision encoder) |
| **Base model** | HuggingFaceTB/SmolVLM2-2.2B-Instruct |

## Quality

Single-image VQA produces coherent, image-grounded answers (the SigLIP vision tower converts
bit-faithfully to the reference, float CPU-parity corr ≈ 1.0). This is the largest SmolVLM2, so it is
notably more capable than [SmolVLM2-500M](https://huggingface.co/litert-community/SmolVLM2-500M).

## ⚠️ Best for single-image VQA — one image per conversation

Ask about **one image per chat** (start a new conversation for a different image).

## Run on Android — Google AI Edge Gallery

> **Update (July 2026):** [Google AI Edge Gallery](https://github.com/google-ai-edge/gallery) **v1.0.16+** can import litert-lm models **directly from Hugging Face** inside the app (tap **+**) — no computer or `adb` needed. The manual steps below are only required on older builds or for sideloading a local file.

Run this model **with image input** in the official
[Google AI Edge Gallery](https://github.com/google-ai-edge/gallery) app — no custom app needed:

1. Push the bundle onto the phone (or download it there directly from this repo):
   `adb push SmolVLM2-2.2B.litertlm /sdcard/Download/`
2. Open the Gallery app, tap the **+** icon (bottom-right) and pick `SmolVLM2-2.2B.litertlm`.
3. In the **Import Model** dialog, **check "Support image"** (required for image input), then tap **Import**.
4. Open the **Ask Image** task, choose the imported model, attach a photo, and ask.

## Run on iPhone / macOS

Use the LiteRT-LM Swift runtime ([swift-litert-lm](https://github.com/google-ai-edge/litert-lm) /
the `LiteRTDemo` sample). Load `SmolVLM2-2.2B.litertlm` with the **vision tower enabled**
(modalities `Modality.textImage` / `[.vision]`), attach a photo, and ask.

## Conversion notes

- LiteRT-LM `fast_vlm` bundle: VISION_ENCODER (`[1,384,384,3]`→SigLIP) + VISION_ADAPTER (pixel-shuffle
  ×3 + Linear → `[1,81,2048]`) + single-token EMBEDDER + PREFILL_DECODE.
- The vision encoder uses the static position-embedding path (the model's dynamic bucketize position
  logic is bypassed — numerically identical for a full 384×384 frame) and bakes the (x−0.5)/0.5
  normalization + NCHW transpose into the graph.
- Single-image, no high-res splitting → a fixed 81 soft tokens; SmolLM2 (Llama) decoder exported with
  externalized (tied) embedder.

## License

Apache-2.0 (SmolVLM2 + SmolLM2). See the
[base model card](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct).
