---
license: apache-2.0
base_model: llava-hf/llava-onevision-qwen2-0.5b-ov-hf
tags:
  - litert
  - litert-lm
  - litertlm
  - on-device
  - edge
  - vlm
  - multimodal
  - llava
  - llava-onevision
  - image-text-to-text
pipeline_tag: image-text-to-text
library_name: litert-lm
---

# LLaVA-OneVision-0.5B — LiteRT-LM (on-device Vision-Language Model)

[llava-hf/llava-onevision-qwen2-0.5b-ov-hf](https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf)
converted to the **LiteRT-LM** (`.litertlm`) format for **on-device image+text** inference with
Google's [LiteRT-LM](https://github.com/google-ai-edge/litert-lm) runtime (the engine behind the
official `litert-community/*` models).

LLaVA-OneVision-0.5B is a compact vision-language model from the LLaVA team: a **SigLIP** vision
encoder + MLP projector feeding a **Qwen2-0.5B** language decoder. This **829 MB** bundle runs it
through LiteRT-LM's `fast_vlm` multimodal path — give it an image and a question, get a grounded
answer, fully offline.

| | |
|---|---|
| **File** | `LLaVA-OneVision-0.5B.litertlm` (~829 MB) |
| **Vision** | SigLIP encoder (384×384, 729 patches, no CLS) + MLP projector, **int8** → 730 image tokens (729 + an `image_newline`) |
| **Decoder** | Qwen2-0.5B (896-dim, 24 layers), **int4** weights (symmetric, **blockwise-32 + OCTAV**); tied embedding INT8 (externalized) |
| **Compute** | integer |
| **Context (KV cache)** | 2048 |
| **Image input** | resized to 384×384 (OpenAI-CLIP normalization baked into the vision encoder) |
| **Base model** | llava-hf/llava-onevision-qwen2-0.5b-ov-hf |

## Quality

Single-image VQA produces coherent, image-grounded answers (CPU-verified; the SigLIP vision tower
converts bit-faithfully to the reference, float CPU-parity corr ≈ 1.0).

## ⚠️ Best for single-image VQA — one image per conversation

Ask about **one image per chat**. This 0.5B model with 730 image tokens per image becomes unreliable
when a **second image is added to the same conversation** (the answer truncates) — start a **new
conversation** for a different image. Single-image VQA, the primary use case, works well.

## Run on iPhone / macOS

Use the LiteRT-LM Swift runtime ([swift-litert-lm](https://github.com/google-ai-edge/litert-lm) /
the `LiteRTDemo` sample). Load `LLaVA-OneVision-0.5B.litertlm` with the **vision tower enabled**
(modalities `Modality.textImage` / `[.vision]` — a vision-only bundle, no audio tower), attach a photo,
and ask a question.

## Run on Android — Google AI Edge Gallery

> **Update (July 2026):** [Google AI Edge Gallery](https://github.com/google-ai-edge/gallery) **v1.0.16+** can import litert-lm models **directly from Hugging Face** inside the app (tap **+**) — no computer or `adb` needed. The manual steps below are only required on older builds or for sideloading a local file.

Run this model **with image input** in the official
[Google AI Edge Gallery](https://github.com/google-ai-edge/gallery) app — no custom app needed
(the bundle carries the tokenizer, chat template, and image preprocessing config):

1. Push the bundle onto the phone (or download it there directly from this repo):
   `adb push LLaVA-OneVision-0.5B.litertlm /sdcard/Download/`
2. Open the Gallery app, tap the **+** icon (bottom-right) and pick `LLaVA-OneVision-0.5B.litertlm` in the file picker.
3. In the **Import Model** dialog, **check "Support image"** (required for image input), pick **GPU** (fast) or **CPU**, then tap **Import**.
4. Open the **Ask Image** task, choose the imported model, attach a photo, and ask.

> **Tip:** ask about **one image per conversation** (start a new chat for a different image) — this 0.5B model is single-image only.

## Conversion notes

- LiteRT-LM `fast_vlm` bundle: VISION_ENCODER (`[1,384,384,3]`→`[1,729,1152]`, SigLIP) + VISION_ADAPTER
  (`[1,729,1152]`→`[1,730,896]`, projector + the learned `image_newline` token) + single-token EMBEDDER
  + PREFILL_DECODE (embeddings-input).
- The vision encoder bakes OpenAI-CLIP normalization + the NCHW transpose into the graph; the single
  base-resolution (no-anyres) path is used so the image always maps to a fixed 730 soft tokens.
- Decoder exported with externalized (tied) embedder.

## License

Apache-2.0 (LLaVA-OneVision + the Qwen2 language component). See the
[base model card](https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf). Converted
artifacts are released under the same terms.
