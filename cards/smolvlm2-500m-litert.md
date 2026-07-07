---
license: apache-2.0
base_model: HuggingFaceTB/SmolVLM2-500M-Video-Instruct
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

# SmolVLM2-500M — LiteRT-LM (on-device Vision-Language Model)

[HuggingFaceTB/SmolVLM2-500M-Video-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct)
(image path) converted to the **LiteRT-LM** (`.litertlm`) format for **on-device image+text** inference
with Google's [LiteRT-LM](https://github.com/google-ai-edge/litert-lm) runtime.

SmolVLM2-500M is a tiny vision-language model from Hugging Face: a **SigLIP** vision encoder +
pixel-shuffle connector feeding a **SmolLM2 (Llama-architecture) 360M** decoder. At just **361 MB** it
is one of the smallest on-device VLMs — give it an image and a question, get a grounded answer, fully
offline.

| | |
|---|---|
| **File** | `SmolVLM2-500M.litertlm` (~361 MB) |
| **Vision** | SigLIP encoder (512×512, 1024 patches, no CLS) + pixel-shuffle ×4 + Linear connector, **int8** → **64 image tokens** |
| **Decoder** | SmolLM2-360M (Llama, 960-dim, 32 layers, GQA 15/5), **int4** weights (blockwise-32 + OCTAV); tied embedding INT8 (externalized) |
| **Compute** | integer |
| **Context (KV cache)** | 2048 |
| **Image input** | resized to 512×512 ((x−0.5)/0.5 normalization baked into the vision encoder) |
| **Base model** | HuggingFaceTB/SmolVLM2-500M-Video-Instruct |

## Quality

Single-image VQA produces coherent, image-grounded answers (CPU-verified; the SigLIP vision tower
converts bit-faithfully, float CPU-parity corr ≈ 1.0). It is a **very small (500M) model** — keep a
sensible `max_tokens` and use sampling (e.g. top-p); at pure greedy it can be repetitive/verbose.

## ⚠️ Best for single-image VQA — one image per conversation

Ask about **one image per chat** (start a new conversation for a different image). Single-image VQA is
the primary use case. (On the GPU backend, a second image in the same conversation may degrade — a
GPU-delegate trait shared across `fast_vlm` models; CPU handles multi-image.)

## Run on iPhone / macOS

Use the LiteRT-LM Swift runtime ([swift-litert-lm](https://github.com/google-ai-edge/litert-lm) /
the `LiteRTDemo` sample). Load `SmolVLM2-500M.litertlm` with the **vision tower enabled**
(modalities `Modality.textImage` / `[.vision]` — vision-only bundle, no audio tower), attach a photo,
ask a question.

## Run on Android — Google AI Edge Gallery

> **Update (July 2026):** [Google AI Edge Gallery](https://github.com/google-ai-edge/gallery) **v1.0.16+** can import litert-lm models **directly from Hugging Face** inside the app (tap **+**) — no computer or `adb` needed. The manual steps below are only required on older builds or for sideloading a local file.

Run this model **with image input** in the official
[Google AI Edge Gallery](https://github.com/google-ai-edge/gallery) app — no custom app needed
(the bundle carries the tokenizer, chat template, and image preprocessing config):

1. Push the bundle onto the phone (or download it there directly from this repo):
   `adb push SmolVLM2-500M.litertlm /sdcard/Download/`
2. Open the Gallery app, tap the **+** icon (bottom-right) and pick `SmolVLM2-500M.litertlm` in the file picker.
3. In the **Import Model** dialog, **check "Support image"** (required for image input), set a sensible **max tokens**, pick **GPU** (fast) or **CPU**, then tap **Import**.
4. Open the **Ask Image** task, choose the imported model, attach a photo, and ask.

> **Tip:** ask about **one image per conversation**. It's a tiny 500M model — keep max-tokens modest so it doesn't ramble.

## Conversion notes

- LiteRT-LM `fast_vlm` bundle: VISION_ENCODER (`[1,512,512,3]`→`[1,1024,768]`, SigLIP) + VISION_ADAPTER
  (`[1,1024,768]`→`[1,64,960]`, pixel-shuffle ×4 + Linear) + single-token EMBEDDER + PREFILL_DECODE.
- The vision encoder uses the static `arange(1024)` position-embedding path (the model's dynamic
  bucketize position logic is bypassed — numerically identical for a full 512×512 frame) and bakes the
  (x−0.5)/0.5 normalization + NCHW transpose into the graph.
- Single-image, no high-res splitting → a fixed 64 soft tokens; SmolLM2 (Llama) decoder exported with
  externalized (tied) embedder.

## License

Apache-2.0 (SmolVLM2 + SmolLM2). See the
[base model card](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct).
