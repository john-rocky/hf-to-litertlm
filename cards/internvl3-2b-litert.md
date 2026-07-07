---
license: apache-2.0
base_model: OpenGVLab/InternVL3-2B
tags:
  - litert
  - litert-lm
  - litertlm
  - on-device
  - edge
  - vlm
  - multimodal
  - internvl
  - image-text-to-text
pipeline_tag: image-text-to-text
library_name: litert-lm
---

# InternVL3-2B — LiteRT-LM (on-device Vision-Language Model)

[OpenGVLab/InternVL3-2B](https://huggingface.co/OpenGVLab/InternVL3-2B) converted to the
**LiteRT-LM** (`.litertlm`) format for **on-device image+text** inference with Google's
[LiteRT-LM](https://github.com/google-ai-edge/litert-lm) runtime (the engine behind the official
`litert-community/*` models, and the same runtime that runs `litert-community/FastVLM-0.5B`).

InternVL3-2B is a compact **vision-language model**: an **InternViT** vision encoder + pixel-shuffle +
MLP projector feeding a **Qwen2.5-1.5B** language decoder. This bundle runs it through LiteRT-LM's
`fast_vlm` multimodal path — give it an image and a question, get a grounded answer, fully on-device.

| | |
|---|---|
| **File** | `InternVL3-2B.litertlm` (~1.43 GB) |
| **Vision** | InternViT-300M encoder + pixel-shuffle + MLP projector, **int8** weights — single **448×448** image → 256 image tokens |
| **Decoder** | Qwen2.5-1.5B, **int4** weights (symmetric, **blockwise-32 + OCTAV** optimal-clipping); input embedding INT8 (externalized section) |
| **Compute** | integer |
| **Context (KV cache)** | 2048 |
| **Image input** | resized to 448×448 (ImageNet normalization is baked into the vision encoder) |
| **Base model** | OpenGVLab/InternVL3-2B |

## Performance (iPhone 17 Pro, CPU)

| | |
|---|---|
| **Load** | ~3–4 s |
| **Decode** | ~20 tok/s |
| **Multi-turn text** | works (ask follow-up questions about the same image) |

The image is described accurately and in detail. The vision tower converts bit-faithfully to the
reference (float CPU-parity corr ≈ 1.0); int8 vision weights keep grounding quality.

## ⚠️ Known limitation — one image per conversation on the GPU backend

Single-image VQA — the primary use case — works great on GPU (~45 tok/s on iPhone 17 Pro). But on the
**GPU (Metal) backend**, a **second image in the *same* conversation** truncates the answer — ask
about **one image per chat** (start a new conversation for a different image).

This is **GPU-delegate-specific, not a model/bundle issue**: on the **CPU backend, multi-image works
perfectly** (verified). The same GPU truncation reproduces with Apple's `litert-community/FastVLM-0.5B`,
so it is general to the runtime's GPU `fast_vlm` path, not specific to this model. (Ruled out as causes:
`max_num_images` — CPU works with it set to 1; and the vision encoder's 5D reshape — a 4D-clean rebuild
still truncates on GPU.) **For reliable multi-image, run on the CPU backend.**

## Run on iPhone / macOS

Use the LiteRT-LM Swift runtime ([swift-litert-lm](https://github.com/google-ai-edge/litert-lm) /
the `LiteRTDemo` sample). Load `InternVL3-2B.litertlm` with the **image (vision) tower enabled**
(modalities `[.vision]`), attach a photo, and ask a question.

> Note for app integrators: this is a **vision-only** bundle (no audio tower). Bring up the engine with
> the **vision** modality only (`Modality.textImage` / `[.vision]`) — requesting the audio tower
> (`.all`) on a bundle with no audio section fails at session creation.

## Run on Android — Google AI Edge Gallery

> **Update (July 2026):** [Google AI Edge Gallery](https://github.com/google-ai-edge/gallery) **v1.0.16+** can import litert-lm models **directly from Hugging Face** inside the app (tap **+**) — no computer or `adb` needed. The manual steps below are only required on older builds or for sideloading a local file.

Run this model **with image input** in the official
[Google AI Edge Gallery](https://github.com/google-ai-edge/gallery) app — no custom app needed
(the bundle carries the tokenizer, chat template, and image preprocessing config):

1. Push the bundle onto the phone (or download it there directly from this repo):
   `adb push InternVL3-2B.litertlm /sdcard/Download/`
2. Open the Gallery app, tap the **+** icon (bottom-right) and pick `InternVL3-2B.litertlm` in the file picker.
3. In the **Import Model** dialog, **check "Support image"** (required for image input), pick **GPU** (fast) or **CPU**, then tap **Import**.
4. Open the **Ask Image** task, choose the imported model, attach a photo, and ask.

> **Tip:** on the **GPU** backend use one image per conversation (a known GPU-delegate trait of `fast_vlm` models); pick **CPU** if you want multiple images in one chat.

## Conversion notes

- LiteRT-LM `fast_vlm` bundle: VISION_ENCODER (`[1,448,448,3]`→`[1,256,4096]`) + VISION_ADAPTER
  (`[1,256,4096]`→`[1,256,1536]`) + single-token EMBEDDER + PREFILL_DECODE (embeddings-input).
- The vision encoder bakes InternVL's ImageNet normalization and the NCHW transpose into the graph
  (the runtime feeds a `[0,1]` NHWC image).
- The InternViT attention is rewritten to be **4D-clean** (qkv split before the head reshape, avoiding
  the 5D `reshape(B,N,3,H,d)` that GPU delegates reject) — numerically identical (corr ≈ 1.0), but it
  keeps the vision encoder almost entirely on the GPU delegate.
- Decoder exported with externalized embedder; InternVL's dynamic-NTK `rope_scaling` is stripped to
  base RoPE (valid since the export cache ≤ the base context window).

## License

**MIT** (the InternVL model) **+ Apache-2.0** (the Qwen2.5 language component). See the
[base model card](https://huggingface.co/OpenGVLab/InternVL3-2B). Converted artifacts are released
under the same terms.
