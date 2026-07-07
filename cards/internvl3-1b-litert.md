---
license: apache-2.0
base_model: OpenGVLab/InternVL3-1B
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

# InternVL3-1B — LiteRT-LM (on-device Vision-Language Model)

[OpenGVLab/InternVL3-1B](https://huggingface.co/OpenGVLab/InternVL3-1B) converted to the
**LiteRT-LM** (`.litertlm`) format for **on-device image+text** inference with Google's
[LiteRT-LM](https://github.com/google-ai-edge/litert-lm) runtime (the engine behind the official
`litert-community/*` models).

InternVL3-1B is the **smallest** InternVL3 vision-language model: an **InternViT** vision encoder +
pixel-shuffle + MLP projector feeding a **Qwen2.5-0.5B** language decoder. At **738 MB** it is a
tiny, fast on-device VLM — give it an image and a question, get a grounded answer, fully offline.
(See [InternVL3-2B-LiteRT](https://huggingface.co/litert-community/InternVL3-2B) for the larger sibling.)

| | |
|---|---|
| **File** | `InternVL3-1B.litertlm` (~738 MB) |
| **Vision** | InternViT-300M encoder (4D-clean attention, GPU-friendly) + pixel-shuffle + MLP projector, **int8** — single **448×448** image → 256 image tokens |
| **Decoder** | Qwen2.5-0.5B (896-dim, 24 layers), **int4** weights (symmetric, **blockwise-32 + OCTAV**); input embedding INT8 (externalized) |
| **Compute** | integer |
| **Context (KV cache)** | 2048 |
| **Image input** | resized to 448×448 (ImageNet normalization baked into the vision encoder) |
| **Base model** | OpenGVLab/InternVL3-1B |

## Quality

Output is coherent and image-grounded (CPU-verified; the vision tower converts bit-faithfully to the
reference, float CPU-parity corr ≈ 1.0). On-device behavior mirrors the larger InternVL3-2B build
(same conversion recipe) — single-image VQA on GPU is fast and accurate; being 0.5B-decoder it is the
fastest/smallest of the family.

## ⚠️ Known limitation — one image per conversation on the GPU backend

Single-image VQA — the primary use case — works great on GPU. But on the **GPU (Metal) backend**, a
**second image in the *same* conversation** truncates the answer — ask about **one image per chat**
(start a new conversation for a different image). This is **GPU-delegate-specific, not a model/bundle
issue**: on the **CPU backend, multi-image works perfectly** (verified), and the same GPU truncation
reproduces with other `fast_vlm` models. **For reliable multi-image, run on the CPU backend.**

## Run on iPhone / macOS

Use the LiteRT-LM Swift runtime ([swift-litert-lm](https://github.com/google-ai-edge/litert-lm) /
the `LiteRTDemo` sample). Load `InternVL3-1B.litertlm` with the **image (vision) tower enabled**
(modalities `Modality.textImage` / `[.vision]` — a vision-only bundle, no audio tower), attach a photo,
and ask a question.

## Run on Android — Google AI Edge Gallery

> **Update (July 2026):** [Google AI Edge Gallery](https://github.com/google-ai-edge/gallery) **v1.0.16+** can import litert-lm models **directly from Hugging Face** inside the app (tap **+**) — no computer or `adb` needed. The manual steps below are only required on older builds or for sideloading a local file.

Run this model **with image input** in the official
[Google AI Edge Gallery](https://github.com/google-ai-edge/gallery) app — no custom app needed
(the bundle carries the tokenizer, chat template, and image preprocessing config):

1. Push the bundle onto the phone (or download it there directly from this repo):
   `adb push InternVL3-1B.litertlm /sdcard/Download/`
2. Open the Gallery app, tap the **+** icon (bottom-right) and pick `InternVL3-1B.litertlm` in the file picker.
3. In the **Import Model** dialog, **check "Support image"** (required for image input), pick **GPU** (fast) or **CPU**, then tap **Import**.
4. Open the **Ask Image** task, choose the imported model, attach a photo, and ask.

> **Tip:** on the **GPU** backend use one image per conversation (a known GPU-delegate trait of `fast_vlm` models); pick **CPU** if you want multiple images in one chat.

## Conversion notes

- LiteRT-LM `fast_vlm` bundle: VISION_ENCODER (`[1,448,448,3]`→`[1,256,4096]`) + VISION_ADAPTER
  (`[1,256,4096]`→`[1,256,896]`) + single-token EMBEDDER + PREFILL_DECODE (embeddings-input).
- The vision encoder bakes ImageNet normalization + the NCHW transpose into the graph, and the
  InternViT attention is rewritten **4D-clean** (qkv split before the head reshape — no GPU-rejected
  5D reshape), numerically identical (corr ≈ 1.0).
- Decoder exported with externalized embedder; InternVL's dynamic-NTK `rope_scaling` is stripped to
  base RoPE (valid since the export cache ≤ the base context window).

## License

**MIT** (the InternVL model) **+ Apache-2.0** (the Qwen2.5 language component). See the
[base model card](https://huggingface.co/OpenGVLab/InternVL3-1B). Converted artifacts are released
under the same terms.
