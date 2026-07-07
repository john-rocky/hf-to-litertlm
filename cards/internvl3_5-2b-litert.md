---
license: apache-2.0
base_model: OpenGVLab/InternVL3_5-2B
tags:
  - litert
  - litert-lm
  - litertlm
  - on-device
  - edge
  - vlm
  - multimodal
  - vision-language
pipeline_tag: image-text-to-text
library_name: litert-lm
---

# InternVL3.5-2B — LiteRT-LM (on-device Vision-Language Model)

[OpenGVLab/InternVL3_5-2B](https://huggingface.co/OpenGVLab/InternVL3_5-2B) converted to the
**LiteRT-LM** (`.litertlm`) format for **on-device image+text** inference with Google's
[LiteRT-LM](https://github.com/google-ai-edge/litert-lm) runtime (the engine behind the official
`litert-community/*` models, and the same runtime that runs `litert-community/FastVLM-0.5B`).

InternVL3.5-2B is a compact **vision-language model**: an **InternViT** vision encoder + pixel-shuffle +
MLP projector feeding a **Qwen3-1.7B** language decoder (the newer Qwen3 backbone is what distinguishes
it from the InternVL3-2B build, which used Qwen2.5-1.5B). This bundle runs it through LiteRT-LM's
`fast_vlm` multimodal path — give it an image and a question, get a grounded answer, fully on-device.

| | |
|---|---|
| **File** | `InternVL3_5-2B.litertlm` (~1.61 GB) |
| **Vision** | InternViT encoder + pixel-shuffle + MLP projector, **int8** weights — single **448×448** image → 256 image tokens |
| **Decoder** | Qwen3-1.7B, **int4** weights (symmetric, **blockwise-32 + OCTAV** optimal-clipping); input embedding INT8 (externalized section) |
| **Compute** | integer |
| **Context (KV cache)** | 2048 |
| **Image input** | resized to 448×448 (ImageNet normalization is baked into the vision encoder) |
| **Base model** | OpenGVLab/InternVL3_5-2B (Apache-2.0) |

## Quality

The vision tower converts **bit-faithfully** to the reference — float CPU-parity **end-to-end
corr ≈ 1.0** (max abs diff ~1e-4), with no FLEX/CUSTOM fallback ops; int8 vision weights preserve
grounding. The Qwen3-1.7B decoder uses the same **blockwise-32 + OCTAV int4** recipe that scores
90.7% GSM8K on the sibling [Ministral-3-3B-Reasoning build](https://huggingface.co/litert-community/Ministral-3-3B-Reasoning-2512).
On a reference eager run the model describes photos accurately and in detail (e.g. a black-and-white
Ansel-Adams-style landscape → "dramatic mountain landscape … snow-capped peaks … a winding river
through a forested valley").

> **On-device performance:** decode/load are expected to be in line with the InternVL3-2B build on the
> same runtime (~20 tok/s CPU, ~45 tok/s GPU on iPhone 17 Pro for single-image VQA). Independent
> on-device measurement for this specific 2B/Qwe3 build is recommended before quoting exact numbers.

## ⚠️ Known limitation — one image per conversation on the GPU backend

Single-image VQA — the primary use case — works on GPU. But on the **GPU (Metal) backend**, a
**second image in the *same* conversation** truncates the answer — ask about **one image per chat**
(start a new conversation for a different image). This is **GPU-delegate-specific, not a model/bundle
issue**: on the **CPU backend, multi-image works**. The same GPU truncation reproduces with Apple's
`litert-community/FastVLM-0.5B`, so it is general to the runtime's GPU `fast_vlm` path, not specific to
this model. **For reliable multi-image, run on the CPU backend.**

## Run on iPhone / macOS

Use the LiteRT-LM Swift runtime ([swift-litert-lm](https://github.com/google-ai-edge/litert-lm) /
the `LiteRTDemo` sample). Load `InternVL3_5-2B.litertlm` with the **image (vision) tower enabled**
(modalities `[.vision]`), attach a photo, and ask a question.

> Note for app integrators: this is a **vision-only** bundle (no audio tower). Bring up the engine with
> the **vision** modality only (`Modality.textImage` / `[.vision]`) — requesting the audio tower
> (`.all`) on a bundle with no audio section fails at session creation.

## Run on Android — Google AI Edge Gallery

Install a recent [Google AI Edge Gallery](https://github.com/google-ai-edge/gallery) (1.0.16+ can
import `.litertlm` directly from Hugging Face), download `InternVL3_5-2B.litertlm`, import it (tap
**+**), attach an image and ask. The bundle already carries the tokenizer and prompt template.

## Conversion notes

- LiteRT-LM `fast_vlm` bundle: VISION_ENCODER (`[1,448,448,3]`→`[1,256,4096]`) + VISION_ADAPTER
  (`[1,256,4096]`→`[1,256,2048]`, matched to the Qwen3-1.7B hidden size) + single-token EMBEDDER +
  PREFILL_DECODE (embeddings-input).
- The vision encoder bakes InternVL's ImageNet normalization and the NCHW transpose into the graph
  (the runtime feeds a `[0,1]` NHWC image).
- The InternViT attention is rewritten **4D-clean** (qkv split before the head reshape, avoiding a 5D
  intermediate) for the GPU delegate.
- Decoder extracted from the InternVLChat wrapper as a standalone `Qwen3ForCausalLM` (dynamic
  rope_scaling stripped; exported with cache ≤ base max so base RoPE is exact).

## License

Apache-2.0, inherited from the base model
[OpenGVLab/InternVL3_5-2B](https://huggingface.co/OpenGVLab/InternVL3_5-2B).
