---
license: apache-2.0
base_model: microsoft/Mage-VL
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

# Mage-VL — LiteRT-LM (on-device Vision-Language Model)

[microsoft/Mage-VL](https://huggingface.co/microsoft/Mage-VL) converted to the **LiteRT-LM** (`.litertlm`) format for **on-device image+text** inference with Google's [LiteRT-LM](https://github.com/google-ai-edge/litert-lm) runtime — the first Mage-VL in this format.

Mage-VL is Microsoft's 4.7B vision-language model: a 24-layer ViT with 3-D rotary position embeddings feeds a **Qwen3-4B** language decoder. It is a strong general describer and a *very* good document reader for its size. This bundle runs the image path through LiteRT-LM's `fast_vlm` runtime — give it an image and a question, get a grounded answer, fully on-device.

| | |
|---|---|
| **File** | `Mage-VL.litertlm` (~2.81 GB) |
| **Vision** | `mage_vl_vision` ViT (24L, 1024-dim, full attention, 3-D rope) made **static 448×448** → 784 patches → 2×2 merge → **196 image tokens**, int8 weights |
| **Adapter** | PatchMerger (LN → 2×2 group → MLP), int8, output at the 2560 text hidden size |
| **Decoder** | Qwen3-4B (36L, hidden 2560, GQA kv8), **int4** weights (symmetric, blockwise-128 + OCTAV); int8 externalized embedder |
| **Context (KV cache)** | 2048 |
| **Image input** | resized to 448×448 (OpenAI-CLIP normalization baked into the encoder) |
| **Base model** | microsoft/Mage-VL (Apache-2.0) |

## Performance (measured)

| Platform | Decode | Time-to-first-token (image turn) | Engine init | Peak footprint |
|---|---|---|---|---|
| **iPhone 17 Pro** (CPU, release build) | **~10 tok/s** | ~7.5–9 s (includes vision encode + prefill) | 0.8 s warm / ~3–4 s first run | **~1.5 GiB** |
| **macOS** (Apple Silicon, LiteRT-LM CPU) | ~66 tok/s | ~2 s | ~1 s | — |

Text-only follow-up turns in the same conversation have ~2 s time-to-first-token on the phone.

## Quality

**Device-verified on an iPhone 17 Pro** and on the desktop LiteRT-LM runtime (macOS CPU):

- **General description / VQA** (photo, on-device): accurate, detailed, and **identical to the desktop runtime output token-for-token** — an Ansel-Adams-style landscape → "a black and white photograph of a mountainous landscape … a winding river cutting through a dense forested valley … jagged, snow-capped mountains … heavy, brooding clouds".
- **Document OCR** (`Extract all the text from this image.`, on-device): **perfect transcription** of a full synthetic report page — every figure, the e-mail address and the phone number, at 448×448.
- Vision tower: static-rewrite vs the reference implementation corr **1.0** (fp32), **0.994** at int8, **zero FLEX/CUSTOM ops**; patch pipeline verified bit-identical to the model's own image processor (max diff 2.4e-7).
- Decoder: the Qwen3-4B text model is re-hosted as a standalone `Qwen3ForCausalLM` (state-dict strictly 1:1, untied lm_head) and quantized with the blockwise-128 + OCTAV int4 recipe that the shipped Qwen3-4B-class LLMs use; desktop 8-question sanity gate 7/8 with no degeneration.

> **No positional compromise.** Unlike Qwen2-VL-family bundles, Mage-VL's language decoder natively uses plain sequential 1-D positions (no M-RoPE), which is exactly what the `fast_vlm` runtime supplies — the deployed decoder contract is mathematically identical to the original model. The 2-D-table-ranking caveat of the Qwen2-VL bundle does not apply here.

> **Image-only bundle.** The base model's video pipeline (neural-codec frame compression) is not included — this bundle handles single images. Send each image in its own message; multiple images in one conversation are untested.

## Run on iPhone / macOS

Use the LiteRT-LM Swift runtime ([swift-litert-lm](https://github.com/google-ai-edge/litert-lm)). Load `Mage-VL.litertlm` with the **vision tower enabled** (`Modality.textImage`), attach a photo, and ask a question.

> Vision-only bundle (no audio tower): bring the engine up with the vision modality only — requesting `.all` fails at session creation on bundles without an audio section.

## Run on Android — Google AI Edge Gallery

Install a recent [Google AI Edge Gallery](https://github.com/google-ai-edge/gallery), download `Mage-VL.litertlm`, import it (tap **+**, enable "Support image"), attach an image and ask. (Verified platforms above are iPhone and macOS; Gallery import follows the same bundle contract as the other fast_vlm models here.)

## Conversion notes

- LiteRT-LM `fast_vlm` bundle: VISION_ENCODER (`[1,448,448,3]`→`[1,784,1024]`) + VISION_ADAPTER (`[1,784,1024]`→`[1,196,2560]`) + single-token EMBEDDER + PREFILL_DECODE (embeddings-input), ChatML prompt with `<|vision_start|>…<|vision_end|>` image markers.
- **Static rewrite of the dynamic-res vision tower.** Mage-VL's ViT is native-resolution (packed patches, `grid_thw`, `cu_seqlens` varlen attention) and does not `torch.export`. The static graph fixes 448×448; a single image is one attention chunk, so the varlen machinery reduces to plain full attention.
- **Direct Conv2d patchify.** `temporal_patch_size=1`, so the patch-embed *is* a stride-16 `Conv2d` — applied to the whole image in raster order (no per-patch reshuffle, no Conv3d fold needed).
- **3-D rope as a constant.** The tower's rotary embedding splits head_dim 4:6:6 over (t,h,w) with interleaved rotation; for a single image t=0, and the (h,w) frequencies are precomputed from raster patch positions and baked into the graph.
- **No GATHER_ND (mobile-GPU-safe graph).** Patches stay in raster order through the (permutation-equivariant) encoder; the 2×2 merge happens in the adapter with 4 strided slices + concat, all ops ≤4D. Static-rewrite corr vs the reference stays 1.0.
- Context is 2048 (vs the trained 262k): the right on-device trade for a 4B decoder — the fp32 KV cache stays ~0.6 GB, keeping the whole session under ~1.5 GiB on the phone.

## License

Apache-2.0, inherited from the base model [microsoft/Mage-VL](https://huggingface.co/microsoft/Mage-VL).
