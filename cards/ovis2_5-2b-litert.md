---
license: apache-2.0
base_model: AIDC-AI/Ovis2.5-2B
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

# Ovis2.5-2B — LiteRT-LM (on-device Vision-Language Model)

[AIDC-AI/Ovis2.5-2B](https://huggingface.co/AIDC-AI/Ovis2.5-2B) converted to the **LiteRT-LM**
(`.litertlm`) format for **on-device image+text** inference with Google's
[LiteRT-LM](https://github.com/google-ai-edge/litert-lm) runtime (the engine behind the official
`litert-community/*` models, and the same runtime that runs `litert-community/FastVLM-0.5B`).

Ovis2.5 is a **SOTA-for-size vision-language model** (OpenCompass ~73.9 for the 2B) with a distinctive
**structural-embedding** vision path: a **Siglip2 NaViT** encoder feeds a *visual tokenizer* that turns
each image patch-group into a probability distribution over a **65 536-word visual vocabulary**, then
embeds it — giving the language model image tokens that live in the same structured space as text.
The language decoder is **Qwen3-1.7B**. This bundle runs the whole thing through LiteRT-LM's `fast_vlm`
multimodal path — give it an image and a question, get a grounded answer, fully on-device.

| | |
|---|---|
| **File** | `Ovis2.5-2B.litertlm` (~2.15 GB) |
| **Vision** | Siglip2-NaViT encoder + visual-tokenizer (head → softmax → visual-vocab embedding), **int8** weights — single **512×512** image → 256 image tokens |
| **Decoder** | Qwen3-1.7B, **int4** weights (symmetric, **blockwise-32 + OCTAV** optimal-clipping); input embedding INT8 (externalized section) |
| **Compute** | integer |
| **Context (KV cache)** | 2048 |
| **Image input** | resized to 512×512 (Siglip normalization is baked into the vision encoder) |
| **Base model** | AIDC-AI/Ovis2.5-2B (Apache-2.0) |

## Quality

The vision tower converts **bit-faithfully** to the reference — float CPU-parity **end-to-end
corr ≈ 1.0** (max abs diff ~3e-6), with **no FLEX/CUSTOM fallback ops**; int8 vision weights keep
end-to-end corr **~0.99**. The Qwen3-1.7B decoder uses the same **blockwise-32 + OCTAV int4** recipe
that scores 90.7% GSM8K on the sibling
[Ministral-3-3B-Reasoning build](https://huggingface.co/litert-community/Ministral-3-3B-Reasoning-2512)
and shipped the [InternVL3.5-2B](https://huggingface.co/litert-community/InternVL3_5-2B) VLM.
On a reference **deployed-path** eager run (fixed-512 vision → 256 tokens → Qwen3-1.7B) the model
describes real photos accurately and in detail (e.g. a black-and-white Ansel-Adams-style landscape →
"snow-capped sharp mountain peaks … a river winding through the valley … cloud layers … black-and-white
contrast with depth of field").

> **Reasoning VLM.** Ovis2.5 is a *thinking* model: it may emit a `<think>…</think>` block before its
> final answer (this matches the base model). Allow enough max-tokens (≥1024) for the answer to follow.

> **On-device performance:** decode/load are expected to be in line with the InternVL3.5-2B build on
> the same runtime (~20 tok/s CPU, ~45 tok/s GPU on iPhone 17 Pro for single-image VQA). Independent
> on-device measurement for this specific build is recommended before quoting exact numbers.

## Run on iPhone / macOS

Use the LiteRT-LM Swift runtime ([swift-litert-lm](https://github.com/google-ai-edge/litert-lm) / the
`LiteRTDemo` sample). Load `Ovis2.5-2B.litertlm` with the **image (vision) tower enabled** (modalities
`[.vision]` / `Modality.textImage`), attach a photo, and ask a question.

> Note for app integrators: this is a **vision-only** bundle (no audio tower). Bring up the engine with
> the **vision** modality only — requesting the audio tower (`.all`) on a bundle with no audio section
> fails at session creation.

## Run on Android — Google AI Edge Gallery

Install a recent [Google AI Edge Gallery](https://github.com/google-ai-edge/gallery) (1.0.16+ can
import `.litertlm` directly from Hugging Face), download `Ovis2.5-2B.litertlm`, import it (tap **+**),
attach an image and ask. The bundle already carries the tokenizer and prompt template.

## Conversion notes

- LiteRT-LM `fast_vlm` bundle: VISION_ENCODER (`[1,512,512,3]`→`[1,256,4608]`) + VISION_ADAPTER
  (`[1,256,4608]`→`[1,256,2048]`, matched to the Qwen3-1.7B hidden size) + single-token EMBEDDER +
  PREFILL_DECODE (embeddings-input).
- **The NaViT static rewrite** is the enabling trick. Ovis's Siglip2-NaViT vision tower is *dynamic
  resolution* (`.item()`/`.tolist()`/grid-loops/`argsort`) and does not `torch.export`. Because the
  config's `fullatt_block_indexes=None` makes every layer use **full attention**, the window-reorder is
  a mathematical no-op — so it can be dropped and replaced with a **precomputed position embedding +
  rotary** and a single full attention over the fixed 512×512 grid (1024 patches). Static-vs-original
  feature corr **0.99999964**.
- The encoder bakes Siglip normalization (`(x-0.5)/0.5`, the runtime feeds a `[0,1]` NHWC image) and
  does patchify **GPU-safe**: the patch-embedding Conv2d is applied to the whole image (raster order),
  then a single **gather** reorders patches into Ovis's hidden-stride "merge" order — all reshapes ≤4D,
  no `>5D` op that GPU delegates reject.
- The adapter is Ovis's visual-tokenizer tail: `head` (Linear 4608→65532 + LayerNorm) → softmax →
  visual-vocabulary embedding (`vte`, 65536×2048). The 256-token bundle carries the visual **atoms**;
  Ovis's two learned image-boundary indicator embeddings are omitted (the `fast_vlm` path splices only
  the atom embeddings) — verified to stay coherent in eager.
- Decoder extracted from the Ovis2_5 wrapper as a standalone `Qwen3ForCausalLM` and exported with cache
  ≤ base max so base RoPE is exact.

## License

Apache-2.0, inherited from the base model [AIDC-AI/Ovis2.5-2B](https://huggingface.co/AIDC-AI/Ovis2.5-2B).
