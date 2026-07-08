---
license: apache-2.0
base_model: Qwen/Qwen2-VL-2B-Instruct
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

# Qwen2-VL-2B-Instruct — LiteRT-LM (on-device Vision-Language Model)

[Qwen/Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) converted to the **LiteRT-LM** (`.litertlm`) format for **on-device image+text** inference with Google's [LiteRT-LM](https://github.com/google-ai-edge/litert-lm) runtime — the first Qwen2-VL-family VLM in this format.

Qwen2-VL is Alibaba's widely-used general-purpose vision-language model: a dynamic-resolution ViT vision encoder feeds the **Qwen2-1.5B** language decoder, giving strong general VQA, description, and document/OCR reading in a phone-sized package. This bundle runs the whole thing through LiteRT-LM's `fast_vlm` multimodal path — give it an image and a question, get a grounded answer, fully on-device.

| | |
|---|---|
| **File** | `Qwen2-VL-2B.litertlm` (~1.78 GB) |
| **Vision** | Qwen2-VL ViT (32L, 1280-dim, full attention) made **static 672×672** → 2304 patches → 2×2 merge → **576 image tokens**, int8 weights |
| **Adapter** | PatchMerger (LN → 2×2 group → MLP), int8, output already at the 1536 text hidden size |
| **Decoder** | Qwen2-1.5B (28L, hidden 1536, GQA kv2), **int4** weights (symmetric, blockwise-32 + OCTAV); int8 externalized embedder |
| **Context (KV cache)** | 4096 |
| **Image input** | resized to 672×672 (OpenAI-CLIP normalization baked into the encoder) |
| **Base model** | Qwen/Qwen2-VL-2B-Instruct (Apache-2.0) |

## Quality

**Device-verified on a Pixel 8a** (Google AI Edge Gallery, vision on GPU + decoder on CPU) and on the desktop LiteRT-LM runtime (macOS CPU):

- **General description / VQA** (photo, on-device): accurate and detailed grounding — an Ansel-Adams-style landscape → "a striking black-and-white photograph … an expansive river winds through a lush, forested valley … to the left … a steep, rocky hillside with rocky cliffs … low hills and mountains, covered in snow, emphasizing the stark nature of the wilderness".
- **Document OCR** (`Read all the text in this image.`, macOS): **perfect transcription** of a synthetic report page — every figure, the e-mail address and the phone number.
- **Counting / reading tables** works (e.g. "there are 5 products listed"); **ranking table cells** across rows is the one weak spot — see the M-RoPE note below.
- Vision tower: static-rewrite vs the reference implementation corr **1.0** (fp32), **0.90** at int8, **zero FLEX/CUSTOM ops** (GPU-clean); the int8 vision is functionally accurate on both VQA and OCR above.
- Decoder: extracted as a standalone Qwen2 model — **bit-exact** fp32 logits vs the original; int4 is the same blockwise-32 + OCTAV recipe used across the shipped Qwen3/Qwen2 LLMs.

> **M-RoPE note & one known limitation.** Qwen2-VL's decoder uses 3-D M-RoPE. The LiteRT-LM `fast_vlm` runtime supplies plain sequential (1-D) positions — for text tokens this is mathematically identical, and it preserves **description, general VQA, counting, and OCR** with no visible loss (A/B-verified against true M-RoPE). The one casualty is **cross-cell comparison reasoning over 2-D structures** — e.g. "which row of this table has the highest value": the 1-D positions flatten the table's 2-D layout, so the model can read every cell correctly (OCR is perfect) but may pick the wrong cell when asked to compare across rows/columns. This is inherent to the runtime (it has no M-RoPE), not the base model or the quantization — the fp32 base with true M-RoPE answers such questions correctly. Use it for reading and describing tables; don't rely on it to *rank* cells.

> **General-purpose VLM.** Unlike an OCR-specialist, Qwen2-VL answers open questions about an image (describe, count, read, reason) as well as extracting text. Ask it anything about the picture — with the 2-D-table-ranking caveat above.

> **One image per chat.** Like the other fast_vlm bundles, send each image in a fresh conversation — a second image in the same chat degrades (context bleed from the first turn was observed on CPU).

## Run on iPhone / macOS

Use the LiteRT-LM Swift runtime ([swift-litert-lm](https://github.com/google-ai-edge/litert-lm)). Load `Qwen2-VL-2B.litertlm` with the **vision tower enabled** (`Modality.textImage`), attach a photo, and ask a question.

> Vision-only bundle (no audio tower): bring the engine up with the vision modality only — requesting `.all` fails at session creation on bundles without an audio section.

## Run on Android — Google AI Edge Gallery

Install a recent [Google AI Edge Gallery](https://github.com/google-ai-edge/gallery), download `Qwen2-VL-2B.litertlm`, import it (tap **+**, enable "Support image"), attach an image and ask.

## Conversion notes

- LiteRT-LM `fast_vlm` bundle: VISION_ENCODER (`[1,672,672,3]`→`[1,2304,1280]`) + VISION_ADAPTER (`[1,2304,1280]`→`[1,576,1536]`) + single-token EMBEDDER + PREFILL_DECODE (embeddings-input).
- **Static rewrite of the dynamic-res vision tower:** Qwen2-VL's ViT is native-resolution (packed patches, 2-D rope, `cu_seqlens`) and does not `torch.export`. The static graph fixes 672×672 and uses full attention (single image = one sequence) with precomputed 2-D rope.
- **Conv3d → Conv2d fold.** The patch-embed is a `Conv3d` (temporal_patch_size=2); for a single image the processor duplicates it into the 2 temporal frames, so the Conv3d over 2 identical slices equals a `Conv2d` with the summed temporal kernel `w[:,:,0]+w[:,:,1]` — GPU-safe, no Conv3d op.
- **No GATHER_ND (this is what lets it run on the phone GPU).** The obvious way to reorder raster patches into the merger's 2×2-block order is a `gather` — but that emits a `GATHER_ND` op, which the mobile GPU delegate cannot compile, so the vision executor fails to create and the whole engine won't load. Instead, patches stay in **raster order** through the encoder (full attention is permutation-equivariant, so ordering is irrelevant as long as each patch carries its own 2-D rope), and the 2×2 merge is done in the adapter with **4 strided slices + concat** (`f[:,0::2,0::2] … f[:,1::2,1::2]`), all ops ≤4D. Static-rewrite corr vs the reference stays **1.0**.
- Decoder: the Qwen2-1.5B text model inside the VLM is re-hosted as a standalone `Qwen2ForCausalLM` (state-dict 1:1, bit-exact logits, lm_head tied to embeddings) and exported with the standard litert-torch path, cache 4096.

## License

Apache-2.0, inherited from the base model [Qwen/Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct).
