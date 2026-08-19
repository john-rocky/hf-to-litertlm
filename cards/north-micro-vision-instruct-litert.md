---
license: apache-2.0
base_model: CohereLabs/North-Micro-Vision-Instruct
tags:
  - litert
  - litert-lm
  - litertlm
  - on-device
  - edge
  - vlm
  - multimodal
  - vision-language
  - cohere
pipeline_tag: image-text-to-text
library_name: litert-lm
---

# North-Micro-Vision-Instruct — LiteRT-LM (on-device Vision-Language Model)

[CohereLabs/North-Micro-Vision-Instruct](https://huggingface.co/CohereLabs/North-Micro-Vision-Instruct) converted to the **LiteRT-LM** (`.litertlm`) format for **on-device image+text** inference with Google's [LiteRT-LM](https://github.com/google-ai-edge/litert-lm) runtime — the first Cohere-family model in this format.

North-Micro-Vision is Cohere's 2.48B VLM: a 400M SigLIP2-SO400M-scale vision tower (Qwen3-VL-style encoder with DeepStack mergers) feeding a 2B Cohere decoder, trained for 11 languages. This bundle runs it through LiteRT-LM's `fast_vlm` path — attach an image, ask a question, get a grounded answer fully on-device.

| | |
|---|---|
| **Files** | `North-Micro-Vision-Instruct_wi8.litertlm` (3.07 GB, **primary**) · `North-Micro-Vision-Instruct_int4.litertlm` (2.19 GB, size-constrained variant) |
| **Vision** | 27-block ViT (hidden 1152, 16 heads, patch 16) made **static 512×512** → 1024 patches → 2×2 merge → **256 image tokens**; DeepStack (3 extra vision embeddings) **folded** into the single image embedding; int8 weights |
| **Adapter** | Patch merger + the three DeepStack mergers, summed; int8; output at the 2048 text hidden size |
| **Decoder** | 2B Cohere decoder (28L, hidden 2048, GQA kv8, parallel attn+MLP blocks, sliding/full 3:1, tied 262k-vocab embedding) — **int8** dynamic weights (wi8) or int4 blockwise-32; int8 externalized embedder; mixed-precision activations (`fp32_fp16`) declared in-bundle |
| **Context (KV cache)** | 4096 |
| **Image input** | resized to 512×512 (normalization `(x/255−0.5)/0.5` baked into the encoder) |
| **Chat format** | Cohere turn tokens (`<|START_OF_TURN_TOKEN|><|USER_TOKEN|>…<|END_OF_TURN_TOKEN|>`), BOS id 2, image rendered as `<|VISION_START|><image_soft_token><|VISION_END|>` |
| **Base model** | CohereLabs/North-Micro-Vision-Instruct (Apache-2.0) |

## Performance (measured)

Apple M4 Max, litert-lm 0.16.0, `litert-lm benchmark -p 256 -d 256 --runs 3 --cache no`, wi8 bundle:

| Backend | Prefill (tok/s) | Decode (tok/s) | Init (s) |
|---|---|---|---|
| CPU | 671.8 | 27.3 | 6.7 |
| GPU | 1236.6 | 80.6 | 1.8 |

(GPU rows verified with a real image generation on `--backend gpu --vision-backend gpu`, not benchmark-only.)

Pixel 8a (litert-lm v0.16.1 CLI, `--disable_cache`), wi8 bundle — the decoder fully delegates to the OpenCL GPU (1380/1380 prefill, 1204/1204 decode nodes):

| Config | Prefill (tok/s) | Decode (tok/s) |
|---|---|---|
| decoder GPU + vision CPU, image turn (271-token prefill) | 124.5 | 4.3–4.4 |
| decoder GPU, text-only (255-token prefill) | 181.1 | 6.4 |

Run the decoder on the GPU on 8 GB-class phones — the CPU backend pages against the 2.5 GB decoder (0.6 tok/s).

iPhone 17 Pro (Metal decoder + Metal vision, cold): the 3.5 GB bundle loads (peak 3.46 GB resident); decode **24.9 tok/s** on a short vision turn / **14.6 tok/s** on a 50-token text turn; vision time-to-first-token 8.1–9.5 s. Same phone, same model through Apple's Core AI runtime (own measurement, 2026-08-14, int8 decoder + fp16 tower): **18.2 tok/s** decode, 21.5 tok/s prefill, image oracle 24/24 — the two runtimes land within ~35% of each other on decode, LiteRT-LM ahead on short vision turns and behind on longer text turns; the honest read is "same class", with the caveat that the LiteRT number for the final int8-vision composition is pending (the iPhone figures above are from the fp16-vision build of the same decoder).

## Quality

- **9-case COCO suite** (3 images × 3 questions, 48-token greedy) against a fp32 PyTorch oracle running the same single-embedding / 1-D-position contract: content-correct and image-grounded on 9/9 (cats on a pink couch with remotes, the two kitchens, colour palettes, "where is this scene"); the int8 vision encoder shifts token choices, so token-exact is 1/9 (the desktop fp16-vision build keeps 5/9 token-exact).
- **8-question text gate: 7/8**, non-degenerate. The one miss ("17 + 25" read as "1.7 + 2.5") reproduces token-for-token on the HF fp32 model — Cohere's per-digit pre-tokenizer, not a conversion artifact.
- **Device**: Pixel 8a Ask-Image 2/2 grounded ("two cats lying on a pink surface, possibly a couch or a bed…", "warm brown wooden counter, black stove, white apron, hanging pots and pans"); iPhone vision probes track the image (no-text fractal → "No, this image…"; text probe → reads the text).
- **int4 variant**: coherent and content-correct on all 9 suite cases but terser and further from the fp32 wording; prefer wi8 where storage allows.

> **What the fast_vlm contract changes, and what it costs.** The released model injects three DeepStack vision embeddings after decoder layers 0/1/2 and uses interleaved M-RoPE. This bundle folds the DeepStack embeddings into the single image embedding (exactly representable; teacher-forced top-1 vs the released model 0.96 fold-only, 0.93 with the runtime's 1-D positions) and the runtime supplies plain sequential positions in place of M-RoPE. Measured effect on probe prompts: **describe / VQA / spatial relations / single-cell lookup preserved**; **2-D table cross-cell questions and digit-dense OCR degrade** (row count off-by-one, "$652,000" read as "$652,000,000", a duplicated word in a dense paragraph). Same class of trade as the Qwen2-VL-2B bundle. Use it for reading and describing; don't rely on it to rank table cells.

> **One image per chat.** Send each image in a fresh conversation, as with the other fast_vlm bundles.

## Run on Android — Google AI Edge Gallery

Install a recent [Google AI Edge Gallery](https://github.com/google-ai-edge/gallery), download `North-Micro-Vision-Instruct_wi8.litertlm`, import it (tap **+**, enable "Support image"), attach an image and ask. Choose the **GPU** accelerator for the decoder.

CLI:
```
litert-lm run North-Micro-Vision-Instruct_wi8.litertlm \
  --prompt "What is in this image?" --attachment photo.jpg \
  --backend gpu --vision-backend cpu
```

## Run on iPhone / macOS

Use the LiteRT-LM Swift runtime ([swift-litert-lm](https://github.com/google-ai-edge/litert-lm)). Load the bundle with the **vision tower enabled** (`Modality.textImage`), attach a photo, and ask. Vision-only bundle (no audio tower): bring the engine up with the vision modality only.

## Conversion notes

- LiteRT-LM `fast_vlm` bundle: VISION_ENCODER (`[1,512,512,3]`→`[1,1024,4608]` — the final block plus the three DeepStack taps, concatenated) + VISION_ADAPTER (`[1,1024,4608]`→`[1,256,2048]`) + single-token EMBEDDER + PREFILL_DECODE (embeddings-input, cache 4096).
- **DeepStack fold.** The three DeepStack mergers' outputs are added to the main merger output inside the adapter, so the decoder receives one embedding per image token that carries all four vision taps.
- **Static rewrite of the dynamic-res tower**: fixed 32×32 grid, precomputed 2-D rope and resampled learned position embedding, full attention; `Conv3d` (temporal 2) folded to `Conv2d` with the summed temporal kernel; patches kept in raster order through the encoder with the 2×2 merge done by strided slices + concat in the adapter (**no GATHER_ND**, which the mobile GPU delegate cannot compile). Every activation keeps a leading batch dim (rank ≥3) — required for correct results on the Metal GPU delegate. LayerNorm inputs are pre-scaled by calibrated powers of two so the tower survives fp16 GPU precision.
- **Decoder re-host.** The Cohere decoder is re-hosted as a standalone `Cohere2ForCausalLM` (parallel block, mean-subtracting LayerNorm, NoPE full-attention layers, logit scale 0.25, tied head) with its rotary layout patched to the checkpoint's half-split convention — text-only logits identical to the original (max |Δ| = 0.0). The decoder section declares `prefer_activation_type = fp32_fp16`: some mobile GPUs accumulate fp16 and overflow at the image-token positions, which blanks the vision conditioning (the model answers as if it saw nothing); the in-bundle declaration selects mixed precision automatically.
- **Tokenizer**: the HF `tokenizer.json` is bundled as-is (byte-level BPE, 255k + 38 special tokens).
- Reproduce: [hf-to-litertlm](https://github.com/john-rocky/hf-to-litertlm) — `bash scripts/reproduce_vlm.sh north-micro-vision` (`REPRODUCE.md` has the full recipe and gotchas).

## License

Apache-2.0, inherited from the base model [CohereLabs/North-Micro-Vision-Instruct](https://huggingface.co/CohereLabs/North-Micro-Vision-Instruct).
