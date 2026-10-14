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
base_model_relation: quantized
---

Measured on device (edge-compat): Galaxy S26 · LiteRT-LM 0.16.0 · GPU · decode 10.4 tok/s · prefill 280 tok/s · TTFT 810 ms · all 1380 ops delegated (2026-08-24); Pixel 8a · LiteRT-LM 0.16.1 · GPU · decode 4.3 tok/s · prefill 124 tok/s · TTFT 2.41 s · all 1380 ops delegated (2026-08-19); iPhone 17 Pro · LiteRT-LM 0.16.0 · GPU · decode 13.6 tok/s · prefill 244 tok/s · TTFT 685 ms (2026-08-19). Record: https://github.com/john-rocky/edge-compat/blob/main/cards/north-micro-vision-instruct-wi8/CARD.md

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

iPhone 17 Pro (Metal decoder + Metal vision, cold, this exact 3.07 GB file): peak resident **2.9–3.0 GB** with the vision executor loaded (2.1 GB text-only); decode **13.6 tok/s** on a 39-token text turn; vision time-to-first-token 2.3–5.0 s cold (302-token image prompt, prefill 322–362 tok/s). Same phone, same model through Apple's Core AI runtime (own measurement, 2026-08-14, int8 decoder + fp16 tower): **18.2 tok/s** decode, 21.5 tok/s prefill, image oracle 24/24. Read it as "same class, Core AI ahead on decode (~1.3×), LiteRT-LM ahead on prefill and vision TTFT"; both numbers are cold single runs on the same device and the same day-class build, not a tuned benchmark.

## Galaxy S26 — GPU backend

Both published bundles run on the Android GPU backend and generate.

| file | GPU backend | delegation | peak |
|---|---|---|---:|
| `North-Micro-Vision-Instruct_int4.litertlm` | runs | 5344 / 5344 ops across 4 subgraphs on `LiteRT GPU` | 1787 MB |
| `North-Micro-Vision-Instruct_wi8.litertlm` | runs | 5344 / 5344 ops across 4 subgraphs on `LiteRT GPU` | 1403 MB |

Measured on a **Samsung Galaxy S26** (SM-S942Q / SM8850, Android 16) with `litert_lm_advanced_main` from litert-lm 0.16.0, `--backend=gpu --sampler_backend=cpu`, prompt `What is the capital of France?`. Peak is the process high-water mark (`VmHWM`) sampled during that same run. Gated 2026-08-25.

The op counts above are the `LiteRT GPU` partitions. XNNPACK additionally takes 1 of the 4 nodes in `main`; the runtime accepts that split.

The gate prompt carries no image, so this covers engine creation and the text path. The vision path on the GPU is not measured here.

**No speed rows, on purpose.** On this handset the GPU backend wins prefill and does not win decode, so a GPU throughput figure only means something beside a CPU row from the same handset, and no S26 CPU row exists for this model yet.

GPU wiring, including the Gallery import toggle: [GPU guide](https://github.com/john-rocky/hf-to-litertlm/blob/main/docs/android-gpu.md).

## On-device demo (Pixel 8a, this exact file, decoder on the OpenCL GPU + vision on CPU)

| | |
|---|---|
| <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png" width="330"/> | <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg" width="330"/> |
| **Q:** Describe this image.<br>**A:** "The image shows two cats lying on a pink surface, which appears to be a couch or a cushion. Both cats are in a relaxed position, with their bodies stretched out and their heads resting on the surface. They are both wearing collars, indicating they are pets. The cats are positioned close to each other, suggesting a sense of companionship…" | **Q:** What is in this image? Mention the colors.<br>**A:** "The image shows a vintage Volkswagen Beetle car parked in front of a building with a yellow wall and two wooden doors. The car is painted in a light teal or turquoise color, and it has a classic design with round wheels and a rounded body shape. The building behind the car has a weathered appearance… The ground is paved with rectangular tiles." |
| **Q:** How many cats are there, and what is between them?<br>**A:** "There are two cats in the image, and they are both lying on a pink surface. Between them, there are two remote controls." | **Q:** Where is the car parked, and what is behind it?<br>**A:** "The car is parked on a paved street, and behind it is a yellow wall with two wooden doors. The setting appears to be an urban or semi-urban area." |

Verbatim outputs from `litert_lm_advanced_main` (litert-lm v0.16.1, `--backend=gpu --vision_backend=cpu --disable_cache`, greedy, 80-token cap) on a Pixel 8a; per turn: TTFT 2.5–2.8 s (≈270-token image prompt, prefill 110–128 tok/s), decode 2.9–3.6 tok/s. The photos are the Hugging Face documentation sample images (COCO cats / Beetle), resized by the runtime to the bundle's 512×512.

## Quality

- **9-case COCO suite** (3 images × 3 questions, 48-token greedy) against a fp32 PyTorch oracle running the same single-embedding / 1-D-position contract: content-correct and image-grounded on 9/9 (cats on a pink couch with remotes, the two kitchens, colour palettes, "where is this scene"); the int8 vision encoder shifts token choices, so token-exact is 1/9 (the desktop fp16-vision build keeps 5/9 token-exact).
- **8-question text gate: 7/8** with the earlier files, non-degenerate. The one miss ("17 + 25" read as "1.7 + 2.5") is the same digit misreading as "17 times 23" (see 2026-10-15 below); the gate has not been re-run on the re-packed files.
- **Device**: Pixel 8a Ask-Image 2/2 grounded ("two cats lying on a pink surface, possibly a couch or a bed…", "warm brown wooden counter, black stove, white apron, hanging pots and pans"); iPhone 17 Pro vision-grounding probes 2/2 ("Does this image contain visible written text?" → **No** on the no-text fractal, **Yes** on the text probe), 8-question text gate 6/8 on-device with the earlier file (the same digit miss plus one rhyme miss).
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
- **Tokenizer**: the model's own `tokenizer.json` is bundled as-is since 2026-10-15 (byte-level BPE, 255k + 38 special tokens); the files before carried the copy transformers' `CohereTokenizer` class writes, which splits every digit (see 2026-10-15 below).
- Reproduce: [hf-to-litertlm](https://github.com/john-rocky/hf-to-litertlm) — `bash scripts/reproduce_vlm.sh north-micro-vision` (`REPRODUCE.md` has the full recipe and gotchas).

### 2026-08-31 — chat template fix: history turns with role 'assistant' (metadata-only, weights unchanged)

The chat template in `North-Micro-Vision-Instruct_int4.litertlm`, `North-Micro-Vision-Instruct_wi8.litertlm` matched history turns only on role `'model'`, so a conversation created through the Conversation API with the standard role `'assistant'` (for example `create_conversation(messages=[...])` restoring a chat) silently dropped every assistant turn from the rendered prompt — the model saw consecutive user turns with its own replies gone. The template's role conditions now also accept `'assistant'`. The engine's own `send_message` flow was never affected.

**Template-only change**: every section of the bundle except the metadata is byte-identical to the previous file (verified by per-section sha256), and single-turn renders, token ids, prefill counts and greedy answers are unchanged (verified on the LiteRT-LM runtime), so the numbers on this card still describe this file — only the file's own sha256 differs. Re-download only matters if you inject conversation history with role `'assistant'`. Details: google-ai-edge/LiteRT-LM#3417 (https://github.com/google-ai-edge/LiteRT-LM/issues/3417).

### 2026-10-15 — tokenizer fix: digits (tokenizer section only, weights unchanged)

**Digits.** Since the 2026-10-15 re-pack (tokenizer section replaced by the upstream tokenizer), "17 times 23" is read as 17 × 23 and answered 391; the earlier files read it as 1.7 × 2.3 on every digit check of litert-lm 0.18.0 (the int4 file answered 3.91).

The earlier files carried the `tokenizer.json` that transformers' `CohereTokenizer` class writes: it splits every digit into its own token, so "17" reached the model as "1", "7". The model's own `tokenizer.json` groups numbers in runs of up to three digits. The tokenizer section is now that file byte for byte (CohereLabs/North-Micro-Vision-Instruct at `fa548d9a`, the commit where upstream made transformers v5 use it verbatim); the vocabulary and token ids are unchanged. Every other section, the metadata included, is byte-identical to the previous file (verified by per-section sha256), so the speed numbers on this card still describe this file. Text other than digits also tokenizes as upstream does now (upper-case contractions, runs of newlines or tabs, decomposed accents), so the verbatim outputs and gate scores above, recorded with the earlier files, can differ on the new ones. The new section tokenizes identically on litert-lm 0.15.0 through 0.18.0 (286 probe texts each).

## License

Apache-2.0, inherited from the base model [CohereLabs/North-Micro-Vision-Instruct](https://huggingface.co/CohereLabs/North-Micro-Vision-Instruct).
