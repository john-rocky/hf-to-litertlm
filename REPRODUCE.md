# Reproducing the shipped LLM conversions

Every dense / reasoning LLM `.litertlm` shipped from this repo is reproducible with one command:

```bash
bash scripts/reproduce_llm.sh <model-key>     # -> out/<key>/model.litertlm
bash scripts/reproduce_llm.sh --list          # all keys
bash scripts/reproduce_llm.sh --all           # everything (heavy)
```

The engine is `scripts/export_simple_template.py`
(`<hf_model> out/<name> templates/<template>.jinja <quant>` + env). Recipes reconstructed **2026-07-06**
from `cards/*.md` + auto-memory + `reports/*` while the memory was fresh (a few env/template fields are
best-inference, flagged below). The tables here are the dense/reasoning LLMs; the **Vision-language
models** section at the bottom covers the VLMs (`scripts/reproduce_vlm.sh`).

## Verified by running (2026-07-07)

Every recipe below was **actually executed** and gated (`verify_quality.py`, 8-question gate, max-tokens 2048):
**16/18 reproduce + pass the gate.** The two exceptions are not recipe errors:
- **`smollm3-3b`** — reproduces + answers **8/8 correct**, but the gate flags one verbose thinking-model
  answer as degenerate (a gate-strictness artifact on the `smollm3_think` template; the shipped model is
  GSM8K 0-pt parity). Recipe is correct.
- **`fastcontext-4b`** — the source repo `microsoft/FastContext-1.0-4B-SFT` is now **private/gated**; it
  can't be re-downloaded without an HF token. Recipe is verbatim-confirmed from `reports/fastcontext-4b-parity.md`.
- **`qwen25-3b`** — the *original* recipe (`gptqrec` dequant-recovery) was found to **fail on the current
  `ai_edge_quantizer`** during this run; switched to the version-robust **`BMIX4_128`** path (now passes 8/8).
- `falcon3-3b` passes 7/8 (its int4 is known-not-parity; shipped withheld). `r1-distill-qwen-1.5b` passes 6/8
  (shallow 1.5B). All others pass 8/8.

**Recipe legend:** `BOCTAV4` = blockwise-32 int4 + OCTAV + int8 embedding (best quality, Mac/Android) ·
`BOCTAV4_128` = blockwise-128 variant (iPhone / 4B, fits the ~2 GiB section) · `BMIX4[_128]` = blockwise
int4 min-max (no OCTAV) + int8 embedding. **Env:** `FORCE_SPM` (BPE→SP tokenizer; auto-enables
`FIX_ADDED_TOKENS` for `<think>` models) · `EXTERNALIZE_EMBEDDER` (split embedding so 3B+ loads on iPhone) ·
`PHI3_STATIC_ROPE` (Phi longrope→static) · `GPTQREC_GCD_FIX` (GPTQ ingest) · `CACHE` / `PREFILL`.

**Template safety:** every recipe here exports with `use_jinja_template=False` — the script swaps the vendor's HF chat template for a minimal ChatML one (`templates/*.jinja`), and the converter packs it as plain prefix/suffix turn markers, so the bundle embeds **no Jinja at all**. A plain litert-torch export instead defaults to `use_jinja_template=True`, embedding the vendor template verbatim; many vendor templates call Python-style methods (`.get()`, `.startswith()`) that LiteRT-LM's minijinja renderer doesn't implement — such a bundle imports fine and then dies on the first message (Edge Gallery: `Failed to apply template: unknown method: map has no method named get`). Triage any bundle with `pip install litert-lm-builder && python -m litert_lm_builder.litertlm_peek_main --litertlm_file model.litertlm`: `prompt_templates`-only = safe; a `jinja_prompt_template` carrying Python-method calls = the crasher.

## Single-command models

| key | HF source | template | quant | env | shipped to |
|---|---|---|---|---|---|
| `fastcontext-4b` | microsoft/FastContext-1.0-4B-SFT | chatml_simple | BOCTAV4 | EXTERNALIZE_EMBEDDER, CACHE=4096 | litert-community/FastContext-1.0-4B-SFT |
| `nanbeige4.1-3b` | Nanbeige/Nanbeige4.1-3B | chatml_simple | BOCTAV4 | FORCE_SPM, EXTERNALIZE_EMBEDDER, CACHE=4096 | litert-community/Nanbeige4.1-3B |
| `nanbeige4.2-3b` | Nanbeige/Nanbeige4.2-3B | chatml_think | BOCTAV4 | FORCE_SPM, EXTERNALIZE_EMBEDDER, CACHE=4096; dedicated `convert_nanbeige42.py` (looped transformer: 22 shared-weight layers ×2 loops → 44 KV slots) | litert-community/Nanbeige4.2-3B |
| `olmo2-1b` | allenai/OLMo-2-0425-1B-Instruct | olmo2_simple | BOCTAV4 | CACHE=4096 | mlboydaisuke/OLMo-2-1B-Instruct-LiteRT |
| `olmo2-7b` | allenai/OLMo-2-1124-7B-Instruct | olmo2_simple | BOCTAV4 | CACHE=4096, EXTERNALIZE_EMBEDDER | *(desktop-only, not published — >2 GiB section)* |
| `polaris-4b` | POLARIS-Project/Polaris-4B-Preview | qwen3_think | BOCTAV4_128 | EXTERNALIZE_EMBEDDER, CACHE=4096 | litert-community/Polaris-4B-Preview |
| `qwen3-1.7b` | Qwen/Qwen3-1.7B | qwen3_think | BOCTAV4 | CACHE=4096 | mlboydaisuke *(dropped→private)* |
| `qwen3-4b-thinking` | Qwen/Qwen3-4B-Thinking-2507 | qwen3_think | **BOCTAV4_128** | EXTERNALIZE_EMBEDDER, CACHE=4096 | litert-community/Qwen3-4B-Thinking-2507 |
| `r1-distill-qwen-1.5b` | deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | deepseek_r1_simple | BOCTAV4 | CACHE=4096 | mlboydaisuke/DeepSeek-R1-Distill-Qwen-1.5B-LiteRT |
| `r1-distill-qwen-7b` | deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | deepseek_r1_simple | BOCTAV4 | CACHE=4096, EXTERNALIZE_EMBEDDER | mlboydaisuke/DeepSeek-R1-Distill-Qwen-7B-LiteRT *(desktop)* |
| `smollm3-3b` | HuggingFaceTB/SmolLM3-3B | smollm3_think | BOCTAV4 | CACHE=4096, EXTERNALIZE_EMBEDDER | mlboydaisuke/SmolLM3-3B-LiteRT |
| `jan-nano` | Menlo/Jan-nano | qwen3_think ⚠ | BOCTAV4_128 | EXTERNALIZE_EMBEDDER, CACHE=4096 | litert-community/Jan-nano |
| `vibethinker-3b` | WeiboAI/VibeThinker-3B | chatml_simple | **BOCTAV4** (block32 ONLY) | CACHE=4096, EXTERNALIZE_EMBEDDER | litert-community/VibeThinker-3B |
| `falcon3-3b` | tiiuae/Falcon3-3B-Instruct | falcon_simple | BMIX4_128 | CACHE=2048 | *(withheld/private — int4 ≠ parity)* |
| `llama32-3b` | meta-llama/Llama-3.2-3B-Instruct | llama_simple | BMIX4 | EXTERNALIZE_EMBEDDER, CACHE=4096 | mlboydaisuke/Llama-3.2-3B-Instruct-LiteRT |

## Models with a prep step (the runner does it automatically)

| key | prep | then export |
|---|---|---|
| `ministral3-3b` | `extract_ministral3_text.py mistralai/Ministral-3-3B-Instruct-2512 → src_models/ministral3-3b-text` (drop pixtral vision) | mistral_simple, BOCTAV4, EXTERNALIZE_EMBEDDER, CACHE=4096 → litert-community/Ministral-3-3B-Instruct-2512 |
| `ministral3-3b-reasoning` | `extract_text_backbone.py mistralai/Ministral-3-3B-Reasoning-2512 → src_models/…-reasoning-text` | mistral_simple, BOCTAV4, **FORCE_SPM**, EXTERNALIZE_EMBEDDER, CACHE=4096 → litert-community/Ministral-3-3B-Reasoning-2512 |
| `phi4-mini-reasoning` | download microsoft/Phi-4-mini-reasoning, set `config.sliding_window=None` | phi_simple, BOCTAV4, **PHI3_STATIC_ROPE**, EXTERNALIZE_EMBEDDER, CACHE=4096 → litert-community/Phi-4-mini-reasoning |
| `qwen25-3b` | `ingest_gptq_dequant.py Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4 Qwen/Qwen2.5-3B-Instruct … fp32clip` (dequantize GPTQ) | chatml_simple, **BMIX4_128**, CACHE=4096 → mlboydaisuke *(NC license, personal only)* |

## Caveats / lower-confidence fields (verify against the card + memory before quoting)

- **`jan-nano` template** ⚠ — no source names it. It's a Qwen3-4B thinking model, so `qwen3_think.jinja`
  (matches the sibling Qwen3-4B-Thinking and the reasoning-template-parity note); `chatml_simple.jinja`
  (what same-base FastContext used) is the alternative. Swap and re-gate if output rambles.
- **`llama32-3b`** was originally exported through the **official litert-torch main** (BPE patch upstream) —
  the current `~/clipconv` reproduces the same `BMIX4` recipe; expect equivalent, not bit-identical.
- **`vibethinker-3b`** needs a runtime stop-token fix (`generation_config.eos_token_id=[151643,151645]` so
  `<|im_end|>` ends ChatML turns) and is **block32-only** (block128 collapses 90→64% GSM8K).
- **`qwen3-4b-thinking`** is **block128-only** (block32 corrupts on iPhone GPU). Reasoning models: eval at
  max-tokens ≥ 2048 or int4 falsely looks degraded.
- **Ministral source ids** — the extract scripts accept an HF id or a local dir; if the plain id 404s, use the
  `*-BF16` variant of the repo (the bf16 weights the extraction was run on).
- **`qwen25-3b` GPTQ path** — the original ship used `recipes/gptqrec_int4_block128.json` + `GPTQREC_GCD_FIX`
  (dequantized-weight-recovery), but the **current `ai_edge_quantizer` rejects blockwise + recovery**
  (`dequantized_weight_recovery.py:163`). The runner therefore uses **`BMIX4_128`** on the `fp32clip`
  dequantized checkpoint — the ingest script's own docstring path (min-max lands the grid on the ±7 levels
  → same recovery, version-robust). Use the recovery recipe only on an ai_edge_quantizer that supports it.
- **Not published** (reproducible, but were withheld): `olmo2-7b` / `r1-distill-qwen-7b` (desktop-only size),
  `falcon3-3b` (naive int4 ≠ bf16 parity), `qwen3-1.7b` (dropped), `qwen25-3b` (Qwen NC license → personal
  namespace only).

## Vision-language models (`fast_vlm`)

VLMs reproduce via `bash scripts/reproduce_vlm.sh <key>` (each runs a `ship_*.sh` that downloads the
source, converts the vision encoder+adapter and the decoder — int4 unless noted — and assembles the
bundle). Details per model in `cards/<name>-litert.md`.

| key | vision | decoder | ship script |
|---|---|---|---|
| `internvl3-1b` | InternViT-448 | Qwen2.5-0.5B | `ship_internvl_1b.sh` |
| `internvl3.5-1b` / `-2b` / `-4b` | InternViT-448 | Qwen3-0.6B / 1.7B / 4B | `ship_internvl3_5_{1b,2b,4b}.sh` |
| `llava-onevision-0.5b` | SigLIP-384 (730 tok) | Qwen2-0.5B | `ship_llavaov.sh` |
| `mage-vl` | **static-448, no GATHER_ND** (196 tok) | Qwen3-4B **int4-b128**, cache 2048 | `ship_magevl.sh` |
| `ovis2.5-2b` | **static-NaViT-512** (256 tok) | Qwen3-1.7B | `ship_ovis_2b.sh` |
| `paddleocr-vl-1.6` | **static-NaViT-560** (400 tok) | ERNIE-4.5-0.3B (**fp16** — int4/int8 corrupt OCR) | `ship_paddleocr_vl.sh` |
| `qwen2-vl-2b` | **static-672, no GATHER_ND** (576 tok) | Qwen2-1.5B int4 | `ship_qwen2vl_2b.sh` |
| `smolvlm2-500m` / `-2.2b` | SigLIP + pixel-shuffle | SmolLM2 / SmolLM2-1.7B | `ship_smolvlm2{,_22b}.sh` |

`qwen2-vl-2b` is the general-purpose Qwen2-VL VLM (describe / VQA / OCR). Two gotchas baked into its scripts: (a) reordering patches into the merger's 2×2-block order with a gather emits a `GATHER_ND` op that the mobile GPU delegate can't compile (the vision executor then fails to create on-device) — so the encoder keeps raster order and the 2×2 merge is done with strided slices + concat in the adapter; (b) the fast_vlm runtime feeds 1-D positions (no M-RoPE), which preserves describe/VQA/OCR/count but degrades cross-cell *ranking* over 2-D tables.

`mage-vl` is Microsoft's 4.7B general VLM (describe / VQA / full-page OCR). Its vision tower is the Qwen2-VL design, but two things make it *simpler* to convert: `temporal_patch_size=1` (the patch-embed already is a stride-16 `Conv2d` — no Conv3d fold) and the text decoder is a **stock Qwen3-4B fed plain 1-D positions**, so the fast_vlm runtime contract is mathematically exact (no M-RoPE caveat at all — the 2-D-table-ranking limitation of `qwen2-vl-2b` does not apply). Its 3-D rope splits head_dim 4:6:6 over (t,h,w) with *interleaved* rotation (not half-split); the scripts precompute it from raster positions with t=0 and verify the whole patch pipeline bit-identical against the model's own processor. Cache is 2048 on purpose: at 4096 the 36-layer/kv8 fp32 KV cache is ~1.2 GB, which more than doubles the on-device session footprint; 2048 keeps the whole phone session ~1.5 GiB. Env gotchas baked into the ship script: python 3.14 breaks torchao's import (use ≤3.13), and torchvision must be the torch-matched pair (`torch==2.12.1 torchvision==0.27.1`) or pip upgrades torch past litert-torch's pin. `magevl_work/precheck_magevl_vision.py` runs the full static-rewrite export precheck on a random-init tower — no checkpoint download — in minutes.

`paddleocr-vl-1.6` is the OCR/document-parsing specialist (task prompts `OCR:` / `Table Recognition:` /
`Formula Recognition:` / …). Two conversion gotchas are baked into its scripts: transformers ≥5.12 loads
remote-code rotary modules with a ZEROED non-persistent `inv_freq` (fix: `rope_init()` after load — and
validate against the native `paddleocr_vl` port, not the remote code), and its 0.36B decoder must ship
fp16 (`RECIPE=WF16`): int4 and dynamic-int8 both measurably corrupt transcription.

VLM quality is gated on vision end-to-end corr (≈1.0 fp32) + eager image grounding, not the 8-question
text gate (image input is device-only on this toolchain). `internvl3-2b` has a card but is reproduced by
adapting `ship_internvl_1b.sh` (model id + dims) — no dedicated script.

## Text-to-speech (Qwen3-TTS, host-loop tflite)

`qwen3tts_work/` converts [Qwen/Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) (Apache-2.0 speech LM, 10 languages, x-vector voice cloning) into three LiteRT graphs plus host tables, and includes the runnable host-side pipeline. Published artifacts: [litert-community/Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/litert-community/Qwen3-TTS-12Hz-0.6B-Base). This one is **not** a `.litertlm` — the speech-LM decode loop (16-codebook embedding sum, 15-step inner AR sub-loop per frame, PCM output) is outside the LiteRT-LM Engine's text loop, so the graphs run under a Compiled Model host loop instead.

```bash
cd qwen3tts_work
# reference dumps (env with qwen-tts==0.1.1, transformers==4.57.3):
python dump_talker_ref.py && python dump_mtp_ref.py && python dump_codec_ref.py
# conversion + verification (env with litert-torch 0.9.1, transformers 5.12.x):
python extract_talker_ckpt.py && python export_talker.py && python verify_talker.py
RECIPE=BOCTAV4 python export_talker.py          # int4 variant
python export_mtp.py && python export_codec.py && python extract_host_tables.py
python assemble_release.py                      # -> out/release/ (published layout)
# synthesize (auto-downloads the published models if out/release is absent):
python synthesize.py --text "Hello from LiteRT." --output hello.wav --model_dir out/release
```

Gates: talker tflite corr 1.0 / top-1 100% (and the synthesized Qwen3 checkpoint is bit-exact vs the talker — the TTS mrope reduces to standard RoPE); MTP 15/15 greedy tokens; codec corr 1.0; end-to-end with `--talker fp32 --greedy` = token-for-token vs the PyTorch reference, waveform corr 1.0, ASR round-trip exact. Known trap: channelwise int8 (tooling default) degenerates — use `RECIPE=BOCTAV4` (blockwise-32).

**Fast MTP (single-graph fold + dynamic int8).** The default `export_mtp.py` graph is a decode step run 17× per audio frame — weight-streaming-bound and the whole pipeline's bottleneck (M4 Max RTF ≈ 2.5). `export_mtp_folded.py` folds all 16 inner steps × 5 layers into ONE graph (in-graph argmax + embedding-table gather, KV kept internal): token-identical (52/52 e2e frames), 41 ms/frame vs 148 (M4 Max), end-to-end RTF 2.18. Data-free int8/int4 on the fold both fail (blockwise-int8 is a no-op; int4 collapses — the 15 lm-heads can't survive 4 bits); calibrated static int16 runs away. What works is **GPTQ int8 → dynamic int8 (DRQ)** so the weights stay int8 in RAM:

```bash
# calibration frames from the fp32 pipeline (extend the sentence list as needed):
TEXT="The quick brown fox jumps over the lazy dog." DUMP_MTP=out/qwen3tts-mtp/calib/calib_1.npz \
  python hostloop_e2e.py
python export_mtp_folded.py                                   # fp32 fold (desktop-exact, 559 MB)
BITS=8 GROUP=32 TAG=int8g32 python gptq_mtp_folded.py         # torch GPTQ (ai-edge-quantizer's is a stub)
python quantize_mtp_folded_v2.py fp16                         # or a plain fp16 fold (desktop only)
```

`gptq_mtp_folded.py` bakes the GPTQ int8 grid; a `dynamic_wi8_afp32` re-quant of that export (`recipe.dynamic_wi8_afp32`) yields a 218 MB graph whose FC weights are int8 in RAM (no explicit-dequantize — that trap re-materializes fp32 at load and defeats the size win). Gate = frame count + no runaway + ASR round-trip (not token match; the int trajectory diverges yet stays intelligible). On a Pixel 8a this cuts the MTP from ≈333 to ≈68 ms/frame (~5×) and the end-to-end RTF from ≈6.7 to ≈3.3; the codec decoder then dominates. `verify_codec_chunking.py` documents a related limit: the fixed-T codec graph's left-context chunking is only seam-clean up to that T, so long utterances want a larger-T codec export.

**Fast codec (mixed-precision split).** With the MTP folded, the codec decoder is the dominant stage (FLOP-bound: upsampling convs). ai-edge-quantizer can't int8 those convs (CONV_2D axis-remap error, TRANSPOSE_CONV unsupported — only the small transformer quantizes), and a plain fp16 cast is a load-time no-op. What works is running the graph with **XNNPACK FORCE_FP16** (`CpuOptions(xnnpack_flags=4)`), ARM-native fp16, no requant. Global FORCE_FP16 is ~2.2× but breaks quality — the 8-layer pre-transformer's large-magnitude activations don't survive fp16 (ASR unintelligible). `export_codec_split.py` splits the decoder at that boundary into Part A (RVQ + pre_conv + pre_transformer, run fp32) and Part B (upsample + SEANet convnet — nearly all the FLOPs, run FORCE_FP16). Part B in fp16 keeps the full 2.2× while waveform corr recovers to 0.997 and ASR is identical to fp32:

```bash
python export_codec_split.py                                 # -> codec_partA/partB (T=64) + bench
CODEC_SPLIT=1 python hostloop_e2e.py                         # end-to-end with the split codec
```

The host loop runs Part A (fp32) → hidden [1,512,T] → Part B (fp16) → PCM. On a Pixel 8a the codec drops from ≈114 to ≈40 ms/frame (2.5×); combined with the folded int8 MTP the end-to-end **RTF falls from ≈6.7 to ≈2.06** (~3.2×), ASR-lossless. Gate the codec on the "Hello…" e2e_ref codes (`hostloop_e2e.py` writes the waveform) — the standalone `codec_equiv_ref` clip isn't speech and won't ASR. `quantize_codec.py` records the int8/fp16 attempts that don't work.


## MiniCPM family (new-style: full-jinja LlmMetadata + post-hoc quantization)

`minicpm_work/` converts the MiniCPM family with the newer packaging used by litert-community/MiniCPM5-1B: the LlmMetadata carries the model's **full `chat_template.jinja` verbatim** plus a `thought` channel (`<think>`/`</think>`), so hybrid-reasoning works natively in LiteRT-LM ≥0.14 (`enable_thinking` via conversation extra context; thinking text arrives on a separate channel). Export unquantized (`--quantization_recipe=""`), then quantize with ai-edge-quantizer, then repackage with `litert-lm-builder`:

```bash
cd minicpm_work
./convert_minicpm.sh minicpm5-1b      # openbmb/MiniCPM5-1B    -> wi8 (int8)
./convert_minicpm.sh minicpm4-0.5b    # openbmb/MiniCPM4-0.5B  -> OCTAV int4-b32 + int8 embed/head
./convert_minicpm.sh minicpm4.1-8b    # openbmb/MiniCPM4.1-8B  -> int4-b32 + int8 embed/head
```

Needs python ≥3.11, `pip install litert-torch litert-lm "transformers==5.6.2"` (litert-torch 0.9.1 breaks with transformers 5.14+). Run/eval with the `litert-lm` CLI or `eval_gsm8k_api.py` (the old prefix/suffix-template harnesses don't apply jinja-only metadata).

What the scripts encode (the non-obvious parts):

- **MiniCPM5-1B** is stock `LlamaForCausalLM`, but its tokenizer contains ~15 **fused `X<|im_end|>\n` merge tokens** — the metadata must list them as string stop tokens or generation can run past end-of-turn (`LlmMetaProto.pbtext` has them all).
- **MiniCPM4/4.1** are custom `MiniCPMForCausalLM` (muP + longrope; remote code needs transformers 4.46 and won't load in 5.x). `prep_minicpm4_as_llama.py` folds them to a stock llama checkpoint (untie; `emb ×scale_emb`; `o/down ×scale_depth/√L`; `lm_head ÷(hidden/dim_model_base)`), and `export_static_longrope.py` strips transformers' `@dynamic_rope_update` (a data-dependent branch torch.export rejects; exact here since long==short factors and factor==1).
- **MiniCPM4 tokenizer**: the HF `tokenizer.json` bundle loses all spaces on decode; the raw `tokenizer.model` bundle lacks the added-token `<|im_end|>`(73440) stop. `fix_sp_added_tokens.py` appends the added tokens to the SP model as USER_DEFINED pieces and the fixed `.spiece` is what gets bundled.
- **Recipe choice (GSM8K n=100, greedy, thinking off)**: MiniCPM5-1B — wi8 **63** vs official artifact 61; data-free int4-b32 lands 48 (both min-max and OCTAV; the official artifact's int4 quantizer is not reproducible from released ai-edge-quantizer, so int8 is the recommended repro target). MiniCPM4-0.5B — bf16 57; OCTAV int4 **50**; min-max int4 collapses to 38 (sub-1B int4 sensitivity), int8 47. MiniCPM4.1-8B — int4-b32 **44/50 (88%)** (8B is int4-robust; desktop-class at 4.9GB).

## LFM2.5 family (hybrid ShortConv + attention)

`lfm_work/` converts LiquidAI's LFM2/2.5 hybrid models (gated short-conv blocks + GQA attention). litert-torch 0.9.1 ships the `lfm2` model support and litert-lm ≥0.14 runs it, but the 0.9.1 exporter has a correctness bug on real prompts: the runtime right-pads prefill chunks whenever the prompt doesn't exactly fill a prefill signature, and the exported ShortConv block saves its conv state from the **last (padded) columns** of the chunk — attention can mask padding, a causal conv cannot — so the first generated token of nearly every reply is corrupted (`"to is a vast..."`). Easy to miss: the model recovers after ~1 token (conv window is 2 wide) and GSM8K still parses answers, it just silently loses ~20pt. **litert-torch 0.9.2 fixes this upstream** (valid-token masking + one-hot matmul state select) and `convert_lfm25.py` auto-detects the version: on 0.9.1 it applies the local `lfm_short_conv_patch.py` (mask-derived valid length + gather from the last *valid* columns); on ≥0.9.2 it skips the patch. Either way engine output is token-identical to an exact per-token decode loop at every prompt length (verified with a padded-prefill harness against the raw loop).

```bash
cd lfm_work
python convert_lfm25.py LiquidAI/LFM2.5-1.2B-Instruct out_lfm25_12b        # -> int8 (export-time recipe, convs included)
python convert_lfm25.py LiquidAI/LFM2.5-1.2B-Instruct out_lfm25_12b_fp --fp  # unquantized, for post-hoc int4:
python ../minicpm_work/quantize_litertlm.py apply out_lfm25_12b_fp/model.litertlm lfm25_int4.litertlm --recipe wi4b32_wi8 --algo octav
```

Same env as MiniCPM (python 3.10+, `litert-torch litert-lm "transformers==5.6.2"`). The non-obvious parts:

- **Quantize convs at export time only.** The export-time dynamic-int8 recipe (the default) safely includes the conv layers. Post-hoc `ALL_SUPPORTED` int8 via ai-edge-quantizer kills them (no output) — post-hoc recipes must stick to linears + embedding (`wi8fc`, `wi4b32_wi8`).
- **Multi-length prefill signatures (1..1024)** let the runtime pick tight chunks (less padding waste); they do NOT fix the conv-state bug by themselves — the remainder chunk still pads unless the length happens to exactly match a signature.
- **Metadata**: bos `<|startoftext|>`, stops `[7 (<|im_end|>), 2]`, the HF `chat_template.jinja` verbatim (tool-calling works), and a `thought` channel declaration (`<think>`/`</think>`).
- **Recipe results (GSM8K n=100, greedy; Instruct/JP at max-tokens 1024, Thinking at 2048)**: Instruct — bf16 **79** · export-time int8 **81** (parity) · post-hoc int8-linears-only 79 · OCTAV int4-b32 72. Thinking — bf16 81 · int8-linears-only **77** · int4 72. JP — bf16 63 · int8-linears-only **65** (parity) · int4 55. Without the conv-state fix the Instruct int8 file lands at 59 — that's the bug, not the quantization.
- **Conv-int8 sensitivity varies by finetune**: export-time conv-int8 is free on Instruct (+2), neutral on Thinking (−1), and costs the JP tune **9 points** (56 vs 65) — A/B the export-time recipe against post-hoc linears-only (`wi8fc`) for every new finetune before picking.
- **GPU works with litert-torch ≥ 0.9.2 exports** (it did not with 0.9.1: the local fix's `index_select` lowered to GATHER_ND and its mask sum brought int64 ops — both rejected by GPU delegates; the upstream 0.9.2 rework emits neither). One catch: the 0.9.2 exporter wraps attention softmax in an `odml.softmax` StableHLO composite that the GPU delegate in released litert-lm 0.14 doesn't accept (engine creation fails with "graph is not fully delegated"; CPU unaffected). `convert_lfm25.py` therefore strips the composite marker by default — the math is unchanged, plain softmax stays — which makes the export fully GPU-delegable on today's runtime (Mac WebGPU, 1.2B int8: ~4750 tok/s prefill / ~195 tok/s decode, 8/8 on the quality gate). Pass `--keep-softmax-composite` to keep the marker for newer runtimes that fuse it.
- **The published 1.2B files predate that 0.9.2 rework, so each repo carries a separate GPU file.** They are 0.9.1-lineage: on a GPU delegate they reach 536 of 579 ops and the engine then aborts (`Hint fully delegated to single delegate is set, but the graph is not fully delegated`), the rejected ops being the local patch's `GATHER_ND` plus the INT64 `ADD`/`CAST`/`SUM` of its mask sum. Measured on a Pixel 8a with `litert_lm_main` built from the litert-lm v0.16.0 tag, and identical to the macOS signature. Rather than replace files that are already device-verified on CPU, each repo now also ships `<Name>_int4_gpu.litertlm`, built with litert-torch 0.9.3 + litert-converter 0.3.1:

```bash
cd lfm_work
python convert_lfm25.py LiquidAI/LFM2.5-1.2B-Instruct out_fp --fp
python ../minicpm_work/quantize_litertlm.py apply out_fp/model.litertlm int4.litertlm \
    --recipe wi4b32_wi8 --algo octav
python add_executor_metadata.py int4.litertlm LFM2.5-1.2B-Instruct_int4_gpu.litertlm
```

  The 2.6B pipeline's `fix_zero_block_scales.py` step is a no-op on this family (measured: 0 zero scales across 0 tensors on all three 1.2B checkpoints) — harmless to run, and the 2.6B does need it. The result delegates **fully** on Android OpenCL (501/501 and 519/519 nodes, zero rejected ops) and runs on the macOS GPU backend; **iOS Metal still cannot create an engine** for this family (upstream LiteRT-LM#3129, two Metal codegen bugs, unchanged through v0.16.0). On a Pixel 8a the GPU's win is prefill and time-to-first-token (263-token prompt: ~190 tok/s vs ~38-54 on CPU; TTFT ~1.4 s vs ~5 s), while decode is bandwidth-bound and roughly equal (~21 tok/s).
- **`litert_lm_main --benchmark_prefill_tokens` / `--benchmark_decode_tokens` are silently ignored** (measured on the v0.16.0 tag build, both backends): runs that pass them still report `Processed 19 tokens` and ~50 decode tokens. A 19-token prefill reads about 3x slower than a 256-token one on the same file, so a row labelled "prefill 256" that came from those flags is mislabelled. Drive a real prompt with `--input_prompt_file` and cap generation with `--max_output_tokens`.
- **litert-lm ≥ 0.15 needs an `ExecutorMetadata` section** for state-carrying (hybrid) models: 0.15 binds the per-layer conv/attention state buffers through a new `ExecutorMetadataProto` section, and files exported with litert-torch ≤ 0.9.2 don't have it — they run fine on 0.14 but fail at inference on 0.15 with `missing some output TensorBuffers` (attention-only models are unaffected). Fix an existing file in place with `python add_executor_metadata.py in.litertlm out.litertlm` (weights unchanged; the result runs on both 0.14 and 0.15 — this is how the published LFM2.5 repos were updated on 2026-08-04). Checking on the composite from the previous bullet: the 0.15 GPU delegate still rejects `odml.softmax`, so the strip default stays.

### LFM2.5-2.6B (the thinking flagship)

`lfm_work/convert_lfm25_26b.py` converts [LiquidAI/LFM2.5-2.6B](https://huggingface.co/LiquidAI/LFM2.5-2.6B) (22 ShortConv + 8 attention layers, vocab 128000, reasons in a `<think>` block by default). Published: [litert-community/LFM2.5-2.6B](https://huggingface.co/litert-community/LFM2.5-2.6B). Same env and mechanics as the 1.2B family above (litert-torch ≥ 0.9.2), plus three 2.6B-specific facts:

```bash
cd lfm_work
python convert_lfm25_26b.py LiquidAI/LFM2.5-2.6B out_26b            # int8 (then add_executor_metadata.py)
python convert_lfm25_26b.py LiquidAI/LFM2.5-2.6B out_26b_fp --fp    # for the int4 pipeline in the script docstring
```

- **Regenerate the metadata for the 2.6B tokenizer.** Stop ids are per-tokenizer, not per-family: `<|im_end|>` is id 7 in the 1.2B and **124900** here (`<|endoftext|>` 124895). `lfm25_26b_LlmMetaProto.pbtext` also replaces the vendor chat template: the vendor one both uses minijinja-fatal Python methods and strips past `<think>` blocks keyed on the last user index, which violates the runtime's incremental render contract (turn-3 failure). The bundled template renders history verbatim and **pre-fills `<think>` in the generation prompt** — load-bearing for quantized builds: with a bare assistant prompt, int4 loses think-discipline first (2,500-token unscaffolded deliberations instead of answers from turn 2); with the pre-fill all variants answer tersely and the thought channel routes cleanly.
- **The zero-scale int4 wall is not ternary-only.** This dense-trained checkpoint carries ~746k all-zero 32-blocks across 26 tensors; OCTAV int4-b32 emits zero scales and the file dies at engine invoke on every runtime version. Pipeline order: quantize → `fix_zero_block_scales.py` → `add_executor_metadata.py`.
- **The int8 file is desktop/Android-class** (~2.7 GB single weight section exceeds what iOS memory-maps for one section); the int4-b32 file (1.67 GB) is the iPhone variant.

**Results (GSM8K n=100, greedy, max-tokens 2048, identical harness both sides)**: bf16 reference **92** · export-time int8 **88** · post-hoc int8-linears-only 89 (conv-int8 harmless on this model — matches Instruct, unlike the JP tune) · OCTAV int4-b32 **83**. Structure gates: 8Q 8/8 CPU (both files), 42-length first-token sweep clean (fresh engine per length), 3-turn conversation exact, Mac WebGPU 8Q 7/8 (int8). Mac M4 Max CPU: int8 434 tok/s prefill@1024 / 38 decode; int4 43.7 decode.

## granite-4.0-h (Mamba2 + attention hybrid) — first Mamba2 hybrid on the released runtime

`granite_work/convert_granite4h.py` converts IBM's granite-4.0-h dense-hybrid models (Mamba2 selective-scan blocks interleaved with grouped-query attention) to `.litertlm`. Published: [litert-community/granite-4.0-h-1b](https://huggingface.co/litert-community/granite-4.0-h-1b). **Requires litert-lm ≥ 0.15 to run** (the hybrid conv/SSM states bind through the `ExecutorMetadata` section).

```bash
cd granite_work
git clone https://github.com/google-ai-edge/litert-torch litert-torch-granite
git -C litert-torch-granite checkout 115a136
git -C litert-torch-granite apply "$(pwd)/granite_hybrid_litert_torch.patch"
PYTHONPATH=litert-torch-granite python convert_granite4h.py ibm-granite/granite-4.0-h-1b out_granite_1b
# GPU ship shape (2026-08-13): declare fp32 activations in the bundle TOML (repack, no re-export)
python ../scripts/set_activation_type.py out_granite_1b/granite-4.0-h-1b_int8.litertlm granite-4.0-h-1b_int8.litertlm --type fp32
```

The patch (against the pinned litert-torch base above) is what makes a state-carrying hybrid convert *and generate correctly* end to end. The non-obvious parts, in the order they were needed:

- **A Mamba2 export-cache layer** (conv `[B, conv_dim, K]` + SSM recurrent `[B, heads, head_dim, state]`) registered for Granite's layer types, so `torch.export` traces the model's own state contract instead of failing on the attention-only default cache.
- **State-continuation tracing.** Prefill graphs trace the chunk-continuation branch (previous conv/SSM state consumed) and the decode graph traces the single-step branch (conv window rolled by one, `has_previous_state` truthy at trace time). Without this the decode graph has no state continuity: it converts, loads and runs — and generates garbage from the second token on. Trap: `torch.export`'s pytree flatten/unflatten REBUILDS cache-layer objects mid-trace, so any mode flag must ride in the pytree context or it silently never reaches the graph.
- **The mamba prefill-pad guard.** The runtime's chunk planner runs prefill chunks *partially filled* (the remainder chunk is padded), and granite's own `apply_mask_to_padding_states` is a no-op at batch 1 — so pads poison the conv/SSM state at chunk-plan-dependent prompt lengths (the Mamba2 analog of the LFM2 ShortConv prefill-pad bug). The guard derives a valid mask from `tokens != 0` (the engine zero-fills pads), makes pad positions identity SSM steps (dt → ~0), and gathers the stored conv window at the last *valid* column with an in-graph one-hot matmul.
- **Quantize convs never.** Post-hoc dynamic int8 on linears + embedding only (`wi8fc`); export-time conv-int8 measurably costs quality on this family (8Q sanity 5/8 vs 6/8 on the 350m — same rule as the LFM2.5 convs).
- Verification bar for the ship: float-graph teacher-forced parity vs the HF reference = correlation 1.000000 with identical top-1 at every checked decode position; int8 sanity gate 8/8 (= the PyTorch reference); first-token robustness sweep against the runtime's real chunk plans at every prompt length 12–60, all clean.

### 2026-08-13 update — the folded scan: 3.5× on CPU, and the GPU wall falls

The published 1b was re-converted from the same weights with the patch's selective scan re-expressed: every contraction that the HF reference spells as broadcast-multiply-reduce (materializing intermediates that scale as `chunks × chunk_len² × heads × d_state`, some at rank 5–6) is rewritten as a **batched matmul with the chunk and head axes folded into the batch axis** — every tensor in the scan is rank ≤ 4, there is no `BROADCAST_TO`, no rank-2 `PAD`, and no int64 index math. The decode path drops its per-token `expand`-materializations for implicit broadcasting. The patch self-checks the folded form against the reference formulation numerically on every export.

What this buys, same weights, same recipe (Mac M4 Max, `litert-lm benchmark -p 256 -d 256 --runs 3 --cache no`, litert-lm 0.16.0):

| granite-4.0-h-1b int8 | prefill | decode |
|---|---|---|
| previous file, CPU | 69 tok/s | 11.4 tok/s |
| re-converted, CPU | **247 tok/s** | **39.2 tok/s** |
| re-converted, GPU | **1684 tok/s** | **134.7 tok/s** |

The CPU speedup is pure graph shape — the matmul form never builds the giant intermediates. The GPU part additionally needed two things:

- **Keep the batch dimension in the pad guard's reductions** (`valid.sum(-1, keepdim=True)`, not `sum(-1)[0]`). A rank-0 tensor entering broadcast arithmetic is *silently miscomputed* by the GPU delegate — the graph is accepted, `is_fully_accelerated` stays true, and every layer's conv state quietly zeroes. If a hybrid delegates 100% and still generates garbage, suspect rank-0 scalars before anything else.
- **Declare fp32 activations** (`scripts/set_activation_type.py`, above). The GPU executor's default fp16 activations overflow on the scan's intermediates; the failure mode is the engine's "Invalid decode and sample result" / all-zero tokens, not an error message.

Ship verification on the re-converted file: float parity vs HF teacher-forced across 8 decode positions — per-position max|logit diff| ≤ 1.1e-4, correlation 1.000000, top-1 and top-5 identical; int8 8-question gate **8/8 on CPU and GPU, on litert-lm 0.15.0 and 0.16.0**; on iPhone 17 Pro (Metal) 8/8 with decode 29.8 tok/s vs 15.9 CPU; on Pixel 8a (OpenCL) every subgraph fully delegated (zero rejections) with correct output. Low-end Android GPU decode does not beat CPU (decode is bandwidth-bound and the fp32-activation path reads 4× the bytes) — the on-device GPU win is Apple hardware, plus prefill/TTFT everywhere.

### granite-4.0-h-350m (fp16 + int8) — and the start_token lesson

Published: [litert-community/granite-4.0-h-350m](https://huggingface.co/litert-community/granite-4.0-h-350m) (fp16 723 MB + int8 436 MB). The 350m initially looked like it had "quantization jitter" (replies ending after a few tokens at some prompt lengths; multi-question prompts answered only at the last question). The actual root cause was **our packaging, not the model or the runtime**: the bundle's `LlmMetadata start_token` makes the engine prepend `<|end_of_text|>` to every prompt, Granite's official chat template has no leading BOS, and at 350M scale that one token flips the greedy trajectory (prepending the same token to the HF reference reproduces the degraded output verbatim — the graph was faithful all along). The 1b is robust to it; the 350m is not.

```bash
cd granite_work
PYTHONPATH=litert-torch-granite python convert_granite4h.py ibm-granite/granite-4.0-h-350m out_granite_350m
python drop_start_token.py out_granite_350m/granite-4.0-h-350m_int8.litertlm granite-4.0-h-350m_int8.litertlm
python ../lfm_work/add_executor_metadata.py out_granite_350m/model.litertlm out_granite_350m/model_fp_meta.litertlm
python make_fp16_variant.py out_granite_350m/model_fp_meta.litertlm granite-4.0-h-350m_fp16.litertlm --drop-start-token
```

Gates on the published files (litert-lm 0.15 CPU, Mac + iPhone 17 Pro): fp16 = 8/8 sanity, greedy output token-identical to the HF fp32 reference on our probes, prompt-length sweep 12–200 clean; int8 = 8/8 sanity, sweep clean except a known 33–37-token band where replies can end early (int8 noise interacting with one prefill chunk shape — fp16 is clean there; documented on the card). On phones ship int8: the CPU runtime unpacks fp16 weights to fp32 in RAM (~3.7 GB peak on iPhone vs ~2.1 GB for int8).

## Qwen3.5 (GatedDeltaNet + attention hybrid) — first Qwen3.5 in LiteRT form

`qwen35_work/convert_qwen35_hybrid.py` converts Qwen3.5 hybrids (gated-delta-rule linear attention interleaved with gated full attention; the 0.8B is 18 + 6 layers) to `.litertlm`, text decoder only (the multimodal checkpoint's vision tower and MTP heads are dropped, matching upstream's own `Qwen3_5ForCausalLM` load contract). Published: [litert-community/Qwen3.5-0.8B](https://huggingface.co/litert-community/Qwen3.5-0.8B). **Requires litert-lm ≥ 0.15 to run** (the conv/recurrent states bind through the `ExecutorMetadata` section).

```bash
cd qwen35_work
git clone https://github.com/google-ai-edge/litert-torch litert-torch-qwen35
git -C litert-torch-qwen35 checkout 115a136
git -C litert-torch-qwen35 apply "$(pwd)/qwen35_hybrid_litert_torch.patch"
PYTHONPATH=litert-torch-qwen35 python convert_qwen35_hybrid.py Qwen/Qwen3.5-0.8B out_qwen35_08b
# GPU ship shape (2026-08-13): declare fp32 activations in the bundle TOML (repack, no re-export)
python ../scripts/set_activation_type.py out_qwen35_08b/Qwen3.5-0.8B_int8.litertlm Qwen3.5-0.8B_int8.litertlm --type fp32
```

The patch (against the pinned litert-torch base, transformers ≥ 5.14) shares the granite hybrid machinery — export-cache layers for `layer_types == "linear_attention"`, state-continuation tracing (fused single-step decode branch + chunked-continuation prefill branch), and the prefill-pad guard (pads become identity delta-rule steps; the conv window is gathered at the last valid column). Qwen3.5-specific parts, in the order they were needed:

- **Constant-eye chunk kernel.** The reference chunked delta rule builds `torch.eye` inside the traced function; it lowers to `STABLEHLO_IOTA`, which no released TFLite kernel set registers — the exported file will not even load into an Interpreter. The kernel is vendored with the identity matrix lifted as a graph constant (`torch.tensor` from python data). The triu masks and aranges constant-fold on their own; only `eye` survives as a runtime op.
- **Unwrap `@force_accelerate_hooks`.** The mixer's `forward` is wrapped without `functools.wraps`, so `inspect.getsource` returns the accelerate wrapper; the real function is fished out of the wrapper's closure before the anchored source replacements.
- **Delta-rule pad identity is on `a`, not dt.** Zeroed pad input alone still decays the recurrent state (softplus(0 + dt_bias) ≠ 0); the guard forces the pre-softplus gate input to −30 on pads so per-token decay is exp(0) = 1, and re-zeroes q/k/v after the conv so pads inject nothing.
- **Chat template replaced.** The stock template strips the `<think>` block from history assistant turns, which violates LiteRT-LM's incremental conversation rendering (each turn's render must string-extend the previous one) and hard-fails turn 2. The bundled simplified ChatML template keeps the empty think block in both the generation prompt and history renders — multi-turn then works exactly (`chat_template_simple.jinja`).
- **Quantize convs never**: post-hoc dynamic int8 on linears + embedding (`wi8fc`), convs and the delta rule stay float — same rule as LFM2.5 / granite.

Verification bar for the ship: float-graph teacher-forced parity vs the HF reference = correlation 1.000000 with identical top-1 at every checked decode position (max|logit diff| ≤ 3.7e-5); an 8-question sanity gate answered **word-for-word identically to HF fp32 by both the float and the published int8 build**; first-token robustness sweep against the runtime's real chunk plans at every prompt length 12–60, all clean; multi-turn state continuity verified.

### 2026-08-13 update — the GPU wall falls (and a delegate bug found on the way)

The published 0.8B was re-converted from the same weights with the vendored chunk kernel re-expressed the same way as granite's scan: contractions as **batched matmuls with chunk and head axes folded into the batch axis**, every tensor rank ≤ 4, no `BROADCAST_TO`, no int64 index math. Two GPU-specific lessons came out of getting it not just delegated but *correct*:

- **Never `PAD` a rank-3 tensor on a non-final axis.** The GPU delegate executes it wrong — a bit-exact one-row shift along the outermost axis (row 0 zeros), silently, with `is_fully_accelerated` still true. In this kernel that shifted every gate to the neighbouring head. Reported upstream with a 1-op repro: [LiteRT#9272](https://github.com/google-ai-edge/LiteRT/issues/9272). The kernel now writes every tail-pad as `torch.cat` with a zeros constant (unsqueezing to rank 4 before padding is also exact, if you prefer `PAD`).
- **Declare fp32 activations** (`scripts/set_activation_type.py`, above). At fp16 the remaining failure is not the converter: one layer-0 head's real weights push fp16 intermediates over range — a property of the checkpoint.

Re-ship verification: parity re-run on the ship export (48 positions, top-1/top-5 100%, Pearson 1.0000); 8-question gate **8/8 on CPU and GPU on both litert-lm 0.15.0 and 0.16.0**; prompt-length robustness with a fresh engine per length 40/40 CPU + 20/20 GPU; on iPhone 17 Pro (Metal) 8/8 with decode 41.4 tok/s vs 14.6 CPU. CPU speed is unchanged (the kernel was already matmul-form on CPU): Mac M4 Max CPU 666/46.7 tok/s, GPU 1972/161.8 (`-p 256 -d 256 --runs 3 --cache no`, 0.16.0). One honest limit: a Pixel 8a cannot compile the full prefill-ladder ship file on its GPU (fp32-expanded weights + ten compiled programs exceed ~3.8 GB available); a reduced dev shape of the same graph runs there fully delegated and correct, and CPU is unaffected.

## LFM2.5-Encoder (bidirectional encoders → plain .tflite)

`lfm_work/convert_lfm25_encoder.py` converts LiquidAI's LFM2.5-Encoder-350M/230M — multilingual (15-language) bidirectional masked-LM encoders on the same LFM2 hybrid backbone — to plain LiteRT `.tflite` for embeddings / retrieval / fill-mask, fully on CPU. These are NOT `export_hf` runs: the models have no KV cache, so the HF eager model (remote code = stock `Lfm2Model` + bidirectional patches) is traced directly with `litert_torch` multi-signature convert. Signatures: `encode_{64,128,256,512}` → `last_hidden_state` `[1,S,1024]` (padded positions zeroed), plus `mlm_128` → masked-LM logits. Published: [litert-community/LFM2.5-Encoder-350M](https://huggingface.co/litert-community/LFM2.5-Encoder-350M), [litert-community/LFM2.5-Encoder-230M](https://huggingface.co/litert-community/LFM2.5-Encoder-230M).

```bash
cd lfm_work
python convert_lfm25_encoder.py LiquidAI/LFM2.5-Encoder-230M out_lfm25_encoder_230m   # -> fp32 + wi8fc int8 + fp16
python verify_lfm25_encoder.py  LiquidAI/LFM2.5-Encoder-230M out_lfm25_encoder_230m   # 15-language parity report
```

Env: litert-torch ≥ 0.9.2, transformers ≥ 5.12 (the encoders' remote code needs 5.x; tested 5.14.1), torch 2.12. The non-obvious parts:

- **transformers' `apply_mask_to_padding_states` is a no-op at batch==1** (`shape[0] > 1` guard) — and batch-1 right-padded input is exactly what static tflite signatures produce. Without a fix, the symmetric (non-causal) ShortConv reads pad-token embeddings past the sequence end and the last valid tokens diverge. The script rebinds the symbol **inside the remote-code module** (patching `transformers.models.lfm2.modeling_lfm2` does nothing — the remote module imported the function by value) to an unconditional mask-multiply; zeroed pad inputs make the padded forward token-exact vs the unpadded one.
- **Gate on pad-CONTENT invariance, not on padded-vs-unpadded diff.** With complete masking, changing the pad-region token ids leaves valid positions bitwise unchanged (the script asserts exactly 0.0). Padded-vs-unpadded absolute diff is shape-dependent reduction-order float noise (~1e-4 on the 350M) — a threshold on it false-fails.
- **Signature inputs must be passed as `sample_kwargs`** — positional `sample_args` names the tflite inputs `args_0/args_1` instead of `input_ids`/`attention_mask`.
- **`wi8fc` int8 = linears + embedding only, convs float** (same rule as the LFM2.5 decoders — `ALL_SUPPORTED` int8 post-hoc kills the conv layers). The embedding lowers to `EMBEDDING_LOOKUP` (not GATHER), so the int8 recipe reaches the 65536×1024 vocab table, and the tied `lm_head` FC dedupes against it in the flatbuffer.
- **Parity (16 sentences over all 15 languages, mean-pooled-embedding cosine vs PyTorch)**: fp32/fp16 **1.000000** (fill-mask top-5 sets identical); wi8fc ≥ **0.9948** (230M) / **0.9966** (350M) min, fill-mask top-1 4/4 / 3/4. Cross-signature outputs (encode_64/128/256 on the same sentence) agree bitwise.
- **int8 is the mobile artifact; fp16 is desktop-only.** On an iPhone 17 Pro the int8 files reproduce desktop outputs bit-exactly (cosine 1.000000, max diff 0.0; encode_512 93 ms on the 230M, 145 ms on the 350M, 6 threads). The fp16 files jetsam at interpreter init even with the increased-memory-limit entitlement — XNNPACK unpacks fp16→fp32 per signature subgraph (×5, no weights cache). Also load one model per process on iOS; two sequential interpreter loads stack XNNPACK's repacked weights and hit the memory limit.
- **transformers 5.x remote-code load check**: assert `rotary_emb.inv_freq.min() > 0` after `from_pretrained` — 5.x meta-device materialization can silently zero init-computed buffers, and since reference and export share the model object, parity would still "pass" on a broken load.

### Task finetunes (Prompt-Router / Policy-Linter / PII-Detector / Spellchecker)

`lfm_work/convert_lfm25_encoder_finetunes.py router|linter|pii|gec` converts LiquidAI's four 350M encoder finetunes; `verify_lfm25_finetunes.py <kind>` runs task-level parity (route probabilities / rule flags / entity tags / edit tags) for fp32 + fp16 + int8. Published: [Prompt-Router](https://huggingface.co/litert-community/LFM2.5-Encoder-350M-Prompt-Router) · [Policy-Linter](https://huggingface.co/litert-community/LFM2.5-Encoder-350M-Policy-Linter) · [PII-Detector](https://huggingface.co/litert-community/LFM2.5-Encoder-350M-PII-Detector) · [Spellchecker](https://huggingface.co/litert-community/LFM2.5-Encoder-350M-Spellchecker). Each has a custom remote-code head — none is "the encoder script unchanged":

- **Router / Linter take pooling matrices as extra float32 graph inputs** (`text_pool` [1,1,S], `category_pool`/`rule_pool` [1,8,S]) — the offset-derived mean-pool weights the vendors' `route()` helpers compute; building them host-side keeps the graph static. All-zero rows for unused lane/rule slots are exact no-ops (their logit collapses to the score bias; ignore beyond your real N).
- **The batch-1 padding rebind must go through `Lfm2ShortConv.slow_forward.__globals__`** — each finetune repo inlines its own copy of the bidirectional patches, so patching a module by name misses them; the installed function's globals always hit the right namespace.
- **Spellchecker's tied vocab heads quantize for free**: `tie_replace` computes `$REPLACE`/`$APPEND` logits as `proj(h) @ embed[:64400].T`, which lowers to FULLY_CONNECTED with a constant weight — the int8 recipe reaches it and the embedding slice dedupes (429 MB total, not +264 MB). Its iterative decode and optional reranker stay host-side.
- **Task parity**: router/linter/gec are task-exact through int8; PII is exact at fp32/fp16, and int8 flipped exactly one tag in our test — an entity-end token with a 0.53-logit fp32 margin (inherently borderline; documented on the card).

### LFM2.5-Embedding-350M (the retrieval bi-encoder on the same backbone)

`lfm_work/convert_lfm25_embedding.py` converts [LiquidAI/LFM2.5-Embedding-350M](https://huggingface.co/LiquidAI/LFM2.5-Embedding-350M) — a multilingual (11-language) dense bi-encoder built on the same `Lfm2BidirectionalModel` backbone as the encoders above — to plain LiteRT `.tflite` for retrieval / RAG / semantic search. Signatures `embed_{64,128,256,512}` → `output_0` `[1,1024]`, **already CLS-pooled and L2-normalized**. Published: [litert-community/LFM2.5-Embedding-350M](https://huggingface.co/litert-community/LFM2.5-Embedding-350M) (int8 371 MB + fp16 712 MB).

```bash
cd lfm_work
python convert_lfm25_embedding.py LiquidAI/LFM2.5-Embedding-350M out_lfm25_embed_350m  # fp32 + wi8fc int8 + fp16
python verify_lfm25_embedding.py  out_lfm25_embed_350m --gates A,B,C,D --prefix-ab     # card oracle + STS17 + retrieval + mechanics
python bench_lfm25_embedding.py   out_lfm25_embed_350m
```

Env: litert-torch ≥ 0.9.2, transformers 5.14.1 (see the version-drift note below), torch 2.12. The non-obvious parts:

- **The repo's remote code does not load on transformers 5.14.1.** It targets 4.56.2 and installs `Lfm2ShortConv.forward = _shortconv_forward`, which forwards `**kwargs` straight into a `_noncausal_shortconv_forward(hidden_states, past_key_values, cache_position, attention_mask)` that does not declare `seq_idx` — which 5.14.1's `Lfm2DecoderLayer` now passes. Every forward dies with `TypeError: ... unexpected keyword argument 'seq_idx'`. The fix is to rebind `forward` to a wrapper that filters kwargs down to `inspect.signature(slow_forward).parameters`, **not** to edit the cached remote code — the filter survives a vendor update.
- **Pick the FA2 pad semantics, and pass the masks yourself.** The vendor supports two behaviours: the sdpa default lets the symmetric ShortConv read pad-token embeddings ("matching the behavior the checkpoints were trained with"), FA2 zeroes pads first ("closest match to the unpadded forward"), and the card reports the two equivalent within 0.002 nDCG across 11 languages. For a static multi-signature export only the FA2 behaviour is viable: under the default, the *same text* routed to `embed_128` vs `embed_512` returns a different vector (max abs 0.005). Force the zeroing and length invariance becomes exact.
- **The vendor's own comment about which mask arrives is already stale.** It says eager/sdpa hand the ShortConv a 4D additive mask "on which this is a no-op". On 5.14.1, `Lfm2Model.forward` builds `{"full_attention": create_causal_mask(...), "conv": create_recurrent_attention_mask(...)}` and the conv layers receive the **2D** pad mask. It is still a no-op, but because of the other half of the guard — `apply_mask_to_padding_states` needs `attention_mask.shape[0] > 1`, and static tflite signatures are exactly batch 1. Read the installed source, not the comment about it.
- **`Lfm2Model.forward` accepts `attention_mask` as a dict** (`if not isinstance(causal_mask_mapping := attention_mask, dict)`) and uses it verbatim, so the script builds both masks and hands them over — the same escape hatch ModernBERT offers. That keeps `create_recurrent_attention_mask` out of the trace, whose "return `None` when the mask is all-ones" branch would otherwise specialize at export time into a graph that ignores `attention_mask` (the identical hazard documented for the Nemotron embedder). Gate it: shortening the mask must move the output (4.8e-02 here).
- **Every invariance check lands on exactly 0.0**, which is what makes the multi-signature file safe to use: hand-built masks vs the vendor's own mask path 0.0, pad-content invariance 0.0, padded-vs-unpadded 0.0, and cross-signature agreement bitwise 0.0 at 64/128/256/512 on fp32, int8 and fp16 alike. (The encoders above only reach ~1e-4 on padded-vs-unpadded, because there the pooled output aggregates every position; here the CLS vector does not.)
- **Read `config_sentence_transformers.json` for the prompt strings; do not retype them from the prose.** The contract is `query: ` / `document: ` **with the trailing space** — prose and config agree, and the space changes tokenization (`Ġ`-prefixed content token, e.g. 3747 vs 3493). Note the sibling ColBERT repo stores a completely different contract (`query_prefix: "[Q] "`, `similarity_fn_name: "MaxSim"`, `do_query_expansion`, a punctuation skiplist) — reading the wrong repo's config is an easy way to ship the wrong prefix.
- **A 50-query retrieval set cannot settle a prefix question.** On NanoSciFact: no prefix 0.8668, no-space 0.8726, documented 0.8540 nDCG@10 — a ~0.019 spread on n=50, inside one standard error, with recall@5 differing by a single query. That is not evidence against the documented prefix; it is a set without the resolution to say anything. Use the graded STS-style gate for resolution, and follow the upstream contract for the prefix.
- **`wi8fc` int8 = FC dynamic-range + EMBEDDING_LOOKUP channelwise, convs float** (same rule as every LFM2 hybrid). Confirm it in the flatbuffer rather than trusting the recipe: 92/92 FULLY_CONNECTED int8, EMBEDDING_LOOKUP int8, all 10 DEPTHWISE_CONV_2D still float32. 354.5M params → fp32 1420.6 MB, int8 371.2 MB, fp16 711.9 MB. Unlike granite-embedding-r2 (65% vocab table) the table is only 19% of the parameters here, so the win comes from the linears.
- **Quality is quantization-flat.** STS17 over 11 language pairs: fp32 mean 0.6720, int8 0.6721. NanoSciFact: fp32 nDCG@10 0.8540 → int8 0.8494. fp16 is bitwise identical to the PyTorch reference on every task metric. (`en-tr` 0.083 and `nl-en` 0.484 are identical on the reference too — Turkish and Dutch are not among the model's 11 languages, so they are a control that the gate discriminates rather than a defect.)
- **int8 is the mobile artifact; fp16 runs on a large phone but is not a phone build.** On an iPhone 17 Pro (6 threads) int8 reproduces desktop outputs **bit-exactly** (cosine 1.000000, max diff 0.0) across en/ja/de/ar/hi, both prompt forms, short and long inputs — embed_128 40–44 ms, embed_512 137–142 ms, peak footprint 1245 MiB for the 4-signature file. fp16 also passes (max diff 2e-07) but peaks at **5799 MiB**, and its first call on each signature costs 355–538 ms of one-time XNNPACK fp16→fp32 unpacking against 73–98 ms warm. Note the per-signature memory cost is model-specific: 4 signatures cost far less here than the ~846 MiB/signature measured on the 2048-d Nemotron embedder.
- **fp16 is slower than fp32 on CPU** (Mac, embed_512: 200.8 ms vs 168.8 ms; int8 123.0 ms). XNNPACK unpacks fp16 weights to fp32 at run time, so the smaller file buys disk, not latency.

### LFM2.5-ColBERT-350M (late interaction, one vector per token)

`lfm_work/convert_lfm25_colbert.py` converts [LiquidAI/LFM2.5-ColBERT-350M](https://huggingface.co/LiquidAI/LFM2.5-ColBERT-350M) — the late-interaction sibling of the embedder above, on the same `Lfm2BidirectionalModel` backbone — to plain LiteRT `.tflite`. Signatures `encode_{32,128,256,512}` → `output_0` `[1,S,128]`: a pylate `Dense(1024→128, bias=False)` head plus a per-token L2 normalize, folded into the graph. Published: [litert-community/LFM2.5-ColBERT-350M](https://huggingface.co/litert-community/LFM2.5-ColBERT-350M) (int8 370 MB + fp16 710 MB).

```bash
cd lfm_work
python convert_lfm25_colbert.py LiquidAI/LFM2.5-ColBERT-350M out_lfm25_colbert_350m  # fp32 + wi8fc int8 + fp16
python verify_lfm25_colbert.py  out_lfm25_colbert_350m --gates A,B,C --unpadded-ref  # ranking + MaxSim retrieval + mechanics
python bench_lfm25_colbert.py   out_lfm25_colbert_350m
```

Env: litert-torch ≥ 0.9.2, transformers 5.14.1, torch 2.12. The non-obvious parts:

- **Zeroing the padding states is exactly WRONG here — the opposite of the embedder above, on the same backbone.** The two repos ship *different* `modeling_lfm2_bidirectional.py` under the same filename: the ColBERT copy gates `apply_mask_to_padding_states` on `_attn_implementation == "flash_attention_2"` and leaves hidden states untouched on eager/sdpa, with the vendor's own note that transformers >=5.x routes the raw 2D pad mask there and "would zero padding/query-expansion states and shift per-token embeddings (hurts ColBERT MaxSim)". `config.json` also sets `disable_flash_attention: true`. **Diff the remote code between sibling repos; never assume a shared filename means shared behaviour.** The converter asserts the gate is present before exporting — without it, query-expansion vectors are silently zeroed with no exception and no NaN, and only retrieval quality suffers.
- **The host contract is most of the work, and it is read out of pylate 1.6.0 rather than guessed.** Queries: tokenize to `query_length - 1` = 31 with `padding="max_length"`, padding with the **EOS token (id 7)** — pylate uses the mask token if the tokenizer has one and EOS otherwise, and this tokenizer has no mask token; insert `[Q] ` (id 64400) at **position 1**, after BOS; leave `attention_mask` **0** on the expansion positions (`attend_to_expansion_tokens: false`); and **score all 32 vectors**, expansion included. Documents: insert `[D] ` (id 64401) at position 1 and keep `skiplist_mask AND attention_mask` (32 punctuation ids). Scoring is MaxSim over unit vectors. `verify_lfm25_colbert.py` is a working reference implementation of all of it.
- **The pad token id is load-bearing.** Because padding states are not zeroed, padding with 0 instead of 7 moves the output by 0.34 max abs. Note `config.json` says `pad_token_id: 0`; pylate does not use it.
- **Cross-signature agreement is bitwise at every kept position**, on fp32, int8 and fp16 alike — so pick a signature for speed and memory, not for quality. The reason is mechanical: the ShortConv kernel is 3, so a token reads one neighbour on each side, and once there is a single pad to the right, more pads change nothing. The drift from padding is binary, not length-proportional. The one position that legitimately differs between signature lengths is the **last** one (its right neighbour is absent at the end of a short signature and a pad inside a long one); it is always masked out host-side, so gate the kept positions, not the whole tensor.
- **Pad the reference the same way you pad the thing you are judging.** Running the torch reference unpadded while the tflite side pads to a signature reports the padding delta as conversion error (it showed up as 0.9957 per-token cos and −0.0152 nDCG@10 before the reference was fixed). Attributed properly on NanoSciFact (600 docs, 50 queries): torch unpadded 0.8777, torch padded 0.8625, fp32 **0.8625**, fp16 **0.8625**, int8 0.8626 — conversion and quantization are both free and the whole delta is the static shape. And −0.0152 on 50 queries is inside one standard error, so report it as unresolved rather than as a measured quality loss.
- **`wi8fc` int8 = FC dynamic-range + EMBEDDING_LOOKUP channelwise, convs float.** 353.5M params, vocab 64402 (the embedder's is 65536 — a bench script that borrowed the sibling's vocab bound produced out-of-range ids). fp32 1416.5 MB, int8 370.1 MB, fp16 709.8 MB. No `GATHER_ND`, no `LOGICAL_AND`.
- **int8 is the mobile artifact.** iPhone 17 Pro, 6 threads: int8 reproduces desktop outputs **bit-exactly** (cosine 1.000000, max diff 0.0) across a real 32-token query with 23 expansion positions, documents in three scripts, and the same document at two signature lengths — encode_32 30 ms, encode_128 40–64 ms, encode_512 180 ms, peak 1254 MiB. fp16 also passes but peaks at 5786 MiB, and is slower than fp32 on CPU because XNNPACK unpacks fp16 weights at run time.

## Nemotron-3-Embed-1B (a bidirectional embedder wearing a decoder's clothes)

`nemotron_work/convert_nemotron3_embed.py` converts [nvidia/Nemotron-3-Embed-1B-BF16](https://huggingface.co/nvidia/Nemotron-3-Embed-1B-BF16) — NVIDIA's multilingual retrieval/RAG embedding model (34 languages, 2048-d, pruned+distilled from Ministral-3-3B) — to plain LiteRT `.tflite`. Like the LFM2.5 encoders this is NOT an `export_hf` run: there is no KV cache in the embedding path, so the HF eager model is traced directly with `litert_torch` multi-signature convert. Signatures `embed_{64,128,256,512}` → `output_0` `[1,2048]`, **already mean-pooled and L2-normalized** — one call in, one finished embedding out.

```bash
cd nemotron_work
python convert_nemotron3_embed.py nvidia/Nemotron-3-Embed-1B-BF16 out_nemotron3_embed_1b  # fp32 + wi8fc int8 + fp16
python verify_nemotron3_embed.py out_nemotron3_embed_1b                                   # card oracle + STS17 + retrieval + mechanics
python bench_nemotron3_embed.py  out_nemotron3_embed_1b
```

Env: litert-torch ≥ 0.9.2, transformers 5.14.1, torch 2.12. The non-obvious parts:

- **`architectures: ["Ministral3Model"]` is the stock causal-LM class — `config.is_causal: false` is what makes it bidirectional.** transformers honours it: `create_causal_mask()` falls back to `create_bidirectional_mask()` and the flag reaches `sdpa_attention_forward` as an explicit `is_causal=False`. `Ministral3Attention.__init__` hardcodes `self.is_causal = True` and is **not** the switch — flipping it changes the output by exactly 0.0. Verify the real thing instead: change the LAST token and check that position 0 moves (it moves by 5.96 here; under causal attention it could not move at all).
- **The bidirectional mask is skipped entirely when nothing is padded.** `create_bidirectional_mask` → `sdpa_mask(allow_is_bidirectional_skip=True)` returns `None` for an all-ones mask. Harmless eagerly, *not* harmless under `torch.export`: the branch specializes at trace time, so tracing with an unpadded sample bakes a graph that ignores `attention_mask` and lets pad tokens into both the attention and the mean. Always trace with right-padded samples, and gate on **pad-content invariance** — scribble random ids over the pad region and require the embedding to be bitwise unchanged (0.0 here, on every variant).
- **Pooling and normalization are not in the HF module.** `config.pooling: "avg"` and `modules.json`'s `2_Normalize` are sentence-transformers metadata that `Ministral3Model` ignores; mean-pool (with `include_prompt: true`) and L2-normalize have to be added to the traced module or you ship raw hidden states.
- **The `query: ` / `passage: ` prefix is mandatory — including for symmetric similarity.** This is an E5-style model; raw text is out of distribution. On STS17 (100 pairs, int8), no prefix vs `query: ` on both sides: en-en 0.636→**0.865**, es-en 0.064→**0.803**, en-ar −0.029→**0.724**. Cross-lingual collapses to noise without it, which is exactly the capability the model is bought for — and the upstream `config_sentence_transformers.json` sets `default_prompt_name: null`, so a plain `SentenceTransformer.encode()` lands in the bad column. Use `encode_query` / `encode_document` or prefix by hand.
- **Build the attention bias yourself; don't let `sdpa_mask` do it.** Feeding a 2D mask routes through transformers' index-based mask construction, which lowers to `GATHER_ND` -> `LOGICAL_AND` -> `SELECT_V2`. Read the constants out of an actual export and it is an identity gather at batch 1 (index grid `[[0,0],[0,1],...]`), an all-True q-side mask, and `where(mask, 0.0, -inf)` — i.e. `bias[0,0,q,k] = 0 if valid else -inf`, with no q dependence. `_preprocess_mask_arguments` early-exits on any 4D mask, so emitting a `[1,1,1,S]` bias directly removes `GATHER_ND` and `LOGICAL_AND` and never materializes the `[1,1,S,S]` mask. Output is **bitwise identical** either way, and it is speed-neutral (an apparent -8% reversed when the A/B was run in the other order — thermal drift, not the change).
- **The GPU wall here is not `GATHER_ND`, it is 5-D tensors.** Compiling for Metal, 256 ops are rejected with `RESHAPE: Tensor dimensions must be less than 5` — the `[1,8,1,S,128] -> [1,8,3,S,128]` expansion inside GQA `repeat_kv` — plus 4 unsupported `BROADCAST_TO`. Only ~62 of ~1100 ops offload and compilation then fails, with or without the `GATHER_ND`. A GPU lane needs `repeat_kv` re-expressed in 4-D (or SDPA's `enable_gqa`); the mask cleanup above is hygiene, not the fix. Checked on litert-converter 0.3.0 and 0.3.1 — identical op histogram, identical rejection profile, bitwise identical weights, so this is a property of the graph rather than of one converter build.
- **`wi8fc` int8 = FC dynamic-range + EMBEDDING_LOOKUP channelwise.** The embedding lowers to `EMBEDDING_LOOKUP` (not GATHER), so the recipe reaches the 131072×2048 vocab table — 24% of all parameters. 1140.9M params → fp32 4567 MB (a fine intermediate; the writer uses buffer offsets, so the 2 GiB flatbuffer figure is not a wall), int8 1167 MB, fp16 2286 MB.
- **Quality is quantization-flat.** Against the base card's own documented similarity matrix all variants reproduce it within the card's own bf16-vs-fp32 gap (max abs 0.0041 for fp32/fp16, 0.0075 for int8) and rank the right document first 4/4. STS17 over 11 language pairs: fp32 mean 0.7292, int8 mean 0.7296 — task-lossless. (`en-tr` scores ~0.04 for every variant: Turkish is not one of the 34 languages NVIDIA evaluated. It is a useful control that the gate discriminates.)
- **Signature count is a memory decision, not a quality one.** Every build returns bitwise identical vectors, and the same text routed through `embed_64/128/256/512` is bitwise identical too. But the interpreter delegates *every* signature subgraph at creation, so peak RAM scales with the signatures **present in the file**, not the ones you call — measured on an iPhone 17 Pro at ~846 MiB per signature: 1 sig → 1028 MiB peak, 2 sigs → 1901 MiB, 4 sigs → 3593 MiB, while the file itself only grows 1148 → 1167 MB. Build `--seqs 512,128` for phones; the 4-signature file wants desktop or the `increased-memory-limit` entitlement.
- **int8 is the mobile artifact.** On an iPhone 17 Pro the int8 builds reproduce desktop outputs **bit-exactly** (cosine 1.000000, max diff 0.0) across five scripts and both signatures; embed_128 103 ms, embed_512 ~400 ms at 6 threads — faster than an M4 Max Mac on the identical file (150 / 561 ms), which does not speed up with more threads. The fp16 file is **desktop-only**: on device it is killed by the OS during interpreter creation, before the first inference.
- **Retrieval sanity** (SciFact-derived, 50 queries / 600 docs, subsampled so not BEIR-comparable): fp32 nDCG@10 0.8683, int8 0.8628, fp16 0.8683. At 50 queries this gate cannot resolve deltas below a couple of percent — Gate B (1100 graded pairs) is the higher-resolution evidence.

There is no `.litertlm` here on purpose: `litert-lm-builder` will happily pack one (`embedding_metadata` + `tflite_model --model_type text_encoder`, and it peeks back clean), but the released runtime has no embedding executor to run it — `Engine()` on such a file aborts. Plain `.tflite` is the working artifact today.

## granite-embedding-*-multilingual-r2 (ModernBERT encoders → plain .tflite)

`granite_embed_work/convert_granite_embedding_r2.py` converts IBM's multilingual ModernBERT bi-encoders to plain LiteRT `.tflite` for retrieval / RAG / semantic search on CPU. Same encoder lane as the LFM2.5 and Nemotron embedders — no KV cache, so the HF eager model is traced directly with `litert_torch` multi-signature convert. Signatures `embed_{64,128,256,512}` → `output_0` `[1,768]`, **already CLS-pooled and L2-normalized**. Published: [litert-community/granite-embedding-311m-multilingual-r2](https://huggingface.co/litert-community/granite-embedding-311m-multilingual-r2) (int8 336 MB + fp16 629 MB).

```bash
cd granite_embed_work
python convert_granite_embedding_r2.py ibm-granite/granite-embedding-311m-multilingual-r2 out_granite_311m
python verify_granite_embedding_r2.py out_granite_311m --prefix-ab
```

Env: litert-torch ≥ 0.9.2, transformers 5.14.1, torch 2.12. The non-obvious parts:

- **Read the pooling config; do not assume.** This model is **CLS pooling** (`1_Pooling/config.json` → `pooling_mode_cls_token: true`, so `hidden[:, 0]`) with **no prompt prefixes**. The Nemotron embedder in this same repo is mean pooling with a *mandatory* `query: `/`passage: ` prefix. Two embedders, opposite contracts — carrying one recipe to the other silently produces wrong vectors.
- **ModernBERT alternates local and global attention.** 22 layers = 8 `full_attention` + 14 `sliding_attention` (every 3rd is global) with a 64-wide half-window, and **each layer type has its own rope frequency set** (global θ 150000 / local 160000) — so assert *both* `inv_freq` buffers are non-zero after load, not just one.
- **Build both masks yourself.** `ModernBertModel.forward` accepts `attention_mask` as a **dict** (`{"full_attention": ..., "sliding_attention": ...}`) and uses it verbatim, which bypasses transformers' `sdpa_mask` entirely: the resulting graph has **zero int64 tensors and no GATHER_ND**. Semantics verified bitwise against transformers at S=128 and S=512 — `full` = pad mask broadcast over queries, `sliding` = pad mask AND `|q − k| ≤ 64`.
- ⚠️ **A sliding window plus right padding creates fully-masked query rows → NaN.** Once a pad position is further than the window from every real token (`q ≥ n_valid + 64`), the sliding row is entirely False and softmax runs over all `-inf`. This is the common case, not an edge case: a 6-token query padded into `embed_512` leaves **442** such rows. It hides well — **eager PyTorch SDPA absorbs it** (so a torch-vs-torch comparison is silent), the lowered SOFTMAX emits **NaN** in fp32/fp16, and **int8 masks it completely** behind a healthy-looking `cos 0.999234`. *An int8 variant scoring better than fp32 is the tell.* Fix: OR the diagonal into the sliding mask (`(valid & band) | (q == k)`) — output-neutral, because for a valid query `k=q` is already allowed and pad rows never reach a valid row (pad **keys** stay masked everywhere). Verified bitwise identical at S=64/128/256/512 with the NaN gone. Gate it with an `isfinite` assert per variant per signature, plus a short-text-in-longest-signature case.
- **`ai_edge_quantizer.export_model` refuses to overwrite**, so a second run leaves *stale* int8/fp16 files beside a freshly exported fp32 — which then get "verified". Delete prior outputs before quantizing.
- **Quality**: reproduces the base card's documented 3×3 cross-lingual matrix to **every published digit** (fp32/fp16 max abs 0.0000, int8 0.0039). STS17 over 11 language pairs: fp32 0.7363 / int8 0.7327 / fp16 0.7364. Retrieval (SciFact-derived, subsampled so not BEIR-comparable) nDCG@10 0.8400 / 0.8360 / 0.8400 with identical recall@5 and hit@1. Cross-signature outputs agree **bitwise**.
- **Prefix contract is per-model — measure it.** An unofficial `query: ` prefix *raises* en-en STS17 from 0.783 to 0.826, which looks like grounds to override the vendor's empty prompts. On retrieval it reverses: nDCG@10 0.836 → 0.832/0.830 and recall@5 0.900 → 0.860. It is a symmetric-similarity artifact; bare text stays the contract. `--prefix-ab` runs this control.
- **int8 is the device artifact.** On an iPhone 17 Pro both files reproduce desktop outputs **bit-exactly** (cosine 1.000000, max diff 0.0), but int8 peaks at **518 MiB** for all four signatures (embed_128 58–120 ms, embed_512 331–354 ms) against fp16's 3700 MiB and up to 6× the latency — XNNPACK expands fp16 weights to fp32 while packing each signature subgraph. At 311M the per-signature packing cost is ~118 MiB, so unlike the 1.14B Nemotron build there is no need to trim signatures or ask for a raised memory limit.

## BitCPM-CANN (ternary / 1.58-bit LLMs — int4 blockwise is a lossless container)

`bitcpm_work/` converts openbmb's BitCPM-CANN family (BitNet-b1.58-style ternary QAT LLMs, Apache-2.0). LiteRT has no native 1-bit/ternary type ([feature request #7713](https://github.com/google-ai-edge/LiteRT/issues/7713)), but it doesn't need one to run these losslessly: BitCPM's QAT quantizer produces weights in {-α, 0, +α} per group of 128 input channels (α from absmean), released "pseudo-quantized" — the ternary values materialized in a plain bf16 checkpoint. Min-max symmetric int4 `BLOCKWISE_32` (blocks subdivide the 128-groups along the same axis) maps every weight onto {-7, 0, +7} with **zero rounding decisions**; the only residual is fp16 rounding of the per-block scale (≤4e-4 relative). `verify_ternary.py` checks both properties on the actual checkpoint before you convert. You pay 4 bits/weight instead of the native ~1.6 (≈2.5× their packed GGUF), but 4× under f16, on stock CPU/GPU int4 kernels.

```bash
cd bitcpm_work
./convert_bitcpm.sh                    # openbmb/BitCPM-CANN-1B -> bitcpm-cann-1b_wi4b32_wi8.litertlm (1.05 GB)
```

Same env as MiniCPM (`litert-torch litert-lm "transformers==5.6.2"`). The non-obvious parts:

- **The 1B is a stock `LlamaForCausalLM`** (no remote code, no muP) in the MiniCPM4 tokenizer family — the pipeline reuses `../minicpm_work`'s longrope-static export wrapper, the ChatML metadata pbtext verbatim (stops `[2, 73440]`), and the SP added-tokens fix (raw `tokenizer.model` lacks `<|im_end|>`=73440 → stop never fires; the 1B repo also ships no `added_tokens.json`, so `prep_bitcpm_as_llama.py` synthesizes one from `tokenizer_config.json`). The 0.5B is MiniCPM4-0.5B-arch (muP) — run it through the `prep_minicpm4_as_llama.py` fold first if you want it.
- **Use `--algo minmax`, not OCTAV.** For ternary blocks, min-max is exact by construction (scale = α/7); OCTAV optimizes MSE for continuous distributions and can pick a non-exact grid. This is the reverse of the usual recommendation for non-ternary models.
- **Embeddings + lm_head are NOT ternary** (full bf16, untied) — they take the usual int8 channelwise treatment.
- **Results (GSM8K n=100, greedy, identical prompt/extraction)**: torch bf16 **63** · .litertlm int4-b32 **60** (parity within n=100 noise; openbmb's own harness reports 61.56). The proof the container is lossless: the same data-free `wi4b32_wi8` recipe on the non-ternary MiniCPM5-1B loses ~13 points (61 → 48); on ternary it's free. 8-question gate 7/8. Mac bench (prefill 256/decode 256): CPU 285/57 tok/s, GPU (WebGPU) 1137/54.

## Ternary-Bonsai-1.7B (PrismML ternary Qwen3 — second ternary family, one new wall)

`bonsai_work/` converts PrismML's Ternary-Bonsai LLMs (ternary g128 with FP16 group scales — vendor-documented — on a stock `Qwen3ForCausalLM`; Apache-2.0; "unpacked" fp16 release = ternary values materialized, same pattern as BitCPM-CANN). The same int4-lossless-container argument applies and `verify_ternary.py` confirms it on the real weights (Bonsai preserves ~0.003% salient outliers in fp, so a handful of blocks quantize normally instead of exactly).

```bash
cd bonsai_work
./convert_bonsai.sh                    # prism-ml/Ternary-Bonsai-1.7B-unpacked -> ternary-bonsai-1.7b_wi4b32_wi8.litertlm (1.11 GB)
```

The non-obvious parts:

- **Stock Qwen3**: yarn rope is static at init (no `@dynamic_rope_update` wrapper needed, unlike longrope); BPE tokenizer bundles as `hf_tokenizer` (no SP fix); metadata pbtext is generated from the repo's `chat_template.jinja` (always-empty `<think>` prefill = non-thinking 2507 style → no thought channel), stops `[151645, 151643]`, **no start_token** (Qwen has no BOS).
- **⭐ Verbatim Qwen-style templates break MULTI-TURN on the runtime (minijinja)**: the runtime's minijinja implements `startswith`/`endswith` (and `[::-1]` slicing) but NOT `.strip()/.lstrip()/.split()`, and the engine's incremental assistant-turn render always executes the template's reasoning-rerender branch — so the FIRST message works and the SECOND dies with `unknown method: string has no method named strip`. Single-turn evals (GSM8K, quality gates) structurally cannot catch this. `convert_bonsai.sh` therefore rewrites the assistant-history branch (lines 34–51 of the upstream template) to a plain verbatim emit — faithful for this non-thinking model; multi-turn re-verified on Mac and Android.
- **⭐ Zero-scale wall (new)**: ternary weights are sparse enough that a 32-weight block can be ALL zeros → ai-edge-quantizer min-max blockwise emits **scale = 0** → XNNPACK refuses to prepare (`unsupported scale value (0.000000) ... for INT4 tensor`). This model hits it (14 blocks across 6 tensors); BitCPM-CANN-1B happened not to. `fix_zero_block_scales.py` patches it post-quant: LiteRT stores blockwise scales in **separate FLOAT16 tensors referenced by the BlockwiseQuantization details table** (not `QuantizationParameters.scale`), so the script walks the raw (lazy) flatbuffers API and replaces zero scales with the tensor's min nonzero scale in place — the affected blocks are all-zero, so dequantization is unchanged. Seconds, vs ~12 min for an object-API round-trip.
- **Results (GSM8K n=100, greedy, identical prompt/extraction)**: torch bf16 **77** · .litertlm int4-b32 **74** — the same −3-within-noise parity as BitCPM-CANN-1B, now across two vendors and two architectures. 8-question gate **8/8**. Mac bench (matched back-to-back): GPU 1576/62 tok/s, CPU 222/33.
- **On-device (Pixel 8a, Google AI Edge Gallery 1.0.15, official in-app benchmark, p256/d256×3)**: Ternary-Bonsai-1.7B CPU **86.5 prefill / 16.3 decode tok/s** (TTFT 3.0s), multi-turn chat correct with the patched template. BitCPM-CANN-1B GPU **133.1 / 7.7 tok/s** (GPU init ~50s) — int4-b32 blockwise runs on the mobile GPU. Ternary-Bonsai crashes at GPU engine creation on the same device (SIGSEGV in the runtime's weight loading; BitCPM with the identical recipe works, so it's file-specific — tied embeddings / odd vocab 151669 / qwen3 q-k-norms / yarn are the differing suspects). Note the Gallery accelerator list is fixed at import time: enable GPU in the import dialog, or the benchmark silently runs CPU.
- **The image sibling**: `prism-ml/bonsai-image-ternary-4B-unpacked` (FLUX.2-Klein pipeline) — the diffusion transformer's 100 block linears verify ternary-g128/int4-exact the same way (`verify_ternary.py` accepts a group-size arg and matches `blocks` names), while the text encoder is a stock fp16 Qwen3-4B (only the DiT is ternarized). The full pipeline conversion is the next section.

## Bonsai Image 4B (ternary DiT text-to-image — the full diffusion pipeline)

`bonsai_image_work/` converts the whole FLUX.2-klein pipeline to three fixed-shape tflite graphs plus a torch-free host loop. Published: [litert-community/Bonsai-Image-ternary-4B](https://huggingface.co/litert-community/Bonsai-Image-ternary-4B); sample PR: [litert-samples#244](https://github.com/google-ai-edge/litert-samples/pull/244).

```bash
cd bonsai_image_work                       # checkpoint auto-downloads; BONSAI_SNAPSHOT overrides
python export_dit.py                       # fp64-rope fix + export -> dit_fp32.tflite (14.4 GiB)
python quantize_dit.py                     # int4-b32 ternary linears + int8 rest -> 2.11 GiB
python fix_zero_block_scales.py dit_int4b32.tflite dit_int4b32.tflite   # zero-scale patch (REQUIRED, 15 scales here)
python export_textenc.py && python quant_textenc.py                     # prompt embedder -> 1.68 GiB int4
python export_vae.py                       # -> 0.19 GiB fp32
python generate.py --model-dir . --prompt "a bonsai tree"               # host loop: no torch, no diffusers
```

The non-obvious parts:

- **⭐ Flux2 rope builds its frequency table in float64** → `tfl.pow` on f64 fails to legalize at the very last conversion stage. `export_dit.py` forces f32 (cost: max rel 2.9e-6, verified with REAL position ids — an all-zero-ids check is vacuous since cos=1/sin=0 at position 0 for any dtype).
- **The ternary container claim survives the full chain**: the finished int4-b32 file uses only {-7, 0, +7} in all 100 block linears — zero rounding decisions, measured in the artifact.
- **Zero-scale wall again**: the DiT hits 15 zero scales across 4 tensors (the LLM sibling hit 14/6) — this is the generic sparse-ternary case, not a model quirk. Patch is mandatory before XNNPACK will prepare.
- **Text encoder exports as a prompt embedder**: the pipeline reads hidden layers (9, 18, 27) only → litert-torch prunes the top 9 of 36 layers + LM head automatically (4.02 B → 3.11 B). Recipe note: dynamic-range quantization floors image fidelity regardless of bit width (activation quantization dominates); weight-only (`quantize_weight_only.py`) is the quality lane, and int8 weight-only is the practical variant.
- **The host loop needs three exact formulas**, all verified against an instrumented reference run before any device work: the sigma schedule is Flux2's `compute_empirical_mu` fit (NOT the scheduler config's linspace+shift — that gives different sigmas), the DiT timestep input equals sigma, and the packed-latent→VAE transform is a per-PACKED-channel affine (the VAE BatchNorm running stats) applied BEFORE 2×2 unpatchify.
- **⚠ Runtime integration**: attach the XNNPACK delegate explicitly (with `num_threads`) when using the C API — the reference-kernel fallback is orders of magnitude slower AND numerically wrong on blockwise int4.
- **Measured**: Mac CPU 8 threads ≈ 19 s/image (512×512, 4 steps); iPhone 17 Pro (XNNPACK, 6 threads) ≈ 64 s/image, 2.9 GiB peak, bit-exact vs desktop (51.2 dB PSNR on the final PNG).

### macOS GPU demo app (DiT on the Apple GPU — 6.6 s/image)

`device/BonsaiAppMac/` is a SwiftUI app that runs the DiT on the Apple GPU through the LiteRT Metal accelerator: **0.76 s/DiT-step, 6.6 s/image total** (vs ≈19 s all-CPU), after a one-time ~40 s Metal compile per launch. Text encoder and VAE stay on CPU, where the CompiledModel path is bit-exact vs the fixture set. GPU vs same-latent CPU pipeline: 31.6 dB PSNR, visually identical.

```bash
cd bonsai_image_work
python export_dit_gpu.py                        # GPU-shaped DiT (rope constants folded, gather-free) -> dit_gpu_fp32.tflite
python quantize_dit.py dit_gpu_fp32.tflite      # -> dit_gpu_int4b32.tflite
python fix_zero_block_scales.py dit_gpu_int4b32.tflite dit_gpu_int4b32.tflite
# (or skip the three steps above: the pre-quantized dit_gpu_int4b32.tflite is published on the model card)
cd device/BonsaiAppMac
./prep_resources.sh && xcodegen generate
xcodebuild -project BonsaiMac.xcodeproj -scheme Bonsai -configuration Release build
```

The pitfalls that cost real time (details in `device/BonsaiAppMac/README.md`): the runtime dylib pair must be **same-generation** (ai-edge-litert 2.1.6 works; the 2026-07-31 LiteRT-main macos_arm64 prebuilt pair SIGSEGVs in an XNNPACK transpose microkernel inside the Metal accelerator during delegate init on this 2.27 GiB DiT — repro CLI in `device/BonsaiAppMac/smoke/`); **fp32 precision is mandatory** (default fp16 corrupts this model's activations), passed as a hand-built TOML opaque option because the prebuilt exports no options helpers; and the GPU-shaped export is required — the CPU-shaped DiT does not run on the Metal delegate.

### Shieldstral-1.0-3B (a single-token safety classifier, not a chat model)

`shieldstral_work/convert_shieldstral.sh` converts [mistralai/Shieldstral-1.0-3B](https://huggingface.co/mistralai/Shieldstral-1.0-3B). Published: [litert-community/Shieldstral-1.0-3B](https://huggingface.co/litert-community/Shieldstral-1.0-3B).

The checkpoint is `Mistral3ForConditionalGeneration` — a pixtral vision tower plus a Ministral3 text decoder — and this recipe ships the **text lane**: `scripts/extract_ministral3_text.py` drops the tower, which is output-neutral for text input (bit-identical yes/no logits on the floor set). The decoder is then the ordinary dense lane, same recipe family as `Ministral-3-3B-Instruct-2512`.

```bash
bash shieldstral_work/convert_shieldstral.sh     # -> out_int4/model.litertlm, out_int8/model.litertlm
```

Three things are specific to this model class and worth knowing before you gate it:

**It answers with one token.** The reply is literally `yes` or `no`, and the published quantity is a softmax over those two logits, thresholded at 0.5. A generic 8-question quality gate is nearly useless here — obvious cases carry ±9 to ±14 logit margins, so every build scores 8/8, including one whose measured numbers were wrong. Build the floor gate from **borderline** items, and treat label agreement plus correlation of the raw logit margin against the source model as the real verdict.

**The scoring API has two traps, both silent.** `Session.run_text_scoring` *mutates* the session, so scoring a second candidate after the same prefill returns a plausible-but-wrong number — give each candidate its own session and prefill. And `create_session(apply_prompt_template=True)` prefills the user **prefix only** (no user suffix, no start token), which scores the model mid-prompt: on this bundle that read 1.060 instead of 8.087 and inverted a clearly-unsafe floor item. Pre-render the whole prompt yourself and pass `apply_prompt_template=False`. The ordinary generate path is unaffected — a bundle can generate perfectly while every scored number is wrong.

**The fixed system prompt is baked into the user prefix** (`templates/shieldstral_simple.jinja`). The model card fixes it as part of the inference contract, and the scoring entry point carries no role information, so shipping it as a separate system slot would make the model's own reference prompt unreachable from the API you actually classify with.

Gate results at n=200 (stratified from the public OpenAI moderation evaluation set, threshold 0.5, identical prompt and extraction on both sides): source fp32 F1 85.0 · bf16 85.7 · int8 CPU 86.1 · int8 GPU 86.1 · int4-b32 CPU 85.7 · int4-b32 GPU 86.2. Every row lands inside a 1.2-point band, and every label flip sits where the reference margin is < 0.7 from zero. int4-b32's main section is 1.82 GiB and runs on an iPhone 17 Pro (~1.8 s per verdict warm, ~1.2 GB peak); int8's 3.33 GiB section is desktop-only.

#### Shieldstral text+image (first pixtral tower here)

`shieldstral_work/vision/` builds the multimodal bundle. Pixtral turns out to be an unusually shallow dynamic-resolution tower: at a fixed square size the per-image crop is the identity, the block-diagonal mask for one image is all zeros (so it collapses to plain full attention), and the meshgrid position ids become a constant. Patchify is a single Conv2d and the sequence stays raster-ordered, so no `GATHER_ND` is introduced — the usual mobile-GPU vision blocker never arises.

```bash
python shieldstral_work/vision/static_pixtral.py  src_models/shieldstral-3b   # corr 1.0 vs eager
python shieldstral_work/vision/verify_adapter.py  src_models/shieldstral-3b   # corr 1.0 vs inputs_embeds
python shieldstral_work/vision/convert_shieldstral_vision.py src_models/shieldstral-3b out_vision --size 560
CACHE=4096 python scripts/export_internvl_decoder.py src_models/shieldstral-3b-text out_decoder
DEC=out_decoder VIS=out_vision python shieldstral_work/vision/build_shieldstral_bundle.py
```

⚠ **The position ids must keep indexing the trained grid.** They are `h * max_width + w` with `max_width = config.image_size // patch_size` (110 here) — substituting the *new* grid width silently re-indexes the rope table and the output stays plausible.

⚠ **The token expansion does not fit the runtime's injection contract as-is.** Pixtral expands one `[IMG]` into a grid of image tokens with `[IMG_BREAK]` ending every row and one `[IMG_END]` at the end, and those markers keep their ordinary *text* embeddings — but the runtime injects one contiguous block at the soft-token position. The escape is to fold the marker embeddings into the adapter: they are constant rows of the decoder's embedding table, so the adapter emits the full block (a `cat` with a constant column — no gather, no dynamic shape). Verify against `inputs_embeds` for the same image.

Two runtime facts: `vision_backend` must be named explicitly (without it the engine loads, the conversation is created, and only the first image message fails), and the scoring API is text-only (`run_prefill(list[str])`), so image documents get the generated verdict but no continuous score.

**Letterbox images before passing them.** The runtime resizes to the declared H×W with no padding. On 100 labelled images, stretching cost 8.5 F1 against the source model where letterboxing cost 3.0, and agreement rose from 92% to 96% — a larger effect than the int8-vs-int4 choice.

## LFM2.5-VL (3B / 1.6B / 450M — LFM2 hybrid text + SigLIP2 vision, native runtime image support)

The first family where the vanilla `litert-torch` VLM path and the runtime's own image pipeline line up end-to-end: `LiquidAI/LFM2.5-VL-3B`, `-1.6B` and `-450M` convert with stock `export_hf --task image_text_to_text` and run **text + image on the released `litert-lm==0.16.0` pip runtime** (`litert-lm run model.litertlm --prompt "..." --attachment img.png`).

```bash
python -m venv .venv && .venv/bin/pip install litert-torch==0.9.3 "transformers==5.14.1" pillow "torch==2.12.1" "torchvision==0.27.1" litert-lm
.venv/bin/python lfm_work/convert_lfm25_vl.py LiquidAI/LFM2.5-VL-3B out_vl3b_int8            # int8 (text+vision dynamic_wi8)
.venv/bin/python lfm_work/convert_lfm25_vl.py LiquidAI/LFM2.5-VL-3B out_vl3b_fp --fp         # fp text for int4
.venv/bin/python lfm_work/quantize_lfm25_vl.py out_vl3b_fp/model.litertlm LFM2.5-VL-3B_int4.litertlm          # int4-b32 octav + wi8 embedder + zero-scale fix + executor metadata
.venv/bin/python lfm_work/quantize_lfm25_vl.py out_vl3b_int8/model.litertlm LFM2.5-VL-3B_int8.litertlm --recipe none  # executor metadata only
LITERT_LM=.venv/bin/litert-lm python lfm_work/gate_lfm25_vl_text.py LFM2.5-VL-3B_int4.litertlm cpu
LITERT_LM=.venv/bin/litert-lm python lfm_work/gate_lfm25_vl_image.py LFM2.5-VL-3B_int4.litertlm cpu
```

⚠ **Pin transformers==5.14.1.** 5.15.0 renamed `Lfm2ShortConv.L_cache` → `conv_kernel_size` and the 0.9.3 export subclass still reads the old attribute (`AttributeError` at model load). torchvision must stay 0.27.x or it drags torch past litert-torch's `<2.13` pin.

⚠ **The metadata pbtext needs `llm_model_type { lfm2 {} }` — empty is correct.** The runtime's `Lfm2DataProcessor` proto defaults already equal this family's processor config (512×512, patch 16, max 1024 patches, pooling 2, mean/std 0.5, boi `<|image_start|>` / eoi `<|image_end|>`). The chat template must render a bare `<image>` per image content part and nothing else: the runtime splits the rendered prompt on it and inserts boi + preprocessed pixels + eoi itself, so a template that emits boi/eoi too would double them.

⚠ **The externalized embedder dodges every text recipe.** VLM exports split the embedder into its own tflite; an `--fp` text export leaves it float32 (1 GB at the 3B's 128k vocab) and a post-hoc recipe applied to `prefill_decode` never reaches it — `quantize_lfm25_vl.py` runs the wi8 recipe on the embedder tflite as a separate pass (1049→264 MB). It also goes through `litert-lm pack/unpack` instead of a single-tflite rebuild, because the older quantize/zero-scale scripts would silently drop the vision sections.

⚠ **The zero-scale int4 wall is inherited by the VL text stack.** The 3B checkpoint carries 746,432 all-zero 32-blocks across 26 tensors — the same dense-checkpoint signature as LFM2.5-2.6B — so the zero-scale fix (folded into `quantize_lfm25_vl.py`) is mandatory; the 450M has none.

Vocab split inside one family: the 3B uses the 128k vocab (stops 124900/124895), the 1.6B and 450M the older 64k one (stops 7/2, same as the 1.2B family) — the converter picks the matching pbtext by model name and refuses sizes it does not know (check `generation_config.json` before adding one).

Gates on the 0.16.0 pip CLI (temperature 0, `--cache no`): 3B int8 AND int4 pass text 7/8 + image 5/5 on cpu and macOS gpu, with image answers verbatim identical to the HF bf16 reference (`verify_lfm25_vl_hf.py`); 1.6B passes text 8/8 on all four configs; 450M int4 passes 8/8 + 4/5. iPhone GPU remains blocked for the ShortConv family (LiteRT-LM#3129); use CPU there.

⚠ **The 64k-vocab models (1.6B, 450M) deterministically miss fine shape/counting fixtures on-device while the torch reference answers them** — and every conversion-side suspect has been eliminated by experiment: exported vision tower AND projector match torch at cosine 1.0000 on identical inputs; an unquantized bundle reproduces the identical misses; runtime patchify layout equals the HF processor's (code-read, both (ph,pw,c) raster); the prompt/token streams match token-for-token (boi/eoi are single tokens in every vocab); double-BOS, fp16 decode, jpeg-level pixel perturbation and prefill-chunk-boundary alignment (probed by forcing the image block exactly onto a 256-token chunk edge via a custom chat template) all change nothing; geometry probes on-device (corner localization, stripe orientation) come back CORRECT. Coarse understanding, OCR and localization are intact — the residue is engine-internal precision on contour/counting answers, the 3B is unaffected, and the next step is instrumenting the runtime to diff the spliced embedding sequence against HF `inputs_embeds`.

## Verify a reproduction

```bash
~/clipconv/bin/python scripts/verify_quality.py out/<key>/model.litertlm --json   # 8-question gate
# parity (dense): scripts/parity_gsm8k.py  ·  reasoning models: run at --max-tokens 2048
# single-token classifiers: label F1 + margin correlation vs the source model, not the 8-question gate
```
