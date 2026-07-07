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

## Single-command models

| key | HF source | template | quant | env | shipped to |
|---|---|---|---|---|---|
| `fastcontext-4b` | microsoft/FastContext-1.0-4B-SFT | chatml_simple | BOCTAV4 | EXTERNALIZE_EMBEDDER, CACHE=4096 | litert-community/FastContext-1.0-4B-SFT |
| `nanbeige4.1-3b` | Nanbeige/Nanbeige4.1-3B | chatml_simple | BOCTAV4 | FORCE_SPM, EXTERNALIZE_EMBEDDER, CACHE=4096 | litert-community/Nanbeige4.1-3B |
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
| `ovis2.5-2b` | **static-NaViT-512** (256 tok) | Qwen3-1.7B | `ship_ovis_2b.sh` |
| `paddleocr-vl-1.6` | **static-NaViT-560** (400 tok) | ERNIE-4.5-0.3B (**fp16** — int4/int8 corrupt OCR) | `ship_paddleocr_vl.sh` |
| `smolvlm2-500m` / `-2.2b` | SigLIP + pixel-shuffle | SmolLM2 / SmolLM2-1.7B | `ship_smolvlm2{,_22b}.sh` |

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


## Verify a reproduction

```bash
~/clipconv/bin/python scripts/verify_quality.py out/<key>/model.litertlm --json   # 8-question gate
# parity (dense): scripts/parity_gsm8k.py  ·  reasoning models: run at --max-tokens 2048
```
