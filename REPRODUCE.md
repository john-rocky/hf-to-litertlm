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

## 2026-08-30 — tokenizer section re-ship (17 files, weights unchanged)

An audit of every published `.litertlm` (91 files) against the upstream tokenizers found that the 14 bundles whose tokenizer was a SentencePiece conversion of a byte-level BPE vocabulary (litert-torch `tokenizer_to_sentencepiece_lib`, our `FORCE_SPM` recipe) mis-tokenize standalone accented/special characters, turn characters without a whole-character piece (emoji, most of Latin Extended-A) into the token the conversion had reused as UNK, and — for `SmolLM3-3B.litertlm` — never match `<|im_end|>` in the prompt (reported upstream as [litert-torch #1205](https://github.com/google-ai-edge/litert-torch/issues/1205); repro and a measured converter patch in [`tools/tokenizer_parity/`](tools/tokenizer_parity/)). Two OLMo-2 bundles carried a re-serialized `tokenizer.json` with the GPT-2 default pre-tokenizer instead of the model's regex, and PaddleOCR-VL's vendor SentencePiece model typed `</s>` as a control symbol the template writes as text.

**Fix = replace only the tokenizer section**, everything else byte-identical (checked per section):

```
# SP-converted bundles and OLMo-2: embed the upstream tokenizer.json (HF tokenizer path)
python tools/tokenizer_parity/swap_tokenizer_section.py in.litertlm <upstream>/tokenizer.json out.litertlm
# PaddleOCR-VL: retype the CONTROL-typed added tokens of the vendor .spiece as USER_DEFINED, then repack
```

Gate on the LiteRT-LM runtime (0.16.1, CPU), old file vs new file: default turn, 7 probe strings, the 223 standalone characters U+00A1–U+017F and every special token tokenize identically to the upstream tokenizer; prefill token count equals the upstream count; the ASCII-only test questions whose ids do not change answer byte-identically old→new; VLM bundles caption an image. Files: InternVL3-1B/2B, InternVL3_5-1B/2B/4B, LLaVA-OneVision-0.5B, Mage-VL, Ministral-3-3B-Reasoning-2512, Ovis2.5-2B, Polaris-4B-Preview, Qwen2-VL-2B, SmolLM3-3B (`SmolLM3-3B.litertlm`), SmolVLM2-2.2B/500M, PaddleOCR-VL-1.6, OLMo-2-1B-Instruct (litert-community and mlboydaisuke). On-device rows on the cards were measured on the previous files; the runtime's tokenizer code is the same on macOS and on device, and the weights and graph are byte-identical.

**Recipe change going forward:** `FORCE_SPM` is retired for byte-level BPE vocabularies — export with the HF tokenizer path (the default), or, if a SentencePiece section is required, convert with `tools/tokenizer_parity/fix_candidate_lib.py` and gate the built bundle with `engine.tokenize` parity on specials, accented characters and emoji.

## Running the converted models on-device (Android)

- **[docs/android-npu.md](docs/android-npu.md)** — Qualcomm NPU (HTP) with on-device JIT
  compilation: the 10 runtime libraries, the two Environment options, the silent-CPU-fallback
  trap, and the SRQ export recipe for LLM bundles (verified on SM8850 / Hexagon v81).
- **[docs/android-gpu.md](docs/android-gpu.md)** — GPU: the AI Edge Gallery import toggle
  (default is CPU-only), the LiteRT-LM API path, and what to expect (prefill gains,
  decode ≈ CPU, ~2× model size in RAM).

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

**Template safety:** every recipe here exports with `use_jinja_template=False` — the script swaps the vendor's HF chat template for a minimal ChatML one (`templates/*.jinja`), and the converter packs it as plain prefix/suffix turn markers, so the bundle embeds **no Jinja at all**. A plain litert-torch export instead defaults to `use_jinja_template=True` and embeds the vendor template verbatim. On older runtimes that was a crasher: vendor templates often call Python-style methods (`.get()`, `.startswith()`) the old minijinja renderer didn't implement, so the bundle imported fine and died on the first message (an earlier Edge Gallery build: `Failed to apply template: unknown method: map has no method named get`). Current runtimes render these templates. Measured 2026-08-24 with a `.startswith()`/`.split()`-heavy vendor template: litert-lm 0.16.0 CLI and litert-mac-verify on Mac, and a 2026-08-21 `litert_lm_main` build on Android CPU all applied the template (prefill length matches the templated prompt) and answered correctly; `strftime_now()` also renders on 0.16.0. Treat embedded Python-method Jinja as a warning for old runtimes, not a blocker; the current Edge Gallery build was not retested. Inspect any bundle's template with `pip install litert-lm-builder && python -m litert_lm_builder.litertlm_peek_main --litertlm_file model.litertlm`.

## Single-command models

| key | HF source | template | quant | env | shipped to |
|---|---|---|---|---|---|
| `fastcontext-4b` | microsoft/FastContext-1.0-4B-SFT | chatml_simple | BOCTAV4 | EXTERNALIZE_EMBEDDER, CACHE=4096 | litert-community/FastContext-1.0-4B-SFT |
| `nanbeige4.1-3b` | Nanbeige/Nanbeige4.1-3B | chatml_simple | BOCTAV4 | FORCE_SPM, EXTERNALIZE_EMBEDDER, CACHE=4096 | litert-community/Nanbeige4.1-3B |
| `nanbeige4.2-3b` | Nanbeige/Nanbeige4.2-3B | chatml_think | BOCTAV4 | FORCE_SPM, EXTERNALIZE_EMBEDDER, CACHE=4096; dedicated `convert_nanbeige42.py` (looped transformer: 22 shared-weight layers ×2 loops → 44 KV slots) | litert-community/Nanbeige4.2-3B |

#### Nanbeige4.2 on a GPU: one TOML line, and it is not the graph

`Nanbeige4.2-3B` shipped CPU-only because its GPU output was a flood of `<unk>` on every backend tried. The
graph was never the problem — it delegates in full, 4716 of 4716 ops, on Adreno and on Metal alike. What
breaks is the **activation dtype**: at fp16 the unrolled loop's second pass reads the graph input instead of
the first pass's output, so the model computes from nothing. Declaring fp32 activations fixes it with the
weights untouched:

```bash
python scripts/set_activation_type.py model.litertlm model_fp32act.litertlm --type fp32
```

Published as `model_fp32act.litertlm` alongside the original. Measured the same day on the same machines:
Apple M4 Max Metal 409 tok/s prefill and 39.7 decode against 57 and 12.9 on CPU — a GPU win both ways; on a
Galaxy S26 the GPU decodes at 3.9 tok/s against the CPU's 3.45, a wash, with the GPU's only clear advantage
being peak memory (2590 MB against 4152 MB).

Two traps this lane paid for, both worth copying:

- **`litert-lm benchmark` reports throughput for a backend that produced no text.** The published card carried
  556 / 52.6 tok/s for the file that emits `<unk>`; the number is real and meaningless. Confirm generation
  before quoting any figure.
- **Neither file starts on an iPhone 17 Pro** at a 512-token budget under an 8-question on-device harness: the
  GPU leg fails at state-buffer allocation and the CPU leg is killed by the OS after engine init. Both fail
  identically, so fp32 activations are not the cause — a 3B at this cache size does not fit that app's budget.
  A smaller cache or a lower token budget may well succeed; this is one configuration, not a verdict on the
  device.
| `olmo2-1b` | allenai/OLMo-2-0425-1B-Instruct | olmo2_simple | BOCTAV4 | CACHE=4096 | mlboydaisuke/OLMo-2-1B-Instruct-LiteRT |
| `olmo2-7b` | allenai/OLMo-2-1124-7B-Instruct | olmo2_simple | BOCTAV4 | CACHE=4096, EXTERNALIZE_EMBEDDER | *(desktop-only, not published — >2 GiB section)* |
| `polaris-4b` | POLARIS-Project/Polaris-4B-Preview | qwen3_think | BOCTAV4_128 | **FORCE_SPM**, EXTERNALIZE_EMBEDDER, CACHE=4096 | litert-community/Polaris-4B-Preview |
| `qwen3-1.7b` | Qwen/Qwen3-1.7B | qwen3_think | BOCTAV4 | CACHE=4096 | mlboydaisuke *(dropped→private)* |
| `qwen3-4b-thinking` | Qwen/Qwen3-4B-Thinking-2507 | qwen3_think | **BOCTAV4_128** | EXTERNALIZE_EMBEDDER, CACHE=4096 | litert-community/Qwen3-4B-Thinking-2507 |
| `r1-distill-qwen-1.5b` | deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | deepseek_r1_simple | BOCTAV4 | CACHE=4096 | mlboydaisuke/DeepSeek-R1-Distill-Qwen-1.5B-LiteRT |
| `r1-distill-qwen-7b` | deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | deepseek_r1_simple | BOCTAV4 | CACHE=4096, EXTERNALIZE_EMBEDDER | mlboydaisuke/DeepSeek-R1-Distill-Qwen-7B-LiteRT *(desktop)* |
| `smollm3-3b` | HuggingFaceTB/SmolLM3-3B | smollm3_think | BOCTAV4 | CACHE=4096, EXTERNALIZE_EMBEDDER | mlboydaisuke/SmolLM3-3B-LiteRT |
| `twil-lm3` | webAI-Official/TwIL-LM3 | smollm3_think | BOCTAV4 | CACHE=4096, EXTERNALIZE_EMBEDDER | mlboydaisuke/TwIL-LM3-LiteRT ⚠ non-commercial |
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
- **`polaris-4b` FORCE_SPM** — added **2026-08-25**; the row shipped without it, and without it the
  tokenizer section does not match the published file. The evidence is in the published bundle's own
  header, which anyone can read with a ranged HTTP request: it carries an `SP_Tokenizer` section, and
  `FORCE_SPM` is the only thing in `export_simple_template.py` that produces one from a BPE source (it
  rebinds `export_lib.export_tokenizer`). Checked against four published controls read the same way —
  `Nanbeige4.1-3B` and `Ministral-3-3B-Reasoning` (FORCE_SPM in recipe) carry `SP_Tokenizer`;
  `Qwen3-4B-Thinking` and `FastContext-4B` (no FORCE_SPM) carry `HF_Tokenizer_Zlib`. `jan-nano` carries
  `HF_Tokenizer_Zlib`, so its row is right as written.
- **`polaris-4b` and `jan-nano` carry one post-export step** that these recipes do not perform: on
  2026-08-22 both published files were repacked to declare a thought channel, so that a reasoning model's
  `<think>...</think>` chain is addressable instead of streaming inline into the answer (without
  `LlmMetadata.channels` a thinking budget is silently ignored). The step adds a `channels` block naming
  the `<think>` / `</think>` markers and repacks; it is metadata-only — every tflite section comes out
  byte-identical — so quality and speed are unaffected, but the published file's sha256 differs from what
  these recipes produce. Bundle `creation_timestamp` reads 2026-08-22 for both.
- **`llama32-3b`** was originally exported through the **official litert-torch main** (BPE patch upstream) —
  re-running with the current default env reproduces the same `BMIX4` recipe; expect equivalent, not bit-identical.
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
| `granite-docling-258m` | SigLIP-512 (SmolVLM2 rail, 64 tok) | granite Llama 576h/30L (**wi8-float** — int4/int8-int corrupt DocTags), cache 4096 | `ship_granite_docling.sh` |
| `internvl3-1b` | InternViT-448 | Qwen2.5-0.5B | `ship_internvl_1b.sh` |
| `internvl3.5-1b` / `-2b` / `-4b` | InternViT-448 | Qwen3-0.6B / 1.7B / 4B | `ship_internvl3_5_{1b,2b,4b}.sh` |
| `llava-onevision-0.5b` | SigLIP-384 (730 tok) | Qwen2-0.5B | `ship_llavaov.sh` |
| `mage-vl` | **static-448, no GATHER_ND** (196 tok) | Qwen3-4B **int4-b128**, cache 2048 | `ship_magevl.sh` |
| `north-micro-vision` | **static-512, deepstack folded, no GATHER_ND** (256 tok) | Cohere2-rehost 2B **int8** + in-bundle `fp32_fp16` hint | `ship_northmv.sh` |
| `ovis2.5-2b` | **static-NaViT-512** (256 tok) | Qwen3-1.7B | `ship_ovis_2b.sh` |
| `paddleocr-vl-1.6` | **static-NaViT-560** (400 tok) | ERNIE-4.5-0.3B (**fp16** — int4/int8 corrupt OCR) | `ship_paddleocr_vl.sh` |
| `qwen2-vl-2b` | **static-672, no GATHER_ND** (576 tok) | Qwen2-1.5B int4 | `ship_qwen2vl_2b.sh` |
| `smolvlm2-500m` / `-2.2b` | SigLIP + pixel-shuffle | SmolLM2 / SmolLM2-1.7B | `ship_smolvlm2{,_22b}.sh` |

`granite-docling-258m` is IBM's document-conversion VLM (the Docling model): page image → DocTags markup (layout + OTSL table structure + formulas), convertible to Markdown/HTML with docling-core. It rides the SmolVLM2 rail **unchanged** (idefics3's vision tower is architecturally SmolVLM2's). Three things its scripts encode. (1) **Quantization: wi8-float only** — int4 and integer-compute int8 corrupt DocTags structure on the 258M decoder (PaddleOCR-VL pattern); int8 weights with float compute keep the table gate exact. (2) **The bundle jinja protects the newline after `<|end_of_text|>`** (`{% endfor -%}`, no leading trim): the model is format-exact, and a `{%-` that eats that one `\n` token turns every image turn into a hallucinated blank page — at every decoder precision, while text-only turns stay fine. (3) **App contract: pre-resize pages to 512×512 BILINEAR** before sending. The base model is trained with tiling; the single-global-512 path this bundle runs is resampling-sensitive *in eager too*, and the runtime's own downscale from larger inputs lands on the bad side. Backend is CPU (the GPU delegate rejects the quantized 576×576 FC at kernel init on macOS WebGPU and Android OpenCL alike; an fp16 decoder runs on Android OpenCL but ~4× slower than CPU). Device-verified: Galaxy S26 CPU converts the synthetic 5×6 table page exactly (25/25 cells, byte-identical across runs) at 35.4 s/page.

`qwen2-vl-2b` is the general-purpose Qwen2-VL VLM (describe / VQA / OCR). Two gotchas baked into its scripts: (a) reordering patches into the merger's 2×2-block order with a gather emits a `GATHER_ND` op that the mobile GPU delegate can't compile (the vision executor then fails to create on-device) — so the encoder keeps raster order and the 2×2 merge is done with strided slices + concat in the adapter; (b) the fast_vlm runtime feeds 1-D positions (no M-RoPE), which preserves describe/VQA/OCR/count but degrades cross-cell *ranking* over 2-D tables.

**Qwen2-VL derivative intake** (`scripts/ship_qwen2vl_derivative.sh`, measured 2026-08-25 on
[numind/NuExtract-2.0-2B](https://huggingface.co/numind/NuExtract-2.0-2B) — 32k downloads, the
top generative Qwen2-VL-2B derivative). Two findings, one of them a wall:

- **Family fact — VLM derivatives train the vision tower too.** Per-tensor diff vs the base:
  339/391 `visual.*` and 315/338 `model.*` tensors differ. Unlike every LLM family measured,
  a derivative conversion re-exports vision from the derivative's own weights (the ship script
  and `convert_qwen2vl_vision.py`'s `MODEL` env encode this). The rails otherwise carry over
  unchanged: decoder re-host is bit-exact (`text-only logits maxdiff=0.000000`), vision
  static-672 export fp32 corr 0.99999998 / int8 end-to-end corr 0.896 — the same 0.90-class
  int8 behavior as the shipped base. Tokenizer files differ only by serialization era
  (added_tokens.json split out, a `#version` merges header; probe tokenization identical).
- **The wall was ours, not the runtime's** (reported as a runtime defect 2026-08-25, retracted on
  LiteRT-LM #3348 on 2026-08-27, corrected here 2026-09-04). The measured facts stand: bundle
  greedy output on a specials-free prompt is byte-identical to HF fp32 for 500+ tokens, and the
  moment the prompt carries `<|im_start|>`/`<|im_end|>` the bundle answers `{"names": []}` where
  HF extracts correctly. The attribution did not: every exclusion experiment was judged by
  comparing greedy outputs, and that measure is degenerate — several *different* wrong-id
  substitutions all reproduced the bundle's output, which was over-read as localizing the
  runtime's encode path. Rebuilt against the published `litert-community/Qwen2-VL-2B` (a fast_vlm
  bundle) with a discriminating measure — teacher-forced `run_text_scoring` after prefill, plus
  prefill-token counts against `engine.tokenize` — the runtime encodes every ChatML special to its
  single correct id on the templated and the raw path alike, and every marker substitution moves
  the score (harness and numbers are on the issue). The runtime's llm / fast_vlm / `tokenize`
  paths share one encode call. What was defective is the bundle's own SentencePiece tokenizer
  section, written by litert-torch's BPE→SentencePiece conversion (litert-torch #1205: no byte
  fallback, pad/eos typed as the UNK piece). `scripts/gate_specials.py` now measures exactly
  that — the engine's encode against the upstream `tokenizer.json` on every added token alone and
  mid-string, plus Latin-1, Ext-A, emoji, CJK, digits and whitespace probes (litert-lm 0.16.0):

  | bundle | tokenizer section | result |
  |---|---|---|
  | `litert-community/Qwen2-VL-2B` as published before 2026-08-30 (sha256 `a9fd5053…`) | SP, converted | **FAIL 6/33** — `<|endoftext|>` splits into 7 tokens, `é` → byte id 165, `ă` → 151643 (it *becomes* `<|endoftext|>`), leading whitespace differs; `<|im_start|>` / `<|im_end|>` correct |
  | `litert-community/Qwen2-VL-2B` as published now (sha256 `cf481776…`) | HF `tokenizer.json` | **PASS 33/33** |
  | gemma-3-270m re-export (SP-native vocab, 6,417 added tokens) | SP, vendor | **PASS 12838/12838** |

  The NuExtract bundle was deleted before the retraction, so which of the #1205 defects hit it
  is inference (its tokenizer serialization splits the added tokens out into `added_tokens.json`,
  the shape that loses them). `ship_qwen2vl_derivative.sh` now ends with this gate, and the note
  above its step 5 gives the HF `tokenizer.json` rebuild for when it fails. The builder also no
  longer writes `start_token = "None"` — that literal went out as the word `None` at the head of
  every prompt on ten published fast_vlm bundles (BOS audit, 2026-08-27). **Verdict: Qwen2-VL
  derivative conversions are mechanically complete; a format-exact derivative ships once it
  passes both `gate_specials.py` and its task gate (`qwen2vl_work/gate_nuextract.py`).
  NuExtract-2.0-2B has not been re-converted and remains unpublished.**

`north-micro-vision` is Cohere's North-Micro-Vision-Instruct (2.48B, apache-2.0) — the first Cohere-family model on this rail, and its vision tower is the Qwen3-VL design (SigLIP2-SO400M dims, 27 blocks, three *deepstack* mergers). Three things the scripts encode. (1) **Deepstack folds into the single fast_vlm image embedding**: HF adds three extra vision embeddings to the residual stream after decoder layers 0/1/2; the fast_vlm contract carries one embedding, so the encoder emits `concat(h27, h8, h16, h24)` and the adapter computes `merger(h27) + Σ deepstack_merger_i(h_i)` — exactly representable, and the ablation (`northmv_work/phase0_deepstack_ablation.py`, `phase0_teacher_forced.py`) shows fold beats drop on every metric (teacher-forced top-1 0.96 fold / 0.93 fold+1-D positions vs the released model, no collapse). (2) **The decoder re-hosts as `Cohere2ForCausalLM` with a rope-layout patch** (`northmv_work/northmv_rope_patch.py`): CohereCompass rotates Llama-style half-split pairs while stock Cohere2 rotates interleaved pairs; patching Cohere2 to the Llama-style math loads the weights verbatim (text logits maxdiff 0.0) *and* replaces the `BROADCAST_TO` + 5-D `CONCATENATION` that stock Cohere2's rope lowers to — both rejected by the GPU delegate. (3) **Vision GPU rules learned the hard way** (all in `convert_northmv_vision.py`): keep every activation rank ≥3 with a leading batch dim (the Metal delegate computes rank-2 elementwise ops against a rank-2 constant *wrong*, silently — the symptom is "this image is a blank canvas"); pre-scale LayerNorm inputs by a calibrated power of two from block 9 on (register-scale activations |x|≈2000 overflow fp16 (x−m)²); and a `clamp` between LN and the following FC stops the converter folding LN-gamma into the int8 weight. Ship quant is int8 vision + int8 (dynamic) decoder with `prefer_activation_type="fp32_fp16"` declared on the decoder section — Mali-class GPUs accumulate fp16 and overflow at the 256 image-token positions, so an fp16 decoder answers image turns as if blind (echoes the question) while text-only stays perfect; the in-bundle declaration fixes it on every runtime without flags. Vision fp16 is desktop-only (compiling it on a Mali GPU hard-crashes the phone). Tokenizer: bundle the HF `tokenizer.json` — the sentencepiece conversion crashes on null-byte pieces and splits digits differently from Cohere's per-digit pre-tokenizer. The runtime's 1-D positions replace M-RoPE, with the same 2-D-table-ranking caveat as `qwen2-vl-2b`. Needs transformers ≥5.16 (`cohere_compass`) for the vision/prep side and the released litert-torch 0.9.3 stack for the decoder export.

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

- **MiniCPM5-1B** is stock `LlamaForCausalLM`. Its metadata lists 14 `X<|im_end|>\n` **string** stop tokens (`LlmMetaProto.pbtext` has them all, matching the published artifact). *Correction 2026-08-25*: an earlier revision of this note attributed them to "fused merge tokens" in the tokenizer — measured, the tokenizer has **no** vocab entry containing `<|im_end|>` besides the special token itself (id 130073; `.<|im_end|>\n` encodes as 3 tokens). The 14 strings are generated by litert-torch's exporter: the template-derived turn-end stop `<|im_end|>\n` expanded with 13 punctuation prefixes (`_STOP_TOKEN_PREFIXES` in `export_hf/core/litert_lm_builder.py`, a defense against SentencePiece greedy token merging). A stock 0.9.3 export of a MiniCPM5 checkpoint emits the same 16-entry stop set as the published artifact by construction.
- **MiniCPM4/4.1** are custom `MiniCPMForCausalLM` (muP + longrope; remote code needs transformers 4.46 and won't load in 5.x). `prep_minicpm4_as_llama.py` folds them to a stock llama checkpoint (untie; `emb ×scale_emb`; `o/down ×scale_depth/√L`; `lm_head ÷(hidden/dim_model_base)`), and `export_static_longrope.py` strips transformers' `@dynamic_rope_update` (a data-dependent branch torch.export rejects; exact here since long==short factors and factor==1).
- **MiniCPM4 tokenizer**: the HF `tokenizer.json` bundle loses all spaces on decode; the raw `tokenizer.model` bundle lacks the added-token `<|im_end|>`(73440) stop. `fix_sp_added_tokens.py` appends the added tokens to the SP model as USER_DEFINED pieces and the fixed `.spiece` is what gets bundled.
- **Recipe choice (GSM8K n=100, greedy, thinking off)**: MiniCPM5-1B — wi8 **63** vs official artifact 61; data-free int4-b32 lands 48 (both min-max and OCTAV; the official artifact's int4 quantizer is not reproducible from released ai-edge-quantizer, so int8 is the recommended repro target). MiniCPM4-0.5B — bf16 57; OCTAV int4 **50**; min-max int4 collapses to 38 (sub-1B int4 sensitivity), int8 47. MiniCPM4.1-8B — int4-b32 **44/50 (88%)** (8B is int4-robust; desktop-class at 4.9GB).

### 2026-08-25 — Finetune intake: MiniCPM5-1B derivatives ride the DEFAULT convert.py path

MiniCPM5-1B is stock `LlamaForCausalLM` (`model_type: llama`), so its finetunes go through
`scripts/convert.py` with **no routing, no env change, no retrofit** — the plain dense rail in the
standard `ltconv040dev`-class env (litert-torch 0.9.3). The per-model recipe above remains what
reproduces the *published base* artifact (its post-hoc wi8 quantization); derivatives don't need it.
Measured on huihui-ai/Huihui-MiniCPM5-1B-abliterated (a real behavioral finetune — range-probed:
attention/MLP projections differ from base, embeddings untouched, structure identical):

- One command → `converted_pass`: stock export (export-time int8, cache 4096), gate **7/8**
  non-degenerate at ~165 tok/s decode on the verifier's default backend — no gate-backend
  override, and the post-export stop-token guard is a no-op (generation_config already declares
  `[1, 130073]`).
- The stock bundle's metadata converges with the published litert-community/MiniCPM5-1B artifact
  **by construction**: stops = ids {1, 130073} + the same 14 `X<|im_end|>\n` string stops (the
  exporter's punctuation-prefix expansion — see the corrected note above), thought channel
  auto-added (`<think>` in the template), `max_num_tokens` 4096 (the artifact says 1024).
- Template proof: embedded jinja **byte-equal 9062/9062** to the checkpoint's
  `chat_template.jinja`; greedy **A ≡ B** (litert-lm 0.16.0 CLI, cpu, top-k 1: think content
  832/832 chars + final answer verbatim-identical after normalizing the runtime's
  `[thought]`/`[/thought]` channel rendering to the raw `<think>` markers). Two B-lane
  requirements: strip the render's leading `<s>` (the engine prepends BOS itself — double-BOS
  shifts the greedy continuation on this family exactly as the LFM2.5 lane measured), and the
  channel normalization above for thinking outputs.
- C-lane contrast is structurally N/A for this family too: every usable top derivative probed
  (7 of 7) ships the base's `chat_template.jinja` byte-identical (md5 65d7b88e).
- **The exit gate refuses defective derivatives here too**: a tool-use DPO derivative of the
  same base converts cleanly but fails the gate 4/8-degenerate — greedy on simple arithmetic
  enters a repetition loop inside `<think>` and never closes it, and the same defect class
  reproduces in HF transformers on the source checkpoint (bf16 greedy degenerates into
  `32<|im_end|>` spam on the same prompt), so the conversion is faithful and the subject
  itself is broken. Second measured case of the publish bar doing its job (after the Qwen3.5
  adapter case above).

**litert-torch 0.9.4 notes (measured 2026-08-25)**: 0.9.4 ships the `qwen3_5` exportables and
emits the `ExecutorMetadata` section natively — for **LFM2.5**, new 0.9.4 exports no longer need
the retrofit (convert.py's guard detects the existing section and no-ops), while the Qwen3.5
env guidance is **unchanged**: the 0.9.4 native path exports and loads, but its output is
degenerate (endless `<|iim_ending|>` repetition on a 2B stock export; cause not yet isolated) —
keep exporting Qwen3.5 from litert-torch main. Also measured: transformers 5.15.x breaks the
lfm2 export on both 0.9.3 and 0.9.4 (`Lfm2ShortConv.L_cache` AttributeError) — pin
transformers 5.14.x for this family.

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

- **The same GPU re-export carries to int8, and for JP/Thinking the recipe must change with it.** The `_gpu` files above are int4; an int8 GPU file is built the same way but the quantizer step is `--recipe wi8fc`, not the exporter's default export-time int8. On Instruct the export-time recipe is the better one (+2 GSM8K) and it is what that repo ships; on JP it costs **9 points** (56 vs 65) and on Thinking 1 point, so those two repos ship post-hoc linears+embedding with convs left float, and their GPU file has to reproduce the same weights. `fix_zero_block_scales.py` has no role here — `wi8fc` has no block scales at all. The retrofit step is not optional: litert-torch 0.9.3 does not emit `ExecutorMetadata`, and `quantize_litertlm.py` rebuilds the bundle with only three sections.

```bash
cd lfm_work
python convert_lfm25.py LiquidAI/LFM2.5-1.2B-JP out_jp_fp --fp
python ../minicpm_work/quantize_litertlm.py apply out_jp_fp/model.litertlm jp_wi8fc.litertlm --recipe wi8fc
python add_executor_metadata.py jp_wi8fc.litertlm LFM2.5-1.2B-JP_int8_gpu.litertlm   # "22 state buffers: 10 linear-attn, 12 kv"
```

  Delete the quantizer's output before a rebuild — `add_executor_metadata.py` removes its own target, but `quantize_litertlm.py` does not, so a stale intermediate can survive a re-run and get packed. Built on litert-torch 0.9.3 / litert-converter 0.3.1 / ai-edge-quantizer 0.8.0 / litert-lm 0.15.0 / transformers 5.14.1 (pin transformers at 5.14.x — 5.15 renames `Lfm2ShortConv.L_cache` and the lfm2 export dies). Measured against the CPU-lineage int8 it replaces, same harness and same day: GSM8K identical (JP 65 vs 65, Thinking 77 vs 77, n=100 greedy), 8-question gate 8/8 (JP) and 7/8 (Thinking) on both backends, and on CPU the two files generate byte-identical text on a fixed greedy prompt — the re-export changes the graph, not the weights. On a Galaxy S26 (SM8850, litert-lm v0.16.0) the new file delegates fully (542/542 across ten prefill signatures, 501/501 and 519/519) and generates, where the file it replaces reaches 536 of 579 and aborts; 205-token prompt, prefill ~940-980 tok/s, decode ~42 tok/s, peak RSS ~570 MB. Worth noting because it contradicts a common rule of thumb: the int8 file's GPU-side peak came in *below* the 736 MB int4 file's, not at the ~4x-file-size fp32 residency the heuristic predicts. Against the int4 GPU file it is ~23% slower to decode and ~10 GSM8K points more accurate, so the two are a real speed/quality pair rather than a strict upgrade. The `_int8_gpu` files were not gated on iOS.

  The 2.6B pipeline's `fix_zero_block_scales.py` step is a no-op on this family (measured: 0 zero scales across 0 tensors on all three 1.2B checkpoints) — harmless to run, and the 2.6B does need it. The result delegates **fully** on Android OpenCL (501/501 and 519/519 nodes, zero rejected ops) and runs on the macOS GPU backend; **iOS Metal runs this family** on litert-lm >= 0.16.0 — but only when `maxNumTokens` is set to **1024**. What was tracked as LiteRT-LM#3129 turned out to be context sizing on the integration side, not a runtime bug: the Metal delegate compiles its kernels against the KV cache `maxNumTokens` pre-allocates, and a value that does not match the exported plan fails engine creation with a shader-compile error. Verified by generation on an iPhone 17 Pro (`_int4_gpu`, v0.16.0 xcframework): decode ~70 tok/s, peak ~560 MB. On a Pixel 8a the GPU's win is prefill and time-to-first-token (263-token prompt: ~190 tok/s vs ~38-54 on CPU; TTFT ~1.4 s vs ~5 s), while decode is bandwidth-bound and roughly equal (~21 tok/s).
- **`litert_lm_main --benchmark_prefill_tokens` / `--benchmark_decode_tokens` are silently ignored** (measured on the v0.16.0 tag build, both backends): runs that pass them still report `Processed 19 tokens` and ~50 decode tokens. A 19-token prefill reads about 3x slower than a 256-token one on the same file, so a row labelled "prefill 256" that came from those flags is mislabelled. Drive a real prompt with `--input_prompt_file` and cap generation with `--max_output_tokens`.
- **litert-lm ≥ 0.15 needs an `ExecutorMetadata` section** for state-carrying (hybrid) models: 0.15 binds the per-layer conv/attention state buffers through a new `ExecutorMetadataProto` section, and files exported with litert-torch ≤ 0.9.2 don't have it — they run fine on 0.14 but fail at inference on 0.15 with `missing some output TensorBuffers` (attention-only models are unaffected). Fix an existing file in place with `python add_executor_metadata.py in.litertlm out.litertlm` (weights unchanged; the result runs on both 0.14 and 0.15 — this is how the published LFM2.5 repos were updated on 2026-08-04). Checking on the composite from the previous bullet: the 0.15 GPU delegate still rejects `odml.softmax`, so the strip default stays.

### 2026-08-24 — Finetune intake: LFM2.5 derivatives ride the RELEASED litert-torch

Finetunes of LFM2.5 go through `scripts/convert.py` in the standard `ltconv040dev`-class
env (litert-torch 0.9.3 — the lfm2 model_ext ships in releases; litert-torch *main*'s
lfm2 patch is currently incompatible with transformers 5.15, `Lfm2ShortConv` has no
`L_cache`, so do NOT use the Qwen3.5 main-venv for this family). Measured on
huihui-ai/Huihui-LFM2.5-1.2B-Instruct-abliterated (real behavioral finetune, old-
generation 1,783 B chat template — the top derivatives all carry it):

- Stock export succeeds in ~64 s, but **0.9.3's bundling emits no ExecutorMetadata
  section** (3 sections), so litert-lm ≥ 0.15 fails at inference — convert.py now
  retrofits it automatically after export (`ensure_executor_metadata()`, reusing
  `lfm_work/add_executor_metadata.py`; 22 state buffers on the 1.2B; needs a
  litert-lm ≥ 0.15 CLI: `$LITERT_LM_CLI`, else PATH, else `~/venvs/lt0160run`).
- Proof: one command → `converted_pass`, gate **8/8** (verifier default backend — the
  bundle runs on WebGPU at ~162 tok/s decode; the odml.softmax-composite GPU rejection
  recorded above for litert-lm 0.14/0.15 does not reproduce on the 0.16 runtime),
  embedded template byte-equal 1783/1783, greedy **A ≡ B** byte-identical.
- Two method notes: the derivative's old-generation template renders IDENTICALLY to the
  base's current one for every CLI-reachable input (system/user/assistant turns — they
  diverge only in the tools path), so a C-lane contrast is structurally N/A for this
  family's typical derivatives; and the B lane must strip the template's leading
  `bos_token` before `--no-template` (the engine prepends BOS itself — double-BOS
  changes the token stream and the greedy output, the `granite-4.1-3b` BOS trap in
  A/B-proof form).

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

### LFM2.5-230M (the smallest decoder) — a template the runtime cannot parse, and a shape that kills the GPU shader compile

`lfm_work/convert_lfm25_230m.py` converts [LiquidAI/LFM2.5-230M](https://huggingface.co/LiquidAI/LFM2.5-230M) (14L: 8 ShortConv + 6 attention, 230M params, tied embeddings). Published: [litert-community/LFM2.5-230M](https://huggingface.co/litert-community/LFM2.5-230M). Same env as the family above (litert-torch 0.9.3, **transformers pinned 5.14.x**). Three facts none of the larger LFM2.5 ships hit:

```bash
cd lfm_work
python convert_lfm25_230m.py out_230m           # export-time int8 (NOT what ships — see below)
python convert_lfm25_230m.py out_230m_fp --fp   # unquantized, the ship pipeline's source
python fix_230m_template.py out_230m_fp/model.litertlm fp_fixed.litertlm
python ../minicpm_work/quantize_litertlm.py apply fp_fixed.litertlm int8.litertlm --recipe wi8fc
python ../minicpm_work/quantize_litertlm.py apply fp_fixed.litertlm int4.litertlm --recipe wi4b32_wi8 --algo octav
python ../bonsai_work/fix_zero_block_scales.py int4.litertlm int4_z.litertlm   # measured no-op here, family pipeline order
python add_executor_metadata.py int8.litertlm LFM2.5-230M_int8.litertlm
python add_executor_metadata.py int4_z.litertlm LFM2.5-230M_int4.litertlm
```

- **The checkpoint's chat template does not run on litert-lm's minijinja — every naive bundle dies on its first message.** Two constructs: HF's `{% generation %}`/`{% endgeneration %}` assistant-mask markers are a *parse* error ("unknown statement generation"), and `message.get("content")` is a *render* error ("map has no method named get"). `fix_230m_template.py` strips the markers and rewrites the two `.get` sites to plain indexing inside the packed bundle (weight-identical); `make_metadata_230m.py` proves the repaired template renders byte-identically to the original through HF `apply_chat_template` on 6 conversation shapes (user/system/multi-turn/past-think/closed/tools) — run it before trusting any edit. String methods (`.split`, `.endswith`) sit in branches the runtime never reaches and need no rewrite.
- **cache_length 4099 + the 1024 prefill signature = GPU engine-creation failure at shader compile** (`CreateShaderModule` validation error, measured on macOS WebGPU). Isolated to the pair: ladder+4096 compiles, 128-only+4099 compiles, 1024-only+4099 fails. The 230M ships **cache 4096** — a clean multiple of the widest signature — and delegates fully on Adreno OpenCL (492/492 nodes on all 11 prefill signatures + decode, Galaxy S26) and on Metal/WebGPU.
- **This tune sits on the JP/Thinking side of the family conv-int8 law.** GSM8K is floor for a 230M non-reasoning model (bf16 23/100 — useless for the A/B), so the recipe decision ran on instruction following (`ifeval_lite_230m.py`, a re-implementation of google/IFEval's mechanically checkable types; numbers comparable only within the harness): bf16 60.0 / export-time conv-int8 53.3 / **post-hoc wi8fc 58.3** (n=120 prompt-level strict). Convs stay float; wi8fc ships as the int8 file.
- One more per-device fact worth recording: **at 230M, int4 is not the fast file everywhere.** iPhone 17 Pro Metal decodes int4 13% faster than int8 (162 vs 143 tok/s), but Galaxy S26 Adreno decodes it 2.8× *slower* (43 vs 122) and Mac is a wash — the blockwise dequant overhead dominates at this size. int8 is the primary recommendation.

### sarashina2.2-0.5b / 1b instruct (SB Intuitions, Japanese) — a SentencePiece vocab whose chat specials are CONTROL pieces, and a BOS the model never saw

`sarashina_work/convert_sarashina.py` converts [sbintuitions/sarashina2.2-0.5b-instruct-v0.1](https://huggingface.co/sbintuitions/sarashina2.2-0.5b-instruct-v0.1) and [-1b-instruct-v0.1](https://huggingface.co/sbintuitions/sarashina2.2-1b-instruct-v0.1) (plain `LlamaForCausalLM`, 24 layers, GQA 16:8, **102,400-entry untied vocab**, MIT). Published: [litert-community/sarashina2.2-0.5b-instruct-v0.1](https://huggingface.co/litert-community/sarashina2.2-0.5b-instruct-v0.1), [litert-community/sarashina2.2-1b-instruct-v0.1](https://huggingface.co/litert-community/sarashina2.2-1b-instruct-v0.1).

```bash
# int8 ship (int8 dynamic on linears + embedding)
python sarashina_work/convert_sarashina.py sbintuitions/sarashina2.2-0.5b-instruct-v0.1 out/s05_int8 templates/sarashina_simple.jinja dynamic_wi8_afp32
# int4 ship (blockwise-32 + OCTAV linears, int8 embedding)
python sarashina_work/convert_sarashina.py sbintuitions/sarashina2.2-0.5b-instruct-v0.1 out/s05_int4 templates/sarashina_simple.jinja BOCTAV4
# same two lines with sarashina2.2-1b-instruct-v0.1 for the 1b
# gates: scripts/verify_quality.py (EN 8Q) + sarashina_work/verify_quality_ja.py (JA 8Q + UTF-8 streaming probe),
#        sarashina_work/tokenizer_parity.py (engine vs HF ids), sarashina_work/gate_multiturn_ja.py,
#        sarashina_work/jcqa_eval.py (JCommonsenseQA, bf16 vs bundle in one harness)
```

The wrapper is `scripts/export_simple_template.py` (structured prompt templates, prefill ladder 1024..1, KV 4096) plus two facts this checkpoint needs:

- **Embed the HF `tokenizer.json`, not the vendor `tokenizer.model`.** litert-torch copies `tokenizer.model` verbatim whenever the HF tokenizer exposes `vocab_file` ending in it. sarashina's SentencePiece model types every chat special (`<|user|>` `<|assistant|>` `<|system|>` `</s>`) as **CONTROL**, and a bare sentencepiece `Encode` never matches CONTROL pieces from text — the template's `<|user|>` comes out as `< | user | >` (5 pieces) and the model never sees its role markers. The HF fast tokenizer matches them as added tokens. Clearing `vocab_file` makes the exporter fall through to `save_pretrained` → `tokenizer.json`. Its decoder chain has no `Strip` step, so the Metaspace per-token-strip trap does not apply. Engine-vs-HF ids: identical on 7 probes (specials-as-text, Japanese, emoji/rare-kanji byte-fallback, Latin-1, whitespace).
- **Write no `start_token`.** `add_bos_token` is false and the official template emits no `<s>`, but the exporter sets `start_token` from `tokenizer.bos_token` regardless, and the runtime then prepends it. Measured on bf16 with the same rendered prompt ± `<s>`: the 0.5b drops the 8-question gate from 6→5 (English) and 8→7 (Japanese); the 1b is unmoved. `NO_START_TOKEN=1` (set by the wrapper) clears `bos_token` before the metadata is built.
- No post-processing: plain attention (no ExecutorMetadata retrofit), template renders on the runtime as-is.
- Per-device fact: **on the Galaxy S26 int4 is not faster than int8 for these files** (0.5b GPU decode 32–40 vs 36 tok/s; CPU 19.5 vs 20–21) — the untied 102,400 vocab makes the int8 embedding + lm_head the dominant per-token cost, so the int4 dequant overhead on the linears buys nothing. int8 is the recommended file; int4 is the size option.

## granite-4.0-h (Mamba2 + attention hybrid) — first Mamba2 hybrid on the released runtime

`granite_work/convert_granite4h.py` converts IBM's granite-4.0-h dense-hybrid models (Mamba2 selective-scan blocks interleaved with grouped-query attention) to `.litertlm`. Published: [litert-community/granite-4.0-h-1b](https://huggingface.co/litert-community/granite-4.0-h-1b). **Requires litert-lm ≥ 0.15 to run** (the hybrid conv/SSM states bind through the `ExecutorMetadata` section).

```bash
cd granite_work
git clone https://github.com/google-ai-edge/litert-torch litert-torch-granite
# the pinned base is no longer reachable from main (measured 2026-08-25), so a
# plain short-SHA checkout fails — fetch the commit by full SHA first
git -C litert-torch-granite fetch origin 115a13607c730c81018bb9789138a3e5e5119e3d
git -C litert-torch-granite checkout 115a13607c730c81018bb9789138a3e5e5119e3d
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

*Followed up 2026-08-27:* robustness is not fidelity, and the 1b's bundle was
left carrying that start_token. The audit of all published bundles confirms
`granite-4.0-h-1b` still prepends `<|end_of_text|>` where the reference stream
has no BOS at all — the same defect the 350m was fixed for, on a model that
merely tolerates it better. Same one-command fix (`drop_start_token.py`).

```bash
cd granite_work
PYTHONPATH=litert-torch-granite python convert_granite4h.py ibm-granite/granite-4.0-h-350m out_granite_350m
python drop_start_token.py out_granite_350m/granite-4.0-h-350m_int8.litertlm granite-4.0-h-350m_int8.litertlm
python ../lfm_work/add_executor_metadata.py out_granite_350m/model.litertlm out_granite_350m/model_fp_meta.litertlm
python make_fp16_variant.py out_granite_350m/model_fp_meta.litertlm granite-4.0-h-350m_fp16.litertlm --drop-start-token
```

Gates on the published files (litert-lm 0.15 CPU, Mac + iPhone 17 Pro): fp16 = 8/8 sanity, greedy output token-identical to the HF fp32 reference on our probes, prompt-length sweep 12–200 clean; int8 = 8/8 sanity, sweep clean except a known 33–37-token band where replies can end early (int8 noise interacting with one prefill chunk shape — fp16 is clean there; documented on the card). On phones ship int8: the CPU runtime unpacks fp16 weights to fp32 in RAM (~3.7 GB peak on iPhone vs ~2.1 GB for int8).

#### The GPU variant, and why the published two are not it

The two files above are CPU files, and the reason is in the graph rather than in the runtime. `transformers`
writes the Mamba2 SSD contractions as broadcast-multiply-reduce, which materialises rank-5 and rank-6
intermediates; every mobile GPU delegate refuses tensors above rank 4. Measured on the published lineage,
`prefill_1024` carries 411 rank-6 and 739 rank-5 tensors, and the Adreno delegate takes 109 of its 3673 ops
before the engine gives up.

The exporter patch has since rewritten those five contractions as batched matmuls with the chunk and head axes
folded into the batch axis. Re-exporting on it gives a graph that is rank ≤ 4 everywhere, and one extra
declaration takes it the rest of the way:

```bash
cd granite_work
PYTHONPATH=litert-torch-granite python convert_granite4h.py ibm-granite/granite-4.0-h-350m out_granite_350m
python drop_start_token.py out_granite_350m/granite-4.0-h-350m_int8.litertlm g350_nobos.litertlm
python ../scripts/set_activation_type.py g350_nobos.litertlm granite-4.0-h-350m_int8_gpu.litertlm --type fp32
```

`prefer_activation_type = "fp32"` is not optional at this size. Without it the file still delegates, but int8
activation noise on the GPU costs real quality and speed — measured on an M4 Max, 5/8 on the 8-question gate
at 76 tok/s decode, against **8/8 at 148 tok/s** with the declaration. The archive had this recorded as a
1b-only rescue; that was single-gate evidence and it did not survive re-measurement.

Gate, Galaxy S26 (Adreno), `litert_lm_advanced_main` v0.16.0: **every one of 12 subgraphs fully delegated in a
single partition, no XNNPack split, correct answer.** 205-token benchmark, n=2: 486 / 484 tok/s prefill,
48.8 / 47.9 tok/s decode, TTFT 0.48 s, peak VmHWM ~2.13 GB. Engine init is 8.6–9.6 s and is separate from TTFT.

Two things this does **not** fix:

- **fp16 stays CPU-only.** The rank wall is gone from it too, but float-casting `EMBEDDING_LOOKUP` leaves
  `Empty quantization params` and a `DEQUANTIZE` the delegate will not take — 3326 of 3536 ops, engine refused.
  The int8 recipe quantizes the same table and delegates. There is no fp16 GPU variant.
- **The engine-reuse band moves, it does not close.** The published int8 file breaks at chat-templated lengths
  33–37 when one `Engine` is reused across a growing shared prefix; on the re-export the band is **37–41**. It
  is the runtime carrying recurrent state across conversations ([LiteRT-LM#3165](https://github.com/google-ai-edge/LiteRT-LM/issues/3165)),
  not the weights, and a fresh `Engine` per conversation is clean at every length tested either way (35–45 on the re-export;
  the hermetic probe's template overhead puts its own floor at 18).

The cheapest way to check a re-export before spending device time is to unpack it and histogram tensor ranks
per subgraph: rank ≥ 5 anywhere in a prefill graph means the delegate will refuse, and no phone is needed to
know it.

### granite-4.0-h finetune intake (Tashkeel-350M-v2) — the recipe rides derivatives unchanged

Measured 2026-08-25 on [Etherll/Tashkeel-350M-v2](https://huggingface.co/Etherll/Tashkeel-350M-v2)
(Arabic diacritization SFT of granite-4.0-h-350m; the most-downloaded granite-4.0-h derivative
on the Hub). Published: [mlboydaisuke/Tashkeel-350M-v2-LiteRT](https://huggingface.co/mlboydaisuke/Tashkeel-350M-v2-LiteRT)
(fp16 exact-parity + int8, measured card + litertlm_manifest.json). Family fact, same as
every LLM family measured so far: the finetune's
`chat_template.jinja` is byte-equal to the base's, and the tokenizer/generation config diffs are
cosmetic (`model_max_length`, eos as list) — so the base recipe applies verbatim:

```bash
cd granite_work
PYTHONPATH=litert-torch-granite python convert_granite4h.py Etherll/Tashkeel-350M-v2 out_tashkeel
python drop_start_token.py out_tashkeel/Tashkeel-350M-v2_int8.litertlm Tashkeel-350M-v2_int8.litertlm
# exact-parity variant (the int8 flips two diacritics on the 10-case gate; fp16 is clean)
python ../lfm_work/add_executor_metadata.py out_tashkeel/model.litertlm out_tashkeel/model_fp_meta.litertlm
python make_fp16_variant.py out_tashkeel/model_fp_meta.litertlm Tashkeel-350M-v2_fp16.litertlm --drop-start-token
python gate_tashkeel.py Tashkeel-350M-v2_fp16.litertlm --backend cpu
```

Three-point proof (litert-lm 0.16.0 CLI, CPU): (a) export ✓ — 481 MB int8, 64 state buffers
(56 conv/SSM + 8 KV), ExecutorMetadata present, stop token = the tokenizer's
`<|end_of_text|>` 100257; (b) embedded template byte-equal 6418/6418 to the checkpoint's;
(c) greedy A/B vs HF fp32 on the model's own task (`gate_tashkeel.py`: the card's worked
example + 9 undiacritized MSA probes, both sides rendering the identical string):
**float 10/10 and fp16 10/10 byte-exact**, int8 8/10 — the two int8 misses are
single-diacritic greedy flips (one also over-runs past the natural end), and both probes are
byte-exact on the float parent, so they are quantization cost, not conversion error — the same
350M-scale int8 sensitivity the base model's card documents (fp16 is the exact-parity variant
there too).
The start_token drop is REQUIRED here as on the base: an HF BOS A/B flips the greedy output
from correct diacritization to garbage (`صلى السلام عليكم`).
Honest note: the model card's own worked example (`اَلسَلَامُ عَلَيْكُمْ`) matches neither the
bundle nor HF fp32 greedy (`السَّلَامُ عَلَيْكُمْ`) — the card example disagrees with the
checkpoint's own greedy output, so HF parity, not the card string, is the certification.

Stock-export failure shape for the record (why the pinned checkout stays): released
litert-torch 0.9.4 has zero mamba support in the wheel; its generic path maps granite's mamba
layers onto the qwen3.5-era linear-attention cache and dies before tracing —
`AttributeError: 'GraniteMoeHybridConfig' object has no attribute 'linear_key_head_dim'`
(`export_hf/core/cache.py`, `create_from_config`). `scripts/convert.py` now routes
`model_type: granitemoehybrid` to this recipe automatically (HYBRID_RECIPE) when the pinned
checkout is present, and refuses with the exact setup command when it is not; task-specific
derivatives gate through `--gate-script` (the generic 8-question gate certifies nothing on a
model that diacritizes its input instead of answering). One-command run, measured end-to-end
from a clean shell:

```bash
python scripts/convert.py Etherll/Tashkeel-350M-v2 --gate-script granite_work/gate_tashkeel.py
```

recipe export 454 s → start_token dropped → task gate on CPU → int8 8/10 (the same two
single-diacritic flips, reproduced deterministically) → exit 1 `converted_gate_failed` with the
bundle and `convert_report.json` written. The exit code is the honest int8-at-350M verdict, not
a conversion failure — the fp16 flow above is the exact-parity finish.

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

### 2026-08-14 — Qwen3.5-4B: two walls the 0.8B never hit

Same driver, same patch, one command (`convert_qwen35_hybrid.py Qwen/Qwen3.5-4B out_qwen35_4b`, then the fp32act repack). Published: [litert-community/Qwen3.5-4B](https://huggingface.co/litert-community/Qwen3.5-4B). The 4B (24 linear-attention + 8 attention layers) is architecturally the 0.8B scaled up — but it trips two size-dependent walls, both now handled in the patch/driver:

- **Grouped value heads reject on GPU.** The 4B is the first Qwen3.5 with `linear_num_value_heads` (32) ≠ `linear_num_key_heads` (16); upstream expands q/k with `repeat_interleave(ratio, dim=2)`, guarded by `if ratio > 1` — so the 0.8B (ratio 1) never traces it. Its rank-5 `unsqueeze→expand→reshape` lowering is rejected by the GPU delegate (`RESHAPE: Tensor dimensions must be less than 5`) and engine creation fails outright. The patch re-expresses it rank-≤4: fold the batch dim, materialize copies with `torch.cat`, re-fold — bitwise-identical to the stock op.
- **Signature count is charged in RAM, and a 248k-vocab 4B pays too much per signature.** With the 0.8B's full prefill ladder (11 lengths + decode = 12 signatures), a 12 GB iPhone jetsam-kills the app at Metal program init #9 — before any inference. Every signature costs engine memory even if never called. The 4B ships a reduced ladder (`QWEN35_PREFILL_LADDER=1024,256,64,16,4,1`, 7 signatures): GPU peak drops to ~4.9 GB on iPhone 17 Pro and the engine creates cleanly, at no correctness cost (the runtime plans slightly coarser chunks; the prompt-length sweep below re-verified against the new plans).

Ship verification (all on the shipped file or its float parent): teacher-forced parity vs HF fp32 across 48 positions — top-1/top-5 100%, Pearson 1.0000, KL ≈ 0; 8-question gate **8/8 on CPU and GPU on both litert-lm 0.15.0 and 0.16.0**; prompt-length robustness, fresh engine per length, 40/40 CPU + 20/20 GPU; iPhone 17 Pro (Metal) answers the composite 8-question probe **word-for-word identically to HF fp32** at 9.3 tok/s decode / 87 tok/s prefill / 4.98 GB peak (CPU: 5.4 / 24.0 / 1.70 GB). Mac M4 Max (`-p 256 -d 256 --runs 3 --cache no`, 0.16.0): GPU 672 / 62.0 tok/s, CPU 243 / 19.9. Honest limits: on the composite probe the CPU int8 path answers 6 of 8 (everything it says is correct; individual questions are 8/8) — the GPU path is the fidelity path; and 8 GB-class Android phones cannot fit the 4.1 GB file (Apple-hardware-first release).

### 2026-08-14 — Qwen3.5-2B: the easy sibling (both 4B walls stay un-tripped)

Same driver, same patch, one command (`convert_qwen35_hybrid.py Qwen/Qwen3.5-2B out_qwen35_2b`, then the fp32act repack). Published: [litert-community/Qwen3.5-2B](https://huggingface.co/litert-community/Qwen3.5-2B). The 2B (18 linear-attention + 6 attention layers, hidden 2048) sits between the 0.8B and 4B and needs **no model-specific work**:

- Its linear-attention heads are ungrouped (`linear_num_key_heads` = `linear_num_value_heads` = 16, ratio 1), so the 4B's grouped-value-head interleave rewrite never traces — same as the 0.8B.
- The **full prefill ladder fits on-phone** (11 prefill lengths + decode = 12 signatures): iPhone 17 Pro GPU peaks at ~5.3 GB during engine creation, comfortably under the 12 GB device's jetsam ceiling — the 4B's reduced-ladder workaround is not needed at this size.

Ship verification (all on the shipped file or its float parent): teacher-forced parity vs HF fp32 across 48 positions — top-1/top-5 100%, Pearson 1.0000, KL ≈ 0; 8-question gate **8/8 on CPU and GPU on both litert-lm 0.15.0 and 0.16.0**; prompt-length robustness, fresh engine per length, 40/40 CPU + 20/20 GPU; iPhone 17 Pro (Metal) answers the composite 8-question probe **word-for-word identically to HF fp32 through answer 7** — including the model's own arithmetic slip on question 1 — then adds a correct 8th answer where fp32 ends its turn one token earlier, at 24.3 tok/s decode / 237.7 tok/s prefill / TTFT 0.73 s / 5.33 GB peak (CPU: 16.2 / 206.5 / 0.77 s / 1.52 GB). Mac M4 Max (`-p 256 -d 256 --runs 3 --cache no`, 0.16.0): GPU 1486 / 114.3 tok/s, CPU 592 / 37.6. Honest limits: on the composite probe the CPU int8 path degrades (3 of 8, deterministic, identical on Mac and iPhone; individual questions are 8/8 everywhere) — the GPU path is the fidelity path; and on a Pixel 8a the OpenCL delegate accepts the whole graph (zero rejections, full delegation on every compiled signature) but engine creation exhausts an 8 GB phone's memory before finishing — Apple-hardware-first, 12 GB+ for Android GPU.

### 2026-08-24 — Finetune intake: stock litert-torch *main* converts Qwen3.5 derivatives

The recipe above reproduces the shipped checkpoints; **finetunes of Qwen3.5 now go through
`scripts/convert.py`** instead. Measured on the 0.8B base plus a real 4B product finetune:

- **litert-torch main ships a `qwen3_5` model_ext** (absent from every released wheel — PyPI
  tops out at 0.9.3, no nightly package, no tag contains it): full reauthored exportables
  routed on `model_type in ('qwen3_5', 'qwen3_5_text')`, no patch, no template surgery.
  One-command install into a **fresh venv**: `pip install 'litert-torch @
  git+https://github.com/google-ai-edge/litert-torch.git'` (measured @ `8379afb` =
  0.10.0.dev; its setup.py pins nightly ai-edge deps — do not share a venv with the released
  0.9.3 pipelines). On 0.9.3 the generic path dies mid-trace
  (`LiteRTLMConvCacheLayer.update_conv_state() takes 2 positional arguments but 3 were
  given`); convert.py pre-empts that with a `model_ext_missing` refusal carrying the install
  action, before any download.
- **Stock export embeds the derivative's own template verbatim and emits the
  ExecutorMetadata section** (conv/recurrent state binding, litert-lm ≥ 0.15) that the
  recipe era added by hand. Qwen/Qwen3.5-0.8B: 226 s, 754 MiB int8, template byte-equal
  7755/7755, CPU gate 6/8 PASS ~47 tok/s. The 6/8 — the shipped wi8fc build is 8/8
  word-for-word — is the stock export-time int8 quantizing convs + delta rule against the
  family rule; that quality notch is the price of the no-recipe path.
- **CPU-only.** The GPU delegate rejects the stock GatedDeltaNet lowering (`TRANSPOSE:
  Permutation for transpose is invalid`, 195/20239 ops delegated) and engine creation fails
  outright — no fallback — so convert.py's exit gate runs `--backend cpu`
  (verify_quality.py grew that flag). GPU-delegable Qwen3.5 remains the per-model recipe
  above.
- **≥3B derivatives get the 4B ship's reduced prefill ladder** (1024,256,64,16,4,1 + decode
  = 7 signatures): signatures are charged in engine RAM even if never called (2026-08-14
  above), and the full-ladder 4B export's converter passes were OOM-killed twice on a
  128 GB host before the reduction (even reduced, the export peaks > 84 GB and swaps hard;
  ~26 min).
- **Proof finetune — numind/NuExtract3** (4.54B multimodal checkpoint, MTP weights in a
  separate file, 123k DL): one command → `converted_pass`, gate **8/8** non-degenerate
  ~20 tok/s CPU, embedded template byte-equal (6,692 chars — including its
  `{% generation %}` blocks, which the current runtime parses), and greedy **A ≡ B**
  byte-identical (runtime-applied template == HF-side render fed raw) with **C ≠ A** (the
  base template's thinking-ON default yields a different answer — the template diff is
  behavior-bearing, and the runtime is applying the derivative's).
- Multi-turn honesty: the stock Qwen3.5 template strips `<think>` from history assistant
  turns, which violates the runtime's incremental render contract (see the recipe above) —
  derivatives inheriting it are single-turn-proven only.
- ⭐ **Qwen3.5 checkpoints under-declare their stop tokens, and both of the stock
  exporter's derivations miss the turn end.** The repos ship no generation_config.json and
  declare only `text_config.eos_token_id = 248044` (`<|endoftext|>`); the exporter's
  template-derived stop never fires because the stock template's probe render hits its own
  `raise_exception` (the "Failed to parse chat template" line in the export log is
  load-bearing). Every stock Qwen3.5 bundle stops ONLY at `<|endoftext|>` —
  `<|im_end|>` = 248046 missing — which the base/NuExtract3 gates masked (those models
  happen to emit `<|endoftext|>` after answering) and a thinking derivative exposed:
  generation runs straight through `<|im_end|>` and the gate reads 0/8 empty.
  `convert.py` now runs `ensure_turn_end_stop()` after every export: if the checkpoint
  tokenizer's `eos_token_id` is missing from the bundle's stop_tokens, it is appended
  (litert_lm_builder unpack → LlmMetadataProto.pbtext → repack); no-op when stops are
  declared correctly (the dense rail).
- **Adapter derivatives ride merge-first unchanged** (cs552-the-expendables/
  patientagent-sft-glm5, r16 α32 targeting the GatedDeltaNet projections):
  `AutoModelForCausalLM` on the multimodal base loads text-only (426 tensors, 0 vision;
  config rewritten to `qwen3_5_text` — why HYBRID_STOCK keys both model_types), and the
  merged hybrid-layer weights verify bitwise as `W + (α/r)·B@A`. That subject then FAILED
  the gate 0/8 for a model-own reason worth knowing: it inherits the base 4B template
  (thinking-ON default) but its SFT never learned to close `<think>` — measured identical
  in HF (greedy emits the answer immediately and never `</think>`), so the runtime files
  the whole answer into the filtered thought channel. The conversion is faithful; the gate
  correctly refuses a model that cannot exit its think block. peft must be installed in
  the export venv (peft 0.20.0 + accelerate 1.14.0 alongside the main stack import-clean).

### 2026-09-04 — Qwen3.5-0.8B / 2B MTP (speculative decoding): built and gated, held by two runtime findings

`mtp_work/` exports Qwen3.5 with a `verify` signature and packs the checkpoint's own 1-layer MTP head as the runtime's drafter section, so the bundle runs under `litert-lm run --speculative-decoding true` (≥ 0.16.0). Recipe, patch, gates and numbers: `mtp_work/README.md`. The short version: Mac CPU decode 1.12× (0.8B) / 1.24× (2B) with acceptance matching the desktop oracle; Galaxy S26 CPU 0.71× on the 0.8B for document text; and a runtime defect — under the flag the turn-final stop token is never committed, so multi-turn transcripts lose `<|im_end|>` after every assistant turn (LiteRT-LM #3439, reproduces on 0.16.0 and 0.16.1; probe script in the directory). Not shipped; a repro bundle is public at `mlboydaisuke/Qwen3.5-0.8B-MTP-repro-LiteRT`.

## Qwen3.5-2B VISION build — the shipped text model was half of a VLM

The Qwen3.5 checkpoints are multimodal, and the text conversion above deliberately drops the vision tower. `qwen35vl_work/` keeps it: the checkpoint's own 24-layer, 1024-dim ViT (no DeepStack — `deepstack_visual_indexes` is empty upstream), wired to LiteRT-LM's `fast_vlm` contract at a static 512x512. Published alongside the text file: [litert-community/Qwen3.5-2B](https://huggingface.co/litert-community/Qwen3.5-2B) (`Qwen3.5-2B-VL_int8.litertlm`).

```bash
cd qwen35vl_work
# vision tower -> two tflites (encoder + adapter); ~10 min on an M4 Max
IMG=512 python convert_qwen35_vision.py out/qwen35vl-vision
python vision_quant_ab.py out/qwen35vl-vision          # fp32/fp16/int8 A/B on real images

# decoder for the fast_vlm contract (externalised embedder), SIX prefill signatures
PREFILL=1024,256,64,16,4,1 CACHE=4096 \
  PYTHONPATH=../qwen35_work/litert-torch-qwen35 \
  python export_qwen35vl_decoder.py Qwen/Qwen3.5-2B out/qwen35vl-decoder

# assemble + bind the hybrid states
DEC=out/qwen35vl-decoder VIS=out/qwen35vl-vision TOK=<tokenizer.json> \
  VENC=vision_encoder_fp16.tflite VADP=vision_adapter_int8.tflite DEC_ACT=fp32 \
  python build_qwen35vl_bundle.py
python ../scripts/add_executor_metadata.py out/.../Qwen3.5-2B-VL.litertlm <final>.litertlm
```

Uses the same patched `litert-torch` worktree as the text build (clone + checkout + apply from the section above).

**Six prefill signatures, not the text build's eleven.** Every exported signature is charged engine memory whether or not it is called. The text build's full 1–1024 ladder peaks around 5.3 GB on an iPhone 17 Pro and fits; attaching the vision tower pushes the same ladder past a 12 GB phone's jetsam ceiling — all three Metal legs were killed after initialising 11 of 12 signatures. Cutting to `1024,256,64,16,4,1` fixed it with no other change, and **also tripled CPU decode** (3.8 → 14.1 tok/s, first token 23.7 s → 5.1 s), because the runtime's XNNPACK weight repack is charged per signature too.

**The pad guard stops arming when the embedder is externalised.** `fast_vlm` moves the embedder into its own tflite, which removes token ids from the decoder graph — so the gated-delta prefill-pad guard, which derived its valid mask from `input_ids != 0`, silently did nothing, and padding would have poisoned the recurrent state during multi-chunk prefill. It now falls back to position monotonicity (`pos[1:] > pos[:-1]`, first slot valid), the same contract the upstream exportable module uses. A hermetic prompt-length sweep (40/40 clean) is what proves the guard armed; nothing else in the gate set catches it.

**Vision export traps, all encoded in `convert_qwen35_vision.py`.** The upstream tower is dynamic-resolution and its `grid_thw` preprocessing aborts `torch.export` outright, so it is re-authored for one static image: Conv3d patch-embed folded to Conv2d over the duplicated temporal frame, learned position embeddings bilinearly resampled to the static grid ahead of time, 2-D rope precomputed, explicit full attention instead of the variable-length split. Patches stay in raster order and the 2x2 merge happens in the adapter as four strided slices and a concat, so no `GATHER_ND` reaches the mobile GPU delegate. Two numerical guards matter: every activation keeps a leading batch dimension (Metal computes rank-2 elementwise against a rank-2 constant silently wrong), and LayerNorm is pre-scaled by a calibrated power of two, because from block 11 the residual stream carries values large enough to overflow fp16 accumulation.

**Ship shape: fp16 encoder, int8 adapter.** The exported tower matches the model's own vision path at correlation 0.9999999992 in fp32. Quantizing the encoder to int8 drops that to 0.97–0.98 on real photographs while the adapter survives int8 at 0.9998, so only the adapter is quantized. An int8-encoder build is 300 MB smaller and still answers correctly — that is the variant to use on Mali, where an fp16 vision encoder is known to crash devices of this class.

**Contract cost.** `fast_vlm` feeds sequential positions, so the checkpoint's M-RoPE collapses to plain RoPE. `hf_oracle_gen.py` measures it: against the full M-RoPE reference over nine image/prompt pairs, one generation is identical and the other eight are fluent same-content paraphrases that diverge deep in the answer. `suite_gate_vl.py` scores a bundle against both references.



### The 0.8B rides the same script — but not the same numbers

Published as `Qwen3.5-0.8B-VL_int8.litertlm` on [litert-community/Qwen3.5-0.8B](https://huggingface.co/litert-community/Qwen3.5-0.8B) (1.30 GB). Identical commands, with the model id and output dirs swapped:

```bash
cd qwen35vl_work
MODEL=Qwen/Qwen3.5-0.8B IMG=512 python convert_qwen35_vision.py out/qwen35vl08-vision
MODEL=Qwen/Qwen3.5-0.8B python vision_quant_ab.py out/qwen35vl08-vision

PREFILL=1024,256,64,16,4,1 CACHE=4096 \
  PYTHONPATH=../qwen35_work/litert-torch-qwen35 \
  python export_qwen35vl_decoder.py Qwen/Qwen3.5-0.8B out/qwen35vl08-decoder

DEC=out/qwen35vl08-decoder VIS=out/qwen35vl08-vision TOK=<tokenizer.json> \
  VENC=vision_encoder_fp16.tflite VADP=vision_adapter_int8.tflite DEC_ACT=fp32 \
  OUT_NAME=Qwen3.5-0.8B-VL_int8.litertlm python build_qwen35vl_bundle.py
python ../scripts/add_executor_metadata.py <out>.litertlm <final>.litertlm
```

**The tower is a different size, so two measured values must be re-derived rather than inherited.** The 0.8B ViT is **12 layers at 768 dim** against the 2B's 24 at 1024. The script calibrates the fp16-safe LayerNorm scales per run, so nothing needs editing — but the resulting table is genuinely different (16 from block 6 on, 512 at the final norm, versus 32 from block 11 for the 2B), and hand-copying a sibling's table would silently mis-scale the tower. The int8 encoder also behaves better here: correlation **0.9916-0.9951** on real photographs against the 2B's 0.974-0.984, so the int8-vision variant is on firmer ground at this size. fp32 parity is 0.9999999999.

**Build the vision tflites with litert-converter 0.4.0, not 0.3.1.** On 0.3.1 the adapter's 2x2 merge (`f[:, 0::2, 0::2, :]`) lowers to **six `GATHER_ND` ops** — a mobile-GPU hard wall — where 0.4.0 gives six `STRIDED_SLICE`, matching the shipped 2B adapter. This happens at every shape including the 2B's, so it is a toolchain variable rather than a geometry one, and **parity is bit-identical across both builds**: no accuracy check catches it. Gate the op histogram (the script prints `enc ops` / `adp ops`) and require `gather`, `flex` and `custom` all empty before anything downstream consumes the file.

**`add_executor_metadata.py` is not optional on a litert-torch 0.9.3 export.** 0.9.3 emits no `ExecutorMetadataProto`, and a fresh bundle dies at engine creation with `INTERNAL: ... No KV cache inputs found.` even though its decoder tflite carries all 48 `kv_cache` inputs. That is what the `_final` suffix means on these artifacts. When it happens, run a known-good sibling bundle on the same CLI first — that exonerates the runtime and the venv in one command.

**The six-signature ladder fit iPhone Metal on the first export**, with no jetsam and no `maxNumTokens` override — unlike the 2B, which needed the ladder cut after three killed Metal legs. Measured on an iPhone 17 Pro: vision turns 65.8 tok/s with grounding 2/2, text probe 45.7 tok/s on Metal at ~3.8 GB peak.

**Where the device and the desktop disagree, run the same prompt on the desktop before blaming the file.** On the composite 8-question probe this bundle answers 8/8 on Mac GPU *and* 8/8 on Mac CPU, while the phone scores 6/8 on Metal and 3/8 on CPU. None of the device misses reproduce on Mac, so they are device-side and not the conversion — the opposite of the 2B, whose arithmetic slip did reproduce on Mac and was therefore the model. The iPhone CPU result was re-run cold three days later after an app reinstall and came back byte-identical, so it is deterministic rather than thermal.

### OvisOCR2 — an OCR finetune rides the 0.8B vision rail unchanged

Published as [mlboydaisuke/OvisOCR2-LiteRT](https://huggingface.co/mlboydaisuke/OvisOCR2-LiteRT)
(1.30 GB). [ATH-MaaS/OvisOCR2](https://huggingface.co/ATH-MaaS/OvisOCR2) is a page-level
document-parsing post-train of Qwen3.5-0.8B, and its `config.json`, `chat_template.jinja`,
`tokenizer_config.json` and `preprocessor_config.json` are **byte-identical** to the base
checkpoint's — only the weights differ. So the 0.8B VL commands apply with the model id
swapped:

```bash
cd qwen35vl_work
MODEL=ATH-MaaS/OvisOCR2 IMG=512 python convert_qwen35_vision.py out/ovisocr2-vision
# OCR is the stricter consumer: A/B the vision quant on DOCUMENT fixtures, not photos
MODEL=ATH-MaaS/OvisOCR2 FIXTURES=<dir> FIXTURE_FILES=table_512.png,formula_512.png,arxiv_512.png \
  python vision_quant_ab.py out/ovisocr2-vision

PREFILL=1024,256,64,16,4,1 CACHE=4096 \
  PYTHONPATH=../qwen35_work/litert-torch-qwen35 \
  python export_qwen35vl_decoder.py ATH-MaaS/OvisOCR2 out/ovisocr2-decoder

DEC=out/ovisocr2-decoder VIS=out/ovisocr2-vision TOK=out/ovisocr2-decoder/tokenizer.json \
  VENC=vision_encoder_fp16.tflite VADP=vision_adapter_int8.tflite DEC_ACT=fp32 \
  OUT_DIR=out/ovisocr2-bundle OUT_NAME=OvisOCR2_int8.litertlm python build_qwen35vl_bundle.py
python ../scripts/add_executor_metadata.py out/ovisocr2-bundle/OvisOCR2_int8.litertlm OvisOCR2_int8_final.litertlm
```

What is checkpoint-specific and had to be re-measured rather than inherited:

- **The fp16-safe LayerNorm table.** Same shape as base 0.8B (16 from block 6, 512 at the
  final norm) but the absolute magnitudes moved (LN absmax 2390.9 → 2803.9). The script
  calibrates per run, so nothing is edited — but a hand-copied sibling table would have been
  silently wrong, again.
- **The vision-quant A/B on document fixtures.** A rendered table page, a formula page and a
  dense arXiv page at 512×512. Ship shape (fp16 encoder + int8 adapter) holds 0.9998+
  end-to-end correlation on all three; the int8/int8 Mali variant holds 0.994–0.998.
  Photo-fixture numbers must not be quoted for an OCR tower — the fixture set is part of
  the measurement.
- **Stop tokens without a `generation_config.json`.** The upstream repo ships none; the ids
  come from `config.json` (`eos_token_id` 248044) plus the tokenizer's `<|im_end|>` (248046),
  and the bundle builder verifies both against the actual tokenizer before baking them.

Two behaviors that belong to the checkpoint, not the conversion — measure the HF fp32
original before blaming the file:

- **Dense pages exceed 512×512.** On a 9-pt arXiv page the transcription starts correct,
  then paraphrases and finally repeats; upstream's own reference inference code ships a
  repeat-cleanup post-processor for this. On pages legible at 512² (the table and formula
  fixtures) the transcription is exact — all 20 table cells, symbol-for-symbol LaTeX — and
  the GPU backend's output is byte-identical to CPU's.
- **General chat is eroded by the OCR post-training.** On a generic 8-question sanity gate
  the model transcribes the prompt instead of answering; the fp32 original does the same.
  It is a document parser; prompt it as one.

## Falcon-H1 (attention ∥ Mamba2 in parallel, every layer) — first fully-hybrid family in LiteRT form

`falcon_h1_work/convert_falcon_h1.py` converts TII's Falcon-H1 Instruct models to `.litertlm`. Every layer (36 on the 0.5B, 24 on the 1.5B, 32 on the 3B, 66 on the 1.5B-Deep) runs a grouped-query attention branch and a Mamba2 selective-scan branch **in parallel** on the same normalized input and sums them — so every layer carries a KV cache *and* conv/SSM recurrent state. Published: [litert-community/Falcon-H1-0.5B-Instruct](https://huggingface.co/litert-community/Falcon-H1-0.5B-Instruct), [litert-community/Falcon-H1-1.5B-Instruct](https://huggingface.co/litert-community/Falcon-H1-1.5B-Instruct), [litert-community/Falcon-H1-3B-Instruct](https://huggingface.co/litert-community/Falcon-H1-3B-Instruct) and [litert-community/Falcon-H1-1.5B-Deep-Instruct](https://huggingface.co/litert-community/Falcon-H1-1.5B-Deep-Instruct). **Requires litert-lm ≥ 0.15 to run.**

```bash
cd falcon_h1_work
git clone https://github.com/google-ai-edge/litert-torch litert-torch-falcon
git -C litert-torch-falcon fetch origin 115a13607c730c81018bb9789138a3e5e5119e3d   # base unreachable from main since 2026-08
git -C litert-torch-falcon checkout 115a13607c730c81018bb9789138a3e5e5119e3d
git -C litert-torch-falcon apply "$(pwd)/falcon_h1_litert_torch.patch"
PYTHONPATH=litert-torch-falcon python convert_falcon_h1.py tiiuae/Falcon-H1-0.5B-Instruct out_falcon_05b
# GPU ship shape: declare fp32 activations in the bundle TOML (repack, no re-export)
python ../scripts/set_activation_type.py out_falcon_05b/Falcon-H1-0.5B-Instruct_int8.litertlm Falcon-H1-0.5B-Instruct_int8.litertlm --type fp32
```

The patch shares the granite/Qwen3.5 hybrid machinery — the folded rank ≤ 4 scan, state-continuation tracing, and the prefill-pad guard ride verbatim (the scan body is byte-identical to granite's, asserted at patch time). What Falcon-H1 needed on top:

- **A composite hybrid cache layer.** KV and conv/SSM state live at the SAME layer index, flattened as `k_i/v_i/mc_i/mr_i`. The class inherits the attention cache layer first (mask/timestamp semantics) and transformers' `LinearAttentionCacheLayerMixin` second — without the mixin, `Cache.has_previous_state(i)` refuses the layer as "an Attention layer". The runtime binds states by tensor name, so co-residency is just packaging.
- **Re-inject the exporter's kwargs.** `FalconH1Model.forward`'s layer loop drops `**kwargs`, so the timestamp indices the export cache needs never reach attention. The patched model pops them and re-injects them at each attention call.
- **`mamba_d_ssm` is not `expand × hidden`.** `FalconH1Config` defaults the SSM intermediate to its own width (and `mamba_d_head` resolves from `'auto'` against it) — shape inference must read the config, not assume the granite/bamba relation.
- **`mup_vector` is a non-persistent model-level buffer** — a prime candidate for transformers 5.x meta-load zeroing (and a silently zeroed buffer would still "pass" parity when reference and export share the model object). The patch zero-checks it after load; the check cannot live in `forward()` because fake-tensor tracing cannot concretize `float(buffer.max())`.

Ship verification (0.5B): float parity vs HF teacher-forced across 8 decode positions — max|logit diff| 7.6e-05, correlation 1.000000, top-1/top-5 identical; 8-question gate **int8 = float = GPU** (all 6/8 with near-verbatim identical text; the two misses are the 0.5B's own level — the float graph misses them the same way); hermetic prompt-length sweep 41/41 clean; Pixel 8a (OpenCL) delegates every subgraph with zero rejections and correct output; iPhone 17 Pro (Metal) decode 52.7 tok/s vs 36.2 CPU. Mac M4 Max: GPU 2650 prefill / 127.5 decode tok/s, CPU 473 / 59.0 (`-p 256 -d 256 --runs 3 --cache no`, litert-lm 0.16.0).

The 1.5B rides the same driver unchanged: parity max|logit diff| 1.6e-04 / correlation 1.000000 / top-1+top-5 identical; 8-question gate CPU 7/8 (the miss is answered identically by the HF bf16 reference) and GPU 6/8 (one borderline int8 greedy flip); hermetic sweep clean; on iPhone 17 Pro the composite quality probe answers 8/8 on BOTH backends with identical text — Metal decode 25.8 tok/s vs 15.5 CPU, TTFT 0.88 s vs 3.16 s. Mac M4 Max: GPU 1447 / 102.9, CPU 238 / 33.7.

The 3B rides the same driver unchanged too — one command, zero code changes (32 all-hybrid layers → 128 state buffers, full 1–1024 prefill ladder, int8 3.15 GiB). Its fp32 export exceeds the TFLite Python Interpreter's flatbuffer cap, so parity runs split: teacher-forced HF fp32 logits in the torch venv, a CompiledModel decode-walk over the same ids in the litert venv (`scripts/parity_logits_bigmodel.py`) — 48 positions, top-1/top-5 100%, Pearson 1.0000, KL ≈ 0. The 3B is the first Falcon-H1 size where int8 drops nothing: the 8-question gate is **8/8 on every lane** (GPU/CPU × litert-lm 0.15.0/0.16.0), hermetic sweep 40/40 CPU + 20/20 GPU, and on iPhone 17 Pro the composite probe answers 8/8 on both backends — Metal decode 14.0 tok/s / prefill 111.5 / TTFT 1.49 s / 3.03 GB peak (CPU: 7.8 / 48.7 / 3.14 s / 1.46 GB); the full 12-signature ladder loads on a 12 GB phone with no memory pressure (falcon's 65k vocab keeps the per-signature RAM tax light — a 248k-vocab 4B needed its ladder cut to 7 signatures). Mac M4 Max: GPU 979 / 65.3, CPU 121 / 20.9. One family-normal log line to expect on GPU: `TopK requires src tensor C dimension to be divisible by 4` (vocab 65537) — generation is unaffected on every size.

The 1.5B-Deep variant takes the same one command as well. It is the deepest shape in the family — 66 all-hybrid layers, so the bundle packages **264 state buffers** (132 conv/SSM + 132 KV, the most on this rail) and quantizes to 1.83 GB int8 from a 5.91 GiB float graph. Same gates, same answers: 48-position parity top-1/top-5 100% with Pearson 1.0000 and KL ≈ 0, 8-question gate 8/8 on all four lanes (GPU/CPU × 0.15.0/0.16.0), hermetic sweep 40/40 CPU + 20/20 GPU. Its bench shows the depth trade running both ways on an M4 Max: prefill **1134 tok/s — above the 3B's 979 —** and decode **51.7, below its 65.3**, because decode walks 66 sequential attention+scan blocks per token while prefill batches over the sequence. If you are timing a conversion of your own, read the stage breakdown and not the wall clock: of this one's 88 minutes, 69 were the HF download and 19 were the graph export — a *faster* export than the 3B's ~26 minutes despite twice the layers.

On an iPhone 17 Pro the Deep answers the composite probe **8/8 on GPU and 8/8 on CPU**, word for word identical apart from one function word: Metal decode 16.5 tok/s / prefill 140.3 / TTFT 1.19 s / 4.89 GB peak (CPU: 9.9 / 103.2 / 1.53 s / 1.74 GB). All 264 state buffers and the full ladder load with room to spare — depth costs nothing on the device side. Budget about a minute for GPU engine creation (8 s on CPU): every prefill signature's Metal kernels are built up front.

### 2026-09-01 — Falcon-H1-Tiny-R-0.6B: the family's first reasoning ship, and the size where two family assumptions break

[litert-community/Falcon-H1-Tiny-R-0.6B](https://huggingface.co/litert-community/Falcon-H1-Tiny-R-0.6B) — 44 all-hybrid layers (176 state buffers), self-emitting `<think>…</think>` reasoner, **requires litert-lm ≥ 0.16**. It does NOT ride `convert_falcon_h1.py` unchanged; `falcon_h1_work/convert_falcon_tinyr.py` carries four measures the Instruct sizes never needed, each one measured rather than assumed (the script's docstring has the details):

```bash
cd falcon_h1_work   # same pinned litert-torch base + falcon_h1_litert_torch.patch as above
hf download tiiuae/Falcon-H1-Tiny-R-0.6B --local-dir src/Falcon-H1-Tiny-R-0.6B
PYTHONPATH=litert-torch-falcon python convert_falcon_tinyr.py src/Falcon-H1-Tiny-R-0.6B out_tinyr
```

1. **The bundle stream must be byte-identical to HF, BOS included.** The upstream template renders `{{bos_token}}`, which the engine's template renderer leaves empty; the builder's start_token puts BOS in a different spot than HF's render. At 0.6B that one-token head difference flips greedy answers (bf16 A/B: 8/8 vs 7/8 — "capital of Japan" became *Hiroshima*). The script hardcodes the literal `<|begin_of_text|>` into the template and suppresses start_token.
2. **int8 stops at the FC boundary — the embedding table stays float.** The family rule "int8 on linears + embedding" fails here: an int8 embedding table (even per-row) gives runaway thinking and greedy fact flips; FC-only int8 restores bf16-class behavior (a float lm_head does NOT — the input embedding is the poison).
3. **The embedder is externalized into its own CPU-side section**, because the GPU delegate accepts an in-graph EMBEDDING_LOOKUP only with an int8 table (a float table crashes engine creation, an fp16 cast is refused as partial delegation) — quality and GPU delegation cannot coexist in one graph.
4. **The pad guard now derives its mask from position monotonicity** (patch updated in place): the externalized decoder graph carries no token ids, so the previous `input_ids != 0` mask silently disabled and CPU pad garbage corrupted the Mamba conv/cumsum state (`17+25 → 19.5`, float export identical — not a quantization bug).

The reasoning is packaged, not just tolerated: the `thought` channel (`<think>`/`</think>`) is declared post-hoc with `tools/add_thought_channel.py` (verified: reasoning arrives on the channel on Mac CPU/GPU and iPhone, and `--thinking-budget 16` cuts at exactly 16 tokens), and both upstream stop tokens `[11, 228]` ride in from `generation_config`. Give sessions a ≥ 2048-token output budget — truncated mid-thought a reasoner produces no final answer.

Ship verification: **GSM8K 100-question greedy at 2048 tokens, scored after `</think>`: the int8 bundle 76/100 vs the bf16 PyTorch model 71/100** on the identical stream and protocol (per-question flips run in both directions — quantization noise, not degradation). 8-question gate: Mac CPU 6/8 / GPU 7/8, iPhone 17 Pro CPU 6/8 / Metal 6/8, zero unclosed thinks; the recurring misses are two fact-recall knife-edges the bf16 model also flips under one-token perturbations. Mac M4 Max (`-p 256 -d 256 --runs 3 --cache no`, litert-lm 0.16.0): GPU 2216 prefill / 97.8 decode / TTFT 0.13 s, CPU 381 / 47.8 / 0.69 s. iPhone 17 Pro (cold, single runs, 148-token composite prompt): **CPU decodes faster than Metal at this size** — 29.4 vs 22.8 tok/s, at 0.82 vs 3.01 GB peak — the GPU-overhead side of the small-model law, opposite to every larger sibling.

### 2026-08-27 — Qwen3.5-4B Mixed INT4: the block size is the only knob that matters

Requested in [LiteRT-LM #1658](https://github.com/google-ai-edge/LiteRT-LM/issues/1658) for 8 GB Android phones: the int8 file is 4.10 GB and cannot fit; a Mixed INT4 build is 2.57 GB. Two commands on top of the recipe above (same pinned base + patch, same float export):

```bash
QWEN35_PREFILL_LADDER=1024,256,64,16,4,1 PYTHONPATH=litert-torch-qwen35 \
  python convert_qwen35_hybrid.py Qwen/Qwen3.5-4B out_qwen35_4b   # float export (reuse if you have it)
python make_int4_4b.py out_qwen35_4b/model.litertlm out_qwen35_4b b32   # -> Qwen3.5-4B_mixed_int4_b32.litertlm
```

`make_int4_4b.py` (this directory) applies `minicpm5_work/quantize_minicpm5.py` post-hoc — int4 **blockwise-32 min-max** on every FULLY_CONNECTED, overridden to int8 channelwise on the lm_head and embedding lookup — then the same template/stop-token/ExecutorMetadata/fp32act steps as the int8 ship. Two things worth copying from how this was chosen:

- **Pick the block size by measurement, not by family habit.** All four candidates (b32/b128 × min-max/OCTAV) were built from the *same* float export and scored on GSM8K n=100, greedy, 2048-token budget, identical harness: int8 control 97, **b32 min-max 93**, b32 OCTAV 92, b128 min-max 90, b128 OCTAV 90. OCTAV bought nothing here; blockwise-128 (the faster-decode habit from dense 4B ships) costs 3 more points on this hybrid. And on today's runtime the iPhone decode gap between b32 and b128 has vanished (11.4 vs 11.9 tok/s Metal) — the historical reason to prefer b128 no longer holds on this rail.
- **Budget ≥2048 tokens when you score a Qwen3.5 on GSM8K, even with thinking disabled.** At 512 the model's verbose step-by-step runs past the budget before the `#### N` line and the extractor grabs stray numbers — it reads as a quantization collapse and is actually truncation (measured: the same questions come back correct at 2048).

Ship verification (all on the shipped file): 8-question gate **8/8 on CPU and GPU on both litert-lm 0.15.0 and 0.16.0**; prompt-length robustness, fresh engine per length, 40/40 CPU + 20/20 GPU; iPhone 17 Pro answers the composite 8-question probe **8/8 on both Metal and CPU** (the int8 file's CPU path answers 6 of 8 on this probe) — Metal decode 11.4 tok/s / prefill 88.7 / TTFT 1.89 s / 5.74 GB peak, CPU 8.9 / 46.7 / 3.12 s / 1.68 GB. Mac M4 Max (`-p 256 -d 256 --runs 3 --cache no`, 0.16.0): GPU 669 / 68.5 tok/s, CPU 100 / 20.1. Honest limits: the GSM8K gap vs int8 is real (93 vs 97); Mac CPU prefill is ~2.4× slower than int8's (int4 unpack cost — decode is equal or faster everywhere); and we have not yet run it on an actual 8 GB Android phone — the 2.57 GB file + ~1.7 GB CPU working set is the sizing evidence, measured on iPhone.

### Falcon-H1 finetune intake — one command, and a gate refusal worth reading

`scripts/convert.py` routes `model_type: falcon_h1` to this recipe the same way as granite
(HYBRID_RECIPE; no start_token drop — the shipped Falcon-H1 bundles keep it and gate clean at
every size). Measured 2026-08-25 on
[megabytes/Falcon-H1-0.5B-Instruct-heretic](https://huggingface.co/megabytes/Falcon-H1-0.5B-Instruct-heretic)
(Heretic v1.1.0 abliteration of the 0.5B, the most-downloaded real Falcon-H1 derivative):

```bash
python scripts/convert.py megabytes/Falcon-H1-0.5B-Instruct-heretic
```

One command from a clean shell: recipe export 796 s → 650 MB int8, 144 state buffers
(72 conv/SSM + 72 KV) → generic 8-question gate on CPU → **5/8, verdict "do NOT publish",
exit 1**. That verdict is the gate working, not the conversion failing — the misses decompose,
measured:

- The graph is faithful: on the divergence probes, the float export ≡ HF fp32 greedy once both
  sides use the bundle's BOS convention (the metadata start_token `<|begin_of_text|>`,
  prepended on the HF side too): `17+25 → "42"`, `thank-you-in-French → "Thank you."` —
  token-identical.
- Two misses are the abliterated model's own answers (`8×7 → "28."` verbatim on HF fp32; the
  French miss is model-level under either BOS convention — abliteration's measured KL 0.0954
  vs base costs real capability at 0.5B scale).
- One miss is an int8 greedy flip (`17+25`: int8 `"22."` vs float/HF `"42"`) — the base 0.5B's
  int8 did not flip this, so the ablated weights quantize worse.

Family fact for derivative templates: the Heretic tool re-serializes `chat_template.jinja`
with CRLF line endings (1167 bytes vs the base's 1151) — content-identical after
universal-newline normalization, which is exactly the form the exporter embeds (bundle
template = 1151 bytes = base's). Compare templates modulo newline convention, or you will
flag re-serialized derivatives as template forks. `tokenizer.json` is byte-equal to the base.

## Zamba2 (Mamba2 backbone + a shared, LoRA-specialized transformer block) — and the metaspace tokenizer trap

`zamba2_work/convert_zamba2.py` converts Zyphra's Zamba2 instruct models to `.litertlm`. The 1.2B is 38 layers: 32 Mamba2 selective-scan layers plus 6 `hybrid` positions where ONE shared transformer block — a single set of attention+MLP weights tied across all six positions, specialized by per-position LoRA adapters (rank 128) — attends over `concat(hidden, original_embedding)` and projects into that position's mamba input. Published: [litert-community/Zamba2-1.2B-instruct](https://huggingface.co/litert-community/Zamba2-1.2B-instruct) and [litert-community/Zamba2-2.7B-instruct](https://huggingface.co/litert-community/Zamba2-2.7B-instruct). **Requires litert-lm ≥ 0.15 to run.**

```bash
cd zamba2_work
git clone https://github.com/google-ai-edge/litert-torch litert-torch-zamba2
git -C litert-torch-zamba2 fetch origin 115a13607c730c81018bb9789138a3e5e5119e3d   # base unreachable from main since 2026-08
git -C litert-torch-zamba2 checkout 115a13607c730c81018bb9789138a3e5e5119e3d
git -C litert-torch-zamba2 apply "$(pwd)/zamba2_litert_torch.patch"
PYTHONPATH=litert-torch-zamba2 python convert_zamba2.py Zyphra/Zamba2-1.2B-instruct out_zamba2_12b
# GPU ship shape: declare fp32 activations in the bundle TOML (repack, no re-export)
python ../scripts/set_activation_type.py out_zamba2_12b/Zamba2-1.2B-instruct_int8.litertlm Zamba2-1.2B-instruct_int8.litertlm --type fp32
```

The scan needed nothing new: Zamba2's reference scan is the same Mamba2 SSD math as granite's in a different spelling, its wrapper is NemotronH's shape (min-only `time_step_min` clamp — the pad-decay floor trap — and the same width-0 `d_mlp` split), and every mixer attribute matches NemotronH's names, so the patch reuses the NemotronH folded rank ≤ 4 forward verbatim and asserts the reuse contract on the upstream source at patch time (layer checker: `zamba2_folded_cache_check.py`, five cache paths, worst 2.26e-04). The cache side adds Zamba2's third config vocabulary for the same mixer geometry (`n_mamba_heads` / `mamba_headdim` / `mamba_ngroups`) to the shape inference, routes the 6 shared-attention positions to the composite hybrid cache layer (KV + conv/SSM state at one index, as Falcon-H1), and the shared block itself traces cleanly: tied weights are stored once, and each position's LoRA adapter is selected by a Python int at trace time. What Zamba2 DID need was new:

- **The metaspace tokenizer trap.** Zamba2 is a Llama/Mistral SP-BPE (metaspace `▁`) tokenizer serialized as `tokenizer.json` — and the runtime executes the HF decoder pipeline **per streamed token**, so the trailing `Strip(" ", start=1)` decoder (meant to trim the ONE artificial leading space the `Prepend("▁")` normalizer adds to a whole sequence) eats the leading space of EVERY piece. Every word starts with `▁` in SP-BPE, so generations come out with all interior spaces missing ("Thesumof17and25is42.") — content correct, formatting destroyed, quality gates failing on word-boundary checks with zero degeneration. Byte-level-BPE models (Falcon/Qwen/Llama-3-style `Ġ` pieces) never hit this: their spaces live inside the piece bytes and their decoders carry no Strip. Fix: drop the Strip decoder from the bundled tokenizer (`fix_tokenizer_strip.py`, wired into the converter) — interior spaces are exact under per-token decode, and the one behavior change is a sequence-initial space the runtime already trims.

Ship verification (1.2B, on the shipped file or its float parent): teacher-forced parity vs HF fp32 across 48 positions (split harness — the fp32 export is 4.98 GB) — top-1/top-5 100%, Pearson 1.0000, KL ≈ 0; 8-question gate **8/8 on every lane** (GPU/CPU × litert-lm 0.15.0/0.16.0); hermetic prompt-length sweep 40/40 CPU + 20/20 GPU; iPhone 17 Pro composite probe **8/8 on CPU**, 6/8 on GPU where both misses (8×7, the rhyme) are questions the **HF fp32 reference itself answers incorrectly** on this composite — the GPU path tracks the reference's own behavior, deterministically and with identical text on Mac and iPhone Metal. iPhone 17 Pro: Metal decode 12.8 tok/s / prefill 96.6 / TTFT 1.63 s / 5.43 GB peak (CPU: 7.4 / 87.3 / 1.73 s / 1.82 GB) — the full 12-signature ladder loads on a 12 GB phone with no memory pressure; the GPU peak includes the six shared-attention positions' wide KV (32 KV heads × 128 head dim at 4096 context) in fp32. Mac M4 Max (`-p 256 -d 256 --runs 3 --cache no`, 0.16.0): GPU 1033 / 74.0 tok/s, CPU 450 / 22.7.

The **2.7B** is the same driver with two differences worth knowing before you run it. First, it is a *two-block* Zamba2 — 54 Mamba2 layers with TWO shared transformer blocks applied alternately at 9 interleaved positions — and **current `transformers` (5.14.x, and main as of 2026-08-14) cannot construct any two-block Zamba2 at all**: `Zamba2Model.get_layers` assigns `block_id` by global layer index while the checkpoint layout and the model's own weight-tie cycle follow hybrid *occurrence* order, so construction raises before weights are read. The patch carries a structure fix (`block_id` by hybrid order, with an anchor assertion), verified key-exact against the published checkpoint; the parity leg drives the reference through `parity_pt_structfix.py` for the same reason. Second, its prefill ladder is cut to six signatures — `ZAMBA2_PREFILL_LADDER=1024,256,64,16,4,1 python convert_zamba2.py Zyphra/Zamba2-2.7B-instruct out_27b` — because a signature costs memory whether or not it is called.

Ship verification (2.7B): parity 48 positions top-1/top-5 100%, Pearson 1.0000, KL ≈ 0; 8-question gate 8/8 on all four lanes; hermetic sweep 40/40 CPU + 20/20 GPU; Mac M4 Max GPU 603 / 43.6, CPU 282 / 14.7. On an iPhone 17 Pro the composite probe is **8/8 on CPU** (prefill 14.3 tok/s, decode 3.6, TTFT 10.2 s, 2.19 GB peak) — and **Metal does not load this size on a 12 GB phone**: engine creation is killed by the OS, deterministically, twice, warm cache included.

That last result is worth a measurement habit, because it is cheap and it saves a conversion. **The Metal path resides the weights in fp32, so phone GPU peak memory runs at a multiple of the int8 file** — 3.98× on the 1.2B (1364 MB file → 5432 MB peak; its CPU run is 1.34×), and 2.80× on Falcon-H1-1.5B-Deep. The multiple is architecture-dependent — Zamba2's wide shared-attention KV inflates its own — so measure it on a sibling that fits, and budget the pessimistic end. At 2.80 GB the 2.7B projects to ≈ 11 GB against 11.7 GB of phone RAM, which is what happens. The tempting fix — cut the prefill ladder — does nothing here: going from 12 signatures to 6 changed neither the failure nor its timing, because the weight term, not the variant term, is what overflows.

### Zamba2 finetune intake — the pre-port serialization wall (measured refusal)

`scripts/convert.py` routes `model_type: zamba2` through HYBRID_RECIPE like the other pinned
families. But the only real Zamba2 derivative on the Hub today
([ssmits/Zamba2-1.2B-instruct-Dutch](https://huggingface.co/ssmits/Zamba2-1.2B-instruct-Dutch))
is serialized in **Zyphra's pre-transformers-port format**, and that is a wall no converter
can cross, measured 2026-08-25 on transformers 5.14.1:

- Its `config.json` speaks the 4.43-era vocabulary (`layers_block_type: ['m','g',…]`,
  `state_size`, `use_mamba_kernels`, …) — current transformers rejects it at construction
  (strict `validate_layer_type`).
- The key VALUES all pair off exactly with the base's modern re-serialization (state_size ↔
  mamba_d_state 128, lora_rank ↔ adapter_rank 128, …), so we tried the honest translation:
  construct under the base's config and load the derivative's weights. The state_dict is a
  different generation too — `model.mamba_layers.*` layout, **684 missing / 404 unexpected
  keys**. Bridging that layout is the transformers port itself, not a converter's thin layer
  (and the port still has open construction bugs upstream:
  [transformers#47994](https://github.com/huggingface/transformers/issues/47994), which breaks
  every `num_mem_blocks > 1` checkpoint even in the modern format).

`convert.py` therefore refuses pre-port checkpoints at the entry gate, before any download
(`preport_serialization`, exit 2), with the concrete action: the model author re-serializes by
loading with transformers ≤ 4.48 and `save_pretrained` with ≥ 4.49. A future new-format
derivative rides the same pinned recipe that shipped the two litert-community Zamba2 models.


## Nemotron-H (Mamba2 + MLP + attention, three layer kinds) — and the registry trap

`nemotron_h_work/convert_nemotron_h.py` converts NVIDIA's Nemotron-H (the 4B: 24 Mamba2 + 24 plain-MLP + 4 attention layers) to `.litertlm`. Published: [litert-community/Nemotron-H-4B-Instruct-128K](https://huggingface.co/litert-community/Nemotron-H-4B-Instruct-128K). **Requires litert-lm ≥ 0.15 to run.**

```bash
cd nemotron_h_work
git clone https://github.com/google-ai-edge/litert-torch litert-torch-nemotron
git -C litert-torch-nemotron fetch origin 115a13607c730c81018bb9789138a3e5e5119e3d   # base unreachable from main since 2026-08
git -C litert-torch-nemotron checkout 115a13607c730c81018bb9789138a3e5e5119e3d
git -C litert-torch-nemotron apply "$(pwd)/nemotron_h_litert_torch.patch"
PYTHONPATH=litert-torch-nemotron python convert_nemotron_h.py nvidia/Nemotron-H-4B-Instruct-128K out_nemotron_4b
# GPU ship shape: declare fp32 activations in the bundle TOML (repack, no re-export)
python ../scripts/set_activation_type.py out_nemotron_4b/Nemotron-H-4B-Instruct-128K_int8.litertlm Nemotron-H-4B-Instruct-128K_int8.litertlm --type fp32
```

The folded scan is a **port, not reuse**: NemotronH's `torch_forward` is an older SSD spelling, not byte-identical to granite's. The lessons, in the order they cost time:

- **Check how the block CONSTRUCTS its mixer before trusting a class swap.** `NemotronHBlock` picks mixer classes from a module-level `MIXER_TYPES` dict bound at import time, so swapping the module's class attribute exports the *unrewritten reference scan* — with no pad guard, and with every parity gate green (reference math is correct math). **A parity gate can never prove which scan form got traced**; only a pad sweep or the GPU sieve can. The patch swaps the dict entry and adds a loud zero-patched-mixers guard.
- **A min-only dt clamp floors the pads.** NemotronH clamps `softplus(dt + dt_bias)` at `time_step_min` with no upper bound, so a zeroed pad input still decays the recurrent state by `exp(time_step_min · A)` per pad. The pad guard forces pads to exact identity (`dt = 0` applied *after* the clamp).
- **`mlp` = a cache-less layer type.** Mamba/attention layers address the cache by absolute layer index, so MLP layers need a layer object that keeps the index while contributing ZERO tensors to the flatten. The dict fallback would have allocated 24 phantom KV buffers — and signatures pay RAM for every buffer that exists, called or not.
- **Config vocabulary.** NemotronH spells the mamba geometry `mamba_num_heads`/`mamba_head_dim`/`ssm_state_size`/`conv_kernel`/`n_groups`, and the SSM intermediate is `heads × head_dim` — equal to `expand × hidden` on granite/bamba by construction, NOT here (7168 vs 6144 on the 4B).
- **Reference forwards need `use_cache=False`** — transformers' own DynamicCache has no `mlp` layer type and KeyErrors on it.
- **Sweep state-carrying models with a fresh engine per prompt length.** Reusing one engine across sweep conversations rewinds a shared prefix only partially; that corrupts linear-attention state and produces false failures indistinguishable from conversion bugs.

Ship verification (4B): float parity vs HF teacher-forced across 8 decode positions — max|logit diff| 5.8e-05, correlation 1.000000, top-1/top-5 identical; 8-question gate **8/8 on CPU and GPU**; hermetic sweep clean at the ship shape; iPhone 17 Pro (Metal) runs with GPU == CPU answers (peak 4.02 GB, increased-memory entitlement). Honest limit: a 4B does not fit an 8 GB Android phone — on a Pixel 8a, engine creation aborts on both backends. Mac M4 Max: GPU 724 prefill / 75.0 decode tok/s, CPU 99 / 20.1.

Derivative intake: `scripts/convert.py` routes `model_type: nemotron_h` through HYBRID_RECIPE
like the other pinned families. The 4B above still has no real derivative, but the family
gained a **new base** in 2026 — `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` (3.4M downloads,
19 finetunes + 14 adapters, the most-downloaded ≤4B model released in 2026 by a wide margin)
— and it rides the same recipe unchanged. Measured 2026-08-26, one command, no new patch.
Published: [litert-community/Nemotron-3-Nano-4B](https://huggingface.co/litert-community/Nemotron-3-Nano-4B)
(2026-08-27; to our knowledge the first Nemotron-3-Nano in LiteRT form — the Hub otherwise
carries only GGUF conversions).

```bash
python scripts/convert.py nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16
# ship step (as for the 4B above): declare fp32 activations so the GPU executor keeps range
python scripts/set_activation_type.py out/.../NVIDIA-Nemotron-3-Nano-4B-BF16_int8.litertlm \
    Nemotron-3-Nano-4B_int8.litertlm --type fp32
```

export 1645 s → 3.84 GiB int8 (4,126,697,184 B), 50 state buffers (42 mamba conv+SSM,
8 KV; 21 mamba + 17 MLP + 4 GQA layers). Gates on the **shipped file**: **CPU 7/8 PASS**
(the one miss is the rhyme item — "violets are purple"), **GPU 8/8** with `--cache no`.
M4 Max `benchmark -p 256 -d 256 --runs 3`: GPU 696 prefill / 74.5 decode tok/s,
CPU 395 / 28.4.

**GPU on this bundle needs `--cache no`** — one-variable measurement, same runner and file
with only the cache flag flipped: with the compiled-graph cache, `litert-lm run --backend
gpu` dies on WebGPU `Invalid BindGroup` validation errors and the 8-question sweep through
`litert-mac-verify` returns token soup (0/8); with `--cache no` the same file answers 8/8.
The cache path is where it goes wrong; the root cause is not isolated further. Note the
exit gate calls `litert-mac-verify`, which has no cache flag — so a routed `convert.py`
run gates this family on CPU and a green CPU gate says nothing about GPU here. The sibling
4B's card documents the same `--cache no` invocation, so treat it as a family-level caveat
rather than a property of this checkpoint.

Chat template embedded
**byte-equal** to the repo's `chat_template.jinja` (10,504/10,504 — note the repo's
`tokenizer_config.json` copy is a *different* 10,497-byte string; the exporter embeds the
one `AutoTokenizer` actually resolves). Reasoning model: ChatML turns, `<think>`,
`eos = <|im_end|>` (id 11) added by the stop-token guard on top of the exported `ids: 2`.

Three things this intake fixed or established, each measured:

- **`auto_map` is not proof of remote code.** The repo declares `auto_map`, and the entry
  gate refused it outright (exit 2) — wrongly. transformers 5.14.1 registers `nemotron_h`
  natively, so without `trust_remote_code` the library class loads and the repo's Python is
  never imported (`AutoConfig.from_pretrained` returns transformers' own `NemotronHConfig`;
  the export ran with `trust_remote_code: False`). The gate now refuses only when `auto_map`
  is present **and** the model_type is absent from `CONFIG_MAPPING` — verified in both
  directions: this model converts, and `kakaocorp/kanana-2-1.3b-instruct` (`kanana2_tiny`,
  unregistered) still refuses with the same structured reason.
- **The ≥3B reduced prefill ladder now reaches the recipe path too.** It was wired only into
  the stock export, so a 4B hybrid — exactly the shape whose signature-count RAM law is
  documented above — was still getting the full 11-signature ladder. `convert_via_recipe`
  passes `PREFILL_LENGTHS` to the driver, and all four drivers read it (Zamba2 keeps its
  older `ZAMBA2_PREFILL_LADDER` name as an alias). This export used 6 prefill + decode.
- **nemotron_h was the one recipe family with a spurious start token and no guard.** Its
  tokenizer says `add_bos_token: False` and its template opens with `<|im_start|>`
  (`<SPECIAL_10>` on the 4B), yet the bundle's `LlmMetadata` carries `start_token: "<s>"` —
  **including the published 4B bundle**, checked directly. Running the stock path's generic
  guard over all four families reproduces every hand-made choice here: it fires on granite
  (which the family flag already dropped) and on nemotron_h, and stays silent on Falcon-H1
  and Zamba2, whose templates render their own BOS. So the guard is now wired into the
  recipe path instead of a fifth per-family flag. Honest limit on the impact: at 4B this
  model is **robust** to the mismatch — the gate scores 7/8 with the start token and 7/8
  without it (same single miss, a rhyme judgment), and HF greedy is byte-identical on 2 of
  3 probes, diverging mid-thought on the third but reaching the same answer. The fix aligns
  the runtime stream with the training stream; it does not rescue a broken model. The
  granite-350m lesson is why it still matters — that mismatch bites at small scale.

Trap already recorded above and confirmed again here: HF reference forwards for this family
need `use_cache=False` (transformers' generic `DynamicCache` KeyErrors on the `mlp` layer
type), which is what the greedy A/B above ran with.

Coverage boundary, stated rather than implied: **RWKV-7 is the one shipped LLM family with
no derivative intake here.** The shipped 0.1B-world size has zero Hub derivatives (measured
2026-08-25; the rwkv7-g1 line's ~27 "finetunes" are almost all the author's own size
re-serializations), the architecture is not transformers-native, and its conversion never
rode this repo's rails — so there is nothing to route until real derivative demand appears.

### 2026-08-27 — Hy-MT2-1.8B intake: one config bake closes the sweep's real gap, and the engine's start_token prepend gets proven

The sweep below left one in-demand family untried with no blocker found:
`tencent/Hy-MT2-1.8B` (`hunyuan_v1_dense`, 226k downloads, 1.2k likes — a 33-language
translation model). It now converts with the standard one command:

```bash
python scripts/convert.py tencent/Hy-MT2-1.8B
```

`converted_pass`: export 120 s → 1.69 GiB int8 (1,815,622,960 B, 2.04B params), gate
6/8 non-degenerate (~50 tok/s decode on the verifier's default backend), template
byte-equal 654/654 (this repo carries only `chat_template.jinja` — no
tokenizer_config copy to disagree with).

Published: [litert-community/Hy-MT2-1.8B](https://huggingface.co/litert-community/Hy-MT2-1.8B)
(2026-08-27; to our knowledge the first Hy-MT2 in LiteRT form — the Hub otherwise carries
GGUF, MLX and FP8 conversions). Gates on the shipped file: 6/8 on CPU **and** GPU with the
same two misses, so the drops are a property of this translation-tuned 1.8B, not a backend.
M4 Max `benchmark -p 256 -d 256 --runs 3 --cache no`: GPU 2008 prefill / 105.8 decode
tok/s (TTFT 0.14 s, after a ≥300 s rest), CPU 211 / 34.0 (TTFT 1.24 s); both backends
gated on real generations before quoting. Unlike the Nemotron hybrids, GPU works with the
default compiled-graph cache on this dense bundle.

**Why stock failed, and why the fix is exact.** The stock export dies in
`torch.export` with `GuardOnDataDependentSymNode` at transformers'
`dynamic_frequency_update` (`if seq_len > max_seq_len_cached`). But transformers
5.14.1 resolves this family's rope STATICALLY at init — its hunyuan modeling computes
`base = rope_theta * alpha**(head_dim/(head_dim-2))` once ("DynamicNTKAlphaRotary")
and never rescales below `max_position_embeddings`. Only the dead growth branch is
untraceable. `convert.py` therefore bakes the resolved base into `rope_theta`
(11,158,839.925) and drops `rope_scaling` before export
(`normalize_static_alpha_rope`). Measured equivalence: `inv_freq` bitwise-equal,
teacher-forced logits bitwise-equal (max |diff| 0.0). Faithful for all positions
≤ 262144 — far past the bundle's 4096 context.

**The engine prepends `start_token` unconditionally — proven, not assumed.** This
family's template renders `<|hy_begin_of_sentence|>` itself and its tokenizer adds no
BOS at encode time, so the exported metadata `start_token` looked like a double BOS.
Output-vs-HF comparison could not decide it: int8 greedy divergence coincidentally
matched the extra-BOS reference on 2 of 3 probes even after the drop. The
discriminator that decides runs entirely inside the runtime: `[start_token]+rest`
(original bundle, `--no-template`, rendered text minus its BOS string) generates
byte-identical greedy output to `[template-BOS]+rest` (dropped bundle, normal
templating) on 3 of 3 probes — same id, same stream, so the default export was
feeding BOS twice. This supersedes the 2026-08-25 note that the guard's silence
on BOS-rendering templates reproduced the right choice — that silence kept a
double BOS.

*Corrected 2026-08-27, after auditing all 88 published bundles.* "The template
renders BOS" is not one shape but two, and only one of them duplicates. The
engine renders the template with **minijinja, which leaves `bos_token`
unbound**, so `{{ bos_token }}` renders **empty** on device and only a BOS
written as a **literal** survives. Read straight out of the runtime:
`Conversation.render_message_to_string` returns `<start_of_turn>user…` for a
template whose source begins `{{ bos_token }}`, and the benchmark prefill
counter reads 16 tokens with a `start_token` against 15 without — one token,
always prepended. The engine's own `tokenize()` adds no special tokens either,
so a tokenizer's `add_bos_token` / post-processor never applies on device.

So Hy-MT2 (a literal) really was doubled, but **Falcon-H1 and LFM2.5 write
`{{ bos_token }}` and are faithful** — their `start_token` supplies the single
BOS the model expects, and dropping it would ship a bundle with no BOS at all.
The families that were actually doubled are the ones with a literal BOS at the
head of the template: Zamba2, SmolVLM2, and Nanbeige (via its affixes).

Honest numbers on the fix: the double-BOS bundle scored 8/8 on the gate; the
stream-faithful bundle scores 6/8 (misses: "Cool" for opposite-of-hot, "pink" for
the rhyme — noise-level for a translation-tuned 1.8B, no degeneration). The ship
follows the training stream, not the lucky gate. On the model's actual task,
runtime greedy matches HF bf16 byte-for-byte on 1 of 3 translation probes; the
other two are valid alternates of the int8-vs-bf16 class (cross-checked: the
no-BOS control arm lands on the plain-HF string on one of them).

Trap for the next family: do not diagnose BOS handling by comparing outputs
against HF arms — quantization noise can mimic the wrong arm. Compare token
streams inside the runtime (`--no-template` against normal templating).

### What a 2026 sweep of the phone band actually finds

To keep the coverage claim honest rather than asserted, the ≤4.5B text-generation models
created in 2026 were enumerated mechanically (2026-08-25/26, HfApi, five angles — trending,
downloads, likes, new-and-recent, and a direct sweep of ~70 major model orgs; parameter
counts read from each repo's own safetensors metadata). Union: **835 models across 93
`model_type` values**, of which **589 (70.5%)** route through a path documented above. Of
the rest, most are research checkpoints (`gpt2`, `opensci`, `ceno`, one-off student models).
What is genuinely outside, in demand order, with the reason:

| model | why it is out |
|---|---|
| `sapientinc/HRM-Text-1B` (248k dl) | transformers-native, but hierarchical recurrence (`H_cycles`/`L_cycles`) with `prefix_lm` — not a prefill/decode contract |
| ~~`tencent/Hy-MT2-1.8B`~~ | was listed here as "no blocker found — untried"; converted 2026-08-27, see the intake above |
| `CohereLabs/tiny-aya-*` (55 ft + 55 ad) | the bases are **gated**; the entry gate refuses on principle, not on capability |
| `ai21labs/AI21-Jamba2-3B` (65k dl) | Mamba-**1** hybrid; the four pinned patches are all Mamba2/SSD, so this needs a new port rather than a recipe |
| `kakaocorp/kanana-2-*`, `ibm-granite/granite-swash-*` | genuinely remote-code / not yet registered in transformers 5.14.1 |
| `Zyphra/ZUNA`, `stabilityai/SAME-*` | not transformers-format checkpoints |

Two things this refuted about the previous session's summary: the phone band's gaps are
**not** only MoE and Gemma (Hy-MT2 and Jamba2 were dense-ish, ungated, and in demand;
Hy-MT2 has since been converted — the intake above — and Jamba2 still stands), and
one apparent gap — the most-downloaded 2026 model in the whole band — was this converter's
own false refusal, fixed above. Separately, the MoE boundary is getting load-bearing:
`facebook/MobileMoE-S/M`, `LFM2.5-8B-A1B`, `ibm-granite/granite-switch-4.1-3b`, and
`arcee-ai` afmoe all landed in the phone band in 2026, and the runtime still ships no MoE
kernel.

## Bamba (arch reference — no published artifact)

`bamba_work/bamba_litert_torch.patch` carries the same folded-scan machinery to IBM's Bamba. `BambaMixer.torch_forward` is byte-identical to granite's, so the extension is a thin wrapper that asserts the source identity at patch time. Arch-verified at tiny scale (E2E float parity correlation 1.000000; 5-path cache checker worst 5.5e-07), and the 9B converts and runs — but every published Bamba checkpoint is a 9B BASE model (no chat template), desktop-class, so nothing is shipped. The patch is here as the third data point on how far the granite scan form travels.

## LFM2.5-Encoder (bidirectional encoders → plain .tflite)

`lfm_work/convert_lfm25_encoder.py` converts LiquidAI's LFM2.5-Encoder-350M/230M — multilingual (15-language) bidirectional masked-LM encoders on the same LFM2 hybrid backbone — to plain LiteRT `.tflite` for embeddings / retrieval / fill-mask, fully on CPU. These are NOT `export_hf` runs: the models have no KV cache, so the HF eager model (remote code = stock `Lfm2Model` + bidirectional patches) is traced directly with `litert_torch` multi-signature convert. Signatures: `encode_{64,128,256,512}` → `last_hidden_state` `[1,S,1024]` (padded positions zeroed), plus `mlm_128` → masked-LM logits. Published: [litert-community/LFM2.5-Encoder-350M](https://huggingface.co/litert-community/LFM2.5-Encoder-350M), [litert-community/LFM2.5-Encoder-230M](https://huggingface.co/litert-community/LFM2.5-Encoder-230M).

```bash
cd lfm_work
python convert_lfm25_encoder.py LiquidAI/LFM2.5-Encoder-230M out_lfm25_encoder_230m   # -> fp32 + wi8fc int8 + fp16
python verify_lfm25_encoder.py  LiquidAI/LFM2.5-Encoder-230M out_lfm25_encoder_230m   # 15-language parity report
```

Env: litert-torch ≥ 0.9.2, transformers ≥ 5.12 (the encoders' remote code needs 5.x; tested 5.14.1), torch 2.12. The non-obvious parts:

- **Mobile-GPU delegation needs a rank-4 respelling of transformers' GQA `repeat_kv` (2026-08-13 re-export, whole family).** The stock HF spelling — `x[:, :, None, :, :].expand(...).reshape(...)` — lowers to a rank-5 `BROADCAST_TO` + rank-5 `RESHAPE`, which the Metal/OpenCL/WebGPU delegates refuse; the original exports ran only ~136/828 ops on GPU and were *slower* than CPU there. `lfm_work/_rank4_repeat_kv.py` respells it as an outer product against a ones row through a matmul (bitwise-identical — asserted at import) and every convert script in this family now calls `install()` before `from_pretrained` and `rebind_all()` after (remote code imports symbols by value, so patching transformers by module name can miss the live call site — the same trap as `apply_mask_to_padding_states`). The graph gate is `assert "BROADCAST_TO" not in hist`. Result: the int8 files fully delegate on Pixel 8a OpenCL (852/852 nodes, `encode_512` ~10 ms) and iPhone 17 Pro Metal, with GPU-vs-fp32-reference cosine at the int8 noise floor (≥0.9948).
- **GPU precision must be fp32.** At fp16 GPU precision the norm reductions overflow (|h| reaches ~50) and every output is NaN — set the GPU accelerator's precision to fp32 (`precision = 2` in the LiteRT GPU options); correctness was verified on-device only in that mode.
- **The 5-signature 350M file cannot GPU-compile on phones** — every signature subgraph materializes its own fp32 copy of the weights on the GPU, and 5×350M exceeds phone memory (the 230M fits). The 350M repo therefore also ships `LFM2.5-Encoder-350M_wi8fc_single-sig.tflite` (encode_512 only) as the phone-GPU artifact; multi-signature use stays on CPU. The PII-Detector and Spellchecker int8 files do not GPU-compile at all (a runtime kernel limit on their 161-label / 128802-vocab int8 heads — consistent across Metal and OpenCL); CPU remains their mobile path.
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

## bekko-embedding-v1-a25m (mmBERT Japanese/multilingual embedder — when 80% of the model is the embedding table)

`bekko_work/convert_bekko_embed.py` converts [hotchpotch/bekko-embedding-v1-a25m](https://huggingface.co/hotchpotch/bekko-embedding-v1-a25m) — an ultra-compact multilingual retrieval model (100+ languages, strong Japanese, 384-d, Matryoshka 256/128/64) — to plain LiteRT `.tflite`. Same ModernBERT encoder lane as the granite-embedding-r2 section above (hand-built `{layer_type: bias}` mask dict, sliding-window diagonal fix, isfinite gates — read that section first; it all transfers). Signatures `embed_{64,128,256,512}` → `output_0` `[1,384]`, **already mean-pooled and L2-normalized**, **no prefixes** (the README FAQ is explicit — a third pooling/prefix contract in three embedders shipped from this repo: read `1_Pooling/config.json` per model, never carry the last model's recipe).

```bash
cd bekko_work
python convert_bekko_embed.py hotchpotch/bekko-embedding-v1-a25m out_bekko_a25m  # fp32 + wi8fc + embt8 + fp16
python verify_bekko_embed.py out_bekko_a25m                # card oracle + JSTS/Matryoshka/STS17 + JSQuAD + mechanics
python verify_bekko_embed.py out_bekko_a25m --gates E --report verify_report_e.json  # cross-variant retrieval
python bench_bekko_embed.py  out_bekko_a25m
```

Env: litert-torch ≥ 0.9.2, transformers 5.14.1, torch 2.12. What is new relative to the granite-r2 section:

- **The model is mostly embedding table, so the quantization question inverts.** 123.2M params, 13L/384h mmBERT (pruned from jhu-clsp/mmBERT-small), Gemma-style vocab 256000: the 256000×384 table is 80% of all parameters and only 25M are "active". The upstream repo's own default ONNX/OpenVINO artifacts quantize ONLY that table to row-wise int8 and call transformer-weight int8 "experimental, not recommended". `embt8` reproduces exactly that recipe (embedding int8 channelwise, float compute: 214 MB, per-vector cos ≥ 0.9999 to torch); `wi8fc` additionally puts dynamic-range int8 on the FCs (142 MB).
- **The vendor's "transformer int8 not recommended" verdict does not transfer to DRQ int8 — but you must gate on tasks, not vectors.** wi8fc's per-vector smoke cos on random ids is 0.985–0.992, which reads as damage. On real tasks it is lossless: JSTS (JGLUE v1.3, 300 pairs) Spearman 0.8179 vs torch 0.8175, with Matryoshka truncation to 256/128/64 intact; JSQuAD retrieval (800 paragraphs, 150 questions) at or above the reference; STS17 spot-checks within 0.0025. Per-vector cosine and task score are different quantities, and at 4× less RAM than the vendor-analog recipe the task numbers are the ones that matter.
- **Gate the RAG deployment shape explicitly: index built with upstream weights, queried with the quantized artifact.** Cross-variant retrieval (documents encoded by PyTorch fp32, queries by wi8fc) scores nDCG@10 0.9186 vs the 0.9179 all-PyTorch control — the int8 embedding space is compatible with a server-built index. A variant can pass same-variant gates while sitting in a slightly rotated space; this is the gate that would catch it.
- **fp16 is strictly dominated here — build it, measure it, don't ship it.** XNNPACK expands fp16 weights to fp32 while packing every signature subgraph: 2387 MiB peak for 4 signatures vs embt8's 575 MiB and wi8fc's 225 MiB, at identical task scores (M4 Max, 8 threads, all signatures invoked).
- **Japanese gates**: JGLUE v1.3 (`datasets/{jsts,jsquad}-v1.3` on github.com/yahoojapan/JGLUE — the v1.1 paths in older writeups are gone). The base card's quickstart scores carry a ~0.028 backend residual that the PyTorch fp32 reference itself reproduces, so the oracle is argmax + the documented cross-lingual similarities (which match to 0.003), not bit-distance to the card.
- **Mechanics**: cross-signature outputs bitwise identical on every variant (the 11-token probe in `embed_512` is also the empty-sliding-window NaN shape — finite everywhere, so the diagonal fix holds), pad-content invariance exactly 0.0, zero int64 tensors, no `GATHER_ND`.


## harrier-oss-v1-0.6b (a causal decoder pooled at the last token — the e5-mistral shape, statically)

`harrier_work/convert_harrier_embed.py` converts [microsoft/harrier-oss-v1-0.6b](https://huggingface.co/microsoft/harrier-oss-v1-0.6b) — Microsoft's multilingual (100+ languages) instruction-prompted embedding model, 596M params, 1024-d — to plain LiteRT `.tflite`. Same encoder lane (direct multi-signature trace), but the opposite attention contract to the bidirectional embedders above: the body is a bare `Qwen3Model` and attention stays **causal** — nothing is flipped; the embedding is the LAST non-padding token's hidden state, L2-normalized, both in-graph. Signatures `embed_{64,128,256,512}` → `output_0` `[1,1024]`.

```bash
cd harrier_work
python convert_harrier_embed.py microsoft/harrier-oss-v1-0.6b out_harrier_0p6b   # fp32 + wi8fc + fp16
python -u verify_harrier_embed.py out_harrier_0p6b --prefix-ab                   # README example + JSTS/STS17 + JSQuAD + mechanics
python -u verify_harrier_embed.py out_harrier_0p6b --gates E --report verify_report_e.json
python bench_harrier_embed.py  out_harrier_0p6b
```

Env: litert-torch ≥ 0.9.2, transformers 5.14.1, torch 2.12. The non-obvious parts:

- **`Qwen3Model` accepts the same `{layer_type: bias}` mask dict as ModernBERT** (`causal_mask_mapping := attention_mask` walrus, key `"full_attention"`), so the hand-built causal+padding bias `[1,1,S,S]` (`allowed(q,k) = (k <= q) & valid[k]`) bypasses `sdpa_mask` entirely — no GATHER of any kind in the graph. No NaN guard is needed in the causal direction: every query row can reach position 0.
- **A last-token pool must rebind to the mask, not to position S−1.** With right padding the pooled position is `sum(mask) − 1`. The static-graph form — `onehot = (arange(S) == sum(mask) − 1); pooled = sum(h * onehot)` — lowers to a single `ONE_HOT` + `SUM`, no dynamic gather. Gate both directions: scribbling pad ids must move nothing (0.0 measured), dropping the last REAL token must move the output a lot (0.13 measured). A graph that unconditionally reads position S−1 passes every all-valid smoke test and silently pools pad garbage in production.
- **The tokenizer appends `<|endoftext|>` itself** (a `TemplateProcessing` post-processor: `tok("summit define")` → `[1242, 1763, 6979, 151643]`), and that appended token is what gets pooled — the e5-mistral EOS convention, delivered by the tokenizer while the vendor's own usage snippet never mentions it. Tokenize with the repo's `tokenizer.json` and the contract is automatic; a hand-rolled BPE without the post-processor pools the last *text* token instead, which is a different (untrained) contract. It is also the same id as PAD — the mask is what distinguishes them.
- **The instruction prefix is mandatory and the degradation is now a number.** Queries carry `Instruct: {task}\nQuery: ` (documents raw; prompts ship in `config_sentence_transformers.json`). Measured with `--prefix-ab` on both torch and the int8 artifact: dropping it costs 0.018 JSTS Spearman and **0.028 JSQuAD nDCG@10**. Note the contrast with granite-r2 above, where an *uninvited* prefix helped STS and hurt retrieval — the prefix contract is per-model in sign as well as size; measure it, never assume it.
- **Quality is quantization-flat, and fp32/fp16 are digit-exact.** fp32 and fp16 tflite reproduce the torch reference to every printed digit on JSTS (0.8553), STS17 (en-en 0.8893 / es-en 0.8345 / ko-ko 0.8874 / en-ar 0.8517) and JSQuAD (nDCG@10 0.9363); wi8fc lands within 0.004 everywhere (JSTS 0.8559, nDCG@10 0.9360), and querying a torch-built index with int8 embeddings scores at the all-torch control (0.9389 vs 0.9363) — the RAG shape of index-on-server, query-on-device holds.
- **wi8fc = FC int8 DRQ + EMBEDDING_LOOKUP int8 channelwise.** 596M params → fp32 2390 MB (fine intermediate), int8 626 MB, fp16 1199 MB. Per-vector smoke cos on random ids reads 0.989–0.997 for int8 — the task gates above are the ones that matter.
- **Signature count is NOT always the RAM lever.** Unlike the 1.14B Nemotron build (~846 MiB/signature), the int8 peak here is signature-independent: 2401 MiB with 4 signatures vs 2351 MiB with 2 (Mac, load+invoke). Ship the flexible 4-signature file. fp16 is still the RAM disaster (11.9 GiB, XNNPACK expands fp16→fp32 per subgraph) — desktop only.


## voyage-4-nano (a "causal" config that is actually a bidirectional embedder)

`voyage_work/convert_voyage_embed.py` converts [voyageai/voyage-4-nano](https://huggingface.co/voyageai/voyage-4-nano) — Voyage AI's multilingual retrieval embedder with a shared embedding space across the voyage-4 series (a device-side nano can query an index built by the larger cloud models), 347M params, 2048-d, MRL-truncatable to 1024/512/256 — to plain LiteRT `.tflite`. Signatures `embed_{64,128,256,512}` → `output_0` `[1,2048]`, mean pool + 1024→2048 projection + L2 normalize in-graph.

```bash
cd voyage_work
python convert_voyage_embed.py voyageai/voyage-4-nano out_voyage_nano   # fp32 + wi8fc + fp16
python -u verify_voyage_embed.py out_voyage_nano --prefix-ab            # README example + JSTS/STS17 + JSQuAD + mechanics
python -u verify_voyage_embed.py out_voyage_nano --gates E --report verify_report_e.json
python bench_voyage_embed.py  out_voyage_nano
```

Env: litert-torch ≥ 0.9.2, transformers 5.14.1, torch 2.12. The non-obvious parts:

- **`architectures: ["Qwen3ForCausalLM"]` is a lie.** `auto_map.AutoModel` points at remote-code `Qwen3BidirectionalModel`: a bare `Qwen3Model` body with every `self_attn.is_causal` flipped False plus a per-token bias-free `linear` 1024→2048 (`config.num_labels`) applied to `last_hidden_state` BEFORE pooling. Convert the config's nominal architecture and you ship a causal model that was never trained that way.
- **The remote code targets transformers 4.51 and breaks twice under 5.14.** It sets no `config_class` (AutoModel.register crashes with `AttributeError: 'NoneType' object has no attribute '__name__'`) and calls `create_causal_mask(input_embeds=…, cache_position=…)` — both renamed/removed in 5.x. The converter imports a vendored copy of the modeling file (shipped next to the script), sets `config_class = Qwen3Config`, and shims the mask call. `trust_remote_code` is never used.
- **The export mask bypasses the vmap path entirely.** The remote forward builds `create_causal_mask(..., or_mask_function=bidirectional)` — vmap machinery that specializes on `attention_mask=None` and emits rank-5 intermediates at trace time. Its semantics reduce to `(causal | valid[k]) & valid[k] = valid[k]`, a key-only gate — so the hand-built `[1,1,S,S]` bias into `Qwen3Model`'s `{"full_attention": bias}` dict is exactly equivalent (gated: cos ≥ 0.9999995 vs the remote path, padded and unpadded) and trace-safe. Full attention + ≥1 valid key ⇒ no empty rows ⇒ no NaN guard needed.
- **Bidirectionality is gated, not assumed** — the inverse of the harrier gate above: editing the LAST valid token must move position 0's hidden state (7.9e-02 measured; a causal graph gives exactly 0).
- **Prompts ride on BOTH sides** (`config_sentence_transformers.json`): queries `"Represent the query for retrieving supporting documents: "`, documents `"Represent the document for retrieval: "`. Measured with `--prefix-ab`: on JSQuAD the query prompt is score-neutral (+0.005 nDCG@10 *without* it, torch included) — opposite sign to harrier's +0.028-with. Third data point that the prefix contract is per-model in sign; measure, never assume.
- **Quality is quantization-flat — consistent with the vendor's QAT claim.** fp32/fp16 reproduce torch to every printed digit (JSTS 0.8406, JSQuAD nDCG@10 0.9292); wi8fc lands within 0.008 everywhere and half the deltas are positive. The shared-space deployment shape holds under int8: torch-built index × wi8fc queries = 0.9283 vs 0.9292 control, and at MRL-256 (truncate + renormalize) 0.8989 vs 0.8990.
- **wi8fc = FC int8 DRQ + EMBEDDING_LOOKUP int8 channelwise.** 347M params → fp32 1389 MB, int8 364 MB (1.15 GiB peak RSS, phone-viable), fp16 696 MB but **7.1 GiB peak RSS** (XNNPACK fp16→fp32 per-signature expansion) — desktop only.

## mLateOn (multilingual ModernBERT ColBERT — late interaction on device)

`mlateon_work/convert_mlateon_colbert.py` converts [lightonai/mLateOn](https://huggingface.co/lightonai/mLateOn) — LightOn's multilingual late-interaction retriever (mmBERT-base, 307M params, 128-d per token, MaxSim), the strongest long-document retriever in its class and trained on nine languages yet generalizing to unseen scripts — to plain LiteRT `.tflite`. Signatures `encode_{32,128,256,512}` → `output_0` `[1,S,128]`, per-token L2-normalized; MaxSim scoring is host-side.

```bash
cd mlateon_work
python convert_mlateon_colbert.py lightonai/mLateOn out_mlateon   # fp32 + wi8fc + fp16
python -u verify_mlateon_colbert.py out_mlateon --gates A,D,B,J   # card oracle + mechanics + NanoSciFact + JSQuAD
python -u verify_mlateon_colbert.py out_mlateon --gates E --bank-cache --report verify_report_e.json
python bench_mlateon_colbert.py out_mlateon
```

Env: litert-torch ≥ 0.9.2, transformers 5.14.1, torch 2.12 (+ pyarrow for the NanoSciFact loader). The non-obvious parts:

- **The pad id is 4 (`<mask>`), not `config.json`'s 0.** PyLate sets `pad_token_id = mask_token_id` whenever the tokenizer has a mask token; the vendor's own `onnx_config.json` says 4. With padding masked out of attention the graph output on real tokens is provably pad-id-independent here (gated 0.0) — but a host should still pad with 4 to match the reference stack.
- **The PyLate head is three all-linear Dense modules and folds exactly.** `use_residual: true` means a PARALLEL bias-free `residual` Linear (`out = linear(x) + residual(x)`), so 768→1536(res) → 1536→768(res) → 768→128 collapses to one 128×768 matrix `W3 @ (W2l+W2r) @ (W1l+W1r)` — folded in float64, gated at 2.4e-6 against the module stack. One FC in the graph instead of five.
- **The sliding-window NaN wall, and why int8 hides it.** ModernBERT alternates full and ±64-window attention (`config.sliding_window` = `local_attention // 2` = 64, inclusive). Under right padding, a pad query row ≥64 past the last valid token has an empty window → all-masked row → SDPA NaN in fp32. There is no pooling to poison — but the NaN sits in the `[1,S,128]` output, and int8 kernels can round it into finite-looking garbage, so **assert finiteness on the fp32 artifact** (the variant that cannot fake it). The hand bias allows the diagonal (`valid[k] | q==k`) — a no-op for valid rows that keeps pad rows finite; the host drops them anyway.
- **No query expansion, unlike classic ColBERT.** `do_query_expansion: false` ⇒ queries are natural-length with `[Q] ` (id 256000) inserted at position 1 after `<bos>`; documents get `[D] ` (256001); `skiplist_words` is empty. Every contract knob differs from LFM2.5-ColBERT above (pad id, expansion, pad-id load-bearing-ness, skiplist) — same architecture class, opposite host contracts, each read out of the shipped config + PyLate source, never assumed.
- **The base card publishes exact MaxSim scores — use them as the oracle.** The converted fp32 reference reproduces `[9.6029, 9.5838, 9.5877, 9.4578]` to all four printed decimals, which anchors tokenizer, prefix insertion, padding, masks, head fold, normalization and MaxSim in a single gate before anything is traced.
- **Quality is quantization-flat, including on an unseen language.** fp32/fp16 match torch to every digit; wi8fc: NanoSciFact nDCG@10 0.8951 vs 0.8882 torch, JSQuAD (Japanese — NOT in the nine training languages) 0.9719 vs 0.9710, and torch-index × int8-queries is identical to control on JSQuAD in every metric. Per-token smoke cos on random ids reads as low as 0.974 for int8 — random ids abuse the 256k SP table; the task gates are the ones that matter.
- **wi8fc = FC int8 DRQ + EMBEDDING_LOOKUP int8 channelwise** (the 256002×768 table is ~64% of params). 307M params → fp32 1233 MB, int8 331 MB (678 MiB peak RSS — the smallest-footprint retriever in this repo), fp16 619 MB but ~6 GiB RSS — desktop only. Encoding runs ~4,600 tok/s at `encode_512` int8 on an M4 Max (late interaction pays at scoring time, not encoding time).


## ettin-reranker-400m-v1 (true cross-encoder reranking on device — and a score that must not be sigmoided)

`ettin_work/convert_ettin_reranker.py` converts [cross-encoder/ettin-reranker-400m-v1](https://huggingface.co/cross-encoder/ettin-reranker-400m-v1) — the Sentence Transformers org's ModernBERT cross-encoder (ettin-400m backbone, 396M params, MSE-distilled from mxbai-rerank-large-v2) — to plain LiteRT `.tflite`. Query and passage go through the model **together**; signatures `score_{128,256,512}` → one raw relevance logit per invoke. The first true cross-encoder in this repo (everything else reranks with bi-encoders or decoder LMs).

```bash
cd ettin_work
python convert_ettin_reranker.py cross-encoder/ettin-reranker-400m-v1 out_ettin   # fp32 + wi8fc + fp16
python -u verify_ettin_reranker.py out_ettin      # card oracle + mechanics + NanoSciFact rerank
python bench_ettin_reranker.py out_ettin
```

Env: litert-torch ≥ 0.9.2, transformers 5.14.1, torch 2.12 (+ pyarrow for the NanoSciFact loader). The non-obvious parts:

- **This is NOT `AutoModelForSequenceClassification`.** It is the sentence-transformers v5 *module-stack* CrossEncoder (`modules.json`): Transformer → Pooling(cls) → Dense 1024→1024 no-bias + **GELU** → LayerNorm → Dense 1024→1 → score. The GELU means the head does NOT fold to one matrix (unlike the all-linear PyLate heads) — implement the stack in-graph as-is. Also note the backbone config's `classifier_pooling: "mean"` is a base-model leftover; the ST module config says `cls`, and the ST path is the vendor path.
- **The score is a raw logit — the activation is part of the model config, not the API.** `CrossEncoder.predict` defaults to sigmoid for single-label models, but this repo pins `activation_fn: torch.nn.Identity` in `config_sentence_transformers.json`, so the published card scores (`[3.6875, 11.6875, 4.75, 9.375]`) are raw. A host that assumes the API default gets card-incomparable scores. Read the config, not the docs.
- **The card's example scores were printed from a bfloat16 run** (the usage snippet passes `dtype: bfloat16`; every value is a multiple of 0.0625). fp32 reproduces them to bf16 tolerance only — max |Δ| 0.054, ranking exact. Gate rank order + tolerance, not decimals, when a card example was run in bf16.
- **Pairs are the tokenizer's own pair template**: `tokenizer(query, passage, truncation="longest_first")` → `[CLS] q [SEP] d [SEP]`, no token_type_ids, pad 50283.
- **The sliding-window NaN wall reaches the score through CLS pooling.** ModernBERT alternates full and ±64-window attention; under right padding an empty pad row goes NaN, and the next layer's `0·NaN` value mixing carries it into every row — including row 0 that the head pools. The hand mask allows the diagonal (`valid[k] | q==k`); gate = padded score equals unpadded score (2.9e-6 eager, exactly 0 across signatures on the artifact).
- **Quantization on a 28L/1024h deep-narrow body is measurable**: NanoSciFact rerank (50 queries × 20 candidates) — torch/fp32/fp16 all nDCG@10 0.9637 / hit@1 0.920 with fp16 at zero pairwise inversions; wi8fc 0.9563 / 0.900 with 2.8% of pairs inverted, max raw-score drift 0.84. Both ship; the card carries the table. 396M → fp32 1588 MB, int8 413 MB (1469 MiB peak RSS), fp16 797 MB but ~6.4 GiB RSS — desktop only. Mac 16-thread: 89.5 ms per candidate at score_256 int8.

## mxbai-edge-colbert-v0-32m (a 32M ColBERT at 39 MB — and the strongest card oracle yet)

`mxbai_work/convert_mxbai_colbert.py` converts [mixedbread-ai/mxbai-edge-colbert-v0-32m](https://huggingface.co/mixedbread-ai/mxbai-edge-colbert-v0-32m) — Mixedbread's edge ColBERT (ettin-32m ModernBERT backbone, 64-d per token, PyLate head) — to plain LiteRT `.tflite`. Signatures `encode_{48,128,256,512}` → `[1,S,64]`, per-token L2-normalized; MaxSim host-side. 48 mirrors the vendor's query length.

```bash
cd mxbai_work
python convert_mxbai_colbert.py mixedbread-ai/mxbai-edge-colbert-v0-32m out_mxbai   # fp32 + wi8fc + fp16
python -u verify_mxbai_colbert.py out_mxbai       # card oracle + mechanics + NanoSciFact + cross-variant
python bench_mxbai_colbert.py out_mxbai
```

Env: same as ettin above. The non-obvious parts:

- **The pad id is 50284 (`[MASK]`), not `config.json`'s 50283.** Same PyLate rule as mLateOn (pad = mask when the tokenizer has one); the vendor's own `onnx_config.json` says 50284. Pad-id-independence on real tokens is gated (0.0), but hosts should match the reference stack.
- **Lowercase first.** `do_lower_case` is set in `sentence_bert_config.json` and the ST Transformer module lowers the text before tokenizing — neither mLateOn nor ettin does this. Same architecture class, third distinct host contract; every knob read out of the shipped configs + PyLate source, never assumed.
- **The skiplist is live here** (unlike mLateOn's empty one): 32 ASCII punctuation token ids, applied to DOCUMENT vectors host-side after encoding; `[CLS]`/`[D]`/`[SEP]` are kept, queries keep everything. Truncation is `query_length−1`/`document_length−1` = 47/511 *before* the `[Q]`/`[D]` insert at position 1.
- **The card publishes scores AND shapes — use both.** The reimplementation reproduces the MaxSim scores `[11.2081, 11.5308, 11.4104, 11.4756]` to all four printed decimals *and* the printed embedding shapes ((12,64) query, (18,64) first doc) pre-export. At 32M the margins are ~0.1, so Mars-on-top is a genuinely tight rank check — and it pins tokenizer, lowercase, prefixes, pad, skiplist, head fold and normalize in one gate.
- **Smoke and task disagree at 32M — trust the task, but run both.** int8's worst per-token smoke cos on random ids at encode_512 is **−0.27** (a destroyed token vector; the worst smoke of any encoder here), yet NanoSciFact damage is only −0.008 nDCG@10 with recall@5/hit@1 unchanged, and torch-index × int8-queries is within 0.007 of control. The 64-d head amplifies per-token noise on abusive inputs; real text does not hit it.
- **wi8fc = FC int8 DRQ + embedding int8 channelwise**: 32M → fp32 131 MB, int8 **39 MB (144 MiB peak RSS — the smallest retriever footprint in this repo)**, fp16 67 MB but 676 MiB RSS (the per-signature fp16→fp32 expansion law holds even at 32M). ~14,800 tok/s at encode_512 int8 on an M4 Max — int8 and fp16 bench identical, so int8 wins on RSS alone.


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

#### The vision bundle needs a different exporter flag than the text one

The published `Shieldstral-1.0-3B-vision_int4.litertlm` does not load on a mobile GPU: the delegate takes 52
of 1187 ops on `prefill_2048` and the engine is refused, on `STABLEHLO_COMPOSITE: odml.softmax`. The text
bundles are unaffected — and the difference is not the vision tower, it is which exporter each lane used.
litert-torch 0.9.2 marks the attention softmax as an `odml.softmax` composite unconditionally, and
litert-converter 0.3.0 has no lowering for it. `scripts/export_simple_template.py` carries an env-gated strip
for exactly this; `scripts/export_internvl_decoder.py`, which the vision lane calls, does not.

Two ways out. The strip is one. The better one needs no source change: **litert-converter 0.3.1 lowers the
composite natively**, so re-exporting the decoder on it produces a builtin `SOFTMAX` and an otherwise
identical graph.

```bash
CACHE=4096 PREFILL=2048,1024,512,256,128,64,32,16,8,4,2,1 RECIPE=BOCTAV4 \
  python scripts/export_internvl_decoder.py src_models/shieldstral-3b-text out_decoder   # on a 0.3.1 venv
DEC=out_decoder VIS=out_vision python shieldstral_work/vision/build_shieldstral_bundle.py
```

⚠ The `PREFILL=` and `RECIPE=` above are not optional. The exporter defaults to `128,512`, which does not
reproduce the shipped bundle — its largest subgraph is `prefill_2048`.

Check the result without a device: the re-exported decoder's subgraphs should be 1187 ops for `prefill_2048`
through `prefill_2`, 1058 for `prefill_1` and 1087 for `decode` — identical, one for one, to the text sibling
that already delegates in full. A whole-file scan should find zero `odml.softmax`; the ~2660 `odml.rms_norm`
occurrences stay, and are not a blocker. On a Galaxy S26 the rebundled file then delegates 15202 of 15202 ops
across all 13 subgraphs, the same numbers as the text file.

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

Gates on the 0.16.0 pip CLI (temperature 0, `--cache no`): 3B int8 AND int4 pass text 7/8 + image 5/5 on cpu and macOS gpu, with image answers verbatim identical to the HF bf16 reference (`verify_lfm25_vl_hf.py`); 1.6B passes text 8/8 on all four configs; 450M int4 passes 8/8 + 4/5. iPhone GPU was not re-tested for these VL files. LiteRT-LM#3129, once read as an iOS block for the ShortConv family, turned out to be an integration-side `maxNumTokens` sizing issue and the text-only 1.2B GPU files do run on iOS Metal at `maxNumTokens` 1024 — but that is measured on the text family, not here, so treat iOS for VL as unmeasured rather than blocked.

⚠ **The 64k-vocab models (1.6B, 450M) deterministically miss fine shape/counting fixtures on-device while the torch reference answers them** — and every conversion-side suspect has been eliminated by experiment: exported vision tower AND projector match torch at cosine 1.0000 on identical inputs; an unquantized bundle reproduces the identical misses; runtime patchify layout equals the HF processor's (code-read, both (ph,pw,c) raster); the prompt/token streams match token-for-token (boi/eoi are single tokens in every vocab); double-BOS, fp16 decode, jpeg-level pixel perturbation and prefill-chunk-boundary alignment (probed by forcing the image block exactly onto a 256-token chunk edge via a custom chat template) all change nothing; geometry probes on-device (corner localization, stripe orientation) come back CORRECT. Coarse understanding, OCR and localization are intact — the residue is engine-internal precision on contour/counting answers, the 3B is unaffected, and the next step is instrumenting the runtime to diff the spliced embedding sequence against HF `inputs_embeds`.

## Verify a reproduction

```bash
~/venvs/ltconv040dev/bin/python scripts/verify_quality.py out/<key>/model.litertlm --json   # 8-question gate
# parity (dense): scripts/parity_gsm8k.py  ·  reasoning models: run at --max-tokens 2048
# single-token classifiers: label F1 + margin correlation vs the source model, not the 8-question gate
```

## granite-4.1-3b (dense) — and the BOS a converted bundle must not prepend

`granite41_work/convert_granite41_3b.sh` converts IBM's dense 3.4B instruct model to `.litertlm` on a pristine released stack (litert-torch 0.9.3 / litert-converter 0.3.1 / ai-edge-quantizer 0.8.0 / litert-lm-builder 0.16.0, transformers 5.14.1) — no patched checkout. Published: [litert-community/granite-4.1-3b](https://huggingface.co/litert-community/granite-4.1-3b). **Requires litert-lm ≥ 0.16 to run.**

```bash
cd granite41_work
pip install "litert-torch==0.9.3" "litert-lm-builder==0.16.0" "transformers==5.14.1"
python -c "from huggingface_hub import snapshot_download; snapshot_download('ibm-granite/granite-4.1-3b', local_dir='src_models/granite-4.1-3b')"
./convert_granite41_3b.sh          # int4 (ship) + int8, six prefill signatures, NO_START_TOKEN=1
```

**The one thing that will bite you on this family.** `litert-lm-builder` writes a `start_token` into the bundle metadata from `tokenizer.bos_token` whenever that field is set, without consulting `add_bos_token`. Granite declares `add_bos_token: False` **and** its BOS is the same token as its EOS (`<|end_of_text|>`), so the runtime prepends a token the model reads as "this document has already ended": it answers by echoing the question back, and the 8-question sanity gate drops from 8/8 to 5/8. It looks exactly like quantization damage and is not — feeding the same rendered prompt to the bf16 PyTorch model with and without that leading token reproduces both halves.

The conversion script sets `NO_START_TOKEN=1`, which clears `bos_token` before the metadata is built. For a bundle you have already built, `strip_start_token.py in.litertlm out.litertlm` unpacks it, drops the field and repacks — weights untouched, about 30 seconds. Check any model whose `tokenizer_config.json` says `add_bos_token: False`; bos == eos is the shape that hurts.

**Six prefill signatures, not eleven.** Every exported signature is charged engine memory whether or not it is called. An eleven-signature build of these weights is killed by iOS during Metal initialisation unless the app caps the context at 1024; `1024,256,64,16,4,1` creates its engine at the bundle's full 4096 on an iPhone 17 Pro, for 12 MB more on disk.

`hf_oracle.py` runs the same eight questions through the HF model in bf16 first, so a later miss on the converted file can be attributed to the conversion rather than to the model.

### granite-4.1 finetune intake — the BOS guard goes generic, and the family fact meets its first exception

Measured 2026-08-25 on [Kezmark/Mordant-3B-Think](https://huggingface.co/Kezmark/Mordant-3B-Think)
(full SFT for image-prompt composition with chain-of-thought; the highest-download real
granite-4.1-3b derivative after excluding a name-declared Claude distill and the
unsloth/mlx mirrors). Hub demand: 19 finetunes + 15 adapters.
Published: [mlboydaisuke/Mordant-3B-Think-LiteRT](https://huggingface.co/mlboydaisuke/Mordant-3B-Think-LiteRT)
(int8, with the measured card and litertlm_manifest.json).

```bash
python scripts/convert.py Kezmark/Mordant-3B-Think
```

One command, clean shell: export 1,621 s → 3.76 GB int8, reduced 7-signature ladder,
externalized embedder, think-aware gate → **8/8 PASS, decode ~71 tok/s CPU**, template
byte-equal 1474/1474, bundle stop = the tokenizer's `<|end_of_text|>` 100257.

Two things this subject taught the intake:

- **The BOS trap generalizes, so the guard did too.** Mordant has `bos == eos ==
  <|end_of_text|>` and a template that never leads with BOS — the shape from the base
  granite-4.1-3b lesson above. Measured on this subject: prepending that one token flips HF
  bf16 greedy from a correct answer to an endless ``` ``` ``` loop. `convert.py` now runs
  `ensure_no_spurious_start_token` after every stock export. *Rewritten 2026-08-27:* it
  renders the chat template **twice** — once with `bos_token` bound (what transformers
  serves, the reference) and once with it empty (what minijinja does on device) — and drops
  the metadata start_token only when `1 + literal-BOS-in-the-device-render` exceeds the
  reference count. That separates `{{ bos_token }}` from a literal BOS, which the earlier
  `add_bos_token` / auto-add test conflated in both directions: it would have dropped
  Falcon-H1's start token (leaving zero BOS) and left Zamba2's genuine duplicate in place.
  The e2e report records `start_token: dropped_spurious_bos` / `dropped_duplicate_bos` /
  `kept_matches_reference`, plus the two counts under `start_token_bos`.
- **The template family fact has its first measured exception.** Every LLM derivative
  measured until now kept the base template byte-for-byte (modulo CRLF re-serialization);
  Mordant *replaces* it with its own 1,474-byte think-form template. The stock path is
  indifferent — it embeds the derivative's own template verbatim either way (byte-equal
  measured) — but "derivative templates always equal the base" is now demonstrated folklore,
  not fact. Same lesson one layer down: the trainer's re-save collapsed `tokenizer.json`'s
  pre_tokenizer (granite's contraction-aware Split → plain ByteLevel); the derivative's own
  tokenizer is what gets bundled and what both sides of any A/B must use.

The reduced 7-signature prefill ladder is now applied to **every ≥3B export**, not just
hybrids — the dense granite-4.1-3b's own 11-signature iOS Metal kill (documented above) is
the same signature-memory law the hybrid 4B hit.

### 2026-08-27 — auditing every published bundle's first token, and what the engine really does

The Hy-MT2 double-BOS finding raised an obvious question about everything already
shipped, so all **88 published `.litertlm` files across 64 repos** were swept. Only the
header is needed — `LlmMetadata` sits in the first few KB, so two HTTP range requests per
file read the `start_token` and the embedded template without downloading any weights
(`make_manifest.py` uses the same pattern). Zero files failed to parse.

**Read the engine, not the output.** The earlier discriminator compared generated text,
which is why it took three probes to survive int8 coincidence. The runtime exposes the
answer directly: `Conversation.render_message_to_string` returns the minijinja render
verbatim, `Engine.tokenize` returns the exact ids, and with `enable_benchmark=True` the
prefill counter returns the assembled prompt length. Repacking one local bundle's metadata
three ways settles the mechanics with no generation at all:

| `start_token` | resolves to | prefill tokens |
|---|---|---|
| `<bos>` | id 2 | **16** |
| the string `"None"` | **id 9336, the word `None`** | **16** |
| absent | — | **15** |

`len(tokenize(render))` is 15 in all three. So: the engine prepends the start token and it
costs exactly one token; the render it prepends onto **does not contain the BOS** even
though the template source begins `{{ bos_token }}` — minijinja leaves `bos_token`
unbound; and `tokenize("hello world")` returns `[23391, 1902]`, so the engine's encode
adds no special tokens regardless of what the tokenizer's post-processor says. (The
bundle's own `tokenizer.json` *does* carry a `TemplateProcessing` that would add `<bos>`;
it is simply not applied.)

The count a bundle actually feeds is therefore `(1 if start_token is the bos) + literal
BOS at the head of the render`, and the reference it must equal is the same template
rendered with `bos_token` bound — what transformers serves, since its chat path encodes
with `add_special_tokens=False`.

**19 of 88 bundles were feeding a first token the model was never trained on**, in three
shapes:

- **The literal string `"None"` (10 bundles)** — `str(None)` reached the metadata for
  models whose tokenizer has no BOS at all, and the engine resolves it to a real vocab
  token. Measured on the published `InternVL3-1B`, the prompt began
  `['None', '<|im_start|>', 'user', …]`. InternVL3-1B/2B, InternVL3_5-1B/2B/4B,
  LLaVA-OneVision-0.5B, Mage-VL, Ovis2.5-2B, Qwen2-VL-2B, SmolLM3-3B.
- **A genuine double BOS (6 bundles)** — the template writes the BOS as a *literal*, so
  the start token lands on top of it. Measured on the published `SmolVLM2-500M`:
  `['<|im_start|>', '<|im_start|>', 'User', …]` against a reference of one. Zamba2-1.2B/2.7B,
  SmolVLM2-500M/2.2B, Nanbeige4.1-3B/4.2-3B.
- **A BOS where the reference has none (3 bundles)** — granite-4.0-h-1b, Nemotron-H-4B,
  Phi-4-mini-reasoning. The 350m was fixed for exactly this in August; the 1b was left
  because it tolerates the token, which is not the same as being faithful to training.

**Falcon-H1, LFM2.5 and gemma-3-270m are clean** — they write `{{ bos_token }}`, which
renders empty on device, so their start token supplies the one BOS the model expects.
Dropping it would have shipped a bundle with *no* BOS. Two more read as defects until the
reference was taken from real token ids rather than declared strings: North-Micro-Vision's
`start_token: ids[2]` **is** `<BOS_TOKEN>`, and PaddleOCR-VL's canonical head is
`<|begin_of_sentence|>`, not the `<s>` its config declares.

**The fix is metadata-only.** `drop_start_token.py` unpacks, removes the block, and
repacks; `section_identity.py` then sha256s every section and requires everything except
`LlmMetadataProto` to be byte-identical, so no requantization can hide in a repack and the
card's measured numbers still stand — only the file's own sha256 changes. Nanbeige needs a
second edit: its affix `system.prefix` is `system\n`, i.e. the leading `<|im_start|>` was
coming from the start token, so a bare drop fixes the no-system path and breaks the system
one. Dropping the start token *and* rewriting that prefix to `<|im_start|>system\n` leaves
the system path's prompt identical (prefill 24 before and after) while fixing the other.

## granite-4.2-3b (dense, thinking) — the think prefill an extracted template silently drops

`granite42_work/convert_granite42_3b.sh` converts IBM's compact reasoning model (3.66B dense, Aug 2026) on a pristine released stack (litert-torch 0.9.3 / litert-converter 0.4.0 / ai-edge-quantizer 0.9.0 / litert-lm-builder 0.16.1, transformers 5.14.1). Published: [litert-community/granite-4.2-3b](https://huggingface.co/litert-community/granite-4.2-3b). **Requires litert-lm ≥ 0.16 to run.**

```bash
pip install "litert-torch==0.9.3" "litert-converter==0.4.0" "ai-edge-quantizer==0.9.0" "litert-lm-builder==0.16.1" "transformers==5.14.1"
python -c "from huggingface_hub import snapshot_download; snapshot_download('ibm-granite/granite-4.2-3b', local_dir='src_models/granite-4.2-3b')"
granite42_work/convert_granite42_3b.sh    # int8 + int4, six prefill signatures, NO_START_TOKEN=1
python tools/add_thought_channel.py granite42_work/out_int8/model.litertlm granite-4.2-3b_int8.litertlm --start '<think>' --end '</think>'
```

4.2 returns from the 4.0/4.1 hybrid line to full attention, so the dense 4.1 rail carries over — six-signature prefill ladder (an eleven-signature build of this shape is killed by iOS at Metal engine init), externalized embedder, `NO_START_TOKEN=1` (4.2 declares `<s>` as BOS but its tokenizer never prepends it — the post-processor adds nothing — so the builder's unconditional `start_token` write must be suppressed). Both quantizations gate **8/8 on CPU and GPU** (Mac, litert-lm 0.16.0), answers arriving cleanly through the thought channel; measured throughput is on the model card.

**What 4.2 adds is the reasoning machinery, and two conversion steps are load-bearing:**

- **The think prefill must survive template extraction.** IBM's template ends the generation prompt with `<|im_start|>assistant\n<think>\n`. The exporter's `parse_chat_template` derives the assistant prefix from an assistant *history* message (`add_generation_prompt=False`), so a jinja that opens `<think>` only in its generation branch exports a bundle whose `model.prefix` is a bare `<|im_start|>assistant\n` — the scaffold is silently gone, and think-discipline is left to the quantized model's choice. `templates/granite42_think.jinja` therefore puts the opener in the assistant history branch too (the chatml_think shape). After any export of a thinking model, read `prompt_templates.model.prefix` out of the bundle before trusting it.
- **The thought channel must be declared.** The structured-template path emits no `LlmMetadata.channels`; without it the runtime streams raw reasoning into the answer and silently ignores any thinking budget. `tools/add_thought_channel.py` adds the block post-export and refuses to pass if anything but `channels` changed.

`granite42_work/hf_oracle.py` is the bf16 reference for the same eight gate questions (scored on the text after `</think>` — the reasoning routinely contains the expected string, e.g. it repeats "0.9" while comparing 0.9 vs 0.11, so scoring the whole output would fake a pass). `granite42_work/gate8q.py` applies the same prefilled-opener-aware scoring to the converted bundle.

## Qwen2.5-Coder-1.5B-Instruct — a 1.5B code model at 1.12 GB, and why the size is the recipe

`qwen25coder_work/convert_qwen25_coder.sh`. Published: [litert-community/Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/litert-community/Qwen2.5-Coder-1.5B-Instruct). **Requires litert-lm ≥ 0.16.**

Decode on this runtime is memory-bandwidth-bound, which has a blunt consequence: putting a bigger model on the GPU does not make it fast, and halving the weights does. Measured on one machine and one protocol (M4 Max, `-p 256 -d 256 --cache no`, quiet, serialized) — this 1.12 GB build against a 2.19 GB 3B-class int4 build converted the same day:

| | Qwen2.5-Coder-1.5B (1.12 GB) | 3B-class int4 (2.19 GB) |
|---|---|---|
| prefill | **3037 tok/s** | 1241 |
| decode | **137.8 tok/s** | 86.3 |
| TTFT | **0.099 s** | 0.233 s |
| init | **2.38 s** | 5.96 s |

**`EXTERNALIZE_EMBEDDER=1` is required.** This model ties its embedding and `lm_head`, so `BOCTAV4` — int4 on every FULLY_CONNECTED, int8 on EMBEDDING_LOOKUP — describes one tensor two ways. The quantizer resolves the conflict by copying the 151936×1536 table once per prefill signature: seven copies, 1.63 GB, **65% of a 2.53 GB file**, with nothing logged and "int4" still true of the linears. `check_bundle_sanity.py <bundle> --params N` makes that a one-line check (exit 1 on duplication); a healthy int4 build lands near 0.7 bytes/param.

**The chat template carries a default system prompt.** Upstream inserts `You are Qwen, created by Alibaba Cloud. You are a helpful assistant.` whenever the caller supplies no system message. A generic ChatML template renders the markers without it, so the shipped bundle would run the model outside the state it was tuned in on every default-path request — quietly. `qwen25_coder_simple.jinja` bakes it into the user prefix and renders byte-identical to upstream for a single turn.

**Gate a code model on code.** An 8-question general-knowledge check certifies nothing here. `eval_code.py` asks for six small functions and **executes** each against assertions — a task passes only if the code imports and every assertion holds, because generated code that reads correctly and raises on the second assertion is exactly how a damaged quantization survives a skim. This build scores 6/6, and 8/8 on the general gate.

## S1-mini (0.6B ASR-transcript normalizer) — the template flag a runtime can't pass, and the stop strings that eat your punctuation

`s1mini_work/convert_s1mini.py` converts [superwhisper/s1-mini](https://huggingface.co/superwhisper/s1-mini) (published: [mlboydaisuke/S1-mini-LiteRT](https://huggingface.co/mlboydaisuke/S1-mini-LiteRT)) — Superwhisper's Qwen3-0.6B finetune that rewrites raw dictation into clean written text — on the released stack (litert-torch 0.9.3, litert-lm-builder ≥ 0.15). One command, int8, 688 MB. License note: Apache 2.0 plus a naming clause — any distribution must keep identifying the model as "S1-mini" by "Superwhisper".

```bash
pip install "litert-torch==0.9.3" "litert-lm-builder>=0.15" "transformers>=4.51" torch
python s1mini_work/convert_s1mini.py            # download (pinned rev) + export + stop-token fix
python s1mini_work/gate_normalize.py out/s1-mini-int8/model.litertlm --backend cpu
```

**The model needs `enable_thinking=False`, and no runtime will ever pass it.** S1-mini keeps Qwen3's template, whose default renders the thinking-mode prompt; the card warns that form usually yields no usable output, and the trained form is the `enable_thinking=False` render (assistant turn opens with an empty `<think>` scaffold). A LiteRT-LM engine applies the embedded template with no such kwarg — so the verbatim vendor template ships a broken model. The fix is resolved at export time: `s1mini_bundle.jinja` hardcodes the non-thinking render, and also bakes in the exact system prompt the card requires (clients that can't send a system message — most simple chat UIs — get the trained format anyway; clients that send the card's wording lose nothing, duplicates are dropped). Render equivalence to the vendor template with `enable_thinking=False` was verified 12/12 before export, and a follow-up turn's render is a string-extension of the previous turn plus reply, so KV reuse stays valid.

**The exporter's composite stop strings silently eat final punctuation.** litert-torch expands the template's turn suffix into punctuation-prefixed stop variants (`".<|im_end|>\n"`, `"?<|im_end|>\n"`, …) to cope with SentencePiece greedy merging. Qwen3 is BPE — that merge never happens — but the runtime still matches the multi-token stop sequence and strips the **whole** match, so a reply ending `…Thursday."` comes back `…Thursday`. On a punctuation-restoration model this is fatal, and a general-knowledge gate can never see it (regexes match content words, not a trailing period). Measured cleanly: 7 of 7 task cases whose reference ends in "." lost exactly that period, and a metadata-only rewrite (drop every `token_str` stop, keep the two real stop ids) flipped all 7 back. The convert script does that rewrite automatically. If you export other BPE models with this stack, check your bundles.

**int8 only, and that is a measurement, not a guess.** The int8 build reproduces the HF fp32 greedy outputs **byte-for-byte, 10/10 task cases, on CPU and GPU** (`gate_normalize.py` feeds both sides the identical rendered string). int4-b32 was built and rejected: it drops the comma before "and" and discourse words like "hmm" (identically on both backends), and decodes *slower* than int8 at 0.6B (24 vs 31 tok/s CPU, 88 vs 105 tok/s Mac GPU) — block-dequant overhead beats the bandwidth saving at this size. Also note s1-mini ties its embedding and lm_head, so an int4-FC + int8-embedding recipe triggers the vocab-table duplication defect described in the Qwen2.5-Coder section (656 MB "int4", larger than int8; `externalize_embedder=True` clears it) — another reason int8 is the honest recipe here.

**Gate it on its own task.** S1-mini is not a chat model: asked "What is 17 + 25?" it will normalize the question, not answer it, so the 8-question gate certifies nothing. `gate_normalize.py` runs 10 dictation-normalization cases (all four registers, lists, email, numbers/time/currency, and the pure-filler case whose correct output is the empty string) against HF fp32 on the identical rendered string, with a BOS A/B check first. Mac numbers for this build (p256/d256, runs 3, `--cache no`, litert-lm 0.16.0, M4 Max): GPU 3777 prefill / 146.7 decode tok/s, TTFT 0.075 s; CPU 466 / 34.6, TTFT 0.58 s.

## granite-speech-5.0-470m-turboctc (CTC conformer ASR → plain .tflite)

First ASR in the catalog. `granite_speech_work/` converts IBM's 473M CTC conformer to plain LiteRT `.tflite` — one encoder forward, no decoder loop; the host does argmax-collapse-decode in ~10 lines. Published: [litert-community/granite-speech-5.0-470m-turboctc](https://huggingface.co/litert-community/granite-speech-5.0-470m-turboctc) (int8 518 MB + fp16 994 MB).

```sh
hf download ibm-granite/granite-speech-5.0-470m-turboctc --local-dir granite_speech_work/hf_model
python3 granite_speech_work/fetch_fixtures.py         # 20 LibriSpeech dev-clean clips
python3 granite_speech_work/eager_gate.py             # oracle: WER 3.79% + buffer checks
python3 granite_speech_work/convert_granite_speech_ctc.py
python3 granite_speech_work/verify_tflite.py          # 20/20 transcript match per variant
```

Env: torch 2.12 + torchaudio 2.11 (the processor's mel front-end needs torchaudio) + transformers 5.14 + litert_torch 0.9.3 + ai-edge-quantizer + ai-edge-litert.

**The Hub repo is two-faced (as of 2026-09-01), and `AutoModel` fails on released transformers.** `config.json`/`model.safetensors` are re-exported for the unreleased stock `GraniteSpeech5ForCTC` (model_type `granite_speech5_ctc`, HF-standard tensor names, no `auto_map`), while the bundled remote-code `.py` files are the older packaged `CtcConformerForCTC` with stock-3.3 granite_speech naming. `load_model.py` loads through the remote code with a 16-rule full-key state-dict rename; the nasty pair is `conv.norm.*` (BatchNorm) → `conv.batch_norm.*` while `norm_conv.*` (LayerNorm) → `conv.norm.*` — rename per full key in one pass, never cascading prefix rewrites. `load_state_dict(strict=True)` over all 550 tensors proves the mapping; a 20-clip WER gate proves the wiring.

**Shaw's re-association is the difference between a 923 MB and a 518 MB int8 file.** The encoder's rel-pos bias `einsum(q, rel_pos_emb(dists[:blk,:blk]))` constant-folds under `torch.export` into a `[blk, blk, 128]` fp32 constant per layer × per distinct block length × per signature (~0.5 GB across the 3-signature export) that FC-only dynamic int8 cannot reach (it feeds BATCH_MATMUL). Rewriting as `gather(q @ W.T, dists)` (`shaw_patch.py`) keeps the single shared `[1025, 128]` table and is **bitwise identical** in eager on all three windows. Cost: 17 GATHER_ND ops — the mobile GPU delegate then refuses to compile the graph (measured on a Galaxy S26), which is fine because this is a CPU ship.

**Fixed windows are semantically cheap because attention is block-local (128 frames), but not free.** Signatures `transcribe_{5,10,30}s` zero-pad audio to the window; padding shifts the 128-frame block boundaries, and on the 20-clip fixture set 19/20 transcribe identically vs unpadded with one single-word change (corpus WER 3.79% → 4.24%). Route each clip to the smallest window that fits.

**Verification is transcript-exact, not just cosine-close.** int8 and fp16 both match the fp32 eager transcripts 20/20; on a Galaxy S26 (CompiledModel CPU) the transcripts are again exact on every signature and the int8 logits are bit-identical to the desktop run of the same file (max abs diff 0.0). Speed: int8 transcribes a 30 s window in 239 ms on an M4 Max (8 threads) and 666 ms on the S26 — ~126× / ~45× real-time; fp16 is ~3× slower on CPU (XNNPACK fp32 repack), use int8 on device.

## VibeVoice-ASR-BitNet (speech-to-text LLM, audio-in `.litertlm`) — first audio-in bundle in the catalog

`vibevoice_asr_work/` converts Microsoft's [VibeVoice-ASR-BitNet](https://huggingface.co/microsoft/VibeVoice-ASR-BitNet) (σ-VAE acoustic + semantic conv encoders at 24 kHz → a Qwen2.5-1.5B-shaped **BitNet/ternary** LM, MIT) into one `.litertlm` that LiteRT-LM's released runtime (≥ 0.16.1) drives end-to-end through its **generic audio path** — the runtime decodes and resamples the clip, frames raw PCM into 3200-sample frames, runs the bundled audio encoder, and splices the 7.5 Hz embeddings into the ChatML prompt. No runtime patch, no litert-torch patch.

```sh
hf download microsoft/VibeVoice-ASR-BitNet model-00001-of-00003.safetensors model-00002-of-00003.safetensors \
   model-00003-of-00003.safetensors config.json tokenizer.json tokenizer_config.json vocab.json generation_config.json \
   --local-dir vibevoice_asr_work/hf
python3 vibevoice_asr_work/fetch_fixtures.py                    # 20 LibriSpeech dev-clean clips at 24 kHz
python3 vibevoice_asr_work/load_bitnet.py                       # legacy -> transformers-native, ternarize, strict load; hf_native/ + lm_native/
python3 vibevoice_asr_work/eager_gate.py                        # fp32 oracle: WER 2.68 % (12/448)
python3 vibevoice_asr_work/export_audio_encoder.py --seconds 30 --quant fp32,wi8fc
EXTERNALIZE_EMBEDDER=1 CACHE=2048 PREFILL=512,128,32 \
  python3 scripts/export_simple_template.py vibevoice_asr_work/lm_native out/vibevoice-lm templates/chatml_simple.jinja BMIX4_128
litert-lm unpack out/vibevoice-lm/model.litertlm --output-dir out/vibevoice-lm/unpack   # the two LM tflites
python3 vibevoice_asr_work/patch_tokenizer.py vibevoice_asr_work/lm_native/tokenizer.json vibevoice_asr_work/lm_native/tokenizer_audio.json
DEC=out/vibevoice-lm/unpack EMB=tf_lite_embedder DECODE=tf_lite_prefill_decode \
  ENC=vibevoice_asr_work/out/audio_encoder/audio_encoder_30s_wi8fc.tflite \
  TOK=vibevoice_asr_work/lm_native/tokenizer_audio.json OUT=out/vibevoice-bundle \
  python3 vibevoice_asr_work/build_bundle.py
python3 vibevoice_asr_work/mac_gate.py --bundle out/vibevoice-bundle/VibeVoice-ASR-BitNet.litertlm   # runtime gate: WER 2.68 %
```

Env: torch 2.13 + transformers 5.14.1 (has `vibevoice_asr` natively) + litert-torch 0.9.4 + ai-edge-quantizer 0.9.0 + litert-lm-builder 0.16.1 for everything but the LM export, which goes through the lane's usual `export_simple_template.py` (litert-torch 0.9.3); the gate needs `litert-lm-api` 0.16.1 (python) — `litert-lm` 0.16.0 supplies `unpack`/`benchmark`.

**"BitNet" is applied at conversion, per tensor, and it is the model.** The Hub checkpoint stores fp32 latent weights; VibeASR.cpp (the vendor runtime) ternarizes the 7 projections per LM layer with a per-tensor absmean scale (`s = 1/mean|w|`, `round(w·s).clamp(−1,1)/s`) at GGUF-conversion time. `load_bitnet.py` does exactly that (α 0.030–0.070, 37 % zeros); run un-ternarized, the LM emits `<|im_end|>` spam. A per-tensor ternary is exactly representable by int4 blockwise min-max at any block size (every block is {−α, 0, +α} → scale α/7), so the LM ships as `BMIX4_128` — the same driver every dense LLM in this repo uses — and the Mac runtime transcript WER equals the fp32 oracle to the word.

**The tokenizer has no string for the speech markers.** The repo ships the Qwen2 tokenizer (added tokens end at 151645) while the model was trained with 151646/151647/151648 (`<|object_ref_start|>` / `<|object_ref_end|>` / `<|box_start|>` in Qwen2.5) as speech start/end/pad — VibeASR.cpp hard-codes the ids. The generic runtime path tokenizes `start_of_audio_token` and `audio_suffix` as text, so `patch_tokenizer.py` adds the three strings as special tokens; they land on exactly those ids by construction (asserted), ordinary text tokenizes identically.

**The audio encoder is one single-signature tflite with the vendor's normaliser folded in.** `audio` f32 `[1, 225, 3200]` (30 s of raw 24 kHz PCM as the runtime frames it, zero-padded) → `features` f32 `[1, 225, 1536]`: RMS-normalise to −25 dBFS over the non-zero frames + peak clip, acoustic encoder (mean latents — the HF class adds `vae_std·randn` even at inference; the vendor runtime uses the mean), semantic encoder, projector. The output must be named literally `features` (return a dict — positional returns become `output_0`) and every reduction keeps `keepdim=True` (the GPU delegate rejects rank-0 tensors). int8 dynamic on the linears (`wi8fc`, 951 MB; the vendor ships the VAE at int8) is WER-lossless; the window is 30 s because a 10 s window costs 7 words on the 20-clip set from context resets at chunk boundaries (the engine chunks by window with no overlap).

**Bundle metadata is the runtime contract, written by hand in `build_bundle.py`:** `GenericModel{audio_enabled, delimiter_regex/audio_token_regex on <|box_start|>, start_of_audio_token, audio_suffix, skip_mel_spectrogram_extraction, 24000 Hz, frame = hop = 3200}` + ChatML templates + a jinja that emits the vendor system prompt and renders an audio item as `<|box_start|>\n` (default instruction "Please transcribe it." when the turn has no text). Without an audio adapter section the executor treats the encoder as a streaming encoder with window = 225 frames and (buffering off by default) zero-pads short clips; valid tokens = valid frames.

**Gates.** fp32 oracle 12/448 = 2.68 % (vendor prompt; adding the generation prompt changes nothing; dropping the duration sentence 2.90 %); encoder int8 through the eager LM 2.68 %; LiteRT-LM 0.16.1 python CPU 2.68 %, LM on Metal GPU 2.46 %; Pixel 8a `litert_lm_advanced_main` v0.16.1 CPU 3.12 % (one clip flips "On the" → "Under" — Arm int8 numerics), Galaxy S26 CPU see the card. LM speed: M4 Max CPU 357 / 59.3 tok/s, Metal 2045 / 138.6 tok/s (`-p 256 -d 256 --cache no`). The audio encoder is **CPU-only** on Metal/WebGPU: the 30 s stem dispatches 90 001 workgroups against a 65 535 limit and the engine returns empty text — a shape problem (a ≤ 5.5 s window would fit), not an op problem.
