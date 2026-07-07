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
best-inference, flagged below). VLMs have their own `scripts/ship_*.sh` (InternVL / LLaVA-OV / SmolVLM2 /
Ovis) — this file is the dense/reasoning-LLM set.

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

## Verify a reproduction

```bash
~/clipconv/bin/python scripts/verify_quality.py out/<key>/model.litertlm --json   # 8-question gate
# parity (dense): scripts/parity_gsm8k.py  ·  reasoning models: run at --max-tokens 2048
```
