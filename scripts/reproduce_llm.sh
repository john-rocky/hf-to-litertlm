#!/bin/bash
# One-command reproduction of every dense / reasoning LLM .litertlm shipped from this repo.
#
#   bash scripts/reproduce_llm.sh <model-key>     # reproduce one model -> out/<key>/model.litertlm
#   bash scripts/reproduce_llm.sh --list          # list model keys + one-line recipe
#   bash scripts/reproduce_llm.sh --all           # reproduce every model serially (heavy)
#
# Recipes reconstructed 2026-07-06 from cards/ + auto-memory + reports/ (see REPRODUCE.md
# for the full table, sources, and per-model caveats). The engine is
# scripts/export_simple_template.py; recipe legend:
#   BOCTAV4     = blockwise-32  int4 + OCTAV + int8 embedding  (best quality; Mac/Android)
#   BOCTAV4_128 = blockwise-128 int4 + OCTAV + int8 embedding  (iPhone / 4B: fits <2GiB section)
#   BMIX4 / BMIX4_128 = blockwise int4 (min-max, no OCTAV) + int8 embedding
# ENV: FORCE_SPM (BPE->SP tokenizer, auto FIX_ADDED_TOKENS for <think> models),
#      EXTERNALIZE_EMBEDDER (split embedding so 3B+ loads on iPhone), PHI3_STATIC_ROPE (Phi),
#      GPTQREC_GCD_FIX (GPTQ ingest), CACHE (KV length), PREFILL.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-$HOME/clipconv/bin/python}"
EXPORT="$PY scripts/export_simple_template.py"
SM=src_models

dl() { # dl <hf_id> <local_dir>  — download once if missing
  [ -d "$2" ] || $PY -c "from huggingface_hub import snapshot_download; snapshot_download('$1', local_dir='$2')"
}

reproduce() {
  local key="$1"
  case "$key" in
  # ---- clean single-command exports ----
  fastcontext-4b)      EXTERNALIZE_EMBEDDER=1 CACHE=4096 $EXPORT microsoft/FastContext-1.0-4B-SFT out/$key templates/chatml_simple.jinja BOCTAV4 ;;
  nanbeige4.1-3b)      FORCE_SPM=1 EXTERNALIZE_EMBEDDER=1 CACHE=4096 $EXPORT Nanbeige/Nanbeige4.1-3B out/$key templates/chatml_simple.jinja BOCTAV4 ;;
  olmo2-1b)            CACHE=4096 $EXPORT allenai/OLMo-2-0425-1B-Instruct out/$key templates/olmo2_simple.jinja BOCTAV4 ;;
  olmo2-7b)            CACHE=4096 EXTERNALIZE_EMBEDDER=1 $EXPORT allenai/OLMo-2-1124-7B-Instruct out/$key templates/olmo2_simple.jinja BOCTAV4 ;;  # desktop-only (>2GiB section)
  polaris-4b)          EXTERNALIZE_EMBEDDER=1 CACHE=4096 $EXPORT POLARIS-Project/Polaris-4B-Preview out/$key templates/qwen3_think.jinja BOCTAV4_128 ;;
  qwen3-1.7b)          CACHE=4096 $EXPORT Qwen/Qwen3-1.7B out/$key templates/qwen3_think.jinja BOCTAV4 ;;  # ship dropped→private
  qwen3-4b-thinking)   EXTERNALIZE_EMBEDDER=1 CACHE=4096 $EXPORT Qwen/Qwen3-4B-Thinking-2507 out/$key templates/qwen3_think.jinja BOCTAV4_128 ;;  # block128 ONLY
  r1-distill-qwen-1.5b) CACHE=4096 $EXPORT deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B out/$key templates/deepseek_r1_simple.jinja BOCTAV4 ;;
  r1-distill-qwen-7b)  CACHE=4096 EXTERNALIZE_EMBEDDER=1 $EXPORT deepseek-ai/DeepSeek-R1-Distill-Qwen-7B out/$key templates/deepseek_r1_simple.jinja BOCTAV4 ;;  # desktop-only
  smollm3-3b)          CACHE=4096 EXTERNALIZE_EMBEDDER=1 $EXPORT HuggingFaceTB/SmolLM3-3B out/$key templates/smollm3_think.jinja BOCTAV4 ;;
  jan-nano)            EXTERNALIZE_EMBEDDER=1 CACHE=4096 $EXPORT Menlo/Jan-nano out/$key templates/qwen3_think.jinja BOCTAV4_128 ;;  # template: qwen3_think (card didn't name it; chatml_simple = alt)
  vibethinker-3b)      CACHE=4096 EXTERNALIZE_EMBEDDER=1 $EXPORT WeiboAI/VibeThinker-3B out/$key templates/chatml_simple.jinja BOCTAV4 ;;  # block32 ONLY; runtime stop-token eos=[151643,151645]
  falcon3-3b)          CACHE=2048 $EXPORT tiiuae/Falcon3-3B-Instruct out/$key templates/falcon_simple.jinja BMIX4_128 ;;  # ship withheld/private (int4≠parity)
  llama32-3b)          EXTERNALIZE_EMBEDDER=1 CACHE=4096 $EXPORT meta-llama/Llama-3.2-3B-Instruct out/$key templates/llama_simple.jinja BMIX4 ;;  # shipped via official litert-torch main

  # ---- needs a PREP step (extract text decoder / patch config / GPTQ ingest) ----
  ministral3-3b)       # Mistral3 multimodal -> extract text decoder first (generic extractor, same as reasoning)
    [ -f $SM/ministral3-3b-text/config.json ] || $PY scripts/extract_text_backbone.py mistralai/Ministral-3-3B-Instruct-2512 $SM/ministral3-3b-text
    EXTERNALIZE_EMBEDDER=1 CACHE=4096 $EXPORT $SM/ministral3-3b-text out/$key templates/mistral_simple.jinja BOCTAV4 ;;
  ministral3-3b-reasoning)
    [ -d $SM/ministral-3-3b-reasoning-text ] || $PY scripts/extract_text_backbone.py mistralai/Ministral-3-3B-Reasoning-2512 $SM/ministral-3-3b-reasoning-text
    EXTERNALIZE_EMBEDDER=1 FORCE_SPM=1 CACHE=4096 $EXPORT $SM/ministral-3-3b-reasoning-text out/$key templates/mistral_simple.jinja BOCTAV4 ;;
  phi4-mini-reasoning) # LongRoPE->static (env) + sliding_window=None (config edit) before export
    dl microsoft/Phi-4-mini-reasoning $SM/phi4-mini-reasoning
    $PY -c "import json;p='$SM/phi4-mini-reasoning/config.json';c=json.load(open(p));c['sliding_window']=None;json.dump(c,open(p,'w'),indent=2)"
    PHI3_STATIC_ROPE=1 EXTERNALIZE_EMBEDDER=1 CACHE=4096 $EXPORT $SM/phi4-mini-reasoning out/$key templates/phi_simple.jinja BOCTAV4 ;;
  qwen25-3b)           # 2-step: dequantize the official GPTQ-int4 ckpt (fp32clip lands the grid on
                       # the +/-7 levels), then BMIX4_128 min-max re-derives those scales exactly.
                       # (The original ship used recipes/gptqrec_int4_block128.json + GPTQREC_GCD_FIX,
                       # but current ai_edge_quantizer rejects blockwise dequant-recovery; BMIX4_128 is
                       # the version-robust path the ingest script's own docstring recommends.)
    [ -d $SM/qwen25-3b-gptq-dequant ] || $PY scripts/ingest_gptq_dequant.py Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4 Qwen/Qwen2.5-3B-Instruct $SM/qwen25-3b-gptq-dequant fp32clip
    CACHE=4096 $EXPORT $SM/qwen25-3b-gptq-dequant out/$key templates/chatml_simple.jinja BMIX4_128 ;;  # personal-namespace only (NC license)

  *) echo "unknown model key: '$key' (run --list)"; return 2 ;;
  esac
  echo "REPRODUCED $key -> out/$key/model.litertlm"
}

KEYS="fastcontext-4b nanbeige4.1-3b olmo2-1b olmo2-7b polaris-4b qwen3-1.7b qwen3-4b-thinking \
r1-distill-qwen-1.5b r1-distill-qwen-7b smollm3-3b jan-nano vibethinker-3b falcon3-3b llama32-3b \
ministral3-3b ministral3-3b-reasoning phi4-mini-reasoning qwen25-3b"

case "${1:-}" in
  --list|"") echo "model keys:"; for k in $KEYS; do echo "  $k"; done;
             echo; echo "see REPRODUCE.md for the full recipe table + caveats" ;;
  --all)     for k in $KEYS; do echo "=== $k ==="; reproduce "$k"; done ;;
  *)         reproduce "$1" ;;
esac
