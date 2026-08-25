#!/bin/bash
# Convert a Qwen2-VL-2B DERIVATIVE (full finetune) to a fast_vlm .litertlm.
#
#   bash scripts/ship_qwen2vl_derivative.sh <hf_repo> [<name>]
#   e.g. bash scripts/ship_qwen2vl_derivative.sh numind/NuExtract-2.0-2B
#
# Family fact (measured 2026-08-25 on NuExtract-2.0-2B): Qwen2-VL derivatives
# train BOTH the decoder AND the vision tower (339/391 visual.* tensors differ
# from the base) — so unlike the LLM families, the vision encoder re-exports
# from the derivative's own weights; nothing is reused from the base bundle.
# Same rails as ship_qwen2vl_2b.sh otherwise: static-672 vision (no GATHER_ND),
# decoder re-hosted as Qwen2ForCausalLM (bit-exact, verified at prep), int4
# BOCTAV4 decoder + int8 vision, ChatML jinja with
# <|vision_start|><image_soft_token><|vision_end|>.
#
# Env: PY = python with litert-torch 0.9.3 stack + torchvision
#      (default ~/venvs/ltconv040dev/bin/python).
set -euo pipefail
cd "$(dirname "$0")/.."

REPO=${1:?usage: ship_qwen2vl_derivative.sh <hf_repo> [<name>]}
NAME=${2:-$(basename "$REPO")}
PY=${PY:-$HOME/venvs/ltconv040dev/bin/python}

echo "### 0. download $REPO"
SNAP=$(HF_HUB_DISABLE_XET=1 $PY -c "from huggingface_hub import snapshot_download; print(snapshot_download('$REPO'))")

echo "### 1. extract decoder (bit-exact Qwen2ForCausalLM re-host, verified)"
$PY qwen2vl_work/prep_qwen2vl_decoder.py "$SNAP" "src_models/$NAME-llm" 2>&1 | tail -3

echo "### 2. export decoder (BOCTAV4 int4 + int8 embedder)"
CACHE=4096 PREFILL=128,512,1024 \
  $PY scripts/export_internvl_decoder.py "src_models/$NAME-llm" "out/$NAME-decoder" 2>&1 | tail -2

echo "### 3. SP tokenizer"
OMP_NUM_THREADS=1 $PY - <<PYEOF 2>&1 | tail -1
from transformers import AutoTokenizer
from litert_torch.generative.tools import tokenizer_to_sentencepiece_lib as tok_spm
tok = AutoTokenizer.from_pretrained("src_models/$NAME-llm", trust_remote_code=True, use_fast=False)
vf = getattr(tok, "vocab_file", None)
if vf and not str(vf).endswith((".model",".spiece",".spm")): tok.vocab_file = None
open("out/$NAME-decoder/tokenizer.spiece","wb").write(tok_spm.convert(tok))
print("SP tokenizer written")
PYEOF

echo "### 4. vision encoder+adapter from the DERIVATIVE's weights (static-672, int8)"
IMG=672 MODEL="$SNAP" $PY qwen2vl_work/convert_qwen2vl_vision.py "out/$NAME-vision" 2>&1 | tail -2

echo "### 5. assemble fast_vlm bundle"
DEC="out/$NAME-decoder" VIS="out/$NAME-vision" TOK=sp IMAGE_SIZE=672 CACHE=4096 \
  IMG_RENDER='<|vision_start|><image_soft_token><|vision_end|>' \
  OUT_NAME="$NAME.litertlm" \
  $PY scripts/build_internvl_bundle.py 2>&1 | tail -2
ls -la "out/internvl-bundle/$NAME.litertlm" | awk '{print "BUNDLE:", $NF, $5/1e9" GB"}'
echo "### DONE — gate it (task gate for task-specific derivatives, e.g."
echo "###   $PY qwen2vl_work/gate_nuextract.py out/internvl-bundle/$NAME.litertlm --backend cpu)"
