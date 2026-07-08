#!/bin/bash
# Full Qwen/Qwen2-VL-2B-Instruct -> fast_vlm .litertlm pipeline (10th VLM, 1st
# Qwen2-VL-family general-purpose VLM on litert-community).
#
# Vision = Qwen2-VL ViT made static-672 (Conv3d folded to Conv2d, full attention,
# precomputed 2D rope, GPU-safe 2x2 merge with strided slices — NO GATHER_ND).
# Decoder = Qwen2-1.5B re-hosted as Qwen2ForCausalLM (bit-exact), int4 BOCTAV4.
#
# KEY device gotcha (baked into convert_qwen2vl_vision.py): reordering patches
# into the merger's 2x2-block order with a gather emits GATHER_ND, which the
# mobile GPU delegate cannot compile -> vision executor fails to
# create on-device. Keep patches in raster order through the (permutation-
# equivariant full-attention) encoder + do the 2x2 merge in the adapter with
# strided slices + concat. Resolution/attention size is NOT the issue.
#
# Known limit (documented): the fast_vlm runtime feeds 1D positions (no M-RoPE),
# which preserves describe/VQA/OCR/count but degrades 2D-table cross-cell ranking.
set -e
cd ~/code/litertlm-convert
PY=.venv/bin/python
LLM=src_models/qwen2-vl-2b-llm
DECO=out/qwen2vl-decoder
VISO=out/qwen2vl-vision-nogather

echo "### 0. download"
[ -f src_models/qwen2-vl-2b/config.json ] || \
  .venv/bin/hf download Qwen/Qwen2-VL-2B-Instruct --local-dir src_models/qwen2-vl-2b

echo "### 1. extract Qwen2 decoder (bit-exact Qwen2ForCausalLM wrap)"
$PY qwen2vl_work/prep_qwen2vl_decoder.py 2>&1 | tail -3

echo "### 2. export decoder (BOCTAV4 int4 + externalize + single_token_embedder)"
CACHE=4096 PREFILL=128,512,1024 \
  $PY scripts/export_internvl_decoder.py "$LLM" "$DECO" 2>&1 | tail -2

echo "### 3. SP tokenizer (Qwen2 BPE -> sentencepiece)"
OMP_NUM_THREADS=1 $PY - <<PYEOF 2>&1 | tail -1
from transformers import AutoTokenizer
from litert_torch.generative.tools import tokenizer_to_sentencepiece_lib as tok_spm
tok = AutoTokenizer.from_pretrained("$LLM", trust_remote_code=True, use_fast=False)
vf = getattr(tok, "vocab_file", None)
if vf and not str(vf).endswith((".model",".spiece",".spm")): tok.vocab_file = None
open("$DECO/tokenizer.spiece","wb").write(tok_spm.convert(tok))
print("SP tokenizer written")
PYEOF

echo "### 4. vision encoder+adapter (static-672, Conv3d-fold, NO GATHER_ND, int8)"
IMG=672 $PY qwen2vl_work/convert_qwen2vl_vision.py "$VISO" 2>&1 | tail -2

echo "### 5. build fast_vlm bundle (ChatML, <|vision_start|><image_soft_token><|vision_end|>)"
DEC="$DECO" VIS="$VISO" TOK=sp IMAGE_SIZE=672 CACHE=4096 \
  IMG_RENDER='<|vision_start|><image_soft_token><|vision_end|>' \
  OUT_NAME=Qwen2-VL-2B.litertlm \
  $PY scripts/build_internvl_bundle.py 2>&1 | tail -2
ls -la out/internvl-bundle/Qwen2-VL-2B.litertlm | awk '{print "BUNDLE:", $NF, $5/1e9" GB"}'
echo "### DONE"
