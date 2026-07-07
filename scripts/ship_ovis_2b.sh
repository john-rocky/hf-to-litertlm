#!/bin/bash
# Full AIDC-AI/Ovis2.5-2B -> fast_vlm .litertlm pipeline.
# Vision = Siglip2Navit made static-single-512 (ovis_work/ovis_static.py); adapter =
# Ovis visual-tokenizer tail (head->softmax->vte). Decoder = Qwen3-1.7B (hidden 2048),
# same BOCTAV4 recipe as the InternVL3_5-2B / Ministral ships.
set -e
cd ~/code/litertlm-convert
PY=~/clipconv/bin/python
LLM=src_models/ovis2_5-2b-llm
DECO=out/ovis-decoder
VISO=out/ovis-vision
GREP='Loading weights|it/s|Redirects|register_constant|FlashAttention|torch_dtype.*deprecated|KernelPreference|ScaleCalculation|FutureWarning|copyreg|InitializeLog|XNNPACK|arithmetic ops|incorrect regex|fix_mistral|Fetching|MISSING|newly initialized|downstream task'

echo "### 1. extract Qwen3 decoder (model.llm)"
$PY scripts/prep_ovis_decoder.py "$LLM" 2>&1 | grep -vE "$GREP" | tail -5

echo "### 2. export decoder (BOCTAV4 + externalize + single_token_embedder)"
CACHE=2048 PREFILL=128,512 $PY scripts/export_internvl_decoder.py "$LLM" "$DECO" 2>&1 | grep -vE "$GREP" | tail -3

echo "### 3. SP tokenizer (Qwen3 BPE -> sentencepiece)"
$PY - <<PYEOF 2>&1 | grep -vE "$GREP" | tail -2
from transformers import AutoTokenizer
from litert_torch.generative.tools import tokenizer_to_sentencepiece_lib as tok_spm
tok = AutoTokenizer.from_pretrained("$LLM", trust_remote_code=True, use_fast=False)
vf = getattr(tok, "vocab_file", None)
if vf and not str(vf).endswith((".model",".spiece",".spm")): tok.vocab_file = None
open("$DECO/tokenizer.spiece","wb").write(tok_spm.convert(tok))
print("SP tokenizer written")
PYEOF

echo "### 4. vision encoder+adapter (static-512, raster-conv patchify, int8)"
$PY scripts/convert_ovis_vision.py "$VISO" 2>&1 | grep -vE "$GREP" | tail -2
cat "$VISO/result.json" | $PY -c "import sys,json;d=json.load(sys.stdin);print('vision:',{k:d.get(k) for k in ['ok','enc_out','adp_out','end2end_corr','end2end_int8_corr','enc_int8_mb','adp_int8_mb']})"

echo "### 5. build fast_vlm bundle (IMAGE_SIZE=512, ChatML, bare <image_soft_token>)"
DEC="$DECO" VIS="$VISO" TOK=sp IMAGE_SIZE=512 CACHE=2048 \
  IMG_RENDER='<image_soft_token>' OUT_NAME=Ovis2.5-2B.litertlm \
  $PY scripts/build_internvl_bundle.py 2>&1 | tail -3
ls -la out/internvl-bundle/Ovis2.5-2B.litertlm | awk '{print "BUNDLE:", $NF, $5/1e9" GB"}'
echo "### DONE"
