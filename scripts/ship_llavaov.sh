#!/bin/bash
# Full LLaVA-OneVision-0.5B -> fast_vlm .litertlm pipeline (different-family proof).
set -e
cd "$(dirname "$0")/.."
PY=~/clipconv/bin/python
SRC=src_models/llava-ov-0.5b
LLM=src_models/llava-ov-0.5b-llm
DECO=out/llavaov-decoder
VISO=out/llavaov-vision
G='Loading weights|it/s|Redirects|register_constant|FlashAttention|torch_dtype.*deprecated|KernelPreference|ScaleCalculation|FutureWarning|copyreg|InitializeLog|XNNPACK|arithmetic ops|incorrect regex|fix_mistral'

echo "### 1. vision encoder+adapter (SigLIP + projector + image_newline)"
$PY scripts/convert_llavaov_vision.py "$SRC" "$VISO" 2>&1 | grep -vE "$G" | tail -2
cat "$VISO/result.json" | $PY -c "import sys,json;d=json.load(sys.stdin);print('vision:',{k:d.get(k) for k in ['ok','enc_out','adp_out','enc_ops','adp_ops','end2end_corr','error_head']})"

echo "### 2. extract Qwen2-0.5B decoder"
$PY scripts/prep_llavaov_decoder.py "$SRC" "$LLM" 2>&1 | grep -vE "$G" | tail -3

echo "### 3. export decoder (BOCTAV4 + externalize + single_token_embedder)"
CACHE=2048 PREFILL=128,1024 $PY scripts/export_internvl_decoder.py "$LLM" "$DECO" 2>&1 | grep -vE "$G" | tail -3

echo "### 4. SP tokenizer"
$PY - <<PYEOF 2>&1 | grep -vE "$G" | tail -2
from transformers import AutoTokenizer
from litert_torch.generative.tools import tokenizer_to_sentencepiece_lib as tok_spm
tok = AutoTokenizer.from_pretrained("$LLM")
vf = getattr(tok, "vocab_file", None)
if vf and not str(vf).endswith((".model",".spiece",".spm")): tok.vocab_file = None
open("$DECO/tokenizer.spiece","wb").write(tok_spm.convert(tok))
print("SP tokenizer written")
PYEOF

echo "### 5. quantize vision int8"
$PY - <<PYEOF 2>&1 | tail -3
from ai_edge_quantizer import quantizer
import ai_edge_quantizer.recipe as r
import os
for name in ["vision_encoder","vision_adapter"]:
    quantizer.Quantizer(f"$VISO/{name}.tflite", r.dynamic_wi8_afp32()).quantize().export_model(f"$VISO/{name}_int8.tflite")
    print(name,"int8",round(os.path.getsize(f"$VISO/{name}_int8.tflite")/1e6,1),"MB")
PYEOF

echo "### 6. build fast_vlm bundle (384, bare <image_soft_token>)"
DEC="$DECO" VIS="$VISO" TOK=sp IMAGE_SIZE=384 IMG_RENDER='<image_soft_token>' OUT_NAME=llava-onevision-qwen2-0.5b.litertlm $PY scripts/build_internvl_bundle.py 2>&1 | tail -2
ls -la out/internvl-bundle/llava-onevision-qwen2-0.5b.litertlm | awk '{print "BUNDLE:", $5/1e9" GB"}'
echo "### DONE"
