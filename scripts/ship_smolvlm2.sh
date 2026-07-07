#!/bin/bash
set -e
cd "$(dirname "$0")/.."
PY=~/clipconv/bin/python
SRC=src_models/smolvlm2-500m
LLM=src_models/smolvlm2-500m-llm
DECO=out/smolvlm2-decoder
VISO=out/smolvlm2-vision
G='Loading weights|it/s|Redirects|register_constant|FlashAttention|torch_dtype.*deprecated|KernelPreference|ScaleCalculation|FutureWarning|copyreg|InitializeLog|XNNPACK|arithmetic ops|incorrect regex|fix_mistral'
echo "### 1. vision (SigLIP static-pos + pixel-shuffle x4 + connector -> 64x960)"
$PY scripts/convert_smolvlm2_vision.py "$SRC" "$VISO" 2>&1 | grep -vE "$G" | tail -2
cat "$VISO/result.json" | $PY -c "import sys,json;d=json.load(sys.stdin);print('vision:',{k:d.get(k) for k in ['ok','enc_out','adp_out','enc_ops','adp_ops','end2end_corr','error_head']})"
echo "### 2. extract SmolLM2 (Llama) decoder"
$PY scripts/prep_smolvlm2_decoder.py "$SRC" "$LLM" 2>&1 | grep -vE "$G" | tail -3
echo "### 3. export decoder (BOCTAV4)"
CACHE=2048 PREFILL=128,512 $PY scripts/export_internvl_decoder.py "$LLM" "$DECO" 2>&1 | grep -vE "$G" | tail -3
echo "### 4. SP tokenizer"
$PY - <<PYEOF 2>&1 | grep -vE "$G" | tail -2
from transformers import AutoTokenizer
from litert_torch.generative.tools import tokenizer_to_sentencepiece_lib as tok_spm
tok = AutoTokenizer.from_pretrained("$LLM")
vf = getattr(tok, "vocab_file", None)
if vf and not str(vf).endswith((".model",".spiece",".spm")): tok.vocab_file = None
try:
    open("$DECO/tokenizer.spiece","wb").write(tok_spm.convert(tok)); print("SP tokenizer written")
except Exception as e: print("SP FAILED (will use HF):", type(e).__name__, str(e)[:120])
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
echo "### 6. build fast_vlm bundle"
DEC="$DECO" VIS="$VISO" CACHE=2048 OUT_NAME=SmolVLM2-500M.litertlm $PY scripts/build_smolvlm_bundle.py 2>&1 | tail -2
ls -la out/internvl-bundle/SmolVLM2-500M.litertlm | awk '{print "BUNDLE:", $5/1e9" GB"}'
echo "### DONE"
