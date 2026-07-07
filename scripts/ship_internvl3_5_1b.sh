#!/bin/bash
# Full InternVL3_5-1B -> fast_vlm .litertlm pipeline. Adapted from ship_internvl_1b.sh.
# Only difference vs InternVL3-2B: the decoder is Qwen3-1.7B (not Qwen2.5), so use the
# generic prep (real decoder class) and hidden=2048 flows through automatically (vision
# adapter dims come from the model's real projector weights).
set -e
cd ~/code/litertlm-convert
PY=~/clipconv/bin/python
SRC=src_models/internvl3_5-1b
LLM=src_models/internvl3_5-1b-llm
DECO=out/internvl3_5-1b-decoder
VISO=out/internvl3_5-1b-vision
GREP='Loading weights|it/s|Redirects|register_constant|FlashAttention|torch_dtype.*deprecated|KernelPreference|ScaleCalculation|FutureWarning|copyreg|InitializeLog|XNNPACK|arithmetic ops|incorrect regex|fix_mistral|Fetching'

echo "### 0. download OpenGVLab/InternVL3_5-1B"
$PY -c "from huggingface_hub import snapshot_download; snapshot_download('OpenGVLab/InternVL3_5-1B', local_dir='$SRC', ignore_patterns=['*.pth','*.bin.index.json.bak'])" 2>&1 | grep -vE "$GREP" | tail -2

echo "### 1. extract Qwen3 decoder (generic prep)"
$PY scripts/prep_internvl_decoder_generic.py "$SRC" "$LLM" 2>&1 | grep -vE "$GREP" | tail -5

echo "### 2. export decoder (BOCTAV4 + externalize + single_token_embedder)"
CACHE=2048 PREFILL=128,512 $PY scripts/export_internvl_decoder.py "$LLM" "$DECO" 2>&1 | grep -vE "$GREP" | tail -3

echo "### 3. SP tokenizer"
$PY - <<PYEOF 2>&1 | grep -vE "$GREP" | tail -2
from transformers import AutoTokenizer
from litert_torch.generative.tools import tokenizer_to_sentencepiece_lib as tok_spm
tok = AutoTokenizer.from_pretrained("$LLM", trust_remote_code=True, use_fast=False)
vf = getattr(tok, "vocab_file", None)
if vf and not str(vf).endswith((".model",".spiece",".spm")): tok.vocab_file = None
open("$DECO/tokenizer.spiece","wb").write(tok_spm.convert(tok))
print("SP tokenizer written")
PYEOF

echo "### 4. vision encoder+adapter (4D-clean GPU4D)"
GPU4D=1 $PY scripts/convert_internvl_vision_split.py "$SRC" "$VISO" 2>&1 | grep -vE "$GREP" | tail -2
cat "$VISO/result.json" | $PY -c "import sys,json;d=json.load(sys.stdin);print('vision:',{k:d.get(k) for k in ['ok','enc_out','adp_out','end2end_corr']})"

echo "### 5. quantize vision int8"
$PY - <<PYEOF 2>&1 | tail -3
from ai_edge_quantizer import quantizer
import ai_edge_quantizer.recipe as r
import os
for name in ["vision_encoder","vision_adapter"]:
    q=quantizer.Quantizer(f"$VISO/{name}.tflite", r.dynamic_wi8_afp32())
    q.quantize().export_model(f"$VISO/{name}_int8.tflite")
    print(name,"int8",round(os.path.getsize(f"$VISO/{name}_int8.tflite")/1e6,1),"MB")
PYEOF

echo "### 6. build fast_vlm bundle"
DEC="$DECO" VIS="$VISO" TOK=sp OUT_NAME=InternVL3_5-1B.litertlm $PY scripts/build_internvl_bundle.py 2>&1 | tail -2
ls -la out/internvl-bundle/InternVL3_5-1B.litertlm | awk '{print "BUNDLE:", $NF, $5/1e9" GB"}'
echo "### DONE"
