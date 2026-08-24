#!/bin/bash
# ibm-granite/granite-docling-258M (idefics3, apache-2.0) -> fast_vlm .litertlm
# IBM's document-conversion VLM (the Docling model): page image -> DocTags markup
# (layout + OTSL tables + formulas), convertible to Markdown/HTML with docling-core.
#
# Rides the SmolVLM2 rail UNCHANGED: idefics3_vision == smolvlm_vision (SigLIP-base
# p16-512, 1024 patches, bucketize dynamic positions -> same static-pos monkeypatch),
# connector = pixel-shuffle x4 + Linear 12288->576 (64 soft tokens), inner decoder =
# plain Llama (576h/30L/kv3, granite 100k vocab, tied embedding).
# Wrapper differences only: granite chat format (<|start_of_role|>...<|end_of_text|>),
# HF tokenizer.json (GPT2 byte-BPE + DocTags added tokens; no SP attempt — digit/tag
# fidelity is the product), NO start_token (add_bos_token=False; a prepended BOS
# double-opens the first role turn), stop=<|end_of_text|>. CACHE=4096 (pages run long).
#
# Recipe = WI8_FLOAT (int8 weights, FLOAT compute): int4 and integer-compute int8
# corrupt DocTags structure on this 258M decoder (same family as PaddleOCR-VL).
# Backend = CPU. The GPU delegate rejects the quantized 576x576 FC at kernel init
# (macOS WebGPU and Android OpenCL alike); an fp16 decoder runs on Android OpenCL
# but ~4x slower than CPU.
#
# ⚠ App contract: pre-resize pages to 512x512 BILINEAR before sending. The model's
# single-global-512 mode is resampling-sensitive (present in eager too) — the
# runtime's own downscale from larger images degrades output to a hallucinated page.
set -e
cd "$(dirname "$0")/.."
PY="${PY:-$HOME/venvs/ltconv040dev/bin/python}"
SRC=src_models/granite-docling-258m
LLM=src_models/granite-docling-258m-llm
RECIPE=${RECIPE:-WI8_FLOAT}
DECO=out/docling-decoder-$(echo "$RECIPE" | tr 'A-Z' 'a-z')
VISO=out/docling-vision
G='Loading weights|it/s|Redirects|register_constant|FlashAttention|torch_dtype.*deprecated|KernelPreference|ScaleCalculation|FutureWarning|copyreg|InitializeLog|XNNPACK|arithmetic ops|incorrect regex|fix_mistral|pad_token_id'

echo "### 0. download"
[ -f "$SRC/model.safetensors" ] || \
  $PY -c "from huggingface_hub import snapshot_download; snapshot_download('ibm-granite/granite-docling-258M', local_dir='$SRC')"

echo "### 1. vision (SigLIP static-pos + pixel-shuffle x4 + connector -> 64x576)"
$PY scripts/convert_smolvlm2_vision.py "$SRC" "$VISO" 2>&1 | grep -vE "$G" | tail -2
cat "$VISO/result.json" | $PY -c "import sys,json;d=json.load(sys.stdin);print('vision:',{k:d.get(k) for k in ['ok','enc_out','adp_out','enc_ops','adp_ops','end2end_corr','error_head']})"

echo "### 2. extract granite Llama decoder"
$PY scripts/prep_smolvlm2_decoder.py "$SRC" "$LLM" 2>&1 | grep -vE "$G" | tail -3

echo "### 3. export decoder ($RECIPE, CACHE 4096)"
CACHE=4096 PREFILL=128,512,1024 RECIPE=$RECIPE \
  $PY scripts/export_internvl_decoder.py "$LLM" "$DECO" 2>&1 | grep -vE "$G" | tail -3

echo "### 4. quantize vision int8"
$PY - <<PYEOF 2>&1 | tail -3
from ai_edge_quantizer import quantizer
import ai_edge_quantizer.recipe as r
import os
for name in ["vision_encoder","vision_adapter"]:
    quantizer.Quantizer(f"$VISO/{name}.tflite", r.dynamic_wi8_afp32()).quantize().export_model(f"$VISO/{name}_int8.tflite")
    print(name,"int8",round(os.path.getsize(f"$VISO/{name}_int8.tflite")/1e6,1),"MB")
PYEOF

echo "### 5. build fast_vlm bundle (granite template, HF tokenizer)"
DEC="$DECO" VIS="$VISO" CACHE=4096 IMAGE_SIZE=512 \
  OUT_NAME=granite-docling-258M.litertlm \
  $PY docling_work/build_granite_docling_bundle.py 2>&1 | tail -2

echo "### 6. gate: synthetic 5x6 table page (make it, pre-resize 512 BILINEAR, eyeball the DocTags)"
$PY docling_work/make_table_page.py
$PY -c "from PIL import Image; Image.open('docling_work/table_page.png').resize((512,512), Image.BILINEAR).save('docling_work/table_page_512.png'); print('wrote docling_work/table_page_512.png')"
echo "gate: run the bundle on docling_work/table_page_512.png with the prompt"
echo "  'Convert this page to docling.'  (e.g. litert-lm run out/docling-bundle/granite-docling-258M.litertlm --backend cpu --attachment docling_work/table_page_512.png --prompt 'Convert this page to docling.')"
echo "expect: title 'Quarterly Sales Report 2025', one 5x6 <otsl> grid, 25/25 cells, clean </doctag> stop"
echo "### DONE"
