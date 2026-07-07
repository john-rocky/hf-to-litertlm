#!/bin/bash
# Full PaddlePaddle/PaddleOCR-VL-1.6 -> fast_vlm .litertlm pipeline (9th VLM,
# 1st document-parsing/OCR VLM on litert-community).
#
# Vision = NaViT-style SigLIP made static-560 (raster patch-conv, precomputed
# interp pos-embed + 2D rope, full attn; GPU-safe 2x2 merge in the adapter).
# Decoder = ERNIE-4.5-0.3B re-hosted as LlamaForCausalLM (bit-exact), fp16
# weights (WF16) — int4/int8 corrupt transcription on a 0.36B decoder and the
# int8+FLOAT-compute graph deadlocks the runtime.
#
# ⚠ transformers 5.12 gotcha (baked into convert_paddleocr_vision.py): remote-
# code SigLIPRotaryEmbedding's non-persistent inv_freq loads as ZEROS (meta-init
# + no original_inv_freq) -> rope silently identity. rope_init() after load.
set -e
cd ~/code/litertlm-convert
PY=.venv/bin/python
LLM=src_models/paddleocr-vl-1.6-llm
DECO=out/paddleocr-decoder-f16
VISO=out/paddleocr-vision

echo "### 0. download"
[ -f src_models/paddleocr-vl-1.6/model.safetensors ] || \
  .venv/bin/hf download PaddlePaddle/PaddleOCR-VL-1.6 --local-dir src_models/paddleocr-vl-1.6

echo "### 1. extract ERNIE decoder as LlamaForCausalLM (bit-exact wrap)"
$PY paddleocr_work/prep_paddleocr_decoder.py 2>&1 | tail -3

echo "### 2. export decoder (WF16 fp16 weights + externalize + single_token_embedder)"
RECIPE=WF16 CACHE=4096 PREFILL=128,512,1024 \
  $PY scripts/export_internvl_decoder.py "$LLM" "$DECO" 2>&1 | tail -2

echo "### 3. vision encoder+adapter (static-560, rope_init fix, int8)"
$PY paddleocr_work/convert_paddleocr_vision.py "$VISO" 2>&1 | tail -2

echo "### 4. build fast_vlm bundle (PaddleOCR template + added-tokens SP fix)"
DEC="$DECO" VIS="$VISO" IMAGE_SIZE=560 CACHE=4096 OUT_NAME=PaddleOCR-VL-1.6.litertlm \
  $PY paddleocr_work/build_paddleocr_bundle.py 2>&1 | tail -2

echo "### 5. runtime verify (optional: any LiteRT-LM host that loads fast_vlm bundles)"
VLMTEST=${VLMTEST:-$HOME/code/vlmtest/.build/release/vlmtest}
if [ -x "$VLMTEST" ]; then
  "$VLMTEST" out/paddleocr-bundle/PaddleOCR-VL-1.6.litertlm \
    paddleocr_work/testdocs/para.png skip 1 cpu "OCR:" 2>/dev/null | tail -2
  "$VLMTEST" out/paddleocr-bundle/PaddleOCR-VL-1.6.litertlm \
    paddleocr_work/testdocs/table.png skip 1 cpu "Table Recognition:" 2>/dev/null | tail -2
else
  echo "(skipped: set VLMTEST to a LiteRT-LM test CLI; expected outputs are in cards/paddleocr-vl-1.6-litert.md)"
fi
echo "### DONE"
