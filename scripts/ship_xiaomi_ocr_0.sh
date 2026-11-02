#!/bin/bash
# xiaomi-research/Xiaomi-OCR-0 (0.87B OCR VLM, apache-2.0, a post-train of Qwen/Qwen3.5-0.8B-Base) -> fast_vlm files that
# read one 1344x768 picture (height x width = 84 x 48 patches of 16 = 1,008 image tokens):
#   Xiaomi-OCR-0_int8.litertlm   the Qwen3.5 hybrid decoder with int8 fully connected layers (the 36 GatedDeltaNet
#                                decay/gate projections float) and an fp32 activation preference; int8 embedder; vision
#                                encoder fp16 with an fp16 activation preference (read by the GPU executor); vision
#                                adapter int8
#   Xiaomi-OCR-0_fp16w.litertlm  with FILES="int8 fp16w": the same, the decoder's fully connected weights fp16
#
# Vision   = the checkpoint's 12-layer ViT (no DeepStack) and 2x2 merger for one static 1344x768 picture
#            (qwen35vl_work/convert_qwen35_vision.py, IMG_H / IMG_W), the fp16-safe LayerNorm pre-scales pinned to
#            xiaomiocr_work/ln_scales_1344x768.json.
# Decoder  = the Qwen3.5 hybrid rail (qwen35vl_work/export_qwen35vl_decoder.py + the litert-torch patch pinned in
#            qwen35_work), run on the checkpoint itself: its text config, safetensors keys and chat template are the base's.
# Bundle   = qwen35vl_work/build_qwen35vl_bundle.py (IMAGE_H / IMAGE_W, VENC_ACT): ChatML with the empty think block, no
#            start token, stop tokens <|endoftext|> 248044 and <|im_end|> 248046, the repository's own tokenizer.json; then
#            scripts/add_executor_metadata.py binds the 48 hybrid state buffers.
#
# Environments (absolute paths, given by the caller):
#   PY_EXPORT   Python 3.12: litert-torch 0.9.2, litert-converter 0.3.0, ai-edge-quantizer 0.8.0, ai-edge-litert 2.1.6,
#               litert-lm-builder 0.15.0, transformers 5.14.1, torch 2.12.1, tokenizers (litert-torch is overridden by the
#               patched checkout below through PYTHONPATH)
#   PY_VISION   Python 3.14: litert-torch 0.9.3, litert-converter 0.4.0, ai-edge-quantizer 0.9.0, ai-edge-litert 2.2.0,
#               transformers 5.14.1, torch 2.13.0, pillow (converter 0.4.0 lowers the 2x2 merge to STRIDED_SLICE; 0.3.x
#               writes GATHER_ND)
#   LITERT_LM   a litert-lm CLI with pack / unpack (0.18.1 was used)
#   FILES       which files (default "int8"; "int8 fp16w" for both)
#
#   bash scripts/ship_xiaomi_ocr_0.sh [OUT]        # OUT defaults to out/xiaomiocr
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)
: "${PY_EXPORT:?}" "${PY_VISION:?}" "${LITERT_LM:?}"
FILES=${FILES:-int8}
OUT=$(mkdir -p "${1:-out/xiaomiocr}" && cd "${1:-out/xiaomiocr}" && pwd)
W=$OUT/work
mkdir -p "$W"

echo "### 0. source checkpoint (revision bbd8cf8d)"
SRC=$(HF_HUB_DISABLE_XET=1 hf download xiaomi-research/Xiaomi-OCR-0 --revision bbd8cf8d21f256b3644b44cb860f3a7cecb93c30 | tail -1)
SRC=${SRC#path=}   # huggingface_hub 2.x prints "path=<dir>" outside a terminal, 1.x the bare directory
[ -f "$SRC/config.json" ] || { echo "no checkpoint at '$SRC'"; exit 1; }
echo "SRC=$SRC"

echo "### 1. litert-torch at the pinned base + the Qwen3.5 hybrid patch"
if [ ! -d qwen35_work/litert-torch-qwen35 ]; then
  git clone https://github.com/john-rocky/litert-torch qwen35_work/litert-torch-qwen35
  git -C qwen35_work/litert-torch-qwen35 fetch --depth 1 origin 115a13607c730c81018bb9789138a3e5e5119e3d
  git -C qwen35_work/litert-torch-qwen35 checkout --detach 115a13607c730c81018bb9789138a3e5e5119e3d
  git -C qwen35_work/litert-torch-qwen35 apply "$ROOT/qwen35_work/qwen35_hybrid_litert_torch.patch"
fi

echo "### 2. decoder: EMBEDDER + PREFILL_DECODE (externalized embedder, prefill 1024/256/64/16/4/1, KV 4096), int8"
mkdir -p "$W/decoder" "$W/tmp"
PREFILL=1024,256,64,16,4,1 CACHE=4096 QWEN35_FLOAT_FC_REGEX='Linear_in_proj_(a|b);' PYTHONHASHSEED=0 \
  PYTHONPATH="$ROOT/qwen35_work/litert-torch-qwen35" TMPDIR="$W/tmp" \
  "$PY_EXPORT" qwen35vl_work/export_qwen35vl_decoder.py "$SRC" "$W/decoder"
if [[ " $FILES " == *" fp16w "* ]]; then
  echo "### 2b. the fp16 weight-only decoder from the same float export (the 36 in_proj_a / in_proj_b stay fp32)"
  "$PY_EXPORT" xiaomiocr_work/quant_fp16w.py "$W/decoder/model.tflite" "$W/decoder/model_fp16w.tflite"
fi

echo "### 3. vision 1344x768: the TFLite pair (fp32 parity against the model's own vision path, int8 parts), then the"
echo "###    fp16 parts and the fp32 / fp16 / int8 correlation on the repository's four example pictures"
IMG_H=1344 IMG_W=768 LN_SCALES=xiaomiocr_work/ln_scales_1344x768.json MODEL="$SRC" \
  "$PY_VISION" qwen35vl_work/convert_qwen35_vision.py "$W/vision"
IMG_H=1344 IMG_W=768 MODEL="$SRC" FIXTURES="$SRC/example_pics" \
  FIXTURE_FILES=receipt.jpg,dense-equations.jpg,handwrite-formula.jpg,qianziwen.jpeg \
  "$PY_VISION" qwen35vl_work/vision_quant_ab.py "$W/vision"

echo "### 4. the bundles: sections + LlmMetadata (fast_vlm 1344x768, stops 248044 / 248046), then the ExecutorMetadata"
echo "###    section that binds the 48 hybrid state buffers"
for f in $FILES; do
  case $f in
    int8)  decode=prefill_decode_wi8 ;;
    fp16w) decode=fp16w ;;
    *) echo "unknown file '$f' (int8 | fp16w)"; exit 2 ;;
  esac
  DEC="$W/decoder" DECODE=$decode EMB=embedder_wi8 VIS="$W/vision" TOK="$SRC/tokenizer.json" \
    VENC=vision_encoder_fp16.tflite VADP=vision_adapter_int8.tflite IMAGE_H=1344 IMAGE_W=768 CACHE=4096 \
    DEC_ACT=fp32 VENC_ACT=fp16 OUT_DIR="$W/bundle" OUT_NAME="pre_$f.litertlm" \
    "$PY_EXPORT" qwen35vl_work/build_qwen35vl_bundle.py
  "$PY_EXPORT" scripts/add_executor_metadata.py "$W/bundle/pre_$f.litertlm" "$OUT/Xiaomi-OCR-0_$f.litertlm" \
    --litert-lm "$LITERT_LM" --python "$PY_EXPORT"
done
ls -l "$OUT"/*.litertlm
