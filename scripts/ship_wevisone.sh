#!/bin/bash
# tencent/WeVisOne-VL-2B (2.44B VLM, apache-2.0) -> WeVisOne-VL-2B_f16.litertlm of litert-community/WeVisOne-VL-2B:
#   fast_vlm at a fixed 512x512 input (256 image tokens), fp16 weights in every graph (decoder, embedder, vision encoder,
#   vision adapter), no prefer_activation_type declared; the tokenizer section = the repo's own tokenizer.json.
#
# DeepStack kept exact (docs/deepstack-wide-embedding.md): the adapter emits 8196-wide image rows (the main embedding +
# the three DeepStack features + an image flag), the decoder adds each DeepStack part after layers 0 / 1 / 2, and
# computes Qwen3-VL's M-RoPE positions in the graph from the runtime's 1-D position and the flag
# (wevisone_work/tools/wv_patch.py, installed in the export process only).
#
# Environments (absolute paths, given by the caller):
#   PY_REF     Python 3.12: torch 2.11.0, transformers 5.12.1, safetensors 0.8.0, numpy 2.2.6, Pillow 12.2.0, tokenizers
#              (text checkpoint, vision constants, the drawn test pictures)
#   PY_EXPORT  Python 3.12: litert-torch 0.10.0, litert-converter 0.5.0, ai-edge-litert 2.3.0, ai-edge-quantizer 0.10.0,
#              litert-lm-builder 0.18.1, torch 2.14.1, transformers 5.14.1, jinja2
#   LITERT_LM  the litert-lm 0.18.1 CLI (describe / unpack / pack)
#
#   bash scripts/ship_wevisone.sh      # -> wevisone_work/out/r2/bundle/WeVisOne-VL-2B_f16.litertlm
#   FP32=1 bash scripts/ship_wevisone.sh   # also WeVisOne-VL-2B_fp32.litertlm, the float32 form for desktop CPUs
#   INT8=1 bash scripts/ship_wevisone.sh   # also WeVisOne-VL-2B_int8.litertlm: int8 weights in the decoder and the
#                                          #   token embedder, the same float16 vision graphs, no activation type declared
#   F16=0 INT8=1 bash scripts/ship_wevisone.sh   # the int8 file only (skips the float16 decoder and its file)
set -euo pipefail
cd "$(dirname "$0")/../wevisone_work"
: "${PY_REF:?}" "${PY_EXPORT:?}" "${LITERT_LM:?}"
REV=3dad9ce8bcb6fbe0aea945bf8860bd2cfc3663bf
NAME=WeVisOne-VL-2B_f16

echo "### 0. source checkpoint (revision 3dad9ce8, without the pixel decoder)"
SNAP=$(HF_HUB_DISABLE_XET=1 hf download tencent/WeVisOne-VL-2B --revision $REV --exclude "mask_decoder.pt" | tail -1)
export WEVISONE_SNAPSHOT="$SNAP"
echo "SNAP=$SNAP"

echo "### 1. the drawn test pictures and the NASA photos"
bash fixtures/fetch_public.sh "$PY_REF"

echo "### 2. text-only checkpoint derived from the VLM"
PY_REF="$PY_REF" bash tools/text_ckpt.sh

echo "### 3. vision tower at 512x512: constants, fp32 export with the published LayerNorm scales, fp16"
export LN_SCALES="$(pwd)/ln_scales_512.json"
"$PY_REF" -I tools/vision_taps.py consts
"$PY_EXPORT" -I tools/wv_vision_r2.py export
"$PY_EXPORT" -I tools/wv_vision_r2.py quant

echo "### 4. decoder: fp32 export (wide input, image flag, derived M-RoPE), then fp16 weights"
PYTHONHASHSEED=0 "$PY_EXPORT" -I tools/wv_export_decoder.py --out out/r2/dec_fp32
if [ "${F16:-1}" = 1 ]; then
  "$PY_EXPORT" -I tools/wv_quant_decoder.py fp16 --src out/r2/dec_fp32 --dst out/r2/dec_fp16
fi

if [ "${F16:-1}" = 1 ]; then
echo "### 5. the .litertlm, its executor metadata, the section check"
"$PY_EXPORT" -I tools/build_wv_bundle.py --build --name $NAME \
  --decoder out/r2/dec_fp16/prefill_decode_fp16.tflite --embedder out/r2/dec_fp16/embedder_fp16.tflite \
  --encoder out/r2/vision/vision_encoder_fp16.tflite --adapter out/r2/vision/vision_adapter_fp16.tflite \
  --act "" --image-size 512
python3 ../scripts/add_executor_metadata.py out/r2/bundle/${NAME}_noexec.litertlm out/r2/bundle/$NAME.litertlm \
  --litert-lm "$LITERT_LM" --python "$PY_EXPORT"
LITERT_LM="$LITERT_LM" "$PY_EXPORT" -I tools/wv_bundle_check.py $NAME
rm -f out/r2/bundle/${NAME}_noexec.litertlm
ls -l out/r2/bundle/$NAME.litertlm
fi

if [ "${FP32:-0}" = 1 ]; then
  echo "### 6. (FP32=1) the float32 form: the fp32 graphs of steps 3 and 4, PREFILL_DECODE declared fp32"
  F=WeVisOne-VL-2B_fp32
  "$PY_EXPORT" -I tools/build_wv_bundle.py --build --name $F \
    --decoder out/r2/dec_fp32/model.tflite --embedder out/r2/dec_fp32/embedder.tflite \
    --encoder out/r2/vision/vision_encoder_fp32.tflite --adapter out/r2/vision/vision_adapter_fp32.tflite \
    --act fp32 --image-size 512
  python3 ../scripts/add_executor_metadata.py out/r2/bundle/${F}_noexec.litertlm out/r2/bundle/$F.litertlm \
    --litert-lm "$LITERT_LM" --python "$PY_EXPORT"
  LITERT_LM="$LITERT_LM" "$PY_EXPORT" -I tools/wv_bundle_check.py $F
  rm -f out/r2/bundle/${F}_noexec.litertlm
  ls -l out/r2/bundle/$F.litertlm
fi

if [ "${INT8:-0}" = 1 ]; then
  echo "### 7. (INT8=1) the int8 form: step 4's fp32 export with int8 weights (dynamic_wi8_afp32) in the decoder and"
  echo "###    the token embedder, the float16 vision graphs of step 3, no activation type declared"
  I=WeVisOne-VL-2B_int8
  "$PY_EXPORT" -I tools/wv_quant_decoder.py wi8 --src out/r2/dec_fp32 --dst out/r2/dec_wi8
  "$PY_EXPORT" -I tools/build_wv_bundle.py --build --name $I \
    --decoder out/r2/dec_wi8/prefill_decode_wi8.tflite --embedder out/r2/dec_wi8/embedder_wi8.tflite \
    --encoder out/r2/vision/vision_encoder_fp16.tflite --adapter out/r2/vision/vision_adapter_fp16.tflite \
    --act "" --image-size 512
  python3 ../scripts/add_executor_metadata.py out/r2/bundle/${I}_noexec.litertlm out/r2/bundle/$I.litertlm \
    --litert-lm "$LITERT_LM" --python "$PY_EXPORT"
  LITERT_LM="$LITERT_LM" "$PY_EXPORT" -I tools/wv_bundle_check.py $I
  rm -f out/r2/bundle/${I}_noexec.litertlm
  ls -l out/r2/bundle/$I.litertlm
fi
