#!/bin/bash
# baidu/Unlimited-OCR-2 (3.34B OCR VLM, MIT: DeepEncoder vision + a 12-layer DeepSeek-V2 MoE decoder with DASH routed
# attention and page2d positions) -> a fast_vlm file that reads one 1024 x 1024 page:
#   Unlimited-OCR-2_int8.litertlm   decoder with int8 fully connected layers (the DASH router q / k and the position
#                                   table float), its KV cache = positions 0-287 pinned + a ring of the last 128
#                                   (416 slots for a 4,096-token context), fp32 activation preference; int8 embedder;
#                                   vision encoder and adapter with fp16 weights
#
# Vision   = unlimitedocr2_work/uocr_vision.py: the checkpoint's SAM ViT-B + CLIP-L + projector for one static
#            1024 x 1024 page (the author's base mode), re-authored in plain torch; vision_export.py exports the
#            encoder / adapter pair, checks it against the checkpoint's own deepencoder.py classes on four document
#            pictures, and writes the fp16-weight copies.
# Decoder  = unlimitedocr2_work/export_ring.py: the 12 layers with the model's DASH attention written statically for
#            one page (uocr_dash.py), the 64 routed experts of each MoE layer as two fully connected ops (all experts
#            computed, the unselected ones weighted 0), the KV cache as 288 pinned + 128 ring slots (uocr_ring.py);
#            prefill_280 + decode; the embedder in its own graph. quant_decoder.py: int8 channelwise.
# Bundle   = unlimitedocr2_work/build_bundle.py: fast_vlm 1024 x 1024, max_num_tokens 4,096, start token 0 (BOS),
#            stop token 1 (EOS), the template uocr2_fixed.jinja (every page = <PAGE> + the picture, then the author's
#            instruction "Multi page parsing."), the repository's tokenizer.json, and an ExecutorMetadata section that
#            declares the 24 KV caches LOCAL (executor_pbtext.py), so LiteRT-LM keeps the 4,096-token context.
#
# Environments (absolute paths, given by the caller):
#   PY_EXPORT   Python 3.14: litert-torch 0.9.3, litert-converter 0.4.0, ai-edge-quantizer 0.9.0, ai-edge-litert 2.2.0,
#               torch 2.13.0, torchvision 0.28.0, transformers 5.14.1, safetensors, pillow, numpy
#   PY_BUNDLE   Python with litert-lm-builder 0.19.0 (the bundle and the read-back)
#   LITERT_LM   a litert-lm 0.19.0 CLI (unpack, for the read-back)
#
#   bash scripts/ship_unlimited_ocr_2.sh [OUT]        # OUT defaults to out/unlimitedocr2
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)
: "${PY_EXPORT:?}" "${PY_BUNDLE:?}" "${LITERT_LM:?}"
OUT=$(mkdir -p "${1:-out/unlimitedocr2}" && cd "${1:-out/unlimitedocr2}" && pwd)
W=$OUT/work
L=$ROOT/unlimitedocr2_work
mkdir -p "$W"
export PYTHONHASHSEED=0

echo "### 0. source checkpoint (revision 04809669) and the four example pictures of Xiaomi-OCR-0 (revision bbd8cf8d)"
SRC=$(HF_HUB_DISABLE_XET=1 hf download baidu/Unlimited-OCR-2 --revision 0480966938291445591f69f08a13b041c073a96a \
  --include '*.json' --include '*.py' --include '*.safetensors' | tail -1)
SRC=${SRC#path=}   # huggingface_hub 2.x prints "path=<dir>" outside a terminal, 1.x the bare directory
[ -f "$SRC/config.json" ] || { echo "no checkpoint at '$SRC'"; exit 1; }
PICS=$(HF_HUB_DISABLE_XET=1 hf download xiaomi-research/Xiaomi-OCR-0 --revision bbd8cf8d21f256b3644b44cb860f3a7cecb93c30 \
  --include 'example_pics/*' | tail -1)
PICS=${PICS#path=}
[ -f "$PICS/example_pics/receipt.jpg" ] || { echo "no example pictures at '$PICS'"; exit 1; }
echo "SRC=$SRC"

echo "### 1. vision 1024 x 1024: the TFLite pair (fp32 parity against the checkpoint's deepencoder.py), fp16 weights"
MODEL="$SRC" FIXTURES="$PICS/example_pics" \
  FIXTURE_FILES=receipt.jpg,dense-equations.jpg,handwrite-formula.jpg,qianziwen.jpeg \
  "$PY_EXPORT" "$L/vision_export.py" "$W/vision" --stage all

echo "### 2. decoder: DASH for one page, pinned + ring KV (416 slots, context 4,096), prefill_280 + decode; embedder"
MODEL="$SRC" "$PY_EXPORT" "$L/export_ring.py" "$W/decoder" --prefill 280 --ctx 4096 --embedder

echo "### 3. int8: every decoder FC channelwise (RopeTable / DashQ / DashK float), the embedding table channelwise"
"$PY_EXPORT" "$L/quant_decoder.py" "$W/decoder/decoder_fp32.tflite" "$W/decoder/decoder_int8.tflite" --form int8 \
  --block 32 --json "$W/decoder/quant_int8.json"
"$PY_EXPORT" "$L/quant_decoder.py" "$W/decoder/embedder_fp32.tflite" "$W/decoder/embedder_int8.tflite" --form emb8 \
  --json "$W/decoder/quant_emb8.json"
"$PY_EXPORT" "$L/executor_pbtext.py" "$W/decoder/quant_int8.json" "$W/executor_local.pbtext" --kv-type local

echo "### 4. the bundle, then the read-back through litert-lm unpack"
"$PY_BUNDLE" "$L/build_bundle.py" "$OUT/Unlimited-OCR-2_int8.litertlm" \
  --embedder "$W/decoder/embedder_int8.tflite" --decoder "$W/decoder/decoder_int8.tflite" \
  --tokenizer "$SRC/tokenizer.json" --jinja "$L/uocr2_fixed.jinja" --type fast_vlm --image-size 1024 \
  --vision-encoder "$W/vision/vision_encoder_fp16.tflite" --vision-adapter "$W/vision/vision_adapter_fp16.tflite" \
  --cache 4096 --executor "$W/executor_local.pbtext" < /dev/null
"$PY_BUNDLE" "$L/readback.py" "$OUT/Unlimited-OCR-2_int8.litertlm" --unpack-dir "$W/readback" \
  --tok-sha c55c165e32c5a67830362a53e1f548c8930dcc8469efa30ff3252a982799c589 --template "$L/uocr2_fixed.jinja" \
  --kv 12 --kv-type local --kv-shapes '1,11,416,128;1,10,128,416' --mask-ctx 4096 --image 1024 \
  --litert-lm "$LITERT_LM" --tflite-python "$PY_EXPORT" --out "$W/readback.json" < /dev/null
ls -l "$OUT/Unlimited-OCR-2_int8.litertlm"
