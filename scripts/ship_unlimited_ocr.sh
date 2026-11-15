#!/bin/bash
# baidu/Unlimited-OCR (3.3B OCR VLM, MIT: DeepEncoder vision + a 12-layer DeepSeek-V2 MoE decoder whose attention keeps
# the prompt and a ring of the last 128 generated positions) -> a fast_vlm file that reads one 1024 x 1024 page:
#   Unlimited-OCR_int8.litertlm   decoder with int8 fully connected layers (the position table float), its KV cache =
#                                 the 277 prompt positions pinned + a ring of the last 128 (405 slots for a 4,096-token
#                                 context), prefill_278 + decode, fp32 activation preference; int8 embedder; vision
#                                 encoder and adapter with fp16 weights; the decoder's and the embedder's weights in
#                                 TFLiteWeights sections at the end of the file
#
# Vision   = unlimitedocr1_work/vision_export.py with uocr_vision.py: the Unlimited-OCR-2 files (the two checkpoints'
#            vision towers have the same structure, each with its own weights; this uocr_vision.py also reads a
#            checkpoint without the CLIP position_ids tensor), checked against this checkpoint's own deepencoder.py
#            classes on four document pictures; the copies with fp16 weights go into the bundle.
# Decoder  = unlimitedocr1_work/export_ring1.py: uocr1_decoder.py writes the model's own decode attention (the prompt's
#            277 positions pinned, the last 128 generated positions in a ring, plain multi-head attention, 1-D RoPE from
#            a constant table) around the MLP / MoE of unlimitedocr2_work/uocr_decoder.py; prefill_278 (the 277-id
#            prompt in one prefill call) + decode; the embedder in its own graph. quant_decoder.py --ram: int8
#            channelwise on every decoder fully connected layer but the position table, and on the embedding table.
# Bundle   = unlimitedocr2_work/build_bundle.py: fast_vlm 1024 x 1024, max_num_tokens 4,096, start token 0 (BOS), stop
#            token 1 (EOS), the template uocr1_fixed.jinja (the picture, then the author's instruction "document
#            parsing."), the repository's tokenizer.json, and an ExecutorMetadata section that declares the 24 KV
#            caches LOCAL (unlimitedocr2_work/executor_pbtext.py); the read-back through litert-lm unpack.
# Weights  = tools/externalize/externalize_repo.sh --no-hub --keep-float-inline: the int8 weights of the decoder and
#            the embedder move into TFLiteWeights sections (LiteRT's tools at 8761012b, litert-lm-builder 0.19.1), every
#            other section is copied byte for byte, and one CPU run of the result must answer.
#
# Environments: PY_EXPORT, PY_BUNDLE and PY_XR (absolute paths to the python of each), or none of the three:
# unlimitedocr1_work/make_envs.sh then makes them under ENVS (default OUT/envs) from the frozen requirements files of
# that folder (PYTHON314 / PYTHON312 = the interpreters to make them from; default python3.14 / python3.12).
# Disk: about 33 GB under OUT besides the environments (the 11.1 GB fp32 decoder stays in OUT/work) and 6.7 GB in the
# Hugging Face cache. Memory: the decoder export peaks near 40 GB.
#
#   bash scripts/ship_unlimited_ocr.sh [OUT]        # OUT defaults to out/unlimitedocr1
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)
OUT=$(mkdir -p "${1:-out/unlimitedocr1}" && cd "${1:-out/unlimitedocr1}" && pwd)
W=$OUT/work
L=$ROOT/unlimitedocr1_work
L2=$ROOT/unlimitedocr2_work
mkdir -p "$W"
export PYTHONHASHSEED=0 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1

if [ -z "${PY_EXPORT:-}${PY_BUNDLE:-}${PY_XR:-}" ]; then
  echo "### environments from unlimitedocr1_work/requirements-*.txt"
  ENVLINE=$(bash "$L/make_envs.sh" "${ENVS:-$OUT/envs}" | tee /dev/stderr | tail -1)
  eval "$ENVLINE"
fi
: "${PY_EXPORT:?}" "${PY_BUNDLE:?}" "${PY_XR:?}"
XB=$(dirname "$PY_XR")
HF=${HF:-$(dirname "$PY_EXPORT")/hf}

echo "### 0. source checkpoint (revision 8249d20b) and four example pictures of Xiaomi-OCR-0 (revision bbd8cf8d)"
SRC=$(HF_HUB_DISABLE_XET=1 "$HF" download baidu/Unlimited-OCR --revision 8249d20b2822a7d8a4c22c8d8c9069b5ac36bda3 \
  --include '*.json' --include '*.py' --include 'model-00001-of-000001.safetensors' | tail -1)
SRC=${SRC#path=}   # huggingface_hub 2.x prints "path=<dir>" outside a terminal, 1.x the bare directory
[ -f "$SRC/config.json" ] || { echo "no checkpoint at '$SRC'"; exit 1; }
PICS=$(HF_HUB_DISABLE_XET=1 "$HF" download xiaomi-research/Xiaomi-OCR-0 --revision bbd8cf8d21f256b3644b44cb860f3a7cecb93c30 \
  --include 'example_pics/receipt.jpg' --include 'example_pics/dense-equations.jpg' \
  --include 'example_pics/handwrite-formula.jpg' --include 'example_pics/qianziwen.jpeg' | tail -1)
PICS=${PICS#path=}
[ -f "$PICS/example_pics/receipt.jpg" ] || { echo "no example pictures at '$PICS'"; exit 1; }
echo "SRC=$SRC"

echo "### 1. vision 1024 x 1024: the TFLite pair (fp32 parity against the checkpoint's deepencoder.py), fp16 weights"
MODEL="$SRC" FIXTURES="$PICS/example_pics" \
  FIXTURE_FILES=receipt.jpg,dense-equations.jpg,handwrite-formula.jpg,qianziwen.jpeg \
  "$PY_EXPORT" "$L/vision_export.py" "$W/vision" --stage all

echo "### 2. decoder: the prompt pinned + a ring of 128 (405 KV slots, context 4,096), prefill_278 + decode; embedder"
MODEL="$SRC" "$PY_EXPORT" "$L/export_ring1.py" "$W/decoder" --prefill 278 --ctx 4096 --embedder

echo "### 3. int8: every decoder FC channelwise (the RopeTable float), the embedding table channelwise"
"$PY_EXPORT" "$L/quant_decoder.py" "$W/decoder/decoder_fp32.tflite" "$W/decoder/decoder_int8.tflite" --ram \
  --form int8 --block 32 --json "$W/decoder/quant_int8.json"
"$PY_EXPORT" "$L/quant_decoder.py" "$W/decoder/embedder_fp32.tflite" "$W/decoder/embedder_int8.tflite" --ram \
  --form emb8 --json "$W/decoder/quant_emb8.json"
"$PY_EXPORT" "$L2/executor_pbtext.py" "$W/decoder/quant_int8.json" "$W/executor_local.pbtext" --kv-type local

echo "### 4. the bundle, then the read-back through litert-lm unpack"
B=$W/Unlimited-OCR_int8_inline.litertlm
"$PY_BUNDLE" "$L2/build_bundle.py" "$B" \
  --embedder "$W/decoder/embedder_int8.tflite" --decoder "$W/decoder/decoder_int8.tflite" \
  --tokenizer "$SRC/tokenizer.json" --jinja "$L/uocr1_fixed.jinja" --type fast_vlm --image-size 1024 \
  --vision-encoder "$W/vision/vision_encoder_fp16.tflite" --vision-adapter "$W/vision/vision_adapter_fp16.tflite" \
  --cache 4096 --executor "$W/executor_local.pbtext" < /dev/null
"$PY_BUNDLE" "$L2/readback.py" "$B" --unpack-dir "$W/readback" \
  --tok-sha a02f8fd5228c90256bb4f6554c34a579d48f909e5beb232dc4afad870b55a8b4 --template "$L/uocr1_fixed.jinja" \
  --kv 12 --kv-type local --kv-shapes '1,10,405,128;1,10,128,405' --mask-ctx 4096 --image 1024 \
  --litert-lm "$(dirname "$PY_BUNDLE")/litert-lm" --tflite-python "$PY_EXPORT" --out "$W/readback.json" < /dev/null
"$PY_BUNDLE" -c 'import json, sys; s = json.load(open(sys.argv[1]))["prefill_decode_signatures"]; print("signatures", s)
sys.exit(0 if sorted(s) == ["decode", "prefill_278"] else 1)' "$W/readback.json"

echo "### 5. external weights: the int8 weights into TFLiteWeights sections, then one CPU run"
XR_PY="$PY_XR" bash "$ROOT/tools/externalize/fetch_tools.sh" "$W/litert-tools"
XR_PY="$PY_XR" XR_LITERT_LM="$XB/litert-lm" XR_BUILDER="$XB/litert-lm-builder" XR_PEEK="$XB/litert-lm-peek" \
  XR_TOOLS_PATH="$W/litert-tools" bash "$ROOT/tools/externalize/externalize_repo.sh" litert-community/Unlimited-OCR \
  Unlimited-OCR_int8.litertlm --source "$B" --no-hub --keep-float-inline \
  --smoke-attachment "$PICS/example_pics/receipt.jpg" --workdir "$W/externalize" --staging "$OUT" < /dev/null

echo "### 6. the file and its sections"
"$PY_XR" "$L/sections.py" "$OUT/Unlimited-OCR_int8.litertlm" --json "$W/sections.json"
