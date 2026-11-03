#!/bin/bash
# reproduce.sh SRC_DIR WORK_DIR — the two GPU files of litert-community/MiniCPM-V-4
# (MiniCPM-V-4-int8-gpu.litertlm, MiniCPM-V-4-mixed-int4-gpu.litertlm) from openbmb/MiniCPM-V-4 at revision 6f69a823.
#   SRC_DIR   a download of openbmb/MiniCPM-V-4 at 6f69a823 (hf download openbmb/MiniCPM-V-4 --revision 6f69a8235885be89608f3a71ffd40695b379e1e7 --local-dir SRC_DIR)
#   WORK_DIR  about 40 GB free (the fp32 export alone is 13.7 GB)
# Env: PY = Python 3.12 with litert-torch 0.10.0, ai-edge-quantizer 0.10.0, ai-edge-litert 2.3.0, torch 2.14.1,
#      transformers 5.20.2, litert-lm-builder 0.19.0 (default: python3);
#      LT = the bin directory of a venv with the litert-lm 0.19.0 CLI (litert-lm, litert-lm-builder, litert-lm-peek);
#      XR_TOOLS_PATH = LiteRT's externalize tools at 8761012b (../tools/externalize/README.md, fetch_tools.sh).
# The README of this directory says what each step makes and how it was checked.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(dirname "$HERE")
SRC=$(cd "${1:?SRC_DIR: openbmb/MiniCPM-V-4 at 6f69a823}" && pwd)
mkdir -p "${2:?WORK_DIR}"
W=$(cd "$2" && pwd)
PY=${PY:-python3}
LT=${LT:?LT = the bin directory of the litert-lm 0.19.0 CLI}
SHIM=$ROOT/muse_glimmer_work/shim   # stands in for the TensorFlow schema module export_hf imports
D=$W/dec_fp32
mkdir -p "$D" "$W/vision" "$W/bundles"

echo "### 1. text checkpoint (the decoder's weights and config as a minicpmv4_text model)"
"$PY" "$HERE/make_text_dir.py" "$SRC" "$W/textdir"

echo "### 2. fp32 export: embedder in its own graph, prefill 1024 / 256 / 64 / 16, KV cache 4096"
PYTHONHASHSEED=0 PYTHONPATH="$SHIM" "$PY" "$HERE/text_export.py" "$W/textdir" "$D" \
  --prefill 1024,256,64,16 --cache 4096 --quant none

echo "### 3. quantization"
PYTHONPATH="$SHIM" "$PY" "$HERE/quant.py" "$D/model.tflite" "$D/model_wi8.tflite" wi8 --json "$D/quant_wi8.json"
PYTHONPATH="$SHIM" "$PY" "$HERE/quant.py" "$D/embedder.tflite" "$D/embedder_emb8.tflite" emb8 --json "$D/quant_emb8.json"
PYTHONPATH="$SHIM" "$PY" "$HERE/quant_mixed.py" "$D/model.tflite" "$D/model_wi4b32fd16.tflite" \
  --fp16-fc 'Linear_down_proj;' --json "$D/quant_wi4b32fd16.json"

echo "### 4. GPU composites (cache update, runtime batch matmul) added to both decoders, weights unchanged"
for n in wi8 wi4b32fd16; do
  "$PY" "$ROOT/tools/gpu_graph/gpu_graph_retrofit.py" "$D/model_$n.tflite" "$D/model_${n}_gpu.tflite" \
    --decomp exporter --mask add_bcast --report "$D/retrofit_$n.json"
done

echo "### 5. vision encoder + resampler (fp16 weights)"
PYTHONPATH="$SHIM" "$PY" "$HERE/v4_vision.py" export --src "$SRC" --out-dir "$W/vision" --quant fp16

echo "### 6. bundles, then the weights moved to TFLiteWeights sections (+ ExecutorMetadata, a CPU smoke on a photo)"
[ -f "$W/cats.jpg" ] || curl -sL -o "$W/cats.jpg" http://images.cocodataset.org/val2017/000000039769.jpg
for pair in int8:wi8 mixed-int4:wi4b32fd16; do
  name=${pair%%:*}; dec=${pair#*:}
  "$PY" "$HERE/build_bundle.py" --decode "$D/model_${dec}_gpu.tflite" --emb "$D/embedder_emb8.tflite" \
    --venc "$W/vision/vision_encoder_fp16.tflite" --tok "$SRC/tokenizer.json" \
    --out "$W/bundles/pre_$name.litertlm" --max-tokens 4096
  XR_PY=$PY XR_LITERT_LM=$LT/litert-lm XR_BUILDER=$LT/litert-lm-builder XR_PEEK=$LT/litert-lm-peek \
    XR_WORK_ROOT=$W/xr_work XR_STAGING_ROOT=$W/xr_staging \
    bash "$ROOT/tools/externalize/externalize_repo.sh" litert-community/MiniCPM-V-4 "MiniCPM-V-4-$name-gpu.litertlm" \
      --source "$W/bundles/pre_$name.litertlm" --no-hub --executor-metadata auto --smoke-attachment "$W/cats.jpg"
done
ls -l "$W/xr_staging/litert-community__MiniCPM-V-4/"*.litertlm
