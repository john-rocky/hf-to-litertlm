#!/bin/bash
# tencent/WeVisOne-VL-4B (VLM on Qwen3-VL, apache-2.0) -> WeVisOne-VL-4B_int4.litertlm of litert-community/WeVisOne-VL-4B
# (main 2fcc2c19; 3,507,874,304 B, sha256 1c14c17a7e8bd688d888300a68e2f7512b255a275a444a3f22bde395cad4b12f):
#   fast_vlm at a fixed 512x512 input (256 image tokens); PREFILL_DECODE with int4 blockwise-32 weights (every block
#   scale floored at 2^-14), prefer_activation_type fp32 and the GPU KV-cache composites of tools/gpu_graph; the token
#   embedder with int8 weights; float16 vision encoder and adapter; the tokenizer section = the repo's own tokenizer.json.
#
# The WeVisOne-VL-2B layout (scripts/ship_wevisone.sh, docs/deepstack-wide-embedding.md) at the 4B widths: the adapter
# emits 10244-wide image rows (the main embedding + the three DeepStack features + an image flag), the decoder adds each
# DeepStack part after layers 0 / 1 / 2 and computes Qwen3-VL's M-RoPE positions in the graph from the runtime's 1-D
# position and the flag (wevisone4b_work/tools/wv_patch.py, installed in the export process only).
# wevisone4b_work/tools/ holds 4B copies of the 2B tools that needed the 4B widths or paths (the first lines of each say
# what changed); the decoder quantizer, the op table and the test pictures are WeVisOne-VL-2B's (wevisone_work/), unchanged.
#
# Environments (absolute paths, given by the caller), the same as scripts/ship_wevisone.sh:
#   PY_REF     Python 3.12: torch 2.11.0, transformers 5.12.1, safetensors 0.8.0, numpy 2.2.6, Pillow 12.2.0, tokenizers
#              (text checkpoint, vision constants, the drawn test pictures)
#   PY_EXPORT  Python 3.12: litert-torch 0.10.0, litert-converter 0.5.0, ai-edge-litert 2.3.0, ai-edge-quantizer 0.10.0,
#              litert-lm-builder 0.18.1, torch 2.14.1, transformers 5.14.1, jinja2
#   LITERT_LM  the litert-lm 0.18.1 CLI (describe / unpack / pack)
#
#   bash scripts/ship_wevisone_4b.sh [PUBLISHED]   # -> wevisone4b_work/out/bundle/WeVisOne-VL-4B_int4.litertlm
#     PUBLISHED        the published file to compare with (argument or env var; default: its recorded section table)
#     BEFORE_RETROFIT  the published file before step 7 (sha256 8643ea9d...; default: its recorded section table)
#
# Every pack writes a new uuid and creation_timestamp into the file header, so a rebuilt file's whole sha256 differs
# from the published one. Step 8 compares section by section (wevisone4b_work/tools/compare_sections.py), and, when a
# file is given, the bytes outside the sections with those two header values swapped. The intermediate files are
# printed next to the published build's sizes and sha256 as they are made (a DIFF there does not stop the run).
# Needs about 60 GB of free disk beside the 19.3 GB snapshot, and about 56 GB of memory for the decoder export.
set -euo pipefail
cd "$(dirname "$0")/../wevisone4b_work"
: "${PY_REF:?}" "${PY_EXPORT:?}" "${LITERT_LM:?}"
PUBLISHED="${1:-${PUBLISHED:-published}}"
BEFORE_RETROFIT="${BEFORE_RETROFIT:-before_retrofit}"
REV=4a7bed92750ac9d8b8a124cede14759036932dc2
NAME=WeVisOne-VL-4B_int4
PRE=${NAME}_before_retrofit
K=$(pwd)

expect() {  # expect FILE BYTES SHA256: the file's size and sha256 next to the published build's
  local b s
  b=$(wc -c < "$1" | tr -d ' ')
  s=$(shasum -a 256 "$1" | cut -d' ' -f1)
  if [ "$b" = "$2" ] && [ "$s" = "$3" ]; then echo "SAME $1 $b $s"; else echo "DIFF $1 $b $s (published build: $2 $3)"; fi
}

echo "### 0. source checkpoint (revision 4a7bed92, without the pixel decoder)"
SNAP=$(HF_HUB_DISABLE_XET=1 hf download tencent/WeVisOne-VL-4B --revision $REV --exclude "mask_decoder.pt" | tail -1)
SNAP=${SNAP#path=}
[ -f "$SNAP/config.json" ] || { echo "no config.json in the snapshot path: $SNAP"; exit 1; }
export WEVISONE_SNAPSHOT="$SNAP"
echo "SNAP=$SNAP"

echo "### 1. the drawn test pictures and the NASA photos (WeVisOne-VL-2B's fixtures; the LayerNorm scales of step 3"
echo "###    were calibrated on three of them)"
bash ../wevisone_work/fixtures/fetch_public.sh "$PY_REF"

echo "### 2. text-only checkpoint derived from the VLM"
PY_REF="$PY_REF" bash tools/text_ckpt.sh
expect out/text_ckpt/model.safetensors 17660980168 c355691da44757724a59b201bb0cd1d378004e2a352f79e65d3d2eebb669cce3

echo "### 3. vision tower at 512x512: constants, fp32 export with the published LayerNorm scales, fp16"
export LN_SCALES="$K/ln_scales_512.json"
"$PY_REF" -I tools/vision_taps.py consts
"$PY_EXPORT" -I tools/wv_vision.py export
"$PY_EXPORT" -I tools/wv_vision.py quant
expect out/vision/vision_encoder_fp32.tflite 1217361552 e097a590dd63d4dcd72cc8127136f79982cc12a80da9831bab3d052528fafe9a
expect out/vision/vision_adapter_fp32.tflite 436348484 55762f4d87155c3f4d04d8b9a07667f162052af53d414ae8a78a444900545936
expect out/vision/vision_encoder_fp16.tflite 611814736 26adbb94502caed3a3ca8d02f73269393593465b6305567b18e01818659750e4
expect out/vision/vision_adapter_fp16.tflite 218245632 88c674a4139bb8c9ed3e91b06baad110f6871f331f8bc1eaf4258a04b95eb793

echo "### 4. decoder: fp32 export (wide input, image flag, derived M-RoPE)"
# python -I ignores PYTHONHASHSEED; the published build ran this step the same way (with -I).
PYTHONHASHSEED=0 "$PY_EXPORT" -I tools/wv_export_decoder.py --out out/dec_fp32
expect out/dec_fp32/model.tflite 16113974576 1874cde7818a194bc9cb6ca731996669356c8d86066b23771d21caf091560324
expect out/dec_fp32/embedder.tflite 1563445860 f126fd52e76ff9fd0a98e65ec89ae1710e8d27bc926dce6af00a13563fedbf8f

echo "### 5. weight forms (WeVisOne-VL-2B's quantizer): int8 (dynamic_wi8_afp32) for the token embedder, int4"
echo "###    blockwise-32 (dynamic_wi4b32_afp32, block scales floored at 2^-14) for PREFILL_DECODE"
"$PY_EXPORT" -I ../wevisone_work/tools/wv_quant_decoder.py wi8 --src "$K/out/dec_fp32" --dst "$K/out/dec_wi8"
"$PY_EXPORT" -I ../wevisone_work/tools/wv_quant_decoder.py wi4b32 --src "$K/out/dec_fp32" --dst "$K/out/dec_wi4b32"
expect out/dec_wi8/embedder_wi8.tflite 392695648 9494be473647957366b44be45f7bb5723a35167fb72ac156cdd32c8a0b1b1c81
expect out/dec_wi8/prefill_decode_wi8.tflite 4133951184 638b0dd3093b068244be30364ff39afb9c1edfb11a9e658da37456c8df34e284
expect out/dec_wi4b32/prefill_decode_wi4b32.tflite 2282076016 9df0e71865add909872f35aeb7fe5f316b2fbe0dfc67f39ffa7415159af809c5

echo "### 6. the .litertlm, its executor metadata, the section check"
"$PY_EXPORT" -I tools/build_wv_bundle.py --build --name $PRE \
  --decoder out/dec_wi4b32/prefill_decode_wi4b32.tflite --embedder out/dec_wi8/embedder_wi8.tflite \
  --encoder out/vision/vision_encoder_fp16.tflite --adapter out/vision/vision_adapter_fp16.tflite \
  --act fp32 --image-size 512
python3 ../scripts/add_executor_metadata.py out/bundle/${PRE}_noexec.litertlm out/bundle/$PRE.litertlm \
  --litert-lm "$LITERT_LM" --python "$PY_EXPORT"
LITERT_LM="$LITERT_LM" "$PY_EXPORT" -I tools/wv_bundle_check.py $PRE
rm -f out/bundle/${PRE}_noexec.litertlm

echo "### 7. GPU KV-cache composites in PREFILL_DECODE (odml.cache_update, odml.runtime_bmm; tools/gpu_graph)"
LITERT_LM_CLI="$LITERT_LM" "$PY_EXPORT" -I ../tools/gpu_graph/gpu_graph_retrofit.py \
  out/bundle/$PRE.litertlm out/bundle/$NAME.litertlm --decomp exporter --mask add_bcast \
  --report out/bundle/retrofit_report.json --work out/bundle/retrofit_work
ls -l out/bundle/$NAME.litertlm

echo "### 8. every section next to the published file's (and the file before step 7 next to its record)"
rc=0
"$PY_EXPORT" -I tools/compare_sections.py out/bundle/$NAME.litertlm "$PUBLISHED" || rc=1
"$PY_EXPORT" -I tools/compare_sections.py out/bundle/$PRE.litertlm "$BEFORE_RETROFIT" || rc=1
exit $rc
