#!/bin/bash
# Spark-X2.6-1.7B / -4B: the four published .litertlm files from the vendor checkpoints (REPRODUCE.md, Spark-X2.6 entry).
#
#   bash spark_work/x26_build.sh 1.7b      # -> out/Spark-X2.6-1.7B_int8.litertlm, out/Spark-X2.6-1.7B_int4-mlp.litertlm
#   bash spark_work/x26_build.sh 4b        # -> out/Spark-X2.6-4B_int8.litertlm, out/Spark-X2.6-4B_int4.litertlm
#   bash spark_work/x26_build.sh post <in.litertlm> <out.litertlm> [fp32]   # only the GPU graph rewrite (+ fp32 item)
#
# Environment (each a path; the defaults are what PATH gives):
#   PY             python with litert-torch 0.9.3, transformers 5.14.1, torch 2.12.1, ai-edge-quantizer 0.8.0 (the export)
#   PYT            python with ai-edge-litert 2.2.0 + litert-lm-builder, 0.16.1 here (graph rewrite, pack checks,
#                  tokenizer swap, stop order)
#   LITERT_LM_CLI  the litert-lm 0.19.0 CLI (unpack / pack / tokenizer swap / the fp32 item of the 1.7B files)
#   LITERT_LM_0170 the litert-lm 0.17.0 CLI (the fp32 repack of the 4B files, scripts/set_activation_type.py)
#   SM             where the checkpoints go (default src_models)
#   hf (the huggingface_hub CLI, 1.27.0 here) on PATH, for the checkpoint download
# Run from the repository root. On an Apple M4 Max (2026-10-31) each 1.7B export took 8 to 22 min and each 4B export
# 15 to 16 min, the spread from the machine's load; the 4B build peaks at about 56 GB (52.5 GiB) of resident memory,
# the 1.7B build at about 25 GB. Each file's last step, spark_work/x26_stop_order.py, writes its stop list in the
# published file's order (the export writes it in Python set order; see that script).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
PYT=${PYT:-$PY}
CLI=${LITERT_LM_CLI:-litert-lm}
CLI17=${LITERT_LM_0170:-}
SM=${SM:-src_models}
J=templates/spark26_think.jinja
export LITERT_LM_CLI=$CLI PYTHONHASHSEED=0

dl() {  # dl <repo> <revision> <dir>
  [ -f "$3/config.json" ] || hf download "$1" --revision "$2" --local-dir "$3" > /dev/null
}

swap() {  # the repo's tokenizer.json bytes into the tokenizer section (the export writes a save_pretrained copy)
  mkdir -p "$(dirname "$3")"
  $PYT tools/tokenizer_parity/swap_tokenizer_section.py "$1" "$2" "$3"
}

post() {  # in.litertlm out.litertlm [fp32]: the GPU-composite graph (KV writes, attention products), weights unchanged
  local in=$1 out=$2 fp32=${3:-} w
  w=$(dirname "$out")/$(basename "$out" .litertlm).work
  rm -rf "$w"; mkdir -p "$(dirname "$w")"
  "$CLI" unpack "$in" --output-dir "$w/unpack" < /dev/null > /dev/null    # the CLI creates the directory itself
  $PYT -B dory_work/retrofit_gpu.py "$w/unpack/Section2_TFLiteModel_tf_lite_prefill_decode.tflite" \
    "$w/prefill_decode_gpug.tflite" "$w/retrofit.json"
  if [ "$fp32" = fp32 ]; then
    $PYT -B spark_work/x26_pack_check.py "$in" "$w/unpack" "$w/prefill_decode_gpug.tflite" "$w/gpug.litertlm" "$w/pack.json"
    $PYT -B spark_work/x26_declare_fp32.py "$w/gpug.litertlm" "$out" "$w/declare.json"
  else
    $PYT -B spark_work/x26_pack_check.py "$in" "$w/unpack" "$w/prefill_decode_gpug.tflite" "$out" "$w/pack.json"
  fi
}

order() {  # in.litertlm <published name>: out/<name>.litertlm = in with the published file's stop order
  $PYT -B spark_work/x26_stop_order.py "$1" "out/$2.litertlm" "$2"
}

case "${1:-}" in
  1.7b)
    dl XHToken/Spark-X2.6-1.7B f18c032c7675f807c0fdffe281dbe124edba2605 $SM/Spark-X2.6-1.7B
    python3 spark_work/patch_modeling.py $SM/Spark-X2.6-1.7B $SM/Spark-X2.6-1.7B-export
    # int8: dynamic int8 everywhere, 11 prefill signatures (convert_spark.py's default ladder)
    $PY spark_work/convert_spark.py $SM/Spark-X2.6-1.7B-export out/spark26-1.7b-int8 $J dynamic_wi8_afp32
    swap out/spark26-1.7b-int8/model.litertlm $SM/Spark-X2.6-1.7B/tokenizer.json out/spark26-1.7b-int8-tok/model.litertlm
    post out/spark26-1.7b-int8-tok/model.litertlm out/spark26-1.7b-int8-gpug/model.litertlm fp32
    order out/spark26-1.7b-int8-gpug/model.litertlm Spark-X2.6-1.7B_int8
    # int4-mlp: attention int8, MLP gate / up int4 blockwise-32 OCTAV, MLP down int8, lm_head int8, embedder in its own
    # section; the fp32 item first, then the graph rewrite (the published file was made in this order)
    Q17_PLAN=attn=8,mlp_gu=4,mlp_down=8,lm_head=8 EXTERNALIZE_EMBEDDER=1 \
      $PY spark_work/x26_q17_export.py $SM/Spark-X2.6-1.7B-export out/spark26-1.7b-q17 $J Q17
    swap out/spark26-1.7b-q17/model.litertlm $SM/Spark-X2.6-1.7B/tokenizer.json out/spark26-1.7b-q17-tok/model.litertlm
    mkdir -p out/spark26-1.7b-q17-fp32
    $PYT -B spark_work/x26_declare_fp32.py out/spark26-1.7b-q17-tok/model.litertlm out/spark26-1.7b-q17-fp32/model.litertlm \
      out/spark26-1.7b-q17-fp32/declare.json
    post out/spark26-1.7b-q17-fp32/model.litertlm out/spark26-1.7b-q17-gpug/model.litertlm
    order out/spark26-1.7b-q17-gpug/model.litertlm Spark-X2.6-1.7B_int4-mlp ;;
  4b)
    [ -n "$CLI17" ] || { echo "set LITERT_LM_0170 to the litert-lm 0.17.0 CLI (the 4B fp32 repack)" >&2; exit 2; }
    dl XHToken/Spark-X2.6-4B d1b5f26071101c27fcd07728e7a7770e428281f4 $SM/Spark-X2.6-4B
    python3 spark_work/patch_modeling.py $SM/Spark-X2.6-4B $SM/Spark-X2.6-4B-export
    # int8 (the CPU file): dynamic int8 everywhere, 6 prefill signatures; fp32 GPU activations declared (0.17.0 CLI),
    # then the tokenizer; no graph rewrite
    PREFILL=1024,256,64,16,4,1 $PY spark_work/convert_spark.py $SM/Spark-X2.6-4B-export out/spark26-4b-int8 $J dynamic_wi8_afp32
    mkdir -p out/spark26-4b-int8-fp32 out/spark26-4b-int4b128-fp32
    $PY scripts/set_activation_type.py out/spark26-4b-int8/model.litertlm out/spark26-4b-int8-fp32/model.litertlm \
      --type fp32 --litert-lm "$CLI17"
    swap out/spark26-4b-int8-fp32/model.litertlm $SM/Spark-X2.6-4B/tokenizer.json out/spark26-4b-int8-fp32-tok/model.litertlm
    order out/spark26-4b-int8-fp32-tok/model.litertlm Spark-X2.6-4B_int8
    # int4: blockwise-128 OCTAV with the 2^-14 scale floor, embedder in its own section
    EXTERNALIZE_EMBEDDER=1 PREFILL=1024,256,64,16,4,1 \
      $PY spark_work/x26_export_clamped.py $SM/Spark-X2.6-4B-export out/spark26-4b-int4b128 $J BOCTAV4_128
    $PY scripts/set_activation_type.py out/spark26-4b-int4b128/model.litertlm out/spark26-4b-int4b128-fp32/model.litertlm \
      --type fp32 --litert-lm "$CLI17"
    swap out/spark26-4b-int4b128-fp32/model.litertlm $SM/Spark-X2.6-4B/tokenizer.json \
      out/spark26-4b-int4b128-fp32-tok/model.litertlm
    post out/spark26-4b-int4b128-fp32-tok/model.litertlm out/spark26-4b-int4b128-gpug/model.litertlm
    order out/spark26-4b-int4b128-gpug/model.litertlm Spark-X2.6-4B_int4 ;;
  post)
    post "$2" "$3" "${4:-}" ;;
  *)
    sed -n '2,19p' "$0"; exit 2 ;;
esac
