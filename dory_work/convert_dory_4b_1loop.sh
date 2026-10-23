#!/usr/bin/env bash
# convert_dory_4b_1loop.sh OUTDIR [STEP...] - the one-loop file of litert-community/NVIDIA-Nemotron-Labs-Dory1.0-4B:
# OUTDIR/NVIDIA-Nemotron-Labs-Dory1.0-4B_int4_1loop.litertlm = the published int4 file (pinned 61cb6bbe) with the 20
# recurrent attention/MLP pairs applied once per token instead of twice (NVIDIA's n_recurrent_loops = 1; 36 KV caches
# instead of 56). Its int4 weights are the published int4 file's, byte for byte (the GPTQ codes and the block scales);
# no GPTQ run, no calibration set. Every step logs its command to OUTDIR/recipe.log. Steps (in order; a finished step
# leaves OUTDIR/.done/<step> and is skipped on a re-run; name steps to run only those):
#   fetch     the 7 checkpoint files the export reads (nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16 at b366b87f) and the
#             published int4 file, curl into OUTDIR/hf_cache (dory_work/fetch_pinned.sh), each checked against the Hub's
#             record of its commit (dory_work/verify_pinned.py)
#   loops     a model dir = the pinned snapshot with config.json's n_recurrent_loops 2 -> 1 (dory_work/set_loops.py);
#             dory_modeling.py reads the loop count from the config
#   export    the float export of that dir (dory_work/export_dory.py, the same arguments as convert_dory_4b.sh: the fp16
#             rewrite of the 11 blocks of fp16_shift.json, prefill 128 / 512 / 1024, KV 2048, embedder externalized)
#   quantize  int4 = recipe_int4_b32.json with every block scale floored at 2^-14 and the runtime chat template
#             dory_runtime.jinja (dory_work/quantize_dory.py, as for the int4 file); the float export is deleted
#   copy      the published int4 file's codes written into this data-free TFLite, weight by weight, matched by their
#             block scales (dory_work/copy_int4.py; the data-free scales are the published file's), and its metadata
#             and embedder sections cut out of it
#   gpu       the GPU KV composites, as in convert_dory_4b.sh (dory_work/retrofit_gpu.py with
#             tools/gpu_graph/gpu_graph_retrofit.py); with the published metadata and embedder sections and this
#             export's executor metadata (the 36 KV caches)
#   pack      the .litertlm file (dory_work/pack_dory.py); no activation type declared
# Paths are relative to this repository (dory_work/, tools/). Interpreter (env, required):
#   EPY = python with this repository's pinned toolchain (uv.lock: litert-torch 0.9.4, ai-edge-quantizer 0.9.0,
#         litert-converter 0.4.0, ai-edge-litert 2.2.0, torch 2.13.0, transformers 5.14.1, litert-lm-builder 0.18.0)
set -euo pipefail
R=$(cd "$(dirname "$0")/.." && pwd); W=$R/dory_work
[ $# -ge 1 ] || { echo "usage: $0 OUTDIR [STEP...]"; exit 2; }
mkdir -p "$1"; OUT=$(cd "$1" && pwd); shift
: "${EPY:?set EPY}"
export PYTHONHASHSEED=0   # litert-torch 0.9.4 writes the stop tokens, and dory_modeling.py builds its two RoPE tables, in the iteration order of a set of strings
REPO=nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16; REV=b366b87f762ea334ba3eb55912031baba1515810
PUB=litert-community/NVIDIA-Nemotron-Labs-Dory1.0-4B; PUBREV=61cb6bbe177d134fdbc215cf5b81f592f434ad08
SNAP=$OUT/hf_cache/models--nvidia--NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16/snapshots/$REV
NAME=NVIDIA-Nemotron-Labs-Dory1.0-4B
INT4=$OUT/hf_cache/models--litert-community--NVIDIA-Nemotron-Labs-Dory1.0-4B/snapshots/$PUBREV/${NAME}_int4.litertlm
DFILES="config.json generation_config.json special_tokens_map.json tokenizer_config.json tokenizer.json chat_template.jinja model.safetensors"
log() { echo "$*" >> "$OUT/recipe.log"; }
run() { log "### CMD: $*"; "$@"; }

step_fetch() {
  run bash "$W/fetch_pinned.sh" $REPO $REV "$OUT/hf_cache" $DFILES
  run "$EPY" "$W/verify_pinned.py" $REPO $REV "$OUT/hf_cache" "$OUT/fetch_dory.json" $DFILES
  run bash "$W/fetch_pinned.sh" $PUB $PUBREV "$OUT/hf_cache" ${NAME}_int4.litertlm
  run "$EPY" "$W/verify_pinned.py" $PUB $PUBREV "$OUT/hf_cache" "$OUT/fetch_int4.json" ${NAME}_int4.litertlm
}
step_loops() {
  run "$EPY" "$W/set_loops.py" "$SNAP" "$OUT/src_1loop" 1 > "$OUT/loops.log" 2>&1
}
step_export() {
  rm -rf "$OUT/float_1loop"
  run "$EPY" "$W/export_dory.py" "$OUT/src_1loop" "$OUT/float_1loop" "$W/fp16_shift.json" 128,512,1024 \
    > "$OUT/export.log" 2>&1
}
step_quantize() {
  run "$EPY" "$W/quantize_dory.py" "$OUT/float_1loop" "$OUT/int4_1loop" "$W/recipe_int4_b32.json" \
    "$W/dory_runtime.jinja" --int4-floor > "$OUT/quantize_int4.log" 2>&1
  run rm -rf "$OUT/float_1loop"
}
step_copy() {
  run "$EPY" "$W/copy_int4.py" "$OUT/int4_1loop" "$INT4" > "$OUT/copy.log" 2>&1
}
step_gpu() {
  rm -rf "$OUT/int4_1loop_gpu"; mkdir -p "$OUT/int4_1loop_gpu"
  run "$EPY" -B "$W/retrofit_gpu.py" "$OUT/int4_1loop/model_quantized.tflite" \
    "$OUT/int4_1loop_gpu/model_quantized.tflite" "$OUT/retrofit_gpu.json" > "$OUT/gpu.log" 2>&1
  run cp "$OUT/int4_1loop/published_llm_metadata_t.pb" "$OUT/int4_1loop_gpu/llm_metadata_t.pb"
  run cp "$OUT/int4_1loop/executor_metadata.pb" "$OUT/int4_1loop_gpu/executor_metadata.pb"
  run cp "$OUT/int4_1loop/published_embedder_quantized.tflite" "$OUT/int4_1loop_gpu/embedder_quantized.tflite"
}
step_pack() {
  run "$EPY" "$W/pack_dory.py" "$OUT/int4_1loop_gpu" "$SNAP/tokenizer.json" "$OUT/${NAME}_int4_1loop.litertlm" \
    > "$OUT/pack.log" 2>&1
  shasum -a 256 "$OUT/${NAME}_int4_1loop.litertlm" | tee -a "$OUT/recipe.log"
}

STEPS=${*:-fetch loops export quantize copy gpu pack}
log "### start $(date '+%F %T %Z') steps: $STEPS"
for s in $STEPS; do
  case " fetch loops export quantize copy gpu pack " in *" $s "*) ;; *) echo "unknown step $s"; exit 2;; esac
  if [ -f "$OUT/.done/$s" ]; then echo "skip $s (done)"; continue; fi
  log "### step $s $(date '+%F %T %Z')"
  "step_$s"
  mkdir -p "$OUT/.done"; touch "$OUT/.done/$s"
done
log "### done $(date '+%F %T %Z') steps: $STEPS"
