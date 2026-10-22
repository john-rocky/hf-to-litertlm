#!/usr/bin/env bash
# convert_dory_4b.sh OUTDIR [STEP...] - nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16 (pinned b366b87f762ea334ba3eb55912031baba1515810,
# OpenMDW-1.1) -> OUTDIR/NVIDIA-Nemotron-Labs-Dory1.0-4B_int4.litertlm (GPTQ int4 with the GPU KV composites, the GPU
# file) and OUTDIR/NVIDIA-Nemotron-Labs-Dory1.0-4B_int8.litertlm (dynamic int8, the CPU file) in one command; every step
# logs its command to OUTDIR/recipe.log. Steps (in order; a finished step leaves OUTDIR/.done/<step> and is skipped on a
# re-run; name steps to run only those): fetch calib export quantize map hessians gptq write gpu pack
# Paths are relative to this repository (dory_work/, g9v3_work/). Interpreters (env, required):
#   EPY = python with this repository's pinned toolchain (uv.lock: litert-torch 0.9.4, ai-edge-quantizer 0.9.0,
#         litert-converter 0.4.0, ai-edge-litert 2.2.0, torch 2.13.0, transformers 5.14.1, litert-lm-builder 0.18.0)
#   GPY = python 3.12 with dory_work/requirements-gptq.txt (torch 2.11.0, numpy 2.2.6, transformers 4.57.6): the
#         calibration render, the Hessians and GPTQ, on Apple silicon (MPS)
#   CPY = python with transformers and datasets: g9v3_work's calibration-set builder (it checks the set's sha256)
#   fetch     the 7 files the conversion reads, curl into OUTDIR/hf_cache (dory_work/fetch_pinned.sh), checked against
#             the Hub's record of the commit (dory_work/verify_pinned.py); the tokenizer files of ai9stars/G9v3-3B at
#             d9553445 for the calibration-set builder
#   calib     g9v3_work/gptq_quantize.py --calib-only (256 chat samples from three datasets at pinned revisions, sha256
#             26e9cc95...), re-rendered with Dory's template + 64 thinking-on samples (GSM8K training questions and the
#             model's own continuations, dory_work/calib_think_ids.jsonl) = 320 samples (dory_work/calib_dory.py)
#   export    the float export (dory_work/export_dory.py: dory_modeling.py, the fp16 rewrite of the MLP justnorm in
#             the 11 blocks of fp16_shift.json, prefill 128 / 512 / 1024, KV 2048, embedder externalized, no start
#             token, no float bundle written)
#   quantize  int8 = dynamic_wi8_afp32; int4 = recipe_int4_b32.json (int4 blockwise-32 FCs, int8 embedding) with every
#             block scale floored at 2^-14; both with the runtime chat template dory_runtime.jinja in the metadata;
#             the float export is deleted afterwards
#   map       the 289 FULLY_CONNECTED weights of the int4 / int8 files mapped to the checkpoint (dory_work/fc_map.py)
#   hessians  GPTQ Hessians of the 320 samples on the PyTorch reference dory_ref.py (fp32, MPS)
#   gptq      GPTQ codes with the int4 file's own block scales (group 32, damp 0.01, natural column order, MPS)
#   write     the codes written into the int4 TFLite (dory_work/write_codes.py; nothing else changes)
#   gpu       the GPU KV composites (odml.cache_update, odml.runtime_bmm) put into the int4 TFLite by
#             tools/gpu_graph/gpu_graph_retrofit.py, through dory_work/retrofit_gpu.py (it adds the handling of Dory's
#             sliding-window mask; every weight buffer keeps its bytes)
#   pack      the two .litertlm files (dory_work/pack_dory.py): int4 from the composite TFLite, int8 as quantized; no
#             activation type declared
set -euo pipefail
R=$(cd "$(dirname "$0")/.." && pwd); W=$R/dory_work
[ $# -ge 1 ] || { echo "usage: $0 OUTDIR [STEP...]"; exit 2; }
mkdir -p "$1"; OUT=$(cd "$1" && pwd); shift
: "${EPY:?set EPY}" "${GPY:?set GPY}" "${CPY:?set CPY}"
export PYTHONHASHSEED=0   # litert-torch 0.9.4 writes the stop tokens, and dory_modeling.py builds its two RoPE tables, in the iteration order of a set of strings
REPO=nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16; REV=b366b87f762ea334ba3eb55912031baba1515810
G9=ai9stars/G9v3-3B; G9REV=d9553445ff92dbce667381954c9699fbcbc924f9
SNAP=$OUT/hf_cache/models--nvidia--NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16/snapshots/$REV
NAME=NVIDIA-Nemotron-Labs-Dory1.0-4B
DFILES="config.json generation_config.json special_tokens_map.json tokenizer_config.json tokenizer.json chat_template.jinja model.safetensors"
GFILES="config.json special_tokens_map.json tokenizer_config.json tokenizer.json chat_template.jinja"
log() { echo "$*" >> "$OUT/recipe.log"; }
run() { log "### CMD: $*"; "$@"; }

step_fetch() {
  run bash "$W/fetch_pinned.sh" $REPO $REV "$OUT/hf_cache" $DFILES
  run "$EPY" "$W/verify_pinned.py" $REPO $REV "$OUT/hf_cache" "$OUT/fetch_dory.json" $DFILES
  run bash "$W/fetch_pinned.sh" $G9 $G9REV "$OUT/hf_cache" $GFILES
  run "$EPY" "$W/verify_pinned.py" $G9 $G9REV "$OUT/hf_cache" "$OUT/fetch_g9v3.json" $GFILES
}
step_calib() {
  run env HF_HUB_CACHE="$OUT/hf_cache" "$CPY" "$R/g9v3_work/gptq_quantize.py" "$OUT/g9v3_unused" \
    "$OUT/calib_g9v3_256.jsonl" --calib-only > "$OUT/calib_g9v3.log" 2>&1
  run "$GPY" "$W/calib_dory.py" "$SNAP" "$OUT/calib_g9v3_256.jsonl" "$W/calib_think_ids.jsonl" "$OUT/calib" \
    > "$OUT/calib.log" 2>&1
}
step_export() {
  rm -rf "$OUT/float"
  run "$EPY" "$W/export_dory.py" "$SNAP" "$OUT/float" "$W/fp16_shift.json" 128,512,1024 > "$OUT/export.log" 2>&1
}
step_quantize() {
  run "$EPY" "$W/quantize_dory.py" "$OUT/float" "$OUT/int8" dynamic_wi8_afp32 "$W/dory_runtime.jinja" \
    > "$OUT/quantize_int8.log" 2>&1
  run "$EPY" "$W/quantize_dory.py" "$OUT/float" "$OUT/int4" "$W/recipe_int4_b32.json" "$W/dory_runtime.jinja" \
    --int4-floor > "$OUT/quantize_int4.log" 2>&1
  run rm -rf "$OUT/float"
}
step_map() {
  run "$EPY" "$W/fc_map.py" "$OUT/int4/model_quantized.tflite" "$OUT/int8/model_quantized.tflite" "$SNAP" \
    "$OUT/wq" "$OUT/fcmap.json" > "$OUT/map.log" 2>&1
}
step_hessians() {
  run "$GPY" "$W/gptq_hessians.py" "$SNAP" "$OUT/calib/samples.jsonl" "$OUT/hess" > "$OUT/hessians.log" 2>&1
}
step_gptq() {
  run "$GPY" "$W/gptq_dory.py" "$SNAP" "$OUT/hess" "$OUT/wq" "$OUT/fcmap.json" "$OUT/gptq" --consume-h \
    > "$OUT/gptq.log" 2>&1
  run rm -rf "$OUT/hess"
}
step_write() {
  run "$EPY" "$W/write_codes.py" "$OUT/int4" "$OUT/gptq" "$OUT/fcmap.json" > "$OUT/write.log" 2>&1
}
step_gpu() {
  rm -rf "$OUT/int4_gpu"; mkdir -p "$OUT/int4_gpu"
  run "$EPY" -B "$W/retrofit_gpu.py" "$OUT/int4/model_quantized.tflite" "$OUT/int4_gpu/model_quantized.tflite" \
    "$OUT/retrofit_gpu.json" > "$OUT/gpu.log" 2>&1
  for f in llm_metadata_t.pb executor_metadata.pb embedder_quantized.tflite; do
    run cp "$OUT/int4/$f" "$OUT/int4_gpu/$f"
  done
}
step_pack() {
  run "$EPY" "$W/pack_dory.py" "$OUT/int4_gpu" "$SNAP/tokenizer.json" "$OUT/${NAME}_int4.litertlm" > "$OUT/pack_int4.log" 2>&1
  run "$EPY" "$W/pack_dory.py" "$OUT/int8" "$SNAP/tokenizer.json" "$OUT/${NAME}_int8.litertlm" > "$OUT/pack_int8.log" 2>&1
  shasum -a 256 "$OUT/${NAME}"_*.litertlm | tee -a "$OUT/recipe.log"
}

STEPS=${*:-fetch calib export quantize map hessians gptq write gpu pack}
log "### start $(date '+%F %T %Z') steps: $STEPS"
for s in $STEPS; do
  case " fetch calib export quantize map hessians gptq write gpu pack " in *" $s "*) ;; *) echo "unknown step $s"; exit 2;; esac
  if [ -f "$OUT/.done/$s" ]; then echo "skip $s (done)"; continue; fi
  log "### step $s $(date '+%F %T %Z')"
  "step_$s"
  mkdir -p "$OUT/.done"; touch "$OUT/.done/$s"
done
log "### done $(date '+%F %T %Z') steps: $STEPS"
