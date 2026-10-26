#!/usr/bin/env bash
# convert_bolt_e4b.sh OUTDIR [STEP...] - nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16 (pinned 49843eed8db1f1c616c47af75cfd36e07aec0667,
# OpenMDW-1.1; post-trained by NVIDIA from google/gemma-4-E4B-it) -> the int4 file of
# litert-community/NVIDIA-Nemotron-3.5-Bolt-E4B, text only, a 2,048-token KV cache:
#   OUTDIR/NVIDIA-Nemotron-3.5-Bolt-E4B_int4.litertlm  GPU composites (odml.cache_update, odml.runtime_bmm), prefill
#                                                      signatures of 128, 512 and 1024 tokens
# Steps (in order; a finished step leaves OUTDIR/.done/<step> and is skipped on a re-run; name steps to run only those):
#   fetch     the files the export reads, curl into OUTDIR/hf_cache (dory_work/fetch_pinned.sh), checked against the Hub's
#             record of the commit (dory_work/verify_pinned.py: sha256 for LFS files, git blob sha1 for the others). With
#             MODEL_DIR set to a local copy of the revision, its files are linked into OUTDIR/hf_cache instead and checked
#             the same way, and the export reads MODEL_DIR
#   shim      a stub `tensorflow` package in OUTDIR/tfstub: litert-torch at 0a81bf6f imports
#             tensorflow.lite.python.schema_py_generated (model_ext/bonsai_flux2); the stub re-exports ai_edge_litert's
#             schema module under that name (a full TensorFlow install clashes with litert-converter at process start)
#   int4      export_clamped_hf.py = litert-torch's export_hf with the int4 block scales floored at 2^-14; recipe
#             recipe_int4_b32_emb4cw.json (int4 blockwise-32 min-max FULLY_CONNECTED weights, int4 channelwise embedding
#             tables); prefill 128 / 512 / 1024, cache 2048, embedder externalized, no vision or audio encoder,
#             --use_rope_composite=True, lightweight conversion (RESOURCE_CONSTANT_NUMEL_THRESHOLD=30000000), and
#             --apply_gpu_composites=True with the GPU options that flag would also turn on passed False
#   sections  sha256 of every section of the file (sections.py)
# Interpreter (env, required):
#   PY = Python 3.12 with litert-torch 0.10.0 installed editable from a checkout at
#        0a81bf6f3243a6f7c846d05d62afeff4008f2542, torch 2.12.0, transformers 5.19.0, ai-edge-quantizer-nightly
#        0.10.0.dev20261013, ai-edge-litert-nightly 2.4.0.dev20261011, litert-converter 0.5.0, litert-lm-builder 0.18.0
# Memory: an export peaked at 74.1-74.3 GB footprint and 84-87 GB resident on a 128 GB Mac; run one export at a time.
set -euo pipefail
R=$(cd "$(dirname "$0")/.." && pwd); W=$R/nemotron_bolt_work
[ $# -ge 1 ] || { echo "usage: $0 OUTDIR [STEP...]"; exit 2; }
mkdir -p "$1"; OUT=$(cd "$1" && pwd); shift
: "${PY:?set PY}"
# litert-torch writes the LlmMetadata stop tokens in the iteration order of a set of strings
export PYTHONHASHSEED=0
REPO=nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16; REV=49843eed8db1f1c616c47af75cfd36e07aec0667
FILES=(config.json generation_config.json model.safetensors.index.json model-00001-of-00001.safetensors
  model-frozen-shared-kv.safetensors tokenizer.json tokenizer_config.json chat_template.jinja processor_config.json)
NAME=NVIDIA-Nemotron-3.5-Bolt-E4B
SNAP=$OUT/hf_cache/models--${REPO//\//--}/snapshots/$REV
STEPS=("$@"); [ ${#STEPS[@]} -gt 0 ] || STEPS=(fetch shim int4 sections)
mkdir -p "$OUT/.done"
LOG=$OUT/recipe.log
run() { echo "### CMD: $*" | tee -a "$LOG"; "$@" 2>&1 | tee -a "$LOG"; return "${PIPESTATUS[0]}"; }
done_or() { [ -f "$OUT/.done/$1" ] && { echo "skip $1 (done)"; return 0; }; return 1; }
model_dir() { if [ -n "${MODEL_DIR:-}" ]; then (cd "$MODEL_DIR" && pwd); else echo "$SNAP"; fi; }

export_int4() {  # export_int4 <file name> [extra export_hf flags...]
  local dest=$1; shift
  local work=$OUT/work_${dest%.litertlm}
  rm -rf "$work"; mkdir -p "$work"
  run env RESOURCE_CONSTANT_NUMEL_THRESHOLD=30000000 PYTHONPATH="$OUT/tfstub" "$PY" "$W/export_clamped_hf.py" \
    --model="$(model_dir)" --output_dir="$work" --task=text_generation \
    --prefill_lengths=128,512,1024 --cache_length=2048 \
    --externalize_embedder=True \
    --export_vision_encoder=False --export_audio_encoder=False \
    --bundle_litert_lm=True --use_rope_composite=True --experimental_lightweight_conversion=True \
    --quantization_recipe="$W/recipe_int4_b32_emb4cw.json" "$@"
  mv "$work/model.litertlm" "$OUT/$dest"
  rm -rf "$work"
  ls -l "$OUT/$dest" | tee -a "$LOG"
}

for step in "${STEPS[@]}"; do
  done_or "$step" && continue
  case $step in
    fetch)
      if [ -n "${MODEL_DIR:-}" ]; then
        mkdir -p "$SNAP"
        for f in "${FILES[@]}"; do ln -sf "$(model_dir)/$f" "$SNAP/$f"; done
      else
        run bash "$R/dory_work/fetch_pinned.sh" "$REPO" "$REV" "$OUT/hf_cache" "${FILES[@]}"
      fi
      run "$PY" "$R/dory_work/verify_pinned.py" "$REPO" "$REV" "$OUT/hf_cache" "$OUT/fetch_manifest.json" "${FILES[@]}"
      ;;
    shim)
      mkdir -p "$OUT/tfstub/tensorflow/lite/python"
      : > "$OUT/tfstub/tensorflow/__init__.py"
      : > "$OUT/tfstub/tensorflow/lite/__init__.py"
      : > "$OUT/tfstub/tensorflow/lite/python/__init__.py"
      echo "from ai_edge_litert.schema_py_generated import *  # noqa: F401,F403" > "$OUT/tfstub/tensorflow/lite/python/schema_py_generated.py"
      ;;
    int4)
      # --apply_gpu_composites=True also turns every GPU option that is not given on (litert-torch 0a81bf6f,
      # core/exportable_module_config.py); the ones this file does not use are passed False
      export_int4 "${NAME}_int4.litertlm" --apply_gpu_composites=True --use_qkv_norm_rope_composite=False \
        --use_sdpa_composite=False --fuse_gate_up=False --fuse_qkv=False --use_swiglu_composite=False \
        --use_short_conv_composite=False --use_bool_mask=False --enable_gpu_dynamic_cache=False
      ;;
    sections)
      run "$PY" "$W/sections.py" "$OUT/${NAME}_int4.litertlm"
      ;;
    *) echo "unknown step $step"; exit 2 ;;
  esac
  touch "$OUT/.done/$step"
done
