#!/usr/bin/env bash
# convert_qwen3_32b.sh OUTDIR [STEP...] - Qwen/Qwen3-32B (pinned 9216db5781bf21249d130ec9da846c4624c16137, Apache-2.0)
# -> OUTDIR/qwen3_32b_int8_ekv4096.litertlm, the file of litert-community/Qwen3-32B: the released litert-torch 0.10.0's
# export_hf with GPU composites (prefill 1024, KV cache 4096), int8 per-channel min-max weights for every
# FULLY_CONNECTED (lm_head included) and the token table, fp32 GPU activations declared, LiteRT-LM v0.19.1's own Qwen3
# LlmMetadata, the checkpoint's tokenizer.json. QUANT=b128 or QUANT=b32 makes the int4 blockwise file of the same export
# instead (int4 FULLY_CONNECTED, int8 lm_head and token table, block scales floored at 2^-14; named
# *_mixed_int4_ekv4096 / *_mixed_int4b32_ekv4096): for Qwen3-32B it failed the accuracy check (REPRODUCE.md). No float .litertlm is written (the float
# TFLite alone is ~131 GB). REPO=<another Qwen3 dense HF id> REV=<its commit> runs the same steps on that checkpoint
# (the output is then named after it, e.g. qwen3_0_6b_int8_ekv4096.litertlm); it tests the steps and keeps the
# Qwen3-32B metadata (stop token, sampler, template, max_num_tokens 4096).
# Disk for Qwen3-32B: about 430 GB at the peak of the quantize step (snapshot 65.5 GB, F32 copy 131 GB, prefused file
# 30.2 GB, float TFLite 131 GB, the quantized TFLite and the int8 file about 33.5 GB each, the TFLite removed after the
# pack); F32_DIR, PREFUSED and EXPORT_DIR can put the three large ones on another disk.
# Steps (in order; a finished step leaves OUTDIR/.done/<step> and is skipped on a re-run; name steps to run only those):
#   fetch     every file of the commit into OUTDIR/hf_cache (or HF_HUB_CACHE), one file at a time with
#             HF_HUB_DISABLE_XET=1, checked against the Hub's record (qwen35_work/fetch_pinned.py: LFS sha256 + git blob
#             sha1, sizes); skipped when the snapshot directory already exists there
#   upcast    an F32 copy of the checkpoint (qwen35_work/upcast_fp32.py: BF16 -> F32 by a 16-bit shift, exact, every
#             tensor and shard checked): transformers 5.22.1 then hands the export memory-mapped float32 tensors instead
#             of converting every bf16 weight in RAM
#   prefuse   the fused q||k||v of every layer and gate||up of the layers whose gate_proj and up_proj sit in different
#             shards, as one F32 file (qwen3_32b_work/prefuse_qwen3.py): litert-torch 0.10.0's fuse_qkv / fuse_gate_up
#             otherwise make each fused weight a new float32 tensor in RAM (about 80 GB over 64 layers)
#   shim      a stub `tensorflow` package in OUTDIR/tfstub: litert-torch 0.10.0 imports
#             tensorflow.lite.python.schema_py_generated without depending on TensorFlow; the stub re-exports
#             ai_edge_litert's schema module under that name
#   export    qwen3_32b_work/export_qwen3.py (the export_hf flags are in its docstring) with the F32 copy and the
#             prefused file, PYTHONHASHSEED=0, HF_HUB_OFFLINE=1; writes the float TFLite and the executor metadata
#             into EXPORT_DIR (default OUTDIR/export) and no .litertlm. Qwen3-32B on a 128 GB Mac with the F32 copy and
#             the TFLite on an external hard disk: 80 minutes at a peak memory footprint of 6.6 GB
#   quantize  qwen3_32b_work/make_int4_qwen3.py EXPORT_DIR SNAPSHOT/tokenizer.json OUTDIR NAME --block $QUANT --act fp32
#             (QUANT=i8 by default), PYTHONHASHSEED=0. Qwen3-32B from the float TFLite on the external hard disk:
#             51 minutes at a peak memory footprint of 33.6 GB (int8); 30 minutes at 56 GB (QUANT=b128)
#   checks    the weight form of every FULLY_CONNECTED / EMBEDDING_LOOKUP (qwen35_work/fc_dtypes.py, no float regex),
#             the ops whose every input is a constant (qwen35_work/const_ops.py), every block scale normal
#             (qwen35_work/scan_scales.py, blockwise int4 scales: exit 1 if a zero or subnormal scale is left; the int8
#             file has none)
#   sections  type, header items, size and sha256 of every section (mnn_pocket_work/sections_items.py); with
#             REF=<the published qwen3_32b_int8_ekv4096.litertlm> also the comparison section by section, header
#             items included (exit 1 on any difference). The packer writes a new uuid and creation time into the
#             header, so the file's own sha256 differs on every run: compare sections
# Interpreter (env, required): PY = the python of an environment made from mnn_pocket_work/requirements.txt (Python 3.13;
#   litert-torch 0.10.0, ai-edge-quantizer 0.10.1, ai-edge-litert 2.3.0, litert-converter 0.5.0, litert-lm-builder 0.18.1,
#   torch 2.14.1, transformers 5.22.1):
#     uv venv -p 3.13 .venv-qwen3-32b && uv pip install --python .venv-qwen3-32b/bin/python -r mnn_pocket_work/requirements.txt
set -euo pipefail
R=$(cd "$(dirname "$0")/.." && pwd); W=$R/qwen3_32b_work
[ $# -ge 1 ] || { echo "usage: $0 OUTDIR [STEP...]"; exit 2; }
mkdir -p "$1"; OUT=$(cd "$1" && pwd); shift
: "${PY:?set PY to the python of an environment made from mnn_pocket_work/requirements.txt}"
REPO=${REPO:-Qwen/Qwen3-32B}
case "$REPO" in Qwen/Qwen3-32B) DEF_REV=9216db5781bf21249d130ec9da846c4624c16137 ;; *) DEF_REV= ;; esac
REV=${REV:-$DEF_REV}; [ -n "$REV" ] || { echo "REV is required for $REPO (pin the commit)"; exit 2; }
BASE=$(basename "$REPO" | tr 'A-Z' 'a-z' | tr '.-' '__')
QUANT=${QUANT:-i8}
case $QUANT in i8) NAME=${BASE}_int8_ekv4096 ;; b128) NAME=${BASE}_mixed_int4_ekv4096 ;; b32) NAME=${BASE}_mixed_int4b32_ekv4096 ;;
  *) echo "QUANT must be i8, b128 or b32"; exit 2 ;; esac
CACHE=${HF_HUB_CACHE:-$OUT/hf_cache}
SNAP=$CACHE/models--${REPO//\//--}/snapshots/$REV
F32=${F32_DIR:-$OUT/fp32src}
PF=${PREFUSED:-$OUT/fp32fused/prefused.safetensors}
EXP=${EXPORT_DIR:-$OUT/export}
STEPS=("$@"); [ ${#STEPS[@]} -gt 0 ] || STEPS=(fetch upcast prefuse shim export quantize checks sections)
mkdir -p "$OUT/.done"
LOG=$OUT/recipe.log
run() { echo "### CMD: $*" | tee -a "$LOG"; "$@" 2>&1 | tee -a "$LOG"; return "${PIPESTATUS[0]}"; }
done_or() { [ -f "$OUT/.done/$1" ] && { echo "skip $1 (done)"; return 0; }; return 1; }
echo "### start $(date '+%F %T %Z') $REPO @ $REV -> $OUT/$NAME.litertlm" | tee -a "$LOG"

for step in "${STEPS[@]}"; do
  done_or "$step" && continue
  case $step in
    fetch)
      if [ -d "$SNAP" ]; then
        echo "snapshot present: $SNAP" | tee -a "$LOG"
      else
        run env HF_HUB_DISABLE_XET=1 "$PY" -I "$R/qwen35_work/fetch_pinned.py" "$CACHE" "$OUT/hf_cache_manifest.json" "$REPO" "$REV"
      fi
      ;;
    upcast)
      rm -rf "$F32"; mkdir -p "$(dirname "$F32")"
      run "$PY" -I "$R/qwen35_work/upcast_fp32.py" "$SNAP" "$F32" "$OUT/fp32src_manifest.json"
      ;;
    prefuse)
      rm -f "$PF" "$PF.partial"; mkdir -p "$(dirname "$PF")"
      run "$PY" -I "$W/prefuse_qwen3.py" "$SNAP" "$PF" "$OUT/prefused_manifest.json" --qkv all --gate-up boundary
      ;;
    shim)
      mkdir -p "$OUT/tfstub/tensorflow/lite/python"
      : > "$OUT/tfstub/tensorflow/__init__.py"
      : > "$OUT/tfstub/tensorflow/lite/__init__.py"
      : > "$OUT/tfstub/tensorflow/lite/python/__init__.py"
      echo "from ai_edge_litert.schema_py_generated import *  # noqa: F401,F403" > "$OUT/tfstub/tensorflow/lite/python/schema_py_generated.py"
      ;;
    export)
      rm -rf "$EXP" "$OUT/tmp"; mkdir -p "$EXP" "$OUT/tmp"
      run "$PY" -c 'import importlib.metadata as m; print(" ".join(f"{p}=={m.version(p)}" for p in ("litert-torch", "ai-edge-quantizer", "ai-edge-litert", "litert-converter", "litert-lm-builder", "torch", "transformers")))'
      run env HF_HUB_OFFLINE=1 PYTHONPATH="$OUT/tfstub" TMPDIR="$OUT/tmp" PYTHONHASHSEED=0 Q32_PREFUSED="$PF" \
        "$PY" "$W/export_qwen3.py" "$F32" "$EXP"
      ;;
    quantize)
      rm -rf "$OUT/$NAME.litertlm" "$OUT/work_$NAME"; mkdir -p "$OUT/tmp"
      run env PYTHONHASHSEED=0 TMPDIR="$OUT/tmp" "$PY" "$W/make_int4_qwen3.py" "$EXP" "$SNAP/tokenizer.json" "$OUT" "$NAME" \
        --block "$QUANT" --act fp32
      run shasum -a 256 "$OUT/$NAME.litertlm"
      ;;
    checks)
      run "$PY" -I "$R/qwen35_work/fc_dtypes.py" "$OUT/$NAME.litertlm" "$OUT/fc_dtypes.json" '$^'
      run "$PY" -I "$R/qwen35_work/const_ops.py" "$OUT/$NAME.litertlm" "$OUT/const_ops.json"
      run "$PY" -I "$R/qwen35_work/scan_scales.py" "$OUT/$NAME.litertlm" "$OUT/scan_scales.json"
      ;;
    sections)
      if [ -n "${REF:-}" ]; then
        run "$PY" -I "$R/mnn_pocket_work/sections_items.py" "$OUT/$NAME.litertlm" "$REF"
      else
        run "$PY" -I "$R/mnn_pocket_work/sections_items.py" "$OUT/$NAME.litertlm"
      fi
      ;;
    *) echo "unknown step $step"; exit 2 ;;
  esac
  touch "$OUT/.done/$step"
done
echo "### done $(date '+%F %T %Z')" | tee -a "$LOG"
