#!/bin/bash
# granite-4.0-h-350m re-export with the FOLDED (rank<=4 batched-matmul) SSD scan.
#
# Why: the bundles published on litert-community/granite-4.0-h-350m were built
# 2026-08-05, BEFORE the folded scan landed in the granite exporter ext
# (commit 3835a9d, 2026-08-13). They therefore carry the upstream rank-6
# broadcast-reduce contraction, which the S4 S26 Adreno gate rejects with
#   PatchedGraniteMoeHybridMambaLayer_mamba;76 has bad input dims size: 6
#   TRANSPOSE: Permutation for transpose is invalid   -> 109/3673 delegated
# The same exporter with the folded scan delegated the 350m 3537/3537 on Pixel 8a
# OpenCL (3835a9d) and the granite-4.0-h-1b built after it PASSES the S26 gate
# fully delegated (S4 idx 15). This rebuilds the 350m on that rail.
#
# Steps: float export -> post-hoc int8 (linears+embedding; convs/SSM float)
#        -> ExecutorMetadata -> drop start_token (350m REQUIRES no-BOS)
#        -> prefer_activation_type = fp32 variant (the 1b needed it on GPU)
set -euo pipefail
REPO="$HOME/code/litertlm-convert"
PUB="$HOME/code/hf-to-litertlm"
HERE="$REPO/granite_work/gpu_reexport_20260827"
export PATH="$REPO/.venv-092/bin:$PATH"
export PYTHONPATH="$HOME/code/litert-torch"
PY="$REPO/.venv-092/bin/python"
OUT="$HERE/out_350m"

mkdir -p "$OUT"

# 1. float export + post-hoc int8 + executor metadata (the published converter)
if [ ! -f "$OUT/granite-4.0-h-350m_int8.litertlm" ]; then
  "$PY" "$PUB/granite_work/convert_granite4h.py" \
      ibm-granite/granite-4.0-h-350m "$OUT" 2>&1 | tee "$HERE/export_350m.log" | tail -5
fi

INT8="$OUT/granite-4.0-h-350m_int8.litertlm"
ls -la "$INT8"

# 2. no-BOS (required at 350M scale — see granite_work/card_350m_DRAFT.md)
NOBOS="$HERE/granite-4.0-h-350m_int8_nobos.litertlm"
"$PY" "$PUB/granite_work/drop_start_token.py" "$INT8" "$NOBOS"

# 3. fp32-activation variant for the GPU lane (weights untouched, repack only)
FP32ACT="$HERE/granite-4.0-h-350m_int8_gpu.litertlm"
"$PY" "$REPO/scripts/set_activation_type.py" "$NOBOS" "$FP32ACT" --type fp32

ls -la "$HERE"/*.litertlm
echo DONE
