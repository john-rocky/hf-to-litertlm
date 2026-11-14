#!/bin/bash
# granite-4.0-h-350m re-export with the FOLDED (rank<=4 batched-matmul) SSD scan.
#
# Why: the files published on litert-community/granite-4.0-h-350m were built
# 2026-08-05, BEFORE the folded scan landed in granite_hybrid_litert_torch.patch
# (this repo's commit 6e9e3e6, 2026-08-13). They therefore carry the upstream rank-6
# broadcast-reduce contraction, which the Galaxy S26 (Adreno) GPU delegate rejects with
#   PatchedGraniteMoeHybridMambaLayer_mamba;76 has bad input dims size: 6
#   TRANSPOSE: Permutation for transpose is invalid   -> 109/3673 delegated
# The same exporter with the folded scan delegated the 350m 3537/3537 on Pixel 8a
# OpenCL, and the granite-4.0-h-1b built after it runs fully delegated on the
# Galaxy S26 GPU. This rebuilds the 350m the same way.
#
# Steps: float export -> post-hoc int8 (linears+embedding; convs/SSM float)
#        -> ExecutorMetadata -> drop start_token (350m REQUIRES no-BOS)
#        -> prefer_activation_type = fp32 variant (the 1b needed it on GPU)
#
# Environment (the setup is in convert_granite4h.py's docstring):
#   PY            the python of that setup, default: python on PATH. Its folder goes first on
#                 PATH, so the litert-lm CLI the steps call by name is the one installed next to it.
#   LITERT_TORCH  the patched litert-torch checkout, default: granite_work/litert-torch-granite
#   WORK          output folder, default: ./gpu_reexport_350m
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=$(command -v "${PY:-python}") || { echo "PY: no such python (see Environment above)"; exit 2; }
HERE=${WORK:-$PWD/gpu_reexport_350m}
export PATH="$(dirname "$PY"):$PATH"
export PYTHONPATH="${LITERT_TORCH:-$ROOT/granite_work/litert-torch-granite}"
OUT="$HERE/out_350m"

mkdir -p "$OUT"

# 1. float export + post-hoc int8 + executor metadata (the published converter)
if [ ! -f "$OUT/granite-4.0-h-350m_int8.litertlm" ]; then
  "$PY" "$ROOT/granite_work/convert_granite4h.py" \
      ibm-granite/granite-4.0-h-350m "$OUT" 2>&1 | tee "$HERE/export_350m.log" | tail -5
fi

INT8="$OUT/granite-4.0-h-350m_int8.litertlm"
ls -la "$INT8"

# 2. no-BOS (required at 350M scale — drop_start_token.py says why)
NOBOS="$HERE/granite-4.0-h-350m_int8_nobos.litertlm"
"$PY" "$ROOT/granite_work/drop_start_token.py" "$INT8" "$NOBOS"

# 3. fp32-activation variant for the GPU (weights untouched, repack only)
FP32ACT="$HERE/granite-4.0-h-350m_int8_gpu.litertlm"
"$PY" "$ROOT/scripts/set_activation_type.py" "$NOBOS" "$FP32ACT" --type fp32

ls -la "$HERE"/*.litertlm
echo DONE
