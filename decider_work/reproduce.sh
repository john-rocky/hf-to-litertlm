#!/bin/bash
# One-command reproduction of litert-community/decider-0.8b-LiteRT (fp16 exact + dynamic int8) from a clean checkout.
# Needs: python3.12 with decider_work/requirements-lock-export.txt installed as PY (export/repack) — see decider_work/README.md.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-$HOME/venvs/lt092run/bin/python}"
OUT="${1:-out/decider-0.8b}"; mkdir -p "$OUT"
[ -d litert-torch-qwen35 ] || { git clone https://github.com/john-rocky/litert-torch litert-torch-qwen35
  git -C litert-torch-qwen35 checkout 115a13607c730c81018bb9789138a3e5e5119e3d
  git -C litert-torch-qwen35 apply "$(pwd)/qwen35_work/qwen35_hybrid_litert_torch.patch"; }
[ -f "$OUT/fp/model.litertlm" ] || STOP_AFTER_EXPORT=1 PYTHONPATH=litert-torch-qwen35 $PY qwen35_work/convert_qwen35_hybrid.py Mapika/decider-0.8b "$OUT/fp"
for pair in wfp16:fp16 wi8fc:int8; do r="${pair%:*}"; f="${pair#*:}"
  $PY decider_work/scripts/quantize_decider_ab.py apply "$OUT/fp/model.litertlm" "$OUT/${f}_raw.litertlm" --recipe "$r"
  $PY decider_work/scripts/repack_identity.py "$OUT/${f}_raw.litertlm" "$OUT/${f}_tmpl.litertlm" --unpack "$OUT/${f}_unpack"
  $PY scripts/add_executor_metadata.py "$OUT/${f}_tmpl.litertlm" "$OUT/${f}_meta.litertlm"
  $PY scripts/set_activation_type.py "$OUT/${f}_meta.litertlm" "$OUT/decider-0.8b_${f}.litertlm" --type fp32
  rm -rf "$OUT/${f}_unpack" "$OUT/${f}_raw.litertlm" "$OUT/${f}_tmpl.litertlm" "$OUT/${f}_meta.litertlm"
done
sha256sum "$OUT"/decider-0.8b_*.litertlm
