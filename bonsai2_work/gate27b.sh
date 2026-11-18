#!/usr/bin/env bash
# Checks for the 27B bundle. Usage: gate27b.sh <bundle.litertlm> <tag> [quantized.tflite]
set -x
cd "$(dirname "$0")/.."
M=$1; TAG=$2; QT=$3; G=bonsai2_work/gates_$TAG; mkdir -p $G
LT=${LITERT_LM:-litert-lm}
# LITERT_LM: the litert-lm 0.17.1 CLI; RUN_PY: a Python with the litert-lm 0.17.1 API; CHECK_PY: a Python with ai-edge-litert 2.2 or later.
# Each defaults to the one found on PATH.
echo "===== multi-turn (python API, cpu) ====="; ${RUN_PY:-python3} bonsai2_work/multiturn_test.py $M 120 2>&1 | grep -E "^turn|MULTITURN|rror"
if [ -n "$QT" ]; then echo "===== logits parity vs MLX oracle (CompiledModel CPU decode walk, 24 positions) ====="; ${CHECK_PY:-python3} scripts/parity_logits_bigmodel.py lt --tflite $QT --ids bonsai2_work/full_mlx.npz --out $G/lt_full.npz 2>&1 | tail -1; python3 bonsai2_work/cmp_logits.py bonsai2_work/full_mlx.npz $G/lt_full.npz | head -1; fi
echo "===== benchmark CPU / GPU (litert-lm 0.17.1, p256 d256, cache no) ====="
$LT benchmark $M --backend cpu -p 256 -d 256 --runs 2 --cache no 2>&1 | grep -v "^I0\|^W0" | tail -8
$LT benchmark $M --backend gpu -p 256 -d 256 --runs 2 --cache no 2>&1 | grep -v "^I0\|^W0" | tail -8
echo CHECKS_DONE
