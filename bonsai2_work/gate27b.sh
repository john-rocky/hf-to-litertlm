#!/usr/bin/env bash
# Gates for the 27B bundle. Usage: gate27b.sh <bundle.litertlm> <tag> [quantized.tflite]
set -x
cd ~/code/litertlm-convert
M=$1; TAG=$2; QT=$3; G=bonsai2_work/gates_$TAG; mkdir -p $G
LT=$HOME/venvs/lt0171run/bin/litert-lm
echo "===== 8Q CPU (litert-lm 0.17.1) ====="; LITERT_LM=$LT ~/venvs/ltconv040dev/bin/python3 qwen35_work/gate8q_qwen35.py $M cpu $G/gate8q_cpu.json 2>&1 | grep -v "^I0\|^W0" | tail -12
echo "===== 8Q GPU (Mac WebGPU) ====="; LITERT_LM=$LT ~/venvs/ltconv040dev/bin/python3 qwen35_work/gate8q_qwen35.py $M gpu $G/gate8q_gpu.json 2>&1 | grep -v "^I0\|^W0" | tail -12
echo "===== multi-turn (python API, cpu) ====="; ~/venvs/lt0171run/bin/python3 bonsai2_work/multiturn_test.py $M 120 2>&1 | grep -E "^turn|MULTITURN|rror"
if [ -n "$QT" ]; then echo "===== logits parity vs MLX oracle (CompiledModel CPU decode walk, 24 positions) ====="; ~/venvs/ltmain0918/bin/python3 scripts/parity_logits_bigmodel.py lt --tflite $QT --ids bonsai2_work/full_mlx.npz --out $G/lt_full.npz 2>&1 | tail -1; python3 bonsai2_work/cmp_logits.py bonsai2_work/full_mlx.npz $G/lt_full.npz | head -1; fi
echo "===== benchmark CPU / GPU (litert-lm 0.17.1, p256 d256, cache no) ====="
$LT benchmark $M --backend cpu -p 256 -d 256 --runs 2 --cache no 2>&1 | grep -v "^I0\|^W0" | tail -8
$LT benchmark $M --backend gpu -p 256 -d 256 --runs 2 --cache no 2>&1 | grep -v "^I0\|^W0" | tail -8
echo GATES_DONE
