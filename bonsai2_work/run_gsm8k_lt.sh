#!/usr/bin/env bash
# GSM8K n=100, greedy, 2048-token budget, on the .litertlm through litert-mac-verify (same prompt/extraction as the
# MLX-oracle run in gsm8k_mlx_pack.py). Usage: run_gsm8k_lt.sh <bundle.litertlm> <tag> [gpu|cpu]
set -e
cd ~/code/litertlm-convert
VERIFY_EXTRA="--backend ${3:-gpu}" ~/venvs/ltconv040dev/bin/python3 scripts/parity_gsm8k.py --which int4 --n 100 --max-tokens 2048 \
  --litertlm "$1" --greedy --tag "$2"
