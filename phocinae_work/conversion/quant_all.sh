#!/bin/bash
# Quantize one fp32 export into the lane's storage variants (sequential). usage: quant_all.sh <fp32 tflite relative to phocinae_work> [variants...]
set -u
W=${W:-$(cd "$(dirname "$0")/.." && pwd)}
P=${P:-python3}
SRC=$1; shift
VARS=${*:-"v2_fp16fc_i8emb v3_fp16fc_fp16emb v1_wi8fc_i8emb v5_i8emb"}
for v in $VARS; do
  echo "== $v $(date '+%T')"
  "$P" "$W/scripts/quantize.py" --src "$SRC" --variant "$v" 2>&1 | grep -v "W1010\|Warning\|NOTE:\|absl\|I0000"
done
echo "== done $(date '+%T')"
