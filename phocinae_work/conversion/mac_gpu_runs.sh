#!/bin/bash
# Mac Metal correctness runs (one quiet_hold window for the batch). usage: mac_gpu_runs.sh <L> <graph...> ; env TABLE=1 for the embeds form
set -u
W=${W:-$(cd "$(dirname "$0")/.." && pwd)}
P=${P:-python3}
L=$1; shift
for g in "$@"; do
  extra=""; case "$g" in *embeds*) extra="--table exports/tables/tok_embeddings_fp16.bin";; esac
  for mode in "--f32" ""; do
    echo "== $g gpu $mode $(date '+%T')"
    "$P" "$W/scripts/litert_parity.py" --tflite "$g" --L "$L" --accel gpu $mode $extra ${EXTRA:-} 2>&1 | grep -v "W1010\|Warning\|NOTE:\|absl\|I0000\|INFO" | tail -c 1500
  done
done
echo "== done $(date '+%T')"
