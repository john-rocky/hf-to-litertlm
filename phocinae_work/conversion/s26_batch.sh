#!/bin/bash
# S26 legs for one window. usage: s26_batch.sh <L> <ids-graph> <embeds-graph> [legs...]   (legs default: all)
# leg = <accel>:<precision>:<form>  form = ids | embeds. Requires device/HOLD_ACQUIRED (hold_keeper.sh).
set -u
W=${W:-$(cd "$(dirname "$0")/.." && pwd)}
P=${P:-python3}
cd "$W"
L=$1; IDS=$2; EMB=$3; shift 3
LEGS=${*:-"gpu:fp32:ids gpu:fp32:embeds gpu:fp16acc32:ids gpu:fp16acc32:embeds gpu:fp16:ids gpu:fp16:embeds npu:fp32:embeds npu:fp32:ids cpu:fp32:ids cpu:fp32:embeds"}
ROWS=device_rows_L$L.json
[ -f device/HOLD_ACQUIRED ] || { echo "no hold"; exit 4; }
if [ -z "${SKIP_SETUP:-}" ]; then
  echo "== setup $(date '+%T')"
  scripts/device_run.sh setup exports/phocinae_act_head_fp32.tflite fixtures/$ROWS exports/$IDS exports/$EMB exports/tables/tok_embeddings_fp16.bin || exit 5
fi
for leg in $LEGS; do
  IFS=: read -r ACC PREC FORM <<< "$leg"
  if [ "$FORM" = embeds ]; then G=$EMB; T=tok_embeddings_fp16.bin; else G=$IDS; T=""; fi
  echo "== leg $leg $G $(date '+%T')"
  LIMIT=$([ "$ACC" = npu ] && echo 2400 || echo 1200) COOL_MS=${COOL_MS:-5000} scripts/device_run.sh gate "$ACC" "$PREC" "$G" "$ROWS" "$L" "$T"
  STEM=${G%.tflite}; STEM=${STEM#phocinae_}
  REP="device/phocinae_s26_${ACC}_${PREC}_${STEM}${T:+_hostlookup}.json"
  [ -f "$REP" ] && "$P" scripts/device_compare.py "$REP" 2>&1 | grep -v "W1010\|Warning\|NOTE:" | tail -c 1800
  sleep ${REST_S:-20}
done
echo "== batch done $(date '+%T')"
