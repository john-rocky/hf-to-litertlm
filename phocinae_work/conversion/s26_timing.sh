#!/bin/bash
# Timing-only legs on a rested phone (the long parity legs heat the SoC and cap the clocks mid-leg; memory
# android-thermal-freq-cap-before-bench): for each ship candidate, wait for thermal status 0 + no cap + a rest, then one
# short gate (5 rows) followed by the timing rows (5 warm-ups + 20 calls each). Reports get the suffix _timing.
# usage: s26_timing.sh   (waits for device/followon.log "followon done"; env NOWAIT=1 skips the wait)
set -u
W=${W:-$(cd "$(dirname "$0")/.." && pwd)}; cd "$W"
A() { adb -s RFGL80R6A6H "$@"; }
[ -n "${NOWAIT:-}" ] || until grep -q "followon done" device/followon.log; do sleep 15; done
[ -f device/HOLD_ACQUIRED ] || { echo "hold lost"; exit 4; }
rest() {
  S=$(date +%s)
  while :; do
    st=$(A shell dumpsys thermalservice | grep -m1 'Thermal Status' | tr -dc '0-9')
    if [ "${st:-9}" = "0" ] && scripts/device_run.sh ready | grep -q "READY"; then break; fi
    [ $(( $(date +%s) - S )) -gt 900 ] && { echo "rest timeout (thermal $st)"; break; }
    sleep 15
  done
  sleep ${REST_EXTRA:-45}
}
LEGS=${LEGS:-"256:gpu:fp32:ids 256:gpu:fp16acc32:ids 256:gpu:fp16acc32:embeds 256:gpu:fp16:ids 256:gpu:fp16:embeds 256:npu:fp32:embeds 256:cpu:fp32:ids 256:cpu:fp32:embeds 384:gpu:fp32:ids 384:gpu:fp16acc32:embeds 384:gpu:fp16:embeds 384:npu:fp32:embeds 384:cpu:fp32:ids 512:gpu:fp32:ids 512:gpu:fp16acc32:embeds 512:gpu:fp16:embeds 512:npu:fp32:embeds 512:cpu:fp32:ids"}
for leg in $LEGS; do
  IFS=: read -r L ACC PREC FORM <<< "$leg"
  if [ "$FORM" = embeds ]; then G=phocinae_L${L}_embeds_wfp16_shift.tflite; T=tok_embeddings_fp16.bin; else G=phocinae_L${L}_ids_v2_fp16fc_i8emb_shift.tflite; T=""; fi
  A shell run-as com.phocinae.gate ls files/$G >/dev/null 2>&1 || scripts/device_run.sh setup exports/$G fixtures/device_rows_L$L.json
  A shell run-as com.phocinae.gate ls files/device_rows_L$L.json >/dev/null 2>&1 || scripts/device_run.sh setup fixtures/device_rows_L$L.json
  rest
  echo "== timing leg $leg $(date '+%T') thermal=$(A shell dumpsys thermalservice | grep -m1 'Thermal Status' | tr -dc '0-9') kgsl=$(A shell cat /sys/class/kgsl/kgsl-3d0/max_clock_mhz | tr -d '\r')"
  REPORT_SUFFIX=_timing COOL_MS=3000 READY_WAIT=600 scripts/device_run.sh gate "$ACC" "$PREC" "$G" device_rows_L$L.json $L "$T" 5
done
echo "== timing legs done $(date '+%T')"
