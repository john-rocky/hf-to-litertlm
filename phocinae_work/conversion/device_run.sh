#!/bin/bash
# Galaxy S26 gate for the Phocinae-Largha-150M-v1 LiteRT graphs (package com.phocinae.gate, debug build).
# usage: device_run.sh setup <file...>                 install the APK, push the named files (graphs, act head, rows, table) into files/
#        device_run.sh gate <accel> <precision> <graph> <rows-file> <L> [table-file] [limit]
#        device_run.sh ready                           print the thermal / frequency state (and whether every cap is at its top)
#        device_run.sh cleanup                         force-stop, uninstall, remove the staging dir
# Requires device/HOLD_ACQUIRED (hold_keeper.sh: the hold is owned by the keeper's pid for the whole batch).
# Never clears logcat; reads it from a device timestamp taken before each launch. Reports land in device/.
set -u
G=${W:-$(cd "$(dirname "$0")/.." && pwd)}
SERIAL=${SERIAL:-RFGL80R6A6H}
PKG=com.phocinae.gate
OUT=$G/device
STAGE=/data/local/tmp/phocinae_gate
APK=$G/android/gate/app/build/outputs/apk/debug/app-debug.apk
A() { adb -s "$SERIAL" "$@"; }
mkdir -p "$OUT"

state() {
  echo "uptime: $(A shell cat /proc/uptime | tr -d '\r')"
  echo "thermal: $(A shell dumpsys thermalservice | grep -m1 'Thermal Status' | tr -d '\r')"
  echo "battery_temp: $(A shell dumpsys battery | grep -m1 temperature | tr -d '\r ')"
  echo "screen: $(A shell dumpsys power | grep -m1 -E 'mWakefulness=' | tr -d '\r ')"
  echo "keyguard: $(A shell dumpsys window | grep -m1 -E 'isKeyguardShowing|mShowingLockscreen' | tr -d '\r')"
  echo "top: $(A shell dumpsys activity activities | grep -m1 topResumedActivity | tr -d '\r')"
  echo "cpufreq: $(A shell 'for p in /sys/devices/system/cpu/cpufreq/policy*; do echo -n "$(basename $p)=$(cat $p/scaling_max_freq)/$(cat $p/cpuinfo_max_freq) "; done' | tr -d '\r')"
  echo "kgsl: max_clock_mhz=$(A shell cat /sys/class/kgsl/kgsl-3d0/max_clock_mhz 2>/dev/null | tr -d '\r') thermal_pwrlevel=$(A shell cat /sys/class/kgsl/kgsl-3d0/thermal_pwrlevel 2>/dev/null | tr -d '\r') temp=$(A shell cat /sys/class/kgsl/kgsl-3d0/temp 2>/dev/null | tr -d '\r')"
}

ready() {
  # every CPU policy at its top frequency and the GPU not thermally limited (memory android-thermal-freq-cap-before-bench)
  local caps
  caps=$(A shell 'for p in /sys/devices/system/cpu/cpufreq/policy*; do [ "$(cat $p/scaling_max_freq)" = "$(cat $p/cpuinfo_max_freq)" ] || echo capped; done; [ "$(cat /sys/class/kgsl/kgsl-3d0/thermal_pwrlevel 2>/dev/null || echo 0)" = "0" ] || echo gpucapped' | tr -d '\r')
  [ -z "$caps" ]
}

case "$1" in
ready)
  state; if ready; then echo "READY: no cap"; else echo "NOT READY: a cap is active"; fi ;;
setup)
  shift
  [ -f "$OUT/HOLD_ACQUIRED" ] || { echo "no hold: $OUT/HOLD_ACQUIRED missing" >&2; exit 4; }
  state
  A install -r "$APK" || exit 5
  A shell mkdir -p $STAGE
  A shell run-as $PKG mkdir -p files
  for f in "$@"; do
    b=$(basename "$f")
    A push "$f" "$STAGE/$b" >/dev/null || exit 6
    A shell run-as $PKG cp "$STAGE/$b" "files/$b" || exit 7
    local_sha=$(shasum -a 256 "$f" | cut -c1-16); dev_sha=$(A shell run-as $PKG sha256sum "files/$b" | cut -c1-16 | tr -d '\r')
    [ "$local_sha" = "$dev_sha" ] || { echo "sha mismatch after push: $b $local_sha $dev_sha" >&2; exit 8; }
    echo "pushed $b ($dev_sha)"
  done
  A shell rm -rf $STAGE
  A shell run-as $PKG ls -l files
  ;;
gate)
  [ -f "$OUT/HOLD_ACQUIRED" ] || { echo "no hold: $OUT/HOLD_ACQUIRED missing" >&2; exit 4; }
  ACC=$2; PREC=$3; GRAPH=$4; ROWS=$5; L=$6; TABLE=${7:-}; LIM=${8:-0}
  STEM=${GRAPH%.tflite}; STEM=${STEM#phocinae_}
  TAG="phocinae_s26_${ACC}_${PREC}_${STEM}${TABLE:+_hostlookup}${LIM:+_n$LIM}"
  [ "$LIM" = 0 ] && TAG="phocinae_s26_${ACC}_${PREC}_${STEM}${TABLE:+_hostlookup}"
  TAG="${TAG}${REPORT_SUFFIX:-}"
  REPORT="$TAG.json"
  A shell am force-stop $PKG
  A shell run-as $PKG rm -f "files/$REPORT"
  # wait for a rested phone (up to READY_WAIT s), then record the state
  W=${READY_WAIT:-300}; S=$(date +%s)
  until ready; do if [ $(( $(date +%s) - S )) -gt $W ]; then echo "WARN: still capped after $W s, starting anyway"; break; fi; sleep 10; done
  state > "$OUT/$TAG.state_before.txt"
  T0=$(A shell "date '+%m-%d %H:%M:%S.000'" | tr -d '\r')
  # an empty --es value breaks am's argument parsing (the next flag becomes the value): pass the table only when set
  A shell am start -W -n $PKG/.GateActivity --es graph "$GRAPH" --es act phocinae_act_head_fp32.tflite --es accel "$ACC" --es precision "$PREC" \
     ${TABLE:+--es table "$TABLE"} --es rows "$ROWS" --es report "$REPORT" --ei seq "$L" --ei limit "$LIM" --ei timed "${TIMED:-20}" --ei cool_ms "${COOL_MS:-0}" --ei threads "${THREADS:-4}" > "$OUT/$TAG.start.txt"
  LIMIT=${LIMIT:-1200}; S=$(date +%s)
  until A shell run-as $PKG ls "files/$REPORT" >/dev/null 2>&1; do
    if [ $(( $(date +%s) - S )) -gt $LIMIT ]; then echo "TIMEOUT $TAG"; break; fi
    sleep 3
  done
  sleep 1
  A exec-out run-as $PKG cat "files/$REPORT" > "$OUT/$REPORT"
  PID=$(A shell pidof $PKG | tr -d '\r')
  A logcat -d -T "$T0" ${PID:+--pid=$PID} > "$OUT/$TAG.logcat.txt" 2>&1
  state > "$OUT/$TAG.state_after.txt"
  A shell am force-stop $PKG
  grep -iE "Replacing|Partition|accelerator|PHOCINAE_GATE (compiled|report)|DispatchDelegate|LITERT_CL|CompilerPlugin|not supported|Failed to|error" "$OUT/$TAG.logcat.txt" | grep -v "^$" | head -16 > "$OUT/$TAG.delegate.txt"
  cat "$OUT/$TAG.delegate.txt"
  head -3 "$OUT/$TAG.state_before.txt"
  python3 -c "import json,sys; d=json.load(open('$OUT/$REPORT')); print('status', d.get('status'), 'compile_ms', d.get('compile_ms'), 'summary', json.dumps(d.get('summary')), 'error', d.get('error'))"
  ;;
cleanup)
  A shell am force-stop $PKG
  A uninstall $PKG
  A shell rm -rf $STAGE
  state
  ;;
*) echo "usage: $0 setup <files...> | gate <accel> <precision> <graph> <rows> <L> [table] [limit] | ready | cleanup" >&2; exit 2 ;;
esac
