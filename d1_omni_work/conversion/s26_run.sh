#!/bin/bash
# Round 5: one leg of the d1-omni S26 gate app (android/d1omni_gate) = d1_3b_work's scripts/d1_s26_run.sh (round 4) with
# this lane's package, stage dir and the app's extras (sig, keep_scores; sel / full instead of hsel / hfull). Requires
# <out>/HOLD_ACQUIRED (scripts/s26_keeper.sh). Called by scripts/s26_chain.sh; by hand only for `state` / `cleanup` while
# the chain's hold is live.
#   usage: s26_run.sh state | gate <tag> <accel> <precision> <graph-file> <rows-file> <mode> | cleanup
# Env: D1_DEVICE_OUT (the round's dir, required), ADB (default: the SDK's adb; the dry run sets scripts/fake_adb.py),
#   LIMIT (s, default 600: then files/STOP, 40 s later TIMEOUT), GATE_SIG / GATE_THREADS / GATE_WARMUP / GATE_REPS /
#   GATE_REST_MS / GATE_COOL_MS / GATE_LIMIT_ROWS / GATE_KEEP_SCORES / GATE_CPU_CACHE / GATE_GPU_SRC_QUANT -> the app's extras.
# Package com.mlboydaisuke.d1omni.gate only; staging /data/local/tmp/d1omni_gate only; never clears logcat, never pm clear,
# never pkill (nothing is killed except by `am force-stop` of this package), no settings / reboot / Clock, no
# `timeout adb ...` (it kills only the host's client: the phone side keeps running).
# Exit 9 = the phone left adb (hard stop: the caller must not retry); the time goes to <out>/DEVICE_LOST.txt.
set -u
SERIAL=${D1_SERIAL:-RFGL80R6A6H}
ADB=${ADB:-adb}
PKG=com.mlboydaisuke.d1omni.gate
OUT=${D1_DEVICE_OUT:?D1_DEVICE_OUT = the round dir}
STAGE=/data/local/tmp/d1omni_gate
A() { "$ADB" -s "$SERIAL" "$@"; }
[ -f "$OUT/HOLD_ACQUIRED" ] || { echo "no hold: $OUT/HOLD_ACQUIRED missing" >&2; exit 4; }
present() { [ "$(A get-state 2>/dev/null | tr -d '\r')" = "device" ]; }
lost() { echo "DEVICE_LOST $(date '+%F %T') during: $1" | tee -a "$OUT/DEVICE_LOST.txt"; exit 9; }
present || lost "start of $1"

state() {
  echo "time: $(date '+%F %T')"
  echo "uptime: $(A shell cat /proc/uptime | tr -d '\r')"
  local th
  th=$(A shell dumpsys thermalservice | tr -d '\r')
  echo "thermal: $(echo "$th" | grep -m1 'Thermal Status')"
  echo "skin: $(echo "$th" | grep -m1 -oE 'mValue=[-0-9.]+, mType=3, mName=SKIN')"
  echo "hal_temps: $(echo "$th" | grep -oE 'Temperature\{mValue=[-0-9.]+, mType=[0-9]+, mName=[A-Za-z0-9_]+' | head -12 | sed 's/Temperature{//' | tr '\n' ';')"
  echo "battery_temp: $(A shell dumpsys battery | grep -m1 temperature | tr -d '\r ')"
  echo "battery_level: $(A shell dumpsys battery | grep -m1 level | tr -d '\r ')"
  echo "screen: $(A shell dumpsys power | grep -m1 -E 'mWakefulness=' | tr -d '\r ')"
  echo "top: $(A shell dumpsys activity activities | grep -m1 topResumedActivity | tr -d '\r')"
  echo "kgsl: temp=$(A shell cat /sys/class/kgsl/kgsl-3d0/temp 2>/dev/null | tr -d '\r') clock_mhz=$(A shell cat /sys/class/kgsl/kgsl-3d0/clock_mhz 2>/dev/null | tr -d '\r') max_clock_mhz=$(A shell cat /sys/class/kgsl/kgsl-3d0/max_clock_mhz 2>/dev/null | tr -d '\r') thermal_pwrlevel=$(A shell cat /sys/class/kgsl/kgsl-3d0/thermal_pwrlevel 2>/dev/null | tr -d '\r')"
  local fq
  fq=$(A shell 'for p in /sys/devices/system/cpu/cpufreq/policy*; do echo "${p##*/} $(cat $p/scaling_cur_freq) $(cat $p/scaling_max_freq) $(cat $p/cpuinfo_max_freq)"; done' | tr -d '\r')
  echo "freq (policy cur scaling_max cpuinfo_max): $(echo "$fq" | tr '\n' ';')"
  echo "freq_capped: $(echo "$fq" | awk '$3 != $4 {printf "%s:%s/%s ", $1, $3, $4; c=1} END {if (!c) printf "none"}')"
  echo "mem: $(A shell grep -E 'MemAvailable' /proc/meminfo | tr -d '\r')"
  echo "data_free: $(A shell df -h /data | tail -1 | tr -d '\r')"
}

case "$1" in
state)
  state
  ;;
gate)
  TAG=$2; ACC=$3; PREC=$4; GRAPH=$5; ROWSF=$6; MODE=$7
  REPORT="$TAG.json"; SEL="sel_$TAG.f32"; FULL="full_$TAG.f32"
  A shell am force-stop $PKG
  A shell run-as $PKG rm -f "files/$REPORT" "files/$REPORT.partial" "files/$SEL" "files/$FULL" files/STOP
  state > "$OUT/$TAG.state_before.txt"
  # quoted for the phone's shell: unquoted, `date` there splits the format at the space (Kev round 11)
  T0=$(A shell "date '+%m-%d %H:%M:%S.000'" | tr -d '\r')
  A shell am start -W -n $PKG/.GateActivity --es graph "$GRAPH" --es accel "$ACC" --es precision "$PREC" \
     --es rows "$ROWSF" --es report "$REPORT" --es mode "$MODE" --ei threads "${GATE_THREADS:-4}" \
     ${GATE_SIG:+--es sig "$GATE_SIG"} \
     ${GATE_LIMIT_ROWS:+--ei limit "$GATE_LIMIT_ROWS"} ${GATE_WARMUP:+--ei warmup "$GATE_WARMUP"} \
     ${GATE_REST_MS:+--ei rest_ms "$GATE_REST_MS"} ${GATE_REPS:+--ei reps "$GATE_REPS"} ${GATE_COOL_MS:+--ei cool_ms "$GATE_COOL_MS"} \
     ${GATE_KEEP_SCORES:+--ei keep_scores "$GATE_KEEP_SCORES"} ${GATE_CPU_CACHE:+--ei cpu_cache "$GATE_CPU_CACHE"} \
     ${GATE_GPU_SRC_QUANT:+--ei gpu_src_quant "$GATE_GPU_SRC_QUANT"} \
     > "$OUT/$TAG.start.txt" 2>&1
  sleep 1
  RPID=$(A shell pidof $PKG | tr -d '\r')
  { echo "pid: $RPID"; echo "cgroup: $(A shell cat /proc/$RPID/cgroup 2>&1 | tr -d '\r' | tr '\n' ' ')";
    echo "top: $(A shell dumpsys activity activities | grep -m1 topResumedActivity | tr -d '\r')"; } > "$OUT/$TAG.running.txt"
  LIMIT=${LIMIT:-600}; S=$(date +%s); LAST_TOP=$S; RESULT=DONE; STOP_SENT=0
  until A shell run-as $PKG ls "files/$REPORT" >/dev/null 2>&1; do
    present || lost "run $TAG"
    NOWS=$(date +%s)
    if [ "$STOP_SENT" = 0 ] && [ $(( NOWS - S )) -ge "$LIMIT" ]; then
      echo "LIMIT ${LIMIT}s reached for $TAG at $(date '+%T'): writing files/STOP"
      A shell run-as $PKG touch files/STOP; STOP_SENT=$NOWS; RESULT=STOPPED_AT_LIMIT
    fi
    if [ "$STOP_SENT" != 0 ] && [ $(( NOWS - STOP_SENT )) -gt 40 ]; then echo "TIMEOUT $TAG (no report 40 s after STOP)"; RESULT=TIMEOUT; break; fi
    if [ -z "$(A shell pidof $PKG | tr -d '\r')" ]; then
      sleep 2
      A shell run-as $PKG ls "files/$REPORT" >/dev/null 2>&1 && break
      echo "PROCESS_DIED $TAG at $(date '+%T')"; RESULT=PROCESS_DIED; break
    fi
    if [ $(( NOWS - LAST_TOP )) -ge 15 ]; then
      LAST_TOP=$NOWS
      TOP=$(A shell dumpsys activity activities | grep -m1 topResumedActivity | tr -d '\r')
      case "$TOP" in
        *"$PKG"*) ;;
        *) echo "$(date '+%F %T') $TOP" >> "$OUT/$TAG.foreground_lost.txt"; echo "FOREGROUND_LOST (still waiting) $TAG: $TOP" ;;
      esac
    fi
    sleep 3
  done
  sleep 1
  A exec-out run-as $PKG cat "files/$REPORT" > "$OUT/$REPORT" 2>/dev/null
  [ -s "$OUT/$REPORT" ] || rm -f "$OUT/$REPORT"
  if [ "$MODE" = "gate" ] && [ -s "$OUT/$REPORT" ]; then
    # pulled only when the report names the file (a FAILED report has none: no empty file is left behind)
    for kind in sel full; do
      [ $kind = full ] && [ "${GATE_KEEP_SCORES:-0}" != 1 ] && continue
      F=$SEL; [ $kind = full ] && F=$FULL
      want=$(python3 -I -c "import json,sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], ''))" "$OUT/$REPORT" "${kind}_bytes")
      if [ -z "$want" ]; then echo "$kind: none in the report"; continue; fi
      A exec-out run-as $PKG cat "files/$F" > "$OUT/$F" 2>/dev/null
      got=$(stat -f %z "$OUT/$F" 2>/dev/null)
      echo "$kind bytes: report=$want pulled=$got $([ "$want" = "$got" ] && echo MATCH || echo CHECK)"
    done
  fi
  PID=$(A shell pidof $PKG | tr -d '\r'); LPID=${PID:-$RPID}
  A logcat -d -T "$T0" ${LPID:+--pid=$LPID} > "$OUT/$TAG.logcat.txt" 2>&1
  A logcat -b all -d -T "$T0" 2>/dev/null | grep -E "d1omni\.gate|D1OMNI_GATE" > "$OUT/$TAG.logcat_pkg_all_buffers.txt"
  state > "$OUT/$TAG.state_after.txt"
  A shell am force-stop $PKG
  A shell run-as $PKG rm -f "files/$SEL" "files/$FULL"
  echo "result=$RESULT status=$(python3 -I -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get('status'), d.get('error', ''))" "$OUT/$REPORT" 2>/dev/null || echo NO_REPORT) pid=${LPID:-none} wall_s=$(( $(date +%s) - S ))"
  grep -E "Replacing|Partitioned|LITERT_CL|D1OMNI_GATE|Failed|failed|rror" "$OUT/$TAG.logcat.txt" | head -15
  grep -iE "lowmemory|lmk|kill" "$OUT/$TAG.logcat_pkg_all_buffers.txt" | head -5
  grep -E "thermal|skin|battery_temp|freq_capped|uptime|kgsl|mem:" "$OUT/$TAG.state_before.txt" "$OUT/$TAG.state_after.txt"
  cat "$OUT/$TAG.running.txt"
  ;;
cleanup)
  A shell am force-stop $PKG
  A uninstall $PKG
  A shell rm -rf $STAGE
  state
  echo "package left: '$(A shell pm list packages $PKG | tr -d '\r')'"
  echo "stage left: $(A shell ls -d $STAGE 2>&1 | tr -d '\r')"
  ;;
*)
  echo "usage: $0 state | gate <tag> <accel> <precision> <graph> <rows> <mode> | cleanup" >&2; exit 2
  ;;
esac
