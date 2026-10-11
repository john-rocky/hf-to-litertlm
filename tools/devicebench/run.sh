#!/bin/bash
# run.sh: one leg of the lane's measurement app on the phone. Requires <round dir>/HOLD_ACQUIRED (the keeper's). Called
# by chain.sh; by hand only for `state` / `cleanup` while the chain's hold is live:
#   LANE_ENV=<lane.env> DEVICE_OUT=<round dir> bash run.sh state | cleanup
#   usage: run.sh state | gate <tag> <accel> <precision> <graph-file> <rows-file> <mode> | cleanup
# Env: LANE_ENV (the lane's settings: PKG, STAGE, SERIAL, ACTIVITY, APP_INT_EXTRAS, APP_STR_EXTRAS, APP_OUTPUTS,
#   APP_LOG_TAG / APP_LOG_RE), DEVICE_OUT (the round's dir), ADB (the dry run sets fake_adb.py), LIMIT (s, default 600:
#   then files/STOP, 40 s later TIMEOUT), LEG_SIG (the app's `sig` extra, when not empty), LEG_EXTRAS (the plan's extras:
#   threads, default 4, and every key of APP_INT_EXTRAS / APP_STR_EXTRAS that has a value).
# The app contract: am start -W -n <PKG>/<ACTIVITY> with --es graph accel precision rows report mode, --ei threads, the
# extras above; the app reads files/<graph> and files/<rows>, writes files/<tag>.json (status DONE / FAILED ..., and
# <kind>_bytes for each output file <kind>_<tag>.f32 of APP_OUTPUTS, pulled after a gate leg; a "<kind>:<extra>" output
# is pulled only when that extra is 1), stops after the current row when files/STOP appears, and stays alive until
# force-stopped. This package only; the stage dir only; never clears logcat, never pm clear, never pkill (nothing is
# killed except by `am force-stop` of this package), no settings / reboot / Clock, no `timeout adb ...` (it kills only
# the host's client: the phone side keeps running). Exit 9 = the phone left adb (hard stop: the caller must not retry);
# the time goes to <round dir>/DEVICE_LOST.txt.
set -u
# shellcheck source=lib.sh
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
bench_load "${LANE_ENV:?LANE_ENV = the lane.env}"
[ -n "$PKG" ] && [ -n "$STAGE" ] || { echo "$LANE_ENV: PKG and STAGE are needed" >&2; exit 2; }
OUT=${DEVICE_OUT:?DEVICE_OUT = the round dir}
[ -f "$OUT/HOLD_ACQUIRED" ] || { echo "no hold: $OUT/HOLD_ACQUIRED missing" >&2; exit 4; }
lost() { echo "DEVICE_LOST $(date '+%F %T') during: $1" | tee -a "$OUT/DEVICE_LOST.txt"; exit 9; }
present || lost "start of ${1:-}"

out_files() {  # tag -> the app's output file names of APP_OUTPUTS, one per line
  local spec
  for spec in $APP_OUTPUTS; do echo "${spec%%:*}_$1.f32"; done
}

case "${1:-}" in
state)
  phone_state
  ;;
gate)
  TAG=$2; ACC=$3; PREC=$4; GRAPH=$5; ROWSF=$6; MODE=$7
  REPORT="$TAG.json"; EXTRAS=${LEG_EXTRAS:-}
  A shell am force-stop "$PKG"
  # shellcheck disable=SC2046
  A shell run-as "$PKG" rm -f "files/$REPORT" "files/$REPORT.partial" $(out_files "$TAG" | sed 's|^|files/|') files/STOP
  phone_state > "$OUT/$TAG.state_before.txt"
  # quoted for the phone's shell: unquoted, `date` there splits the format at the space
  T0=$(A shell "date '+%m-%d %H:%M:%S.000'" | tr -d '\r')
  set -- am start -W -n "$PKG/$ACTIVITY" --es graph "$GRAPH" --es accel "$ACC" --es precision "$PREC" \
     --es rows "$ROWSF" --es report "$REPORT" --es mode "$MODE" --ei threads "$(xget "$EXTRAS" threads 4)"
  [ -n "${LEG_SIG:-}" ] && set -- "$@" --es sig "$LEG_SIG"
  # an extra goes to the app only with a value (`--es key ""` makes am read the next flag as the value)
  for spec in $APP_INT_EXTRAS; do v=$(xget "$EXTRAS" "${spec%%:*}" ""); [ -n "$v" ] && set -- "$@" --ei "${spec#*:}" "$v"; done
  for spec in $APP_STR_EXTRAS; do v=$(xget "$EXTRAS" "${spec%%:*}" ""); [ -n "$v" ] && set -- "$@" --es "${spec#*:}" "$v"; done
  A shell "$@" > "$OUT/$TAG.start.txt" 2>&1
  sleep 1
  RPID=$(A shell pidof "$PKG" | tr -d '\r')
  { echo "pid: $RPID"; echo "cgroup: $(A shell cat "/proc/$RPID/cgroup" 2>&1 | tr -d '\r' | tr '\n' ' ')";
    echo "top: $(A shell dumpsys activity activities | grep -m1 topResumedActivity | tr -d '\r')"; } > "$OUT/$TAG.running.txt"
  LIMIT=${LIMIT:-600}; S=$(date +%s); LAST_TOP=$S; RESULT=DONE; STOP_SENT=0
  until A shell run-as "$PKG" ls "files/$REPORT" >/dev/null 2>&1; do
    present || lost "run $TAG"
    NOWS=$(date +%s)
    if [ "$STOP_SENT" = 0 ] && [ $(( NOWS - S )) -ge "$LIMIT" ]; then
      echo "LIMIT ${LIMIT}s reached for $TAG at $(date '+%T'): writing files/STOP"
      A shell run-as "$PKG" touch files/STOP; STOP_SENT=$NOWS; RESULT=STOPPED_AT_LIMIT
    fi
    if [ "$STOP_SENT" != 0 ] && [ $(( NOWS - STOP_SENT )) -gt 40 ]; then echo "TIMEOUT $TAG (no report 40 s after STOP)"; RESULT=TIMEOUT; break; fi
    if [ -z "$(A shell pidof "$PKG" | tr -d '\r')" ]; then
      sleep 2
      A shell run-as "$PKG" ls "files/$REPORT" >/dev/null 2>&1 && break
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
  A exec-out run-as "$PKG" cat "files/$REPORT" > "$OUT/$REPORT" 2>/dev/null
  # exec-out carries the phone's stderr in the same stream: a report that is not there arrives as cat's error text,
  # which is kept aside (not as the leg's report)
  if [ -s "$OUT/$REPORT" ] && ! python3 -I -c "import json,sys; json.load(open(sys.argv[1]))" "$OUT/$REPORT" 2>/dev/null; then
    mv "$OUT/$REPORT" "$OUT/$TAG.report_pull.txt"
    echo "report: not JSON, kept as $TAG.report_pull.txt: $(head -c 160 "$OUT/$TAG.report_pull.txt" | tr '\n' ' ')"
  fi
  [ -s "$OUT/$REPORT" ] || rm -f "$OUT/$REPORT"
  if [ "$MODE" = "gate" ] && [ -s "$OUT/$REPORT" ]; then
    # pulled only when the report names the file (a FAILED report has none: no empty file is left behind)
    for spec in $APP_OUTPUTS; do
      kind=${spec%%:*}; F=${kind}_$TAG.f32
      case "$spec" in *:*) [ "$(xget "$EXTRAS" "${spec#*:}" 0)" = 1 ] || continue ;; esac
      want=$(python3 -I -c "import json,sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], ''))" "$OUT/$REPORT" "${kind}_bytes")
      if [ -z "$want" ]; then echo "$kind: none in the report"; continue; fi
      A exec-out run-as "$PKG" cat "files/$F" > "$OUT/$F" 2>/dev/null
      got=$(stat -f %z "$OUT/$F" 2>/dev/null)
      echo "$kind bytes: report=$want pulled=$got $([ "$want" = "$got" ] && echo MATCH || echo CHECK)"
    done
  fi
  PID=$(A shell pidof "$PKG" | tr -d '\r'); LPID=${PID:-$RPID}
  A logcat -d -T "$T0" ${LPID:+--pid=$LPID} > "$OUT/$TAG.logcat.txt" 2>&1
  A logcat -b all -d -T "$T0" 2>/dev/null | grep -E "$APP_LOG_RE" > "$OUT/$TAG.logcat_pkg_all_buffers.txt"
  phone_state > "$OUT/$TAG.state_after.txt"
  A shell am force-stop "$PKG"
  # shellcheck disable=SC2046
  A shell run-as "$PKG" rm -f $(out_files "$TAG" | sed 's|^|files/|')
  echo "result=$RESULT status=$(python3 -I -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get('status'), d.get('error', ''))" "$OUT/$REPORT" 2>/dev/null || echo NO_REPORT) pid=${LPID:-none} wall_s=$(( $(date +%s) - S ))"
  grep -E "Replacing|Partitioned|LITERT_CL|${APP_LOG_TAG:+$APP_LOG_TAG|}Failed|failed|rror" "$OUT/$TAG.logcat.txt" | head -15
  grep -iE "lowmemory|lmk|kill" "$OUT/$TAG.logcat_pkg_all_buffers.txt" | head -5
  grep -E "thermal|skin|battery_temp|freq_capped|uptime|kgsl|gpu:|cooling:|mem:" "$OUT/$TAG.state_before.txt" "$OUT/$TAG.state_after.txt"
  cat "$OUT/$TAG.running.txt"
  ;;
cleanup)
  A shell am force-stop "$PKG"
  A uninstall "$PKG"
  A shell rm -rf "$STAGE"
  phone_state
  echo "package left: '$(A shell pm list packages "$PKG" | tr -d '\r')'"
  echo "stage left: $(A shell ls -d "$STAGE" 2>&1 | tr -d '\r')"
  ;;
*)
  echo "usage: $0 state | gate <tag> <accel> <precision> <graph> <rows> <mode> | cleanup" >&2; exit 2
  ;;
esac
