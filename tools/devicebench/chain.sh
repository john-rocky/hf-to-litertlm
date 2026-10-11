#!/bin/bash
# chain.sh: the device gate chain for a measurement app (an Android activity that runs one graph over a rows file and
# writes a report), shared by every lane. A lane keeps only its settings (<lane>/device/lane.env), its plans, rows,
# graphs, APK, host scorer and stand-in; this file, lib.sh, run.sh and keeper.sh are not copied into lanes.
#   bash chain.sh check <round> <lane.env>         the plan and its inputs, sha256s; no adb, no hold
#   bash chain.sh mac-baseline <round> <lane.env>  the Mac CPU same-file run of every gate (and generic) leg's graph and
#                                                  rows through the lane's stand-in, then lane_score_mac; no adb; heavy:
#                                                  run it in the Mac's measurement window
#   bash chain.sh run <round> <lane.env>           the phone: hold, legs, cleanup, release
#   bash chain.sh cleanup <round> <lane.env>       by hand after a hard stop: uses the dead chain's live hold or takes
#                                                  the phone again, removes the app, releases
# (LANE_ENV in the environment can stand for the third argument.)
# Round dir R = <device dir>/<round>/ (ROUND_DIR overrides). Plan = PLAN, else the first of R/plan_<round>.tsv,
# R/plan.tsv, <device dir>/plan_<round>.tsv. One leg per line (# comments; whitespace-separated; extras key=value):
#   tag  limit_s  accel  precision  graph  rows  mode  ready  [extras]
#   tag = the leg's name (report R/<tag>.json); limit_s = the app's time limit (capped so the leg ends by RUN_END_MIN);
#   accel gpu | cpu; precision fp32 | fp16acc32 | fp16 | default (gpu) or - (cpu); graph = a file of GRAPH_DIR (the app
#   gets sig=<SIG_OF applied to the name> when lane.env sets SIG_OF); rows = a file of ROWS_DIR (the side files it names
#   go along: every rows[].prefix_file or sets[].rows[].prefix_file, and a generic rows file's inputs[].file); mode
#   gate | timing | generic | generic_timing; ready = the ready gate's mode: clean (wait for a clean phone up to
#   REST_MAX, then run warm), gate (do not wait for clean, but never at thermal status >= 4), timing (run only from
#   clean, else leave it for the next hold) or uncapped (run when no CPU policy is capped, the GPU's thermal level is 0
#   (kgsl thermal_pwrlevel, or on Mali the largest state of its GPU cooling devices, mType 3) and the thermal status is
#   at most 1; else as timing).
#   extras: rest=<s> (rest since the previous leg, default 60), keep=<graph,...> (graphs that stay on the phone),
#   guard=<kB> (memory guard: MemAvailable below it -> am force-stop; default MEM_GUARD_KB; 0 = off), sample=<s> (the
#   sampler's period, default 2), poll=<s> (the ready gate's poll period), needs=<tag> (run only when that leg is
#   DONE), needs_pass=<tag> (run only when that gate leg's verdict, written by the lane's scorer, says PASS; waits up to
#   VERDICT_WAIT_S for it), once=1 (one attempt per round dir, whatever its result), whole=1 (start only when the whole
#   limit fits before RUN_END_MIN), prep=<kind>:<leg> (a generic leg whose rows file and inputs lane_prep makes from
#   that earlier generic leg's pulled output; skipped when that leg is not DONE), threads=<n> (default 4), and the
#   app's extras named in APP_INT_EXTRAS / APP_STR_EXTRAS (an extra goes to the app only when it has a value).
# A leg whose report says DONE is skipped, so a later hold continues the plan where the last one stopped.
# Order of a run: keeper (keeper.sh, from a copy in R: queue_cli.py wait <hold file> <script name> <keeper pid>) ->
# HOLD_ACQUIRED -> stay-on read (never changed) -> the GPU's driver read once (kgsl, mali or none: where the sampler and
# the ready gate read the GPU) -> where the cooling devices are read, once (the thermal service's list, else
# /sys/class/thermal; R/s_gpu.txt) -> the thermal service's dump as it is and the sysfs cooling devices, once
# (R/s_thermal_raw.txt: its cached and current temperatures and its cooling devices with their types;
# R/s_cooling_sysfs.txt) -> transfer check (another session's adb transfer on the Mac; this app,
# a *runner process, or another app matching PEER_APP_RE that is in front or whose CPU time grows, on the phone = no
# leg, release) -> /proc/uptime -> the APK (sha256 checked) installed -> the plan's rows files and their side files into
# the app's files/ -> per leg: the graph into files/ (exec-in, byte count polled, toybox sha256sum on the phone; other
# graphs and their XNNPACK cache files removed unless kept, /data free checked) -> ready gate -> the leg (run.sh: am
# start, poll, STOP at the limit, pull, logcat by pid) with the 2 s sampler (device clock, the GPU's clock and level,
# CPU caps, MemAvailable, the app's VmHWM / VmRSS / VmSwap, the GPU's memory), the foreground check and the memory guard
# -> lmkd (lines naming this package only) / memory guard / reboot (/proc/uptime went down) / other apps' CPU TIME ->
# the lane's host scorer of a DONE leg in the background (lane_score) -> cleanup (force-stop, uninstall, the stage dir
# removed, checked) -> RELEASE_HOLD -> the keeper releases.
# Clock (minutes after the hold): no leg starts after SKIP_MIN (25), every leg's limit ends by RUN_END_MIN (28), the
# keeper releases at END_MIN (30) at the latest. Exit 9 = the phone left adb or rebooted (hard stop; the hold is
# released). Exit 5 = inputs, setup or another session's job on the phone (no leg). Exit 3 = no hold.
# The settings (lane.env; README.md lists them): LANE PKG STAGE SERIAL HOLD (required), DEVICE GRAPH_DIR ROWS_DIR
# RESULTS_DIR APK APK_SHA SCRIPT_NAME ACTIVITY SIG_OF APP_INT_EXTRAS APP_STR_EXTRAS APP_OUTPUTS APP_LOG_TAG APP_LOG_RE
# PEER_APP_RE PY ADB STANDIN MEM_GUARD_KB and the clock, and the hooks lane_score / lane_score_mac / lane_prep.
# DRY_RUN=1 = the fake phone (dryrun.sh): ADB must be fake_adb.py and the hold file, ROUND_DIR and RESULTS_DIR must lie
# under this folder's cache/.
set -u
CHAIN_NAME=chain.sh
# shellcheck source=lib.sh
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
CMD=${1:-}; ROUND=${2:-}
[ -n "$ROUND" ] && [ -n "${3:-${LANE_ENV:-}}" ] || { echo "usage: $0 check | mac-baseline | run | cleanup <round> <lane.env>" >&2; exit 2; }
bench_load "${3:-$LANE_ENV}"
for v in PKG STAGE; do
  [ -n "${!v:-}" ] || { echo "$LANE_ENV: $v is not set (chain.sh needs it)" >&2; exit 2; }
done
PLAN=$(plan_file)
RUN=$BENCH/run.sh
OWN_NAME=$PKG; STAGE_DIR=$STAGE
OWN_PROC_RE="$PKG_RE|[A-Za-z0-9_.]*runner"
export DEVICE_OUT=$R
dry_run_guard

status_of() { python3 -I -c "import json,sys; print(json.load(open(sys.argv[1])).get('status','NONE'))" "$R/$1.json" 2>/dev/null || echo NONE; }
done_leg() { [ "$(status_of "$1")" = DONE ]; }
sig_of() { [ -n "$SIG_OF" ] && echo "$1" | sed -nE "$SIG_OF"; }
prefix_files_of() {  # rows file -> the prefix files its rows (or timing sets) name, one per line
  python3 -I -c "
import json, sys
d = json.load(open(sys.argv[1]))
rows = d.get('rows') or [r for s in d.get('sets', []) for r in s['rows']]
print('\n'.join(sorted({r['prefix_file'] for r in rows if isinstance(r, dict) and r.get('prefix_file')})))" "$1"
}
generic_files_of() {  # rows file -> the input files a generic rows file names, one per line; nothing otherwise
  python3 -I -c "
import json, sys
d = json.load(open(sys.argv[1]))
if d.get('kind') == 'generic':
    print('\n'.join(sorted({i['file'] for i in d['inputs']})))" "$1"
}
input_sha() {  # name -> its sha256 from R/inputs.sha256, else from R/gen.sha256 (files made during the run)
  local s
  s=$(awk -v f="$1" '$2 == f {print $1}' "$R/inputs.sha256")
  [ -z "$s" ] && [ -s "$R/gen.sha256" ] && s=$(awk -v f="$1" '$2 == f {s = $1} END {print s}' "$R/gen.sha256")
  echo "$s"
}
on_device() { A shell run-as "$PKG" ls files 2>/dev/null | tr -d '\r' | grep '\.tflite$'; }

dev_sha() {  # name -> MATCH / CHECK line; fails unless both sides are non-empty and equal
  local name=$1 want got
  want=$(input_sha "$name"); got=$(A shell run-as "$PKG" toybox sha256sum "files/$name" 2>/dev/null | cut -d' ' -f1 | tr -d '\r')
  echo "sha256 $name phone=${got:-<none>} mac=${want:-<none>} $([ -n "$got" ] && [ -n "$want" ] && [ "$got" = "$want" ] && echo MATCH || echo CHECK)"
  [ -n "$got" ] && [ -n "$want" ] && [ "$got" = "$want" ]
}

xfer() {  # src name: exec-in into the app's files/ (no staged copy); the client may return before the phone has
  # written everything -> poll the byte count until it equals the source's, or stops growing for 30 s, or XFER_MAX_S;
  # no `timeout` on the client
  local src=$1 name=$2 want got last=-1 still=0 t0 rc
  want=$(stat -L -f %z "$src"); t0=$(now)
  A exec-in "run-as $PKG sh -c 'cat > files/$name'" < "$src"; rc=$?
  present || hard_stop "device_lost_in_transfer_$name"
  while :; do
    got=$(A shell run-as "$PKG" stat -c %s "files/$name" 2>/dev/null | tr -d '\r')
    [ -n "$got" ] && [ "$got" = "$want" ] && break
    if [ "$got" = "$last" ]; then still=$(( still + 5 )); else still=0; fi
    last=$got
    { [ "$still" -ge 30 ] || [ $(( $(now) - t0 )) -ge "$XFER_MAX_S" ]; } && break
    sleep 5
  done
  echo "exec-in $name rc=$rc bytes ${got:-<none>}/$want in $(( $(now) - t0 ))s"
  if [ "$rc" = 0 ] && [ -n "$got" ] && [ "$got" = "$want" ]; then return 0; fi
  A shell run-as "$PKG" rm -f "files/$name"; return 1
}

push_cp() {  # src name: a small file staged in STAGE, copied into files/ with run-as, the staged copy removed
  local src=$1 name=$2 want got
  want=$(stat -L -f %z "$src")
  A shell mkdir -p "$STAGE"
  A push "$src" "$STAGE/$name" >/dev/null || { present || hard_stop "device_lost_in_push_$name"; return 1; }
  got=$(A shell stat -c %s "$STAGE/$name" | tr -d '\r')
  [ -n "$got" ] && [ "$got" = "$want" ] || { echo "staged size mismatch $name: $got != $want"; A shell rm -f "$STAGE/$name"; return 1; }
  A shell run-as "$PKG" cp "$STAGE/$name" "files/$name" || { A shell rm -f "$STAGE/$name"; return 1; }
  got=$(A shell run-as "$PKG" stat -c %s "files/$name" | tr -d '\r')
  A shell rm -f "$STAGE/$name"
  echo "push+cp $name bytes $got/$want"
  [ -n "$got" ] && [ "$got" = "$want" ]
}

send_small() {  # src name: exec-in, else the staged push; then the sha256 on the phone
  xfer "$1" "$2" || push_cp "$1" "$2" || return 1
  dev_sha "$2"
}

ensure_graph() {  # name [keep ...]: the graph in files/ with its sha256 checked; other graphs not kept removed first,
  # each with its XNNPACK weight cache file (an app's cpu_cache=1 writes files/<graph>.xnnpack_cache next to the graph:
  # the /data check must count the space that cache held)
  local g=$1; shift
  local keep=" $g $* " x need free
  if on_device | grep -qx "$g" && dev_sha "$g"; then echo "graph $g already on the phone"; return 0; fi
  echo "files before the swap to $g: $(A shell run-as "$PKG" ls -l files 2>&1 | tr -d '\r' | awk 'NF >= 5 {printf "%s=%s ", $NF, $5}')"
  for x in $(on_device); do
    case "$keep" in *" $x "*) ;; *) A shell "run-as $PKG sh -c 'rm -f files/$x files/$x.xnnpack_cache*'"; echo "removed $x (and files/$x.xnnpack_cache* if any)" ;; esac
  done
  need=$(( $(stat -L -f %z "$GRAPH_DIR/$g") + DATA_MARGIN )); free=$(avail_bytes)
  [ "${free:-0}" -ge "$need" ] || { echo "data free ${free:-?} < $need for $g"; return 6; }
  xfer "$GRAPH_DIR/$g" "$g" || return 6
  dev_sha "$g" || return 7
  echo "data free after $g: $(avail_bytes) B"
}

swap_to() {  # name [keep ...]: ensure_graph with its log; a failure is written to s_skipped.txt
  local rc
  ( ensure_graph "$@" ) >> "$R/s_graphs.txt" 2>&1; rc=$?
  present || hard_stop "device_lost_in_graph_swap"
  [ $rc = 0 ] || { echo "graph swap to $1 failed rc $rc $(ts)" | tee -a "$R/s_skipped.txt"; return 1; }
  echo "graph $1 ready $(ts)"
}

# the sampler's commands (lib.sh sample_cmd / slow_cmd), set once the GPU's driver is read after the hold
SAMPLE_CMD=""; SLOW_CMD=""

monitor() {  # tag runner_pid period_s mem_guard_kb: foreground check, 2 s samples, wake keeping, memory guard
  local tag=$1 rp=$2 per=$3 guard=$4 n=0 fg=pending pid cg top out w every ma
  every=$(( 15 / per )); [ $every -lt 1 ] && every=1
  while kill -0 "$rp" 2>/dev/null; do
    if [ "$fg" = pending ] && [ -s "$R/$tag.running.txt" ] && grep -q '^top:' "$R/$tag.running.txt"; then
      # no process 1 s after the start = it died or never started: not a foreground loss (the report or the runner's
      # PROCESS_DIED decides the leg)
      if grep -q '^pid: *$' "$R/$tag.running.txt"; then
        fg=gone; echo "$(ts) no app process at the foreground check" >> "$R/$tag.fg.txt"
      elif grep -q 'cpuset:/top-app' "$R/$tag.running.txt" && grep '^top:' "$R/$tag.running.txt" | grep -q "$PKG"; then
        fg=ok; echo "$(ts) foreground ok at start (cpuset top-app, the activity top-resumed)" >> "$R/$tag.fg.txt"
      else
        A shell input keyevent KEYCODE_WAKEUP; sleep 3
        pid=$(A shell pidof "$PKG" | tr -d '\r')
        if [ -z "$pid" ]; then
          fg=gone; echo "$(ts) not foreground at start; no app process after KEYCODE_WAKEUP" >> "$R/$tag.fg.txt"
        else
          cg=$(A shell cat "/proc/$pid/cgroup" 2>&1 | tr -d '\r' | tr '\n' ' ')
          top=$(A shell dumpsys activity activities | grep -m1 topResumedActivity | tr -d '\r')
          echo "$(ts) not foreground at start; after KEYCODE_WAKEUP: pid=$pid cgroup=$cg top=$top" >> "$R/$tag.fg.txt"
          if echo "$cg" | grep -q 'cpuset:/top-app' && echo "$top" | grep -q "$PKG"; then
            fg=ok_after_wakeup; echo "$(ts) foreground after one KEYCODE_WAKEUP" >> "$R/$tag.fg.txt"
          else
            fg=fail; echo "$(ts) FOREGROUND FAIL -> force-stop" >> "$R/$tag.fg.txt"; date '+%F %T' > "$R/$tag.FG_FAIL"
            A shell am force-stop "$PKG"
          fi
        fi
      fi
    fi
    out=$(A shell "$SAMPLE_CMD" 2>&1 | tr -d '\r')
    if [ "$guard" -gt 0 ]; then
      ma=$(echo "$out" | awk '/MemAvailable/ {print $2; exit}')
      if [ -n "$ma" ] && [ "$ma" -lt "$guard" ]; then
        A shell am force-stop "$PKG"
        echo "$(ts) MemAvailable $ma kB < $guard kB -> force-stop" | tee -a "$R/$tag.MEMGUARD"
      fi
    fi
    if [ $(( n % every )) = 0 ]; then
      out="$out
$(A shell "$SLOW_CMD" 2>&1 | tr -d '\r')"
      w=$(echo "$out" | grep -m1 -oE 'mWakefulness=[A-Za-z]+')
      # only while the app is known to be in front (a WAKEUP before am start lights the screen and caps the prime
      # cluster in state_before)
      if [ -n "$w" ] && [ "$w" != "mWakefulness=Awake" ] && { [ "$fg" = ok ] || [ "$fg" = ok_after_wakeup ]; }; then
        A shell input keyevent KEYCODE_WAKEUP
        echo "$(ts) $tag $w -> KEYCODE_WAKEUP" >> "$R/wake_keeper.log"
      fi
    fi
    printf '=== %s n=%d fg=%s\n%s\n' "$(date '+%T')" "$n" "$fg" "$out" >> "$R/$tag.samples.txt"
    n=$((n + 1)); sleep "$per"
  done
}

score_leg() {  # tag rows mode: the lane's host scorer (lane_score in lane.env) of a DONE leg, in the background, from
  # LANE_DIR. Contract: a gate (or generic) leg's verdict is the first word of R/<tag>.verdict (PASS / FAIL / ...),
  # which needs_pass reads; a lane without lane_score gets no host scoring.
  declare -F lane_score > /dev/null || return 0
  ( cd "$LANE_DIR" && lane_score "$1" "$2" "$3" "$CUR_GRAPH" ) < /dev/null &
  SCORE_PIDS="$SCORE_PIDS $!"
}

pull_out() {  # tag: a generic leg's out_<tag>.f32 (the byte count its report names), then removed from the phone
  local tag=$1 want got
  want=$(python3 -I -c "import json,sys; print(json.load(open(sys.argv[1])).get('out_bytes', ''))" "$R/$tag.json" 2>/dev/null)
  if [ -z "$want" ]; then echo "out: none in the report"; return 0; fi
  A exec-out run-as "$PKG" cat "files/out_$tag.f32" > "$R/out_$tag.f32" 2>/dev/null
  present || hard_stop "device_lost_in_pull_$tag"
  got=$(stat -f %z "$R/out_$tag.f32" 2>/dev/null)
  echo "out bytes: report=$want pulled=$got $([ "$want" = "$got" ] && echo MATCH || echo CHECK)"
  A shell run-as "$PKG" rm -f "files/out_$tag.f32"
  [ "$want" = "$got" ] || rm -f "$R/out_$tag.f32"
}

prep_leg() {  # tag prep rows: <kind>:<earlier generic leg> -> this leg's rows file and its inputs, made by the lane's
  # lane_prep from that leg's pulled output into R/gen/, then sent with their sha256 checked on the phone
  local tag=$1 prep=$2 rows=$3 kind src srows f
  case "$prep" in *:*) kind=${prep%%:*}; src=${prep#*:} ;; *) echo "prep $tag: $prep is not <kind>:<leg>" >> "$R/s_prep.txt"; return 1 ;; esac
  declare -F lane_prep > /dev/null || { echo "prep $tag: lane.env defines no lane_prep" >> "$R/s_prep.txt"; return 1; }
  done_leg "$src" || { echo "prep $tag: $src is not DONE ($(status_of "$src"))" >> "$R/s_prep.txt"; return 1; }
  [ -s "$R/out_$src.f32" ] || { echo "prep $tag: no $R/out_$src.f32" >> "$R/s_prep.txt"; return 1; }
  srows=$(printf '%s\n' "${PLAN_LINES[@]}" | awk -v t="$src" '$1 == t {print $6; exit}')
  mkdir -p "$GEN"
  ( cd "$LANE_DIR" && lane_prep "$kind" "$R/$src.json" "$R/out_$src.f32" "$ROWS_DIR/$srows" "$rows" "$GEN" "$R/gen.sha256" ) >> "$R/s_prep.txt" 2>&1 \
    || { echo "prep $tag: $kind from $src failed" >> "$R/s_prep.txt"; return 1; }
  for f in "$rows" $(generic_files_of "$GEN/$rows"); do
    send_small "$GEN/$f" "$f" >> "$R/s_prep.txt" 2>&1 || { echo "prep $tag: sending $f failed" >> "$R/s_prep.txt"; return 1; }
  done
  echo "prep $tag: $rows and its inputs made from $src and sent $(ts)" | tee -a "$R/s_prep.txt"
}

verdict_of() {  # tag -> PASS / FAIL / NONE, waiting up to VERDICT_WAIT_S for the background scorer of that leg
  local tag=$1 t0
  t0=$(now)
  while [ ! -s "$R/$tag.verdict" ] && [ $(( $(now) - t0 )) -lt "$VERDICT_WAIT_S" ] && [ "$(status_of "$tag")" = DONE ]; do sleep 2; done
  [ -s "$R/$tag.verdict" ] && cut -d' ' -f1 "$R/$tag.verdict" || echo NONE
}

LEG_STATUS=NONE; LEG_END=0; STOP_ALL=0; LAST_UPTIME=0; CUR_GRAPH=""
leg() {  # tag limit_s accel precision graph rows mode sample_s guard_kb extras
  local tag=$1 lim=$2 accel=$3 prec=$4 graph=$5 rows=$6 mode=$7 per=$8 guard=$9 extras=${10} rc rp mp cap up
  if [ "$(now)" -ge "$SKIP_AFTER" ]; then
    echo "SKIP $tag at $(date '+%T') (past skip_after $(hms "$SKIP_AFTER"))" | tee -a "$R/s_skipped.txt"; LEG_STATUS=SKIPPED; return
  fi
  cap=$(( RUN_END - $(now) )); [ $cap -lt "$lim" ] && lim=$cap
  echo "$READY_NOTE" > "$R/$tag.ready.txt"
  ps_time "$R/$tag.ps_before.txt"
  echo "=== leg $tag start $(date '+%T') limit ${lim}s ready: $READY_NOTE"
  LIMIT=$lim LEG_SIG=$(sig_of "$graph") LEG_EXTRAS=$extras \
    bash "$RUN" gate "$tag" "$accel" "$prec" "$graph" "$rows" "$mode" > "$R/$tag.run.txt" 2>&1 &
  rp=$!
  monitor "$tag" "$rp" "$per" "$guard" &
  mp=$!
  wait "$rp"; rc=$?
  wait "$mp" 2>/dev/null
  LEG_END=$(now)
  cat "$R/$tag.run.txt"
  [ $rc = 9 ] && hard_stop "device_lost_in_$tag"
  if [ "$mode" = generic ] && [ -s "$R/$tag.json" ]; then
    pull_out "$tag" > "$R/$tag.pull.txt" 2>&1
    cat "$R/$tag.pull.txt"
  fi
  ps_time "$R/$tag.ps_after.txt"
  others_time "$R/$tag.ps_before.txt" "$R/$tag.ps_after.txt" "$R/$tag.others_time.txt"
  [ -s "$R/$tag.others_time.txt" ] && echo "other processes' CPU TIME grew during $tag: $(tr '\n' ' ' < "$R/$tag.others_time.txt")"
  [ -s "$R/$tag.samples.txt" ] && memory_line "$R/$tag.samples.txt" | tee "$R/$tag.memory.txt"
  LEG_STATUS=$(status_of "$tag")
  [ -f "$R/$tag.FG_FAIL" ] && LEG_STATUS=FG_FAIL
  if grep -q "PROCESS_DIED" "$R/$tag.run.txt"; then
    A logcat -b all -d -T "$(date -r $(( LEG_END - lim - 120 )) '+%m-%d %H:%M:%S.000')" > "$R/$tag.logcat_all_since_start.txt" 2>&1
    grep -iE "lowmemorykiller|lmkd|kill.*$PKG_RE|$PKG_RE.*kill" "$R/$tag.logcat_all_since_start.txt" | head -40 > "$R/$tag.kill_lines.txt"
    # an lmkd kill OF THIS APP only = lmkd lines naming the package; our own `am force-stop` (memory guard) is not lmkd
    if grep -iE "(lowmemorykiller|lmkd).*$PKG_RE" "$R/$tag.logcat_all_since_start.txt" > "$R/$tag.lmkd_app_lines.txt" && [ -s "$R/$tag.lmkd_app_lines.txt" ]; then
      echo "lmkd kill of $tag -> no more legs (recorded)"; STOP_ALL=1; LEG_STATUS=LMKD_KILLED
    elif [ -f "$R/$tag.MEMGUARD" ]; then
      echo "$tag stopped by the memory guard (not lmkd)"; LEG_STATUS=MEMGUARD_STOPPED
    fi
  fi
  echo "$LEG_STATUS" > "$R/$tag.leg_status"
  up=$(uptime_s)
  if [ -n "$up" ] && [ "$LAST_UPTIME" -gt 0 ] && [ "$up" -lt "$LAST_UPTIME" ]; then
    echo "uptime ${up}s < ${LAST_UPTIME}s before $tag: the phone rebooted" | tee "$R/$tag.REBOOT"
    hard_stop "phone_rebooted_during_$tag"
  fi
  [ -n "$up" ] && LAST_UPTIME=$up
  echo "=== leg $tag status $LEG_STATUS rc $rc end $(date '+%T')"
  if [ "$LEG_STATUS" = DONE ]; then
    if [ "$mode" = gate ] && [ -s "$R/${GATE_OUT}_$tag.f32" ]; then score_leg "$tag" "$rows" gate; fi
    if [ "$mode" = timing ] || [ "$mode" = generic_timing ]; then score_leg "$tag" "$rows" timing; fi
    if [ "$mode" = generic ] && [ -s "$R/out_$tag.f32" ]; then score_leg "$tag" "$rows" generic; fi
  fi
}

leg_fg() {  # a leg whose app could not stay in front: pause for R/RESUME (a person unlocks the phone), then one more try
  leg "$@"
  [ "$LEG_STATUS" = FG_FAIL ] || return 0
  echo "$1 $(ts)" > "$R/PAUSED_FG"
  echo "PAUSED: foreground lost in $1 at $(date '+%T'); waiting for $R/RESUME (until $(hms $(( SKIP_AFTER - 60 ))))"
  until [ -f "$R/RESUME" ]; do
    present || hard_stop "device_lost_while_paused"
    [ "$(now)" -ge $(( SKIP_AFTER - 60 )) ] && { echo "no RESUME by $(date '+%T'): remaining legs skipped"; STOP_ALL=1; return 1; }
    sleep 5
  done
  rm -f "$R/RESUME" "$R/PAUSED_FG"
  echo "RESUME seen $(date '+%T')"
  local t=$1; shift
  leg "${t}_r2" "$@"
  [ "$LEG_STATUS" = FG_FAIL ] && { echo "foreground lost again: remaining legs skipped"; STOP_ALL=1; return 1; }
  return 0
}

cleanup_all() {  # the app's cleanup (run.sh cleanup), then what is left on the phone, read back
  bash "$RUN" cleanup > "$R/s9_cleanup.txt" 2>&1; local rc=$?
  cat "$R/s9_cleanup.txt"
  [ $rc = 9 ] && { echo "the phone left adb during the cleanup $(ts)" > "$R/CHAIN_STOPPED"; return 9; }
  { echo "pm list: '$(A shell pm list packages "$PKG" | tr -d '\r')'"
    echo "stage: $(A shell ls -d "$STAGE" 2>&1 | tr -d '\r')"
    echo "run-as files: $(A shell run-as "$PKG" ls -l files 2>&1 | tr -d '\r' | tr '\n' ' ')"
    echo "data_free: $(A shell df -h /data | tail -1 | tr -d '\r')"; } | tee "$R/s9_cleanup_check.txt"
  CLEANED=1
}

small_files() {  # the plan's rows files and the side files they name, unique, one per line; a prep leg's rows file is
  # made during the run and is not listed
  local line tag lim accel prec graph rows mode ready extras
  for line in "${PLAN_LINES[@]}"; do
    read -r tag lim accel prec graph rows mode ready extras <<< "$line"
    [ -n "$(xget "$extras" prep "")" ] && continue
    echo "$rows"
    [ -s "$ROWS_DIR/$rows" ] && prefix_files_of "$ROWS_DIR/$rows" && generic_files_of "$ROWS_DIR/$rows"
  done | grep . | LC_ALL=C sort -u
}

check_inputs() {  # sha256 of the APK, the plan's graphs, rows files and side files -> R/inputs.sha256; the leg table
  mkdir -p "$R"
  read_plan
  local line tag lim accel prec graph rows mode ready extras f prep src hint ok=0
  echo "plan: $PLAN"
  [ -s "$APK" ] || { echo "missing APK $APK"; ok=1; }
  [ -n "$APK_SHA" ] || { echo "no APK sha256 (APK_SHA or $APK.sha256)"; ok=1; }
  [ -s "$APK" ] && [ "$(shasum -a 256 "$APK" | cut -d' ' -f1)" != "$APK_SHA" ] && { echo "APK sha256 differs from $APK_SHA"; ok=1; }
  : > "$R/inputs.sha256.tmp"
  for line in "${PLAN_LINES[@]}"; do
    read -r tag lim accel prec graph rows mode ready extras <<< "$line"
    case "$lim" in
      ''|*[!0-9]*)
        hint=""; case "$lim" in npu*|gpu|cpu) hint=" (the NPU chain's columns: tag accel precision graph fx ...: use npu_chain.sh)" ;; esac
        echo "$tag: limit_s '$lim' is not a number$hint"; ok=1 ;;
    esac
    case "$accel" in gpu|cpu) ;; *) echo "$tag: accel $accel"; ok=1 ;; esac
    case "$mode" in gate|timing|generic|generic_timing) ;; *) echo "$tag: mode $mode"; ok=1 ;; esac
    case "$ready" in clean|gate|timing|uncapped) ;; *) echo "$tag: ready $ready"; ok=1 ;; esac
    [ "$accel" = cpu ] || case "$prec" in fp32|fp16acc32|fp16|default) ;; *) echo "$tag: precision $prec"; ok=1 ;; esac
    [ -s "$GRAPH_DIR/$graph" ] || { echo "$tag: missing graph $GRAPH_DIR/$graph"; ok=1; }
    prep=$(xget "$extras" prep "")
    if [ -n "$prep" ]; then
      src=""; case "$prep" in *:*) src=${prep#*:} ;; esac
      [ -n "$src" ] && [ "$mode" = generic ] \
        && printf '%s\n' "${PLAN_LINES[@]}" | awk -v t="$src" '$1 == t && $7 == "generic" {f = 1} END {exit f ? 0 : 1}' \
        || { echo "$tag: prep $prep needs mode generic and an earlier generic leg $src"; ok=1; }
      declare -F lane_prep > /dev/null || { echo "$tag: prep $prep needs lane_prep in lane.env"; ok=1; }
    else
      [ -s "$ROWS_DIR/$rows" ] || { echo "$tag: missing rows $ROWS_DIR/$rows"; ok=1; }
    fi
    [ -z "$SIG_OF" ] || [ -n "$(sig_of "$graph")" ] || echo "$tag: SIG_OF gives no signature for $graph (the app resolves it itself)"
    for f in "$GRAPH_DIR/$graph"; do
      [ -s "$f" ] || continue
      awk -v f="$(basename "$f")" '$2 == f {found = 1} END {exit found ? 0 : 1}' "$R/inputs.sha256.tmp" && continue
      echo "$(shasum -a 256 "$f" | cut -d' ' -f1) $(basename "$f")" >> "$R/inputs.sha256.tmp"
    done
    printf '%-34s %-6s %-5s %-10s %-40s %-20s %-6s %-6s %s\n' "$tag" "$lim" "$accel" "$prec" "$graph" "$rows" "$mode" "$ready" "$extras"
  done
  for f in $(small_files); do
    [ -s "$ROWS_DIR/$f" ] || { echo "missing small file $ROWS_DIR/$f"; ok=1; continue; }
    echo "$(shasum -a 256 "$ROWS_DIR/$f" | cut -d' ' -f1) $f" >> "$R/inputs.sha256.tmp"
  done
  # "<sha256> <name>", one line per file (what input_sha reads)
  LC_ALL=C sort -k2 -u "$R/inputs.sha256.tmp" > "$R/inputs.sha256"; rm -f "$R/inputs.sha256.tmp"
  echo "inputs:"; sed 's/^/  /' "$R/inputs.sha256"
  return $ok
}

standin_report() {  # report json -> one line: status, error, rows, layout check
  python3 -I -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get('status'), d.get('error', ''), d.get('summary', {}).get('count'), d.get('layout_equals_host_build_inputs', d.get('out_bytes')))" "$1"
}

mac_generic_leg() {  # tag graph rows extras: the Mac CPU run of a generic leg through the stand-in + lane_score_mac; a
  # prep leg's inputs are made by lane_prep from the Mac run of its source leg (the Mac chain of the phone's legs)
  local tag=$1 graph=$2 rows=$3 extras=$4 base prep kind src sline sgraph srows sbase f rowsdir=$ROWS_DIR x
  local -a standin
  read -r -a standin <<< "$STANDIN"
  base="mac_${graph%.tflite}__${rows%.json}"
  prep=$(xget "$extras" prep "")
  if [ -n "$prep" ]; then
    kind=${prep%%:*}; src=${prep#*:}
    sline=$(printf '%s\n' "${PLAN_LINES[@]}" | awk -v t="$src" '$1 == t {print; exit}')
    read -r x x x x sgraph srows x <<< "$sline"
    sbase="mac_${sgraph%.tflite}__${srows%.json}"
    [ -s "$R/mac/$sbase.json" ] || { echo "$base: no Mac run of $src ($sbase) to make its $kind from"; return 0; }
    declare -F lane_prep > /dev/null || { echo "$base: prep $prep needs lane_prep in lane.env"; return 0; }
    rowsdir=$R/mac/gen_$tag
    ( cd "$LANE_DIR" && lane_prep "$kind" "$R/mac/$sbase.json" "$R/mac/out_$sbase.f32" "$ROWS_DIR/$srows" "$rows" "$rowsdir" "" ) \
      > "$R/mac/$base.prep.txt" 2>&1 || { echo "$base: $kind from $sbase failed ($R/mac/$base.prep.txt)"; return 0; }
  fi
  if [ ! -s "$R/mac/$base.json" ]; then
    ln -sf "$GRAPH_DIR/$graph" "$R/mac/$graph"; ln -sf "$rowsdir/$rows" "$R/mac/$rows"
    for f in $(generic_files_of "$rowsdir/$rows"); do ln -sf "$rowsdir/$f" "$R/mac/$f"; done
    ( cd "$LANE_DIR" && "${standin[@]}" --files-dir "$R/mac" --graph "$graph" --rows "$rows" --report "$base.json" \
        --mode generic --accel cpu --threads "$MAC_THREADS" ) > "$R/mac/$base.stdout.txt" 2>&1
    echo "$base: $(standin_report "$R/mac/$base.json")"
    rm -f "$R/mac/$graph" "$R/mac/$rows"
    for f in $(generic_files_of "$rowsdir/$rows"); do rm -f "$R/mac/$f"; done
  else
    echo "$base: already there"
  fi
  if declare -F lane_score_mac > /dev/null && [ -s "$R/mac/$base.json" ]; then
    ( cd "$LANE_DIR" && lane_score_mac "$base" "$graph" "$rows" generic "$rowsdir" )
  fi
}

mac_baseline() {  # the Mac CPU run of every gate leg's graph + rows through the stand-in (the same files as on the
  # phone), then lane_score_mac
  read_plan
  [ -n "$STANDIN" ] || { echo "mac-baseline needs STANDIN (the app's stand-in) in lane.env" >&2; exit 5; }
  mkdir -p "$R/mac"
  local line tag lim accel prec graph rows mode ready extras base f sig
  local -a standin
  read -r -a standin <<< "$STANDIN"
  for line in "${PLAN_LINES[@]}"; do
    read -r tag lim accel prec graph rows mode ready extras <<< "$line"
    if [ "$mode" = generic ]; then mac_generic_leg "$tag" "$graph" "$rows" "$extras"; continue; fi
    [ "$mode" = gate ] || continue
    base="mac_${graph%.tflite}__${rows%.json}"
    if [ ! -s "$R/mac/$base.json" ]; then
      sig=$(sig_of "$graph")
      ln -sf "$GRAPH_DIR/$graph" "$R/mac/$graph"; ln -sf "$ROWS_DIR/$rows" "$R/mac/$rows"
      for f in $(prefix_files_of "$ROWS_DIR/$rows"); do ln -sf "$ROWS_DIR/$f" "$R/mac/$f"; done
      ( cd "$LANE_DIR" && "${standin[@]}" --files-dir "$R/mac" --graph "$graph" --rows "$rows" --report "$base.json" \
          --mode gate --accel cpu --threads "$MAC_THREADS" ${sig:+--sig "$sig"} ) > "$R/mac/$base.stdout.txt" 2>&1
      echo "$base: $(standin_report "$R/mac/$base.json")"
      rm -f "$R/mac/$graph" "$R/mac/$rows"
      for f in $(prefix_files_of "$ROWS_DIR/$rows"); do rm -f "$R/mac/$f"; done
    else
      echo "$base: already there"
    fi
    if declare -F lane_score_mac > /dev/null && [ -s "$R/mac/$base.json" ]; then
      ( cd "$LANE_DIR" && lane_score_mac "$base" "$graph" "$rows" gate "$ROWS_DIR" )
    fi
  done
}

run_chain() {
  mkdir -p "$R"
  rm -f "$R/CHAIN_DONE" "$R/CHAIN_STOPPED" "$R/PAUSED_FG" "$R/RESUME" "$R/RELEASE_HOLD"
  echo "$$" > "$R/chain.pid"
  echo "chain $ROUND pid $$ $(ts); this file sha256 $(shasum -a 256 "$0" | cut -c1-16); lib.sh sha256 $(shasum -a 256 "$BENCH/lib.sh" | cut -c1-16); run.sh sha256 $(shasum -a 256 "$RUN" | cut -c1-16); lane.env $LANE_ENV sha256 $(shasum -a 256 "$LANE_ENV" | cut -c1-16); plan $PLAN sha256 $(shasum -a 256 "$PLAN" | cut -c1-16)"
  check_inputs > "$R/s0_inputs.txt" 2>&1 || { cat "$R/s0_inputs.txt"; echo "inputs check failed: no hold taken"; exit 5; }
  read_plan
  [ "$(LC_ALL=C sort -u <<< "$(printf '%s\n' "${PLAN_LINES[@]}" | awk '{print $1}')" | wc -l | tr -d ' ')" = "${#PLAN_LINES[@]}" ] || { echo "duplicate leg tags in the plan"; exit 5; }
  mkdir -p "$RESULTS_DIR"
  take_hold
  READY_NOTE=""; LEG_END=$(now)
  present || hard_stop "device_absent_at_start"
  echo "stay_on_while_plugged_in at $(ts): $(A shell settings get global stay_on_while_plugged_in | tr -d '\r') (read only)" | tee "$R/s_stayon.txt"
  gpu_detect || { echo "the GPU's driver was not read (got '$GPU_KIND'): no GPU reads"; GPU_KIND=none; }
  cooling_detect || true
  export GPU_KIND COOLING_SRC COOLING_SYSFS
  SAMPLE_CMD=$(sample_cmd "$PKG" "MemAvailable|MemFree"); SLOW_CMD=$(slow_cmd)
  echo "gpu at $(ts): $GPU_KIND (the ready gate's GPU level: $(gpu_level_name))" | tee "$R/s_gpu.txt"
  gpu_probe | sed 's/^/  /' >> "$R/s_gpu.txt"
  cooling_note | tee -a "$R/s_gpu.txt"
  thermal_raw
  if ! transfer_check; then
    echo "OTHER SESSION'S TRANSFER / PROCESS ON THE PHONE -> no leg, release $(ts)"
    echo "other_session_job $(ts)" > "$R/CHAIN_STOPPED"; CLEANED=1; exit 5
  fi
  LAST_UPTIME=$(uptime_s); echo "phone uptime at the hold: ${LAST_UPTIME}s"
  bash "$RUN" state > "$R/s0_state_initial.txt" 2>&1; cat "$R/s0_state_initial.txt"
  local ok line tag lim accel prec graph rows mode ready extras f
  (
    echo "setup start $(ts)"
    [ "$(shasum -a 256 "$APK" | cut -d' ' -f1)" = "$APK_SHA" ] || { echo "APK sha256 mismatch"; exit 5; }
    echo "data free before: $(avail_bytes) B"
    A install -r "$APK" || exit 5
    A shell run-as "$PKG" mkdir -p files
    for f in $(small_files); do
      send_small "$ROWS_DIR/$f" "$f" || exit 6
    done
    A shell rm -rf "$STAGE"
    echo "setup end $(ts); data free $(avail_bytes) B"
  ) > "$R/s0_setup.txt" 2>&1
  ok=$?
  cat "$R/s0_setup.txt"
  present || hard_stop "device_lost_in_setup"
  [ $ok = 0 ] || { echo "setup failed rc $ok $(ts)"; echo "setup_failed $(ts)" > "$R/CHAIN_STOPPED"; exit 5; }
  local rest per guard needs needs_pass once whole keep need v prep
  for line in "${PLAN_LINES[@]}"; do
    read -r tag lim accel prec graph rows mode ready extras <<< "$line"
    [ "$STOP_ALL" = 0 ] || { echo "SKIP $tag (stopped)" | tee -a "$R/s_skipped.txt"; continue; }
    done_leg "$tag" && { echo "DONE before: $tag"; continue; }
    rest=$(xget "$extras" rest 60); per=$(xget "$extras" sample 2); guard=$(xget "$extras" guard "$MEM_GUARD_KB")
    needs=$(xget "$extras" needs ""); needs_pass=$(xget "$extras" needs_pass ""); once=$(xget "$extras" once 0); whole=$(xget "$extras" whole 0)
    keep=$(xget "$extras" keep "" | tr ',' ' ')
    if [ -n "$needs" ] && ! done_leg "$needs"; then echo "SKIP $tag: needs $needs DONE ($(status_of "$needs"))" | tee -a "$R/s_skipped.txt"; continue; fi
    if [ -n "$needs_pass" ]; then
      v=$(verdict_of "$needs_pass")
      [ "$v" = PASS ] || { echo "SKIP $tag: needs_pass $needs_pass (verdict $v)" | tee -a "$R/s_skipped.txt"; continue; }
    fi
    if [ "$once" = 1 ] && [ -f "$R/$tag.attempted" ]; then echo "SKIP $tag: once=1 and tried at $(cat "$R/$tag.attempted")" | tee -a "$R/s_skipped.txt"; continue; fi
    [ "$(now)" -lt "$SKIP_AFTER" ] || { echo "SKIP $tag at $(date '+%T') (past skip_after $(hms "$SKIP_AFTER"))" | tee -a "$R/s_skipped.txt"; continue; }
    need=$(( rest + lim + 30 ))
    if [ "$whole" = 1 ] && [ $(( RUN_END - $(now) )) -lt "$need" ]; then
      echo "DEFER $tag at $(date '+%T'): $(( RUN_END - $(now) ))s left before run_end < ${need}s (rest + the whole limit)" | tee -a "$R/s_skipped.txt"; continue
    fi
    prep=$(xget "$extras" prep "")
    if [ -n "$prep" ]; then
      prep_leg "$tag" "$prep" "$rows" || { echo "SKIP $tag: prep $prep failed (s_prep.txt)" | tee -a "$R/s_skipped.txt"; continue; }
    fi
    # shellcheck disable=SC2086
    swap_to "$graph" $keep || continue
    CUR_GRAPH=$graph
    ready_gate "$tag" "$LEG_END" "$rest" "$ready" "$(xget "$extras" poll "$POLL")"
    [ "${READY_SKIP:-0}" = 1 ] && { echo "SKIP $tag at $(date '+%T'): $READY_NOTE (left for the next hold)" | tee -a "$R/s_skipped.txt"; continue; }
    [ "$(now)" -lt "$SKIP_AFTER" ] || { echo "SKIP $tag at $(date '+%T') (past skip_after after the ready gate)" | tee -a "$R/s_skipped.txt"; continue; }
    date '+%F %T' > "$R/$tag.attempted"
    leg_fg "$tag" "$lim" "$accel" "$prec" "$graph" "$rows" "$mode" "$per" "$guard" "$extras"
  done
  cleanup_all || hard_stop "device_lost_in_cleanup"
  bash "$RUN" state > "$R/s9_state_end.txt" 2>&1; cat "$R/s9_state_end.txt"
  echo "chain $ROUND legs done $(ts)"
  date '+%F %T' > "$R/CHAIN_DONE"
  release_hold
  wait_scores
  echo "host scorers done $(ts)"
}

case "$CMD" in
  check) check_inputs ;;
  mac-baseline) mac_baseline ;;
  run) run_chain ;;
  cleanup) mkdir -p "$R"; cleanup_cmd ;;
  *) echo "usage: $0 check | mac-baseline | run | cleanup <round> <lane.env>" >&2; exit 2 ;;
esac
