#!/bin/bash
# Round 5: the d1-omni Galaxy S26 chain = d1_3b_work's scripts/d1_s26_chain.sh (round 4, rehearsed three times against a
# fake phone) with this lane's package, graphs and rows (the six-input decision graph, media prefixes as files), the
# scorer scripts/s26_score.py, a `needs_pass=` gate on a leg's parity verdict, and a broader "another lane's app is busy"
# check. One hold of at most 30 minutes per run.
#   bash scripts/s26_chain.sh check <round>         the plan and its inputs, sha256s; no adb, no hold
#   bash scripts/s26_chain.sh mac-baseline <round>  the Mac CPU same-file run of every gate leg's graph and rows
#                                                   (scripts/fake_runner.py standalone) + its score; no adb; heavy: run it
#                                                   under ~/code/standup/tools/quiet/quiet_wait.py --
#   bash scripts/s26_chain.sh run <round>           the phone: hold, legs, cleanup, release
#   bash scripts/s26_chain.sh cleanup <round>       by hand after a hard stop: uses the dead chain's live hold or takes the
#                                                   phone again, removes the app, releases
# Round dir R = d1_omni_work/device/<round>/ (D1_R overrides); the plan = device/plan_<round>.tsv (D1_PLAN overrides).
# plan, one leg per line (# comments; whitespace-separated; the extras are key=value):
#   tag  limit_s  accel  precision  graph  rows  mode  ready  [extras]
#   tag = the leg's name (report R/<tag>.json); limit_s = the app's time limit (capped so the leg ends by RUN_END_MIN);
#   accel gpu | cpu; precision fp32 | fp16acc32 | fp16 | default (gpu) or - (cpu); graph = a file of D1_GRAPH_DIR
#   (d1omni_decide_L<L>_<form>.tflite: the app gets sig=decide_<L>); rows = a file of D1_ROWS_DIR (its prefix files go
#   along); mode gate | timing; ready = the ready gate's mode for this leg: clean (wait for a clean phone up to REST_MAX,
#   then run warm), gate (do not wait for clean, but never at thermal status >= 4) or timing (run only from clean, else
#   leave it for the next hold).
#   extras: rest=<s> (rest since the previous leg, default 60), keep=<graph,...> (graphs that stay on the phone),
#   guard=<kB> (memory guard: MemAvailable below it -> am force-stop; default MEM_GUARD_KB; 0 = off), sample=<s> (the
#   sampler's period, default 2), needs=<tag> (run only when that leg is DONE), needs_pass=<tag> (run only when that
#   gate leg's parity verdict, written by the host scorer, says PASS; waits up to 120 s for the verdict), once=1 (one
#   attempt per lane, whatever its result), whole=1 (start only when the whole limit fits before RUN_END_MIN), and the
#   app's extras threads= warmup= reps= rest_ms= cool_ms= limit_rows= keep_scores= cpu_cache= gpu_src_quant=.
# A leg whose report says DONE is skipped, so a later hold continues the plan where the last one stopped.
# Order of a run: keeper (scripts/s26_keeper.sh: queue_cli.py wait <hold file> d1c-<round>-s26 <keeper pid>; the entry is
# enqueued by hand before the chain starts) -> HOLD_ACQUIRED -> stay-on read (never changed) -> transfer check (another
# session's adb transfer on the Mac; this app, a *runner process, or another com.mlboydaisuke.* app that is in front or
# whose CPU time grows, on the phone = no leg, release) -> /proc/uptime -> the APK (sha256 checked) installed -> the
# plan's rows files and their prefix files into the app's files/ -> per leg: the graph into files/ (exec-in, byte count
# polled, toybox sha256sum on the phone; other graphs removed unless kept, /data free checked) -> ready gate -> the leg
# (scripts/s26_run.sh: am start, poll, STOP at the limit, pull, logcat by pid) with the 2 s sampler (device clock, kgsl,
# CPU caps, MemAvailable, the app's VmHWM / VmRSS / VmSwap, kgsl page_alloc), the foreground check and the memory guard
# -> lmkd (lines naming this package only) / memory guard / reboot (/proc/uptime went down) / other apps' CPU TIME -> the
# host scorer of a DONE leg in the background (scripts/s26_score.py gate | timing -> D1_RESULTS_DIR/s26_parity_<tag>.json
# | s26_timing_<tag>.json) -> cleanup (force-stop, uninstall, the stage dir removed, checked) -> RELEASE_HOLD -> the
# keeper releases.
# Clock (minutes after the hold): no leg starts after SKIP_MIN (25), every leg's limit ends by RUN_END_MIN (28), the
# keeper releases at END_MIN (30) at the latest. Exit 9 = the phone left adb (hard stop; the hold is released).
# Env: D1_APK / D1_APK_SHA (default device/app-debug.<round>.apk and the sha256 in device/app-debug.<round>.apk.sha256),
# D1_HOLD_FILE (default the shared S26 hold file), D1_SCRIPT_NAME (default d1c-<round>-s26), D1_GRAPH_DIR (default
# out/), D1_ROWS_DIR (default device/), D1_RESULTS_DIR (default results/), ADB (default the SDK's adb), PY (the scorer's
# and the stand-in's python, default ~/venvs/lt094dev/bin/python), D1_DRY_RUN=1 (requires ADB = scripts/fake_adb.py and
# a hold file under d1_omni_work/cache/; the dry run's driver is scripts/dryrun.sh).
# Kept out (Kev r11-r17, d1a r4): no `timeout adb` (it kills only the Mac's client) - waits poll the phone instead; every
# remote `date` format quoted; `mkdir -p $STAGE` before a staged push; `LC_ALL=C sort` everywhere; no function defined
# inside another function and no name reused; lmkd = lines naming this package only, our own force-stop is
# MEMGUARD_STOPPED; time checked before every graph transfer; the foreground check judges a live process only (no process
# 1 s after the start = fg=gone; the report or PROCESS_DIED decides that leg); never a bare `wait` (the keeper is a child:
# the host scorers' pids are collected and waited for by pid); `stat -L -f %z` for sizes; this file is never edited while
# it runs (write a new file and mv it over this one: a running bash keeps the old inode).
# Round 9 additions (round 5's plans, names and paths are unchanged):
#   modes generic / generic_timing: the app's generic mode (a generic rows file: the vision tower, the projector, the
#   audio graph); a generic leg's out_<tag>.f32 is pulled after the leg (the size the report names) and removed from the
#   phone; its input files (the rows file's `inputs[].file`) go to the phone with the other small files;
#   scripts/s26_score.py generic scores it (vs the oracle npz, the Mac stores, the Mac CPU same-file run).
#   extras prep=soft:<tower leg> (a projector leg): before the leg, scripts/s26_score.py soft makes the leg's rows file
#   and its soft input from that tower leg's pulled features (host unshuffle) into R/gen/, and both go to the phone
#   (sha256 in R/gen.sha256); the leg is skipped when the tower leg is not DONE or the files do not arrive.
#   ready mode uncapped: run when no CPU policy is capped, kgsl thermal_pwrlevel is 0 and the thermal status is at most 1
#   (the caps decide a call's speed; status 1 alone outlasts the caps by 15-75 s, round 5 s_ready.txt); like timing,
#   not reached by REST_MAX_T or the hold's deadline -> skipped. extras poll=<s> = the ready gate's poll period.
#   Result files carry the round: s26_parity_<tag>_<round>.json, s26_timing_<tag>_<round>.json,
#   s26_macbase_<base>_<round>.json (D1_RESULT_SUFFIX overrides; r5 and the round 5 dry runs keep the bare names).
set -u
K=${K:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
CA=${CA:?set CA to the directory of the shared-device hold tools (queue_cli.py, hold_cli.py)}
CMD=${1:-}; ROUND=${2:-}
[ -n "$ROUND" ] || { echo "usage: $0 check | mac-baseline | run | cleanup <round>" >&2; exit 2; }
R=${D1_R:-$K/device/$ROUND}
RUN=$K/scripts/s26_run.sh
KEEPER=$K/scripts/s26_keeper.sh
SCORE=$K/scripts/s26_score.py
STANDIN=$K/scripts/fake_runner.py
PY=${PY:-python3}
ADB=${ADB:-adb}
SERIAL=${D1_SERIAL:-RFGL80R6A6H}
PKG=com.mlboydaisuke.d1omni.gate
STAGE=/data/local/tmp/d1omni_gate
APK=${D1_APK:-$K/device/app-debug.$ROUND.apk}
APK_SHA=${D1_APK_SHA:-$(cut -d' ' -f1 "$APK.sha256" 2>/dev/null)}
HOLD_FILE=${D1_HOLD_FILE:-$CA/s2_npu_sweep/.device_hold}
SCRIPT_NAME=${D1_SCRIPT_NAME:-d1c-$ROUND-s26}
PLAN=${D1_PLAN:-$K/device/plan_$ROUND.tsv}
GRAPH_DIR=${D1_GRAPH_DIR:-$K/out}
ROWS_DIR=${D1_ROWS_DIR:-$K/device}
RESULTS=${D1_RESULTS_DIR:-$K/results}
DRY_RUN=${D1_DRY_RUN:-0}
SKIP_MIN=${SKIP_MIN:-25}; RUN_END_MIN=${RUN_END_MIN:-28}; END_MIN=${END_MIN:-30}
REST_MAX=${REST_MAX:-300}; REST_MAX_T=${REST_MAX_T:-600}; POLL=${POLL:-15}
MEM_GUARD_KB=${MEM_GUARD_KB:-1500000}
DATA_MARGIN=${DATA_MARGIN:-2000000000}
XFER_MAX_S=${XFER_MAX_S:-900}
VERDICT_WAIT_S=${VERDICT_WAIT_S:-120}
case "$ROUND" in r5|dryrun|dryrun-lost|dryrun-busy) SFX_DEFAULT="" ;; *) SFX_DEFAULT="_$ROUND" ;; esac
SFX=${D1_RESULT_SUFFIX-$SFX_DEFAULT}
GEN=$R/gen
export D1_DEVICE_OUT=$R ADB D1_SERIAL=$SERIAL

if [ "$DRY_RUN" = 1 ]; then
  case "$(basename "$ADB")" in fake_adb.py) ;; *) echo "D1_DRY_RUN=1 needs ADB=$K/scripts/fake_adb.py (got $ADB)" >&2; exit 2 ;; esac
  case "$HOLD_FILE" in "$K"/cache/*) ;; *) echo "D1_DRY_RUN=1 needs a hold file under $K/cache (got $HOLD_FILE)" >&2; exit 2 ;; esac
  case "$RESULTS" in "$K"/results|"$K"/results/) echo "D1_DRY_RUN=1 needs D1_RESULTS_DIR outside results/" >&2; exit 2 ;; esac
elif [ "$(basename "$ADB")" = fake_adb.py ]; then
  echo "the fake adb outside a dry run (set D1_DRY_RUN=1)" >&2; exit 2
fi

A() { "$ADB" -s "$SERIAL" "$@"; }
now() { date +%s; }
ts() { date '+%F %T'; }
hms() { date -r "$1" '+%T'; }
present() { [ "$(A get-state 2>/dev/null | tr -d '\r')" = "device" ]; }
hard_stop() { echo "HARD STOP: $1 $(ts)"; echo "$1 $(ts)" > "$R/CHAIN_STOPPED"; exit 9; }
thermal() { A shell dumpsys thermalservice | grep -m1 'Thermal Status' | tr -d '\r' | grep -oE '[0-9]+$'; }
cpu_capped() { A shell 'for p in /sys/devices/system/cpu/cpufreq/policy*; do echo "${p##*/} $(cat $p/scaling_max_freq) $(cat $p/cpuinfo_max_freq)"; done' | tr -d '\r' | awk '$2 != $3 {printf "%s:%s/%s ", $1, $2, $3}'; }
gpulvl() { A shell cat /sys/class/kgsl/kgsl-3d0/thermal_pwrlevel 2>/dev/null | tr -d '\r'; }
uptime_s() { A shell cat /proc/uptime 2>/dev/null | tr -d '\r' | awk '{printf "%d", $1}'; }
avail_bytes() { A shell df -k /data | tail -1 | awk '{print $4 * 1024}' | tr -d '\r'; }
status_of() { python3 -I -c "import json,sys; print(json.load(open(sys.argv[1])).get('status','NONE'))" "$R/$1.json" 2>/dev/null || echo NONE; }
done_leg() { [ "$(status_of "$1")" = DONE ]; }
xget() {  # extras key default -> the value of key=... in the extras, else the default
  local kv
  for kv in $1; do case "$kv" in "$2"=*) echo "${kv#*=}"; return ;; esac; done
  echo "$3"
}
sig_of() { echo "$1" | sed -nE 's/^d1omni_decide_L([0-9]+)_.*\.tflite$/decide_\1/p'; }
prefix_files_of() {  # rows file -> the prefix files its rows (or timing sets) name, one per line
  python3 -I -c "
import json, sys
d = json.load(open(sys.argv[1]))
rows = d.get('rows') or [r for s in d.get('sets', []) for r in s['rows']]
print('\n'.join(sorted({r['prefix_file'] for r in rows if r.get('prefix_file')})))" "$1"
}
input_sha() {  # name -> its sha256 from R/inputs.sha256, else from R/gen.sha256 (files made during the run, round 9)
  local s
  s=$(awk -v f="$1" '$2 == f {print $1}' "$R/inputs.sha256")
  [ -z "$s" ] && [ -s "$R/gen.sha256" ] && s=$(awk -v f="$1" '$2 == f {s = $1} END {print s}' "$R/gen.sha256")
  echo "$s"
}
generic_files_of() {  # rows file -> the input files a generic rows file names (round 9), one per line; nothing otherwise
  python3 -I -c "
import json, sys
d = json.load(open(sys.argv[1]))
if d.get('kind') == 'generic':
    print('\n'.join(sorted({i['file'] for i in d['inputs']})))" "$1"
}
on_device() { A shell run-as $PKG ls files 2>/dev/null | tr -d '\r' | grep '\.tflite$'; }

dev_sha() {  # name -> MATCH / CHECK line; fails unless both sides are non-empty and equal
  local name=$1 want got
  want=$(input_sha "$name"); got=$(A shell run-as $PKG toybox sha256sum "files/$name" 2>/dev/null | cut -d' ' -f1 | tr -d '\r')
  echo "sha256 $name phone=${got:-<none>} mac=${want:-<none>} $([ -n "$got" ] && [ -n "$want" ] && [ "$got" = "$want" ] && echo MATCH || echo CHECK)"
  [ -n "$got" ] && [ -n "$want" ] && [ "$got" = "$want" ]
}

xfer() {  # src name: exec-in into the app's files/ (no staged copy); the client may return before the phone has
  # written everything (Kev r14) -> poll the byte count until it equals the source's, or stops growing for 30 s, or
  # XFER_MAX_S; no `timeout` on the client
  local src=$1 name=$2 want got last=-1 still=0 t0 rc
  want=$(stat -L -f %z "$src"); t0=$(now)
  A exec-in "run-as $PKG sh -c 'cat > files/$name'" < "$src"; rc=$?
  present || hard_stop "device_lost_in_transfer_$name"
  while :; do
    got=$(A shell run-as $PKG stat -c %s "files/$name" 2>/dev/null | tr -d '\r')
    [ -n "$got" ] && [ "$got" = "$want" ] && break
    if [ "$got" = "$last" ]; then still=$(( still + 5 )); else still=0; fi
    last=$got
    { [ "$still" -ge 30 ] || [ $(( $(now) - t0 )) -ge "$XFER_MAX_S" ]; } && break
    sleep 5
  done
  echo "exec-in $name rc=$rc bytes ${got:-<none>}/$want in $(( $(now) - t0 ))s"
  if [ "$rc" = 0 ] && [ -n "$got" ] && [ "$got" = "$want" ]; then return 0; fi
  A shell run-as $PKG rm -f "files/$name"; return 1
}

push_cp() {  # src name: a small file staged in $STAGE, copied into files/ with run-as, the staged copy removed
  local src=$1 name=$2 want got
  want=$(stat -L -f %z "$src")
  A shell mkdir -p $STAGE
  A push "$src" "$STAGE/$name" >/dev/null || { present || hard_stop "device_lost_in_push_$name"; return 1; }
  got=$(A shell stat -c %s "$STAGE/$name" | tr -d '\r')
  [ -n "$got" ] && [ "$got" = "$want" ] || { echo "staged size mismatch $name: $got != $want"; A shell rm -f "$STAGE/$name"; return 1; }
  A shell run-as $PKG cp "$STAGE/$name" "files/$name" || { A shell rm -f "$STAGE/$name"; return 1; }
  got=$(A shell run-as $PKG stat -c %s "files/$name" | tr -d '\r')
  A shell rm -f "$STAGE/$name"
  echo "push+cp $name bytes $got/$want"
  [ -n "$got" ] && [ "$got" = "$want" ]
}

send_small() {  # src name: exec-in, else the staged push; then the sha256 on the phone
  xfer "$1" "$2" || push_cp "$1" "$2" || return 1
  dev_sha "$2"
}

ensure_graph() {  # name [keep ...]: the graph in files/ with its sha256 checked; other graphs not kept removed first
  local g=$1; shift
  local keep=" $g $* " x need free
  if on_device | grep -qx "$g" && dev_sha "$g"; then echo "graph $g already on the phone"; return 0; fi
  for x in $(on_device); do
    case "$keep" in *" $x "*) ;; *) A shell run-as $PKG rm -f "files/$x"; echo "removed $x" ;; esac
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

SAMPLE_CMD='echo "devtime $(date +%s)"; p=$(pidof com.mlboydaisuke.d1omni.gate); echo "kgsl $(cat /sys/class/kgsl/kgsl-3d0/clock_mhz) $(cat /sys/class/kgsl/kgsl-3d0/max_clock_mhz) $(cat /sys/class/kgsl/kgsl-3d0/thermal_pwrlevel) $(cat /sys/class/kgsl/kgsl-3d0/temp)"; for q in /sys/devices/system/cpu/cpufreq/policy*; do echo "cpu ${q##*/} $(cat $q/scaling_cur_freq) $(cat $q/scaling_max_freq) $(cat $q/cpuinfo_max_freq)"; done; grep -E "MemAvailable|MemFree" /proc/meminfo; echo "pid $p"; [ -n "$p" ] && grep -E "VmRSS|VmHWM|VmSwap" /proc/$p/status; echo "kgsl_page_alloc $(cat /sys/class/kgsl/kgsl/page_alloc 2>/dev/null)"'
SLOW_CMD='t=$(dumpsys thermalservice); echo "$t" | grep -m1 "Thermal Status"; echo "$t" | grep -m1 -oE "mValue=[-0-9.]+, mType=3, mName=SKIN"; dumpsys battery | grep -m1 temperature; dumpsys power | grep -m1 -oE "mWakefulness=[A-Za-z]+"'

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
        fg=ok; echo "$(ts) foreground ok at start (cpuset top-app, GateActivity top-resumed)" >> "$R/$tag.fg.txt"
      else
        A shell input keyevent KEYCODE_WAKEUP; sleep 3
        pid=$(A shell pidof $PKG | tr -d '\r')
        if [ -z "$pid" ]; then
          fg=gone; echo "$(ts) not foreground at start; no app process after KEYCODE_WAKEUP" >> "$R/$tag.fg.txt"
        else
          cg=$(A shell cat /proc/$pid/cgroup 2>&1 | tr -d '\r' | tr '\n' ' ')
          top=$(A shell dumpsys activity activities | grep -m1 topResumedActivity | tr -d '\r')
          echo "$(ts) not foreground at start; after KEYCODE_WAKEUP: pid=$pid cgroup=$cg top=$top" >> "$R/$tag.fg.txt"
          if echo "$cg" | grep -q 'cpuset:/top-app' && echo "$top" | grep -q "$PKG"; then
            fg=ok_after_wakeup; echo "$(ts) foreground after one KEYCODE_WAKEUP" >> "$R/$tag.fg.txt"
          else
            fg=fail; echo "$(ts) FOREGROUND FAIL -> force-stop" >> "$R/$tag.fg.txt"; date '+%F %T' > "$R/$tag.FG_FAIL"
            A shell am force-stop $PKG
          fi
        fi
      fi
    fi
    out=$(A shell "$SAMPLE_CMD" 2>&1 | tr -d '\r')
    if [ "$guard" -gt 0 ]; then
      ma=$(echo "$out" | awk '/MemAvailable/ {print $2; exit}')
      if [ -n "$ma" ] && [ "$ma" -lt "$guard" ]; then
        A shell am force-stop $PKG
        echo "$(ts) MemAvailable $ma kB < $guard kB -> force-stop" | tee -a "$R/$tag.MEMGUARD"
      fi
    fi
    if [ $(( n % every )) = 0 ]; then
      out="$out
$(A shell "$SLOW_CMD" 2>&1 | tr -d '\r')"
      w=$(echo "$out" | grep -m1 -oE 'mWakefulness=[A-Za-z]+')
      # only while the app is known to be in front (Kev r6: a WAKEUP before am start lit the screen and capped the
      # prime cluster in state_before)
      if [ -n "$w" ] && [ "$w" != "mWakefulness=Awake" ] && { [ "$fg" = ok ] || [ "$fg" = ok_after_wakeup ]; }; then
        A shell input keyevent KEYCODE_WAKEUP
        echo "$(ts) $tag $w -> KEYCODE_WAKEUP" >> "$R/wake_keeper.log"
      fi
    fi
    printf '=== %s n=%d fg=%s\n%s\n' "$(date '+%T')" "$n" "$fg" "$out" >> "$R/$tag.samples.txt"
    n=$((n + 1)); sleep "$per"
  done
}

ready_gate() {  # name since_epoch min_rest_s mode [poll_s] -> READY_NOTE, READY_SKIP
  # clean = wait for clean up to REST_MAX, then run warm; gate = heat does not change a verdict: run without waiting for
  # clean, but never at thermal status >= 4 (not below 4 by REST_MAX -> skipped); timing = run only from clean (not
  # clean by REST_MAX_T or the hold's deadline -> skipped, left for the next hold); uncapped (round 9) = run when the
  # clocks are uncapped (no CPU cap, kgsl thermal_pwrlevel 0) at thermal status <= 1, else as timing
  local name=$1 since=$2 minrest=$3 mode=$4 poll=${5:-$POLL} T C G lim
  READY_SKIP=0
  while [ $(( $(now) - since )) -lt "$minrest" ]; do present || hard_stop "device_lost_in_rest_before_$name"; sleep 5; done
  lim=$REST_MAX; { [ "$mode" = timing ] || [ "$mode" = uncapped ]; } && lim=$REST_MAX_T
  while true; do
    T=$(thermal); C=$(cpu_capped); G=$(gpulvl)
    echo "$(date '+%T') ready? $name: thermal ${T:-?} cap ${C:-none} kgsl_pwrlevel ${G:-?} rested $(( $(now) - since ))s mode $mode" | tee -a "$R/s_ready.txt"
    if [ "${T:-9}" = 0 ] && [ -z "$C" ] && [ "${G:-9}" = 0 ]; then READY_NOTE="clean"; break; fi
    if [ "$mode" = uncapped ] && [ -z "$C" ] && [ "${G:-9}" = 0 ] && [ "${T:-9}" -le 1 ] 2>/dev/null; then
      READY_NOTE="uncapped (thermal ${T:-?}, no CPU cap, kgsl_pwrlevel 0)"; break
    fi
    if [ "$mode" = gate ] && [ "${T:-9}" -le 3 ] 2>/dev/null; then
      READY_NOTE="gate leg, not waited for clean (thermal ${T:-?}, cap ${C:-none}, kgsl_pwrlevel ${G:-?})"; break
    fi
    if [ $(( $(now) - since )) -ge "$lim" ]; then
      [ "$mode" = clean ] && { READY_NOTE="warm (thermal ${T:-?}, cap ${C:-none}, kgsl_pwrlevel ${G:-?})"; break; }
      READY_NOTE="not ready after ${lim}s in $mode mode (thermal ${T:-?}, cap ${C:-none}, kgsl_pwrlevel ${G:-?})"; READY_SKIP=1; break
    fi
    if [ "$(now)" -ge $(( SKIP_AFTER - 20 )) ]; then
      [ "$mode" = clean ] && { READY_NOTE="warm-deadline (thermal ${T:-?}, cap ${C:-none}, kgsl_pwrlevel ${G:-?})"; break; }
      READY_NOTE="not ready by the hold's deadline in $mode mode (thermal ${T:-?}, cap ${C:-none}, kgsl_pwrlevel ${G:-?})"; READY_SKIP=1; break
    fi
    sleep "$poll"
  done
  READY_NOTE="$READY_NOTE; rested $(( $(now) - since ))s since $(hms "$since")"
  echo "$name ready gate: $READY_NOTE" | tee -a "$R/s_ready.txt"
}

ps_time() {  # file: the phone's processes with their cumulative CPU TIME (another app's TIME that grows = contamination)
  A shell ps -A -o TIME,NAME 2>/dev/null | tr -d '\r' | LC_ALL=C sort -k2 > "$1"
}

others_time() {  # before after out: the processes other than this app whose CPU TIME grew by >= 2 s during the leg
  awk -v pkg="$PKG" '
    function secs(t,   d, a, n, s) { d = 0; if (index(t, "-")) { split(t, a, "-"); d = a[1]; t = a[2] } n = split(t, a, ":"); s = 0
      for (i = 1; i <= n; i++) s = s * 60 + a[i]; return d * 86400 + s }
    FNR == 1 { next }
    NR == FNR { before[$2] = secs($1); next }
    $2 != pkg && ($2 in before) && secs($1) - before[$2] >= 2 { printf "%s +%ds\n", $2, secs($1) - before[$2] }' "$1" "$2" > "$3"
}

memory_line() {  # samples file -> the leg's peaks
  # empty = never sampled; a sampled 0 prints 0 (VmSwap is often 0 kB all along)
  awk '/^VmHWM:/ {if (h == "" || $2 + 0 > h) h = $2 + 0} /^VmSwap:/ {if (s == "" || $2 + 0 > s) s = $2 + 0}
       /^MemAvailable:/ {if (m == "" || $2 + 0 < m) m = $2 + 0} /^kgsl_page_alloc [0-9]/ {if (k == "" || $2 + 0 > k) k = $2 + 0}
       END {printf "VmHWM_max_kB %s VmSwap_max_kB %s MemAvailable_min_kB %s kgsl_page_alloc_max_B %s\n", h, s, m, k}' "$1"
}

round3_refs() {  # graph -> the --round3 options of the same graph file's Mac runs in round 3 / 4 (when they exist)
  local form L f
  L=$(echo "$1" | sed -nE 's/^d1omni_decide_L([0-9]+)_(.*)\.tflite$/\1/p'); form=$(echo "$1" | sed -nE 's/^d1omni_decide_L([0-9]+)_(.*)\.tflite$/\2/p')
  [ -n "$L" ] || return 0
  for f in cpu gpu_fp32; do
    [ -s "$K/results/litert_${f}_parity_L${L}_${form}.json" ] && printf -- '--round3 mac_%s=results/litert_%s_parity_L%s_%s.json ' "$f" "$f" "$L" "$form"
  done
}

score_leg() {  # tag rows mode: the host scorer of a DONE leg, in the background; a gate leg's verdict -> R/<tag>.verdict
  # (round 9: mode generic = scripts/s26_score.py generic, verdict FINITE / NONFINITE; the vision / audio bar is the end
  # to end, scored after the run)
  local tag=$1 rows=$2 mode=$3 base same="" r3 rows_path
  if [ "$mode" = gate ]; then
    base="mac_$(echo "$CUR_GRAPH" | sed 's/\.tflite$//')__$(echo "$rows" | sed 's/\.json$//')"
    [ -s "$R/mac/$base.json" ] && same="--same-file mac_cpu8=$R/mac/$base.json:$R/mac/sel_$base.f32"
    r3=$(round3_refs "$CUR_GRAPH")
    ( cd "$K" && "$PY" "$SCORE" gate --report "$R/$tag.json" --sel "$R/sel_$tag.f32" --rows "$ROWS_DIR/$rows" \
        --logcat "$R/$tag.logcat.txt" --samples "$R/$tag.samples.txt" --state-before "$R/$tag.state_before.txt" \
        --state-after "$R/$tag.state_after.txt" $same $r3 --out "$RESULTS/s26_parity_$tag$SFX.json" > "$R/$tag.score.txt" 2>&1
      v=$(python3 -I -c "import json,sys; print('PASS' if json.load(open(sys.argv[1]))['summary']['bar_pass'] else 'FAIL')" "$RESULTS/s26_parity_$tag$SFX.json" 2>/dev/null || echo NONE)
      echo "$v $(date '+%F %T')" > "$R/$tag.verdict" ) < /dev/null &
  elif [ "$mode" = generic ]; then
    base="mac_$(echo "$CUR_GRAPH" | sed 's/\.tflite$//')__$(echo "$rows" | sed 's/\.json$//')"
    rows_path=$ROWS_DIR/$rows
    if [ -s "$GEN/$rows" ]; then
      rows_path=$GEN/$rows          # a prep leg: its Mac run has other inputs; the scorer runs the Mac on the same ones
    elif [ -s "$R/mac/$base.json" ]; then
      same="--same-file mac_cpu8=$R/mac/$base.json:$R/mac/out_$base.f32"
    fi
    ( cd "$K" && "$PY" "$SCORE" generic --report "$R/$tag.json" --out-file "$R/out_$tag.f32" --rows "$rows_path" \
        --logcat "$R/$tag.logcat.txt" --samples "$R/$tag.samples.txt" --state-before "$R/$tag.state_before.txt" \
        --state-after "$R/$tag.state_after.txt" $same --prefix-out "$R/$tag.prefix.npz" \
        --out "$RESULTS/s26_parity_$tag$SFX.json" > "$R/$tag.score.txt" 2>&1
      v=$(python3 -I -c "import json,sys; print('FINITE' if json.load(open(sys.argv[1]))['summary']['finite_pass'] else 'NONFINITE')" "$RESULTS/s26_parity_$tag$SFX.json" 2>/dev/null || echo NONE)
      echo "$v $(date '+%F %T')" > "$R/$tag.verdict" ) < /dev/null &
  else
    ( cd "$K" && "$PY" "$SCORE" timing --report "$R/$tag.json" --samples "$R/$tag.samples.txt" \
        --state-before "$R/$tag.state_before.txt" --state-after "$R/$tag.state_after.txt" --logcat "$R/$tag.logcat.txt" \
        --out "$RESULTS/s26_timing_$tag$SFX.json" > "$R/$tag.score.txt" 2>&1 ) < /dev/null &
  fi
  SCORE_PIDS="$SCORE_PIDS $!"
}

pull_out() {  # tag: a generic leg's out_<tag>.f32 (the byte count its report names), then removed from the phone
  local tag=$1 want got
  want=$(python3 -I -c "import json,sys; print(json.load(open(sys.argv[1])).get('out_bytes', ''))" "$R/$tag.json" 2>/dev/null)
  if [ -z "$want" ]; then echo "out: none in the report"; return 0; fi
  A exec-out run-as $PKG cat "files/out_$tag.f32" > "$R/out_$tag.f32" 2>/dev/null
  present || hard_stop "device_lost_in_pull_$tag"
  got=$(stat -f %z "$R/out_$tag.f32" 2>/dev/null)
  echo "out bytes: report=$want pulled=$got $([ "$want" = "$got" ] && echo MATCH || echo CHECK)"
  A shell run-as $PKG rm -f "files/out_$tag.f32"
  [ "$want" = "$got" ] || rm -f "$R/out_$tag.f32"
}

prep_leg() {  # tag prep rows: soft:<tower leg> -> the projector rows file and its soft input, made from that leg's pulled
  # features (scripts/s26_score.py soft, host unshuffle) into R/gen/, then sent with their sha256 checked on the phone
  local tag=$1 prep=$2 rows=$3 src srows f
  case "$prep" in soft:*) src=${prep#soft:} ;; *) echo "prep $tag: unknown $prep" >> "$R/s_prep.txt"; return 1 ;; esac
  done_leg "$src" || { echo "prep $tag: $src is not DONE ($(status_of "$src"))" >> "$R/s_prep.txt"; return 1; }
  [ -s "$R/out_$src.f32" ] || { echo "prep $tag: no $R/out_$src.f32" >> "$R/s_prep.txt"; return 1; }
  srows=$(printf '%s\n' "${PLAN_LINES[@]}" | awk -v t="$src" '$1 == t {print $6; exit}')
  mkdir -p "$GEN"
  ( cd "$K" && "$PY" "$SCORE" soft --report "$R/$src.json" --out-file "$R/out_$src.f32" --rows "$ROWS_DIR/$srows" \
      --name "$rows" --out-dir "$GEN" --sha-file "$R/gen.sha256" ) >> "$R/s_prep.txt" 2>&1 \
    || { echo "prep $tag: soft from $src failed" >> "$R/s_prep.txt"; return 1; }
  for f in "$rows" $(generic_files_of "$GEN/$rows"); do
    send_small "$GEN/$f" "$f" >> "$R/s_prep.txt" 2>&1 || { echo "prep $tag: sending $f failed" >> "$R/s_prep.txt"; return 1; }
  done
  echo "prep $tag: $rows and its inputs made from $src and sent $(ts)" | tee -a "$R/s_prep.txt"
}

wait_scores() {  # the host scorers still running (never `wait` for every child: the keeper is one of them)
  local p
  for p in $SCORE_PIDS; do wait "$p" 2>/dev/null; done
  SCORE_PIDS=""
}

verdict_of() {  # tag -> PASS / FAIL / NONE, waiting up to VERDICT_WAIT_S for the background scorer of that leg
  local tag=$1 t0
  t0=$(now)
  while [ ! -s "$R/$tag.verdict" ] && [ $(( $(now) - t0 )) -lt "$VERDICT_WAIT_S" ] && [ "$(status_of "$tag")" = DONE ]; do sleep 2; done
  [ -s "$R/$tag.verdict" ] && cut -d' ' -f1 "$R/$tag.verdict" || echo NONE
}

LEG_STATUS=NONE; LEG_END=0; STOP_ALL=0; LAST_UPTIME=0; CUR_GRAPH=""; SCORE_PIDS=""
leg() {  # tag limit_s accel precision graph rows mode sample_s guard_kb extras
  local tag=$1 lim=$2 accel=$3 prec=$4 graph=$5 rows=$6 mode=$7 per=$8 guard=$9 extras=${10} rc rp mp cap up
  if [ "$(now)" -ge "$SKIP_AFTER" ]; then
    echo "SKIP $tag at $(date '+%T') (past skip_after $(hms "$SKIP_AFTER"))" | tee -a "$R/s_skipped.txt"; LEG_STATUS=SKIPPED; return
  fi
  cap=$(( RUN_END - $(now) )); [ $cap -lt "$lim" ] && lim=$cap
  echo "$READY_NOTE" > "$R/$tag.ready.txt"
  ps_time "$R/$tag.ps_before.txt"
  echo "=== leg $tag start $(date '+%T') limit ${lim}s ready: $READY_NOTE"
  LIMIT=$lim GATE_SIG=$(sig_of "$graph") GATE_THREADS=$(xget "$extras" threads 4) GATE_WARMUP=$(xget "$extras" warmup "") \
    GATE_REPS=$(xget "$extras" reps "") GATE_REST_MS=$(xget "$extras" rest_ms "") GATE_COOL_MS=$(xget "$extras" cool_ms "") \
    GATE_LIMIT_ROWS=$(xget "$extras" limit_rows "") GATE_KEEP_SCORES=$(xget "$extras" keep_scores "") \
    GATE_CPU_CACHE=$(xget "$extras" cpu_cache "") GATE_GPU_SRC_QUANT=$(xget "$extras" gpu_src_quant "") \
    "$RUN" gate "$tag" "$accel" "$prec" "$graph" "$rows" "$mode" > "$R/$tag.run.txt" 2>&1 &
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
    grep -iE "lowmemorykiller|lmkd|kill.*$PKG|$PKG.*kill" "$R/$tag.logcat_all_since_start.txt" | head -40 > "$R/$tag.kill_lines.txt"
    # an lmkd kill OF THIS APP only = lmkd lines naming the package; our own `am force-stop` (memory guard) is not lmkd
    if grep -iE "(lowmemorykiller|lmkd).*$PKG" "$R/$tag.logcat_all_since_start.txt" > "$R/$tag.lmkd_app_lines.txt" && [ -s "$R/$tag.lmkd_app_lines.txt" ]; then
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
    if [ "$mode" = gate ] && [ -s "$R/sel_$tag.f32" ]; then score_leg "$tag" "$rows" gate; fi
    if [ "$mode" = timing ] || [ "$mode" = generic_timing ]; then score_leg "$tag" "$rows" timing; fi
    if [ "$mode" = generic ] && [ -s "$R/out_$tag.f32" ]; then score_leg "$tag" "$rows" generic; fi
  fi
}

leg_fg() {  # a leg whose app could not stay in front: pause for R/RESUME (the user unlocks the phone), then one more try
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

cleanup_all() {
  "$RUN" cleanup > "$R/s9_cleanup.txt" 2>&1; local rc=$?
  cat "$R/s9_cleanup.txt"
  [ $rc = 9 ] && { echo "the phone left adb during the cleanup $(ts)" > "$R/CHAIN_STOPPED"; return 9; }
  { echo "pm list: '$(A shell pm list packages $PKG | tr -d '\r')'"
    echo "stage: $(A shell ls -d $STAGE 2>&1 | tr -d '\r')"
    echo "run-as files: $(A shell run-as $PKG ls -l files 2>&1 | tr -d '\r' | tr '\n' ' ')"
    echo "data_free: $(A shell df -h /data | tail -1 | tr -d '\r')"; } | tee "$R/s9_cleanup_check.txt"
  CLEANED=1
}

transfer_check() {  # another session's adb transfer on the Mac, or a busy app of this lane / another lane on the phone
  local busy
  {
    echo "=== transfer check $(ts)"
    echo "host adb processes:"; ps -Ao pid,ppid,etime,comm,args | awk '$4 ~ /(^|\/)adb$/ && $0 ~ / (push|pull|install|exec-in|exec-out|sync) /' | grep . || echo "  none"
    echo "phone:"; A shell "cat /proc/uptime; df -h /data | tail -1; ps -A -o PID,ETIME,TIME,NAME | grep -i -E 'gate|runner|push|sync|install|dex2oat|litert|com\.mlboy' || echo '  no app / runner / install process'"
    echo "top resumed: $(A shell dumpsys activity activities | grep -m1 topResumedActivity | tr -d '\r')"
  } > "$R/x0_transfer_check.txt" 2>&1
  ps_time "$R/x0_ps_a.txt"; sleep 5; ps_time "$R/x0_ps_b.txt"
  others_time "$R/x0_ps_a.txt" "$R/x0_ps_b.txt" "$R/x0_others_time.txt"
  { echo "CPU TIME grown in 5 s (>= 2 s):"; sed 's/^/  /' "$R/x0_others_time.txt"; } >> "$R/x0_transfer_check.txt"
  cat "$R/x0_transfer_check.txt"
  if sed -n '/host adb processes:/,/phone:/p' "$R/x0_transfer_check.txt" | grep -v "^host\|^phone\|none" | grep -q "push\|install\|exec-in\|exec-out\|pull\|sync"; then
    return 1
  fi
  # phone side: this package or a test runner at all; another com.mlboydaisuke.* app only when it is in front or its CPU
  # time grows (a cached, idle app of another lane is recorded, not a stop: memory shared-device-coordination); system
  # daemons (installd, com.sec.spp.push, gatekeeperd) are recorded only (Kev r14)
  if sed -n '/^phone:/,/^top resumed:/p' "$R/x0_transfer_check.txt" | awk '{print $NF}' | grep -qE '^(com\.mlboydaisuke\.d1omni\.gate|[A-Za-z0-9_.]*runner)$'; then
    return 1
  fi
  grep '^top resumed:' "$R/x0_transfer_check.txt" | grep -q 'com\.mlboydaisuke\.' && return 1
  busy=$(awk '{print $1}' "$R/x0_others_time.txt" | grep -E '^com\.mlboydaisuke\.' || true)
  [ -n "$busy" ] && { echo "busy on the phone: $busy"; return 1; }
  return 0
}

RELEASED=0; CLEANED=0; HELD=0
release_hold() {
  [ "$RELEASED" = 1 ] && return 0
  RELEASED=1
  date '+%F %T' > "$R/RELEASE_HOLD"
  echo "RELEASE_HOLD written $(ts)"
  local i=0
  while [ ! -f "$R/HOLD_RELEASED" ] && [ $i -lt 30 ]; do sleep 1; i=$((i + 1)); done
  [ -f "$R/HOLD_RELEASED" ] && echo "keeper released the hold at $(cat "$R/HOLD_RELEASED")" || echo "keeper has not released within 30 s (it does at its limit)"
}

on_exit() {  # any exit after the hold was seen: clean the phone if it is there and not cleaned yet, then release
  local rc=$?
  [ "$HELD" = 1 ] || exit $rc
  if [ "$CLEANED" = 0 ] && present; then echo "exit $rc before the cleanup: cleaning now $(ts)"; cleanup_all; fi
  release_hold
  wait_scores
  echo "chain exit $rc $(ts)"
  exit $rc
}

read_plan() {  # -> PLAN_LINES (the plan's legs, comments and blank lines dropped), read once: a later edit of the file
  # does not reach a running chain
  [ -s "$PLAN" ] || { echo "no plan: $PLAN" >&2; exit 5; }
  PLAN_LINES=()
  local line
  while IFS= read -r line || [ -n "$line" ]; do
    line=${line%%#*}
    [ -n "$(echo "$line" | tr -d ' \t')" ] || continue
    PLAN_LINES+=("$line")
  done < "$PLAN"
  [ ${#PLAN_LINES[@]} -gt 0 ] || { echo "the plan has no leg" >&2; exit 5; }
}

small_files() {  # the plan's rows files and the prefix files (round 9: and generic input files) they name, unique, one
  # per line; a prep leg's rows file is made during the run and is not listed
  local line tag lim accel prec graph rows mode ready extras f
  for line in "${PLAN_LINES[@]}"; do
    read -r tag lim accel prec graph rows mode ready extras <<< "$line"
    [ -n "$(xget "$extras" prep "")" ] && continue
    echo "$rows"
    [ -s "$ROWS_DIR/$rows" ] && prefix_files_of "$ROWS_DIR/$rows" && generic_files_of "$ROWS_DIR/$rows"
  done | grep . | LC_ALL=C sort -u
}

check_inputs() {  # sha256 of the APK, the plan's graphs, rows files and prefix files -> R/inputs.sha256; the leg table
  mkdir -p "$R"
  read_plan
  local line tag lim accel prec graph rows mode ready extras f prep src ok=0
  [ -s "$APK" ] || { echo "missing APK $APK"; ok=1; }
  [ -n "$APK_SHA" ] || { echo "no APK sha256 (D1_APK_SHA or $APK.sha256)"; ok=1; }
  [ -s "$APK" ] && [ "$(shasum -a 256 "$APK" | cut -d' ' -f1)" != "$APK_SHA" ] && { echo "APK sha256 differs from $APK_SHA"; ok=1; }
  : > "$R/inputs.sha256.tmp"
  for line in "${PLAN_LINES[@]}"; do
    read -r tag lim accel prec graph rows mode ready extras <<< "$line"
    case "$accel" in gpu|cpu) ;; *) echo "$tag: accel $accel"; ok=1 ;; esac
    case "$mode" in gate|timing|generic|generic_timing) ;; *) echo "$tag: mode $mode"; ok=1 ;; esac
    case "$ready" in clean|gate|timing|uncapped) ;; *) echo "$tag: ready $ready"; ok=1 ;; esac
    [ "$accel" = cpu ] || case "$prec" in fp32|fp16acc32|fp16|default) ;; *) echo "$tag: precision $prec"; ok=1 ;; esac
    [ -s "$GRAPH_DIR/$graph" ] || { echo "$tag: missing graph $GRAPH_DIR/$graph"; ok=1; }
    prep=$(xget "$extras" prep "")
    if [ -n "$prep" ]; then
      src=${prep#soft:}
      [ "$prep" = "soft:$src" ] && [ "$mode" = generic ] \
        && printf '%s\n' "${PLAN_LINES[@]}" | awk -v t="$src" '$1 == t && $7 == "generic" {f = 1} END {exit f ? 0 : 1}' \
        || { echo "$tag: prep $prep needs mode generic and an earlier generic leg $src"; ok=1; }
    else
      [ -s "$ROWS_DIR/$rows" ] || { echo "$tag: missing rows $ROWS_DIR/$rows"; ok=1; }
    fi
    [ -n "$(sig_of "$graph")" ] || echo "$tag: graph name without L (the app resolves the signature itself)"
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

mac_generic_leg() {  # tag graph rows extras: the Mac CPU (8 threads) run of a generic gate leg (round 9) + its score; a
  # prep leg's inputs are made from the Mac run of its source leg (the Mac chain, e.g. tower -> unshuffle -> projector)
  local tag=$1 graph=$2 rows=$3 extras=$4 base prep src sline sgraph srows sbase f rowsdir=$ROWS_DIR x
  base="mac_${graph%.tflite}__${rows%.json}"
  prep=$(xget "$extras" prep "")
  if [ -n "$prep" ]; then
    src=${prep#soft:}
    sline=$(printf '%s\n' "${PLAN_LINES[@]}" | awk -v t="$src" '$1 == t {print; exit}')
    read -r x x x x sgraph srows x <<< "$sline"
    sbase="mac_${sgraph%.tflite}__${srows%.json}"
    [ -s "$R/mac/$sbase.json" ] || { echo "$base: no Mac run of $src ($sbase) to make its soft from"; return 0; }
    rowsdir=$R/mac/gen_$tag
    ( cd "$K" && "$PY" "$SCORE" soft --report "$R/mac/$sbase.json" --out-file "$R/mac/out_$sbase.f32" \
        --rows "$ROWS_DIR/$srows" --name "$rows" --out-dir "$rowsdir" ) > "$R/mac/$base.soft.txt" 2>&1 \
      || { echo "$base: soft from $sbase failed ($R/mac/$base.soft.txt)"; return 0; }
  fi
  if [ ! -s "$R/mac/$base.json" ]; then
    ln -sf "$GRAPH_DIR/$graph" "$R/mac/$graph"; ln -sf "$rowsdir/$rows" "$R/mac/$rows"
    for f in $(generic_files_of "$rowsdir/$rows"); do ln -sf "$rowsdir/$f" "$R/mac/$f"; done
    ( cd "$K" && "$PY" "$STANDIN" --files-dir "$R/mac" --graph "$graph" --rows "$rows" --report "$base.json" \
        --mode generic --accel cpu --threads 8 ) > "$R/mac/$base.stdout.txt" 2>&1
    echo "$base: $(python3 -I -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get('status'), d.get('error', ''), d.get('summary', {}).get('count'), d.get('out_bytes'))" "$R/mac/$base.json")"
    rm -f "$R/mac/$graph" "$R/mac/$rows"
    for f in $(generic_files_of "$rowsdir/$rows"); do rm -f "$R/mac/$f"; done
  else
    echo "$base: already there"
  fi
  if [ ! -s "$RESULTS/s26_macbase_$base$SFX.json" ] && [ -s "$R/mac/$base.json" ]; then
    ( cd "$K" && "$PY" "$SCORE" generic --report "$R/mac/$base.json" --out-file "$R/mac/out_$base.f32" --rows "$rowsdir/$rows" \
        --prefix-out "$R/mac/$base.prefix.npz" --tag "$base" --out "$RESULTS/s26_macbase_$base$SFX.json" ) > "$R/mac/$base.score.txt" 2>&1
    echo "$base score rc $?: $(python3 -I -c "import json,sys; d=json.load(open(sys.argv[1])); print(json.dumps(d['summary']))" "$RESULTS/s26_macbase_$base$SFX.json" 2>/dev/null | cut -c1-900)"
  fi
}

mac_baseline() {  # the Mac CPU (8 threads) run of every gate leg's graph + rows, the same files as on the phone, and its score
  read_plan
  mkdir -p "$R/mac"
  local line tag lim accel prec graph rows mode ready extras base f r3
  for line in "${PLAN_LINES[@]}"; do
    read -r tag lim accel prec graph rows mode ready extras <<< "$line"
    if [ "$mode" = generic ]; then mac_generic_leg "$tag" "$graph" "$rows" "$extras"; continue; fi
    [ "$mode" = gate ] || continue
    base="mac_${graph%.tflite}__${rows%.json}"
    if [ ! -s "$R/mac/$base.json" ]; then
      ln -sf "$GRAPH_DIR/$graph" "$R/mac/$graph"; ln -sf "$ROWS_DIR/$rows" "$R/mac/$rows"
      for f in $(prefix_files_of "$ROWS_DIR/$rows"); do ln -sf "$ROWS_DIR/$f" "$R/mac/$f"; done
      ( cd "$K" && "$PY" "$STANDIN" --files-dir "$R/mac" --graph "$graph" --rows "$rows" --report "$base.json" \
          --mode gate --accel cpu --threads 8 --sig "$(sig_of "$graph")" ) > "$R/mac/$base.stdout.txt" 2>&1
      echo "$base: $(python3 -I -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get('status'), d.get('error', ''), d.get('summary', {}).get('count'), d.get('layout_equals_host_build_inputs'))" "$R/mac/$base.json")"
      rm -f "$R/mac/$graph" "$R/mac/$rows"
      for f in $(prefix_files_of "$ROWS_DIR/$rows"); do rm -f "$R/mac/$f"; done
    else
      echo "$base: already there"
    fi
    if [ ! -s "$RESULTS/s26_macbase_$base$SFX.json" ] && [ -s "$R/mac/$base.json" ]; then
      r3=$(round3_refs "$graph")
      ( cd "$K" && "$PY" "$SCORE" gate --report "$R/mac/$base.json" --sel "$R/mac/sel_$base.f32" --rows "$ROWS_DIR/$rows" \
          $r3 --tag "$base" --out "$RESULTS/s26_macbase_$base$SFX.json" ) > "$R/mac/$base.score.txt" 2>&1
      echo "$base score rc $?: $(python3 -I -c "import json,sys; d=json.load(open(sys.argv[1])); s=d['summary']; print(s['rows'], s['max_abs_dp'], s['argmax_outside_near_tie'], s['near_tie'], s['bar_pass'], json.dumps(d['round3_same_file']))" "$RESULTS/s26_macbase_$base$SFX.json" 2>/dev/null)"
    fi
  done
}

take_hold() {  # the keeper is its own process (its pid owns the hold, so the hold outlives a chain that dies): start it,
  # wait for HOLD_ACQUIRED, then set the clock and the exit trap (clean the phone if needed, release)
  rm -f "$R/RELEASE_HOLD" "$R/HOLD_RELEASED" "$R/GAVE_UP"
  echo "===== keeper for $SCRIPT_NAME started by chain pid $$ $(ts)" >> "$R/keeper.log"
  nohup bash "$KEEPER" "$R" "$HOLD_FILE" "$SCRIPT_NAME" >> "$R/keeper.log" 2>&1 < /dev/null &
  echo "$!" > "$R/keeper.pid"
  echo "keeper pid $(cat "$R/keeper.pid") waiting in line for $HOLD_FILE as $SCRIPT_NAME $(ts)"
  until [ -f "$R/HOLD_ACQUIRED" ]; do
    [ -f "$R/GAVE_UP" ] && { echo "keeper gave up $(ts)"; echo gave_up > "$R/CHAIN_STOPPED"; exit 3; }
    kill -0 "$(cat "$R/keeper.pid")" 2>/dev/null || { echo "keeper died before the hold $(ts)"; exit 3; }
    sleep 2
  done
  HELD=1
  trap on_exit EXIT
  ACQ=$(date -j -f '%Y-%m-%d %H:%M:%S' "$(sed -n 's/^pid [0-9]* since //p' "$R/HOLD_ACQUIRED")" +%s 2>/dev/null || now)
  SKIP_AFTER=$(( ACQ + SKIP_MIN * 60 )); RUN_END=$(( ACQ + RUN_END_MIN * 60 )); END_AT=$(( ACQ + END_MIN * 60 ))
  echo "HOLD_SEEN $(ts): acquired $(hms "$ACQ"), skip_after $(hms "$SKIP_AFTER"), run_end $(hms "$RUN_END"), released by $(hms "$END_AT")"
}

cleanup_cmd() {  # by hand after a hard stop: never while a chain of this round runs; the live hold of a dead chain is
  # used and then released, else a hold is taken for the cleanup
  local cp kp
  cp=$(cat "$R/chain.pid" 2>/dev/null); kp=$(cat "$R/keeper.pid" 2>/dev/null)
  if [ -n "$cp" ] && kill -0 "$cp" 2>/dev/null && ps -o command= -p "$cp" | grep -q s26_chain; then
    echo "a chain of this round is running (pid $cp): not cleaning under it" >&2; exit 4
  fi
  echo "$$" > "$R/chain.pid"
  if [ -f "$R/HOLD_ACQUIRED" ] && [ -n "$kp" ] && kill -0 "$kp" 2>/dev/null; then
    echo "the hold of this round is live (keeper $kp, the chain is gone): cleaning, then releasing"
    HELD=1; trap on_exit EXIT
  else
    take_hold
  fi
  present || hard_stop "device_absent_at_cleanup"
  cleanup_all
  release_hold
}

run_chain() {
  mkdir -p "$R"
  rm -f "$R/CHAIN_DONE" "$R/CHAIN_STOPPED" "$R/PAUSED_FG" "$R/RESUME" "$R/RELEASE_HOLD"
  echo "$$" > "$R/chain.pid"
  echo "chain $ROUND pid $$ $(ts); this file sha256 $(shasum -a 256 "$0" | cut -c1-16); plan $PLAN sha256 $(shasum -a 256 "$PLAN" | cut -c1-16)"
  check_inputs > "$R/s0_inputs.txt" 2>&1 || { cat "$R/s0_inputs.txt"; echo "inputs check failed: no hold taken"; exit 5; }
  read_plan
  [ "$(LC_ALL=C sort -u <<< "$(printf '%s\n' "${PLAN_LINES[@]}" | awk '{print $1}')" | wc -l | tr -d ' ')" = "${#PLAN_LINES[@]}" ] || { echo "duplicate leg tags in the plan"; exit 5; }
  mkdir -p "$RESULTS"
  take_hold
  READY_NOTE=""; LEG_END=$(now)
  present || hard_stop "device_absent_at_start"
  echo "stay_on_while_plugged_in at $(ts): $(A shell settings get global stay_on_while_plugged_in | tr -d '\r') (read only)" | tee "$R/s_stayon.txt"
  if ! transfer_check; then
    echo "OTHER SESSION'S TRANSFER / PROCESS ON THE PHONE -> no leg, release $(ts)"
    echo "other_session_job $(ts)" > "$R/CHAIN_STOPPED"; CLEANED=1; exit 5
  fi
  LAST_UPTIME=$(uptime_s); echo "phone uptime at the hold: ${LAST_UPTIME}s"
  "$RUN" state > "$R/s0_state_initial.txt" 2>&1; cat "$R/s0_state_initial.txt"
  local ok line tag lim accel prec graph rows mode ready extras f
  (
    echo "setup start $(ts)"
    [ "$(shasum -a 256 "$APK" | cut -d' ' -f1)" = "$APK_SHA" ] || { echo "APK sha256 mismatch"; exit 5; }
    echo "data free before: $(avail_bytes) B"
    A install -r "$APK" || exit 5
    A shell run-as $PKG mkdir -p files
    for f in $(small_files); do
      send_small "$ROWS_DIR/$f" "$f" || exit 6
    done
    A shell rm -rf $STAGE
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
    swap_to "$graph" $keep || continue
    CUR_GRAPH=$graph
    ready_gate "$tag" "$LEG_END" "$rest" "$ready" "$(xget "$extras" poll "$POLL")"
    [ "${READY_SKIP:-0}" = 1 ] && { echo "SKIP $tag at $(date '+%T'): $READY_NOTE (left for the next hold)" | tee -a "$R/s_skipped.txt"; continue; }
    [ "$(now)" -lt "$SKIP_AFTER" ] || { echo "SKIP $tag at $(date '+%T') (past skip_after after the ready gate)" | tee -a "$R/s_skipped.txt"; continue; }
    date '+%F %T' > "$R/$tag.attempted"
    leg_fg "$tag" "$lim" "$accel" "$prec" "$graph" "$rows" "$mode" "$per" "$guard" "$extras"
  done
  cleanup_all || hard_stop "device_lost_in_cleanup"
  "$RUN" state > "$R/s9_state_end.txt" 2>&1; cat "$R/s9_state_end.txt"
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
  *) echo "usage: $0 check | mac-baseline | run | cleanup <round>" >&2; exit 2 ;;
esac
