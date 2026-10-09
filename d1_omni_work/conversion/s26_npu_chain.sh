#!/bin/bash
# Round 10: the d1-omni NPU chain on the Galaxy S26: the decision graph on the Qualcomm HTP (LiteRT 2.2.0 JIT through the
# Qualcomm compiler plugin, NPU + CPU) and the same runner on the GPU for the A/B, with a C-API runner on LiteRT 2.2.0
# (not included; staged as device/r10/stage/d1c_npu_runner) and the NPU runtime libraries of LiteRT 2.2.0 and the
# SoC vendor (not included; staged as device/r10/stage/libs and checked against their SHA256SUMS = $LIBS_SUMS; never
# copied outside the stage and the phone's /data/local/tmp/d1omni_npu). Built from round 5 / 9's scripts/s26_chain.sh
# (the keeper and the named waiting line, the ready gate, the 2 s sampler, others' CPU TIME, the transfer check, the
# 25 / 28 / 30-minute clock) plus the runner's stage, command, library paths, the logcat since a device mark and the
# runner killed by pidof after a host-side stop. One hold of at most 30 minutes per run.
#   bash scripts/s26_npu_chain.sh check <round>      the plan and its inputs with sha256 (graphs, runner, libs, fixtures); no adb
#   bash scripts/s26_npu_chain.sh run <round>        the phone: hold, setup, legs, cleanup, release
#   bash scripts/s26_npu_chain.sh cleanup <round>    by hand after a hard stop: the dead chain's live hold, or a new hold
# Round dir R = d1_omni_work/device/<round>/ (D1_R overrides); plan = R/plan_<round>.tsv (D1_PLAN overrides); one leg per
# line (# comments; whitespace-separated; extras key=value):
#   tag  accel  precision  graph  fx  count  warmup  rounds  ab  cache  compile_s  ready  [extras]
#   accel npu+cpu | npu | gpu; precision - (npu) | fp32 | fp16acc32 (gpu); graph = L128_f16safe_fp16 | L256_f16safe_fp16 |
#   L128_fp16 (out/d1omni_decide_<graph>.tflite); fx = fx_L128 | fx_L256 (device/r10/stage/<fx>/ + <fx>.manifest.json);
#   count = fixtures of the parity pass (000..count-1); warmup / rounds / ab = the latency phase on fixture ab (rounds 0 =
#   none); cache = - | new:<dir> (a JIT cold run: <dir> removed first) | use:<dir> (a cached run: skipped unless <dir>
#   is on the phone); compile_s = the compile watch (no `compile_end` line by then -> the runner is killed on the phone,
#   COMPILE_TIMEOUT, the next leg; never retried); ready = gate | clean | uncapped (round 9's modes).
#   extras: rest=<s> (since the previous leg, default 30), needs=<tag> (run only when that leg is DONE: a cached leg never
#   reuses the cache of a compile that did not end), guard=<kB> (MemAvailable below it -> the runner is killed,
#   OOM_GUARD; default MEM_GUARD_KB), sample=<s> (default 2).
# Phone layout: /data/local/tmp/d1omni_npu/ = the runner + the 12 .so (flat: --libdir, LD_LIBRARY_PATH and the
# ADSP_LIBRARY_PATH head all point there; the compiler plugin's DT_NEEDED libQnnIr.so etc. resolve only through
# LD_LIBRARY_PATH = memory npu-jit-srq-litertlm), fx_L128/ and fx_L256/ (the fixtures), the graph files, the JIT cache
# dirs (one per leg; a cache is about 1.3 GB, kept only while a later leg uses it), out_<tag>/ (the dumps, pulled and
# removed after each leg). Nothing else on the phone is touched; the cleanup removes that one directory.
# Per leg: the time checks -> the graph on the phone (bytes polled, toybox sha256sum; graphs and caches no later leg needs
# are removed first; /data free >= the graph + 1.4 GB per JIT cache + DATA_MARGIN) -> the ready gate -> the state ->
# a device clock mark -> the runner (adb shell, stdout + stderr -> R/<tag>.stdout.txt) with the 2 s sampler, the compile
# watch, the memory guard and the run-end watch -> any runner left on the phone killed by `pidof` (a host-side stop
# kills only the adb client: Kev round 11) -> /proc/uptime (went down = the phone rebooted: hard stop) -> the state ->
# logcat -b all since the mark (R/<tag>.logcat_all.txt; the runner pid's key lines -> R/<tag>.delegate.txt; an lmkd line
# naming the runner = hard stop) -> the dumps pulled (count and bytes checked) -> out_<tag>/ removed -> the cache size ->
# R/<tag>.leg.json -> scripts/npu_score.py leg in the background -> items no later leg needs removed.
# Clock (minutes after the hold): no leg starts after SKIP_MIN (25), every runner is stopped by RUN_END_MIN (28) (a
# compile watch never reaches past it), the keeper releases at END_MIN (30) at the latest. Exit 9 = the phone left adb
# or rebooted, or lmkd killed the runner (hard stop: the phone is cleaned if it is there, the hold is released).
# Env: D1_HOLD_FILE (default the shared S26 hold file), D1_SCRIPT_NAME (default d1c-<round>-s26), D1_NPU_RUNNER (the
# runner file to stage; default device/r10/stage/d1c_npu_runner), D1_NPU_LIBS (default device/r10/stage/libs),
# D1_RESULTS_DIR (default results/), ADB (default the SDK's adb), PY (the scorer's python, default
# ~/venvs/lt094dev/bin/python), D1_DRY_RUN=1 (requires ADB = scripts/fake_adb10.py, a hold file under d1_omni_work/cache/
# and a results dir outside results/; the dry run's driver is scripts/dryrun10.sh).
# Round 12 (generalized; a round 10 call behaves as before): the graph column takes any out/ file: L<n>_* = the decision
# graph out/d1omni_decide_<graph>.tflite (round 10's names), <name>.tflite = out/<name>.tflite, anything else =
# out/d1omni_<graph>.tflite (e.g. audio_T1001_f16safe_fp16, vision_tower_f16safe_fp16, projector_fp16); the fx column
# takes any fixture dir whose manifest sits in device/<round>/ (D1_FXROOT overrides; scripts/npu_fixtures.py write-specs):
# the dump size check reads the manifest's out_elems (output 0's elements; round 10's manifests: L), the ab fixture check
# reads its first input's name; the runner and the libs default to device/<round>/stage/; extras pair=<leg> (the
# scorer's GPU precision pairs) is carried in the plan only.
# Kept out (Kev r11-r17, rounds 5 / 9): no `timeout adb` (waits poll the phone); every remote `date` format quoted;
# `mkdir -p` before a staged push; `LC_ALL=C sort` on the Mac side of every sorted hash; no pkill / pgrep -f (the runner is
# found by `pidof d1c_npu_runner`, a name only this lane uses); never a bare `wait` (the keeper is a child); this file is
# never edited while it runs (write a new file and mv it over this one).
set -u
K=${K:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
CA=${CA:?set CA to the directory of the shared-device hold tools (queue_cli.py, hold_cli.py)}
CMD=${1:-}; ROUND=${2:-}
[ -n "$ROUND" ] || { echo "usage: $0 check | run | cleanup <round>" >&2; exit 2; }
R=${D1_R:-$K/device/$ROUND}
KEEPER=$K/scripts/s26_npu_keeper.sh
SCORE=$K/scripts/npu_score.py
PY=${PY:-python3}
ADB=${ADB:-adb}
SERIAL=${D1_SERIAL:-RFGL80R6A6H}
STAGE=/data/local/tmp/d1omni_npu
RNAME=d1c_npu_runner
RUNNER_BIN=${D1_NPU_RUNNER:-$K/device/$ROUND/stage/d1c_npu_runner}
RUNNER_SHA_REF=102a0f0a815df1a29a2db55cdcb71c9177f95de596c6f660f9596a27c22b3c94   # the staged runner binary (not included)
LIBS=${D1_NPU_LIBS:-$K/device/$ROUND/stage/libs}
LIBS_SUMS=${LIBS_SUMS:?set LIBS_SUMS to the SHA256SUMS file of the NPU runtime libraries}
FXROOT=${D1_FXROOT:-$K/device/$ROUND}
HOLD_FILE=${D1_HOLD_FILE:-$CA/s2_npu_sweep/.device_hold}
SCRIPT_NAME=${D1_SCRIPT_NAME:-d1c-$ROUND-s26}
PLAN=${D1_PLAN:-$R/plan_$ROUND.tsv}
RESULTS=${D1_RESULTS_DIR:-$K/results}
DRY_RUN=${D1_DRY_RUN:-0}
SKIP_MIN=${SKIP_MIN:-25}; RUN_END_MIN=${RUN_END_MIN:-28}; END_MIN=${END_MIN:-30}
REST_MAX=${REST_MAX:-180}; REST_MAX_T=${REST_MAX_T:-300}; POLL=${POLL:-10}
MEM_GUARD_KB=${MEM_GUARD_KB:-1000000}
DATA_MARGIN=${DATA_MARGIN:-2000000000}
CACHE_EST=${CACHE_EST:-1400000000}
XFER_MAX_S=${XFER_MAX_S:-300}
LIBPATH_ADSP="$STAGE;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp"

if [ "$DRY_RUN" = 1 ]; then
  case "$(basename "$ADB")" in fake_adb10.py) ;; *) echo "D1_DRY_RUN=1 needs ADB=$K/scripts/fake_adb10.py (got $ADB)" >&2; exit 2 ;; esac
  case "$HOLD_FILE" in "$K"/cache/*) ;; *) echo "D1_DRY_RUN=1 needs a hold file under $K/cache (got $HOLD_FILE)" >&2; exit 2 ;; esac
  case "$RESULTS" in "$K"/results|"$K"/results/) echo "D1_DRY_RUN=1 needs D1_RESULTS_DIR outside results/" >&2; exit 2 ;; esac
elif [ "$(basename "$ADB")" = fake_adb10.py ]; then
  echo "the fake adb outside a dry run (set D1_DRY_RUN=1)" >&2; exit 2
fi

A() { "$ADB" -s "$SERIAL" "$@" < /dev/null; }
now() { date +%s; }
ts() { date '+%F %T'; }
hms() { date -r "$1" '+%T'; }
present() { [ "$("$ADB" -s "$SERIAL" get-state 2>/dev/null < /dev/null | tr -d '\r')" = "device" ]; }
hard_stop() { echo "HARD STOP: $1 $(ts)"; echo "$1 $(ts)" > "$R/CHAIN_STOPPED"; exit 9; }
thermal() { A shell dumpsys thermalservice | grep -m1 'Thermal Status' | tr -d '\r' | grep -oE '[0-9]+$'; }
cpu_capped() { A shell 'for p in /sys/devices/system/cpu/cpufreq/policy*; do echo "${p##*/} $(cat $p/scaling_max_freq) $(cat $p/cpuinfo_max_freq)"; done' | tr -d '\r' | awk '$2 != $3 {printf "%s:%s/%s ", $1, $2, $3}'; }
gpulvl() { A shell cat /sys/class/kgsl/kgsl-3d0/thermal_pwrlevel 2>/dev/null | tr -d '\r'; }
uptime_s() { A shell cat /proc/uptime 2>/dev/null | tr -d '\r' | awk '{printf "%d", $1}'; }
avail_bytes() { A shell df -k /data | tail -1 | awk '{print $4 * 1024}' | tr -d '\r'; }
xget() {  # extras key default
  local kv
  for kv in $1; do case "$kv" in "$2"=*) echo "${kv#*=}"; return ;; esac; done
  echo "$3"
}
graph_file() { case "$1" in L[0-9]*_*) echo "d1omni_decide_$1.tflite" ;; *.tflite) echo "$1" ;; *) echo "d1omni_$1.tflite" ;; esac; }
runner_pid() { A shell pidof $RNAME 2>/dev/null | tr -d '\r'; }
dev_sha() { A shell sha256sum "$1" 2>/dev/null | cut -d' ' -f1 | tr -d '\r'; }
manifest_of() { echo "$FXROOT/$1.manifest.json"; }
jq_get() { python3 -I -c "import json,sys; d=json.load(open(sys.argv[1])); print(d[sys.argv[2]])" "$1" "$2"; }
out_elems() { python3 -I -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get('out_elems') or d['L'])" "$1"; }
first_input() { python3 -I -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['order']['inputs_subgraph_order'][0]['name'])" "$1"; }
krel() { case "$1" in "$K"/*) echo "${1#$K/}" ;; *) echo "$1" ;; esac; }

state() {  # round 5's s26_run.sh `state` lines (scripts/s26_score.py parse_state reads them)
  echo "time: $(date '+%F %T')"
  echo "uptime: $(A shell cat /proc/uptime | tr -d '\r')"
  local th fq
  th=$(A shell dumpsys thermalservice | tr -d '\r')
  echo "thermal: $(echo "$th" | grep -m1 'Thermal Status')"
  echo "skin: $(echo "$th" | grep -m1 -oE 'mValue=[-0-9.]+, mType=3, mName=SKIN')"
  echo "battery_temp: $(A shell dumpsys battery | grep -m1 temperature | tr -d '\r ')"
  echo "battery_level: $(A shell dumpsys battery | grep -m1 level | tr -d '\r ')"
  echo "screen: $(A shell dumpsys power | grep -m1 -E 'mWakefulness=' | tr -d '\r ')"
  echo "top: $(A shell dumpsys activity activities | grep -m1 topResumedActivity | tr -d '\r')"
  echo "kgsl: temp=$(A shell cat /sys/class/kgsl/kgsl-3d0/temp 2>/dev/null | tr -d '\r') clock_mhz=$(A shell cat /sys/class/kgsl/kgsl-3d0/clock_mhz 2>/dev/null | tr -d '\r') max_clock_mhz=$(A shell cat /sys/class/kgsl/kgsl-3d0/max_clock_mhz 2>/dev/null | tr -d '\r') thermal_pwrlevel=$(A shell cat /sys/class/kgsl/kgsl-3d0/thermal_pwrlevel 2>/dev/null | tr -d '\r')"
  fq=$(A shell 'for p in /sys/devices/system/cpu/cpufreq/policy*; do echo "${p##*/} $(cat $p/scaling_cur_freq) $(cat $p/scaling_max_freq) $(cat $p/cpuinfo_max_freq)"; done' | tr -d '\r')
  echo "freq (policy cur scaling_max cpuinfo_max): $(echo "$fq" | tr '\n' ';')"
  echo "freq_capped: $(echo "$fq" | awk '$3 != $4 {printf "%s:%s/%s ", $1, $3, $4; c=1} END {if (!c) printf "none"}')"
  echo "mem: $(A shell grep -E 'MemAvailable' /proc/meminfo | tr -d '\r')"
  echo "data_free: $(A shell df -h /data | tail -1 | tr -d '\r')"
  echo "stage: $(A shell "du -sk $STAGE 2>/dev/null" | tr -d '\r')"
}

SAMPLE_CMD='echo "devtime $(date +%s)"; p=$(pidof d1c_npu_runner); echo "kgsl $(cat /sys/class/kgsl/kgsl-3d0/clock_mhz) $(cat /sys/class/kgsl/kgsl-3d0/max_clock_mhz) $(cat /sys/class/kgsl/kgsl-3d0/thermal_pwrlevel) $(cat /sys/class/kgsl/kgsl-3d0/temp)"; for q in /sys/devices/system/cpu/cpufreq/policy*; do echo "cpu ${q##*/} $(cat $q/scaling_cur_freq) $(cat $q/scaling_max_freq) $(cat $q/cpuinfo_max_freq)"; done; grep -E "MemAvailable|MemFree|SwapFree" /proc/meminfo; echo "pid $p"; [ -n "$p" ] && grep -E "VmRSS|VmHWM|VmSwap" /proc/$p/status; echo "kgsl_page_alloc $(cat /sys/class/kgsl/kgsl/page_alloc 2>/dev/null)"'
SLOW_CMD='t=$(dumpsys thermalservice); echo "$t" | grep -m1 "Thermal Status"; echo "$t" | grep -m1 -oE "mValue=[-0-9.]+, mType=3, mName=SKIN"; dumpsys battery | grep -m1 temperature; dumpsys power | grep -m1 -oE "mWakefulness=[A-Za-z]+"'

ready_gate() {  # name since_epoch min_rest_s mode -> READY_NOTE, READY_SKIP (round 9's modes)
  local name=$1 since=$2 minrest=$3 mode=$4 T C G lim
  READY_SKIP=0
  while [ $(( $(now) - since )) -lt "$minrest" ]; do present || hard_stop "device_lost_in_rest_before_$name"; sleep 5; done
  lim=$REST_MAX; [ "$mode" = uncapped ] && lim=$REST_MAX_T
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
    if [ $(( $(now) - since )) -ge "$lim" ] || [ "$(now)" -ge $(( SKIP_AFTER - 20 )) ]; then
      if [ "$mode" = gate ]; then
        READY_NOTE="not below thermal 4 by the limit (thermal ${T:-?})"; READY_SKIP=1; break
      fi
      READY_NOTE="warm after ${lim}s or the deadline in $mode mode (thermal ${T:-?}, cap ${C:-none}, kgsl_pwrlevel ${G:-?})"; break
    fi
    sleep "$POLL"
  done
  READY_NOTE="$READY_NOTE; rested $(( $(now) - since ))s since $(hms "$since")"
  echo "$name ready gate: $READY_NOTE" | tee -a "$R/s_ready.txt"
}

ps_time() { A shell ps -A -o TIME,NAME 2>/dev/null | tr -d '\r' | LC_ALL=C sort -k2 > "$1"; }
others_time() {  # before after out: processes other than the runner whose CPU TIME grew by >= 2 s
  awk -v me="$RNAME" '
    function secs(t,   d, a, n, s) { d = 0; if (index(t, "-")) { split(t, a, "-"); d = a[1]; t = a[2] } n = split(t, a, ":"); s = 0
      for (i = 1; i <= n; i++) s = s * 60 + a[i]; return d * 86400 + s }
    FNR == 1 { next }
    NR == FNR { before[$2] = secs($1); next }
    $2 != me && ($2 in before) && secs($1) - before[$2] >= 2 { printf "%s +%ds\n", $2, secs($1) - before[$2] }' "$1" "$2" > "$3"
}
memory_line() {
  awk '/^VmHWM:/ {if (h == "" || $2 + 0 > h) h = $2 + 0} /^VmSwap:/ {if (s == "" || $2 + 0 > s) s = $2 + 0}
       /^MemAvailable:/ {if (m == "" || $2 + 0 < m) m = $2 + 0}
       END {printf "VmHWM_max_kB %s VmSwap_max_kB %s MemAvailable_min_kB %s\n", h, s, m}' "$1"
}

transfer_check() {  # another session's adb transfer on the Mac, or a runner / app / install busy on the phone
  local busy
  {
    echo "=== transfer check $(ts)"
    echo "host adb processes:"; ps -Ao pid,ppid,etime,comm,args | awk '$4 ~ /(^|\/)adb$/ && $0 ~ / (push|pull|install|exec-in|exec-out|sync) /' | grep . || echo "  none"
    echo "phone:"; A shell "cat /proc/uptime; df -h /data | tail -1; ps -A -o PID,ETIME,TIME,NAME | grep -i -E 'gate|runner|push|sync|install|dex2oat|litert|com\.mlboy' || echo '  no app / runner / install process'"
    echo "top resumed: $(A shell dumpsys activity activities | grep -m1 topResumedActivity | tr -d '\r')"
    echo "our stage before setup: $(A shell ls -d $STAGE 2>&1 | tr -d '\r')"
  } > "$R/x0_transfer_check.txt" 2>&1
  ps_time "$R/x0_ps_a.txt"; sleep 5; ps_time "$R/x0_ps_b.txt"
  others_time "$R/x0_ps_a.txt" "$R/x0_ps_b.txt" "$R/x0_others_time.txt"
  { echo "CPU TIME grown in 5 s (>= 2 s):"; sed 's/^/  /' "$R/x0_others_time.txt"; } >> "$R/x0_transfer_check.txt"
  cat "$R/x0_transfer_check.txt"
  # a host adb transfer stops the run when it names this phone's serial or names no serial at all (it may reach this
  # phone); a transfer to another phone (-s <other serial>) is recorded only
  local tl
  tl=$(sed -n '/host adb processes:/,/phone:/p' "$R/x0_transfer_check.txt" | grep -v "^host\|^phone\|none" | grep "push\|install\|exec-in\|exec-out\|pull\|sync")
  if [ -n "$tl" ] && { echo "$tl" | grep -q -- " -s $SERIAL" || echo "$tl" | grep -qv -- " -s "; }; then
    if [ "$DRY_RUN" = 1 ]; then echo "(dry run: a host adb transfer is recorded, not a stop - the fake phone is not on USB)"; else return 1; fi
  fi
  # phone side: a native runner of any lane (a process name ending in "runner"), another com.mlboydaisuke.* app in front
  # or whose CPU time grows (an idle cached app of another lane is recorded only: memory shared-device-coordination)
  if sed -n '/^phone:/,/^top resumed:/p' "$R/x0_transfer_check.txt" | awk '{print $NF}' | grep -qE '^[A-Za-z0-9_.]*runner$'; then
    return 1
  fi
  grep '^top resumed:' "$R/x0_transfer_check.txt" | grep -q 'com\.mlboydaisuke\.' && return 1
  busy=$(awk '{print $1}' "$R/x0_others_time.txt" | grep -E '^com\.mlboydaisuke\.' || true)
  [ -n "$busy" ] && { echo "busy on the phone: $busy"; return 1; }
  return 0
}

push_file() {  # src dst: adb push, then the byte count polled (the client can return early: Kev r14) and the sha256
  local src=$1 dst=$2 want got last=-1 still=0 t0 rc h d
  want=$(stat -L -f %z "$src"); t0=$(now)
  A push "$src" "$dst" > /dev/null 2>&1; rc=$?
  present || hard_stop "device_lost_in_push_$(basename "$dst")"
  while :; do
    got=$(A shell stat -c %s "$dst" 2>/dev/null | tr -d '\r')
    [ -n "$got" ] && [ "$got" = "$want" ] && break
    if [ "$got" = "$last" ]; then still=$(( still + 5 )); else still=0; fi
    last=$got
    { [ "$still" -ge 30 ] || [ $(( $(now) - t0 )) -ge "$XFER_MAX_S" ]; } && break
    sleep 5
  done
  h=$(shasum -a 256 "$src" | cut -d' ' -f1); d=$(dev_sha "$dst")
  echo "push $(basename "$dst") rc=$rc bytes ${got:-<none>}/$want sha256 phone=${d:-<none>} mac=$h $([ "$h" = "$d" ] && echo MATCH || echo MISMATCH) in $(( $(now) - t0 ))s"
  [ "$rc" = 0 ] && [ "$got" = "$want" ] && [ "$h" = "$d" ]
}

push_fx() {  # fx: the fixture dir -> STAGE/<fx>, then the file count and the concatenated sha256 (byte-sorted names)
  local fx=$1 man src want_n want_sha got_n got_sha t0
  man=$(manifest_of "$fx"); src=$K/$(jq_get "$man" dir)
  want_n=$(jq_get "$man" files); want_sha=$(jq_get "$man" concat_sha256_bytesorted)
  t0=$(now)
  A shell mkdir -p "$STAGE/$fx"
  A push "$src/." "$STAGE/$fx/" > /dev/null 2>&1
  present || hard_stop "device_lost_in_push_$fx"
  got_n=$(A shell "ls $STAGE/$fx | wc -l" | tr -d '\r ')
  got_sha=$(A shell "cd $STAGE/$fx && cat \$(ls | LC_ALL=C sort) | sha256sum" | cut -d' ' -f1 | tr -d '\r')   # byte order (r12: a locale sort put 000_mel_valid before 000_mel)
  echo "push $fx files $got_n/$want_n concat sha256 phone=${got_sha:-<none>} mac=$want_sha $([ "$got_sha" = "$want_sha" ] && echo MATCH || echo MISMATCH) in $(( $(now) - t0 ))s"
  [ "$got_n" = "$want_n" ] && [ "$got_sha" = "$want_sha" ]
}

setup_stage() {  # the runner + the 12 .so (sha256 against SHA256SUMS) + the plan's fixture dirs
  local f name want got bad=0 fx t0
  t0=$(now)
  echo "setup start $(ts); data free before: $(avail_bytes) B"
  A shell mkdir -p $STAGE
  for f in "$LIBS"/*.so; do
    name=$(basename "$f")
    want=$(awk -v n="$name" '$2 == n {print $1}' "$LIBS_SUMS")
    [ -n "$want" ] && [ "$(shasum -a 256 "$f" | cut -d' ' -f1)" = "$want" ] || { echo "LIB SHA256 MISMATCH ON THE MAC $name"; bad=1; continue; }
    A push "$f" "$STAGE/$name" > /dev/null 2>&1 || { echo "push failed $name"; bad=1; continue; }
    got=$(dev_sha "$STAGE/$name")
    [ "$got" = "$want" ] && echo "lib $name sha256 MATCH" || { echo "lib $name sha256 phone=${got:-<none>} want=$want MISMATCH"; bad=1; }
  done
  [ "$(ls "$LIBS"/*.so | wc -l | tr -d ' ')" = "$(wc -l < "$LIBS_SUMS" | tr -d ' ')" ] || { echo "lib count differs from SHA256SUMS"; bad=1; }
  want=$(shasum -a 256 "$RUNNER_BIN" | cut -d' ' -f1)
  if [ "$DRY_RUN" != 1 ] && [ "$want" != "$RUNNER_SHA_REF" ]; then echo "runner sha256 $want != the Kev record $RUNNER_SHA_REF"; bad=1; fi
  A push "$RUNNER_BIN" "$STAGE/$RNAME" > /dev/null 2>&1 || { echo "push failed runner"; bad=1; }
  A shell chmod 755 "$STAGE/$RNAME"
  got=$(dev_sha "$STAGE/$RNAME")
  [ "$got" = "$want" ] && echo "runner $RNAME sha256 MATCH ($want)" || { echo "runner sha256 phone=${got:-<none>} mac=$want MISMATCH"; bad=1; }
  for fx in $(printf '%s\n' "${PLAN_LINES[@]}" | awk '{print $5}' | LC_ALL=C sort -u); do
    push_fx "$fx" || bad=1
  done
  echo "device: $(A shell getprop ro.product.model | tr -d '\r') $(A shell getprop ro.build.version.release | tr -d '\r') $(A shell getprop ro.build.display.id | tr -d '\r')"
  A shell "ls -la $STAGE | head -40; df -h /data | tail -1" | tr -d '\r'
  echo "setup end $(ts) in $(( $(now) - t0 ))s; data free $(avail_bytes) B; bad=$bad"
  return $bad
}

later_needs() {  # index -> the graph files and cache dirs that the legs after it use (one per line: g:<file> / c:<dir>)
  local i=$1 j line x
  for ((j = i + 1; j < ${#PLAN_LINES[@]}; j++)); do
    line=${PLAN_LINES[$j]}
    read -r x x x graph x x x x x cache x <<< "$line"
    echo "g:$(graph_file "$graph")"
    case "$cache" in new:*|use:*) echo "c:${cache#*:}" ;; esac
  done | LC_ALL=C sort -u
}

evict() {  # index [keep_cache]: remove the graphs and caches on the phone that no leg after <index> uses; with
  # keep_cache = also every other cache (the space is short: the cached legs that needed them then skip)
  local i=$1 keep=${2:-} need item f
  need=$(later_needs "$i")
  for f in $(A shell "ls $STAGE" | tr -d '\r' | grep '\.tflite$'); do
    echo "$need" | grep -qx "g:$f" || { A shell rm -f "$STAGE/$f"; echo "removed graph $f $(ts)"; }
  done
  for f in $(A shell "ls -d $STAGE/cache_* 2>/dev/null" | tr -d '\r'); do
    item=$(basename "$f")
    if ! echo "$need" | grep -qx "c:$item" || { [ -n "$keep" ] && [ "$item" != "$keep" ]; }; then
      A shell rm -rf "$STAGE/$item"; echo "removed cache $item $(ts)"
    fi
  done
}

ensure_graph() {  # index graph tag cold(0/1) cache dir: the graph on the phone with its sha256 checked, room for its JIT
  # cache; first everything that neither this leg nor a later one uses is removed
  local i=$1 g=$2 cold=$3 cdir=${4:-} file src need free
  file=$(graph_file "$g"); src=$K/out/$file
  evict "$((i - 1))"
  if [ "$(A shell "ls $STAGE/$file 2>/dev/null" | tr -d '\r')" = "$STAGE/$file" ] && [ "$(dev_sha "$STAGE/$file")" = "$(shasum -a 256 "$src" | cut -d' ' -f1)" ]; then
    echo "graph $file already on the phone (sha256 MATCH)"; need=0
  else
    need=$(stat -L -f %z "$src")
  fi
  [ "$cold" = 1 ] && need=$(( need + CACHE_EST ))
  need=$(( need + DATA_MARGIN )); free=$(avail_bytes)
  if [ "${free:-0}" -lt "$need" ]; then
    echo "data free ${free:-?} < $need: also removing the caches only later cached legs need"; evict "$((i - 1))" "${cdir:-none}"; free=$(avail_bytes)
  fi
  [ "${free:-0}" -ge "$need" ] || { echo "data free ${free:-?} < $need for $file"; return 6; }
  if [ "$(A shell "ls $STAGE/$file 2>/dev/null" | tr -d '\r')" != "$STAGE/$file" ]; then
    push_file "$src" "$STAGE/$file" || { A shell rm -f "$STAGE/$file"; echo "graph $file push failed"; return 7; }
  fi
  echo "data free with $file: $(avail_bytes) B"
}

monitor() {  # tag client_pid period guard_kb compile_limit_s t0: samples, the compile watch, the memory guard, run end
  local tag=$1 cp=$2 per=$3 guard=$4 climit=$5 t0=$6 n=0 out ma p every killed=""
  every=$(( 10 / per )); [ $every -lt 1 ] && every=1
  while kill -0 "$cp" 2>/dev/null; do
    out=$(A shell "$SAMPLE_CMD" 2>&1 | tr -d '\r')
    if [ $(( n % every )) = 0 ]; then out="$out
$(A shell "$SLOW_CMD" 2>&1 | tr -d '\r')"; fi
    printf '=== %s n=%d\n%s\n' "$(date '+%T')" "$n" "$out" >> "$R/$tag.samples.txt"
    if [ -z "$killed" ]; then
      p=$(echo "$out" | awk '/^pid / {print $2; exit}')
      ma=$(echo "$out" | awk '/MemAvailable/ {print $2; exit}')
      if [ "$guard" -gt 0 ] && [ -n "$ma" ] && [ "$ma" -lt "$guard" ] && [ -n "$p" ]; then
        killed=guard; echo "$(ts) MemAvailable $ma kB < $guard kB -> kill -9 runner pid $p" | tee -a "$R/$tag.killed.txt"
        A shell kill -9 "$p"
      elif ! grep -q '^compile_end model\[0\]' "$R/$tag.stdout.txt" 2>/dev/null && [ $(( $(now) - t0 )) -ge "$climit" ] && [ -n "$p" ]; then
        killed=compile_timeout; echo "$(ts) no compile_end after ${climit}s -> kill -9 runner pid $p" | tee -a "$R/$tag.killed.txt"
        A shell kill -9 "$p"
      elif [ "$(now)" -ge "$RUN_END" ] && [ -n "$p" ]; then
        killed=run_end; echo "$(ts) run end $(hms "$RUN_END") reached -> kill -9 runner pid $p" | tee -a "$R/$tag.killed.txt"
        A shell kill -9 "$p"
      fi
    fi
    if ! present; then sleep 3; present || { echo "$(ts) device lost during $tag" >> "$R/$tag.killed.txt"; echo lost > "$R/$tag.lost"; return; }; fi
    n=$((n + 1)); sleep "$per"
  done
  [ -n "$killed" ] && echo "$killed" > "$R/$tag.killed_by"
}

LEG_END=0; LAST_UPTIME=0; SCORE_PIDS=""; STOP_ALL=0
run_leg() {  # index tag accel prec graph fx count warm rounds ab cache climit ready extras
  local i=$1 tag=$2 accel=$3 prec=$4 graph=$5 fx=$6 count=$7 warm=$8 rounds=$9 ab=${10} cache=${11} climit=${12} ready=${13} extras=${14:-}
  local rest per guard file cdir cold=0 cache_flag="" prec_flag="" out cmd cp mp rc up0 up1 mark status killed pid ndump nbad csize t0 L OUTN
  rest=$(xget "$extras" rest 30); per=$(xget "$extras" sample 2); guard=$(xget "$extras" guard "$MEM_GUARD_KB")
  file=$(graph_file "$graph"); out=$STAGE/out_$tag
  L=$(jq_get "$(manifest_of "$fx")" L)
  OUTN=$(out_elems "$(manifest_of "$fx")")
  [ -e "$R/$tag.stdout.txt" ] && { echo "SKIP $tag: R/$tag.stdout.txt exists (never overwritten)" | tee -a "$R/s_skipped.txt"; return; }
  [ "$(now)" -lt "$SKIP_AFTER" ] || { echo "SKIP $tag at $(date '+%T') (past skip_after $(hms "$SKIP_AFTER"))" | tee -a "$R/s_skipped.txt"; return; }
  local needs; needs=$(xget "$extras" needs "")
  if [ -n "$needs" ] && [ "$(python3 -I -c "import json,sys; print(json.load(open(sys.argv[1]))['status'])" "$R/$needs.leg.json" 2>/dev/null)" != DONE ]; then
    echo "SKIP $tag: needs $needs DONE (a cache of a compile that did not end is never reused: no JIT retry)" | tee -a "$R/s_skipped.txt"; return
  fi
  case "$cache" in
    new:*) cdir=${cache#new:}; cold=1 ;;
    use:*) cdir=${cache#use:}
           [ -n "$(A shell "ls $STAGE/$cdir 2>/dev/null" | tr -d '\r')" ] || { echo "SKIP $tag: cache $cdir is not on the phone" | tee -a "$R/s_skipped.txt"; return; } ;;
    *) cdir="" ;;
  esac
  ( ensure_graph "$i" "$graph" "$cold" "$cdir" ) >> "$R/s_graphs.txt" 2>&1; rc=$?
  present || hard_stop "device_lost_in_graph_$tag"
  [ $rc = 0 ] || { echo "SKIP $tag: graph $file not ready (rc $rc, s_graphs.txt)" | tee -a "$R/s_skipped.txt"; return; }
  [ "$cold" = 1 ] && A shell rm -rf "$STAGE/$cdir"
  [ -n "$cdir" ] && cache_flag="--cache-dir $STAGE/$cdir"
  case "$accel" in gpu) prec_flag="--gpu-precision $prec" ;; esac
  ready_gate "$tag" "$LEG_END" "$rest" "$ready"
  [ "${READY_SKIP:-0}" = 1 ] && { echo "SKIP $tag at $(date '+%T'): $READY_NOTE" | tee -a "$R/s_skipped.txt"; return; }
  [ "$(now)" -lt "$SKIP_AFTER" ] || { echo "SKIP $tag at $(date '+%T') (past skip_after after the ready gate)" | tee -a "$R/s_skipped.txt"; return; }
  local left=$(( RUN_END - $(now) - 45 )) cl=$climit
  [ "$left" -ge 30 ] || { echo "SKIP $tag at $(date '+%T'): ${left}s of compile time left before run_end" | tee -a "$R/s_skipped.txt"; return; }
  [ "$left" -lt "$cl" ] && cl=$left
  echo "$READY_NOTE" > "$R/$tag.ready.txt"
  state > "$R/$tag.state_before.txt" 2>&1
  ps_time "$R/$tag.ps_before.txt"
  up0=$(uptime_s)
  mark=$(A shell "date '+%m-%d %H:%M:%S.000'" | tr -d '\r')   # quoted for the phone's shell (Kev round 11)
  cmd="cd $STAGE && rm -rf $out && LD_LIBRARY_PATH=$STAGE ADSP_LIBRARY_PATH='$LIBPATH_ADSP' ./$RNAME --accel $accel $prec_flag --burst 1 --libdir $STAGE $cache_flag --fixtures $STAGE/$fx --count $count --warmup $warm --rounds $rounds --ab-fixture $ab --dump full --out $out --model $STAGE/$file"
  echo "$cmd" > "$R/$tag.cmd.txt"
  echo "=== leg $tag start $(date '+%T') device mark '$mark' uptime $up0 compile watch ${cl}s guard ${guard} kB ready: $READY_NOTE"
  t0=$(now)
  "$ADB" -s "$SERIAL" shell "$cmd" > "$R/$tag.stdout.txt" 2>&1 < /dev/null &
  cp=$!
  monitor "$tag" "$cp" "$per" "$guard" "$cl" "$t0" &
  mp=$!
  wait "$cp"; rc=$?
  wait "$mp" 2>/dev/null
  LEG_END=$(now)
  [ -f "$R/$tag.lost" ] && hard_stop "device_lost_in_$tag"
  present || hard_stop "device_lost_after_$tag"
  pid=$(runner_pid)
  if [ -n "$pid" ]; then echo "$(ts) runner pid $pid still alive after rc=$rc -> kill -9" | tee -a "$R/$tag.killed.txt"; A shell kill -9 "$pid"; sleep 2; fi
  up1=$(uptime_s)
  if [ -n "$up1" ] && [ -n "$up0" ] && [ "$up1" -lt "$up0" ]; then
    echo "uptime ${up1}s < ${up0}s: the phone rebooted during $tag" | tee "$R/$tag.REBOOT"; hard_stop "phone_rebooted_during_$tag"
  fi
  state > "$R/$tag.state_after.txt" 2>&1
  ps_time "$R/$tag.ps_after.txt"
  others_time "$R/$tag.ps_before.txt" "$R/$tag.ps_after.txt" "$R/$tag.others_time.txt"
  A logcat -b all -d -T "$mark" > "$R/$tag.logcat_all.txt" 2>&1
  local rpid
  rpid=$(sed -nE 's/^pid=([0-9]+) .*/\1/p' "$R/$tag.stdout.txt" | head -1)
  if [ -n "$rpid" ]; then
    grep -E "^[0-9-]+ [0-9:.]+ +$rpid +[0-9]+ " "$R/$tag.logcat_all.txt" | grep -iE "Replacing|Partitioned|DispatchDelegate|QNN|Qnn|qnn|compiler plugin|cached model|reserializ|Unsupported|not supported|failed|error|fatal|abort|scudo" | head -400 > "$R/$tag.delegate.txt"
  fi
  grep -iE "lowmemorykiller|lmkd" "$R/$tag.logcat_all.txt" | head -80 > "$R/$tag.lmkd_lines.txt"
  killed=$(cat "$R/$tag.killed_by" 2>/dev/null)
  mkdir -p "$R/$tag.dump"
  A pull "$out/model0/." "$R/$tag.dump/" > /dev/null 2>&1
  ndump=$(ls "$R/$tag.dump" | wc -l | tr -d ' ')
  nbad=$(find "$R/$tag.dump" -type f -name '*.f32' ! -size "$(( OUTN * 4 ))c" | wc -l | tr -d ' ')
  A shell rm -rf "$out"
  csize=""; [ -n "$cdir" ] && csize=$(A shell "du -sk $STAGE/$cdir 2>/dev/null" | tr -d '\r' | awk '{print $1}')
  if [ -n "$killed" ]; then
    case "$killed" in guard) status=OOM_GUARD ;; compile_timeout) status=COMPILE_TIMEOUT ;; run_end) status=LIMIT_STOP ;; *) status=KILLED ;; esac
  elif grep -q '^done wall_ms=' "$R/$tag.stdout.txt" && [ "$rc" = 0 ]; then
    status=DONE
  else
    status=RUNNER_FAILED
  fi
  if [ -n "$rpid" ] && grep -E "$RNAME|\($rpid\)|pid $rpid[^0-9]" "$R/$tag.lmkd_lines.txt" > "$R/$tag.lmkd_runner.txt" && [ -s "$R/$tag.lmkd_runner.txt" ]; then
    status=LMKD_KILLED
  fi
  python3 -I - "$R/$tag.leg.json" <<EOF
import json, sys
doc = {"tag": "$tag", "index": $i, "accel": "$accel", "precision": "$prec", "graph": "$graph", "graph_file": "out/$file",
       "fx": "$fx", "manifest": "$(krel "$(manifest_of "$fx")")", "mac_dir": "$(krel "$FXROOT")/mac/$graph/model0", "L": $L,
       "out_elems": $OUTN, "extras": "$extras",
       "count": $count, "warmup": $warm, "rounds": $rounds, "ab_fixture": "$ab", "cache": "$cache", "cache_dir": "$cdir",
       "compile_watch_s": $cl, "compile_limit_plan_s": $climit, "guard_kb": $guard, "ready": "$ready",
       "ready_note": open("$R/$tag.ready.txt").read().strip(), "status": "$status", "rc": $rc, "killed_by": "$killed" or None,
       "start": "$(date -r "$t0" '+%F %T')", "end": "$(date -r "$LEG_END" '+%F %T')", "wall_s": $(( LEG_END - t0 )),
       "device_mark": "$mark", "uptime_before": int("${up0:-0}" or 0), "uptime_after": int("${up1:-0}" or 0),
       "dumps": $ndump, "dumps_bad_size": $nbad, "cache_kb_after": int("${csize:-0}" or 0) if "$cdir" else None,
       "data_free_after_b": int("$(avail_bytes)" or 0), "memory": "$(memory_line "$R/$tag.samples.txt" 2>/dev/null)",
       "others_cpu_time": open("$R/$tag.others_time.txt").read().split("\n")[:20],
       "lmkd_lines": sum(1 for _ in open("$R/$tag.lmkd_lines.txt")), "runner_pid": int("${rpid:-0}" or 0),
       "cmd": open("$R/$tag.cmd.txt").read().strip(), "dry_run": "$DRY_RUN" == "1"}
open(sys.argv[1], "w").write(json.dumps(doc, indent=1) + "\n")
EOF
  echo "=== leg $tag status $status rc $rc killed ${killed:-none} dumps $ndump (bad size $nbad) cache ${csize:-none} kB end $(date '+%T') uptime $up0 -> $up1"
  grep -E "^(model\[0\]=|parity_summary|summary|FATAL|ERROR|FIXTURE_ERROR|done)" "$R/$tag.stdout.txt" | tail -6
  grep -E "Partitioned subgraph|Replacing [0-9]+ out of" "$R/$tag.stdout.txt" | head -4
  [ -s "$R/$tag.others_time.txt" ] && echo "other processes' CPU TIME grew during $tag: $(tr '\n' ' ' < "$R/$tag.others_time.txt")"
  [ -s "$R/$tag.lmkd_lines.txt" ] && echo "lmkd lines during $tag: $(wc -l < "$R/$tag.lmkd_lines.txt" | tr -d ' ') (R/$tag.lmkd_lines.txt)"
  ( cd "$K" && "$PY" "$SCORE" leg --tag "$tag" --round "$ROUND" --round-dir "$R" --results "$RESULTS" > "$R/$tag.score.txt" 2>&1 ) < /dev/null &
  SCORE_PIDS="$SCORE_PIDS $!"
  if [ "$status" = LMKD_KILLED ]; then hard_stop "lmkd_killed_the_runner_in_$tag"; fi
  ( evict "$i" ) >> "$R/s_graphs.txt" 2>&1
}

cleanup_all() {
  {
    local p
    p=$(runner_pid); [ -n "$p" ] && { echo "runner pid $p alive at the cleanup -> kill -9"; A shell kill -9 "$p"; sleep 1; }
    A shell rm -rf $STAGE
    echo "stage left: $(A shell ls -d $STAGE 2>&1 | tr -d '\r')"
    echo "runner left: '$(runner_pid)'"
    echo "data_free: $(A shell df -h /data | tail -1 | tr -d '\r')"
    state
  } > "$R/s9_cleanup.txt" 2>&1
  cat "$R/s9_cleanup.txt"
  present || { echo "the phone left adb during the cleanup $(ts)" > "$R/CHAIN_STOPPED"; return 9; }
  CLEANED=1
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
wait_scores() { local p; for p in $SCORE_PIDS; do wait "$p" 2>/dev/null; done; SCORE_PIDS=""; }
on_exit() {
  local rc=$?
  [ "$HELD" = 1 ] || exit $rc
  if [ "$CLEANED" = 0 ] && present; then echo "exit $rc before the cleanup: cleaning now $(ts)"; cleanup_all; fi
  release_hold
  wait_scores
  echo "chain exit $rc $(ts)"
  exit $rc
}

read_plan() {
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

check_inputs() {
  mkdir -p "$R"
  read_plan
  local line tag accel prec graph fx count warm rounds ab cache climit ready extras ok=0 f man n
  for line in "${PLAN_LINES[@]}"; do
    read -r tag accel prec graph fx count warm rounds ab cache climit ready extras <<< "$line"
    case "$accel" in npu+cpu|npu|gpu) ;; *) echo "$tag: accel $accel"; ok=1 ;; esac
    case "$accel" in gpu) case "$prec" in fp32|fp16acc32|fp16|default) ;; *) echo "$tag: precision $prec"; ok=1 ;; esac ;;
                     *) [ "$prec" = - ] || { echo "$tag: precision $prec for $accel"; ok=1; } ;; esac
    case "$ready" in gate|clean|uncapped) ;; *) echo "$tag: ready $ready"; ok=1 ;; esac
    case "$cache" in -|new:cache_*|use:cache_*) ;; *) echo "$tag: cache $cache (- | new:cache_<x> | use:cache_<x>)"; ok=1 ;; esac
    f=$K/out/$(graph_file "$graph"); [ -s "$f" ] || { echo "$tag: missing graph $f"; ok=1; }
    man=$(manifest_of "$fx"); [ -s "$man" ] || { echo "$tag: missing $man"; ok=1; continue; }
    n=$(jq_get "$man" count); [ "$count" -le "$n" ] || { echo "$tag: count $count > the $n fixtures of $fx"; ok=1; }
    [ -f "$K/$(jq_get "$man" dir)/${ab}_$(first_input "$man").f32" ] || { echo "$tag: no fixture $ab in $fx"; ok=1; }
    printf '%-26s %-8s %-5s %-18s %-8s %4s %3s %3s %4s %-24s %4s %-9s %s\n' "$tag" "$accel" "$prec" "$graph" "$fx" "$count" "$warm" "$rounds" "$ab" "$cache" "$climit" "$ready" "$extras"
  done
  echo "inputs (sha256):"
  for graph in $(printf '%s\n' "${PLAN_LINES[@]}" | awk '{print $4}' | LC_ALL=C sort -u); do
    f=$K/out/$(graph_file "$graph"); echo "  $(shasum -a 256 "$f" | cut -c1-16)… $(stat -L -f %z "$f") B out/$(graph_file "$graph")"
  done
  echo "  runner $(shasum -a 256 "$RUNNER_BIN" | cut -c1-16)… $(stat -L -f %z "$RUNNER_BIN") B $RUNNER_BIN"
  [ "$DRY_RUN" = 1 ] || [ "$(shasum -a 256 "$RUNNER_BIN" | cut -d' ' -f1)" = "$RUNNER_SHA_REF" ] || { echo "runner sha256 differs from the Kev record"; ok=1; }
  ( cd "$LIBS" && shasum -a 256 -c "$LIBS_SUMS" ) > "$R/s0_libs_check.txt" 2>&1 || { echo "libs sha256 check failed (R/s0_libs_check.txt)"; ok=1; }
  echo "  libs: $(grep -c ': OK$' "$R/s0_libs_check.txt")/$(wc -l < "$LIBS_SUMS" | tr -d ' ') OK against $LIBS_SUMS"
  for fx in $(printf '%s\n' "${PLAN_LINES[@]}" | awk '{print $5}' | LC_ALL=C sort -u); do
    man=$(manifest_of "$fx")
    echo "  $fx: $(jq_get "$man" files) files, $(jq_get "$man" bytes) B, concat sha256 $(jq_get "$man" concat_sha256_bytesorted | cut -c1-16)…"
  done
  return $ok
}

take_hold() {
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

cleanup_cmd() {
  local cp kp
  cp=$(cat "$R/chain.pid" 2>/dev/null); kp=$(cat "$R/keeper.pid" 2>/dev/null)
  if [ -n "$cp" ] && kill -0 "$cp" 2>/dev/null && ps -o command= -p "$cp" | grep -q s26_npu_chain; then
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
  rm -f "$R/CHAIN_DONE" "$R/CHAIN_STOPPED" "$R/RELEASE_HOLD"
  echo "$$" > "$R/chain.pid"
  echo "chain $ROUND pid $$ $(ts); this file sha256 $(shasum -a 256 "$0" | cut -c1-16); plan $PLAN sha256 $(shasum -a 256 "$PLAN" | cut -c1-16)"
  check_inputs > "$R/s0_inputs.txt" 2>&1 || { cat "$R/s0_inputs.txt"; echo "inputs check failed: no hold taken"; exit 5; }
  cat "$R/s0_inputs.txt"
  read_plan
  [ "$(printf '%s\n' "${PLAN_LINES[@]}" | awk '{print $1}' | LC_ALL=C sort -u | wc -l | tr -d ' ')" = "${#PLAN_LINES[@]}" ] || { echo "duplicate leg tags"; exit 5; }
  mkdir -p "$RESULTS"
  take_hold
  LEG_END=$(now)
  present || hard_stop "device_absent_at_start"
  if ! transfer_check; then
    echo "OTHER SESSION'S TRANSFER / PROCESS ON THE PHONE -> no leg, release $(ts)"
    echo "other_session_job $(ts)" > "$R/CHAIN_STOPPED"; CLEANED=1; exit 5
  fi
  LAST_UPTIME=$(uptime_s); echo "phone uptime at the hold: ${LAST_UPTIME}s"
  state > "$R/s0_state_initial.txt" 2>&1; cat "$R/s0_state_initial.txt"
  setup_stage > "$R/s0_setup.txt" 2>&1; local ok=$?
  cat "$R/s0_setup.txt"
  present || hard_stop "device_lost_in_setup"
  [ $ok = 0 ] || { echo "setup failed $(ts)"; echo "setup_failed $(ts)" > "$R/CHAIN_STOPPED"; exit 5; }
  local i line tag accel prec graph fx count warm rounds ab cache climit ready extras
  for ((i = 0; i < ${#PLAN_LINES[@]}; i++)); do
    line=${PLAN_LINES[$i]}
    read -r tag accel prec graph fx count warm rounds ab cache climit ready extras <<< "$line"
    [ "$STOP_ALL" = 0 ] || { echo "SKIP $tag (stopped)" | tee -a "$R/s_skipped.txt"; continue; }
    run_leg "$i" "$tag" "$accel" "$prec" "$graph" "$fx" "$count" "$warm" "$rounds" "$ab" "$cache" "$climit" "$ready" "$extras"
  done
  cleanup_all || hard_stop "device_lost_in_cleanup"
  echo "chain $ROUND legs done $(ts)"
  date '+%F %T' > "$R/CHAIN_DONE"
  release_hold
  wait_scores
  echo "host scorers done $(ts)"
}

case "$CMD" in
  check) check_inputs ;;
  run) run_chain ;;
  cleanup) mkdir -p "$R"; cleanup_cmd ;;
  *) echo "usage: $0 check | run | cleanup <round>" >&2; exit 2 ;;
esac
