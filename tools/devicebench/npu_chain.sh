#!/bin/bash
# npu_chain.sh: the device chain for a native C-API runner on the phone (the NPU through the vendor's compiler plugin
# and dispatch library, and the same runner on the GPU for an A/B), shared by every lane. lib.sh holds the hold, the
# clock, the ready gate and the transfer check; a lane keeps its settings (NPU_* in lane.env), plans, fixtures, graphs,
# the runner binary with its runtime libraries (staged on the Mac, never published) and its scorer (lane_npu_score).
#   bash npu_chain.sh check <round> <lane.env>      the plan and its inputs with sha256 (graphs, runner, libs, fixtures); no adb
#   bash npu_chain.sh run <round> <lane.env>        the phone: hold, setup, legs, cleanup, release
#   bash npu_chain.sh cleanup <round> <lane.env>    by hand after a hard stop: the dead chain's live hold, or a new hold
# Round dir R = <device dir>/<round>/ (ROUND_DIR overrides); the plan = PLAN, else the first of R/plan_<round>.tsv,
# R/plan.tsv, <device dir>/plan_<round>.tsv. One leg per line (# comments; whitespace-separated; extras key=value):
#   tag  accel  precision  graph  fx  count  warmup  rounds  ab  cache  compile_s  ready  [extras]
#   accel npu+cpu | npu | gpu; precision - (npu) | fp32 | fp16acc32 | fp16 | default (gpu); graph = a name that the
#   lane's lane_graph_file maps to a file of GRAPH_DIR (default: <graph>.tflite, or <graph> when it ends in .tflite);
#   fx = a fixture dir whose manifest FX_ROOT/<fx>.manifest.json names it (dir, files, bytes, concat_sha256_bytesorted,
#   count, L, out_elems, order.inputs_subgraph_order); count = fixtures of the parity pass (000..count-1); warmup / rounds
#   / ab = the latency phase on fixture ab (rounds 0 = none); cache = - | new:<dir> (a JIT cold run: <dir> removed first)
#   | use:<dir> (a cached run: skipped unless <dir> is on the phone); compile_s = the compile watch (no `compile_end` line
#   by then -> the runner is killed on the phone, COMPILE_TIMEOUT, the next leg; never retried); ready = gate | clean |
#   uncapped (lib.sh ready_gate; an uncapped leg not uncapped by the limit runs warm).
#   extras: rest=<s> (since the previous leg, default 30), needs=<tag> (run only when that leg is DONE: a cached leg never
#   reuses the cache of a compile that did not end), guard=<kB> (MemAvailable below it -> the runner is killed,
#   OOM_GUARD; default MEM_GUARD_KB), sample=<s> (default 2); other keys (e.g. pair=<leg> for the scorer) ride along.
# Phone layout: NPU_STAGE/ = the runner + the runtime .so files (flat: --libdir, LD_LIBRARY_PATH and the head of
# ADSP_LIBRARY_PATH all point there; a compiler plugin's DT_NEEDED libraries resolve only through LD_LIBRARY_PATH), the
# fixture dirs, the graph files, the JIT cache dirs (one per cold leg, kept only while a later leg uses them), out_<tag>/
# (the dumps, pulled and removed after each leg). Nothing else on the phone is touched; the cleanup removes NPU_STAGE.
# Per leg: the time checks -> the graph on the phone (bytes polled, sha256sum; graphs and caches no later leg needs are
# removed first; /data free >= the graph + CACHE_EST per JIT cache + DATA_MARGIN) -> the ready gate -> the state -> a
# device clock mark -> the runner (adb shell; its stdout streams to R/<tag>.stdout.txt, its stderr goes to
# NPU_STAGE/<tag>.stderr.txt on the phone: one stream per file, so a runner line never lands behind a fragment of a
# LiteRT log line; the adb client's and the phone shell's own messages, such as `Killed`, still reach stdout.txt) with
# the 2 s sampler, the compile watch, the memory guard and the run-end watch -> any runner left on the phone killed by
# `pidof` (a host-side stop kills only the adb client) -> /proc/uptime (went down = the phone rebooted: hard stop) -> the
# state -> logcat -b all since the mark (R/<tag>.logcat_all.txt) -> the stderr file pulled to R/<tag>.stderr.txt (bytes
# checked) and removed -> R/<tag>.delegate.txt (the runner's own delegation lines from stdout.txt and stderr.txt, then
# the runner pid's key lines of the logcat; an lmkd line naming the runner = hard stop) -> the dumps pulled (count and
# bytes checked) -> out_<tag>/ removed -> the cache size -> R/<tag>.leg.json -> lane_npu_score in the background -> what
# no later leg needs removed.
# The runner's contract (a C-API runner built for the phone): ./<NPU_RUNNER_NAME> --accel <accel> [--gpu-precision <p>]
# --burst 1 --libdir <stage> [--cache-dir <dir>] --fixtures <dir> --count <n> --warmup <n> --rounds <n> --ab-fixture <id>
# --dump full --out <dir> --model <file>; it prints `pid=<pid> ...` first, `compile_end model[0]` after the compile,
# `done wall_ms=` at the end, and dumps <out>/model0/<id>.f32 (out_elems float32 each).
# Clock (minutes after the hold): no leg starts after SKIP_MIN (25), every runner is stopped by RUN_END_MIN (28) (a
# compile watch never reaches past it), the keeper releases at END_MIN (30) at the latest. Exit 9 = the phone left adb
# or rebooted, or lmkd killed the runner (hard stop: the phone is cleaned if it is there, the hold is released).
# Settings (lane.env; README.md lists them): LANE SERIAL HOLD NPU_RUNNER_NAME (at most 15 characters: the phone's
# process name) NPU_STAGE (required); NPU_RUNNER NPU_RUNNER_SHA NPU_LIBS NPU_LIBS_SUMS NPU_ADSP_PATH FX_ROOT GRAPH_DIR
# RESULTS_DIR PY ADB CACHE_EST and the hooks lane_graph_file / lane_npu_score. DRY_RUN=1 = the fake phone
# (dryrun_npu.sh): ADB must be fake_adb.py and the hold file, ROUND_DIR and RESULTS_DIR must lie under this folder's
# cache/; the runner may be the lane's stand-in (its recorded sha256 is not checked).
set -u
CHAIN_NAME=npu_chain.sh
# shellcheck source=lib.sh
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
CMD=${1:-}; ROUND=${2:-}
[ -n "$ROUND" ] && [ -n "${3:-${LANE_ENV:-}}" ] || { echo "usage: $0 check | run | cleanup <round> <lane.env>" >&2; exit 2; }
# the NPU chain's own defaults (the environment and lane.env win)
REST_MAX=${REST_MAX:-180}; REST_MAX_T=${REST_MAX_T:-300}; POLL=${POLL:-10}; MEM_GUARD_KB=${MEM_GUARD_KB:-1000000}
XFER_MAX_S=${XFER_MAX_S:-300}
bench_load "${3:-$LANE_ENV}"
for v in NPU_RUNNER_NAME NPU_STAGE; do
  [ -n "${!v:-}" ] || { echo "$LANE_ENV: $v is not set (npu_chain.sh needs it)" >&2; exit 2; }
done
[ ${#NPU_RUNNER_NAME} -le 15 ] || { echo "NPU_RUNNER_NAME $NPU_RUNNER_NAME is longer than 15 characters (the phone's comm, pidof)" >&2; exit 2; }
PLAN=$(plan_file)
RNAME=$NPU_RUNNER_NAME
STAGE=$NPU_STAGE; STAGE_DIR=$NPU_STAGE
RUNNER_BIN=${NPU_RUNNER:-$DEVICE_DIR/$ROUND/stage/$RNAME}
RUNNER_SHA_REF=${NPU_RUNNER_SHA:-}
LIBS=${NPU_LIBS:-$DEVICE_DIR/$ROUND/stage/libs}
LIBS_SUMS=${NPU_LIBS_SUMS:-$LIBS/SHA256SUMS}
FXROOT=${FX_ROOT:-$DEVICE_DIR/$ROUND}
CACHE_EST=${CACHE_EST:-1400000000}
LIBPATH_ADSP=${NPU_ADSP_PATH:-$STAGE;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp}
OWN_NAME=$RNAME
OWN_PROC_RE='[A-Za-z0-9_.]*runner'
UNCAPPED_FALLBACK=warm
dry_run_guard

graph_file() {  # plan graph name -> the file name in GRAPH_DIR (the lane's lane_graph_file, else <name>[.tflite])
  if declare -F lane_graph_file > /dev/null; then lane_graph_file "$1"; return; fi
  case "$1" in *.tflite) echo "$1" ;; *) echo "$1.tflite" ;; esac
}
runner_pid() { A shell pidof "$RNAME" 2>/dev/null | tr -d '\r'; }
stage_sha() { A shell sha256sum "$1" 2>/dev/null | cut -d' ' -f1 | tr -d '\r'; }
manifest_of() { echo "$FXROOT/$1.manifest.json"; }
jq_get() { python3 -I -c "import json,sys; d=json.load(open(sys.argv[1])); print(d[sys.argv[2]])" "$1" "$2"; }
out_elems() { python3 -I -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get('out_elems') or d['L'])" "$1"; }
first_input() { python3 -I -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['order']['inputs_subgraph_order'][0]['name'])" "$1"; }
krel() { case "$1" in "$LANE_DIR"/*) echo "${1#"$LANE_DIR"/}" ;; *) echo "$1" ;; esac; }

# the sampler's commands (lib.sh sample_cmd / slow_cmd), set once the GPU's driver is read after the hold
SAMPLE_CMD=""; SLOW_CMD=""

push_file() {  # src dst: adb push, then the byte count polled (the client can return early) and the sha256
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
  h=$(shasum -a 256 "$src" | cut -d' ' -f1); d=$(stage_sha "$dst")
  echo "push $(basename "$dst") rc=$rc bytes ${got:-<none>}/$want sha256 phone=${d:-<none>} mac=$h $([ "$h" = "$d" ] && echo MATCH || echo MISMATCH) in $(( $(now) - t0 ))s"
  [ "$rc" = 0 ] && [ "$got" = "$want" ] && [ "$h" = "$d" ]
}

push_fx() {  # fx: the fixture dir -> STAGE/<fx>, then the file count and the concatenated sha256 (byte-sorted names)
  local fx=$1 man src want_n want_sha got_n got_sha t0
  man=$(manifest_of "$fx"); src=$LANE_DIR/$(jq_get "$man" dir)
  want_n=$(jq_get "$man" files); want_sha=$(jq_get "$man" concat_sha256_bytesorted)
  t0=$(now)
  A shell mkdir -p "$STAGE/$fx"
  A push "$src/." "$STAGE/$fx/" > /dev/null 2>&1
  present || hard_stop "device_lost_in_push_$fx"
  got_n=$(A shell "ls $STAGE/$fx | wc -l" | tr -d '\r ')
  # byte order: a locale sort puts 000_mel_valid before 000_mel
  got_sha=$(A shell "cd $STAGE/$fx && cat \$(ls | LC_ALL=C sort) | sha256sum" | cut -d' ' -f1 | tr -d '\r')
  echo "push $fx files $got_n/$want_n concat sha256 phone=${got_sha:-<none>} mac=$want_sha $([ "$got_sha" = "$want_sha" ] && echo MATCH || echo MISMATCH) in $(( $(now) - t0 ))s"
  [ "$got_n" = "$want_n" ] && [ "$got_sha" = "$want_sha" ]
}

setup_stage() {  # the runner + the runtime .so files (sha256 against NPU_LIBS_SUMS) + the plan's fixture dirs
  local f name want got bad=0 fx t0
  t0=$(now)
  echo "setup start $(ts); data free before: $(avail_bytes) B"
  A shell mkdir -p "$STAGE"
  for f in "$LIBS"/*.so; do
    name=$(basename "$f")
    want=$(awk -v n="$name" '$2 == n || $2 == "*" n {print $1}' "$LIBS_SUMS")
    [ -n "$want" ] && [ "$(shasum -a 256 "$f" | cut -d' ' -f1)" = "$want" ] || { echo "LIB SHA256 MISMATCH ON THE MAC $name"; bad=1; continue; }
    A push "$f" "$STAGE/$name" > /dev/null 2>&1 || { echo "push failed $name"; bad=1; continue; }
    got=$(stage_sha "$STAGE/$name")
    [ "$got" = "$want" ] && echo "lib $name sha256 MATCH" || { echo "lib $name sha256 phone=${got:-<none>} want=$want MISMATCH"; bad=1; }
  done
  [ "$(ls "$LIBS"/*.so | wc -l | tr -d ' ')" = "$(grep -c . "$LIBS_SUMS" | tr -d ' ')" ] || { echo "lib count differs from $LIBS_SUMS"; bad=1; }
  want=$(shasum -a 256 "$RUNNER_BIN" | cut -d' ' -f1)
  if [ "$DRY_RUN" != 1 ] && [ -n "$RUNNER_SHA_REF" ] && [ "$want" != "$RUNNER_SHA_REF" ]; then echo "runner sha256 $want != the recorded $RUNNER_SHA_REF"; bad=1; fi
  A push "$RUNNER_BIN" "$STAGE/$RNAME" > /dev/null 2>&1 || { echo "push failed runner"; bad=1; }
  A shell chmod 755 "$STAGE/$RNAME"
  got=$(stage_sha "$STAGE/$RNAME")
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
  local i=$1 j line x graph cache
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

stage_graph() {  # index graph cold(0/1) cache_dir: the graph on the phone with its sha256 checked, room for its JIT
  # cache; first everything that neither this leg nor a later one uses is removed
  local i=$1 g=$2 cold=$3 cdir=${4:-} file src need free
  file=$(graph_file "$g"); src=$GRAPH_DIR/$file
  evict "$((i - 1))"
  if [ "$(A shell "ls $STAGE/$file 2>/dev/null" | tr -d '\r')" = "$STAGE/$file" ] && [ "$(stage_sha "$STAGE/$file")" = "$(shasum -a 256 "$src" | cut -d' ' -f1)" ]; then
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

runner_monitor() {  # tag client_pid period guard_kb compile_limit_s t0: samples, the compile watch, the memory guard, run end
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

STDERR_BYTES_PHONE=""; STDERR_BYTES=""
pull_stderr() {  # tag: the runner's stderr file on the phone -> R/<tag>.stderr.txt, its byte count checked against the
  # phone's (a second pull when they differ), then removed on the phone; STDERR_BYTES_PHONE / STDERR_BYTES for leg.json
  local tag=$1 src=$STAGE/$1.stderr.txt n=0
  STDERR_BYTES_PHONE=$(A shell stat -c %s "$src" 2>/dev/null | tr -d '\r'); STDERR_BYTES=""
  case "$STDERR_BYTES_PHONE" in ''|*[!0-9]*) echo "stderr $tag: no $src on the phone"; STDERR_BYTES_PHONE=""; return 1 ;; esac
  while [ $n -lt 2 ] && [ "$STDERR_BYTES" != "$STDERR_BYTES_PHONE" ]; do
    A pull "$src" "$R/$tag.stderr.txt" > /dev/null 2>&1
    STDERR_BYTES=$(stat -f %z "$R/$tag.stderr.txt" 2>/dev/null); n=$((n + 1))
  done
  A shell rm -f "$src"
  echo "stderr $tag.stderr.txt bytes phone=$STDERR_BYTES_PHONE pulled=${STDERR_BYTES:-<none>} $([ "$STDERR_BYTES" = "$STDERR_BYTES_PHONE" ] && echo MATCH || echo CHECK)"
  [ "$STDERR_BYTES" = "$STDERR_BYTES_PHONE" ]
}

LEG_END=0; LAST_UPTIME=0; STOP_ALL=0
run_leg() {  # index tag accel prec graph fx count warm rounds ab cache climit ready extras
  local i=$1 tag=$2 accel=$3 prec=$4 graph=$5 fx=$6 count=$7 warm=$8 rounds=$9 ab=${10} cache=${11} climit=${12} ready=${13} extras=${14:-}
  local rest per guard file cdir cold=0 cache_flag="" prec_flag="" out cmd cp mp rc up0 up1 mark status killed pid ndump nbad csize t0 L OUTN
  local needs left cl rpid
  rest=$(xget "$extras" rest 30); per=$(xget "$extras" sample 2); guard=$(xget "$extras" guard "$MEM_GUARD_KB")
  file=$(graph_file "$graph"); out=$STAGE/out_$tag
  L=$(jq_get "$(manifest_of "$fx")" L)
  OUTN=$(out_elems "$(manifest_of "$fx")")
  { [ -e "$R/$tag.stdout.txt" ] || [ -e "$R/$tag.stderr.txt" ]; } && { echo "SKIP $tag: R/$tag.stdout.txt or .stderr.txt exists (never overwritten)" | tee -a "$R/s_skipped.txt"; return; }
  [ "$(now)" -lt "$SKIP_AFTER" ] || { echo "SKIP $tag at $(date '+%T') (past skip_after $(hms "$SKIP_AFTER"))" | tee -a "$R/s_skipped.txt"; return; }
  needs=$(xget "$extras" needs "")
  if [ -n "$needs" ] && [ "$(python3 -I -c "import json,sys; print(json.load(open(sys.argv[1]))['status'])" "$R/$needs.leg.json" 2>/dev/null)" != DONE ]; then
    echo "SKIP $tag: needs $needs DONE (a cache of a compile that did not end is never reused: no JIT retry)" | tee -a "$R/s_skipped.txt"; return
  fi
  case "$cache" in
    new:*) cdir=${cache#new:}; cold=1 ;;
    use:*) cdir=${cache#use:}
           [ -n "$(A shell "ls $STAGE/$cdir 2>/dev/null" | tr -d '\r')" ] || { echo "SKIP $tag: cache $cdir is not on the phone" | tee -a "$R/s_skipped.txt"; return; } ;;
    *) cdir="" ;;
  esac
  ( stage_graph "$i" "$graph" "$cold" "$cdir" ) >> "$R/s_graphs.txt" 2>&1; rc=$?
  present || hard_stop "device_lost_in_graph_$tag"
  [ $rc = 0 ] || { echo "SKIP $tag: graph $file not ready (rc $rc, s_graphs.txt)" | tee -a "$R/s_skipped.txt"; return; }
  [ "$cold" = 1 ] && A shell rm -rf "$STAGE/$cdir"
  [ -n "$cdir" ] && cache_flag="--cache-dir $STAGE/$cdir"
  case "$accel" in gpu) prec_flag="--gpu-precision $prec" ;; esac
  ready_gate "$tag" "$LEG_END" "$rest" "$ready"
  [ "${READY_SKIP:-0}" = 1 ] && { echo "SKIP $tag at $(date '+%T'): $READY_NOTE" | tee -a "$R/s_skipped.txt"; return; }
  [ "$(now)" -lt "$SKIP_AFTER" ] || { echo "SKIP $tag at $(date '+%T') (past skip_after after the ready gate)" | tee -a "$R/s_skipped.txt"; return; }
  left=$(( RUN_END - $(now) - 45 )); cl=$climit
  [ "$left" -ge 30 ] || { echo "SKIP $tag at $(date '+%T'): ${left}s of compile time left before run_end" | tee -a "$R/s_skipped.txt"; return; }
  [ "$left" -lt "$cl" ] && cl=$left
  echo "$READY_NOTE" > "$R/$tag.ready.txt"
  phone_state > "$R/$tag.state_before.txt" 2>&1
  ps_time "$R/$tag.ps_before.txt"
  up0=$(uptime_s)
  mark=$(A shell "date '+%m-%d %H:%M:%S.000'" | tr -d '\r')   # quoted for the phone's shell
  cmd="cd $STAGE && rm -rf $out && LD_LIBRARY_PATH=$STAGE ADSP_LIBRARY_PATH='$LIBPATH_ADSP' ./$RNAME --accel $accel $prec_flag --burst 1 --libdir $STAGE $cache_flag --fixtures $STAGE/$fx --count $count --warmup $warm --rounds $rounds --ab-fixture $ab --dump full --out $out --model $STAGE/$file 2>$STAGE/$tag.stderr.txt"
  echo "$cmd" > "$R/$tag.cmd.txt"
  echo "=== leg $tag start $(date '+%T') device mark '$mark' uptime $up0 compile watch ${cl}s guard ${guard} kB ready: $READY_NOTE"
  t0=$(now)
  "$ADB" -s "$SERIAL" shell "$cmd" > "$R/$tag.stdout.txt" 2>&1 < /dev/null &
  cp=$!
  runner_monitor "$tag" "$cp" "$per" "$guard" "$cl" "$t0" &
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
  phone_state > "$R/$tag.state_after.txt" 2>&1
  ps_time "$R/$tag.ps_after.txt"
  others_time "$R/$tag.ps_before.txt" "$R/$tag.ps_after.txt" "$R/$tag.others_time.txt"
  A logcat -b all -d -T "$mark" > "$R/$tag.logcat_all.txt" 2>&1
  pull_stderr "$tag"
  rpid=$(sed -nE 's/^pid=([0-9]+) .*/\1/p' "$R/$tag.stdout.txt" | head -1)
  # the runner's own delegation lines first ("<file>:<line>"; LiteRT writes them to the runner's stderr, and the HTP
  # partition line is in no logcat), then the runner pid's key lines of the logcat
  ( cd "$R" && grep -H -E "Partitioned subgraph|Replacing [0-9]+ out of|cached model|compiler plugins? were applied" "$tag.stdout.txt" "$tag.stderr.txt" 2>/dev/null | head -40 ) > "$R/$tag.delegate.txt"
  if [ -n "$rpid" ]; then
    grep -E "^[0-9-]+ [0-9:.]+ +$rpid +[0-9]+ " "$R/$tag.logcat_all.txt" | grep -iE "Replacing|Partitioned|DispatchDelegate|QNN|Qnn|qnn|compiler plugin|cached model|reserializ|Unsupported|not supported|failed|error|fatal|abort|scudo" | head -400 >> "$R/$tag.delegate.txt"
    # the delegation evidence alone (a long vendor log can push it out of the first 400 lines above)
    grep -E "^[0-9-]+ [0-9:.]+ +$rpid +[0-9]+ " "$R/$tag.logcat_all.txt" | grep -E "Replacing [0-9]+ out of|Partitioned subgraph|cached model" > "$R/$tag.delegate_key.txt"
  fi
  grep -iE "lowmemorykiller|lmkd" "$R/$tag.logcat_all.txt" | head -80 > "$R/$tag.lmkd_lines.txt"
  killed=$(cat "$R/$tag.killed_by" 2>/dev/null)
  mkdir -p "$R/$tag.dump"
  A pull "$out/model0/." "$R/$tag.dump/" > /dev/null 2>&1
  ndump=$(find "$R/$tag.dump" -type f | wc -l | tr -d ' ')
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
doc = {"tag": "$tag", "index": $i, "accel": "$accel", "precision": "$prec", "graph": "$graph", "graph_file": "$(krel "$GRAPH_DIR")/$file",
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
       "stderr": {"file": "$tag.stderr.txt", "bytes_phone": int("$STDERR_BYTES_PHONE") if "$STDERR_BYTES_PHONE" else None,
                  "bytes_pulled": int("$STDERR_BYTES") if "$STDERR_BYTES" else None},
       "cmd": open("$R/$tag.cmd.txt").read().strip(), "dry_run": "$DRY_RUN" == "1"}
open(sys.argv[1], "w").write(json.dumps(doc, indent=1) + "\n")
EOF
  echo "=== leg $tag status $status rc $rc killed ${killed:-none} dumps $ndump (bad size $nbad) cache ${csize:-none} kB end $(date '+%T') uptime $up0 -> $up1"
  grep -E "^(model\[0\]=|parity_summary|summary|FATAL|ERROR|FIXTURE_ERROR|done)" "$R/$tag.stdout.txt" | tail -6
  grep -hE "^(FATAL|ERROR)" "$R/$tag.stderr.txt" 2>/dev/null | tail -3
  grep -hE "Partitioned subgraph|Replacing [0-9]+ out of" "$R/$tag.stdout.txt" "$R/$tag.stderr.txt" 2>/dev/null | head -4
  [ -s "$R/$tag.others_time.txt" ] && echo "other processes' CPU TIME grew during $tag: $(tr '\n' ' ' < "$R/$tag.others_time.txt")"
  [ -s "$R/$tag.lmkd_lines.txt" ] && echo "lmkd lines during $tag: $(wc -l < "$R/$tag.lmkd_lines.txt" | tr -d ' ') (R/$tag.lmkd_lines.txt)"
  if declare -F lane_npu_score > /dev/null; then
    ( cd "$LANE_DIR" && lane_npu_score "$tag" > "$R/$tag.score.txt" 2>&1 ) < /dev/null &
    SCORE_PIDS="$SCORE_PIDS $!"
  fi
  if [ "$status" = LMKD_KILLED ]; then hard_stop "lmkd_killed_the_runner_in_$tag"; fi
  ( evict "$i" ) >> "$R/s_graphs.txt" 2>&1
}

cleanup_all() {  # the runner killed if alive, the stage removed, read back
  {
    local p
    p=$(runner_pid); [ -n "$p" ] && { echo "runner pid $p alive at the cleanup -> kill -9"; A shell kill -9 "$p"; sleep 1; }
    A shell rm -rf "$STAGE"
    echo "stage left: $(A shell ls -d "$STAGE" 2>&1 | tr -d '\r')"
    echo "runner left: '$(runner_pid)'"
    echo "data_free: $(A shell df -h /data | tail -1 | tr -d '\r')"
    phone_state
  } > "$R/s9_cleanup.txt" 2>&1
  cat "$R/s9_cleanup.txt"
  present || { echo "the phone left adb during the cleanup $(ts)" > "$R/CHAIN_STOPPED"; return 9; }
  CLEANED=1
}

check_inputs() {  # the plan's legs, every graph's sha256, the runner, the libs against their sums, the fixture manifests
  mkdir -p "$R"
  read_plan
  local line tag accel prec graph fx count warm rounds ab cache climit ready extras ok=0 f man n hint
  echo "plan: $PLAN"
  for line in "${PLAN_LINES[@]}"; do
    read -r tag accel prec graph fx count warm rounds ab cache climit ready extras <<< "$line"
    case "$accel" in
      npu+cpu|npu|gpu) ;;
      *) hint=""; case "$accel" in *[0-9]*) hint=" (a number: the gate chain's columns? use chain.sh)" ;; esac
         echo "$tag: accel $accel$hint"; ok=1 ;;
    esac
    case "$accel" in gpu) case "$prec" in fp32|fp16acc32|fp16|default) ;; *) echo "$tag: precision $prec"; ok=1 ;; esac ;;
                     *) [ "$prec" = - ] || { echo "$tag: precision $prec for $accel"; ok=1; } ;; esac
    case "$ready" in gate|clean|uncapped) ;; *) echo "$tag: ready $ready"; ok=1 ;; esac
    case "$cache" in -|new:cache_*|use:cache_*) ;; *) echo "$tag: cache $cache (- | new:cache_<x> | use:cache_<x>)"; ok=1 ;; esac
    case "$count$warm$rounds$climit" in *[!0-9]*) echo "$tag: count / warmup / rounds / compile_s must be numbers ($count $warm $rounds $climit)"; ok=1 ;; esac
    f=$GRAPH_DIR/$(graph_file "$graph"); [ -s "$f" ] || { echo "$tag: missing graph $f"; ok=1; }
    man=$(manifest_of "$fx")
    if [ -s "$man" ]; then
      n=$(jq_get "$man" count); [ "$count" -le "$n" ] 2>/dev/null || { echo "$tag: count $count > the $n fixtures of $fx"; ok=1; }
      [ -f "$LANE_DIR/$(jq_get "$man" dir)/${ab}_$(first_input "$man").f32" ] || { echo "$tag: no fixture $ab in $LANE_DIR/$(jq_get "$man" dir)"; ok=1; }
    else
      echo "$tag: missing $man"; ok=1
    fi
    printf '%-26s %-8s %-5s %-18s %-8s %4s %3s %3s %4s %-24s %4s %-9s %s\n' "$tag" "$accel" "$prec" "$graph" "$fx" "$count" "$warm" "$rounds" "$ab" "$cache" "$climit" "$ready" "$extras"
  done
  echo "inputs (sha256):"
  for graph in $(printf '%s\n' "${PLAN_LINES[@]}" | awk '{print $4}' | LC_ALL=C sort -u); do
    f=$GRAPH_DIR/$(graph_file "$graph")
    [ -s "$f" ] && echo "  $(shasum -a 256 "$f" | cut -c1-16)… $(stat -L -f %z "$f") B $(krel "$f")"
  done
  if [ -s "$RUNNER_BIN" ]; then
    echo "  runner $(shasum -a 256 "$RUNNER_BIN" | cut -c1-16)… $(stat -L -f %z "$RUNNER_BIN") B $(krel "$RUNNER_BIN")"
    [ "$DRY_RUN" = 1 ] || [ -z "$RUNNER_SHA_REF" ] || [ "$(shasum -a 256 "$RUNNER_BIN" | cut -d' ' -f1)" = "$RUNNER_SHA_REF" ] || { echo "runner sha256 differs from the recorded $RUNNER_SHA_REF"; ok=1; }
  else
    echo "  missing runner $RUNNER_BIN"; ok=1
  fi
  if [ -d "$LIBS" ] && [ -s "$LIBS_SUMS" ]; then
    ( cd "$LIBS" && shasum -a 256 -c "$LIBS_SUMS" ) > "$R/s0_libs_check.txt" 2>&1 || { echo "libs sha256 check failed (R/s0_libs_check.txt)"; ok=1; }
    echo "  libs: $(grep -c ': OK$' "$R/s0_libs_check.txt")/$(grep -c . "$LIBS_SUMS" | tr -d ' ') OK against $LIBS_SUMS"
  else
    echo "  missing libs $LIBS or their sums $LIBS_SUMS"; ok=1
  fi
  for fx in $(printf '%s\n' "${PLAN_LINES[@]}" | awk '{print $5}' | LC_ALL=C sort -u); do
    man=$(manifest_of "$fx")
    [ -s "$man" ] || continue
    echo "  $fx: $(jq_get "$man" files) files, $(jq_get "$man" bytes) B, concat sha256 $(jq_get "$man" concat_sha256_bytesorted | cut -c1-16)…$([ -d "$LANE_DIR/$(jq_get "$man" dir)" ] || echo " (dir $(jq_get "$man" dir) missing)")"
    [ -d "$LANE_DIR/$(jq_get "$man" dir)" ] || ok=1
  done
  return $ok
}

run_chain() {
  mkdir -p "$R"
  rm -f "$R/CHAIN_DONE" "$R/CHAIN_STOPPED" "$R/RELEASE_HOLD"
  echo "$$" > "$R/chain.pid"
  echo "chain $ROUND pid $$ $(ts); this file sha256 $(shasum -a 256 "$0" | cut -c1-16); lib.sh sha256 $(shasum -a 256 "$BENCH/lib.sh" | cut -c1-16); lane.env $LANE_ENV sha256 $(shasum -a 256 "$LANE_ENV" | cut -c1-16); plan $PLAN sha256 $(shasum -a 256 "$PLAN" | cut -c1-16)"
  check_inputs > "$R/s0_inputs.txt" 2>&1 || { cat "$R/s0_inputs.txt"; echo "inputs check failed: no hold taken"; exit 5; }
  cat "$R/s0_inputs.txt"
  read_plan
  [ "$(printf '%s\n' "${PLAN_LINES[@]}" | awk '{print $1}' | LC_ALL=C sort -u | wc -l | tr -d ' ')" = "${#PLAN_LINES[@]}" ] || { echo "duplicate leg tags"; exit 5; }
  mkdir -p "$RESULTS_DIR"
  take_hold
  LEG_END=$(now)
  present || hard_stop "device_absent_at_start"
  gpu_detect || { echo "the GPU's driver was not read (got '$GPU_KIND'): no GPU reads"; GPU_KIND=none; }
  cooling_detect || true
  export GPU_KIND COOLING_SRC COOLING_SYSFS
  SAMPLE_CMD=$(sample_cmd "$RNAME" "MemAvailable|MemFree|SwapFree"); SLOW_CMD=$(slow_cmd)
  echo "gpu at $(ts): $GPU_KIND (the ready gate's GPU level: $(gpu_level_name))" | tee "$R/s_gpu.txt"
  gpu_probe | sed 's/^/  /' >> "$R/s_gpu.txt"
  cooling_note | tee -a "$R/s_gpu.txt"
  thermal_raw
  if ! transfer_check; then
    echo "OTHER SESSION'S TRANSFER / PROCESS ON THE PHONE -> no leg, release $(ts)"
    echo "other_session_job $(ts)" > "$R/CHAIN_STOPPED"; CLEANED=1; exit 5
  fi
  LAST_UPTIME=$(uptime_s); echo "phone uptime at the hold: ${LAST_UPTIME}s"
  phone_state > "$R/s0_state_initial.txt" 2>&1; cat "$R/s0_state_initial.txt"
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
  *) echo "usage: $0 check | run | cleanup <round> <lane.env>" >&2; exit 2 ;;
esac
