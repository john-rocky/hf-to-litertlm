#!/bin/bash
# lib.sh: the shared half of the device gate harness, sourced by chain.sh and run.sh (and npu_chain.sh). It defines
# functions and the lane's settings and runs nothing: the lane's lane.env (bench_load), the phone's basic reads (the
# GPU's driver: Adreno kgsl or Arm Mali), the plan, the owned hold (keeper, clock, release, cleanup by hand), the ready
# gate, the sampler's commands, the process accounting and the transfer check. The chain that sources it defines
# cleanup_all (what its own cleanup removes from the phone).
# Kept out (traps of the chains this body was cut from): no `timeout adb` (it kills only the Mac's client; waits poll
# the phone instead); every remote `date` format quoted; `LC_ALL=C sort`; no function defined inside another function
# and no name reused; never a bare `wait` (the keeper is a child: background jobs are waited for by pid); `stat -L -f
# %z` for sizes (macOS host); a body file is never edited in place while a chain may run (write a new file and mv it
# over the old one: a running bash keeps the old inode); the keeper runs from a copy in the round dir.

BENCH=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

re_escape() {  # text -> the text as a literal in an extended regular expression
  printf '%s' "$1" | sed 's/[][\.*^$+?(){}|/]/\\&/g'
}

bench_load() {  # <lane.env>: the lane's settings, then the body's defaults for what the lane leaves out
  # lane.env is bash: VAR=${VAR:-value} lines (so the caller's environment wins) and, optionally, the hook functions
  # lane_score / lane_score_mac / lane_prep / lane_graph_file / lane_npu_score. It may use ROUND, LANE_DIR, DEVICE_DIR,
  # BENCH and CA, set before it is read. LANE, SERIAL and HOLD are required here; each chain checks its own.
  # CA = the hold tools' dir: hold/ in this folder when it has queue_cli.py, else this folder's parent.
  [ -f "${1:-}" ] || { echo "no lane.env: ${1:-<none>}" >&2; exit 2; }
  LANE_ENV="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
  DEVICE_DIR=${DEVICE_DIR:-$(dirname "$LANE_ENV")}
  LANE_DIR=${LANE_DIR:-$(dirname "$DEVICE_DIR")}
  if [ -z "${CA:-}" ]; then
    if [ -f "$BENCH/hold/queue_cli.py" ]; then CA=$BENCH/hold; else CA=$(dirname "$BENCH"); fi
  fi
  ROUND=${ROUND:-}
  # shellcheck source=/dev/null
  . "$LANE_ENV"
  local v
  for v in LANE SERIAL HOLD; do
    [ -n "${!v:-}" ] || { echo "$LANE_ENV: $v is not set" >&2; exit 2; }
  done
  DEVICE=${DEVICE:-phone}
  R=${ROUND_DIR:-$DEVICE_DIR/$ROUND}
  GRAPH_DIR=${GRAPH_DIR:-$LANE_DIR/out}; ROWS_DIR=${ROWS_DIR:-$DEVICE_DIR}; RESULTS_DIR=${RESULTS_DIR:-$LANE_DIR/results}
  APK=${APK:-$DEVICE_DIR/app-debug.$ROUND.apk}; APK_SHA=${APK_SHA:-$(cut -d' ' -f1 "$APK.sha256" 2>/dev/null)}
  SCRIPT_NAME=${SCRIPT_NAME:-$LANE-$ROUND-$DEVICE}
  ACTIVITY=${ACTIVITY:-.GateActivity}
  SIG_OF=${SIG_OF:-}
  APP_INT_EXTRAS=${APP_INT_EXTRAS-limit_rows:limit warmup rest_ms reps cool_ms cpu_cache gpu_src_quant}
  APP_STR_EXTRAS=${APP_STR_EXTRAS-}
  APP_OUTPUTS=${APP_OUTPUTS:-sel full:keep_scores}
  GATE_OUT=${APP_OUTPUTS%% *}; GATE_OUT=${GATE_OUT%%:*}
  PKG=${PKG:-}; STAGE=${STAGE:-}; PKG_RE=$(re_escape "$PKG")
  APP_LOG_TAG=${APP_LOG_TAG:-}
  APP_LOG_RE=${APP_LOG_RE:-$PKG_RE${APP_LOG_TAG:+|$APP_LOG_TAG}}
  PEER_APP_RE=${PEER_APP_RE-$([ -z "$PKG" ] || re_escape "$(echo "$PKG" | cut -d. -f1-2).")}
  PY=${PY:-python3}; ADB=${ADB:-adb}; STANDIN=${STANDIN:-}
  DRY_RUN=${DRY_RUN:-0}
  SKIP_MIN=${SKIP_MIN:-25}; RUN_END_MIN=${RUN_END_MIN:-28}; END_MIN=${END_MIN:-30}
  REST_MAX=${REST_MAX:-300}; REST_MAX_T=${REST_MAX_T:-600}; POLL=${POLL:-15}
  MEM_GUARD_KB=${MEM_GUARD_KB:-1500000}; DATA_MARGIN=${DATA_MARGIN:-2000000000}; XFER_MAX_S=${XFER_MAX_S:-900}
  VERDICT_WAIT_S=${VERDICT_WAIT_S:-120}; MAC_THREADS=${MAC_THREADS:-8}
  GPU_KIND=${GPU_KIND:-}; COOLING_SRC=${COOLING_SRC:-}; COOLING_SYSFS=${COOLING_SYSFS:-}
  GEN=$R/gen
  export LANE_ENV ROUND CA SERIAL ADB PY DRY_RUN
  if [ "$DRY_RUN" = 1 ]; then
    export FAKE_PKG=$PKG FAKE_ACTIVITY=$ACTIVITY FAKE_SERIAL=$SERIAL FAKE_STANDIN=$STANDIN FAKE_RUNNER_NAME=${NPU_RUNNER_NAME:-}
  fi
}

dry_run_guard() {  # a dry run touches nothing outside this folder's cache/; the fake adb never runs outside a dry run
  if [ "$DRY_RUN" = 1 ]; then
    [ "$ADB" = "$BENCH/fake_adb.py" ] || { echo "DRY_RUN=1 needs ADB=$BENCH/fake_adb.py (got $ADB)" >&2; exit 2; }
    case "$HOLD" in "$BENCH"/cache/*) ;; *) echo "DRY_RUN=1 needs a hold file under $BENCH/cache (got $HOLD)" >&2; exit 2 ;; esac
    case "$RESULTS_DIR" in "$BENCH"/cache/*) ;; *) echo "DRY_RUN=1 needs RESULTS_DIR under $BENCH/cache (got $RESULTS_DIR)" >&2; exit 2 ;; esac
    case "$R" in "$BENCH"/cache/*) ;; *) echo "DRY_RUN=1 needs ROUND_DIR under $BENCH/cache (got $R)" >&2; exit 2 ;; esac
    [ -n "$STANDIN" ] || { echo "DRY_RUN=1 needs STANDIN (the app's stand-in) in lane.env" >&2; exit 2; }
  elif [ "$(basename "$ADB")" = fake_adb.py ]; then
    echo "the fake adb outside a dry run (set DRY_RUN=1)" >&2; exit 2
  fi
}

A() {  # the adb client on this phone; stdin is /dev/null except for exec-in (inside a `while read` loop an adb call
  # would otherwise eat the loop's input)
  if [ "${1:-}" = exec-in ]; then "$ADB" -s "$SERIAL" "$@"; else "$ADB" -s "$SERIAL" "$@" < /dev/null; fi
}
now() { date +%s; }
ts() { date '+%F %T'; }
hms() { date -r "$1" '+%T'; }
present() { [ "$(A get-state 2>/dev/null | tr -d '\r')" = "device" ]; }
hard_stop() { echo "HARD STOP: $1 $(ts)"; echo "$1 $(ts)" > "$R/CHAIN_STOPPED"; exit 9; }
cpu_capped() { A shell 'for p in /sys/devices/system/cpu/cpufreq/policy*; do echo "${p##*/} $(cat $p/scaling_max_freq) $(cat $p/cpuinfo_max_freq)"; done' | tr -d '\r' | awk '$2 != $3 {printf "%s:%s/%s ", $1, $2, $3}'; }

# The GPU's driver decides where its clock and thermal level are read. Adreno (kgsl): /sys/class/kgsl/kgsl-3d0/
# (clock_mhz, max_clock_mhz, thermal_pwrlevel, temp; readable by the shell user). Arm Mali: /sys/class/misc/mali0/device/,
# where the shell user reads cur_freq (kHz), total_gpu_mem and dma_buf_gpu_mem (bytes) but not the frequency limits or
# the governor; its thermal level is the state of its GPU cooling devices in `dumpsys thermalservice` (the largest
# mValue among the devices of mType 3; 0 = not capped) and its temperature the HAL sensor G3D. Neither: no GPU reads, and
# the ready gate never sees a clean phone.
# Temperatures: `dumpsys thermalservice` prints "Cached temperatures:" (the values of the HAL's last callback, which can be
# old: SKIN 37.9 °C there against 30.1 °C current on a Galaxy S26) before "Current temperatures from HAL:", then the
# HAL's cooling devices and the thresholds. Every temperature (skin, G3D, hal_temps) is read from the current section only
# (hal_current); the thermal status is a line of its own, outside both.
# Cooling devices, read from one source per run (cooling_detect): hal when the HAL's cooling-device list has any device
# (the CPU / GPU ones chosen by their type, never by their name: mType 2 = CPU, 3 = GPU, android.os.CoolingDevice
# TYPE_CPU / TYPE_GPU; the Pixel 8a lists thermal-cpufreq-0/1/2 and thermal-gpufreq-0), else sysfs when the shell user
# reads /sys/class/thermal/cooling_device*/type (the CPU, GPU, DSP and memory-bus devices, chosen by their type name:
# cpufreq-*, cpu-cluster*, gpu, kgsl, cdsp*, ddr-cdev; a Galaxy S26's HAL lists none), else none.
COOLING_ERE='CoolingDevice\{mValue=[0-9]+, mType=[23], mName=[^}]*\}'
HAL_TEMPS_SED='/^Current temperatures from HAL/,/^Current cooling devices from HAL/p'
HAL_COOLING_SED='/^Current cooling devices from HAL/,/^Temperature static thresholds from HAL/p'
SYSFS_THERMAL=/sys/class/thermal
SYSFS_COOLING_ERE='^(cpufreq-[A-Za-z0-9_.-]*|cpu-cluster[A-Za-z0-9_.-]*|gpu|kgsl|cdsp[A-Za-z0-9_.-]*|ddr-cdev)$'
KGSL=/sys/class/kgsl/kgsl-3d0
MALI=/sys/class/misc/mali0/device
hal_current() { sed -n "$HAL_TEMPS_SED"; }    # dumpsys thermalservice -> the HAL's current temperatures (no cached one)
hal_cooling() { sed -n "$HAL_COOLING_SED"; }  # dumpsys thermalservice -> the HAL's cooling devices
gpu_detect() {  # GPU_KIND = kgsl | mali | none, read from the phone once per process (chain.sh exports it to run.sh)
  [ -n "$GPU_KIND" ] && return 0
  GPU_KIND=$(A shell "if [ -d $KGSL ]; then echo kgsl; elif [ -d $MALI ]; then echo mali; else echo none; fi" | tr -d '\r')
  case "$GPU_KIND" in kgsl|mali|none) return 0 ;; *) GPU_KIND=""; return 1 ;; esac
}
MALI_FILES="cur_freq hint_min_freq dvfs_period capacity_headroom dma_buf_gpu_mem total_gpu_mem max_freq min_freq scaling_max_freq hint_max_freq governor power_policy available_frequencies gpu_top clock_info dvfs_table"
gpu_probe() {  # Mali only: which of the GPU's sysfs files the shell user can read, for the record of the run: "readable
  # <file>", or "<file>: <cat's error>" (Permission denied, No such file or directory, ...). Judged by cat itself: a
  # file the shell user may not even stat fails `[ -e ]` while it exists
  [ "$GPU_KIND" = mali ] || return 0
  A shell "for f in $MALI_FILES; do e=\$(cat $MALI/\$f 2>&1 > /dev/null | head -n 1); if [ -z \"\$e\" ]; then echo \"readable \$f\"; else echo \"\$f: \${e##*: }\"; fi; done" | tr -d '\r'
}
gpu_level_name() {  # the name of the GPU's thermal level in the ready gate's lines
  case "$GPU_KIND" in kgsl) echo kgsl_pwrlevel ;; mali) echo gpu_cooling ;; *) echo gpu_level ;; esac
}
cooling_detect() {  # COOLING_SRC = hal | sysfs | none and, for sysfs, COOLING_SYSFS = the kept devices ("<dir>:<type>
  # ...", in the phone's glob order), read from the phone once per process (chain.sh exports both to run.sh)
  [ -n "$COOLING_SRC" ] && return 0
  local n
  n=$(A shell dumpsys thermalservice | tr -d '\r' | hal_cooling | grep -c 'CoolingDevice{')
  if [ "${n:-0}" -gt 0 ] 2>/dev/null; then COOLING_SRC=hal; return 0; fi
  COOLING_SYSFS=$(A shell 'for d in '"$SYSFS_THERMAL"'/cooling_device*; do echo "${d##*/} $(cat $d/type 2>/dev/null)"; done' \
    | tr -d '\r' | awk -v re="$SYSFS_COOLING_ERE" 'NF == 2 && $2 ~ re {printf "%s%s:%s", s, $1, $2; s = " "}')
  if [ -n "$COOLING_SYSFS" ]; then COOLING_SRC=sysfs; else COOLING_SRC=none; fi
}
cooling_sysfs_cmd() {  # -> the phone's command printing "cooling <type> <cur_state>/<max_state>" per kept sysfs device
  echo 'for p in '"$COOLING_SYSFS"'; do d='"$SYSFS_THERMAL"'/${p%%:*}; echo "cooling ${p#*:} $(cat $d/cur_state 2>/dev/null)/$(cat $d/max_state 2>/dev/null)"; done'
}
cooling_line() {  # dumpsys thermalservice (the text) -> the state files' cooling: value: hal = "<name>=<mValue>(type
  # <mType>);" per CPU / GPU device (mType 2 / 3), sysfs = "<type>=<cur_state>/<max_state>;" per kept device, none = n/a
  case "$COOLING_SRC" in
    hal) echo "$1" | hal_cooling | grep -oE "$COOLING_ERE" | sed -E 's/^CoolingDevice\{mValue=([0-9]+), mType=([0-9]+), mName=([^}]*)\}/\3=\1(type \2)/' | tr '\n' ';' ;;
    sysfs) A shell "$(cooling_sysfs_cmd)" | tr -d '\r' | awk '$1 == "cooling" {printf "%s=%s;", $2, $3}' ;;
    *) printf 'n/a' ;;
  esac
}
cooling_note() {  # one line for s_gpu.txt: where this run reads the cooling devices
  case "$COOLING_SRC" in
    hal) echo "cooling: hal (the thermal service's cooling-device list; the CPU / GPU devices by mType 2 / 3)" ;;
    sysfs) echo "cooling: sysfs (the thermal service lists no cooling device; $SYSFS_THERMAL, kept by type: $COOLING_SYSFS)" ;;
    *) echo "cooling: none (the thermal service lists no cooling device and $SYSFS_THERMAL/cooling_device*/type gave none of the kept types)" ;;
  esac
}
cooling_max() {  # mType < dumpsys thermalservice -> the largest state (mValue) among the HAL's cooling devices of that
  # type; empty when the phone lists none of it
  hal_cooling | grep -oE "CoolingDevice\{mValue=[0-9]+, mType=$1, mName=[^}]*\}" | sed -E 's/^CoolingDevice\{mValue=([0-9]+),.*/\1/' | LC_ALL=C sort -n | tail -n 1
}
thermal_raw() {  # once after the hold: the phone's `dumpsys thermalservice` as it is (R/s_thermal_raw.txt: its cached and
  # current temperatures and the HAL's cooling devices with their mType and mValue), every sysfs cooling device with its
  # type and cur_state / max_state, or cat's error (R/s_cooling_sysfs.txt), and one line about both
  local sk_c sk_h
  { echo "dumpsys thermalservice at $(ts):"; A shell dumpsys thermalservice | tr -d '\r'; } > "$R/s_thermal_raw.txt" 2>&1
  { echo "$SYSFS_THERMAL/cooling_device* at $(ts) (dir type cur_state/max_state):"
    A shell 'for d in '"$SYSFS_THERMAL"'/cooling_device*; do echo "${d##*/} $(cat $d/type 2>&1) $(cat $d/cur_state 2>&1)/$(cat $d/max_state 2>&1)"; done' | tr -d '\r'
  } > "$R/s_cooling_sysfs.txt" 2>&1
  sk_c=$(sed -n '/^Cached temperatures/,/^HAL Ready/p' "$R/s_thermal_raw.txt" | grep -m1 -oE 'mValue=[-0-9.]+, mType=3, mName=[A-Za-z0-9_-]*SKIN' | sed -E 's/^mValue=([-0-9.]+),.*/\1/')
  sk_h=$(hal_current < "$R/s_thermal_raw.txt" | grep -m1 -oE 'mValue=[-0-9.]+, mType=3, mName=[A-Za-z0-9_-]*SKIN' | sed -E 's/^mValue=([-0-9.]+),.*/\1/')
  echo "thermal service at the hold: skin ${sk_h:-?} °C (current; the cached section says ${sk_c:-?}), $(hal_cooling < "$R/s_thermal_raw.txt" | grep -c 'CoolingDevice{' | tr -d ' ') cooling devices listed, CPU (mType 2) $(hal_cooling < "$R/s_thermal_raw.txt" | grep -cE 'CoolingDevice\{mValue=[0-9]+, mType=2,' | tr -d ' '), GPU (mType 3) $(hal_cooling < "$R/s_thermal_raw.txt" | grep -cE 'CoolingDevice\{mValue=[0-9]+, mType=3,' | tr -d ' '); sysfs cooling devices read $(grep -cE '^cooling_device[0-9]+ [^ ]+ [0-9]+/[0-9]+$' "$R/s_cooling_sysfs.txt" | tr -d ' '); cooling read from ${COOLING_SRC:-?} (s_thermal_raw.txt, s_cooling_sysfs.txt)"
}
read_levels() {  # -> T (the thermal status), C (the capped CPU policies), G (the GPU's thermal level: kgsl
  # thermal_pwrlevel, or on Mali the largest state of its GPU cooling devices (mType 3) from the same dumpsys as T; empty
  # when there is none)
  local th
  [ -n "$GPU_KIND" ] || gpu_detect || true
  th=$(A shell dumpsys thermalservice | tr -d '\r')
  T=$(echo "$th" | grep -m1 'Thermal Status' | grep -oE '[0-9]+$')
  C=$(cpu_capped)
  case "$GPU_KIND" in
    kgsl) G=$(A shell cat $KGSL/thermal_pwrlevel 2>/dev/null | tr -d '\r') ;;
    mali) G=$(echo "$th" | cooling_max 3) ;;
    *) G="" ;;
  esac
}

sample_cmd() {  # process name, /proc/meminfo fields (an extended regular expression) -> the 2 s sampler's command on the
  # phone: devtime, the GPU line (kgsl: "kgsl <clock MHz> <max clock MHz> <thermal_pwrlevel> <temp m°C>"; mali: "gpu mali
  # <cur_freq kHz> <total_gpu_mem B> <dma_buf_gpu_mem B>" with "?" for a file it cannot read; none: "gpu none"), one
  # "cpu <policy> <cur> <max> <cpuinfo max>" line per policy, the meminfo fields, the process's pid and VmRSS / VmHWM /
  # VmSwap and, on kgsl, the GPU's page_alloc
  local gpu tail=''
  case "$GPU_KIND" in
    kgsl) gpu='echo "kgsl $(cat '$KGSL'/clock_mhz) $(cat '$KGSL'/max_clock_mhz) $(cat '$KGSL'/thermal_pwrlevel) $(cat '$KGSL'/temp)"'
          tail='; echo "kgsl_page_alloc $(cat /sys/class/kgsl/kgsl/page_alloc 2>/dev/null)"' ;;
    mali) gpu='g() { v=$(cat "$1" 2>/dev/null | head -n 1); echo "${v:-?}"; }; echo "gpu mali $(g '$MALI'/cur_freq) $(g '$MALI'/total_gpu_mem) $(g '$MALI'/dma_buf_gpu_mem)"' ;;
    *) gpu='echo "gpu none"' ;;
  esac
  echo 'echo "devtime $(date +%s)"; p=$(pidof '"$1"'); '"$gpu"'; for q in /sys/devices/system/cpu/cpufreq/policy*; do echo "cpu ${q##*/} $(cat $q/scaling_cur_freq) $(cat $q/scaling_max_freq) $(cat $q/cpuinfo_max_freq)"; done; grep -E "'"$2"'" /proc/meminfo; echo "pid $p"; [ -n "$p" ] && grep -E "VmRSS|VmHWM|VmSwap" /proc/$p/status'"$tail"
}

slow_cmd() {  # -> the sampler's slower command (every ~15 s): the thermal status, the skin temperature (SKIN, or a
  # name ending in SKIN such as VIRTUAL-SKIN) from the HAL's current temperatures, the cooling devices of COOLING_SRC
  # (hal: one "cooling <name> <state> type=<mType>" line per CPU / GPU device, mType 2 / 3 whatever its name; sysfs: one
  # "cooling <type> <cur_state>/<max_state>" line per kept device; none: no line), on Mali the GPU's temperature (G3D,
  # from the HAL's current temperatures), the battery temperature, the screen state (dumpsys's "Failed to write ...
  # Broken pipe" when grep -m1 stops reading goes to /dev/null)
  local g3d='' cool=''
  [ -n "$COOLING_SRC" ] || cooling_detect || true
  [ "$GPU_KIND" = mali ] && g3d='; echo "$c" | grep -m1 -oE "mValue=[-0-9.]+, mType=[0-9]+, mName=G3D"'
  case "$COOLING_SRC" in
    hal) cool='; echo "$t" | sed -n "'"$HAL_COOLING_SED"'" | grep -oE "'"$COOLING_ERE"'" | sed -E "s/^CoolingDevice\{mValue=([0-9]+), mType=([0-9]+), mName=([^}]*)\}/cooling \3 \1 type=\2/"' ;;
    sysfs) cool="; $(cooling_sysfs_cmd)" ;;
  esac
  echo 't=$(dumpsys thermalservice); c=$(echo "$t" | sed -n "'"$HAL_TEMPS_SED"'"); echo "$t" | grep -m1 "Thermal Status"; echo "$c" | grep -m1 -oE "mValue=[-0-9.]+, mType=3, mName=[A-Za-z0-9_-]*SKIN"'"$cool$g3d"'; dumpsys battery 2>/dev/null | grep -m1 temperature; dumpsys power 2>/dev/null | grep -m1 -oE "mWakefulness=[A-Za-z]+"'
}

uptime_s() { A shell cat /proc/uptime 2>/dev/null | tr -d '\r' | awk '{printf "%d", $1}'; }
avail_bytes() { A shell df -k /data | tail -1 | awk '{print $4 * 1024}' | tr -d '\r'; }
xget() {  # extras key default -> the value of key=... in the extras, else the default
  local kv
  for kv in $1; do case "$kv" in "$2"=*) echo "${kv#*=}"; return ;; esac; done
  echo "$3"
}

plan_file() {  # -> the plan of ROUND: PLAN, else the first of <device dir>/<round>/plan_<round>.tsv,
  # <device dir>/<round>/plan.tsv, <device dir>/plan_<round>.tsv (the three places lanes have kept their plans)
  local f
  if [ -n "${PLAN:-}" ]; then echo "$PLAN"; return; fi
  for f in "$DEVICE_DIR/$ROUND/plan_$ROUND.tsv" "$DEVICE_DIR/$ROUND/plan.tsv" "$DEVICE_DIR/plan_$ROUND.tsv"; do
    [ -s "$f" ] && { echo "$f"; return; }
  done
  echo "$DEVICE_DIR/$ROUND/plan_$ROUND.tsv"
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

ready_gate() {  # name since_epoch min_rest_s mode [poll_s] -> READY_NOTE, READY_SKIP
  # clean = wait for clean up to REST_MAX, then run warm; gate = heat does not change a verdict: run without waiting for
  # clean, but never at thermal status >= 4 (not below 4 by REST_MAX -> skipped); timing = run only from clean (not
  # clean by REST_MAX_T or the hold's deadline -> skipped, left for the next hold); uncapped = run when the clocks are
  # uncapped (no CPU cap, the GPU's thermal level 0: kgsl thermal_pwrlevel, or Mali's GPU cooling devices (mType 3))
  # at thermal status <= 1, else as timing (UNCAPPED_FALLBACK=warm: run warm instead, the NPU chain's rule). A phone
  # whose GPU level cannot be read (GPU_KIND none) is never clean or uncapped.
  local name=$1 since=$2 minrest=$3 mode=$4 poll=${5:-$POLL} T C G lim gl
  READY_SKIP=0
  [ -n "$GPU_KIND" ] || gpu_detect || true
  gl=$(gpu_level_name)
  while [ $(( $(now) - since )) -lt "$minrest" ]; do present || hard_stop "device_lost_in_rest_before_$name"; sleep 5; done
  lim=$REST_MAX; { [ "$mode" = timing ] || [ "$mode" = uncapped ]; } && lim=$REST_MAX_T
  while true; do
    read_levels
    echo "$(date '+%T') ready? $name: thermal ${T:-?} cap ${C:-none} $gl ${G:-?} rested $(( $(now) - since ))s mode $mode" | tee -a "$R/s_ready.txt"
    if [ "${T:-9}" = 0 ] && [ -z "$C" ] && [ "${G:-9}" = 0 ]; then READY_NOTE="clean"; break; fi
    if [ "$mode" = uncapped ] && [ -z "$C" ] && [ "${G:-9}" = 0 ] && [ "${T:-9}" -le 1 ] 2>/dev/null; then
      READY_NOTE="uncapped (thermal ${T:-?}, no CPU cap, $gl 0)"; break
    fi
    if [ "$mode" = gate ] && [ "${T:-9}" -le 3 ] 2>/dev/null; then
      READY_NOTE="gate leg, not waited for clean (thermal ${T:-?}, cap ${C:-none}, $gl ${G:-?})"; break
    fi
    if [ $(( $(now) - since )) -ge "$lim" ]; then
      { [ "$mode" = clean ] || { [ "$mode" = uncapped ] && [ "${UNCAPPED_FALLBACK:-skip}" = warm ]; }; } && { READY_NOTE="warm (thermal ${T:-?}, cap ${C:-none}, $gl ${G:-?})"; break; }
      READY_NOTE="not ready after ${lim}s in $mode mode (thermal ${T:-?}, cap ${C:-none}, $gl ${G:-?})"; READY_SKIP=1; break
    fi
    if [ "$(now)" -ge $(( SKIP_AFTER - 20 )) ]; then
      { [ "$mode" = clean ] || { [ "$mode" = uncapped ] && [ "${UNCAPPED_FALLBACK:-skip}" = warm ]; }; } && { READY_NOTE="warm-deadline (thermal ${T:-?}, cap ${C:-none}, $gl ${G:-?})"; break; }
      READY_NOTE="not ready by the hold's deadline in $mode mode (thermal ${T:-?}, cap ${C:-none}, $gl ${G:-?})"; READY_SKIP=1; break
    fi
    sleep "$poll"
  done
  READY_NOTE="$READY_NOTE; rested $(( $(now) - since ))s since $(hms "$since")"
  echo "$name ready gate: $READY_NOTE" | tee -a "$R/s_ready.txt"
}

phone_state() {  # the phone's state lines (a scorer reads them as "key: value"): time, uptime, thermal, the skin and
  # HAL temperatures (the HAL's current temperatures only, never its cached section), the cooling devices (cooling_line:
  # "<name>=<mValue>(type <mType>);" for the HAL's mType 2 and 3, "<type>=<cur_state>/<max_state>;" for the kept sysfs
  # devices, or n/a), battery, screen, the front activity, the GPU (kind; kgsl: its line as before; mali: cur_freq,
  # memory, gpu_cooling = the largest state of its mType 3 cooling devices, and G3D), CPU frequencies and caps,
  # MemAvailable, /data free and, with STATE_STAGE set, the size of that stage dir
  echo "time: $(date '+%F %T')"
  echo "uptime: $(A shell cat /proc/uptime | tr -d '\r')"
  local th cur fq mf
  [ -n "$GPU_KIND" ] || gpu_detect || true
  [ -n "$COOLING_SRC" ] || cooling_detect || true
  th=$(A shell dumpsys thermalservice | tr -d '\r'); cur=$(echo "$th" | hal_current)
  echo "thermal: $(echo "$th" | grep -m1 'Thermal Status')"
  echo "skin: $(echo "$cur" | grep -m1 -oE 'mValue=[-0-9.]+, mType=3, mName=[A-Za-z0-9_-]*SKIN')"
  echo "hal_temps: $(echo "$cur" | grep -oE 'Temperature\{mValue=[-0-9.]+, mType=[0-9]+, mName=[A-Za-z0-9_-]+' | sed 's/Temperature{//' | tr '\n' ';')"
  echo "cooling: $(cooling_line "$th")"
  echo "battery_temp: $(A shell dumpsys battery | grep -m1 temperature | tr -d '\r ')"
  echo "battery_level: $(A shell dumpsys battery | grep -m1 level | tr -d '\r ')"
  echo "screen: $(A shell dumpsys power | grep -m1 -E 'mWakefulness=' | tr -d '\r ')"
  echo "top: $(A shell dumpsys activity activities | grep -m1 topResumedActivity | tr -d '\r')"
  case "$GPU_KIND" in
    kgsl)
      echo "gpu: kind=kgsl"
      echo "kgsl: temp=$(A shell cat $KGSL/temp 2>/dev/null | tr -d '\r') clock_mhz=$(A shell cat $KGSL/clock_mhz 2>/dev/null | tr -d '\r') max_clock_mhz=$(A shell cat $KGSL/max_clock_mhz 2>/dev/null | tr -d '\r') thermal_pwrlevel=$(A shell cat $KGSL/thermal_pwrlevel 2>/dev/null | tr -d '\r')" ;;
    mali)
      mf=$(A shell 'for f in cur_freq total_gpu_mem dma_buf_gpu_mem; do v=$(cat '"$MALI"'/$f 2>/dev/null | head -n 1); echo "$f=${v:-?}"; done' | tr -d '\r' | tr '\n' ' ')
      echo "gpu: kind=mali ${mf}gpu_cooling=$(echo "$th" | cooling_max 3) G3D=$(echo "$cur" | grep -m1 -oE 'mValue=[-0-9.]+, mType=[0-9]+, mName=G3D' | sed -E 's/^mValue=([-0-9.]+),.*/\1/')" ;;
    *) echo "gpu: kind=${GPU_KIND:-unknown} (no GPU clock or level read)" ;;
  esac
  fq=$(A shell 'for p in /sys/devices/system/cpu/cpufreq/policy*; do echo "${p##*/} $(cat $p/scaling_cur_freq) $(cat $p/scaling_max_freq) $(cat $p/cpuinfo_max_freq)"; done' | tr -d '\r')
  echo "freq (policy cur scaling_max cpuinfo_max): $(echo "$fq" | tr '\n' ';')"
  echo "freq_capped: $(echo "$fq" | awk '$3 != $4 {printf "%s:%s/%s ", $1, $3, $4; c=1} END {if (!c) printf "none"}')"
  echo "mem: $(A shell grep -E 'MemAvailable' /proc/meminfo | tr -d '\r')"
  echo "data_free: $(A shell df -h /data | tail -1 | tr -d '\r')"
  [ -z "${STATE_STAGE:-}" ] || echo "stage: $(A shell "du -sk $STATE_STAGE 2>/dev/null" | tr -d '\r')"
}

ps_time() {  # file: the phone's processes with their cumulative CPU TIME (another app's TIME that grows = contamination)
  A shell ps -A -o TIME,NAME 2>/dev/null | tr -d '\r' | LC_ALL=C sort -k2 > "$1"
}

others_time() {  # before after out: the processes other than ours (OWN_NAME: the app, or the runner) whose CPU TIME grew
  # by >= 2 s in between
  awk -v pkg="${OWN_NAME:-$PKG}" '
    function secs(t,   d, a, n, s) { d = 0; if (index(t, "-")) { split(t, a, "-"); d = a[1]; t = a[2] } n = split(t, a, ":"); s = 0
      for (i = 1; i <= n; i++) s = s * 60 + a[i]; return d * 86400 + s }
    FNR == 1 { next }
    NR == FNR { before[$2] = secs($1); next }
    $2 != pkg && ($2 in before) && secs($1) - before[$2] >= 2 { printf "%s +%ds\n", $2, secs($1) - before[$2] }' "$1" "$2" > "$3"
}

memory_line() {  # samples file -> the leg's peaks
  # empty = never sampled; a sampled 0 prints 0 (VmSwap is often 0 kB all along); the GPU's memory: kgsl page_alloc, or
  # on Mali total_gpu_mem (printed only when a Mali line was sampled)
  awk '/^VmHWM:/ {if (h == "" || $2 + 0 > h) h = $2 + 0} /^VmSwap:/ {if (s == "" || $2 + 0 > s) s = $2 + 0}
       /^MemAvailable:/ {if (m == "" || $2 + 0 < m) m = $2 + 0} /^kgsl_page_alloc [0-9]/ {if (k == "" || $2 + 0 > k) k = $2 + 0}
       /^gpu mali / && $4 ~ /^[0-9]+$/ {if (g == "" || $4 + 0 > g) g = $4 + 0}
       END {printf "VmHWM_max_kB %s VmSwap_max_kB %s MemAvailable_min_kB %s", h, s, m
            if (k != "" || g == "") printf " kgsl_page_alloc_max_B %s", k
            if (g != "") printf " mali_total_gpu_mem_max_B %s", g
            printf "\n"}' "$1"
}

host_transfers() {  # the Mac's adb processes in the middle of a transfer (a dry run looks for the fake adb's instead:
  # another session's real transfer does not touch the fake phone)
  if [ "$DRY_RUN" = 1 ]; then
    ps -Ao pid,ppid,etime,comm,args | ME="$ADB" awk 'index($0, ENVIRON["ME"]) && $0 ~ / (push|pull|install|exec-in|exec-out|sync) /'
  else
    ps -Ao pid,ppid,etime,comm,args | awk '$4 ~ /(^|\/)adb$/ && $0 ~ / (push|pull|install|exec-in|exec-out|sync) /'
  fi
}

transfer_check() {  # another session's adb transfer on the Mac, or a busy process of this lane / another lane on the
  # phone: OWN_PROC_RE (this app, a *runner) at all; another app matching PEER_APP_RE only when it is in front or its
  # CPU time grows (a cached, idle app of another lane is recorded, not a stop); system daemons whose names contain
  # install / push are recorded only
  local busy watch='gate|runner|push|sync|install|dex2oat|litert'
  [ -n "$PKG_RE" ] && watch="$watch|$PKG_RE"
  [ -n "$PEER_APP_RE" ] && watch="$watch|$PEER_APP_RE"
  {
    echo "=== transfer check $(ts)"
    echo "host adb processes:"; host_transfers | grep . || echo "  none"
    echo "phone:"; A shell "cat /proc/uptime; df -h /data | tail -1; ps -A -o PID,ETIME,TIME,NAME | grep -i -E '$watch' || echo '  no app / runner / install process'"
    echo "top resumed: $(A shell dumpsys activity activities | grep -m1 topResumedActivity | tr -d '\r')"
    echo "our stage before setup: $(A shell ls -d "${STAGE_DIR:-$STAGE}" 2>&1 | tr -d '\r')"
  } > "$R/x0_transfer_check.txt" 2>&1
  ps_time "$R/x0_ps_a.txt"; sleep 5; ps_time "$R/x0_ps_b.txt"
  others_time "$R/x0_ps_a.txt" "$R/x0_ps_b.txt" "$R/x0_others_time.txt"
  { echo "CPU TIME grown in 5 s (>= 2 s):"; sed 's/^/  /' "$R/x0_others_time.txt"; } >> "$R/x0_transfer_check.txt"
  cat "$R/x0_transfer_check.txt"
  # a host adb transfer stops the run when it names this phone's serial or names no serial (it may reach this phone);
  # a transfer to another phone (-s <other serial>) is recorded only
  local tl
  tl=$(sed -n '/host adb processes:/,/phone:/p' "$R/x0_transfer_check.txt" | grep -v "^host\|^phone\|none" | grep "push\|install\|exec-in\|exec-out\|pull\|sync")
  if [ -n "$tl" ] && { echo "$tl" | grep -q -- " -s $SERIAL" || echo "$tl" | grep -qv -- " -s "; }; then
    return 1
  fi
  if sed -n '/^phone:/,/^top resumed:/p' "$R/x0_transfer_check.txt" | awk '{print $NF}' | grep -qE "^(${OWN_PROC_RE})$"; then
    return 1
  fi
  [ -n "$PEER_APP_RE" ] || return 0
  grep '^top resumed:' "$R/x0_transfer_check.txt" | grep -qE "$PEER_APP_RE" && return 1
  busy=$(awk '{print $1}' "$R/x0_others_time.txt" | grep -E "^($PEER_APP_RE)" || true)
  [ -n "$busy" ] && { echo "busy on the phone: $busy"; return 1; }
  return 0
}

SCORE_PIDS=""
wait_scores() {  # the host scorers still running (never `wait` for every child: the keeper is one of them)
  local p
  for p in $SCORE_PIDS; do wait "$p" 2>/dev/null; done
  SCORE_PIDS=""
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

take_hold() {  # the keeper is its own process (its pid owns the hold, so the hold outlives a chain that dies): start it
  # from a copy in the round dir (an edit of keeper.sh never reaches a running keeper), wait for HOLD_ACQUIRED, then set
  # the clock and the exit trap (clean the phone if needed, release)
  [ -f "$CA/queue_cli.py" ] && [ -f "$CA/hold_cli.py" ] || { echo "CA=$CA has no queue_cli.py / hold_cli.py: no hold" >&2; exit 3; }
  rm -f "$R/RELEASE_HOLD" "$R/HOLD_RELEASED" "$R/GAVE_UP"
  cp "$BENCH/keeper.sh" "$R/keeper.sh.tmp" && mv "$R/keeper.sh.tmp" "$R/keeper.sh"
  echo "===== keeper for $SCRIPT_NAME started by chain pid $$ $(ts) (keeper.sh sha256 $(shasum -a 256 "$R/keeper.sh" | cut -c1-16))" >> "$R/keeper.log"
  nohup bash "$R/keeper.sh" "$R" "$HOLD" "$SCRIPT_NAME" >> "$R/keeper.log" 2>&1 < /dev/null &
  echo "$!" > "$R/keeper.pid"
  echo "keeper pid $(cat "$R/keeper.pid") waiting in line for $HOLD as $SCRIPT_NAME $(ts)"
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
  if [ -n "$cp" ] && kill -0 "$cp" 2>/dev/null && ps -o command= -p "$cp" | grep -qF "$CHAIN_NAME"; then
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
