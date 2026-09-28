#!/bin/bash
# S26 per-signature benchmark (LiteRT benchmark_model in /data/local/tmp/litert-cli). Usage:
#   audio8_tts_work/device_bench.sh <local.tflite> <device-name> <signature|-> <cpu|gpu> [num_runs] [num_threads]
# Before every speed leg it waits (up to 10 min) until no CPU policy is frequency-capped
# (scaling_max_freq == cpuinfo_max_freq on every policy) and records the caps + thermal status.
# Assumes the S26 hold is already owned by the caller.
set -u
export ANDROID_SERIAL=${ANDROID_SERIAL:-RFGL80R6A6H}
LOCAL=$1; NAME=$2; SIG=$3; BE=$4; RUNS=${5:-20}; THR=${6:-4}
D=/data/local/tmp/audio8_gate; BM=/data/local/tmp/litert-cli/benchmark_model
adb shell "test -f $D/$NAME" 2>/dev/null || adb push "$LOCAL" "$D/$NAME" >/dev/null 2>&1
freq_state() {  # prints "policyN max/cinfo" per policy; returns 1 if any policy is capped
  local capped=0
  while read -r p mx ci; do
    echo -n "$p $mx/$ci "; [ "$mx" != "$ci" ] && capped=1
  done < <(adb shell 'for p in /sys/devices/system/cpu/cpufreq/policy*; do echo $(basename $p) $(cat $p/scaling_max_freq) $(cat $p/cpuinfo_max_freq); done' | tr -d '\r')
  echo; return $capped
}
for i in $(seq 1 60); do freq_state > /tmp/freq_$$.txt && break; sleep 10; done
echo "### $NAME sig=$SIG $BE runs=$RUNS thr=$THR | freq: $(cat /tmp/freq_$$.txt) | thermal: $(adb shell 'dumpsys thermalservice 2>/dev/null | grep -m1 -i "status"' | tr -d '\r')"
rm -f /tmp/freq_$$.txt
SIGF=""; [ "$SIG" != "-" ] && SIGF="--signature_to_run_for=$SIG"
if [ "$BE" = gpu ]; then BEF="--use_gpu=true"; else BEF="--num_threads=$THR"; fi
adb shell "cd $D && $BM --graph=$NAME $SIGF $BEF --num_runs=$RUNS --warmup_runs=3 2>&1" \
  | grep -E "Replacing|not supported|Inference \(avg\)|Inference \(min\)|footprint|ERROR|Failed" | grep -v "^ERROR: Following" | head -14 | cut -c1-200
