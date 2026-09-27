#!/bin/zsh
# Per-graph CPU (XNNPACK) latency on the connected Android device, with the same
# benchmark_model binary the LiteRT CLI uses (`litert benchmark --android --cpu`).
#   bench_android.sh <graph.tflite> [...]
# Prints the benchmark's timing and peak-memory lines per graph; pushes each model,
# runs it, and removes it again so the device does not keep the 2 GiB files.
set -e
BIN=${BIN:-~/.cache/litert-cli/binaries/arm64-v8a/benchmark_model}
THREADS=${THREADS:-6}
RUNS=${RUNS:-3}
R=/data/local/tmp/bonsai_size_bench
adb shell mkdir -p $R
adb push "$BIN" $R/benchmark_model >/dev/null
adb shell chmod +x $R/benchmark_model
adb shell getprop ro.product.model
for f in "$@"; do
  b=$(basename "$f")
  echo "=== $b ($(du -h "$f" | cut -f1)) threads=$THREADS runs=$RUNS"
  adb push "$f" $R/$b >/dev/null
  adb shell "$R/benchmark_model --graph=$R/$b --use_xnnpack=true --num_threads=$THREADS \
      --num_runs=$RUNS --warmup_runs=1 --report_peak_memory_footprint=true" 2>&1 \
    | grep -E "Replacing|Model initialization|Warmup \(first\)|Inference \(avg\)|Inference \(min\)|Inference \(max\)|Init footprint|Overall footprint|Peak memory" || true
  adb shell rm -f $R/$b
done
