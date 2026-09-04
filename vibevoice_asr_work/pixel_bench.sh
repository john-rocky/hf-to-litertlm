#!/bin/sh
# Device LM speed rows: litert_lm_advanced_main --benchmark, 213-word text prompt, CPU and GPU.
# usage: pixel_bench.sh <serial> <bundle-basename> <tag> <libs-dir-on-phone>
S=$1; B=$2; TAG=$3; L=${4:-/data/local/tmp/g41_gate}; D=/data/local/tmp/vv_gate
for BE in cpu gpu; do
  adb -s $S shell "cd $D && LD_LIBRARY_PATH=$L:$D ./litert_lm_advanced_main --backend=$BE --sampler_backend=cpu --model_path=$D/$B --max_num_tokens=2048 --input_prompt_file=$D/long_prompt.txt --benchmark > $D/bench_${TAG}_$BE.log 2>&1 < /dev/null; echo EXIT=\$?"
  adb -s $S pull $D/bench_${TAG}_$BE.log bench/pixel_${TAG}_$BE.log >/dev/null
  sleep 60
done
# audio prompt under --benchmark (does the report include the audio encoder?)
adb -s $S shell "cd $D && LD_LIBRARY_PATH=$L:$D ./litert_lm_advanced_main --backend=cpu --audio_backend=cpu --sampler_backend=cpu --model_path=$D/$B --max_num_tokens=2048 --input_prompt='[audio:$D/clip02.wav] This is a 12.49 seconds audio, please transcribe it.' --benchmark > $D/bench_${TAG}_audio_cpu.log 2>&1 < /dev/null; echo EXIT=\$?"
adb -s $S pull $D/bench_${TAG}_audio_cpu.log bench/pixel_${TAG}_audio_cpu.log >/dev/null
echo BENCH_DONE
