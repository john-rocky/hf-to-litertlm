#!/bin/sh
# S26 LM speed rows (--benchmark with fixed fake token counts 256/256 — this binary honours
# --benchmark_{prefill,decode}_tokens (probe: bench/s26_probe_faketokens.log); a real text prompt
# stops after 1 token because the bare streaming template makes the model emit <|text_chunk_end|>) + engine-load-only runs, for both
# bundles and both LM backends.  Each speed leg waits for SKIN < 40 C and an uncapped CPU
# (scaling_max_freq == cpuinfo_max_freq on every policy), records both, rests 180 s after.
# usage: s26_bench.sh <serial>   (device dir /data/local/tmp/vvs_gate, libs docling_gate)
S=$1; D=/data/local/tmp/vvs_gate; L=/data/local/tmp/docling_gate; OUT=vibevoice_asr_streaming_work/bench
mkdir -p $OUT
adb -s $S shell "cp /data/local/tmp/vv_gate/long_prompt.txt $D/ 2>/dev/null; ls -la $D/long_prompt.txt"
ready() {  # prints "SKIN=<c> cap=<policy6 max>/<cpuinfo max>"; returns 0 when uncapped and cool
  adb -s $S shell 'skin=$(dumpsys thermalservice | grep "mName=SKIN" | head -1 | sed "s/.*mValue=\([0-9.]*\).*/\1/"); ok=1; caps=""; for p in /sys/devices/system/cpu/cpufreq/policy*; do a=$(cat $p/scaling_max_freq); b=$(cat $p/cpuinfo_max_freq); caps="$caps $(basename $p)=$a/$b"; [ "$a" = "$b" ] || ok=0; done; hot=$(echo "$skin >= 40" | bc 2>/dev/null); [ "$hot" = "1" ] && ok=0; echo "SKIN=$skin$caps ok=$ok"'
}
wait_ready() {
  i=0; while [ $i -lt 60 ]; do r=$(ready); echo "$r"; echo "$r" | grep -q "ok=1" && return 0; i=$((i+1)); sleep 20; done; echo "NOT READY after 20 min"; return 1
}
for B in VibeVoice-ASR-Streaming-1.5B.litertlm VibeVoice-ASR-Streaming-1.5B_int4.litertlm; do
  case $B in *_int4*) T=int4;; *) T=wi8;; esac
  for BE in cpu gpu; do
    echo "=== leg $T $BE $(date)"; wait_ready | tail -1 | tee $OUT/s26_${T}_${BE}_ready.txt
    adb -s $S shell "cd $D && LD_LIBRARY_PATH=$L:$D ./litert_lm_advanced_main --backend=$BE --sampler_backend=cpu --model_path=$D/$B --max_num_tokens=2048 --input_prompt_file=$D/long_prompt.txt --benchmark --benchmark_prefill_tokens=256 --benchmark_decode_tokens=256 > $D/bench_${T}_$BE.log 2>&1 < /dev/null; echo EXIT=\$?"
    adb -s $S pull $D/bench_${T}_$BE.log $OUT/s26_${T}_$BE.log >/dev/null
    grep -i "Processed\|prefill speed\|decode speed\|first token\|Peak private" $OUT/s26_${T}_$BE.log | cut -c1-120
    # engine-load-only (multi_turns with an empty first line -> exits right after load)
    adb -s $S shell "cd $D && t0=\$(date +%s.%N); printf '\n' | LD_LIBRARY_PATH=$L:$D ./litert_lm_advanced_main --multi_turns=true --backend=$BE --audio_backend=cpu --sampler_backend=cpu --model_path=$D/$B --max_num_tokens=2048 > /dev/null 2> $D/load_${T}_$BE.log; t1=\$(date +%s.%N); echo LOAD_ONLY_S=\$(echo \"\$t1 - \$t0\" | bc)" | tee $OUT/s26_${T}_${BE}_load.txt
    sleep 180
  done
done
echo S26_BENCH_DONE
