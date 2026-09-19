#!/usr/bin/env bash
# Bonsai 2 27B float export (text-only, Hadamard ops in-graph, Qwen3.5 hybrid patch), lightweight conversion, stops at model.tflite.
cd ~/code/litertlm-convert
export PYTHONPATH=qwen35_work/litert-torch-qwen35 QWEN35_PREFILL_LADDER=1024,256,64,16,4,1 CACHE_LENGTH=4096
echo "start $(date +%T)"
/usr/bin/time -l ~/venvs/ltconv040dev/bin/python3 bonsai2_work/export_driver.py bonsai2_work/hf_bf16 bonsai2_work/out_fp \
  --keep_temporary_files=True --experimental_lightweight_conversion=True > bonsai2_work/out_fp/export.log 2>&1 &
PID=$!
while kill -0 $PID 2>/dev/null; do
  P=$(pgrep -f "export_driver.py bonsai2_work/hf_bf16" | tail -1)
  R=$(ps -o rss= -p $P 2>/dev/null | tr -d ' '); F=$(vm_stat | awk '/Pages free/{gsub("\\.","",$3); printf "%.1f", $3*16384/1e9}')
  echo "$(date +%T) rss_GB=$(( ${R:-0} / 1048576 )) free_GB=$F stage=$(grep -v '^W0' bonsai2_work/out_fp/export.log | grep -E '^\(' | tail -1 | cut -c1-90)"
  sleep 60
done
echo "end $(date +%T)"; grep -E "maximum resident|peak memory|real " bonsai2_work/out_fp/export.log
