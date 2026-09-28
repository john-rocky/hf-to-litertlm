#!/bin/zsh
# Round 6: after the cache-no cells, the same decision with the runtime's default mode (a cache folder next to the
# process's choice: out/r6_bench_cache/<variant>_<backend>/, emptied before run 0 = cold; runs 1-2 warm).
cd "$(dirname $0)/.."
while pgrep -f 'r6_bench_after_checks.sh' > /dev/null; do sleep 10; done
echo "# $(date '+%F %T %Z') disk-cache cells start"
python3 -B -u scripts/r6_mac_bench.py --cells fp16:cpu:disk,v7c:cpu:disk,fp16:gpu:disk,v7c:gpu:disk \
  --out results/r6/mac_bench_disk.json --first-wait 300 --gpu-first-rest 120 --gpu-rest 120
echo "# $(date '+%F %T %Z') disk-cache cells exit $?"
