#!/bin/zsh
# Round 6 (informational): the reference's GPU rule (decoder on ai-edge-litert CompiledModel GPU = Metal on this Mac,
# one padded prefill chunk per slot; embedder + vision on CPU) on the public fixtures, v7c then fp16, minimal venv.
#   scripts/r6_run_gpu_checks.sh   (from decider2bv_work/; waits for scripts/r6_run_checks.sh to finish)
cd "$(dirname $0)/.."
while pgrep -f r6_run_checks.sh > /dev/null; do sleep 5; done
for v in v7c fp16; do
  echo "# $(date '+%F %T %Z') gpu check $v"
  out/venv-ref/bin/python -B publish/reference/check_fixtures.py --bundle out/bundle_r4/$v/decider-2b-vision_$v.litertlm \
    --backend gpu --cache-dir out/ref_cache --out results/r6/check_${v}_gpu.json > logs/r6/check_${v}_gpu.log 2> logs/r6/check_${v}_gpu.stderr
  echo "# $(date '+%F %T %Z') gpu check $v exit $?"
done
echo "# $(date '+%F %T %Z') done"
