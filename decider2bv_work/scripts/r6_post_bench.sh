#!/bin/zsh
# Round 6: after both bench series: re-run the contended cache-no rows, refresh the public fixtures (image source text
# of the v2 rows), re-run the reference checks (CPU + bit identity, then GPU), run the card's usage and the runtime
# example once, and rebuild the performance summary. Sequential, one heavy process at a time.
cd "$(dirname $0)/.."
while pgrep -f 'r6_bench_disk_after.sh' > /dev/null; do sleep 5; done
echo "# $(date '+%F %T %Z') redo contended"
python3 -B -u scripts/r6_mac_bench.py --redo-contended --first-wait 120 --gpu-first-rest 0
python3 -B -u scripts/r6_mac_bench.py --redo-contended --runs 4 --cells fp16:cpu:disk,v7c:cpu:disk,fp16:gpu:disk,v7c:gpu:disk \
  --out results/r6/mac_bench_disk.json --first-wait 120 --gpu-first-rest 120 --gpu-rest 120
# (--runs 4: a cache-folder cell's run 0 is the cold run that writes the cache; runs 1-3 are the three warm runs)
echo "# $(date '+%F %T %Z') public fixtures"
python3 -B scripts/r6_make_public_fixtures.py
echo "# $(date '+%F %T %Z') checks"
scripts/r6_run_checks.sh
echo "# $(date '+%F %T %Z') usage + runtime example"
out/venv-ref/bin/python -B scripts/r6_usage_test.py --bundle out/bundle_r4/v7c/decider-2b-vision_v7c.litertlm > logs/r6/usage_test_v7c.log 2>&1; tail -1 logs/r6/usage_test_v7c.log
out/venv-readout/bin/python -B publish/reference/runtime_example.py --bundle out/bundle_r4/v7c/decider-2b-vision_v7c.litertlm \
  --image publish/reference/fixtures/images/game_pong_atari_up_160x210.png \
  --context "You play Pong (Atari) and control the right paddle. Move the paddle so the ball hits it; the ball bounces off paddles and walls. Missing the ball loses a point. The image shows the current game screen." \
  --question "What should you do right now?" --options "move paddle up" "move paddle down" "stay" --cache-dir :nocache \
  > results/r6/runtime_example_v7c.txt 2> logs/r6/runtime_example_v7c.stderr < /dev/null
echo "runtime_example exit $?"; cat results/r6/runtime_example_v7c.txt
python3 -B scripts/r6_perf_table.py
echo "# $(date '+%F %T %Z') gpu checks"
scripts/r6_run_gpu_checks.sh
echo "# $(date '+%F %T %Z') post-bench done"
