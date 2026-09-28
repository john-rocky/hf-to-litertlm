#!/bin/zsh
# Round 6: start the Mac measurements only after this round's own checks have finished (they would contend).
cd "$(dirname $0)/.."
while pgrep -f 'r6_run_checks.sh|r6_run_gpu_checks.sh' > /dev/null; do sleep 10; done
echo "# $(date '+%F %T %Z') own checks finished; bench starts"
python3 -B -u scripts/r6_mac_bench.py
echo "# $(date '+%F %T %Z') bench exit $?"
