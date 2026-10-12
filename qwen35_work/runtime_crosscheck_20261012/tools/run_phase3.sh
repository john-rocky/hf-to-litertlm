#!/bin/bash
# Phase 3 (after phase 2): digit test "What is 17 times 23?" at the bundle default, fresh CPU process x3, for the card lines.
set -u
cd "$(dirname "$0")/.." || exit 2
QW="python3 $HOME/code/standup/tools/quiet/quiet_wait.py --max-wait 21600 --"
PY="uv run --no-project --with litert-lm==0.18.0 python -I"
ts() { date "+%Y-%m-%d %H:%M:%S"; }
run() { local out=""; local prev=""; for a in "$@"; do [ "$prev" = "--out" ] && out="$a"; prev="$a"; done; if [ -n "$out" ] && [ -s "$out" ]; then echo "### SKIP (exists): $out"; return 0; fi; echo "### CMD: $*"; echo "### AT: $(ts)"; $QW "$@"; echo "### EXIT: $? AT: $(ts)"; }
echo "### PHASE3 waiting for phase 2 $(ts)"
until grep -q "PHASE2 DONE" logs/phase2.log 2>/dev/null; do sleep 20; done
echo "### PHASE3 (RE)START $(ts) pid $$"
shasum -a 256 bundles/Qwen3.5-4B_int8.litertlm bundles/Qwen3.5-4B_mixed_int4.litertlm bundles/Qwen3.5-2B_int8.litertlm
for L in 4B_int8 4B_mixed_int4 2B_int8; do M=bundles/Qwen3.5-${L}.litertlm
  for R in 1 2 3; do
    run $PY tools/digit_probe.py --model "$M" --label "$L" --run $R --out results/digit/${L}_r${R}.json
  done
done
echo "### PHASE3 DONE $(ts)"
