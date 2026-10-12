#!/bin/bash
# Phase 2 (after phase 1): the workaround re-packs (issue 1d) and the 4B mixed_int4 state probes (card number for the 4B card).
set -u
cd "$(dirname "$0")/.." || exit 2
QW="python3 $HOME/code/standup/tools/quiet/quiet_wait.py --max-wait 21600 --"
PY="uv run --no-project --with litert-lm==0.18.0 python -I"
ts() { date "+%Y-%m-%d %H:%M:%S"; }
run() { local out=""; local prev=""; for a in "$@"; do [ "$prev" = "--out" ] && out="$a"; prev="$a"; done; if [ -n "$out" ] && [ -s "$out" ]; then echo "### SKIP (exists): $out"; return 0; fi; echo "### CMD: $*"; echo "### AT: $(ts)"; $QW "$@"; echo "### EXIT: $? AT: $(ts)"; }
echo "### PHASE2 waiting for phase 1 $(ts)"
until grep -q "PHASE1 DONE" logs/phase1.log 2>/dev/null; do sleep 20; done
echo "### PHASE2 (RE)START $(ts) pid $$"
shasum -a 256 workaround/Qwen3.5-2B_int8_qwen3type.litertlm workaround/Qwen3.5-0.8B_int8_qwen3type.litertlm bundles/Qwen3.5-4B_mixed_int4.litertlm
# --- 1d: workaround re-pack (llm_model_type qwen3 + models/qwen3 template with the JSON instruction)
for L in 2B 0.8B; do M=workaround/Qwen3.5-${L}_int8_qwen3type.litertlm
  for R in 1 2 3; do
    run $PY tools/tc_probe.py --model "$M" --label "${L}_qwen3type" --arm workaround --automatic 1 --thinking off --run $R --out results/tc/${L}_workaround_auto_off_r${R}.json
  done
  run $PY tools/tc_probe.py --model "$M" --label "${L}_qwen3type" --arm workaround --automatic 0 --thinking off --run 1 --out results/tc/${L}_workaround_noauto_off_r1.json
done
# --- 4B mixed_int4 (Hub main 37209c42…): state probes for the 4B card note
M=bundles/Qwen3.5-4B_mixed_int4.litertlm; L=4Bq4
for R in 1 2 3; do
  run $PY tools/leak_probe.py qset --model "$M" --label "$L" --mode shared --run $R --out results/leak/${L}_qset_shared_r${R}.json
done
for R in 1 2 3; do
  run $PY tools/leak_probe.py sysswitch --model "$M" --label "$L" --mode shared --run $R --out results/leak/${L}_sysswitch_shared_r${R}.json
  run $PY tools/leak_probe.py sysswitch --model "$M" --label "$L" --mode fresh --run $R --out results/leak/${L}_sysswitch_fresh_r${R}.json
done
for Q in $(seq 0 19); do
  run $PY tools/leak_probe.py qset --model "$M" --label "$L" --mode fresh --question $Q --run 1 --out results/leak/${L}_qset_fresh_q$(printf %02d $Q).json
done
echo "### PHASE2 DONE $(ts)"
