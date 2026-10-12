#!/bin/bash
# Phase 1: everything that needs no re-pack. One fresh process per condition, sequential, through quiet_wait.
# Usage: nohup bash tools/run_phase1.sh > logs/phase1.log 2>&1 &
set -u
cd "$(dirname "$0")/.." || exit 2
QW="python3 $HOME/code/standup/tools/quiet/quiet_wait.py --max-wait 21600 --"
PY="uv run --no-project --with litert-lm==0.18.0 python -I"
TPL=templates/qwen3_5_main_03d3979f.jinja
model_of() { case "$1" in 2B) echo bundles/Qwen3.5-2B_int8.litertlm;; 0.8B) echo bundles/Qwen3.5-0.8B_int8.litertlm;; *) echo "bad label $1" >&2; exit 2;; esac; }
ORDER=(2B 0.8B)
ts() { date "+%Y-%m-%d %H:%M:%S"; }
run() { local out=""; local prev=""; for a in "$@"; do [ "$prev" = "--out" ] && out="$a"; prev="$a"; done; if [ -n "$out" ] && [ -s "$out" ]; then echo "### SKIP (exists): $out"; return 0; fi; echo "### CMD: $*"; echo "### AT: $(ts)"; $QW "$@"; echo "### EXIT: $? AT: $(ts)"; }

echo "### PHASE1 (RE)START $(ts) pid $$"
echo "### versions"; $PY -c "import importlib.metadata as m; print('litert-lm', m.version('litert-lm'))"
for L in "${ORDER[@]}"; do M=$(model_of "$L"); shasum -a 256 "$M"; done

# --- Issue 1: tool calling, bundle as published (template has no tools branch) and canonical qwen3_5 template
for L in "${ORDER[@]}"; do M=$(model_of "$L")
  for R in 1 2 3; do
    run $PY tools/tc_probe.py --model "$M" --label "$L" --arm bundle --automatic 1 --run $R --out results/tc/${L}_bundle_auto_r${R}.json
  done
  for R in 1 2 3; do
    run $PY tools/tc_probe.py --model "$M" --label "$L" --arm canonical --template $TPL --automatic 1 --thinking off --run $R --out results/tc/${L}_canonical_auto_off_r${R}.json
  done
  run $PY tools/tc_probe.py --model "$M" --label "$L" --arm canonical --template $TPL --automatic 0 --thinking off --run 1 --out results/tc/${L}_canonical_noauto_off_r1.json
  run $PY tools/tc_probe.py --model "$M" --label "$L" --arm canonical --template $TPL --automatic 1 --thinking unset --run 1 --out results/tc/${L}_canonical_auto_unset_r1.json
done

# --- Issue 2a: system-prompt switch on one Engine vs the control
for L in "${ORDER[@]}"; do M=$(model_of "$L")
  for R in 1 2 3; do
    run $PY tools/leak_probe.py sysswitch --model "$M" --label "$L" --mode shared --run $R --out results/leak/${L}_sysswitch_shared_r${R}.json
    run $PY tools/leak_probe.py sysswitch --model "$M" --label "$L" --mode fresh --run $R --out results/leak/${L}_sysswitch_fresh_r${R}.json
  done
done

# --- Issue 2b: 20 questions, one Engine (two rounds) vs one process per question
for L in "${ORDER[@]}"; do M=$(model_of "$L")
  for R in 1 2 3; do
    run $PY tools/leak_probe.py qset --model "$M" --label "$L" --mode shared --run $R --out results/leak/${L}_qset_shared_r${R}.json
  done
done
for L in "${ORDER[@]}"; do M=$(model_of "$L")
  for Q in $(seq 0 19); do
    run $PY tools/leak_probe.py qset --model "$M" --label "$L" --mode fresh --question $Q --run 1 --out results/leak/${L}_qset_fresh_q$(printf %02d $Q).json
  done
done
echo "### PHASE1 DONE $(ts)"
