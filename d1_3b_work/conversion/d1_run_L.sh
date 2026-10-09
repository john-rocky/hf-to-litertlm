#!/usr/bin/env bash
# d1_run_L.sh <source> <tag> <L>: the per-window chain of the d1-3B row graphs, one command per L.
#   source = tiny | a snapshot directory holding LiquidAI/d1-3B's model.safetensors; tag = output prefix (tiny, real);
#   L = the window (64 .. 4096).
# Steps (round 2 ran each on the tiny model; ROUND2.md):
#   graph_check  d1_graph_check.py --L L      torch: D1Prefill vs the provider (bar 1e-5), pad content bit-equal, tree
#   export       d1_export.py --L L           litert_torch -> exports/{tag}_rowprefill_L{L}_fp32.tflite + op scan
#   prelint      tools/rank3_prelint.py       rank <= 3 forms (attention BMM rows are expected, see ROUND2.md)
#   cpu          d1_check.py --accel cpu      CompiledModel CPU vs the torch rows
#   gpu_f32 / gpu_default                     Metal, float32 / default precision, inside the GPU window (soft: the fp32
#                                             file keeps a float embedding table, which Kev's Metal runs refused at
#                                             0.8B (the Kev-0.8B LiteRT conversion); a failure is recorded and the chain goes on)
#   storage_v2 / storage_v3 / storage_v1      d1_storage.py: fp16 FC + int8 table / int8 table only / int8 FC (dynamic,
#                                             integer compute) + int8 table (round 5)
#   cpu_<v> / gpu_f32_<v>                     v2 and v3 on the CPU and on Metal float32, compared with the fp32 file
#   gpu_default_v2                            the candidate at Metal default precision (soft; the fp16 norm note in REPRODUCE.md)
#   cpu_v1_wi8fc / gpu_f32_v1_wi8fc           v1 on the CPU (the S26 CPU form of the round-4 estimate) and on Metal
#                                             float32 (soft: int8 FC weights on a GPU run as float GEMMs; not the
#                                             form v1 is for)
# CPU-heavy steps wait for a closed measurement window (quiet_wait.py); GPU steps take it (quiet_hold.py, one window per
# run). The chain stops at the first failing step except the soft ones; every step logs to logs/{tag}_L{L}_<step>.log;
# existing outputs are never overwritten (the scripts refuse), so a rerun of a finished step needs its outputs moved away,
# or D1_RESUME=1: a step whose result json exists is skipped (and said so), the rest run (a chain that stopped, or a step
# added later). D1_NORM_SCALE=<json or file> runs graph_check and export with that per-site norm pre-scale
# (d1_prefill_graph.apply_norm_scale; use its own tag, e.g. realns); every later step reads the exported file.
# Round 6b (the real weights):
#   D1_REFERENCE=<results/reference_<tag>.json> D1_TABLE=<readout_table.safetensors> [D1_REFERENCE_HIDDEN=<npz>]
#   [D1_NEAR_TIE_FROM=<results/reference_summary.json>]: every d1_check.py step also runs the reference's question rows
#   that fit L and its red arms (d1_check.py --reference); the float32 file on the CPU and v2 on Metal float32 stop the
#   chain when the bar (near ties apart) or a red arm fails (--stop-on-bar, the stop conditions).
#   D1_GRAPH_BAR=<x>: graph_check's bar (default the script's 1e-5).
#   D1_UNTIL=<step>: stop (exit 0) after that step (run the rest later with D1_RESUME=1).
#   D1_MEM_GIB=<n> [D1_MEM_WAIT_MIN=<m> (30)]: before every step but prelint, wait until vm_stat's free + inactive +
#   speculative pages are >= n GiB and the swap in use did not grow over 60 s (logs/{tag}_L{L}_memgate.log); after m
#   minutes without that, the chain stops (exit 3) so that the Mac's memory can be freed.
#   gpu_default_v3 / gpu_default_v1_wi8fc (soft): v3 and v1 at Metal default precision too (18 steps).
# Round 6c (the long buckets):
#   D1_FORMS=<comma list of fp32, fp32gpu, v2, v3, v1> (default all = the 18 steps): fp32 = the float32 file's CPU
#   gate (required: the variants compare with its CPU run), fp32gpu = the float32 file on Metal float32 / default,
#   v2 / v3 / v1 = build that variant and run its steps. `D1_FORMS=fp32,v2` = graph_check, export, prelint, cpu,
#   storage_v2, cpu_v2, gpu_f32_v2 (stops on the bar), gpu_default_v2 (soft).
#   D1_CPU_ROWS=smallest-head [D1_CPU_HEAD=20]: the CPU steps run the reference questions whose smallest bucket is L +
#   the first D1_CPU_HEAD questions + the red arms (d1_check.py --ref-subset); the Metal steps run every question
#   that fits L.
#   D1_GRAPH_ROWS=2: graph_check's check 2 on n = L and L/2+1 only (d1_graph_check.py --rows 2).
#   D1_RETIRE_FP32=1: a last step retire_fp32 deletes the float32 file after the gate (d1_export.py --retire: re-hash
#   against the export record, results/{tag}_L{L}_fp32_deleted.json, ledger results/{tag}_files.json).
#   D1_WIN_PREFIX=<label prefix> (default d1a): the GPU window label is <prefix>-{tag}-L{L}.
#   D1_MEM_SWAP_S=<s> (default 60): how long memgate watches the swap for growth.
#   D1_EMBEDS=1 D1_EMBED_TABLE=<embed_table.safetensors> (with D1_FORMS=fp32,v2e): the embeds variant
#   (D1PrefillEmbeds): no graph_check (the export guard takes the tag's passing check of L, whose check 4 is the
#   embeds graph), export --embeds, prelint, cpu (the host's float32 rows of the bfloat16 table), storage_v2e (fp16
#   FC, no table), cpu / gpu_f32 (stops on the bar) / gpu_default (soft) of v2e, retire_fp32 --embeds; every check
#   compares with the ids float32 file's CPU run of the same L (cache/{tag}/litert_{tag}_rowprefill_L{L}_fp32_cpu.npz);
#   logs and results carry `_embeds` (logs/{tag}_embeds_L{L}_<step>.log).
# Round 8:
#   D1_VCPU_ROWS=all | smallest-head (with D1_REFERENCE): the row set of the variants' CPU steps (cpu_v2 / cpu_v3 /
#   cpu_v2e) when it differs from D1_CPU_ROWS (which stays the float32 file's CPU set); all = every question that fits
#   L + the red arms. Unset = D1_CPU_ROWS, as before.
#   D1_QW=<command> / D1_QH=<command> (required): what wraps the CPU-heavy steps and the GPU steps (D1_QH takes a
#   <label> before its --); every step runs with D1_STEP_LABEL=<window prefix>-<tag>[-embeds]-L<L>-<step> in
#   its environment, so a wrapper can take one window per step (round 8: cache/real/image/r8_hold.sh =
#   quiet_hold.py d1a-r8-...-<step>, the order taken by the flock).
# Round 10:
#   D1_COMPARE=<npz> (K-relative): the run every check step compares with, in place of the ids float32 file's CPU run of
#   the same L (a new bucket has none: L128 compares with cache/real/litert_real_rowprefill_L256_fp32_cpu.npz; the check
#   rows then differ in n and are not compared, the questions are, by hsel/<id>/<qid>; d1_check.py round 10).
set -euo pipefail
[ $# -eq 3 ] || { echo "usage: $0 <tiny|snapshot dir> <tag> <L>" >&2; exit 2; }
SRC=$1; TAG=$2; L=$3
K=$(cd "$(dirname "$0")/.." && pwd)
REPO=$(cd "$K/.." && pwd)
PY=${D1_PY:-$HOME/venvs/lt094dev/bin/python}
QW=${D1_QW:?set D1_QW: a command that runs what follows its --}
QH=${D1_QH:?set D1_QH: a command that takes a label and runs what follows its --}
S=$K/scripts
LOG=$K/logs
KIND=""
EXPARG=()
EMBARG=()
if [ "${D1_EMBEDS:-0}" = 1 ]; then
  [ -n "${D1_EMBED_TABLE:-}" ] || { echo "D1_EMBEDS needs D1_EMBED_TABLE" >&2; exit 2; }
  KIND=_embeds
  EXPARG=(--embeds)
  EMBARG=(--embed-table "$D1_EMBED_TABLE")
fi
F=exports/${TAG}_rowprefill${KIND}_L${L}_fp32.tflite
CMP=${D1_COMPARE:-cache/${TAG}/litert_${TAG}_rowprefill_L${L}_fp32_cpu.npz}
if [ -n "$KIND" ] && [ ! -e "$K/$CMP" ]; then   # the embeds chain compares with an ids run made before it
  echo "no compare npz $CMP (set D1_COMPARE)" >&2; exit 2
fi
WIN="${D1_WIN_PREFIX:-d1a}-${TAG}${KIND//_/-}-L${L}"
FORMS=",${D1_FORMS:-fp32,fp32gpu,v2,v3,v1},"
has() { case "$FORMS" in *",$1,"*) return 0 ;; esac; return 1; }
has fp32 || { echo "D1_FORMS must list fp32 (the variants compare with its CPU run)" >&2; exit 2; }

RESUME=${D1_RESUME:-0}
NSARG=()
[ -n "${D1_NORM_SCALE:-}" ] && NSARG=(--norm-scale "$D1_NORM_SCALE")
GBAR=()
[ -n "${D1_GRAPH_BAR:-}" ] && GBAR=(--bar "$D1_GRAPH_BAR")
[ -n "${D1_GRAPH_ROWS:-}" ] && GBAR+=(--rows "$D1_GRAPH_ROWS")
REFARG=()
if [ -n "${D1_REFERENCE:-}" ]; then
  [ -n "${D1_TABLE:-}" ] || { echo "D1_REFERENCE needs D1_TABLE" >&2; exit 2; }
  REFARG=(--reference "$D1_REFERENCE" --table "$D1_TABLE")
  [ -n "${D1_REFERENCE_HIDDEN:-}" ] && REFARG+=(--reference-hidden "$D1_REFERENCE_HIDDEN")
  [ -n "${D1_NEAR_TIE_FROM:-}" ] && REFARG+=(--near-tie-from "$D1_NEAR_TIE_FROM")
fi
STOPBAR=()
[ -n "${D1_REFERENCE:-}" ] && STOPBAR=(--stop-on-bar)
CPUREF=()
if [ -n "${D1_REFERENCE:-}" ] && [ -n "${D1_CPU_ROWS:-}" ]; then
  CPUREF=(--ref-subset "$D1_CPU_ROWS" --ref-head "${D1_CPU_HEAD:-20}")
fi
VCPUREF=(${CPUREF[@]+"${CPUREF[@]}"})
if [ -n "${D1_REFERENCE:-}" ] && [ -n "${D1_VCPU_ROWS:-}" ]; then
  VCPUREF=(--ref-subset "$D1_VCPU_ROWS" --ref-head "${D1_CPU_HEAD:-20}")
fi
MEMLOG=$LOG/${TAG}${KIND}_L${L}_memgate.log
# memgate <step>: the memory condition of round 6b (header); no-op unless D1_MEM_GIB is set
memgate() {
  [ -n "${D1_MEM_GIB:-}" ] || return 0
  local name=$1 t0 now s0 s1 avail
  t0=$(date +%s)
  while :; do
    s0=$(sysctl -n vm.swapusage | awk '{for (i = 1; i <= NF; i++) if ($i == "used") { v = $(i + 2); sub(/M$/, "", v); print v }}')
    sleep "${D1_MEM_SWAP_S:-60}"
    s1=$(sysctl -n vm.swapusage | awk '{for (i = 1; i <= NF; i++) if ($i == "used") { v = $(i + 2); sub(/M$/, "", v); print v }}')
    avail=$(vm_stat | awk '/page size of/ { ps = $8 } /^Pages free/ { f = $3 } /^Pages inactive/ { i = $3 } /^Pages speculative/ { s = $3 }
      END { gsub(/\./, "", f); gsub(/\./, "", i); gsub(/\./, "", s); printf "%.1f", (f + i + s) * ps / 1073741824 }')
    now=$(date +%s)
    if awk -v a="$avail" -v g="$D1_MEM_GIB" -v s0="$s0" -v s1="$s1" 'BEGIN { exit !(a >= g && s1 <= s0) }'; then
      echo "[$(date +%H:%M:%S)] $name go: avail ${avail} GiB, swap used ${s0} -> ${s1} MB, waited $((now - t0)) s" >> "$MEMLOG"
      return 0
    fi
    echo "[$(date +%H:%M:%S)] $name wait: avail ${avail} GiB (need ${D1_MEM_GIB}), swap used ${s0} -> ${s1} MB" >> "$MEMLOG"
    if [ $((now - t0)) -ge $((${D1_MEM_WAIT_MIN:-30} * 60)) ]; then
      echo "MEMGATE TIMEOUT $name after $((now - t0)) s (log: $MEMLOG)"; exit 3
    fi
  done
}
R=results/${TAG}_rowprefill${KIND}_L${L}
SOFT_FAILED=()
SKIPPED=()
# step|soft <name> <result json (K-relative) that marks the step done> <command...>
until_check() {
  if [ -n "${D1_UNTIL:-}" ] && [ "$D1_UNTIL" = "$1" ]; then
    echo "[$(date +%H:%M:%S)] stop after $1 (D1_UNTIL); continue with D1_RESUME=1"; exit 0
  fi
}
step() {
  local name=$1 done=$2; shift 2
  local log=$LOG/${TAG}${KIND}_L${L}_${name}.log
  if [ "$RESUME" = 1 ] && [ -e "$K/$done" ]; then echo "[$(date +%H:%M:%S)] skip $name ($done exists)"; SKIPPED+=("$name"); until_check "$name"; return; fi
  case "$name" in prelint | retire_fp32) ;; *) memgate "$name" ;; esac
  echo "[$(date +%H:%M:%S)] $name"
  if ! D1_STEP_LABEL="${D1_WIN_PREFIX:-d1a}-${TAG}${KIND//_/-}-L${L}-${name}" "$@" > "$log" 2>&1; then
    echo "FAIL $name (log: $log)"; tail -5 "$log"; exit 1
  fi
  until_check "$name"
}
soft() {
  local name=$1 done=$2; shift 2
  local log=$LOG/${TAG}${KIND}_L${L}_${name}.log
  if [ "$RESUME" = 1 ] && [ -e "$K/$done" ]; then echo "[$(date +%H:%M:%S)] skip $name ($done exists)"; SKIPPED+=("$name"); until_check "$name"; return; fi
  memgate "$name"
  echo "[$(date +%H:%M:%S)] $name (soft)"
  if ! D1_STEP_LABEL="${D1_WIN_PREFIX:-d1a}-${TAG}${KIND//_/-}-L${L}-${name}" "$@" > "$log" 2>&1; then
    echo "SOFT FAIL $name (log: $log), going on"; tail -5 "$log"; SOFT_FAILED+=("$name")
  fi
  until_check "$name"
}

cd "$K"
FCMP=()
if [ -z "$KIND" ]; then
  step graph_check "results/${TAG}_graph_check_L${L}.json" \
    "$QW" -- "$PY" "$S/d1_graph_check.py" --source "$SRC" --tag "$TAG" --L "$L" ${NSARG[@]+"${NSARG[@]}"} \
    ${GBAR[@]+"${GBAR[@]}"}
else
  FCMP=(--compare "$CMP")     # the embeds float32 file against the ids float32 file's CPU run
fi
step export "results/${TAG}_export${KIND}_L${L}.json" \
  "$QW" -- "$PY" "$S/d1_export.py" --source "$SRC" --tag "$TAG" --L "$L" ${EXPARG[@]+"${EXPARG[@]}"} \
  ${NSARG[@]+"${NSARG[@]}"}
step prelint "results/${TAG}${KIND}_rank3_prelint_L${L}.json" \
  "$PY" "$REPO/tools/rank3_prelint.py" "$K/$F" --json "$K/results/${TAG}${KIND}_rank3_prelint_L${L}.json"
step cpu "${R}_fp32_cpu_check.json" "$QW" -- "$PY" "$S/d1_check.py" --tflite "$F" --accel cpu \
  ${FCMP[@]+"${FCMP[@]}"} ${EMBARG[@]+"${EMBARG[@]}"} ${REFARG[@]+"${REFARG[@]}"} ${CPUREF[@]+"${CPUREF[@]}"} \
  ${STOPBAR[@]+"${STOPBAR[@]}"}
if has fp32gpu; then
  soft gpu_f32 "${R}_fp32_gpu_f32_check.json" "$QH" "$WIN" -- "$PY" "$S/d1_check.py" --tflite "$F" --accel gpu --f32 \
    --compare "$CMP" ${REFARG[@]+"${REFARG[@]}"}
  soft gpu_default "${R}_fp32_gpu_default_check.json" "$QH" "$WIN" -- "$PY" "$S/d1_check.py" --tflite "$F" --accel gpu \
    --compare "$CMP" ${REFARG[@]+"${REFARG[@]}"}
fi
for v in v2:v2_fp16fc_i8emb v3:v3_i8emb v1:v1_wi8fc v2e:v2e_fp16fc; do
  has "${v%%:*}" || continue
  step "storage_${v%%:*}" "${R}_${v#*:}_quant.json" \
    "$QW" -- "$PY" "$S/d1_storage.py" --tflite "$F" --variant "${v%%:*}"
done
for v in v2:v2_fp16fc_i8emb v3:v3_i8emb v2e:v2e_fp16fc; do
  has "${v%%:*}" || continue
  v=${v#*:}
  G=exports/${TAG}_rowprefill${KIND}_L${L}_${v}.tflite
  SB=()
  case "$v" in v2_fp16fc_i8emb | v2e_fp16fc) SB=(${STOPBAR[@]+"${STOPBAR[@]}"}) ;; esac
  step "cpu_$v" "${R}_${v}_cpu_check.json" "$QW" -- "$PY" "$S/d1_check.py" --tflite "$G" --accel cpu --compare "$CMP" \
    ${EMBARG[@]+"${EMBARG[@]}"} ${REFARG[@]+"${REFARG[@]}"} ${VCPUREF[@]+"${VCPUREF[@]}"}
  step "gpu_f32_$v" "${R}_${v}_gpu_f32_check.json" \
    "$QH" "$WIN" -- "$PY" "$S/d1_check.py" --tflite "$G" --accel gpu --f32 --compare "$CMP" \
    ${EMBARG[@]+"${EMBARG[@]}"} ${REFARG[@]+"${REFARG[@]}"} ${SB[@]+"${SB[@]}"}
done
if has v2; then
  soft gpu_default_v2 "${R}_v2_fp16fc_i8emb_gpu_default_check.json" \
    "$QH" "$WIN" -- "$PY" "$S/d1_check.py" --tflite "exports/${TAG}_rowprefill_L${L}_v2_fp16fc_i8emb.tflite" \
    --accel gpu --compare "$CMP" ${REFARG[@]+"${REFARG[@]}"}
fi
if has v2e; then
  soft gpu_default_v2e "${R}_v2e_fp16fc_gpu_default_check.json" \
    "$QH" "$WIN" -- "$PY" "$S/d1_check.py" --tflite "exports/${TAG}_rowprefill${KIND}_L${L}_v2e_fp16fc.tflite" \
    --accel gpu --compare "$CMP" ${EMBARG[@]+"${EMBARG[@]}"} ${REFARG[@]+"${REFARG[@]}"}
fi
if has v3; then
  soft gpu_default_v3 "${R}_v3_i8emb_gpu_default_check.json" \
    "$QH" "$WIN" -- "$PY" "$S/d1_check.py" --tflite "exports/${TAG}_rowprefill_L${L}_v3_i8emb.tflite" \
    --accel gpu --compare "$CMP" ${REFARG[@]+"${REFARG[@]}"}
fi
if has v1; then
  G=exports/${TAG}_rowprefill_L${L}_v1_wi8fc.tflite
  step cpu_v1_wi8fc "${R}_v1_wi8fc_cpu_check.json" "$QW" -- "$PY" "$S/d1_check.py" --tflite "$G" --accel cpu \
    --compare "$CMP" ${REFARG[@]+"${REFARG[@]}"} ${CPUREF[@]+"${CPUREF[@]}"}
  soft gpu_f32_v1_wi8fc "${R}_v1_wi8fc_gpu_f32_check.json" \
    "$QH" "$WIN" -- "$PY" "$S/d1_check.py" --tflite "$G" --accel gpu --f32 --compare "$CMP" ${REFARG[@]+"${REFARG[@]}"}
  soft gpu_default_v1_wi8fc "${R}_v1_wi8fc_gpu_default_check.json" \
    "$QH" "$WIN" -- "$PY" "$S/d1_check.py" --tflite "$G" --accel gpu --compare "$CMP" ${REFARG[@]+"${REFARG[@]}"}
fi
"$PY" - "$K" "$TAG" "$L" "$KIND" <<'PYEOF'
import json, sys
from pathlib import Path
K, tag, L, kind = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
gc = json.loads((K / f"results/{tag}_graph_check_L{L}.json").read_text())["summary"]
ex = json.loads((K / f"results/{tag}_export{kind}_L{L}.json").read_text())
print(f"graph_check pass={gc['pass']} parity_max={gc['parity_max_abs_real']:.3e} pad_bit_equal={gc['pad_content_bit_equal']}")
print(f"export {ex['status']} ops={ex['operator_count']} custom={ex['stops']['custom_op_count']} "
      f"int64={ex['int64_tensor_count']} broadcast_to={ex['broadcast_to_count']} bytes={ex['bytes']}")
print("| file | accel | delegated / total | max abs real vs torch | vs fp32 file | non-finite | argmax (non near-tie / near-tie) "
      "| max dp | mean dp | red arms | bar (near ties apart) |")
print("|---|---|---|---:|---:|---:|---|---:|---:|---|---|")
for p in sorted((K / "results").glob(f"{tag}_rowprefill{kind}_L{L}_*_check.json")):
    d = json.loads(p.read_text())
    rep = d["delegation"]["replacing"]
    dl = ", ".join(f"{r['delegated']}/{r['total']} {r['delegate']}" for r in rep) or "-"
    cmp = d.get("max_abs_real_vs_compare")
    rp = (d.get("reference_parity") or {}).get("summary")
    ref = ("| - | - | - | - | - |" if not rp else
           f"| {rp['non_near_tie_argmax']} / {rp['near_tie_argmax']} | {rp['max_abs_dp']:.3e} | "
           f"{rp['mean_abs_dp_all_options']:.3e} | {rp['line']['red_arms']} | {rp['bar_near_tie_apart']} |")
    print(f"| {Path(d['tflite']).name} | {d['accel']} | {dl} | {d.get('max_abs_real_vs_torch_d1prefill', float('nan')):.3e} | "
          f"{'-' if cmp is None else format(cmp, '.3e')} | {d.get('nonfinite_all_total')} " + ref)
PYEOF
if [ "${D1_RETIRE_FP32:-0}" = 1 ]; then
  step retire_fp32 "results/${TAG}${KIND}_L${L}_fp32_deleted.json" \
    "$QW" -- "$PY" "$S/d1_export.py" --retire --tag "$TAG" --L "$L" ${EXPARG[@]+"${EXPARG[@]}"}
fi
echo "[$(date +%H:%M:%S)] done: $TAG$KIND L$L; soft failures: ${SOFT_FAILED[*]:-none}; skipped (D1_RESUME): ${SKIPPED[*]:-none}"
