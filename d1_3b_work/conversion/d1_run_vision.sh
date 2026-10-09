#!/usr/bin/env bash
# d1_run_vision.sh <source> <tag>: the picture graphs of d1-3B, one command.
#   source = tiny | a snapshot directory holding LiquidAI/d1-3B's model.safetensors; tag = export prefix (tiny, real).
# Steps (round 3 ran each on the tiny VL model, ROUND3.md; round 6d on the real weights, ROUND6D.md):
#   host_check     d1_vision_host_check.py (venv-ref) once: pictures, processor tensors, acceptance 1-3; skipped when
#                  results/vision_host_check.json already says pass (it never overwrites)
#   extra          d1v_extra_pictures.py (venv-ref) when D1V_EXTRA names pictures: their processor tensors
#   torch_check    d1_vision_torch_check.py: VisionTower / Projector vs transformers on every tile (round 3 bar 1e-5
#                  absolute; D1V_REL_BAR = a relative bar), the HF rows saved for the LiteRT checks
#   tower          d1_vision_graph.py --write tower -> exports/{tag}_vision_tower_fp32.tflite + scan
#   tower_prelint  tools/rank3_prelint.py (attention BMM and post_layernorm affine rows are expected, see ROUND3.md)
#   tower_v2       d1v_storage.py --variant v2 when D1V_STORAGE=v2 (fp16 FC weights) -> {tag}_vision_tower_v2_fp16fc
#   per file (fp32, then v2):  _cpu  d1_vision_lrt_check.py --accel cpu (vs torch and transformers, pad noise)
#                  _gpu_f32 / _gpu_default   Metal, inside the GPU window (soft: a refusal is recorded, the chain goes
#                  on; lfm25vl_work saw SigLIP2 vision files refused by every GPU delegate)
#   projector, projector_prelint, projector_v2 and their checks   the same for the projector
#   e2e            host/test_d1_vision_tiny.py --accel cpu (tiny only: it needs the tiny text graph and the provider's
#                  reference on the tiny VL model; a snapshot source skips it until the real text graphs exist)
# Env (round 6d; all optional, the defaults are round 3's behaviour):
#   D1V_RTAG       results / cache prefix (results/{rtag}_*, cache/{rtag}/); unset = round 3 names
#   D1V_ROOT       output root (K-relative or absolute) for exports / results / cache / logs (a rerun next to a first run)
#   D1V_LOG_PREFIX prefix of the log files; D1V_WIN = the GPU window label (default d1a-{tag}-vision)
#   D1V_TABLE      the position table file; D1V_EXTRA = space-separated name:image pairs (K-relative images)
#   D1V_STORAGE    v2 to make and check the fp16-FC variants; D1V_REL_BAR = the torch check's relative bar
# CPU-heavy steps wait for a closed measurement window (quiet_wait.py); GPU steps take it (quiet_hold.py, one window
# per run). The chain stops at the first failing step except the soft ones; every step logs to
# {root}/logs/{log_prefix}{rtag}_vision_<step>.log; existing outputs are never overwritten (the scripts refuse).
set -euo pipefail
[ $# -eq 2 ] || { echo "usage: $0 <tiny|snapshot dir> <tag>" >&2; exit 2; }
SRC=$1; TAG=$2
K=$(cd "$(dirname "$0")/.." && pwd)
REPO=$(cd "$K/.." && pwd)
PY=${D1_PY:-$HOME/venvs/lt094dev/bin/python}
REF_PY=${D1_REF_PY:-$K/venv-ref/bin/python}
QW=${D1_QW:?set D1_QW: a command that runs what follows its --}
QH=${D1_QH:?set D1_QH: a command that takes a label and runs what follows its --}
S=$K/scripts
RTAG=${D1V_RTAG:-}
ROOT=${D1V_ROOT:-}
LOGP=${D1V_LOG_PREFIX:-}
WIN=${D1V_WIN:-d1a-${TAG}-vision}
TABLE=${D1V_TABLE:-}
EXTRA=${D1V_EXTRA:-}
STORAGE=${D1V_STORAGE:-}
REL_BAR=${D1V_REL_BAR:-}
case "$ROOT" in "") ROOTD=$K ;; /*) ROOTD=$ROOT ;; *) ROOTD=$K/$ROOT ;; esac
NAME=${RTAG:-$TAG}
LOG=$ROOTD/logs
mkdir -p "$LOG" "$ROOTD/exports" "$ROOTD/results"
OUT_ARGS=(--tag "$TAG")
[ -n "$RTAG" ] && OUT_ARGS+=(--rtag "$RTAG")
[ -n "$ROOT" ] && OUT_ARGS+=(--root "$ROOT")
[ -n "$LOGP" ] && OUT_ARGS+=(--log-prefix "$LOGP")
PIC_ARGS=()
[ -n "$TABLE" ] && PIC_ARGS+=(--table "$TABLE")
EXTRA_PAIRS=()
EXTRA_DIR=cache/${RTAG:-vision}
[ -n "$ROOT" ] && EXTRA_DIR=$(cd "$ROOTD" && pwd)/cache/${RTAG:-vision}
for e in $EXTRA; do
  n=${e%%:*}
  EXTRA_PAIRS+=("$e")
  PIC_ARGS+=(--extra "$n:${e#*:}:${EXTRA_DIR#$K/}/proc_$n.npz")
done

SOFT_FAILED=()
step() {
  local name=$1; shift
  local log=$LOG/${LOGP}${NAME}_vision_${name}.log
  echo "[$(date +%H:%M:%S)] $name"
  if ! "$@" > "$log" 2>&1; then
    echo "FAIL $name (log: $log)"; tail -5 "$log"; exit 1
  fi
}
soft() {
  local name=$1; shift
  local log=$LOG/${LOGP}${NAME}_vision_${name}.log
  echo "[$(date +%H:%M:%S)] $name (soft)"
  if ! "$@" > "$log" 2>&1; then
    echo "SOFT FAIL $name (log: $log), going on"; tail -5 "$log"; SOFT_FAILED+=("$name")
  fi
}
checks() {   # checks <step prefix> <K-relative tflite>
  local G=$1 F=$2
  step "${G}_cpu" "$QW" -- "$PY" "$S/d1_vision_lrt_check.py" --tflite "$F" --accel cpu --source "$SRC" ${OUT_ARGS[@]+"${OUT_ARGS[@]}"} ${PIC_ARGS[@]+"${PIC_ARGS[@]}"}
  soft "${G}_gpu_f32" "$QH" "$WIN" -- "$PY" "$S/d1_vision_lrt_check.py" --tflite "$F" --accel gpu --f32 --source "$SRC" ${OUT_ARGS[@]+"${OUT_ARGS[@]}"} ${PIC_ARGS[@]+"${PIC_ARGS[@]}"}
  soft "${G}_gpu_default" "$QH" "$WIN" -- "$PY" "$S/d1_vision_lrt_check.py" --tflite "$F" --accel gpu --source "$SRC" ${OUT_ARGS[@]+"${OUT_ARGS[@]}"} ${PIC_ARGS[@]+"${PIC_ARGS[@]}"}
}

cd "$K"
if "$PY" -c 'import json,sys; sys.exit(0 if json.load(open("results/vision_host_check.json"))["summary"]["pass"] else 1)' 2>/dev/null; then
  echo "[$(date +%H:%M:%S)] host_check: results/vision_host_check.json passes, not rerun"
else
  step host_check "$QW" -- "$REF_PY" "$S/d1_vision_host_check.py"
fi
if [ ${#EXTRA_PAIRS[@]} -gt 0 ]; then
  XARGS=(--out-dir "${EXTRA_DIR#$K/}" --rtag "$NAME")
  [ -n "$ROOT" ] && XARGS+=(--root "$ROOT")
  step extra "$REF_PY" "$S/d1v_extra_pictures.py" ${EXTRA_PAIRS[@]+"${EXTRA_PAIRS[@]}"} ${XARGS[@]+"${XARGS[@]}"}
fi
TC_ARGS=()
[ -n "$REL_BAR" ] && TC_ARGS+=(--rel-bar "$REL_BAR")
step torch_check "$QW" -- "$PY" "$S/d1_vision_torch_check.py" --source "$SRC" ${OUT_ARGS[@]+"${OUT_ARGS[@]}"} ${PIC_ARGS[@]+"${PIC_ARGS[@]}"} ${TC_ARGS[@]+"${TC_ARGS[@]}"}
EXP=${ROOTD#$K/}/exports
[ "$ROOTD" = "$K" ] && EXP=exports
for G in tower projector; do
  STEM=$([ "$G" = tower ] && echo "${TAG}_vision_tower" || echo "${TAG}_projector")
  F=$EXP/${STEM}_fp32.tflite
  RN=${NAME}${STEM#$TAG}
  step "$G" "$QW" -- "$PY" "$S/d1_vision_graph.py" --source "$SRC" ${OUT_ARGS[@]+"${OUT_ARGS[@]}"} --write "$G"
  step "${G}_prelint" "$PY" "$REPO/tools/rank3_prelint.py" "$K/$F" --json "$ROOTD/results/${RN}_fp32_rank3_prelint.json"
  checks "$G" "$F"
  if [ "$STORAGE" = v2 ]; then
    F2=$EXP/${STEM}_v2_fp16fc.tflite
    step "${G}_v2" "$QW" -- "$PY" "$S/d1v_storage.py" --tflite "$F" --variant v2 ${OUT_ARGS[@]+"${OUT_ARGS[@]}"}
    step "${G}_v2_prelint" "$PY" "$REPO/tools/rank3_prelint.py" "$K/$F2" --json "$ROOTD/results/${RN}_v2_fp16fc_rank3_prelint.json"
    checks "${G}_v2" "$F2"
  fi
done
if [ "$SRC" = tiny ] && [ -z "$ROOT" ]; then
  step e2e "$QW" -- "$PY" "$K/host/test_d1_vision_tiny.py" --accel cpu --tag "$TAG"
elif [ "$SRC" = tiny ]; then
  echo "[$(date +%H:%M:%S)] e2e: skipped under D1V_ROOT (host/test_d1_vision_tiny.py reads and writes K's exports / results)"
else
  echo "[$(date +%H:%M:%S)] e2e: skipped for a snapshot source (needs the real text graph and the real reference)"
fi
"$PY" - "$ROOTD" "$NAME" <<'PYEOF'
import json, sys
from pathlib import Path
R, name = Path(sys.argv[1]), sys.argv[2]
tc = json.loads((R / f"results/{name}_vision_torch_check.json").read_text())["summary"]
print(f"torch_check pass={tc['pass']} tower_vs_hf={tc['tower_vs_hf_uncut_max']:.3e} (rel {tc['tower_vs_hf_uncut_rel_max']:.3e}) "
      f"projector_vs_hf={tc['projector_chain_vs_hf_max']:.3e} (rel {tc['projector_chain_vs_hf_rel_max']:.3e}) "
      f"pad_bits={tc['pad_bits_equal']}/{tc['pad_tiles']} unshuffle_bits={tc['unshuffle_bits_equal']}/{tc['tiles']}")
for p in sorted((R / "results").glob(f"{name}_*_tflite.json")) + sorted((R / "results").glob(f"{name}_*_quant.json")):
    w = json.loads(p.read_text())
    if "sha256" in w:
        print(f"{p.name}: {w['status']} ops={w['operator_count']} bytes={w['bytes']} sha256={w['sha256']}")
    else:
        print(f"{p.name}: checks_pass={w['checks_pass']} bytes={w['output']['bytes']} sha256={w['output']['sha256']}")
print("| file | accel | delegated / total | max abs vs torch | vs transformers (rel) | pad bits | non-finite |")
print("|---|---|---|---:|---:|---|---:|")
for p in sorted((R / "results").glob(f"{name}_*_check.json")):
    d = json.loads(p.read_text())
    if "kind" not in d and d.get("status") != "FAIL":
        continue
    rep = d["delegation"]["replacing"]
    dl = ", ".join(f"{r['delegated']}/{r['total']} {r['delegate']}" for r in rep) or "-"
    if d.get("status") == "FAIL":
        print(f"| {Path(d['tflite']).name} | {d['accel']} | {dl} | FAIL: {d.get('error', '')[:120]} | | | |")
        continue
    print(f"| {Path(d['tflite']).name} | {d['accel']} | {dl} | {d.get('max_abs_vs_torch', float('nan')):.3e} | "
          f"{d.get('max_abs_vs_hf', float('nan')):.3e} ({d.get('max_rel_vs_hf', float('nan')):.3e}) | "
          f"{d.get('pad_bits_equal')}/{d.get('pad_tiles')} | {d.get('nonfinite_total')} |")
e2e = R / f"results/{name}_vision_e2e_cpu.json"
if e2e.exists():
    s = json.loads(e2e.read_text())["summary"]
    print(f"e2e cpu pass={s['pass']} hidden_slot={s['hidden_slot_vs_provider_max']:.3e} dp={s['max_abs_dp']:.3e} "
          f"ids={s['ids_equal']}/{s['requests']} order_bits={s['insertion_order_bits_equal']}/{s['requests']}")
PYEOF
echo "[$(date +%H:%M:%S)] done: $TAG vision; soft failures: ${SOFT_FAILED[*]:-none}"
