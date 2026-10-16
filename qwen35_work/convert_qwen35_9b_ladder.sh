#!/usr/bin/env bash
# convert_qwen35_9b.sh <HF id> OUTDIR [REVISION] - Qwen3.5-9B-class GatedDeltaNet + attention hybrids (Qwen/Qwen3.5-9B and its
# finetunes with the same config, e.g. ornith-ai/Ornith-1.0-9B) -> OUTDIR/<name>_int8.litertlm + OUTDIR/<name>_mixed_int4.litertlm
# (text decoder only) in one command; every step logs its command to OUTDIR/recipe.log. <name> = the HF id's basename.
#   0 checkpoint: REVISION (default: Qwen/Qwen3.5-9B's c202236235762e1c871ad0ccb60c8ee5ba337b9a; other ids need it) in a scratch HF
#     cache (fetch_pinned.py: LFS sha256 + git blob sha1 of every file, one file at a time, HF_HUB_DISABLE_XET=1; reused when
#     HF_HUB_CACHE already holds the snapshot), offline afterwards
#   1 float export (QWEN35_PY = a Python with transformers 5.14.1, torch 2.12.1, ai-edge-quantizer 0.8.0 and litert-torch's
#     dependencies; litert-torch itself = LITERT_TORCH_DIR, a checkout of 115a13607c730c81018bb9789138a3e5e5119e3d with
#     ../decider2bv_work/deps/qwen35_hybrid_litert_torch.patch applied, on PYTHONPATH; see REPRODUCE.md):
#     convert_qwen35_hybrid_lw.py STOP_AFTER_EXPORT=1 = convert_qwen35_hybrid.py + experimental_lightweight_conversion with
#     RESOURCE_CONSTANT_NUMEL_THRESHOLD=100000000 (constants under 100 MB fold as in the default export), prefill ladder
#     1024,256,64,16,4,1, KV cache 4096. Peaks near 50 GB of RAM (each quantize step below near 37 GB): run ONE export at a time,
#     on a machine with plenty of free RAM.
#   2 int8: the same script with SKIP_EXPORT=1 = post-hoc dynamic int8 on FULLY_CONNECTED + EMBEDDING_LOOKUP
#     (../minicpm5_work/quantize_minicpm5.py wi8fc) with QWEN35_FLOAT_FC_REGEX='Linear_in_proj_(a|b);' (the 48 GatedDeltaNet
#     decay/gate projections stay float), the rail's simple template + <|im_end|> stop, ExecutorMetadata (the 64 state buffers);
#     then ../scripts/set_activation_type.py --type fp32 (the GPU's fp16 activation default breaks these weights)
#   3 int4: make_int4_9b.py b32 = blockwise-32 min-max int4 FCs + int8 channelwise lm_head / embedding, the same float regex,
#     template + stop + ExecutorMetadata + fp32 activations
#   4 metadata: the prompt template qwen35_text_current.jinja (thinking off: every assistant turn and the generation prompt carry an
#     empty think block; renders as the vendor template with enable_thinking=False) via dump_llm_metadata.py +
#     make_new_pbtext.py --template-only + ../g9v3_work/repack_metadata.py (litert-lm 0.18.0 CLI + builder; only jinja_prompt_template
#     changes, every other section asserted identical)
#   5 checks: fc_dtypes.py (FC / embedding weight forms; float FCs = the regex's), scan_scales.py (int4 block scales),
#     `litert-lm describe`, sha256. The file sha256 changes from run to run (the packer writes a uuid and a timestamp): compare
#     sections.
# env: QWEN35_PY (python for steps 0-3), LITERT_TORCH_DIR (default qwen35_work/litert-torch-qwen35), RUN_PY (a Python with
#      litert-lm 0.18.0 = litert-lm-builder 0.18.0), LITERT_LM (the litert-lm 0.18.0 CLI), CHECK_PY (a Python with ai-edge-litert
#      2.2.0 + litert-lm-builder, for step 5), HF_HUB_CACHE (default OUTDIR/hf_cache).
set -euo pipefail
Q=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$Q/.." && pwd)
[ $# -ge 2 ] || { echo "usage: $0 <HF id> OUTDIR [REVISION]"; exit 2; }
HFID=$1; mkdir -p "$2"; OUT=$(cd "$2" && pwd)
case "$HFID" in Qwen/Qwen3.5-9B) DEF_REV=c202236235762e1c871ad0ccb60c8ee5ba337b9a ;; *) DEF_REV= ;; esac
REV=${3:-$DEF_REV}; [ -n "$REV" ] || { echo "REVISION is required for $HFID (pin the commit)"; exit 2; }
NAME=$(basename "$HFID")
: "${QWEN35_PY:?set QWEN35_PY}" "${RUN_PY:?set RUN_PY}" "${LITERT_LM:?set LITERT_LM}" "${CHECK_PY:?set CHECK_PY}"
WT=${LITERT_TORCH_DIR:-$Q/litert-torch-qwen35}
CLI_DIR=$(cd "$(dirname "$LITERT_LM")" && pwd)
export HF_HUB_CACHE=${HF_HUB_CACHE:-$OUT/hf_cache}
SNAP=$HF_HUB_CACHE/models--${HFID/\//--}/snapshots/$REV
log() { echo "$*" >> "$OUT/recipe.log"; }
run() { log "### CMD: $*"; "$@"; }
log "### start $(date '+%F %T %Z') $HFID @ $REV HF_HUB_CACHE=$HF_HUB_CACHE"
[ -d "$SNAP" ] || run env HF_HUB_DISABLE_XET=1 "$QWEN35_PY" -I "$Q/fetch_pinned.py" "$HF_HUB_CACHE" "$OUT/hf_cache_manifest.json" "$HFID" "$REV"
export HF_HUB_OFFLINE=1
ENV=(env PYTHONPATH="$WT" PATH="$CLI_DIR:$PATH" "QWEN35_PREFILL_LADDER=${QWEN35_PREFILL_LADDER:-1024,256,64,16,4,1}" "QWEN35_FLOAT_FC_REGEX=Linear_in_proj_(a|b);"
     RESOURCE_CONSTANT_NUMEL_THRESHOLD=100000000 QWEN35_NAME="$NAME" TMPDIR="$OUT/tmp")
mkdir -p "$OUT/tmp" "$OUT/out"
step() {  # step NAME OUTPUT CMD... : run once, the step's log in OUTDIR/NAME.log; a failure removes the partial output and stops
  local n=$1 outp=$2; shift 2
  if run "$@" > "$OUT/$n.log" 2>&1 && [ -s "$outp" ]; then return 0; fi
  rm -f "$outp"; log "### $n: FAILED (see $OUT/$n.log)"; exit 1
}
[ -s "$OUT/out/model.litertlm" ] || step export "$OUT/out/model.litertlm" "${ENV[@]}" STOP_AFTER_EXPORT=1 "$QWEN35_PY" "$Q/convert_qwen35_hybrid_lw.py" "$HFID" "$OUT/out"
[ -s "$OUT/out/${NAME}_int8.litertlm" ] || step int8 "$OUT/out/${NAME}_int8.litertlm" "${ENV[@]}" SKIP_EXPORT=1 "$QWEN35_PY" "$Q/convert_qwen35_hybrid_lw.py" "$HFID" "$OUT/out"
run env PATH="$CLI_DIR:$PATH" python3 "$ROOT/scripts/set_activation_type.py" "$OUT/out/${NAME}_int8.litertlm" "$OUT/out/${NAME}_int8_fp32act.litertlm" --type fp32 --litert-lm "$LITERT_LM" > "$OUT/set_fp32act.log" 2>&1
[ -s "$OUT/out/${NAME}_mixed_int4.litertlm" ] || step int4 "$OUT/out/${NAME}_mixed_int4.litertlm" "${ENV[@]}" "$QWEN35_PY" "$Q/make_int4_9b.py" "$OUT/out/model.litertlm" "$OUT/out" b32
TPL=$Q/qwen35_text_current.jinja
for q in int8 int4; do
  if [ $q = int8 ]; then IN=$OUT/out/${NAME}_int8_fp32act.litertlm; DST=$OUT/${NAME}_int8.litertlm
  else IN=$OUT/out/${NAME}_mixed_int4.litertlm; DST=$OUT/${NAME}_mixed_int4.litertlm; fi
  run "$RUN_PY" -I "$Q/dump_llm_metadata.py" "$IN" "$OUT/meta_$q"
  run "$RUN_PY" -I "$Q/make_new_pbtext.py" "$OUT/meta_$q" "$TPL" "$OUT/$q.pbtext" --template-only
  EXPECT=$(python3 -c 'import json,sys; print(",".join(json.load(open(sys.argv[1]))["changed_fields"]))' "$OUT/$q.pbtext.check.json")
  run "$RUN_PY" -I "$ROOT/g9v3_work/repack_metadata.py" "$IN" "$DST" --pbtext "$OUT/$q.pbtext" \
    --report "$OUT/repack_$q.json" --expect-changed "$EXPECT" --cli "$LITERT_LM" --scratch-parent "$OUT/tmp"
  run "$LITERT_LM" describe "$DST" < /dev/null > "$OUT/describe_$q.txt" 2>&1
  run "$CHECK_PY" -I "$Q/fc_dtypes.py" "$DST" "$OUT/fc_dtypes_$q.json" > "$OUT/fc_dtypes_$q.txt" 2>&1
done
run "$CHECK_PY" -I "$Q/scan_scales.py" "$OUT/${NAME}_mixed_int4.litertlm" "$OUT/scan_int4.json" > "$OUT/scan_int4.txt" 2>&1
shasum -a 256 "$OUT/${NAME}_int8.litertlm" "$OUT/${NAME}_mixed_int4.litertlm" | tee -a "$OUT/recipe.log"
log "### done $(date '+%F %T %Z')"
