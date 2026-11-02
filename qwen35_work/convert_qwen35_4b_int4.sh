#!/usr/bin/env bash
# convert_qwen35_4b_int4.sh <HF id> OUTDIR [REVISION] - a Qwen3.5-4B text decoder (GatedDeltaNet + attention hybrid; a finetune such
# as ccore/Qwen3.5-4B-3D-Coder) -> OUTDIR/<name>_mixed_int4.litertlm in one command: Mixed INT4 at the KV cache length and prefill
# ladder you choose, the model repo's tokenizer.json in the tokenizer section, a thinking-off-only template, fp32 GPU activations
# declared. <name> = the HF id's basename. Every step logs its command to OUTDIR/recipe.log as `### CMD:` and its output to
# OUTDIR/<step>.log; a step whose output exists is skipped when the command runs again, and STOP_AFTER=fetch | export | int4 |
# tokenizer | switch | noswitch stops after that step.
#   0 checkpoint: REVISION (default for ccore/Qwen3.5-4B-3D-Coder: 196c03ff49d8d99f49235ad19c7d48f940bc7523; other ids need it) in a
#     scratch HF cache (fetch_pinned.py: every file of the commit, LFS sha256 + git blob sha1 checked; reused when HF_HUB_CACHE
#     already holds the snapshot), offline afterwards
#   1 float export (export_qwen35_float.py: litert-torch export_hf, no quantization) with QWEN35_CACHE_LENGTH (default 32768) and
#     QWEN35_PREFILL_LADDER (default 1024,256,64,16,4,1), PYTHONHASHSEED=0, the pinned litert-torch checkout (LITERT_TORCH_DIR) first on
#     PYTHONPATH -> OUTDIR/export/model.litertlm (float32 weights: about 17 GB for a 4B; the export peaked at about 57 GB of memory on
#     a 128 GB Mac)
#   2 int4: build_qwen35_quant.py int4 = blockwise-32 min-max int4 FULLY_CONNECTED weights, int8 channelwise lm_head and embedding, the
#     48 GatedDeltaNet decay / gate projections ('Linear_in_proj_(a|b);') float, block scales floored at 2^-14
#     (../falcon_h1_work/quantize_clamped.py), stops 248046 + 248044, the ExecutorMetadata section, fp32 GPU activations declared
#     -> OUTDIR/int4/<name>_mixed_int4.litertlm
#   3 tokenizer: use_repo_tokenizer.sh with the snapshot's tokenizer.json (EXPECT_TOKENIZER_SHA256 optional; the export writes the
#     tokenizer transformers builds from the tokenizer class, not the repo's file)
#   4 switch: use_switch_template.sh = the thinking-switch metadata of the Qwen3.5-4B files (qwen35_text_T1.jinja, a `thought` channel,
#     supports_thinking true, supports_function_calling false)
#   5 noswitch: the thinking-off-only metadata on top of step 4: dump_llm_metadata.py -> make_noswitch_pbtext.py with
#     qwen35_text_T1_noswitch.jinja (qwen35_text_T1.jinja whose generation prompt always opens with an empty think block) ->
#     ../g9v3_work/repack_metadata.py: template, `thought` channel removed, supports_thinking false; every other field and section kept
#   6 fp32: ../scripts/set_activation_type.py --type fp32 once more (steps 3-5 unpack and pack the bundle; the tool reads the
#     declaration back from every TFLiteModel section) -> OUTDIR/<name>_mixed_int4.litertlm
#   7 checks: fc_dtypes.py, scan_scales.py (no zero or subnormal int4 scale), bundle_sections.py (every section's sha256),
#     dump_llm_metadata.py, `litert-lm describe`, sha256. The file sha256 changes from run to run (the packer writes a uuid and a
#     timestamp): compare sections.
# env: QWEN35_PY (a Python with transformers 5.14.1, torch 2.12.1, ai-edge-quantizer 0.8.0 and litert-torch's dependencies: steps 0-2
#      and 6), LITERT_TORCH_DIR (default qwen35_work/litert-torch-qwen35 = 115a13607c730c81018bb9789138a3e5e5119e3d with
#      qwen35_hybrid_litert_torch.patch applied; REPRODUCE.md), RUN_PY (a Python with litert-lm 0.18.0 = litert-lm-builder 0.18.0),
#      LITERT_LM (the litert-lm 0.18.0 CLI; it also unpacks and packs in steps 1-2 and 6), CHECK_PY (a Python with ai-edge-litert,
#      litert-lm-builder and tokenizers), HF_HUB_CACHE (default OUTDIR/hf_cache), QWEN35_CACHE_LENGTH, QWEN35_PREFILL_LADDER,
#      EXPECT_TOKENIZER_SHA256, EXEC_META_LITERT_LM (the CLI add_executor_metadata.py packs with; unset = its default,
#      ~/venvs/lt0150run/bin/litert-lm = litert-lm 0.15.0), STOP_AFTER.
set -euo pipefail
Q=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$Q/.." && pwd)
[ $# -ge 2 ] || { echo "usage: $0 <HF id> OUTDIR [REVISION]"; exit 2; }
HFID=$1; mkdir -p "$2"; OUT=$(cd "$2" && pwd)
case "$HFID" in ccore/Qwen3.5-4B-3D-Coder) DEF_REV=196c03ff49d8d99f49235ad19c7d48f940bc7523 ;; *) DEF_REV= ;; esac
REV=${3:-$DEF_REV}; [ -n "$REV" ] || { echo "REVISION is required for $HFID (pin the commit)"; exit 2; }
NAME=$(basename "$HFID")
: "${QWEN35_PY:?set QWEN35_PY}" "${RUN_PY:?set RUN_PY}" "${LITERT_LM:?set LITERT_LM}" "${CHECK_PY:?set CHECK_PY}"
WT=${LITERT_TORCH_DIR:-$Q/litert-torch-qwen35}
CLI_DIR=$(cd "$(dirname "$LITERT_LM")" && pwd)
KV=${QWEN35_CACHE_LENGTH:-32768}
LADDER=${QWEN35_PREFILL_LADDER:-1024,256,64,16,4,1}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$OUT/hf_cache}
SNAP=$HF_HUB_CACHE/models--${HFID/\//--}/snapshots/$REV
EXP=$OUT/export
INT4=$OUT/int4/${NAME}_mixed_int4.litertlm
TOKF=$OUT/post/tok/${NAME}_mixed_int4.tok.litertlm
SWF=$OUT/post/sw/${NAME}_mixed_int4.sw.litertlm
NS=$OUT/post/ns/${NAME}_mixed_int4.ns.litertlm
DST=$OUT/${NAME}_mixed_int4.litertlm
log() { echo "$*" >> "$OUT/recipe.log"; }
run() { log "### CMD: $*"; "$@"; }
stop_after() { if [ "${STOP_AFTER:-}" = "$1" ]; then log "### stop after $1 (STOP_AFTER) $(date '+%F %T %Z')"; exit 0; fi; }
step() {  # step NAME OUTPUT CMD... : run once, the step's log in OUTDIR/NAME.log; a failure removes the partial output and stops
  local n=$1 outp=$2; shift 2
  log "### $n start $(date '+%F %T %Z')"
  if run "$@" > "$OUT/$n.log" 2>&1 && [ -s "$outp" ]; then log "### $n end $(date '+%F %T %Z')"; return 0; fi
  rm -f "$outp"; log "### $n: FAILED (see $OUT/$n.log)"; exit 1
}
log "### start $(date '+%F %T %Z') $HFID @ $REV KV $KV ladder $LADDER HF_HUB_CACHE=$HF_HUB_CACHE"
[ -d "$SNAP" ] || run env HF_HUB_DISABLE_XET=1 "$QWEN35_PY" -I "$Q/fetch_pinned.py" "$HF_HUB_CACHE" "$OUT/hf_cache_manifest.json" "$HFID" "$REV"
export HF_HUB_OFFLINE=1
stop_after fetch
mkdir -p "$OUT/tmp" "$OUT/int4" "$OUT/post/tok" "$OUT/post/sw" "$OUT/post/ns" "$OUT/checks"
export TMPDIR=$OUT/tmp
if [ ! -s "$EXP/model.litertlm" ]; then
  rm -rf "$EXP"; mkdir -p "$EXP"
  step export "$EXP/model.litertlm" env PYTHONPATH="$WT" PYTHONHASHSEED=0 PATH="$CLI_DIR:$PATH" TOKENIZERS_PARALLELISM=false \
    QWEN35_CACHE_LENGTH="$KV" QWEN35_PREFILL_LADDER="$LADDER" "$QWEN35_PY" "$Q/export_qwen35_float.py" "$HFID" "$EXP"
fi
stop_after export
if [ ! -s "$INT4" ]; then
  step int4 "$INT4" env PYTHONHASHSEED=0 PATH="$CLI_DIR:$PATH" "QWEN35_FLOAT_FC_REGEX=Linear_in_proj_(a|b);" \
    "$QWEN35_PY" "$Q/build_qwen35_quant.py" int4 "$EXP/model.litertlm" "$OUT/int4" "$NAME"
fi
stop_after int4
if [ ! -s "$DST" ] && [ ! -s "$TOKF" ]; then
  rm -f "$TOKF.tok.json" "$TOKF.tok.tokenizer.json"
  step tokenizer "$TOKF" env RUN_PY="$RUN_PY" LITERT_LM="$LITERT_LM" CHECK_PY="$CHECK_PY" \
    bash "$Q/use_repo_tokenizer.sh" "$INT4" "$TOKF" "$SNAP/tokenizer.json"
fi
stop_after tokenizer
if [ ! -s "$DST" ] && [ ! -s "$SWF" ]; then
  rm -rf "$SWF.meta" "$SWF.pbtext" "$SWF.pbtext.check.json" "$SWF.repack.json" "$SWF.repack.evidence"
  step switch "$SWF" env RUN_PY="$RUN_PY" LITERT_LM="$LITERT_LM" bash "$Q/use_switch_template.sh" "$TOKF" "$SWF"
fi
stop_after switch
if [ ! -s "$DST" ] && [ ! -s "$NS" ]; then
  rm -rf "$NS.meta" "$NS.pbtext" "$NS.pbtext.check.json" "$NS.repack.json" "$NS.repack.evidence"
  step noswitch_meta "$NS.meta/LlmMetadataProto.bin" "$RUN_PY" -I "$Q/dump_llm_metadata.py" "$SWF" "$NS.meta"
  step noswitch_pbtext "$NS.pbtext" "$RUN_PY" -I "$Q/make_noswitch_pbtext.py" "$NS.meta" "$Q/qwen35_text_T1_noswitch.jinja" "$NS.pbtext"
  EXPECT=$(python3 -c 'import json,sys; print(",".join(json.load(open(sys.argv[1]))["changed_fields"]))' "$NS.pbtext.check.json")
  step noswitch "$NS" "$RUN_PY" -I "$ROOT/g9v3_work/repack_metadata.py" "$SWF" "$NS" --pbtext "$NS.pbtext" \
    --report "$NS.repack.json" --expect-changed "$EXPECT" --cli "$LITERT_LM"
fi
stop_after noswitch
if [ ! -s "$DST" ]; then
  step fp32 "$DST" env PATH="$CLI_DIR:$PATH" "$QWEN35_PY" -I "$ROOT/scripts/set_activation_type.py" "$NS" "$DST" --type fp32
  run rm -f "$TOKF" "$SWF" "$NS"
fi
C=$OUT/checks
rm -rf "$C/meta"
run "$CHECK_PY" -I "$Q/fc_dtypes.py" "$DST" "$C/fc_dtypes.json" > "$C/fc_dtypes.txt" 2>&1
run "$CHECK_PY" -I "$Q/scan_scales.py" "$DST" "$C/scan_scales.json" > "$C/scan_scales.txt" 2>&1
run "$RUN_PY" -I "$Q/bundle_sections.py" "$C/sections.json" "$DST" > "$C/sections.txt" 2>&1
run "$RUN_PY" -I "$Q/dump_llm_metadata.py" "$DST" "$C/meta" > "$C/meta.txt" 2>&1
run "$LITERT_LM" describe "$DST" < /dev/null > "$C/describe.txt" 2>&1
shasum -a 256 "$DST" | tee -a "$OUT/recipe.log"
log "### done $(date '+%F %T %Z')"
