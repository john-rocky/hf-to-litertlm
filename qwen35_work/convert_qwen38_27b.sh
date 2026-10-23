#!/usr/bin/env bash
# convert_qwen38_27b.sh <HF id> OUTDIR [REVISION] - Qwen/Qwen3.8-27B (a 27B GatedDeltaNet + attention hybrid of the Qwen3.5
# architecture) -> OUTDIR/<name>_mixed_int4.litertlm (text decoder only, thinking off; Mixed INT4 with int4 blocks of 128) in one command, without
# the float .litertlm a 27B would need next to its float TFLite (108 GB each). Every step logs its command to OUTDIR/recipe.log and its
# output to OUTDIR/<step>.log; a step whose output exists is skipped when the command runs again, and STOP_AFTER=fetch | upcast |
# export | int4 | tokenizer | noswitch stops after that step (to run the download and the copy apart from the heavy steps). <name> = the HF id's
# basename. Disk for the 27B: about 310 GB (snapshot 56 GB, F32 copy 111 GB, float TFLite 108 GB, the int4 file twice, 15 GB each).
#   0 checkpoint: REVISION (default: Qwen/Qwen3.8-27B's 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0; other ids need it) in a scratch HF
#     cache (fetch_pinned.py: LFS sha256 + git blob sha1 of every file, one file at a time, HF_HUB_DISABLE_XET=1; reused when
#     HF_HUB_CACHE already holds the snapshot), offline afterwards
#   1 F32 copy (upcast_fp32.py SNAPSHOT F32_DIR): BF16 -> F32 by a 16-bit shift (exact), the same tensor names and shapes, so that
#     transformers 5.14.1 hands the export the memory-mapped F32 tensors instead of making a float32 copy of every weight in RAM (the
#     27B: 108 GB); F32_DIR defaults to OUTDIR/fp32src and takes twice the checkpoint's size
#   2 float export (QWEN35_PY = a Python with transformers 5.14.1, torch 2.12.1, ai-edge-quantizer 0.8.0 and litert-torch's
#     dependencies; litert-torch itself = LITERT_TORCH_DIR, a checkout of 115a13607c730c81018bb9789138a3e5e5119e3d with
#     qwen35_hybrid_litert_torch.patch applied, on PYTHONPATH; see REPRODUCE.md): convert_qwen38_hybrid_lw.py F32_DIR OUTDIR/export =
#     the export of convert_qwen35_hybrid_lw.py (lightweight conversion, RESOURCE_CONSTANT_NUMEL_THRESHOLD=100000000, prefill ladder
#     1024,256,64,16,4,1, KV cache 4096) writing the float TFLite and the packager's LlmMetadata but no float .litertlm, with a
#     read-only +-inf check in place of litert-torch's in-place clamp of every float constant (that clamp turns every page of a
#     memory-mapped F32 checkpoint into anonymous memory). The 27B on a 137 GB Mac with the F32 copy and the TFLite on an external
#     hard disk: 66 minutes, peak memory footprint 11.8 GB, a 107.7 GB TFLite
#   3 int4: make_int4_27b.py OUTDIR/export OUTDIR/out b128 TEMPLATE = int4 blockwise-128 min-max on the FULLY_CONNECTED ops, int8
#     channelwise on the lm_head and the embedding, the GatedDeltaNet decay / gate projections ('Linear_in_proj_(a|b);') float, block
#     scales floored at 2^-14, the quantized TFLite written into the output file's mmap; LlmMetadata = the export's with TEMPLATE
#     (default qwen38_text_Q1.jinja: Qwen3.8's chat format with a thinking switch, off by default), a `thought` channel,
#     supports_thinking true, supports_function_calling false (make_new_pbtext.py); ExecutorMetadata; fp32 GPU activations declared.
#     The 27B: 18 minutes, peak memory footprint 46.2 GB
#   4 tokenizer: use_repo_tokenizer.sh with the snapshot's tokenizer.json (the export writes the tokenizer transformers builds from
#     the tokenizer class, not the repo's file)
#   5 noswitch: use_noswitch_template.sh = the LlmMetadata of a thinking-off-only file (qwen38_text_noswitch.jinja = the switch
#     template's thinking-off branch, no `thought` channel, supports_thinking false; every other section unchanged): with the thinking
#     switch, LiteRT-LM 0.18.1's default conversation settings broke three-turn conversations with thinking on (REPRODUCE.md)
#   6 checks: fc_dtypes.py, const_ops.py, scan_scales.py, bundle_sections.py (every section's sha256), `litert-lm describe`, sha256.
#     The file sha256 changes from run to run (the packer writes a uuid and a timestamp): compare sections.
# env: QWEN35_PY (steps 0-3), LITERT_TORCH_DIR (default qwen35_work/litert-torch-qwen35), RUN_PY (a Python with litert-lm 0.18.1 =
#      litert-lm-builder 0.18.1), LITERT_LM (the litert-lm 0.18.1 CLI), CHECK_PY (a Python with ai-edge-litert 2.2.0,
#      litert-lm-builder and tokenizers: step 5 and the tokenizer check), HF_HUB_CACHE (default OUTDIR/hf_cache), F32_DIR, TEMPLATE,
#      STOP_AFTER.
set -euo pipefail
Q=$(cd "$(dirname "$0")" && pwd)
[ $# -ge 2 ] || { echo "usage: $0 <HF id> OUTDIR [REVISION]"; exit 2; }
HFID=$1; mkdir -p "$2"; OUT=$(cd "$2" && pwd)
case "$HFID" in Qwen/Qwen3.8-27B) DEF_REV=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 ;; *) DEF_REV= ;; esac
REV=${3:-$DEF_REV}; [ -n "$REV" ] || { echo "REVISION is required for $HFID (pin the commit)"; exit 2; }
NAME=$(basename "$HFID")
: "${QWEN35_PY:?set QWEN35_PY}" "${RUN_PY:?set RUN_PY}" "${LITERT_LM:?set LITERT_LM}" "${CHECK_PY:?set CHECK_PY}"
WT=${LITERT_TORCH_DIR:-$Q/litert-torch-qwen35}
CLI_DIR=$(cd "$(dirname "$LITERT_LM")" && pwd)
export HF_HUB_CACHE=${HF_HUB_CACHE:-$OUT/hf_cache}
SNAP=$HF_HUB_CACHE/models--${HFID/\//--}/snapshots/$REV
F32=${F32_DIR:-$OUT/fp32src}
TPL=${TEMPLATE:-$Q/qwen38_text_Q1.jinja}
EXP=$OUT/export
INT4=$OUT/out/${NAME}_mixed_int4_b128.litertlm
SWAP=$OUT/out/${NAME}_mixed_int4.swap.litertlm
DST=$OUT/${NAME}_mixed_int4.litertlm
log() { echo "$*" >> "$OUT/recipe.log"; }
run() { log "### CMD: $*"; "$@"; }
stop_after() { if [ "${STOP_AFTER:-}" = "$1" ]; then log "### stop after $1 (STOP_AFTER) $(date '+%F %T %Z')"; exit 0; fi; }
step() {  # step NAME OUTPUT CMD... : run once, the step's log in OUTDIR/NAME.log; a failure removes the partial output and stops
  local n=$1 outp=$2; shift 2
  if run "$@" > "$OUT/$n.log" 2>&1 && [ -s "$outp" ]; then return 0; fi
  rm -f "$outp"; log "### $n: FAILED (see $OUT/$n.log)"; exit 1
}
log "### start $(date '+%F %T %Z') $HFID @ $REV HF_HUB_CACHE=$HF_HUB_CACHE"
[ -d "$SNAP" ] || run env HF_HUB_DISABLE_XET=1 "$QWEN35_PY" -I "$Q/fetch_pinned.py" "$HF_HUB_CACHE" "$OUT/hf_cache_manifest.json" "$HFID" "$REV"
export HF_HUB_OFFLINE=1
stop_after fetch
if [ ! -s "$OUT/fp32src_manifest.json" ]; then
  rm -rf "$F32"
  step upcast "$OUT/fp32src_manifest.json" "$QWEN35_PY" -I "$Q/upcast_fp32.py" "$SNAP" "$F32" "$OUT/fp32src_manifest.json"
fi
stop_after upcast
mkdir -p "$OUT/tmp" "$OUT/out"
if [ ! -s "$EXP/export_artifacts.json" ]; then
  rm -rf "$EXP"; mkdir -p "$EXP"
  step export "$EXP/export_artifacts.json" env PYTHONPATH="$WT" PATH="$CLI_DIR:$PATH" QWEN35_PREFILL_LADDER=1024,256,64,16,4,1 \
    "QWEN35_FLOAT_FC_REGEX=Linear_in_proj_(a|b);" RESOURCE_CONSTANT_NUMEL_THRESHOLD=100000000 QWEN35_NAME="$NAME" TMPDIR="$OUT/tmp" \
    "$QWEN35_PY" "$Q/convert_qwen38_hybrid_lw.py" "$F32" "$EXP"
fi
stop_after export
if [ ! -s "$INT4" ]; then
  rm -rf "$OUT/out/work_b128"
  step int4 "$INT4" env QWEN35_NAME="$NAME" RUN_PY="$RUN_PY" LITERT_LM="$LITERT_LM" PATH="$CLI_DIR:$PATH" TMPDIR="$OUT/tmp" \
    "$QWEN35_PY" "$Q/make_int4_27b.py" "$EXP" "$OUT/out" b128 "$TPL"
fi
stop_after int4
if [ ! -s "$DST" ] && [ ! -s "$SWAP" ]; then
  rm -f "$SWAP.tok.json" "$SWAP.tok.tokenizer.json"
  step tokenizer "$SWAP" env RUN_PY="$RUN_PY" LITERT_LM="$LITERT_LM" CHECK_PY="$CHECK_PY" TMPDIR="$OUT/tmp" \
    bash "$Q/use_repo_tokenizer.sh" "$INT4" "$SWAP" "$SNAP/tokenizer.json"
fi
stop_after tokenizer
if [ ! -s "$DST" ]; then
  rm -rf "$DST.meta" "$DST.pbtext" "$DST.pbtext.check.json" "$DST.repack.json" "$DST.repack.json.evidence"
  step noswitch "$DST" env RUN_PY="$RUN_PY" LITERT_LM="$LITERT_LM" bash "$Q/use_noswitch_template.sh" "$SWAP" "$DST"
  run rm -f "$SWAP"
fi
stop_after noswitch
run "$CHECK_PY" -I "$Q/fc_dtypes.py" "$DST" "$OUT/fc_dtypes.json" > "$OUT/fc_dtypes.txt" 2>&1
run "$CHECK_PY" -I "$Q/const_ops.py" "$DST" "$OUT/const_ops.json" > "$OUT/const_ops.txt" 2>&1
run "$CHECK_PY" -I "$Q/scan_scales.py" "$DST" "$OUT/scan_int4.json" > "$OUT/scan_int4.txt" 2>&1
run "$RUN_PY" -I "$Q/bundle_sections.py" "$OUT/sections.json" "$DST" > "$OUT/sections.txt" 2>&1
run "$LITERT_LM" describe "$DST" < /dev/null > "$OUT/describe.txt" 2>&1
shasum -a 256 "$DST" | tee -a "$OUT/recipe.log"
log "### done $(date '+%F %T %Z')"
