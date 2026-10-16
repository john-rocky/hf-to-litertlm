#!/usr/bin/env bash
# convert_falcon_h1r_7b.sh OUTDIR [HF id] [REVISION] - tiiuae/Falcon-H1R-7B (falcon_h1: 44 layers of attention ∥ Mamba2, self-emitted
# <think>…</think> reasoning; Falcon-LLM License) -> OUTDIR/Falcon-H1R-7B_int8.litertlm + OUTDIR/Falcon-H1R-7B_int4.litertlm, the two
# files of litert-community/Falcon-H1R-7B, in one command. Every step logs its command to OUTDIR/recipe.log.
#   0 litert-torch: OUTDIR/litert-torch-falcon = a clone of https://github.com/john-rocky/litert-torch at the pinned base
#     115a13607c730c81018bb9789138a3e5e5119e3d (fetched by hash: it is unreachable from main) + falcon_h1_litert_torch.patch
#     (TORCH_SRC = an existing litert-torch clone that has the commit, to skip the clone)
#   1 checkpoint: REVISION (default a6f74bf181389908efd6970878d7ee2b42f5d417) into a scratch HF cache (fetch_pinned.py: LFS sha256
#     + git blob sha1 of every file, one file at a time, HF_HUB_DISABLE_XET=1; reused when HF_HUB_CACHE already holds the snapshot)
#   2 float export (convert_falcon_h1r.py STOP_AFTER_EXPORT=1, PYTHONPATH = the clone): litert-torch export_hf, prefill
#     1024,256,64,16,4,1, cache 4096, the vendor template bundled, experimental_lightweight_conversion with
#     RESOURCE_CONSTANT_NUMEL_THRESHOLD=100000000 (30.4 GB float bundle in OUTDIR/out/)
#   3 int8: the same script, SKIP_EXPORT=1 RECIPE=wi8fc = ../minicpm5_work/quantize_minicpm5.py wi8fc (dynamic int8 FCs + int8
#     embedding) -> LlmMetadata (falcon_h1r_dual.jinja, no start token, supports_thinking; make_pbtext_h1r.py +
#     ../granite_work/repack_metadata.py) -> ../scripts/add_executor_metadata.py -> ../scripts/set_activation_type.py --type fp32
#     -> ../tools/add_thought_channel.py --start '<think>' --end '</think>'
#   4 int4: the same with RECIPE=wi4b32_wi8 (int4 blockwise-32 min-max FCs + int8 lm_head + int8 embedding), run through
#     quantize_clamped.py = every blockwise int4 scale floored at 2^-14 BEFORE quantizing (clamp-before-quantize): the plain
#     quantizer leaves 96 zero + 748 subnormal fp16 block scales in 7 Mamba-mixer tensors of this model, and XNNPACK then refuses
#     the CPU path of the int4 file (it prepares a blockwise int4 FC only for normal scales)
#   5 checks: `litert-lm describe`, fc_dtypes.py (FC / embedding weight forms), inspect_bundle.py (sections, model type, stop tokens),
#     scan_scales.py (int4 block scales), sha256
# A 7B export loads the model in fp32 (~30 GB) and the converter needs a multiple of that: run ONE export (or quantization) at a
# time, on a machine with plenty of free RAM (measured on a 128 GiB Mac: export peak 61 GB in 78-83 minutes, of which ~60 are
# litert-torch's runtime constant folding; each quantization ~30 GB in 7-8 minutes).
# The file sha256 never reproduces (the packer writes a new uuid and timestamp); compare the section sha256 values with the table in
# REPRODUCE.md (inspect_bundle.py / the litert-lm-builder header read).
# Environments:
#   EXPORT_PY   a python with torch 2.12.1, transformers 5.14.1, ai-edge-quantizer 0.8.0, ai-edge-litert 2.1.6, litert-converter
#               0.3.0, litert-lm-builder 0.15.0, huggingface_hub (litert-torch comes from the clone on PYTHONPATH)
#   RUN_PY      a python with litert-lm 0.18.0 (litert-lm-builder 0.18.0 reads and verifies the metadata)
#   LITERT_LM   the litert-lm 0.18.0 CLI (unpack / pack / describe), default "$(dirname "$RUN_PY")/litert-lm"
#   CHECK_PY    a python with ai-edge-litert (the TFLite schema) + litert-lm-builder, for fc_dtypes.py / scan_scales.py
#               (default EXPORT_PY)
#   HF_HUB_CACHE optional: a Hugging Face cache that already holds the pinned commit (default OUTDIR/hf_cache, a 15.2 GB download);
#               never the shared ~/.cache/huggingface/hub (fetch_pinned.py rewrites refs/main)
#   KEEP_FLOAT=1 keeps OUTDIR/out/model.litertlm
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
[ $# -ge 1 ] || { echo "usage: $0 OUTDIR [HF id] [REVISION]"; exit 2; }
mkdir -p "$1"; OUT=$(cd "$1" && pwd)
HFID=${2:-tiiuae/Falcon-H1R-7B}; REV=${3:-a6f74bf181389908efd6970878d7ee2b42f5d417}
NAME=$(basename "$HFID")
BASE=115a13607c730c81018bb9789138a3e5e5119e3d
PATCH=$HERE/falcon_h1_litert_torch.patch
EPY=${EXPORT_PY:-python}; RPY=${RUN_PY:-python}; CPY=${CHECK_PY:-$EPY}
CLI=${LITERT_LM:-$(dirname "$(command -v "$RPY")")/litert-lm}
WT=$OUT/litert-torch-falcon
export HF_HUB_CACHE=${HF_HUB_CACHE:-$OUT/hf_cache}
SNAP=$HF_HUB_CACHE/models--${HFID/\//--}/snapshots/$REV
log() { echo "$*" >> "$OUT/recipe.log"; }
run() { log "### CMD: $*"; "$@"; }
log "### start $(date '+%F %T %Z') $HFID @ $REV HF_HUB_CACHE=$HF_HUB_CACHE"
if [ ! -d "$WT" ]; then
  if [ -n "${TORCH_SRC:-}" ]; then
    run git -C "$TORCH_SRC" worktree add --detach "$WT" "$BASE"
  else
    run git clone https://github.com/john-rocky/litert-torch "$WT"
    run git -C "$WT" fetch origin "$BASE"
    run git -C "$WT" checkout --detach "$BASE"
  fi
  run git -C "$WT" apply "$PATCH"
fi
log "### litert-torch $(git -C "$WT" rev-parse HEAD) status: $(git -C "$WT" status --short | tr '\n' ' ')"
[ -d "$SNAP" ] || run env HF_HUB_DISABLE_XET=1 "$EPY" -I "$HERE/fetch_pinned.py" "$HF_HUB_CACHE" "$OUT/hf_cache_manifest.json" "$HFID" "$REV"
export HF_HUB_OFFLINE=1
mkdir -p "$OUT/tmp" "$OUT/out"
ENV=(env PYTHONPATH="$WT" PATH="$(dirname "$CLI"):$PATH" RESOURCE_CONSTANT_NUMEL_THRESHOLD=100000000 FH1R_NAME="$NAME"
     FH1R_RECIPE_LOG="$OUT/recipe.log" TMPDIR="$OUT/tmp" RUN_PY="$RPY" LITERT_LM="$CLI")
[ -s "$OUT/out/model.litertlm" ] || run "${ENV[@]}" STOP_AFTER_EXPORT=1 "$EPY" "$HERE/convert_falcon_h1r.py" "$HFID" "$OUT/out" > "$OUT/export.log" 2>&1
[ -s "$OUT/${NAME}_int8.litertlm" ] || run "${ENV[@]}" SKIP_EXPORT=1 RECIPE=wi8fc FH1R_WORK="$OUT/work_int8" \
  "$EPY" "$HERE/convert_falcon_h1r.py" "$HFID" "$OUT/out" "$OUT/${NAME}_int8.litertlm" > "$OUT/int8.log" 2>&1
[ -s "$OUT/${NAME}_int4.litertlm" ] || run "${ENV[@]}" SKIP_EXPORT=1 RECIPE=wi4b32_wi8 FH1R_WORK="$OUT/work_int4" \
  "$EPY" "$HERE/convert_falcon_h1r.py" "$HFID" "$OUT/out" "$OUT/${NAME}_int4.litertlm" > "$OUT/int4.log" 2>&1
for q in int8 int4; do
  F=$OUT/${NAME}_$q.litertlm
  run "$CLI" describe "$F" < /dev/null > "$OUT/describe_$q.txt" 2>&1
  run "$CPY" -I "$HERE/fc_dtypes.py" "$F" "$OUT/fc_dtypes_$q.json" '(?!x)x' > "$OUT/fc_dtypes_$q.txt" 2>&1
  run "$RPY" -I "$HERE/inspect_bundle.py" "$F" "$OUT/inspect_$q.json" > "$OUT/inspect_$q.txt" 2>&1
done
run "$CPY" -I "$HERE/scan_scales.py" "$OUT/${NAME}_int4.litertlm" "$OUT/scan_int4.json" > "$OUT/scan_int4.txt" 2>&1 || true
[ -n "${KEEP_FLOAT:-}" ] || { log "### removed $(stat -f %z "$OUT/out/model.litertlm" 2>/dev/null || stat -c %s "$OUT/out/model.litertlm") $OUT/out/model.litertlm"; rm -f "$OUT/out/model.litertlm"; }
shasum -a 256 "$OUT/${NAME}_int8.litertlm" "$OUT/${NAME}_int4.litertlm" | tee -a "$OUT/recipe.log"
log "### done $(date '+%F %T %Z')"
