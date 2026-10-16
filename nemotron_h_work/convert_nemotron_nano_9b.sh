#!/usr/bin/env bash
# convert_nemotron_nano_9b.sh OUTDIR [HF id] [REVISION] - nvidia/NVIDIA-Nemotron-Nano-9B-v2 (nemotron_h: 56 layers = 27 Mamba2 + 25 MLP +
# 4 attention; reasoning switched by `/think` / `/no_think` in the system or user text; NVIDIA Open Model License) ->
# OUTDIR/NVIDIA-Nemotron-Nano-9B-v2_int8.litertlm + OUTDIR/NVIDIA-Nemotron-Nano-9B-v2_int4.litertlm, the two files of
# litert-community/NVIDIA-Nemotron-Nano-9B-v2, in one command. Every step logs its command to OUTDIR/recipe.log.
#   0 litert-torch: a worktree of commit 115a13607c730c81018bb9789138a3e5e5119e3d (fetched from github.com/john-rocky/litert-torch,
#     where it stays reachable) + nemotron_h_litert_torch.patch (the Nemotron-H rail: folded rank <= 4 scan, pad guard, cache-less
#     MLP layer, the mixer class swapped in NemotronH's MIXER_TYPES table) + nemotron_h_vendor_dt.patch (the Mamba2 dt limit = NVIDIA's time_step_limit,
#     default (0.0, inf) = no clamp, instead of transformers' lower clamp at time_step_min = 0.001; 339 of this model's 3,456 heads have
#     softplus(dt_bias) < 0.001) -> OUTDIR/litert-torch-nemotron
#   1 checkpoint: REVISION (default 6533e8de2c68e4536bf7c411d7a3ce5734111476) into a scratch HF cache (fetch_pinned.py: LFS sha256 +
#     git blob sha1 of every file, one file at a time, HF_HUB_DISABLE_XET=1; reused when HF_HUB_CACHE already holds the snapshot),
#     offline afterwards
#   2 float export (convert_nemotron_nano_9b.py STOP_AFTER_EXPORT=1): litert-torch export_hf, prefill 1024,256,64,16,4,1 + decode,
#     KV cache 4096, experimental_lightweight_conversion with RESOURCE_CONSTANT_NUMEL_THRESHOLD=100000000 -> OUTDIR/out/model.litertlm
#     (35.6 GB). A 9B export loads the model in fp32 and the converter needs more: run ONE export at a time, on a machine with
#     plenty of free RAM (measured on a 128 GB Mac: export peak 40.8 GB, 34 min; each quantize step 35.5 GB, about 5 min with
#     TMPDIR on a local disk)
#   3 int8: the same script with SKIP_EXPORT=1 RECIPE=wi8fc = minicpm5_work/quantize_minicpm5.py (dynamic int8 FCs + int8 embedding;
#     convolutions and the scan stay float) -> LlmMetadata (nemotron_nano_dual.jinja, no start token, supports_thinking;
#     make_pbtext.py + granite_work/repack_metadata.py, litert-lm 0.18.0) -> scripts/add_executor_metadata.py ->
#     scripts/set_activation_type.py --type fp32 -> tools/add_thought_channel.py --start '<think>\n' --end '\n</think>'
#   4 int4: the same with RECIPE=wi4b32_wi8 (int4 blockwise-32 min-max FCs + int8 lm_head + int8 embedding) through
#     quantize_clamped.py (blockwise int4 scales floored at 2^-14 before quantizing)
#   5 checks: `litert-lm describe`, fc_dtypes.py (FC / embedding weight forms), inspect_bundle.py (sections, state buffers, metadata),
#     scan_scales.py (int4 block scales), sha256. The file sha256 never reproduces (the packer writes a new uuid and creation
#     timestamp); compare the section sha256 values (REPRODUCE.md) instead.
# Environments:
#   EXPORT_PY   a python 3.12 with torch==2.12.1 transformers==5.14.1 ai-edge-quantizer==0.8.0 litert-converter==0.3.0
#               ai-edge-litert==2.1.6 litert-lm-builder==0.15.0 huggingface_hub tokenizers (litert_torch itself comes from the
#               patched worktree on PYTHONPATH); default `python`
#   RUN_PY      a python with litert-lm==0.18.0 (its litert-lm-builder reads, edits and verifies the metadata); default `python`
#   LITERT_LM   the litert-lm 0.18.0 CLI (unpack / pack / describe); default next to RUN_PY
#   CHECK_PY    a python with ai-edge-litert and litert-lm-builder for fc_dtypes.py / scan_scales.py; default EXPORT_PY
#   LITERT_TORCH_SRC  a litert-torch clone that has commit 115a136 (default: cloned into OUTDIR/litert-torch-src)
#   HF_HUB_CACHE  optional: a Hugging Face cache that already holds the pinned commit (default OUTDIR/hf_cache, a 17.8 GB download);
#               never the shared ~/.cache/huggingface/hub (fetch_pinned.py rewrites refs/main)
#   KEEP_FLOAT=1  keep OUTDIR/out/model.litertlm
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
[ $# -ge 1 ] || { echo "usage: $0 OUTDIR [HF id] [REVISION]"; exit 2; }
mkdir -p "$1"; OUT=$(cd "$1" && pwd)
HFID=${2:-nvidia/NVIDIA-Nemotron-Nano-9B-v2}; REV=${3:-6533e8de2c68e4536bf7c411d7a3ce5734111476}
NAME=$(basename "$HFID")
BASE=115a13607c730c81018bb9789138a3e5e5119e3d
EPY=${EXPORT_PY:-python}; RPY=${RUN_PY:-python}
CLI=${LITERT_LM:-$(dirname "$(command -v "$RPY")")/litert-lm}
CPY=${CHECK_PY:-$EPY}
SRC=${LITERT_TORCH_SRC:-$OUT/litert-torch-src}
WT=$OUT/litert-torch-nemotron
export HF_HUB_CACHE=${HF_HUB_CACHE:-$OUT/hf_cache}
SNAP=$HF_HUB_CACHE/models--${HFID/\//--}/snapshots/$REV
log() { echo "$*" >> "$OUT/recipe.log"; }
run() { log "### CMD: $*"; "$@"; }
log "### start $(date '+%F %T %Z') $HFID @ $REV HF_HUB_CACHE=$HF_HUB_CACHE"
if [ ! -d "$WT" ]; then
  if [ ! -d "$SRC" ]; then
    run git clone https://github.com/john-rocky/litert-torch "$SRC"
    run git -C "$SRC" fetch origin "$BASE"
  fi
  run git -C "$SRC" worktree add --detach "$WT" "$BASE"
  run git -C "$WT" apply "$HERE/nemotron_h_litert_torch.patch"
  run git -C "$WT" apply "$HERE/nemotron_h_vendor_dt.patch"
fi
log "### worktree $(git -C "$WT" rev-parse HEAD) status: $(git -C "$WT" status --short | tr '\n' ' ')"
log "### nemotron_h/patch.py sha256 $(shasum -a 256 "$WT/litert_torch/generative/export_hf/model_ext/nemotron_h/patch.py" | cut -d' ' -f1)"
[ -d "$SNAP" ] || run env HF_HUB_DISABLE_XET=1 "$EPY" "$HERE/fetch_pinned.py" "$HF_HUB_CACHE" "$OUT/hf_cache_manifest.json" "$HFID" "$REV"
export HF_HUB_OFFLINE=1
mkdir -p "$OUT/tmp" "$OUT/out"
ENV=(env PYTHONPATH="$WT" PATH="$(dirname "$CLI"):$PATH" RUN_PY="$RPY" LITERT_LM="$CLI" RESOURCE_CONSTANT_NUMEL_THRESHOLD=100000000
     NN9B_NAME="$NAME" NN9B_RECIPE_LOG="$OUT/recipe.log" TMPDIR="$OUT/tmp")
[ -s "$OUT/out/model.litertlm" ] || run "${ENV[@]}" STOP_AFTER_EXPORT=1 "$EPY" "$HERE/convert_nemotron_nano_9b.py" "$HFID" "$OUT/out" \
  > "$OUT/export.log" 2>&1
[ -s "$OUT/${NAME}_int8.litertlm" ] || run "${ENV[@]}" SKIP_EXPORT=1 RECIPE=wi8fc NN9B_WORK="$OUT/work_int8" \
  "$EPY" "$HERE/convert_nemotron_nano_9b.py" "$HFID" "$OUT/out" "$OUT/${NAME}_int8.litertlm" > "$OUT/int8.log" 2>&1
[ -s "$OUT/${NAME}_int4.litertlm" ] || run "${ENV[@]}" SKIP_EXPORT=1 RECIPE=wi4b32_wi8 NN9B_WORK="$OUT/work_int4" \
  "$EPY" "$HERE/convert_nemotron_nano_9b.py" "$HFID" "$OUT/out" "$OUT/${NAME}_int4.litertlm" > "$OUT/int4.log" 2>&1
for q in int8 int4; do
  F=$OUT/${NAME}_$q.litertlm
  run "$CLI" describe "$F" < /dev/null > "$OUT/describe_$q.txt" 2>&1
  run "$CPY" "$HERE/fc_dtypes.py" "$F" "$OUT/fc_dtypes_$q.json" '(?!x)x' > "$OUT/fc_dtypes_$q.txt" 2>&1
  run "$RPY" "$HERE/inspect_bundle.py" "$F" "$OUT/inspect_$q.json" > "$OUT/inspect_$q.txt" 2>&1
done
run "$CPY" "$HERE/scan_scales.py" "$OUT/${NAME}_int4.litertlm" "$OUT/scan_int4.json" > "$OUT/scan_int4.txt" 2>&1 || true
[ -n "${KEEP_FLOAT:-}" ] || { log "### removed $(wc -c < "$OUT/out/model.litertlm" | tr -d ' ') $OUT/out/model.litertlm"; rm -f "$OUT/out/model.litertlm"; }
shasum -a 256 "$OUT/${NAME}_int8.litertlm" "$OUT/${NAME}_int4.litertlm" | tee -a "$OUT/recipe.log"
log "### done $(date '+%F %T %Z')"
