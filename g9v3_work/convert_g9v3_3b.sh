#!/usr/bin/env bash
# convert_g9v3_3b.sh OUTDIR - ai9stars/G9v3-3B (pinned d9553445ff92dbce667381954c9699fbcbc924f9, Apache-2.0) ->
# OUTDIR/G9v3-3B_int8.litertlm + OUTDIR/G9v3-3B_int4.litertlm in one command; every step logs its command to OUTDIR/recipe.log.
# Paths are relative to this repository (scripts/, recipes/, g9v3_work/). Interpreters (env, required):
#   EPY = python with litert-torch 0.9.4, litert-converter 0.4.0, ai-edge-quantizer 0.9.0, transformers 5.14.1 (export, ingest)
#   RPY = python with litert-lm 0.18.0 (litert-lm-builder 0.18.0: metadata dump / repack)   CLI = that venv's `litert-lm`
#   GPY = python 3.12 with gptqmodel 7.3.5 + datasets + requests + torchvision, patched once with
#         `GPY g9v3_work/patch_gptqmodel_mps.py` (Apple silicon / MPS; the GPTQ step takes about 25 min on an M4 Max)
#   0 checkpoint: the pinned commit in a dedicated HF cache (g9v3_work/fetch_pinned.py; reused when HF_HUB_CACHE holds it), offline
#     afterwards
#   1 export (scripts/export_simple_template.py, embedded template g9v3_work/g9v3.jinja, KV 4096, prefill signatures
#     1024,256,64,16,4,1, embedder externalized, HF tokenizer.json, start token kept: the runtime renders the template's bos_token as ""
#     and prepends the start token once):
#       int8 = dynamic_wi8_afp32 (channelwise int8 weights, float activations)
#       int4 = GPTQ-calibrated int4 in three steps:
#              a. g9v3_work/gptq_quantize.py OUTDIR/gptq_g128 OUTDIR/calib_g9v3_256.jsonl (sym, desc_act false, group 128, damp 0.01,
#                 true_sequential; 256 calibration samples built from three datasets at pinned revisions and checked by sha256)
#              b. scripts/ingest_gptq_dequant.py OUTDIR/gptq_g128 ai9stars/G9v3-3B OUTDIR/hf_gptq fp32clip (dequantized fp32 checkpoint)
#              c. export from OUTDIR/hf_gptq with recipes/gptqrec_int4_block128.json (dequantized_weight_recovery = blockwise-128 int4
#                 on the GPTQ grid, int8 lm_head, int8 embedding) + GPTQREC_GCD_FIX=1, through g9v3_work/export_clamped.py (block
#                 scales below 2^-14 are raised to 2^-14 before the payload is written; this checkpoint has 365 GPTQ scales below
#                 2^-14 and 416 all-zero blocks, which XNNPACK refuses as subnormal / zero fp16 scales)
#   2 metadata (RPY): g9v3_work/make_pbtext.py dump + edit (stops = ids 1 + 130073, no sampler_params, channel thought '<think>\n' /
#     '\n</think>', supports_thinking true, supports_function_calling false, the template = g9v3_work/g9v3.jinja), int4 scale scan
#     (g9v3_work/scan_scales.py: no zero / subnormal block scale), then g9v3_work/repack_metadata.py (every other section identical)
#   3 GPU activations, declared per file on both TFLiteModel sections (scripts/set_activation_type.py on the undeclared file - the
#     script does not rewrite an existing declaration; weights byte-identical; the CPU path does not read the declaration):
#       int8 = fp32 (with fp16 GPU activations the int8 file scored GSM8K 69 vs 84 with fp32 and 85 on CPU)
#       int4 = fp16 (GSM8K GPU fp16 81, fp32 78, CPU 78)
#   4 `litert-lm describe` of both files
# Every section of both files reproduced byte for byte (the whole-file sha256 differs: the builder writes a new uuid and timestamp).
set -euo pipefail
R=$(cd "$(dirname "$0")/.." && pwd); W=$R/g9v3_work
[ $# -ge 1 ] || { echo "usage: $0 OUTDIR"; exit 2; }
mkdir -p "$1"; OUT=$(cd "$1" && pwd)
: "${EPY:?set EPY}" "${RPY:?set RPY}" "${GPY:?set GPY}" "${CLI:?set CLI}"
TEMPLATE=$W/g9v3.jinja; PREFILL=1024,256,64,16,4,1
export HF_HUB_CACHE=${HF_HUB_CACHE:-$OUT/hf_cache}
log() { echo "$*" >> "$OUT/recipe.log"; }
run() { log "### CMD: $*"; "$@"; }
log "### start $(date '+%F %T %Z') template=$TEMPLATE ($(shasum -a 256 "$TEMPLATE" | cut -c1-16)) HF_HUB_CACHE=$HF_HUB_CACHE"
if [ ! -d "$HF_HUB_CACHE/models--ai9stars--G9v3-3B/snapshots/d9553445ff92dbce667381954c9699fbcbc924f9" ]; then
  run "$EPY" "$W/fetch_pinned.py" "$HF_HUB_CACHE" "$OUT/hf_cache_manifest.json"
fi
export HF_HUB_OFFLINE=1
for q in int8 int4; do
  SRC=ai9stars/G9v3-3B; XENV=
  if [ $q = int8 ]; then RECIPE=dynamic_wi8_afp32; SCRIPT=$R/scripts/export_simple_template.py
  else
    RECIPE=$R/recipes/gptqrec_int4_block128.json; SCRIPT=$W/export_clamped.py; SRC=$OUT/hf_gptq; XENV=GPTQREC_GCD_FIX=1
    if [ ! -s "$OUT/gptq_g128/model.safetensors" ]; then
      run env HF_HUB_OFFLINE=0 "$GPY" "$W/gptq_quantize.py" "$OUT/gptq_g128" "$OUT/calib_g9v3_256.jsonl" > "$OUT/gptq.log" 2>&1
    fi
    [ -s "$OUT/hf_gptq/model.safetensors" ] || \
      run "$EPY" "$R/scripts/ingest_gptq_dequant.py" "$OUT/gptq_g128" ai9stars/G9v3-3B "$OUT/hf_gptq" fp32clip > "$OUT/ingest.log" 2>&1
  fi
  run env $XENV USE_JINJA=1 PREFILL=$PREFILL CACHE=4096 EXTERNALIZE_EMBEDDER=1 \
    "$EPY" "$SCRIPT" "$SRC" "$OUT/out_$q" "$TEMPLATE" $RECIPE > "$OUT/export_$q.log" 2>&1
  run "$RPY" -I "$W/make_pbtext.py" dump "$OUT/out_$q/model.litertlm" "$OUT/raw_$q.pbtext"
  run "$RPY" -I "$W/make_pbtext.py" edit "$OUT/raw_$q.pbtext" "$OUT/$q.pbtext" "$OUT/$q.edit.json" --jinja "$TEMPLATE"
  EXPECT=$(python3 -c 'import json,sys; print(",".join(json.load(open(sys.argv[1]))["changed_fields"]))' "$OUT/$q.edit.json")
  if [ $q = int4 ]; then run "$EPY" -I "$W/scan_scales.py" "$OUT/out_$q/model.litertlm" "$OUT/scan_$q.json"; fi
  run "$RPY" -I "$W/repack_metadata.py" "$OUT/out_$q/model.litertlm" "$OUT/meta_$q.litertlm" \
    --pbtext "$OUT/$q.pbtext" --report "$OUT/repack_$q.json" --expect-changed "$EXPECT" --cli "$CLI"
  ACT=fp32
  if [ $q = int4 ]; then ACT=fp16; fi
  PATH=$(dirname "$CLI"):$PATH run python3 "$R/scripts/set_activation_type.py" "$OUT/meta_$q.litertlm" "$OUT/G9v3-3B_$q.litertlm" \
    --type $ACT --litert-lm "$CLI" > "$OUT/set_${ACT}act_$q.log" 2>&1
  run "$CLI" describe "$OUT/G9v3-3B_$q.litertlm" < /dev/null > "$OUT/describe_$q.txt" 2>&1
done
shasum -a 256 "$OUT"/G9v3-3B_*.litertlm | tee -a "$OUT/recipe.log"
log "### done $(date '+%F %T %Z')"
