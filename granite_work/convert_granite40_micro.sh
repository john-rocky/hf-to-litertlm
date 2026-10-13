#!/usr/bin/env bash
# convert_granite40_micro.sh OUTDIR [6|11] - ibm-granite/granite-4.0-micro (pinned 56111ae135df9c53a78c99028e7bc24035a9e979,
# Apache-2.0) -> OUTDIR/granite-4.0-micro_int8[_11sig].litertlm + OUTDIR/granite-4.0-micro_int4[_11sig].litertlm in one command;
# the two files of litert-community/granite-4.0-micro are the 6-signature build. Every step logs its command to OUTDIR/recipe.log.
#   0 checkpoint: the pinned commit in a scratch HF cache (fetch_pinned.py: LFS sha256 + git blob sha1 of every file;
#     reused when the cache already holds it), offline afterwards
#   1 dense rewrite (to_dense_granite.py): granitemoehybrid with 40 attention layers and 0 experts -> GraniteForCausalLM
#     (shared_mlp.input_linear rows 0..8191 -> mlp.gate_proj, rows 8192..16383 -> mlp.up_proj, output_linear -> down_proj;
#     config = granite-4.1-3b's key set), weights moved byte for byte
#   2 proof (dense_check.py, fp32 CPU): original vs rewrite on 3 prompts - max |logit delta| <= 1e-3, argmax on every
#     position, greedy 50 identical; the run stops here on FAIL (SKIP_PROOF=1 skips it)
#   3 export (scripts/export_simple_template.py) with USE_JINJA=1 (the vendor chat_template.jinja verbatim, default system
#     message included), NO_START_TOKEN=1 (bos == eos == <|end_of_text|>, add_bos_token False), CACHE=4096,
#     EXTERNALIZE_EMBEDDER=1, PREFILL = 1024,256,64,16,4,1 (6) or 1024,512,...,1 (11), the HF tokenizer.json:
#       int8 = dynamic_wi8_afp32 (channelwise int8 weights, float activations)
#       int4 = BOCTAV4 (blockwise-32 OCTAV int4 + int8 embedding)
#   4 metadata: make_pbtext.py dump + edit (stop = id 100257 only, supports_thinking false, supports_function_calling
#     false; checks: no start_token, no channels, max_num_tokens 4096, generic_model, jinja == the vendor file),
#     repack_metadata.py (every other section proven identical; OUTDIR/repack_<q>.json lists each section's sha256)
#   5 checks: int4 scale scan (scan_scales.py, all normal), `litert-lm describe` of both files, file sha256
# The file sha256 never reproduces (the packer writes a new uuid and timestamp); compare the section sha256 values in
# OUTDIR/repack_int8.json / repack_int4.json (`sections[].new_sha256`) with the table in REPRODUCE.md.
# Environments (released wheels):
#   EXPORT_PY       a python with litert-torch==0.9.4 litert-converter==0.4.0 ai-edge-quantizer==0.9.0 transformers==5.14.1
#                   (torch 2.13.0)
#   RUN_PY          a python with litert-lm==0.18.0 (its litert-lm-builder reads and verifies the metadata)
#   LITERT_LM       the litert-lm 0.18.0 CLI (unpack / pack / describe), default "$(dirname "$RUN_PY")/litert-lm"
#   GRANITE_HF_CACHE optional: a Hugging Face cache that already holds the pinned commit (default OUTDIR/hf_cache, a 6.8 GB
#                   download); never the shared ~/.cache/huggingface/hub (fetch_pinned.py rewrites refs/main)
#   SKIP_PROOF=1    skip step 2
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
[ $# -ge 1 ] || { echo "usage: $0 OUTDIR [6|11]"; exit 2; }
mkdir -p "$1"; OUT=$(cd "$1" && pwd); SHAPE=${2:-6}
EPY=${EXPORT_PY:-python}; RPY=${RUN_PY:-python}
CLI=${LITERT_LM:-$(dirname "$(command -v "$RPY")")/litert-lm}
PIN=56111ae135df9c53a78c99028e7bc24035a9e979
export HF_HUB_CACHE=${GRANITE_HF_CACHE:-$OUT/hf_cache}
SNAP=$HF_HUB_CACHE/models--ibm-granite--granite-4.0-micro/snapshots/$PIN
DENSE=$OUT/src_dense/granite-4.0-micro-dense
case "$SHAPE" in
  6) PREFILL=1024,256,64,16,4,1; SUF= ;;
  11) PREFILL=1024,512,256,128,64,32,16,8,4,2,1; SUF=_11sig ;;
  *) echo "shape must be 6 or 11"; exit 2 ;;
esac
log() { echo "$*" >> "$OUT/recipe.log"; }
run() { log "### CMD: $*"; "$@"; }
log "### start $(date '+%F %T %Z') shape=$SHAPE HF_HUB_CACHE=$HF_HUB_CACHE"
[ -d "$SNAP" ] || run "$EPY" "$HERE/fetch_pinned.py" "$HF_HUB_CACHE" "$OUT/hf_cache_manifest.json"
export HF_HUB_OFFLINE=1
[ -d "$DENSE" ] || run "$EPY" -I "$HERE/to_dense_granite.py" "$SNAP" "$DENSE" "$OUT/to_dense_report.json"
if [ -z "${SKIP_PROOF:-}" ] && [ ! -e "$OUT/dense_check.json" ]; then
  run "$EPY" -I "$HERE/dense_check.py" "$SNAP" "$DENSE" "$OUT/dense_check.json"
fi
export VENDOR_TEMPLATE=$SNAP/chat_template.jinja
for q in int8 int4; do
  if [ $q = int8 ]; then RECIPE=dynamic_wi8_afp32; else RECIPE=BOCTAV4; fi
  run env USE_JINJA=1 NO_START_TOKEN=1 PREFILL=$PREFILL CACHE=4096 EXTERNALIZE_EMBEDDER=1 \
    "$EPY" "$ROOT/scripts/export_simple_template.py" "$DENSE" "$OUT/out_$q$SUF" "$DENSE/chat_template.jinja" $RECIPE > "$OUT/export_$q$SUF.log" 2>&1
  run "$RPY" -I "$HERE/make_pbtext.py" dump "$OUT/out_$q$SUF/model.litertlm" "$OUT/raw_$q$SUF.pbtext"
  run "$RPY" -I "$HERE/make_pbtext.py" edit "$OUT/raw_$q$SUF.pbtext" "$OUT/$q$SUF.pbtext" "$OUT/$q$SUF.edit.json"
  EXPECT=$(python3 -c 'import json,sys; print(",".join(json.load(open(sys.argv[1]))["changed_fields"]))' "$OUT/$q$SUF.edit.json")
  if [ $q = int4 ]; then run "$EPY" -I "$HERE/scan_scales.py" "$OUT/out_$q$SUF/model.litertlm" "$OUT/scan_$q$SUF.json"; fi
  run "$RPY" -I "$HERE/repack_metadata.py" "$OUT/out_$q$SUF/model.litertlm" "$OUT/granite-4.0-micro_$q$SUF.litertlm" \
    --pbtext "$OUT/$q$SUF.pbtext" --report "$OUT/repack_$q$SUF.json" --expect-changed "$EXPECT" --cli "$CLI"
  run "$CLI" describe "$OUT/granite-4.0-micro_$q$SUF.litertlm" < /dev/null > "$OUT/describe_$q$SUF.txt" 2>&1
done
shasum -a 256 "$OUT"/granite-4.0-micro_*.litertlm | tee -a "$OUT/recipe.log"
log "### done $(date '+%F %T %Z')"
