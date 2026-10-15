#!/usr/bin/env bash
# convert_granite41_8b.sh OUTDIR - ibm-granite/granite-4.1-8b (pinned 1504002f650e656a0a3789d99574df12e3e94ed0, Apache-2.0,
# GraniteForCausalLM 40L / hidden 4096 / tied 100352 x 4096 embeddings) -> OUTDIR/granite-4.1-8b_int8.litertlm +
# OUTDIR/granite-4.1-8b_int4.litertlm, the two files of litert-community/granite-4.1-8b, in one command (the exported graphs are
# kept as OUTDIR/granite-4.1-8b_{int8,int4}_plain.litertlm). Every step logs its command to OUTDIR/recipe.log.
#   0 checkpoint: the pinned commit in a scratch HF cache (fetch_pinned.py: LFS sha256 + git blob sha1 of every file, files one
#     at a time, HF_HUB_DISABLE_XET=1; reused when HF_HUB_CACHE already holds it), offline afterwards
#   1 export (export_lightweight.py = scripts/export_simple_template.py + export(..., experimental_lightweight_conversion=True)
#     under LIGHTWEIGHT=1: large constants stay lazy through the MLIR lowering and are folded by the LiteRT runtime afterwards;
#     an export of this model through the default path needed more than 92 GiB of RAM in the converter passes) with
#     RESOURCE_CONSTANT_NUMEL_THRESHOLD=100000000 (bytes: only constants above 1e8 B - the MLP weights and the embedding table -
#     stay lazy; with the 1 MiB default the converter left o_proj x residual_multiplier as 40 fp32 weights behind an unfolded
#     MUL), USE_JINJA=1 (the vendor chat_template.jinja verbatim; it has no default system message), NO_START_TOKEN=1 (bos ==
#     eos == <|end_of_text|>, add_bos_token False), CACHE=4096, EXTERNALIZE_EMBEDDER=1, PREFILL=1024,256,64,16,4,1, the HF
#     tokenizer.json (no FORCE_SPM):
#       int8 = dynamic_wi8_afp32 (channelwise int8 weights, float activations)
#       int4 = BOCTAV4 (blockwise-32 OCTAV int4 + int8 embedding)
#     An 8B export loads the model in fp32 (35 GB) and the converter needs more than that: run ONE export at a time, on a machine
#     with plenty of free RAM (measured on a 128 GB Mac: 50-56 GiB at the peak of each export).
#   2 metadata (litert-lm 0.18.0): granite_work/make_pbtext.py dump + edit with VENDOR_TEMPLATE = the snapshot's
#     chat_template.jinja (stop = id 100257 only, supports_thinking false, supports_function_calling false; checks: no
#     start_token, no channels, max_num_tokens 4096, generic_model, jinja == the vendor file), granite_work/repack_metadata.py
#     (every other section identical) -> X_plain.litertlm
#   3 checks: int4 scale scan (granite_work/scan_scales.py, all normal), `litert-lm describe` of the files, sha256
#   4 GPU graph (weights byte-identical; GPU_GRAPH=0 skips it and leaves the _plain files as the result): for int8 and int4,
#     tools/gpu_graph/gpu_graph_retrofit.py X_plain.litertlm X.litertlm --decomp exporter --mask add_bcast
#     (unpack -> prefill_decode: the two DYNAMIC_UPDATE_SLICE KV writes per layer -> odml.cache_update, BATCH_MATMUL ->
#     odml.runtime_bmm, + param_tensor -> pack with the litert-lm 0.18.0 CLI; every other section asserted byte-identical)
# The file sha256 never reproduces (the packer writes a new uuid and timestamp); compare the section sha256 values in
# OUTDIR/gpu_graph_int8.json / gpu_graph_int4.json (and OUTDIR/repack_int{8,4}.json for the _plain files) with the table in
# REPRODUCE.md.
# Environments (released wheels):
#   EXPORT_PY   a python with litert-torch==0.9.4 litert-converter==0.4.0 ai-edge-quantizer==0.9.0 transformers==5.14.1
#               (torch 2.13.0)
#   RUN_PY      a python with litert-lm==0.18.0 (its litert-lm-builder reads and verifies the metadata)
#   LITERT_LM   the litert-lm 0.18.0 CLI (unpack / pack / describe), default "$(dirname "$RUN_PY")/litert-lm"
#   HF_HUB_CACHE optional: a Hugging Face cache that already holds the pinned commit (default OUTDIR/hf_cache, a 17.6 GB
#               download); never the shared ~/.cache/huggingface/hub (fetch_pinned.py rewrites refs/main)
#   GPU_GRAPH=0 skip step 4
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
[ $# -ge 1 ] || { echo "usage: $0 OUTDIR"; exit 2; }
mkdir -p "$1"; OUT=$(cd "$1" && pwd)
EPY=${EXPORT_PY:-python}; RPY=${RUN_PY:-python}
CLI=${LITERT_LM:-$(dirname "$(command -v "$RPY")")/litert-lm}
PIN=1504002f650e656a0a3789d99574df12e3e94ed0
NAME=granite-4.1-8b
export HF_HUB_CACHE=${HF_HUB_CACHE:-$OUT/hf_cache}
SNAP=$HF_HUB_CACHE/models--ibm-granite--granite-4.1-8b/snapshots/$PIN
PREFILL=1024,256,64,16,4,1
log() { echo "$*" >> "$OUT/recipe.log"; }
run() { log "### CMD: $*"; "$@"; }
log "### start $(date '+%F %T %Z') HF_HUB_CACHE=$HF_HUB_CACHE"
[ -d "$SNAP" ] || run env HF_HUB_DISABLE_XET=1 "$EPY" "$HERE/fetch_pinned.py" "$HF_HUB_CACHE" "$OUT/hf_cache_manifest.json"
export HF_HUB_OFFLINE=1
export VENDOR_TEMPLATE=$SNAP/chat_template.jinja
for q in int8 int4; do
  if [ $q = int8 ]; then RECIPE=dynamic_wi8_afp32; else RECIPE=BOCTAV4; fi
  run env USE_JINJA=1 NO_START_TOKEN=1 PREFILL=$PREFILL CACHE=4096 EXTERNALIZE_EMBEDDER=1 LIGHTWEIGHT=1 RESOURCE_CONSTANT_NUMEL_THRESHOLD=100000000 \
    "$EPY" "$HERE/export_lightweight.py" "$SNAP" "$OUT/out_$q" "$SNAP/chat_template.jinja" $RECIPE > "$OUT/export_$q.log" 2>&1
  run "$RPY" -I "$ROOT/granite_work/make_pbtext.py" dump "$OUT/out_$q/model.litertlm" "$OUT/raw_$q.pbtext"
  run "$RPY" -I "$ROOT/granite_work/make_pbtext.py" edit "$OUT/raw_$q.pbtext" "$OUT/$q.pbtext" "$OUT/$q.edit.json"
  EXPECT=$(python3 -c 'import json,sys; print(",".join(json.load(open(sys.argv[1]))["changed_fields"]))' "$OUT/$q.edit.json")
  if [ $q = int4 ]; then run "$EPY" -I "$ROOT/granite_work/scan_scales.py" "$OUT/out_$q/model.litertlm" "$OUT/scan_$q.json"; fi
  run "$RPY" -I "$ROOT/granite_work/repack_metadata.py" "$OUT/out_$q/model.litertlm" "$OUT/${NAME}_${q}_plain.litertlm" \
    --pbtext "$OUT/$q.pbtext" --report "$OUT/repack_$q.json" --expect-changed "$EXPECT" --cli "$CLI"
  run "$CLI" describe "$OUT/${NAME}_${q}_plain.litertlm" < /dev/null > "$OUT/describe_${q}_plain.txt" 2>&1
done
if [ "${GPU_GRAPH:-1}" != 0 ]; then
  for q in int8 int4; do
    run env LITERT_LM_CLI="$CLI" "$EPY" "$ROOT/tools/gpu_graph/gpu_graph_retrofit.py" "$OUT/${NAME}_${q}_plain.litertlm" \
      "$OUT/${NAME}_$q.litertlm" --decomp exporter --mask add_bcast --report "$OUT/gpu_graph_$q.json" > "$OUT/gpu_graph_$q.log" 2>&1
    run "$CLI" describe "$OUT/${NAME}_$q.litertlm" < /dev/null > "$OUT/describe_$q.txt" 2>&1
  done
fi
shasum -a 256 "$OUT"/${NAME}_*.litertlm | tee -a "$OUT/recipe.log"
log "### done $(date '+%F %T %Z')"
