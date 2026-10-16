#!/usr/bin/env bash
# convert_olmo3_7b_think.sh OUTDIR - allenai/Olmo-3-7B-Think (pinned d97e442d7cc678210054dbcc9b440894d62c89a4, Apache-2.0,
# Olmo3ForCausalLM 32L dense: 24 sliding-window (4096) + 8 full-attention layers, MHA 32:32, full-width QK-norm, post-norm, YaRN
# RoPE on the full layers) -> OUTDIR/Olmo-3-7B-Think_int8.litertlm + OUTDIR/Olmo-3-7B-Think_int4.litertlm, the two files of
# litert-community/Olmo-3-7B-Think (cache 4096), + OUTDIR/Olmo-3-7B-Think_int4_c2048.litertlm (the same int4 weights with a
# 2048-token cache), in one command. Every step logs its command to OUTDIR/recipe.log.
#   0 checkpoint: the pinned commit in a scratch HF cache (olmo3_work/fetch_pinned.py: LFS sha256 + git blob sha1 of every file,
#     files one at a time, HF_HUB_DISABLE_XET=1; reused when HF_HUB_CACHE already holds it), offline afterwards
#   1 export: granite41_work/export_lightweight.py (= scripts/export_simple_template.py + export(...,
#     experimental_lightweight_conversion=True) under LIGHTWEIGHT=1) with RESOURCE_CONSTANT_NUMEL_THRESHOLD=100000000 (bytes),
#     USE_JINJA=1 with olmo3_work/olmo3_think_dual.jinja (Ai2's chat_template.jinja + the content-parts macro, `.get` rewritten; the
#     generation prompt ends `<|im_start|>assistant\n<think>`; built by olmo3_work/make_template.py), NO_START_TOKEN=1 (bos == eos
#     == <|endoftext|>; the vendor render starts at <|im_start|> and transformers prepends nothing), EXTERNALIZE_EMBEDDER=1,
#     PREFILL=1024,256,64,16,4,1, the HF tokenizer.json path (no FORCE_SPM). The HF modeling is traced as is (the sliding layers:
#     litert-torch's sliding mask over a full-length cache; with a cache no longer than the window it is the causal mask; YaRN
#     cos / sin on the 8 full layers):
#       int8       = dynamic_wi8_afp32 (channelwise int8 weights, float activations), CACHE=4096
#       int4       = BOCTAV4 (blockwise-32 OCTAV int4 + int8 embedding), CACHE=4096, through olmo3_work/export_clamped_lw.py =
#                    every block scale clamped to at least 2^-14 before the weights are quantized (ai-edge-quantizer 0.9.0 writes 1,574,919 zero +
#                    2,031,784 subnormal fp16 block scales into this model's FC weights without it; XNNPACK refuses non-normal
#                    scales; the scan in step 3 stops the script if any is left)
#       int4_c2048 = BOCTAV4 the same way, CACHE=2048
#     An 8B-class export loads the model in fp32 (~30 GB) and peaks near 55 GB with LIGHTWEIGHT=1 (measured on a 128 GB Mac:
#     52.6-56.9 GB per export): run ONE export at a time, on a machine with plenty of free RAM.
#   2 metadata (litert-lm 0.18.0): olmo3_work/make_pbtext.py dump + edit (stop = ids 100265 + 100257 only, supports_thinking
#     true, supports_function_calling false, sampler_params cleared if present; checks: no start_token, channel thought
#     `<think>` / `</think>`, max_num_tokens 4096 / 2048, generic_model, jinja == olmo3_think_dual.jinja)
#   3 tokenizer section = the repo's tokenizer.json bytes (tools/tokenizer_parity/swap_tokenizer_section.py when the exporter's
#     save_pretrained re-write differs; olmo3_work/tok_section.py checks), the int4 scale scan (granite_work/scan_scales.py),
#     granite_work/repack_metadata.py (every other section identical), `litert-lm describe`
#   4 graph rewrite (the GPU-composites form; off by default, GPU_GRAPH=1 runs it): tools/gpu_graph/gpu_graph_retrofit.py
#     X.litertlm X_gpu_graph.litertlm --decomp exporter --mask add_bcast refuses this graph ("prefill_1024: mask ADD 7 adds to
#     [1, 1024, 4096], not a [bk, g, T, C] view": the mask input also feeds the sliding-window layers' mask construction,
#     litert-torch exportable_module.py create_sliding_mask(...) + mask), so the published files are the exporter's graphs
#   5 sha256 of every bundle
# The file sha256 never reproduces (the packer writes a new uuid and timestamp); compare the section sha256 values in
# OUTDIR/repack_{int8,int4,int4_c2048}.json with the table in REPRODUCE.md.
# Environments (released wheels):
#   EXPORT_PY   a python with litert-torch==0.9.4 litert-converter==0.4.0 ai-edge-quantizer==0.9.0 transformers==5.14.1
#               (torch 2.13.0)
#   RUN_PY      a python with litert-lm==0.18.0 (its litert-lm-builder reads and verifies the metadata)
#   LITERT_LM   the litert-lm 0.18.0 CLI (unpack / pack / describe), default "$(dirname "$RUN_PY")/litert-lm"
#   HF_HUB_CACHE optional: a Hugging Face cache that already holds the pinned commit (default OUTDIR/hf_cache, a 14.6 GB
#               download); never the shared ~/.cache/huggingface/hub (fetch_pinned.py rewrites refs/main)
#   GPU_GRAPH=1 try step 4 (it fails on this model)
#   KEEP_RAW=1  keep OUTDIR/out_*/model.litertlm (default: removed once its file is built)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
[ $# -ge 1 ] || { echo "usage: $0 OUTDIR"; exit 2; }
mkdir -p "$1"; OUT=$(cd "$1" && pwd)
EPY=${EXPORT_PY:-python}; RPY=${RUN_PY:-python}
CLI=${LITERT_LM:-$(dirname "$(command -v "$RPY")")/litert-lm}
PIN=d97e442d7cc678210054dbcc9b440894d62c89a4
NAME=Olmo-3-7B-Think
export HF_HUB_CACHE=${HF_HUB_CACHE:-$OUT/hf_cache}
SNAP=$HF_HUB_CACHE/models--allenai--Olmo-3-7B-Think/snapshots/$PIN
TPL=$HERE/olmo3_think_dual.jinja
PREFILL=1024,256,64,16,4,1
log() { echo "$*" >> "$OUT/recipe.log"; }
run() { log "### CMD: $*"; "$@"; }
log "### start $(date '+%F %T %Z') HF_HUB_CACHE=$HF_HUB_CACHE"
[ -d "$SNAP" ] || run env HF_HUB_DISABLE_XET=1 "$EPY" "$HERE/fetch_pinned.py" "$HF_HUB_CACHE" "$OUT/hf_cache_manifest.json"
export HF_HUB_OFFLINE=1
for q in int8 int4 int4_c2048; do
  case $q in int8) RECIPE=dynamic_wi8_afp32; CACHE=4096 ;; int4) RECIPE=BOCTAV4; CACHE=4096 ;; int4_c2048) RECIPE=BOCTAV4; CACHE=2048 ;; esac
  XS=$ROOT/granite41_work/export_lightweight.py; [ $RECIPE = BOCTAV4 ] && XS=$HERE/export_clamped_lw.py
  run env USE_JINJA=1 NO_START_TOKEN=1 PREFILL=$PREFILL CACHE=$CACHE EXTERNALIZE_EMBEDDER=1 LIGHTWEIGHT=1 RESOURCE_CONSTANT_NUMEL_THRESHOLD=100000000 \
    "$EPY" "$XS" "$SNAP" "$OUT/out_$q" "$TPL" $RECIPE > "$OUT/export_$q.log" 2>&1
  MAXN=4096; [ $q = int4_c2048 ] && MAXN=2048
  run "$RPY" -I "$HERE/make_pbtext.py" dump "$OUT/out_$q/model.litertlm" "$OUT/raw_$q.pbtext"
  run env MAX_NUM=$MAXN "$RPY" -I "$HERE/make_pbtext.py" edit "$OUT/raw_$q.pbtext" "$OUT/$q.pbtext" "$OUT/$q.edit.json"
  EXPECT=$(python3 -c 'import json,sys; print(",".join(json.load(open(sys.argv[1]))["changed_fields"]))' "$OUT/$q.edit.json")
  SRC=$OUT/out_$q/model.litertlm
  rc=0; run "$RPY" -I "$HERE/tok_section.py" "$SRC" "$SNAP/tokenizer.json" "$OUT/tok_raw_$q.json" || rc=$?
  if [ $rc = 1 ]; then
    # the swapped copy goes to OUTDIR/q/: repack_metadata.py refuses an output (bundle, report, scratch) in its input's directory
    mkdir -p "$OUT/q"
    run env LITERT_LM_CLI="$CLI" "$RPY" -I "$ROOT/tools/tokenizer_parity/swap_tokenizer_section.py" "$SRC" "$SNAP/tokenizer.json" "$OUT/q/${q}_tok.litertlm"
    SRC=$OUT/q/${q}_tok.litertlm
    run "$RPY" -I "$HERE/tok_section.py" "$SRC" "$SNAP/tokenizer.json" "$OUT/tok_swapped_$q.json"
  elif [ $rc != 0 ]; then exit $rc; fi
  case $q in int4*) run "$EPY" -I "$ROOT/granite_work/scan_scales.py" "$SRC" "$OUT/scan_$q.json" ;; esac
  run "$RPY" -I "$ROOT/granite_work/repack_metadata.py" "$SRC" "$OUT/${NAME}_$q.litertlm" \
    --pbtext "$OUT/$q.pbtext" --report "$OUT/repack_$q.json" --expect-changed "$EXPECT" --cli "$CLI"
  run "$CLI" describe "$OUT/${NAME}_$q.litertlm" < /dev/null > "$OUT/describe_$q.txt" 2>&1
  run "$RPY" -I "$HERE/tok_section.py" "$OUT/${NAME}_$q.litertlm" "$SNAP/tokenizer.json" "$OUT/tok_final_$q.json"
  rm -f "$OUT/q/${q}_tok.litertlm"
  [ -n "${KEEP_RAW:-}" ] || rm -f "$OUT/out_$q/model.litertlm"
done
if [ "${GPU_GRAPH:-0}" != 0 ]; then
  for q in int8 int4 int4_c2048; do
    run env LITERT_LM_CLI="$CLI" "$EPY" "$ROOT/tools/gpu_graph/gpu_graph_retrofit.py" "$OUT/${NAME}_$q.litertlm" \
      "$OUT/${NAME}_${q}_gpu_graph.litertlm" --decomp exporter --mask add_bcast --report "$OUT/gpu_graph_$q.json" > "$OUT/gpu_graph_$q.log" 2>&1
    run "$CLI" describe "$OUT/${NAME}_${q}_gpu_graph.litertlm" < /dev/null > "$OUT/describe_${q}_gpu_graph.txt" 2>&1
  done
fi
shasum -a 256 "$OUT"/${NAME}_*.litertlm | tee -a "$OUT/recipe.log"
log "### done $(date '+%F %T %Z')"
