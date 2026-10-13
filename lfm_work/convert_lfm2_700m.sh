#!/usr/bin/env bash
# convert_lfm2_700m.sh OUTDIR - LiquidAI/LFM2-700M -> the two files of litert-community/LFM2-700M, one command:
#   OUTDIR/LFM2-700M_int8.litertlm  the exporter's default recipe (dynamic_wi8_afp32): channelwise int8 weights on the
#                                   linears, the ShortConv convolutions and the embedding; float activations
#   OUTDIR/LFM2-700M_int4.litertlm  int4 blockwise-32 (OCTAV) on the linears, channelwise int8 lm_head + embedding,
#                                   ShortConv convolutions float
# both with the LiteRT-LM models/lfm2 configuration (commit 86a7e6d0; max_num_tokens 4096, supports_thinking false),
# an 11-signature prefill ladder (1024 ... 1) and a 4096-token KV cache.
#
# Environments (released wheels):
#   EXPORT_PY      a python with litert-torch==0.9.4 litert-converter==0.4.0 ai-edge-quantizer==0.9.0 transformers==5.14.1
#                  (torch 2.13.0; transformers 5.15 breaks the lfm2 export)
#   RUN_PY         a python with litert-lm==0.18.0 (its litert-lm-builder 0.18.0 verifies the repack)
#   LITERT_LM      the litert-lm 0.18.0 CLI (unpack / pack), e.g. "$(dirname "$RUN_PY")/litert-lm"
#   LFM2_HF_CACHE  optional: a Hugging Face cache that already holds the pinned commit (default OUTDIR/hf_cache, a 1.5 GB
#                  download); never the shared ~/.cache/huggingface/hub (fetch_pinned.py rewrites refs/main)
#
# Steps: 0 the checkpoint at commit 86f49fc9, every file checked against the Hub -> 1 export x2: the exporter's default
# recipe (the int8 file's weights) and unquantized (the int4 file's base) -> 2 post-hoc int4, every section kept ->
# 3 int4 zero-scale patch (a measured no-op for this checkpoint: 92 blockwise tensors, no zero or subnormal scale) ->
# 4 the lfm2 configuration x2, every other section proven identical. The file sha256 never reproduces (the packer writes
# a new uuid and timestamp); compare the SECTION sha256 lines that step 4 prints with the table in REPRODUCE.md.
# Exit 0 = every step ran and every check passed; the first failing step stops the run.
set -euo pipefail
[ $# -eq 1 ] || { echo "usage: $0 OUTDIR (must not exist)"; exit 2; }
OUT=$1
[ -e "$OUT" ] && { echo "refusing: $OUT exists"; exit 2; }
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
EXPORT_PY=${EXPORT_PY:-python}
RUN_PY=${RUN_PY:-python}
LITERT_LM=${LITERT_LM:-litert-lm}
export LITERT_LM
MODEL=LiquidAI/LFM2-700M
REVISION=86f49fc9a3800c3a325b7320bde179c318062583
mkdir -p "$OUT"/q
OUT=$(cd "$OUT" && pwd)
HF_CACHE=${LFM2_HF_CACHE:-$OUT/hf_cache}
cd "$OUT"
echo "### $(date '+%F %T %Z') OUT=$PWD HF_CACHE=$HF_CACHE"
"$LITERT_LM" --version

# 0. the checkpoint at the pinned commit (export_hf takes no revision), every file checked against the Hub's record
"$EXPORT_PY" "$HERE/fetch_pinned.py" "$MODEL" "$REVISION" "$HF_CACHE" fetch_manifest.json
# 1. exports, offline from that cache: the exporter's default recipe (= the int8 file) and unquantized (= the int4 base)
HF_HUB_CACHE="$HF_CACHE" "$EXPORT_PY" "$HERE/export_lfm2_700m.py" out_wi8
HF_HUB_CACHE="$HF_CACHE" "$EXPORT_PY" "$HERE/export_lfm2_700m.py" out_fp --fp
# 2. post-hoc int4, every section kept (the recipe is minicpm_work/quantize_litertlm.py build_recipe)
"$EXPORT_PY" "$HERE/quantize_keep_sections.py" out_fp/model.litertlm q/int4.litertlm --recipe wi4b32_wi8 --algo octav --report q/quant_int4.report.json
# 3. int4: patch zero blockwise scales in place, if any (prints the count; 0 for this checkpoint)
"$EXPORT_PY" "$ROOT/minicpm_work/fix_zero_scales_inplace.py" q/int4.litertlm q/int4_zfix.litertlm
# 4. the lfm2 configuration (whole LlmMetadata replaced, every other section checked)
"$RUN_PY" "$HERE/repack_official_lfm2.py" out_wi8/model.litertlm LFM2-700M_int8.litertlm --cli "$LITERT_LM"
"$RUN_PY" "$HERE/repack_official_lfm2.py" q/int4_zfix.litertlm LFM2-700M_int4.litertlm --cli "$LITERT_LM"
ls -l LFM2-700M_int8.litertlm LFM2-700M_int4.litertlm
echo "### done $(date '+%F %T %Z')"
