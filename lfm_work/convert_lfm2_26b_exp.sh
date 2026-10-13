#!/usr/bin/env bash
# convert_lfm2_26b_exp.sh OUTDIR [--prefill L1,L2,...] - LiquidAI/LFM2-2.6B-Exp -> the two files of litert-community/LFM2-2.6B-Exp,
# one command:
#   OUTDIR/LFM2-2.6B-Exp_int8.litertlm  dynamic int8 on the linears (FULLY_CONNECTED) + channelwise int8 embedding,
#                                   ShortConv convolutions float
#   OUTDIR/LFM2-2.6B-Exp_int4.litertlm  int4 blockwise-32 (OCTAV) on the linears, channelwise int8 lm_head + embedding;
#                                   zero block scales patched and those blocks' payload zeroed
# both with the LiteRT-LM models/lfm2 configuration (commit 86a7e6d0; max_num_tokens 4096; supports_thinking kept true:
# LFM2-2.6B-Exp, like LFM2-2.6B, writes a reasoning trace between <think> and </think> for some prompts; measured, its
# model card does not say so), an 11-signature prefill ladder
# (1024 ... 1; --prefill overrides it, e.g. 1024,256,64,16,4,1) and a 4096-token KV cache.
#
# Environments (released wheels):
#   EXPORT_PY      a python with litert-torch==0.9.4 litert-converter==0.4.0 ai-edge-quantizer==0.9.0 transformers==5.14.1
#                  (torch 2.13.0; transformers 5.15 breaks the lfm2 export)
#   RUN_PY         a python with litert-lm==0.18.0 (its litert-lm-builder 0.18.0 verifies the repack)
#   LITERT_LM      the litert-lm 0.18.0 CLI (unpack / pack), e.g. "$(dirname "$RUN_PY")/litert-lm"
#   LFM2_HF_CACHE  optional: a Hugging Face cache that already holds the pinned commit (default OUTDIR/hf_cache, a 5.1 GB
#                  download); never the shared ~/.cache/huggingface/hub (fetch_pinned.py rewrites refs/main)
#
# Steps: 0 the checkpoint at commit 2251c865, every file checked against the Hub -> 1 export, unquantized (about 12 min
# on an M4 Max, 10.3 GB) -> 2 post-hoc quantization x2, every section kept -> 3 int4: zero block scales patched (746,432
# on this checkpoint) and the payload of those blocks zeroed, then every scale must be a normal fp16 -> 4 the lfm2
# configuration x2, every other section proven identical. The file sha256 never reproduces (the packer writes a new
# uuid and timestamp); compare the SECTION sha256 lines that step 4 prints with the table in REPRODUCE.md.
# Exit 0 = every step ran and every check passed; the first failing step stops the run.
set -euo pipefail
usage="usage: $0 OUTDIR (must not exist) [--prefill L1,L2,...]"
[ $# -ge 1 ] || { echo "$usage"; exit 2; }
OUT=$1; shift
PREFILL=()
if [ $# -gt 0 ]; then
  { [ "$1" = --prefill ] && [ $# -eq 2 ]; } || { echo "$usage"; exit 2; }
  PREFILL=(--prefill "$2")
fi
[ -e "$OUT" ] && { echo "refusing: $OUT exists"; exit 2; }
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
EXPORT_PY=${EXPORT_PY:-python}
RUN_PY=${RUN_PY:-python}
LITERT_LM=${LITERT_LM:-litert-lm}
export LITERT_LM
MODEL=LiquidAI/LFM2-2.6B-Exp
REVISION=2251c865342649ffb4a9f57d07ab2428bb552cf5
mkdir -p "$OUT"/q
OUT=$(cd "$OUT" && pwd)
HF_CACHE=${LFM2_HF_CACHE:-$OUT/hf_cache}
cd "$OUT"
echo "### $(date '+%F %T %Z') OUT=$PWD HF_CACHE=$HF_CACHE ${PREFILL[*]:-}"
"$LITERT_LM" --version

# 0. the checkpoint at the pinned commit (export_hf takes no revision), every file checked against the Hub's record
"$EXPORT_PY" "$HERE/fetch_pinned.py" "$MODEL" "$REVISION" "$HF_CACHE" fetch_manifest.json
# 1. export, unquantized (the base of both recipes), offline from that cache; bash 3.2 (macOS) needs the ${a[@]+...} form
HF_HUB_CACHE="$HF_CACHE" "$EXPORT_PY" "$HERE/export_lfm2_26b_exp.py" out_fp --fp ${PREFILL[@]+"${PREFILL[@]}"}
# 2. post-hoc quantization, every section kept (the recipes are minicpm_work/quantize_litertlm.py build_recipe)
"$EXPORT_PY" "$HERE/quantize_keep_sections.py" out_fp/model.litertlm q/wi8fc.litertlm --recipe wi8fc --report q/quant_wi8fc.report.json
"$EXPORT_PY" "$HERE/quantize_keep_sections.py" out_fp/model.litertlm q/int4.litertlm --recipe wi4b32_wi8 --algo octav --report q/quant_int4.report.json
# 3. int4: patch the zero blockwise scales in place, zero those blocks' payload, refuse any zero or subnormal scale left
"$EXPORT_PY" "$ROOT/minicpm_work/fix_zero_scales_inplace.py" q/int4.litertlm q/int4_zfix.litertlm --zero-payload --require-normal
# 4. the lfm2 configuration (whole LlmMetadata replaced, every other section checked)
"$RUN_PY" "$HERE/repack_official_lfm2.py" q/wi8fc.litertlm LFM2-2.6B-Exp_int8.litertlm --supports-thinking true --cli "$LITERT_LM"
"$RUN_PY" "$HERE/repack_official_lfm2.py" q/int4_zfix.litertlm LFM2-2.6B-Exp_int4.litertlm --supports-thinking true --cli "$LITERT_LM"
ls -l LFM2-2.6B-Exp_int8.litertlm LFM2-2.6B-Exp_int4.litertlm
echo "### done $(date '+%F %T %Z')"
