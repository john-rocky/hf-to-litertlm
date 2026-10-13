#!/usr/bin/env bash
# convert_lfm25_350m.sh OUTDIR - LiquidAI/LFM2.5-350M -> the two files of litert-community/LFM2.5-350M, one command:
#   OUTDIR/LFM2.5-350M_int8.litertlm  dynamic int8 on the linears (FULLY_CONNECTED) + channelwise int8 embedding,
#                                     ShortConv convolutions float
#   OUTDIR/LFM2.5-350M_int4.litertlm  int4 blockwise-32 (OCTAV) on the linears, channelwise int8 lm_head + embedding
# both with the LiteRT-LM models/lfm2 configuration (commit 86a7e6d0; max_num_tokens 4096, supports_thinking false),
# an 11-signature prefill ladder (1024 ... 1) and a 4096-token KV cache.
#
# Environments (released wheels):
#   EXPORT_PY  a python with litert-torch==0.9.4 litert-converter==0.4.0 ai-edge-quantizer==0.9.0 transformers==5.14.1
#              (torch 2.13.0; transformers 5.15 breaks the lfm2 export)
#   RUN_PY     a python with litert-lm==0.18.0 (its litert-lm-builder 0.18.0 verifies the repack)
#   LITERT_LM  the litert-lm 0.18.0 CLI (unpack / pack), e.g. "$(dirname "$RUN_PY")/litert-lm"
#
# Steps: 1 export, unquantized -> 2 post-hoc quantization x2, every section kept -> 3 int4 zero-scale patch (a measured
# no-op for this checkpoint: 92 blockwise tensors, no zero scale) -> 4 the lfm2 configuration x2, every other section
# proven identical. The file sha256 never reproduces (the packer writes a new uuid and timestamp); compare the SECTION
# sha256 lines that step 4 prints with the table in REPRODUCE.md.
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
mkdir -p "$OUT"/q
cd "$OUT"
echo "### $(date '+%F %T %Z') OUT=$PWD"
"$LITERT_LM" --version

# 1. export, unquantized (the base of both recipes)
"$EXPORT_PY" "$HERE/export_lfm25_350m.py" out_fp --fp
# 2. post-hoc quantization, every section kept (the recipes are minicpm_work/quantize_litertlm.py build_recipe)
"$EXPORT_PY" "$HERE/quantize_keep_sections.py" out_fp/model.litertlm q/wi8fc.litertlm --recipe wi8fc --report q/quant_wi8fc.report.json
"$EXPORT_PY" "$HERE/quantize_keep_sections.py" out_fp/model.litertlm q/int4.litertlm --recipe wi4b32_wi8 --algo octav --report q/quant_int4.report.json
# 3. int4: patch zero blockwise scales in place, if any (prints the count; 0 for this checkpoint)
"$EXPORT_PY" "$ROOT/minicpm_work/fix_zero_scales_inplace.py" q/int4.litertlm q/int4_zfix.litertlm
# 4. the lfm2 configuration (whole LlmMetadata replaced, every other section checked)
"$RUN_PY" "$HERE/repack_official_lfm2.py" q/wi8fc.litertlm LFM2.5-350M_int8.litertlm --cli "$LITERT_LM"
"$RUN_PY" "$HERE/repack_official_lfm2.py" q/int4_zfix.litertlm LFM2.5-350M_int4.litertlm --cli "$LITERT_LM"
ls -l LFM2.5-350M_int8.litertlm LFM2.5-350M_int4.litertlm
echo "### done $(date '+%F %T %Z')"
