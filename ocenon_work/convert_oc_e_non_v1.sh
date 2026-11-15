#!/usr/bin/env bash
# convert_oc_e_non_v1.sh OUTDIR - p-e-w/Oc-e-Non-v1 (pinned dbf4e269a44be55a1807cf2b362ec7199ef73d33, Apache-2.0; a
# Qwen3-4B-Instruct-2507 derivative: Qwen3ForCausalLM, 36 layers, tied embeddings, no thinking mode) -> .litertlm files:
#   OUTDIR/i8/oc_e_non_v1_int8_ekv4096.litertlm           int8 per-channel min-max weights for every FULLY_CONNECTED
#                                                         (lm_head included) and the token table; fp32 GPU activations
#   OUTDIR/b32/oc_e_non_v1_mixed_int4b32_ekv4096.litertlm int4 blockwise-32 min-max FULLY_CONNECTED weights (block
#                                                         scales floored at 2^-14), the decode signature's lm_head and
#                                                         the token table int8 (the prefill signature's lm_head is
#                                                         int4); fp32 GPU activations declared
#   OUTDIR/b32h/oc_e_non_v1_mixed_int4b32_fp16act_ekv4096.litertlm  the same int4 file with fp16 GPU activations declared
#                                                         (Q32_ACT=fp16: every section the b32 file's, only the TFLite
#                                                         section's prefer_activation_type item differs)
# All from one export by qwen3_32b_work/convert_qwen3_32b.sh: the released litert-torch 0.10.0's export_hf with GPU
# composites (prefill 1,024, KV cache 4,096), the checkpoint's tokenizer.json, and this folder's LlmMetadata
# (oc_e_non_v1_LlmMetadata.pbtext: LiteRT-LM v0.19.1's models/qwen3 configuration with the checkpoint's sampler and
# stop tokens and no thought channel).
# The 32B recipe's F32 copy (upcast) and prefused file only keep a 32B export inside RAM. A 4B export reads the bf16
# snapshot directly (F32_DIR = the snapshot) and builds the fused projections in RAM (Q32_CATVIEW=0): the same float
# values. On a 128 GB Mac the export peaked at a 46.1 GB memory footprint, an int4 pass at 10.1 GB. Disk: 34 GB under
# OUTDIR at the end (the 8.1 GB snapshot, the 16.1 GB float TFLite, the 4.1 GB int8 file and two 3.1 GB int4 files).
# PY = the python of an environment made from mnn_pocket_work/requirements.txt (as for qwen3-32b in REPRODUCE.md).
set -euo pipefail
R=$(cd "$(dirname "$0")/.." && pwd)
[ $# -eq 1 ] || { echo "usage: $0 OUTDIR"; exit 2; }
mkdir -p "$1"; OUT=$(cd "$1" && pwd)
: "${PY:?set PY to the python of an environment made from mnn_pocket_work/requirements.txt}"
export PY REPO=p-e-w/Oc-e-Non-v1 REV=dbf4e269a44be55a1807cf2b362ec7199ef73d33
export Q32_PBTEXT=$R/ocenon_work/oc_e_non_v1_LlmMetadata.pbtext Q32_CATVIEW=0
export HF_HUB_CACHE=${HF_HUB_CACHE:-$OUT/hf_cache}
SNAP=$HF_HUB_CACHE/models--p-e-w--Oc-e-Non-v1/snapshots/$REV
F32_DIR=$SNAP QUANT=i8 bash "$R/qwen3_32b_work/convert_qwen3_32b.sh" "$OUT/i8" fetch shim export quantize checks sections
EXPORT_DIR=$OUT/i8/export QUANT=b32 bash "$R/qwen3_32b_work/convert_qwen3_32b.sh" "$OUT/b32" shim quantize checks sections
EXPORT_DIR=$OUT/i8/export QUANT=b32 Q32_ACT=fp16 bash "$R/qwen3_32b_work/convert_qwen3_32b.sh" "$OUT/b32h" shim quantize checks sections
ls -l "$OUT/i8"/*.litertlm "$OUT/b32"/*.litertlm "$OUT/b32h"/*.litertlm
