#!/usr/bin/env bash
# PrismML Ternary-Bonsai-2-27B (Qwen3.8-27B hybrid, ternary g128 + block-1024 Hadamard rotation) -> LiteRT-LM int4 bundle.
# Every step is a script in this directory; see FINDINGS.md for the recipe, the traps and the gate numbers.
#
#   1. dl_mlx_pack.sh           the 8.6 GB MLX pack (2-bit affine g128 = the exact ternary values) + small files
#   2. dequant_mlx_pack.py      -> hf_bf16/: HF-layout bf16 Qwen3_5 text checkpoint (rotated values kept verbatim,
#                                  conv1d layout + RMSNorm (1+w) offset handled) + hadamard_signs.safetensors
#   3. hf_check.py / mlx_ref.py  token-identical greedy check against PrismML's bundled MLX runtime (oracle)
#   4. export_driver.py         float export with the Qwen3.5 hybrid patch + Hadamard ops in-graph, lightweight
#                                  conversion, stops at model.tflite (the fp32 27B tflite is ~108 GB; no fp32 bundle)
#   5. build_bundle.py          aeq int4 blockwise-32 (rotation FCs float, embedding + lm_head int8), metadata,
#                                  litert-lm-builder pack, zero-scale fix, ExecutorMetadata, prefer_activation_type fp32
set -euo pipefail
cd "$(dirname "$0")/.."
W=bonsai2_work
LT=$HOME/venvs/ltconv040dev            # transformers 5.14.1, torch 2.12.1, ai-edge-quantizer 0.8.0, litert-lm(-builder) 0.15.0
export PATH="$LT/bin:$PATH"
[ -f $W/src/DONE ] || $W/dl_mlx_pack.sh
HF_HUB_DISABLE_XET=1 hf download prism-ml/Ternary-Bonsai-2-27B-mlx-2bit --local-dir $W/src --exclude "model.safetensors" >/dev/null
[ -f $W/hf_bf16/model.safetensors.index.json ] || $LT/bin/python3 $W/dequant_mlx_pack.py $W/src $W/hf_bf16
cp $W/hf_ref/config.json $W/hf_ref/generation_config.json $W/hf_bf16/          # Qwen/Qwen3.8-27B config (text_config = the arch)
cp $W/src/tokenizer.json $W/src/tokenizer_config.json $W/src/chat_template.jinja $W/src/hadamard.json $W/hf_bf16/
PYTHONPATH=qwen35_work/litert-torch-qwen35 QWEN35_PREFILL_LADDER=1024,256,64,16,4,1 CACHE_LENGTH=4096 \
  $LT/bin/python3 $W/export_driver.py $W/hf_bf16 $W/out_fp --keep_temporary_files=True --experimental_lightweight_conversion=True
$LT/bin/python3 $W/build_bundle.py $W/out_fp/model.tflite $W/out_fp/tokenizer.json qwen35_work/chat_template_simple.jinja \
  $W/out/Ternary-Bonsai-2-27B_mixed_int4_b32.litertlm --block b32
