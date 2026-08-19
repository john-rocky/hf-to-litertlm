#!/bin/bash
# CohereLabs/North-Micro-Vision-Instruct (2.48B VLM, apache-2.0) -> fast_vlm .litertlm
# (first Cohere-family model on LiteRT; Qwen3-VL-structure vision tower + Cohere2 decoder).
#
# Vision  = CohereCompass ViT (SigLIP2-SO400M dims, 27 blocks) made static-512 (32x32 patches
#           -> 2x2 merge -> 256 soft tokens): Conv3d folded to Conv2d, learned pos_embed resampled
#           to the grid, precomputed 2-D rope, full attention, raster order through the encoder
#           and a strided-slice 2x2 merge in the adapter (NO GATHER_ND). The three deepstack
#           embeddings (Qwen3-VL "DeepStack", injected after decoder layers 0/1/2 in HF) are
#           FOLDED into the single fast_vlm image embedding (merger + sum of deepstack mergers) --
#           Phase-0 ablation: fold keeps 96% teacher-forced top-1 vs the released model, no
#           collapse on any of the 9 suite cases. Encoder fp16 (int8 costs vision fidelity, see
#           FINDINGS), adapter int8.
# Decoder = Cohere2ForCausalLM re-host of the CohereCompass text model (bit-exact text-only,
#           maxdiff 0.0) with the Cohere2 rope patched to Compass's Llama-style layout
#           (northmv_work/northmv_rope_patch.py; stock Cohere2 rope emits BROADCAST_TO + 5-D
#           CONCATENATION that the GPU delegate rejects). int8 (dynamic_wi8_afp32) primary;
#           int4 BOCTAV4 variant is built for the A/B (terser answers, see FINDINGS).
#           The PREFILL_DECODE section declares prefer_activation_type=fp32_fp16: Mali-CL's
#           fp16 accumulation overflows at the 256 image-token positions and the model
#           answers as if blind (echoes the question); mixed precision fixes it in-bundle.
#           Vision ships int8 (fp16 vision compiled on Mali-CL hard-crashes the device).
# Runtime contract: fast_vlm feeds 1-D sequential positions -> M-RoPE collapses to 1-D RoPE.
#           This is the real quality trade (table row counting, digit-heavy OCR); documented on
#           the card exactly like Qwen2-VL-2B.
#
# Envs: .venv-vl0930-t515 = transformers 5.16.0.dev (cohere_compass) + litert-torch 0.9.3 (vision,
#       prep); .venv-vl093 = released 0.9.3 stack, transformers 5.14.1 (decoder export, bundle).
set -e
cd ~/code/litertlm-convert
PYV=.venv-vl0930-t515/bin/python   # cohere_compass lives only in transformers git-main
PY=.venv-vl093/bin/python          # decoder export + bundle on the released 0.9.3 stack
LLM=src_models/north-micro-vision-llm
DECO=out/northmv-decoder-wi8
VISO=out/northmv-vision-fold-d
RECIPE=${RECIPE:-dynamic_wi8_afp32}   # or BOCTAV4 for the int4 variant

echo "### 1. re-host the text decoder as Cohere2ForCausalLM (bit-exact check inside)"
$PYV northmv_work/prep_northmv_decoder.py CohereLabs/North-Micro-Vision-Instruct "$LLM" 2>&1 | tail -6

echo "### 2. export decoder ($RECIPE, externalize_embedder + single_token_embedder, CACHE 4096)"
mkdir -p "$DECO"
CACHE=4096 PREFILL=128,512,1024 RECIPE=$RECIPE \
  $PY northmv_work/export_northmv_decoder.py "$LLM" "$DECO" 2>&1 | tail -2

echo "### 3. vision encoder+adapter (static-512, deepstack fold, fp16-safe LN, NO GATHER_ND)"
IMG=512 DEEPSTACK=fold $PYV northmv_work/convert_northmv_vision.py "$VISO" 2>&1 | grep -E "end2end|RESULT" | cut -c1-200
$PYV - <<PYEOF 2>&1 | grep "MB$"
import sys, os; sys.path.insert(0, "northmv_work"); import vision_quant_ab as ab
for part in ("encoder", "adapter"):
  for kind in ("fp16", "int8dyn"):
    dst = f"$VISO/vision_{part}_{kind}.tflite"
    ab.quantize(f"$VISO/vision_{part}.tflite", dst, kind); print(dst, round(os.path.getsize(dst)/1e6), "MB")
PYEOF

echo "### 4. bundle (HF tokenizer.json; decoder section declares fp32_fp16 — Mali-CL fp16 goes blind on image turns)"
DEC="$DECO" VIS="$VISO" VENC=vision_encoder_int8dyn.tflite VADP=vision_adapter_int8dyn.tflite \
  TOK=hf HF_TOK="$LLM/tokenizer.json" IMAGE_SIZE=512 CACHE=4096 OUT_DIR=out/northmv-bundle \
  OUT_NAME=North-Micro-Vision-Instruct_wi8.litertlm \
  $PY northmv_work/build_northmv_bundle.py 2>&1 | tail -1
ls -la out/northmv-bundle/North-Micro-Vision-Instruct_wi8.litertlm | awk '{print "BUNDLE:", $NF, $5/1e9" GB"}'
echo "### DONE  (gates: northmv_work/suite_gate_bundle.py, scripts/verify_quality.py)"
