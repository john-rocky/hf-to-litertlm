#!/bin/bash
# microsoft/Mage-VL-Cowork (GUI agent VLM, apache-2.0: a 24-layer ViT + a Qwen3 text decoder with 36 layers) at revision
# 6fbe7f91da477c1c73790dbb1f08d001b8626be1 -> two fast_vlm files that read one 1920x1088 screenshot (68 x 120 patches of
# 16 = 8,160 patches, 2,040 image tokens after the 2x2 merge):
#   Mage-VL-Cowork.litertlm (decoder and vision encoder on the GPU)
#     LlmMetadata     fast_vlm 1088 x 1920, max_num_tokens 4,096, the vendor chat template with six listed edits
#                     (magevlcowork_work/make_cowork_template.py), stop token id 151645 (<|im_end|>), model prefix
#                     "<|im_start|>assistant\n<think>\n", no channel, no start token
#     HF tokenizer    the repository's own tokenizer.json (11,422,654 B)
#     EMBEDDER        int8 embedding table; fp32 declared
#     PREFILL_DECODE  int4 blockwise-128 (OCTAV) weights, signatures prefill_1024 + decode, KV cache 4,096, the GPU KV
#                     composites (72 odml.cache_update, 142 odml.runtime_bmm, param_tensor), rotary positions exact under
#                     fp16 activations (LiteRT-LM #3913); no activation declaration, so the GPU text executor runs its
#                     default fp16 activations
#     VISION_ENCODER  fp32, [1,1088,1920,3] in [0,1] -> [1,8160,1024]; attention in 32 query chunks of 255 rows (each
#                     chunk sees all 8,160 keys); every tensor at rank <= 4; fp32 declared
#     VISION_ADAPTER  fp32, 2x2 strided-slice merge + merger MLP, [1,8160,1024] -> [1,2040,2560]; fp32 declared
#   Mage-VL-Cowork_vision-fp16.litertlm (decoder on the GPU, vision encoder on the CPU)
#     the same file with only its VISION_ENCODER replaced: 22 LayerNorm pre-scales and the ln_pre scale (the fp32 output
#     does not move), FULLY_CONNECTED weights and biases FLOAT16 behind a DEQUANTIZE, model metadata
#     reduced_precision_support = fp16accfp16 (XNNPACK's fp16 path)
#
# Steps: the text decoder is extracted as a standalone Qwen3ForCausalLM and checked against the full model (fp32,
# argmax at every position + a greedy continuation); it is exported with scripts/export_internvl_decoder.py (KV 4,096,
# prefill 128 / 512 / 1,024) and the vision pair is converted; the bundle is assembled; then, without a re-export:
# only prefill_1024 + decode are kept, the GPU KV composites are retrofitted, the decoder's fp32 declaration is removed
# and the rotary split is applied. The second file: the encoder is exported again with the LayerNorm pre-scales
# (its fp32 output must equal the first encoder's bit for bit), edited twice on the flatbuffer and swapped in.
#
# Environments (absolute paths, given by the caller):
#   PY_PREP     Python 3.12: torch 2.12.1, transformers 5.14.1, tokenizers 0.22.2, safetensors 0.8.0
#   PY_EXPORT   Python 3.14: litert-torch 0.9.3, litert-converter 0.4.0, ai-edge-quantizer 0.9.0, ai-edge-litert 2.2.0,
#               litert-lm-builder 0.16.1, transformers 5.14.1, torch 2.13.0, torchao 0.18.0, flatbuffers, numpy,
#               pillow
#   LITERT_LM   the litert-lm CLI 0.19.0 (pack / unpack)
#   PY_SAMPLE   (optional, default PY_EXPORT) a Python with pillow + matplotlib (its DejaVu fonts) that draws the sample
#               screenshot the vision export traces with; the vision file does not depend on the picture's pixels
#               (matplotlib 3.11.2 + Pillow 13.0.0 draw the card's samples/settings.png byte for byte)
#   SRC         (optional) a local copy of the source revision, checked against the Hub, instead of the download
# The source download is 9.5 GB. Maximum resident set size of the three heavy steps when the shipped file was built
# (Apple Silicon Mac): decoder extraction 30.7 GB, decoder export 57.2 GB, vision export 10.2 GB.
#
# Every pack writes a new uuid and creation time into the file header, so the whole-file sha256 never repeats; compare
# the sections. The shipped files (3,803,948,496 B and 3,199,591,888 B), hashed as stored in the container
# (scripts/section_identity.py, sections()); the last lines of this script print the same lists for the new files:
#   0 LlmMetadataProto            4,767  f6ec6b42b5df4631af0e5410393548c02328af4773a360a119f266e5a3de7313
#   1 HF_Tokenizer_Zlib       2,142,149  f8c60e814140eff4ea766b3de0354f1236ac945273984a40b6a51bff3f2b8050
#   2 TFLiteModel           390,781,600  4249034dc10c92661ca65fef347d36587713abf6f5477ccc737245fe121563a2  embedder
#   3 TFLiteModel         2,084,495,952  9c4204efc96341328a547cf1fafc0fdf9a319a785e9f0d51920694a861767c63  prefill_decode
#   4 TFLiteModel         1,217,373,628  25a044784332fa11ce5cf03532c2a1df58feae7b070b0f40444f5e6f8a2a421d  vision_encoder
#   5 TFLiteModel           109,094,352  0164e48c0a691eb436bc1b62419a6c81befe967dc049e1951c9c245140e5c3bd  vision_adapter
#   _vision-fp16: 0-3 and 5 the same bytes, section 4:
#   4 TFLiteModel           613,020,076  08cc1759f599d222bd2424f42bb6097fd2815c7276b91e5e485cbd2ac755c295  vision_encoder
#
#   bash scripts/ship_mage_vl_cowork.sh [OUT]        # OUT defaults to out/magevlcowork
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)
: "${PY_PREP:?}" "${PY_EXPORT:?}" "${LITERT_LM:?}"
OUT=$(mkdir -p "${1:-out/magevlcowork}" && cd "${1:-out/magevlcowork}" && pwd)
W=$OUT/work
mkdir -p "$W/tmp" "$W/bundle"
export TMPDIR="$W/tmp"

check() {   # check REPORT.json KEY...: stop unless every KEY of the report is true
  "$PY_EXPORT" -I -c 'import json, sys
r = json.load(open(sys.argv[1]))
bad = [k for k in sys.argv[2:] if r.get(k) is not True]
sys.exit(f"{sys.argv[1]}: not true: {bad}" if bad else 0)' "$@"
}

echo "### 0. source checkpoint (revision 6fbe7f91)"
if [ -z "${SRC:-}" ]; then
  SRC=$(HF_HUB_DISABLE_XET=1 hf download microsoft/Mage-VL-Cowork --revision 6fbe7f91da477c1c73790dbb1f08d001b8626be1 | tail -1)
  SRC=${SRC#path=}   # huggingface_hub 2.x prints "path=<dir>" outside a terminal, 1.x the bare directory
fi
[ -f "$SRC/config.json" ] || { echo "no checkpoint at '$SRC'"; exit 1; }
echo "SRC=$SRC"

echo "### 1. the text decoder as a standalone Qwen3ForCausalLM (strict 1:1, untied lm_head), checked against the full"
echo "###    model in fp32 (the vendor's GUI agent prompt: argmax at every position, a 48-token greedy continuation)"
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false HF_MODULES_CACHE="$W/hf_modules" HF_HOME="$W/hf_home" \
  /usr/bin/time -l "$PY_PREP" -I magevlcowork_work/prep_cowork_decoder.py "$SRC" "$W/llm" "$W/prep_decoder.json"
check "$W/prep_decoder.json" ok

echo "### 2. the bundle's chat template: the vendor's chat_template.jinja + six listed edits (4,623 B, sha256 5f8cdbd6...)"
"$PY_EXPORT" magevlcowork_work/make_cowork_template.py "$SRC/chat_template.jinja" "$W/cowork_runtime.jinja" \
  --doc "$W/REWRITES.md"

echo "### 3. decoder: EMBEDDER + PREFILL_DECODE (externalized embedder, prefill 128 / 512 / 1,024, KV 4,096), int4"
echo "###    blockwise-128 OCTAV + int8 embedding, block scales floored at 2^-14"
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false HF_HOME="$W/hf_home" \
  CACHE=4096 PREFILL=128,512,1024 RECIPE=BOCTAV4_128 PYTHONHASHSEED=0 \
  /usr/bin/time -l "$PY_EXPORT" magevlcowork_work/export_cowork_decoder.py "$W/llm" "$W/decoder"

echo "### 4. vision 1920x1088: the encoder (raster patches, 255-row attention query chunks) and the adapter, fp32; parity"
echo "###    against the vendor's vision forward on the sample screenshot (also the tracing input) and a noise picture"
"${PY_SAMPLE:-$PY_EXPORT}" magevlcowork_work/make_sample.py "$W/settings.png"
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false HF_MODULES_CACHE="$W/hf_modules" HF_HOME="$W/hf_home" \
  PYTHONHASHSEED=0 IMG_H=1088 IMG_W=1920 MODEL="$SRC" TFL_THREADS=4 QCHUNK=255 \
  /usr/bin/time -l "$PY_EXPORT" magevlcowork_work/convert_cowork_vision.py "$W/vision" --example "$W/settings.png"
check "$W/vision/result.json" ok rank_gt4_ok max_intermediate_ok

echo "### 5. the bundle: LlmMetadata + the repository's tokenizer.json + the four TFLite sections, then fp32 declared on"
echo "###    every TFLite section"
DEC="$W/decoder" VIS="$W/vision" VENC="$W/vision/vision_encoder.tflite" VADP="$W/vision/vision_adapter.tflite" \
  TOK=hf TOK_JSON="$SRC/tokenizer.json" IMAGE_H=1088 IMAGE_W=1920 CACHE=4096 STOP_ID=151645 \
  JINJA_FILE="$W/cowork_runtime.jinja" CHANNEL= OUT_DIR="$W/bundle" OUT_NAME=s5_undeclared.litertlm \
  "$PY_EXPORT" -I magevlcowork_work/build_cowork_bundle.py
"$PY_EXPORT" -I scripts/set_activation_type.py "$W/bundle/s5_undeclared.litertlm" "$W/bundle/s5.litertlm" --type fp32 \
  --litert-lm "$LITERT_LM"
rm -f "$W/bundle/s5_undeclared.litertlm"

echo "### 6. keep the decoder signatures prefill_1024 + decode and drop the subgraphs no signature reaches"
LITERT_LM_CLI="$LITERT_LM" "$PY_EXPORT" -I tools/gpu_graph/shape_variant.py build "$W/bundle/s5.litertlm" \
  "$W/bundle/s6.litertlm" --keep prefill_1024,decode --drop-subgraphs --report "$W/shape_variant_report.json" \
  --work "$W/s6_work"
rm -rf "$W/s6_work"; rm -f "$W/bundle/s5.litertlm"

echo "### 7. the GPU KV composites: odml.cache_update + odml.runtime_bmm with param_tensor (the exporter's decomposition,"
echo "###    the mask added by broadcast)"
LITERT_LM_CLI="$LITERT_LM" "$PY_EXPORT" -I tools/gpu_graph/gpu_graph_retrofit.py "$W/bundle/s6.litertlm" \
  "$W/bundle/s7.litertlm" --decomp exporter --mask add_bcast --report "$W/retrofit_report.json" --work "$W/s7_work"
rm -rf "$W/s7_work"; rm -f "$W/bundle/s6.litertlm"

echo "### 8. remove PREFILL_DECODE's fp32 declaration (the GPU text executor then runs its default fp16 activations)"
"$PY_EXPORT" -I magevlcowork_work/set_section_act.py "$W/bundle/s7.litertlm" "$W/bundle/s8.litertlm" \
  --section prefill_decode --type none --litert-lm "$LITERT_LM" --work "$W/s8_work"
rm -f "$W/bundle/s7.litertlm"

echo "### 9. rotary positions exact under fp16 (LiteRT-LM #3913): position = 2048*h + l, the phase of 2048*h from tables"
"$PY_EXPORT" -I magevlcowork_work/patch_rope_qwen3.py "$W/bundle/s8.litertlm" --split "$OUT/Mage-VL-Cowork.litertlm"
rm -f "$W/bundle/s8.litertlm"
ls -l "$OUT/Mage-VL-Cowork.litertlm"

echo "### 10. the second file's encoder: exported again with the 22 LayerNorm pre-scales (layers 13-23); its fp32 output must"
echo "###     equal the first encoder's bit for bit on the sample screenshot and the noise picture"
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false HF_MODULES_CACHE="$W/hf_modules" HF_HOME="$W/hf_home" \
  PYTHONHASHSEED=0 IMG_H=1088 IMG_W=1920 MODEL="$SRC" TFL_THREADS=4 QCHUNK=255 \
  /usr/bin/time -l "$PY_EXPORT" magevlcowork_work/convert_cowork_vision.py "$W/vision_scaled" --example "$W/settings.png" \
  --ln-prescale magevlcowork_work/vision_ln_prescale.json --plain-dir "$W/vision"
check "$W/vision_scaled/result.json" ok rank_gt4_ok max_intermediate_ok

echo "### 11. ln_pre's input x 8 and its eps x 64: a flat screen no longer gives NaN on XNNPACK's fp16 path"
"$PY_EXPORT" -I magevlcowork_work/fp16_lnpre_scale.py "$W/vision_scaled/vision_encoder.tflite" "$W/vision_scaled/enc_lnpre.tflite" \
  --report "$W/lnpre_report.json"

echo "### 12. FULLY_CONNECTED weights and biases FLOAT16 behind a DEQUANTIZE + the metadata reduced_precision_support"
"$PY_EXPORT" -I magevlcowork_work/fp16_fc_weights.py "$W/vision_scaled/enc_lnpre.tflite" "$W/vision_scaled/enc_fp16.tflite" \
  --report "$W/fc_weights_report.json"

echo "### 13. the second file: the first file with only its vision encoder section replaced"
LITERT_LM_CLI="$LITERT_LM" "$PY_EXPORT" -I magevlcowork_work/swap_section.py "$OUT/Mage-VL-Cowork.litertlm" \
  "$W/vision_scaled/enc_fp16.tflite" "$OUT/Mage-VL-Cowork_vision-fp16.litertlm" --model-type vision_encoder \
  --report "$W/swap_report.json"
ls -l "$OUT/Mage-VL-Cowork_vision-fp16.litertlm"

echo "### sections, hashed as stored (compare with the list at the top of this script)"
"$PY_EXPORT" -I - "$ROOT/scripts" "$OUT/Mage-VL-Cowork.litertlm" "$OUT/Mage-VL-Cowork_vision-fp16.litertlm" <<'PYEOF'
import sys
sys.path.insert(0, sys.argv[1])
from section_identity import sections
for path in sys.argv[2:]:
  print(path)
  for i, (dtype, size, digest) in enumerate(sections(path)):
    print(f"  {i} {dtype:<18} {size:>13,}  {digest}")
PYEOF
