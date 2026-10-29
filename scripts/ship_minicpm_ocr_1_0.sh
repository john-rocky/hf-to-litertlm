#!/bin/bash
# openbmb/MiniCPM-OCR-1.0 (1.3B OCR VLM, apache-2.0) -> the files of litert-community/MiniCPM-OCR-1.0:
#   MiniCPM-OCR-1.0-VL_int8.litertlm     fast_vlm 448: the Qwen3.5-text hybrid decoder (int8 fully connected layers with
#                                        the 36 GatedDeltaNet decay/gate projections float, fp32 activation preference)
#                                        + int8 embedder + vision encoder fp16 + vision adapter fp16 (448x448 -> 256
#                                        tokens, the model's 4x mode)
#   MiniCPM-OCR-1.0-VL_int8vis.litertlm  the same with the vision encoder int8 (dynamic range)
#
# Vision   = the 27-layer ViT of the 4x mode (no in-encoder window merger: every layer attends over all 1,024 patches)
#            and the 2x2 merger, re-authored in plain torch (minicpmocr_work/ocr_vision.py): position table as a
#            constant, fp16-safe LayerNorm with the published power-of-two input scales (minicpmocr_work/ln_scales.json).
# Decoder  = a text-only checkpoint derived from the VLM (tools/vlm_text_ckpt: model.language_model.* -> model.*, tied
#            head, qwen3_5_text config) exported with the Qwen3.5 hybrid rail (qwen35vl_work + qwen35_work) and the
#            litert-torch patch pinned there.
# Template = minicpmocr_work/chat_template_ocr.jinja: ChatML with the model's empty think block; with the extra-context
#            variable slice_cols, the 2nd and later images of one user message are the processor's <slice> tiles.
#
# Environments (absolute paths, given by the caller):
#   PY_EXPORT   Python 3.12: litert-torch 0.9.2, litert-converter 0.3.0, ai-edge-quantizer 0.8.0, ai-edge-litert 2.1.6,
#               litert-lm-builder 0.15.0, transformers 5.14.1, torch 2.12.1 (litert-torch is overridden by the patched
#               checkout below through PYTHONPATH)
#   PY_VISION   Python 3.14: litert-torch 0.9.3, litert-converter 0.4.0, ai-edge-quantizer 0.9.0, ai-edge-litert 2.2.0,
#               torch 2.13.0, safetensors, numpy, pillow
#   PY_HF       Python 3.12: transformers at commit 2df500f356fb0b1eda921500ea8db1b1eed86a5d (minicpm_ocr is on main
#               only, not in v5.21.0), torch 2.14.1, torchvision, pillow, safetensors, accelerate
#   LITERT_LM   a litert-lm CLI with pack / unpack (0.18.1 was used)
#   FIXTURE_SRC a directory of 448x448 PNG pictures for the vision parity check (any text pictures; the LayerNorm scales
#               are the published ones, so these pictures do not change the files)
#
#   bash scripts/ship_minicpm_ocr_1_0.sh [OUT]        # OUT defaults to out/minicpmocr
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)
: "${PY_EXPORT:?}" "${PY_VISION:?}" "${PY_HF:?}" "${LITERT_LM:?}" "${FIXTURE_SRC:?}"
OUT=$(mkdir -p "${1:-out/minicpmocr}" && cd "${1:-out/minicpmocr}" && pwd)
W=$OUT/work
TXT=$W/MiniCPM-OCR-1.0-text
mkdir -p "$W"

echo "### 0. source checkpoint (revision 165531f9)"
SRC=$(HF_HUB_DISABLE_XET=1 hf download openbmb/MiniCPM-OCR-1.0 --revision 165531f9ead5c7f09fba35a9a89fd2e255acf99a | tail -1)
SRC=${SRC#path=}   # huggingface_hub 2.x prints "path=<dir>" outside a terminal, 1.x the bare directory
[ -f "$SRC/config.json" ] || { echo "no checkpoint at '$SRC'"; exit 1; }
echo "SRC=$SRC"

echo "### 1. text-only checkpoint derived from the VLM (tied head), with its logits proof against the full model"
"$PY_HF" tools/vlm_text_ckpt/make_text_ckpt.py "$SRC" "$TXT" --prefix model.language_model. \
    --model-type qwen3_5_text --architectures Qwen3_5ForCausalLM --report "$W/text_ckpt.json" --prove

echo "### 2. litert-torch at the pinned base + the Qwen3.5 hybrid patch"
if [ ! -d qwen35_work/litert-torch-qwen35 ]; then
  git clone https://github.com/john-rocky/litert-torch qwen35_work/litert-torch-qwen35
  git -C qwen35_work/litert-torch-qwen35 fetch --depth 1 origin 115a13607c730c81018bb9789138a3e5e5119e3d
  git -C qwen35_work/litert-torch-qwen35 checkout --detach 115a13607c730c81018bb9789138a3e5e5119e3d
  git -C qwen35_work/litert-torch-qwen35 apply "$ROOT/qwen35_work/qwen35_hybrid_litert_torch.patch"
fi

echo "### 3. decoder: EMBEDDER + PREFILL_DECODE (externalized embedder, prefill 1024/256/64/16/4/1, KV 4096), int8"
mkdir -p "$W/decoder" "$W/tmp"
PREFILL=1024,256,64,16,4,1 CACHE=4096 QWEN35_FLOAT_FC_REGEX='Linear_in_proj_(a|b);' PYTHONHASHSEED=0 \
  PYTHONPATH="$ROOT/qwen35_work/litert-torch-qwen35" TMPDIR="$W/tmp" \
  "$PY_EXPORT" qwen35vl_work/export_qwen35vl_decoder.py "$TXT" "$W/decoder"

echo "### 4. vision: upstream references on FIXTURE_SRC, then the TFLite pair (fp32 -> fp16 / int8 encoder, fp16 adapter)"
"$PY_HF" minicpmocr_work/vision_check.py --snap "$SRC" --images "$FIXTURE_SRC"/*.png --out "$W/vision_ref"
"$PY_VISION" minicpmocr_work/vision_export.py "$W/vision" --snap "$SRC" --refs "$W/vision_ref" \
    --ln-scales minicpmocr_work/ln_scales.json --stage all

echo "### 5. the two bundles: sections + LlmMetadata (fast_vlm 448, stops 248044 / 248046, the template), then the"
echo "###    ExecutorMetadata section that binds the 48 hybrid state buffers"
for v in VL_int8:vision_encoder_fp16.tflite VL_int8vis:vision_encoder_int8.tflite; do
  name=${v%%:*}; venc=${v#*:}
  "$PY_EXPORT" minicpmocr_work/build_ocr_bundle.py --build --dec "$W/decoder" --vis "$W/vision" --venc "$venc" \
      --vadp vision_adapter_fp16.tflite --tok "$SRC/tokenizer.json" --out-dir "$W/bundle" --out-name "pre_$name.litertlm"
  "$PY_EXPORT" scripts/add_executor_metadata.py "$W/bundle/pre_$name.litertlm" "$OUT/MiniCPM-OCR-1.0-$name.litertlm" \
      --litert-lm "$LITERT_LM" --python "$PY_EXPORT"
done
ls -l "$OUT"/*.litertlm
