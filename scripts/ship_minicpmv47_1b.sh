#!/bin/bash
# openbmb/MiniCPM-V-4.7-1B (1.55B VLM, apache-2.0) -> the three files of litert-community/MiniCPM-V-4.7-1B:
#   MiniCPM-V-4.7-1B-VL_int8.litertlm     fast_vlm: the Qwen3.5-text hybrid decoder (int8 fully connected layers with the
#                                         36 GatedDeltaNet decay/gate projections float, fp32 activation preference) +
#                                         int8 embedder + vision encoder fp16 + vision adapter fp16 (448x448, 64 tokens)
#   MiniCPM-V-4.7-1B-VL_int8vis.litertlm  the same with the vision encoder int8 (for 8 GB phones)
#   MiniCPM-V-4.7-1B_int8.litertlm        the text decoder alone (12 prefill signatures, embedder inside the graph)
#
# Vision  = the 27-layer ViT re-authored in plain torch (minicpmv47_work/mv47_vision.py): the in-encoder window merger
#           as a block-diagonal constant attention mask in raster order, 2x2 merges as four strided slices + concat,
#           fp16-safe LayerNorm (power-of-two input scales, ln_scales.json), static 448x448 -> 64 image tokens.
# Decoder = a text-only checkpoint derived from the VLM (make_text_ckpt.py: model.language_model.* -> model.*, the
#           untied lm_head kept) exported with the Qwen3.5 hybrid rail (qwen35_work + qwen35vl_work) and the
#           litert-torch patch pinned there.
# Template = minicpmv47_work/chat_template_mv47_dual.jinja (VL) / chat_template_mv47_text_dual.jinja (text), swapped in
#           last: an assistant turn ends at <|im_end|> and its newline opens the next turn, so the runtime's
#           incremental prefill of a second turn equals the upstream template's tokens.
#
# Environments (absolute paths, given by the caller):
#   PY_EXPORT   Python 3.12: litert-torch 0.9.2, litert-converter 0.3.0, ai-edge-quantizer 0.8.0, ai-edge-litert 2.1.6,
#               litert-lm-builder 0.15.0, transformers 5.14.1, torch 2.12.1 (litert-torch is overridden by the patched
#               checkout below through PYTHONPATH)
#   PY_VISION   Python 3.14: litert-torch 0.9.3, litert-converter 0.4.0, ai-edge-quantizer 0.9.0, ai-edge-litert 2.2.0,
#               torch 2.13.0, pillow
#   PY_HF       Python 3.12: transformers 5.19.0, torch 2.14.1, torchvision, pillow (the HF reference for the vision parity)
#   LITERT_LM   the litert-lm 0.18.0 CLI (unpack / pack inside the drivers)
#   PY_SWAP     Python with litert-lm 0.17.1 (litert-lm-builder 0.17.1); SWAP_CLI = that environment's litert-lm
#   FIXTURE_SRC a directory with cats_512.png, kitchen1_512.png, kitchen2_512.png (the vision parity images; not in
#               this repository, any three photos run the checks; ln_scales.json carries the published scales)
#
#   bash scripts/ship_minicpmv47_1b.sh [OUT]        # OUT defaults to out/minicpmv47
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)
: "${PY_EXPORT:?}" "${PY_VISION:?}" "${PY_HF:?}" "${LITERT_LM:?}" "${PY_SWAP:?}" "${SWAP_CLI:?}" "${FIXTURE_SRC:?}"
OUT=$(mkdir -p "${1:-out/minicpmv47}" && cd "${1:-out/minicpmv47}" && pwd)
W=$OUT/work
TXT=$OUT/MiniCPM-V-4.7-1B-text
mkdir -p "$W"

echo "### 0. source checkpoint (revision 984a89aa, without the README assets)"
SRC=$(HF_HUB_DISABLE_XET=1 hf download openbmb/MiniCPM-V-4.7-1B --revision 984a89aafcd5160164f4f0d04b4be0f4eec03cb8 \
      --exclude "minicpm_v4dot7_assets/*" | tail -1)
echo "SRC=$SRC"

echo "### 1. text-only checkpoint derived from the VLM"
"$PY_EXPORT" minicpmv47_work/make_text_ckpt.py "$SRC" "$TXT"

echo "### 2. litert-torch at the pinned base + the Qwen3.5 hybrid patch"
if [ ! -d qwen35_work/litert-torch-qwen35 ]; then
  git clone https://github.com/john-rocky/litert-torch qwen35_work/litert-torch-qwen35
  git -C qwen35_work/litert-torch-qwen35 fetch --depth 1 origin 115a13607c730c81018bb9789138a3e5e5119e3d
  git -C qwen35_work/litert-torch-qwen35 checkout --detach 115a13607c730c81018bb9789138a3e5e5119e3d
  git -C qwen35_work/litert-torch-qwen35 apply "$ROOT/qwen35_work/qwen35_hybrid_litert_torch.patch"
fi

echo "### 3. text-only file: export + int8 (36 decay/gate FC float) -> fp32 activations -> template -> no start token"
mkdir -p "$OUT/text"
( cd qwen35_work && PATH="$(dirname "$LITERT_LM"):$PATH" QWEN35_FLOAT_FC_REGEX='Linear_in_proj_(a|b);' \
    PYTHONPATH=litert-torch-qwen35 "$PY_EXPORT" convert_qwen35_hybrid.py "$TXT" "$OUT/text" )
T=$OUT/text/MiniCPM-V-4.7-1B-text
"$PY_EXPORT" scripts/set_activation_type.py "${T}_int8.litertlm" "${T}_int8_fp32act.litertlm" --type fp32 --litert-lm "$LITERT_LM"
"$PY_SWAP" tools/chat_template_parts/swap_template.py "${T}_int8_fp32act.litertlm" "${T}_int8_fp32act_dual.litertlm" \
  --jinja tools/chat_template_parts/dual/b3f69d650a63.jinja --cli "$SWAP_CLI" --report "$OUT/text/swap_dual.json"
"$PY_EXPORT" minicpmv47_work/strip_start_token.py "${T}_int8_fp32act_dual.litertlm" "${T}_int8_fp32act_dual_nostart.litertlm" \
  --litert-lm "$LITERT_LM"

echo "### 4. vision tower: fixtures -> HF reference -> eager parity -> fp32 / fp16 / int8 TFLite + A/B"
FIXTURE_SRC="$FIXTURE_SRC" "$PY_VISION" minicpmv47_work/vision_fixtures.py "$W/fixtures"
"$PY_HF" minicpmv47_work/vision_ref.py --src "$SRC" --fixtures "$W/fixtures" --out "$W/ref" --stages "$W/stages"
"$PY_VISION" minicpmv47_work/vision_parity_eager.py --src "$SRC" --ref "$W/ref" --fixtures "$W/fixtures" \
  --stages "$W/stages" --out "$W"
"$PY_VISION" minicpmv47_work/vision_export.py "$OUT/vision" --src "$SRC" --ref "$W/ref" --fixtures "$W/fixtures" \
  --ln-scales minicpmv47_work/ln_scales.json --r2 "$W" --stage all

echo "### 5. VL decoder: externalized embedder, 6 prefill signatures + decode, KV 4096, int8 with the decay/gate FC float"
PREFILL=1024,256,64,16,4,1 CACHE=4096 QWEN35_FLOAT_FC_REGEX='Linear_in_proj_(a|b);' \
  PYTHONPATH=qwen35_work/litert-torch-qwen35 "$PY_EXPORT" qwen35vl_work/export_qwen35vl_decoder.py "$TXT" "$OUT/decoder"

echo "### 6. fast_vlm bundles (encoder fp16 / int8, adapter fp16) + executor metadata (48 state buffers)"
for v in "fp16 MiniCPM-V-4.7-1B-VL_int8" "int8 MiniCPM-V-4.7-1B-VL_int8vis"; do
  set -- $v
  "$PY_EXPORT" minicpmv47_work/build_mv47_bundle.py --build --dec "$OUT/decoder" --vis "$OUT/vision" \
    --venc "vision_encoder_$1.tflite" --vadp vision_adapter_fp16.tflite --tok "$SRC/tokenizer.json" \
    --out-dir "$OUT/bundle" --out-name "$2.litertlm"
  "$PY_EXPORT" scripts/add_executor_metadata.py "$OUT/bundle/$2.litertlm" "$OUT/bundle/$2_final.litertlm" \
    --litert-lm "$LITERT_LM" --python "$PY_EXPORT"
done

echo "### 7. the shipped chat templates (metadata only; every other section keeps its bytes)"
for v in MiniCPM-V-4.7-1B-VL_int8 MiniCPM-V-4.7-1B-VL_int8vis; do
  "$PY_SWAP" tools/chat_template_parts/swap_template.py "$OUT/bundle/${v}_final.litertlm" "$OUT/$v.litertlm" \
    --jinja minicpmv47_work/chat_template_mv47_dual.jinja --cli "$SWAP_CLI" --report "$OUT/swap_$v.json"
done
"$PY_SWAP" tools/chat_template_parts/swap_template.py "${T}_int8_fp32act_dual_nostart.litertlm" \
  "$OUT/MiniCPM-V-4.7-1B_int8.litertlm" --jinja minicpmv47_work/chat_template_mv47_text_dual.jinja \
  --cli "$SWAP_CLI" --report "$OUT/swap_MiniCPM-V-4.7-1B_int8.json"
ls -l "$OUT"/*.litertlm | awk '{print "BUNDLE:", $NF, $5, "B"}'
echo "### DONE (compare sections, not file sha256: every pack writes a new uuid and timestamp into the header)"
