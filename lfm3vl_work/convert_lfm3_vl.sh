#!/usr/bin/env bash
# convert_lfm3_vl.sh OUTDIR [STEP...] - LiquidAI/LFM3-VL-2B (pinned f83d2b0a2296aaa277f394990a9b8b949ddf9691, LFM Open
# License v1.0) -> the two files of litert-community/LFM3-VL-2B-LiteRT, text and one 512 x 512 picture in / text out,
# LiteRT-LM 0.18.1, model type fast_vlm:
#   OUTDIR/LFM3-VL-2B_int4.litertlm  text decoder int4 blockwise-32 (the tied lm_head int8 per channel), token embedder int8
#   OUTDIR/LFM3-VL-2B_f16.litertlm   text decoder with every fully connected weight cast to fp16, token embedder int8
# Both: prefill signatures 1024 / 256 / 64 / 16 / 4 / 1 over a 2,048-token cache, the vision encoder with fp32 weights
# and the projector with fp16 weights (both declared fp32 activations; the same two graphs in both files), the source
# tokenizer.json, the chat template of tools/vl_template.py, no sampler, no activation type declared for the decoder.
# Steps (in order; a finished step leaves OUTDIR/.done/<step> and is skipped on a re-run; name steps to run only those):
#   fetch     the source files, curl into OUTDIR/hf_cache (dory_work/fetch_pinned.sh), checked against the Hub's record
#             of the commit (dory_work/verify_pinned.py: sha256 for LFS files, git blob sha1 for the others), then the
#             code and text files copied and the weights linked into OUTDIR/src. With MODEL_DIR set to a local copy of
#             the revision its files are linked and checked the same way. The folder name matters: transformers names
#             the remote-code module after it, and the decoder graph's tensor names carry that module path
#   template  OUTDIR/vl_template.jinja from the source chat_template.jinja (tools/vl_template.py), sha256 checked
#   vision    the vision encoder and projector for the fixed 512 x 512 grid (tools/export_vision.py, PY_EXPORT)
#   float     the float32 decoder export (tools/vl_decoder.py float = lfm3_work/lfm3_ext with the VL checkpoint)
#   quant     the token embedder int8 (emb_wi8) and the decoder int4 and fp16 (tools/vl_decoder.py quant)
#   pack      the two files (tools/build_vl_bundle.py, PY_EXPORT); then the float export is removed unless KEEP_FLOAT=1
#   sections  sha256 of every section of both files (nemotron_bolt_work/sections.py); with REF_DIR set (a folder that
#             holds the published files under the same names), each file is compared with it section by section
# Interpreters (env, required):
#   PY        = Python 3.12 with litert-torch 0.11.0 installed editable from a checkout at
#               f61d0b792013300ae7a1ca215d8c2f9de5ef4adf, torch 2.14.0, transformers 5.20.2, ai-edge-quantizer-nightly
#               0.10.0.dev20261023 (the decoder: float, quant)
#   PY_EXPORT = Python 3.12 with litert-torch 0.10.0, litert-converter 0.5.0, ai-edge-litert 2.3.0, ai-edge-quantizer
#               0.10.0, torch 2.14.1, litert-lm-builder 0.18.1, tokenizers (vision, pack, sections)
# Memory: the float export peaked at 23.8 GB resident, the quantization at 34.1 GB (2026-10-31). Disk: about 40 GB at
# the pack step (the source 3.8 GB, the float export 16 GB and a temporary copy of its graphs, the quantized and vision
# graphs 6 GB, the two files 6.5 GB); the float export goes after the pack unless KEEP_FLOAT=1.
set -euo pipefail
W=$(cd "$(dirname "$0")" && pwd); R=$(cd "$W/.." && pwd)
[ $# -ge 1 ] || { echo "usage: $0 OUTDIR [STEP...]"; exit 2; }
mkdir -p "$1"; OUT=$(cd "$1" && pwd); shift
: "${PY:?set PY}" "${PY_EXPORT:?set PY_EXPORT}"
export PYTHONHASHSEED=0
unset PYTHONPATH
REPO=LiquidAI/LFM3-VL-2B; REV=f83d2b0a2296aaa277f394990a9b8b949ddf9691
FILES=(LICENSE README.md chat_template.jinja config.json configuration_lfm3.py configuration_lfm3_vl.py
  generation_config.json image_processing_lfm3_vl.py lfm3_conv.py modeling_lfm3.py modeling_lfm3_vl.py
  processing_lfm3_vl.py processor_config.json tokenizer.json tokenizer_config.json video_processing_lfm3_vl.py
  model.safetensors)
NAME=LFM3-VL-2B
SRC=$OUT/src
SNAP=$OUT/hf_cache/models--${REPO//\//--}/snapshots/$REV
TEMPLATE=$OUT/vl_template.jinja
TEMPLATE_SHA=5011892a164b4d4bc426d50267f064bfe471664b6c8159ce1a5f6baf2ea28db7
TOWER=$OUT/tower
DEC=$OUT/dec
STEPS=("$@"); [ ${#STEPS[@]} -gt 0 ] || STEPS=(fetch template vision float quant pack sections)
mkdir -p "$OUT/.done"
LOG=$OUT/recipe.log
run() { echo "### CMD: $*" | tee -a "$LOG"; "$@" 2>&1 | tee -a "$LOG"; return "${PIPESTATUS[0]}"; }
done_or() { [ -f "$OUT/.done/$1" ] && { echo "skip $1 (done)"; return 0; }; return 1; }

pack() {  # pack <file name> <decoder tflite>
  rm -f "$OUT/$1.litertlm"
  run "$PY_EXPORT" -B "$W/tools/build_vl_bundle.py" --name "$1" --out-dir "$OUT" --src "$SRC" --template "$TEMPLATE" \
    --g 512 --model-type fast_vlm --decoder-bundle "$DEC/float/model.litertlm" \
    --embedder-tflite "$DEC/emb_wi8.tflite" --decoder-tflite "$2" \
    --encoder "$TOWER/vision_encoder_g512_v2.tflite" --adapter "$TOWER/vision_adapter_g512_v2_fp16.tflite" \
    --vision-act fp32
}

for step in "${STEPS[@]}"; do
  done_or "$step" && continue
  case $step in
    fetch)
      mkdir -p "$SNAP" "$SRC"
      if [ -n "${MODEL_DIR:-}" ]; then
        for f in "${FILES[@]}"; do ln -sf "$(cd "$MODEL_DIR" && pwd)/$f" "$SNAP/$f"; done
      else
        run bash "$R/dory_work/fetch_pinned.sh" "$REPO" "$REV" "$OUT/hf_cache" "${FILES[@]}"
      fi
      run "$PY" "$R/dory_work/verify_pinned.py" "$REPO" "$REV" "$OUT/hf_cache" "$OUT/fetch_manifest.json" "${FILES[@]}"
      # the code and text files are copied, the weights linked: the remote-code module's hash covers the code files'
      # names relative to this folder and their bytes
      for f in "${FILES[@]}"; do
        rm -f "$SRC/$f"
        if [ "$f" = model.safetensors ]; then ln -s "$SNAP/$f" "$SRC/$f"; else cp -L "$SNAP/$f" "$SRC/$f"; fi
      done
      ;;
    template)
      run "$PY" "$W/tools/vl_template.py" "$SRC/chat_template.jinja" "$TEMPLATE"
      got=$(shasum -a 256 "$TEMPLATE" | cut -d' ' -f1)
      [ "$got" = "$TEMPLATE_SHA" ] || { echo "template sha256 $got, expected $TEMPLATE_SHA"; exit 1; }
      ;;
    vision)
      rm -rf "$TOWER"
      run "$PY_EXPORT" -B "$W/tools/export_vision.py" --src "$SRC" --out "$TOWER" --g 512
      ;;
    float)
      rm -rf "$DEC/float"
      run "$PY" -B "$W/tools/vl_decoder.py" float --src "$SRC" --template "$TEMPLATE" --out "$DEC" --ctx 2048
      [ -f "$DEC/float/model.tflite" ] && [ -f "$DEC/float/embedder.tflite" ] && [ -f "$DEC/float/model.litertlm" ] \
        || { echo "the float export is incomplete in $DEC/float"; exit 1; }
      ;;
    quant)
      for form in emb_wi8 int4 f16; do
        rm -f "$DEC/$form.tflite"
        run "$PY" -B "$W/tools/vl_decoder.py" quant --form "$form" --out "$DEC" --ctx 2048
      done
      ;;
    pack)
      pack "${NAME}_int4" "$DEC/int4.tflite"
      pack "${NAME}_f16" "$DEC/f16.tflite"
      ls -l "$OUT/${NAME}_int4.litertlm" "$OUT/${NAME}_f16.litertlm" | tee -a "$LOG"
      [ -n "${KEEP_FLOAT:-}" ] || rm -rf "$DEC/float"
      ;;
    sections)
      for f in "${NAME}_int4" "${NAME}_f16"; do
        if [ -n "${REF_DIR:-}" ]; then
          run "$PY_EXPORT" -B "$R/nemotron_bolt_work/sections.py" "$OUT/$f.litertlm" "$REF_DIR/$f.litertlm"
        else
          run "$PY_EXPORT" -B "$R/nemotron_bolt_work/sections.py" "$OUT/$f.litertlm"
        fi
      done
      ;;
    *) echo "unknown step $step"; exit 2 ;;
  esac
  touch "$OUT/.done/$step"
done
