#!/usr/bin/env bash
# convert_lfm3_omni.sh OUTDIR [STEP...] - LiquidAI/LFM3-Omni-2B (pinned ce2458cbce4ad970f92032ed94e3d18d30b04f9c, LFM
# Open License v1.0) -> the .litertlm file and the speech-out files of litert-community/LFM3-Omni-2B-LiteRT:
#   OUTDIR/LFM3-Omni-2B_int8.litertlm   text, one picture and audio in, text out (LiteRT-LM 0.18.1, generic_model):
#       the decoder rewritten per channel with smooth_scales.npz (the same float32 function) and then dynamic int8
#       (int8 per-channel weights, float32 activations), a 4,096-token KV cache, the embedder's table int8, the audio
#       encoder in float32, the vision encoder in float32 and the vision adapter with float16 weights at 512 x 512, the
#       source tokenizer.json and the chat template of make_template.py
#   OUTDIR/speech/LFM3-Omni-2B_depth_step_f32.tflite         one depth-decoder pass (3 per speech frame)
#   OUTDIR/speech/LFM3-Omni-2B_depth_frame2_f32.tflite       a whole speech frame in one call: the depth projection,
#       the 3 passes and the sampler; the 12 codes out
#   OUTDIR/speech/LFM3-Omni-2B_frame_embed_f32.tflite        the 12 codes of a frame -> the next decoder input
#   OUTDIR/speech/LFM3-Omni-2B_codec_decoder_f12_f32.tflite  the codec decoder, 12 frames (0.96 s) per call
#   OUTDIR/speech/LFM3-Omni-2B_codec_decoder_f1_f32.tflite   the codec decoder, 1 frame per call
#   OUTDIR/speech/codec_window_f32.npy                       the codec's overlap-add window
# Steps (in order; a finished step leaves OUTDIR/.done/<step> and is skipped on a re-run; name steps to run only those):
#   fetch     the files the conversion reads, curl into OUTDIR/hf_cache (dory_work/fetch_pinned.sh), checked against the
#             Hub's record of the commit (dory_work/verify_pinned.py: sha256 for LFS files, git blob sha1 for the
#             others), then copied (the weights linked) into OUTDIR/hf_small. With MODEL_DIR set to a local copy of the
#             revision its files are linked and checked the same way. The folder name matters: transformers names the
#             remote code module after it (transformers_modules.hf_small.<hash of the code files>), and the decoder
#             graph's tensor names carry that module path
#   template  OUTDIR/lfm3omni_bundle.jinja from the source chat_template.jinja (make_template.py)
#   rewrite   OUTDIR/smooth/model.safetensors: the language-model weights with the per-channel scales of
#             smooth_scales.npz folded in (smooth_weights.py)
#   lm        the decoder in float32 from the rewritten weights (export_lm.py = lfm3_work's lfm3_ext.export_lfm3:
#             embeddings in, logits and the final-norm `hidden` out, prefill 1024 / 256 / 64 / 16 / 4 / 1, KV 4,096).
#             Removes the rewritten weights unless KEEP_FLOAT=1
#   quant     dynamic int8 of the decoder and the embedder (quant_int8.py). Removes the float graphs unless KEEP_FLOAT=1
#   vision    the vision encoder (float32) and adapter (float16 weights) at 512 x 512 (export_vision.py)
#   audio     the audio encoder section, float32 (export_audio.py)
#   bundle    build_bundle.py: the .litertlm file
#   speech    the speech-out graphs: depth step, frame embedding, codec decoder for 12 frames and for 1 frame, and the
#             codec window (export_speech.py, one part per process)
#   frame2    the depth frame graph (export_depth_frame2.py, run as the main script: its wrapper class's module name
#             `__main__` is in the graph's tensor names)
#   sections  sha256 of every file; the .litertlm by section (nemotron_bolt_work/sections.py). With REF_DIR set (a
#             folder laid out like the Hub repo: the .litertlm at the top, the speech files under speech/), each file is
#             compared with it, the .litertlm section by section (its header holds a new id and creation time per pack)
# Interpreters (env, required):
#   PY       = Python 3.12 with litert-torch 0.11.0 installed editable from a checkout at
#              f61d0b792013300ae7a1ca215d8c2f9de5ef4adf, torch 2.14.0, transformers 5.20.2, ai-edge-quantizer-nightly
#              0.10.0.dev20261023, ai-edge-litert-nightly 2.4.0.dev20261022, litert-converter 0.6.0.dev20261022,
#              litert-lm-builder 0.18.1, numpy 2.5.4, safetensors 0.8.0, tokenizers 0.23.3 (steps fetch, template,
#              rewrite, lm, quant, bundle, sections)
#   PY_PARTS = Python 3.12 with litert-torch 0.10.0, litert-converter 0.5.0, ai-edge-litert 2.3.0, ai-edge-quantizer
#              0.10.0, torch 2.14.1, transformers 5.21.0, numpy 2.5.4, safetensors 0.8.0 (steps vision, audio, speech,
#              frame2)
set -euo pipefail
W=$(cd "$(dirname "$0")" && pwd); R=$(cd "$W/.." && pwd)
[ $# -ge 1 ] || { echo "usage: $0 OUTDIR [STEP...]"; exit 2; }
mkdir -p "$1"; OUT=$(cd "$1" && pwd); shift
export PYTHONHASHSEED=0 PYTHONPATH="$R/lfm3_work"
REPO=LiquidAI/LFM3-Omni-2B; REV=ce2458cbce4ad970f92032ed94e3d18d30b04f9c
FILES=(config.json generation_config.json tokenizer.json tokenizer_config.json chat_template.jinja processor_config.json
  configuration_lfm3.py modeling_lfm3.py lfm3_conv.py LICENSE model.safetensors audio_tokenizer/config.json
  audio_tokenizer/model.safetensors)
HF=$OUT/hf_small
SNAP=$OUT/hf_cache/models--${REPO//\//--}/snapshots/$REV
TEMPLATE=$OUT/lfm3omni_bundle.jinja
TEMPLATE_SHA=425f89b67cd00700b69d8a11bb5f0c1c179db88248d470bce707ba67f03b3abd
SCALES=$W/smooth_scales.npz
SCALES_SHA=10a9d1d3642c0c982e8069c3ff2117410d7646e7e0db37a577e8e4a7ddb2128a
SMOOTH=$OUT/smooth/model.safetensors
FLOAT=$OUT/lm_f32_c4096
LM=$OUT/lm_int8_c4096
VISION=$OUT/vision
AUDIO=$OUT/audio/lfm3omni_audio_section_T3000_f32.tflite
BUNDLE=LFM3-Omni-2B_int8.litertlm
SP=$OUT/speech
SPEECH=(LFM3-Omni-2B_depth_step_f32.tflite LFM3-Omni-2B_depth_frame2_f32.tflite LFM3-Omni-2B_frame_embed_f32.tflite
  LFM3-Omni-2B_codec_decoder_f12_f32.tflite LFM3-Omni-2B_codec_decoder_f1_f32.tflite codec_window_f32.npy)
STEPS=("$@"); [ ${#STEPS[@]} -gt 0 ] || STEPS=(fetch template rewrite lm quant vision audio bundle speech frame2 sections)
mkdir -p "$OUT/.done" "$OUT/logs"
LOG=$OUT/recipe.log
# run <step> <command...>: the command line to LOG, its output to logs/<step>.log, its result lines to LOG too
run() {
  local step=$1; shift
  echo "### CMD: $*" >> "$LOG"
  local t0=$SECONDS rc=0
  "$@" >> "$OUT/logs/$step.log" 2>&1 || rc=$?
  grep -aE 'sha256|rewrote|exported|INT8|PACKED|IDENTICAL|DIFFERENT|^(OK|BAD) |PASS|FAIL|Error|Traceback|chars' \
    "$OUT/logs/$step.log" | tail -n 40 >> "$LOG" || true
  echo "    rc=$rc ${SECONDS}s total, $((SECONDS - t0))s this command" >> "$LOG"
  return $rc
}
sha() { shasum -a 256 "$1" | cut -d' ' -f1; }
done_or() { [ -f "$OUT/.done/$1" ] && { echo "skip $1 (done)" | tee -a "$LOG"; return 0; }; return 1; }
need() { local v; for v in "$@"; do [ -n "${!v:-}" ] || { echo "set $v (see the header of $0)"; exit 2; }; done; }

for step in "${STEPS[@]}"; do
  done_or "$step" && continue
  echo "## step $step $(date '+%F %T')" | tee -a "$LOG"
  case $step in
    fetch)
      need PY
      mkdir -p "$SNAP/audio_tokenizer" "$HF/audio_tokenizer"
      if [ -n "${MODEL_DIR:-}" ]; then
        for f in "${FILES[@]}"; do ln -sf "$(cd "$MODEL_DIR" && pwd)/$f" "$SNAP/$f"; done
      else
        run fetch bash "$R/dory_work/fetch_pinned.sh" "$REPO" "$REV" "$OUT/hf_cache" "${FILES[@]}"
      fi
      run fetch "$PY" "$R/dory_work/verify_pinned.py" "$REPO" "$REV" "$OUT/hf_cache" "$OUT/fetch_manifest.json" "${FILES[@]}"
      # the code and text files are copied, the weights linked: the remote code module's hash covers the code files'
      # names relative to this folder and their bytes
      for f in "${FILES[@]}"; do
        rm -f "$HF/$f"
        case $f in *.safetensors) ln -s "$SNAP/$f" "$HF/$f" ;; *) cp -L "$SNAP/$f" "$HF/$f" ;; esac
      done
      ;;
    template)
      need PY
      run template "$PY" "$W/make_template.py" "$HF/chat_template.jinja" "$TEMPLATE"
      got=$(sha "$TEMPLATE")
      [ "$got" = "$TEMPLATE_SHA" ] || { echo "template sha256 $got, expected $TEMPLATE_SHA" | tee -a "$LOG"; exit 1; }
      ;;
    rewrite)
      need PY
      got=$(sha "$SCALES")
      [ "$got" = "$SCALES_SHA" ] || { echo "smooth_scales.npz sha256 $got, expected $SCALES_SHA" | tee -a "$LOG"; exit 1; }
      rm -f "$SMOOTH"
      run rewrite "$PY" "$W/smooth_weights.py" "$SCALES" "$HF" "$SMOOTH"
      ;;
    lm)
      need PY
      rm -rf "$FLOAT"
      run lm "$PY" "$W/export_lm.py" "$HF" "$SMOOTH" "$FLOAT"
      for f in model.tflite embedder.tflite; do
        [ -f "$FLOAT/$f" ] || { echo "no $f in $FLOAT" | tee -a "$LOG"; exit 1; }
      done
      [ -n "${KEEP_FLOAT:-}" ] || rm -f "$SMOOTH"
      ;;
    quant)
      need PY
      rm -rf "$LM"
      run quant "$PY" "$W/quant_int8.py" "$FLOAT" "$LM"
      [ -n "${KEEP_FLOAT:-}" ] || rm -f "$FLOAT/model.tflite" "$FLOAT/embedder.tflite"
      ;;
    vision)
      need PY_PARTS
      run vision "$PY_PARTS" "$W/export_vision.py" "$HF/model.safetensors" "$HF/config.json" "$VISION"
      ;;
    audio)
      need PY_PARTS
      rm -f "$AUDIO"
      run audio "$PY_PARTS" "$W/export_audio.py" "$HF" "$AUDIO"
      ;;
    bundle)
      need PY
      rm -f "$OUT/$BUNDLE"
      run bundle "$PY" "$W/build_bundle.py" --hf "$HF" --lm "$LM" --audio "$AUDIO" \
        --vision-encoder "$VISION/vision_encoder_g512.tflite" --vision-adapter "$VISION/vision_adapter_g512_fp16.tflite" \
        --template "$TEMPLATE" --out "$OUT/$BUNDLE"
      ;;
    speech)
      need PY_PARTS
      mkdir -p "$SP"
      run speech "$PY_PARTS" "$W/export_speech.py" --part depth --hf "$HF" --out "$SP/LFM3-Omni-2B_depth_step_f32.tflite"
      run speech "$PY_PARTS" "$W/export_speech.py" --part frame --hf "$HF" --out "$SP/LFM3-Omni-2B_frame_embed_f32.tflite"
      run speech "$PY_PARTS" "$W/export_speech.py" --part codec12 --hf "$HF" \
        --out "$SP/LFM3-Omni-2B_codec_decoder_f12_f32.tflite" --window "$SP/codec_window_f32.npy"
      run speech "$PY_PARTS" "$W/export_speech.py" --part codec1 --hf "$HF" --out "$SP/LFM3-Omni-2B_codec_decoder_f1_f32.tflite"
      ;;
    frame2)
      need PY_PARTS
      mkdir -p "$SP"
      run frame2 "$PY_PARTS" "$W/export_depth_frame2.py" "$HF/model.safetensors" "$SP/LFM3-Omni-2B_depth_frame2_f32.tflite"
      ;;
    sections)
      need PY
      if [ -n "${REF_DIR:-}" ]; then
        run sections "$PY" "$R/nemotron_bolt_work/sections.py" "$OUT/$BUNDLE" "$REF_DIR/$BUNDLE" || true
      else
        run sections "$PY" "$R/nemotron_bolt_work/sections.py" "$OUT/$BUNDLE"
      fi
      for g in "${SPEECH[@]}"; do
        line="speech/$g $(wc -c < "$SP/$g" | tr -d ' ') B sha256 $(sha "$SP/$g")"
        if [ -n "${REF_DIR:-}" ]; then
          [ "$(sha "$SP/$g")" = "$(sha "$REF_DIR/speech/$g")" ] && line="$line IDENTICAL" \
            || line="$line DIFFERENT (published $(sha "$REF_DIR/speech/$g"))"
        fi
        echo "$line" | tee -a "$LOG"
      done
      ;;
    *) echo "unknown step $step"; exit 2 ;;
  esac
  touch "$OUT/.done/$step"
done
