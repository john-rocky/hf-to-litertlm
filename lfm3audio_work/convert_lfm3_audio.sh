#!/usr/bin/env bash
# convert_lfm3_audio.sh OUTDIR [STEP...] - LiquidAI/LFM3-Audio-2B (pinned 9d49701c372d6011d900f6abb40c6b296fb706dd, LFM
# Open License v1.0) -> the six files of litert-community/LFM3-Audio-2B-LiteRT:
#   OUTDIR/lfm3-audio-2b_int4_b128_c2048_h30.litertlm   text and audio in, text out (LiteRT-LM 0.18.1): the decoder in
#       int4 (blockwise 128; its tied output head and the embedder int8), a 2,048-token KV cache, the audio encoder
#       with fp16 weights, the source tokenizer.json and the chat template of make_template.py
#   OUTDIR/lfm3_audio_depth_proj_r4b.tflite             speech out: the depth decoder's per-frame projection (fp32)
#   OUTDIR/lfm3_audio_depth_pass_r4b.tflite             speech out: one depth-decoder pass, 3 per frame (fp32)
#   OUTDIR/lfm3_audio_frame_embed_f32.tflite            speech out: the 12 codes of a frame -> the next decoder input
#   OUTDIR/lfm3_audio_codec_decoder_f12_r4b2.tflite     speech out: the codec decoder, 12 frames (0.96 s) per call
#   OUTDIR/lfm3_audio_depth_pass_r5b_aot.tflite         speech out: the depth-decoder pass compiled ahead of time for
#       the Hexagon NPU of the Qualcomm SM8850 (Galaxy S26), QAIRT 2.47.0
# Steps (in order; a finished step leaves OUTDIR/.done/<step> and is skipped on a re-run; name steps to run only those):
#   fetch     the files the conversion reads, curl into OUTDIR/hf_cache (dory_work/fetch_pinned.sh), checked against the
#             Hub's record of the commit (dory_work/verify_pinned.py: sha256 for LFS files, git blob sha1 for the
#             others), then copied (the weights linked) into OUTDIR/hf. With MODEL_DIR set to a local copy of the
#             revision its files are linked and checked the same way. The folder name matters: transformers names the
#             remote code module after it (transformers_modules.hf.<hash of the code files>), and the decoder graph's
#             tensor names carry that module path
#   template  OUTDIR/lfm3audio_bundle.jinja from the source chat_template.jinja (make_template.py)
#   lm        the decoder in float32 (lfm3_work's lfm3_ext.export_lfm3: decoder weights under `lfm.model.`, the `lfm`
#             block of config.json, embeddings in, logits and the final-norm `hidden` out, the embedder outside with
#             one token per call, prefill signatures 1024 / 256 / 64 / 16 / 4 / 1, KV cache 2,048); the 8 GB float
#             .litertlm the export also writes is removed (the bundle step takes the graphs and the metadata files)
#   quant     lm_quant.py with recipe_int4_b128.json: every FULLY_CONNECTED int4 blockwise-128 except the tied output
#             head (int8 per channel), the embedder's table int8 per channel; block scales floored at 2^-14. Removes
#             the float graphs unless KEEP_FLOAT=1
#   audio     the audio encoder section (scripts/export_parts.py --part audio = r4b_parts.AudioSectionR4b), then its
#             fully connected weights cast to fp16 (scripts/quant_fp16w.py)
#   bundle    build_bundle.py: the .litertlm file
#   parts     the speech-out graphs (scripts/export_parts.py --part frame / depth / codec12)
#   npu       the NPU file: the depth-decoder pass traced with the HTP rewrites (scripts/export_parts.py --part
#             depth_npu), compiled ahead of time for SM8850 in the image of npu/Dockerfile (docker, linux/amd64)
#   sections  sha256 of every file; the .litertlm by section (nemotron_bolt_work/sections.py). With REF_DIR set (a
#             folder that holds the published files under the same names), each file is compared with it, the
#             .litertlm section by section (its header holds a new id and creation time per pack)
# Interpreters (env, required):
#   PY       = Python 3.12 with litert-torch 0.11.0 installed editable from a checkout at
#              f61d0b792013300ae7a1ca215d8c2f9de5ef4adf, torch 2.14.0, transformers 5.20.2, ai-edge-quantizer-nightly
#              0.10.0.dev20261023, ai-edge-litert-nightly 2.4.0.dev20261022, litert-converter 0.6.0.dev20261022,
#              litert-lm-builder 0.18.1, numpy 2.5.4, safetensors 0.8.0, tokenizers 0.23.3 (steps fetch, template, lm,
#              quant, bundle, sections)
#   PY_PARTS = Python 3.12 with litert-torch 0.10.0, litert-converter 0.5.0, ai-edge-litert 2.3.0, ai-edge-quantizer
#              0.10.0, torch 2.14.0, transformers 5.20.2, numpy 2.5.4, safetensors 0.8.0 (steps audio, parts, npu)
#   AOT_IMAGE (default npu-aot-qc:2.2.0-full) = the image built from npu/Dockerfile (step npu)
set -euo pipefail
W=$(cd "$(dirname "$0")" && pwd); R=$(cd "$W/.." && pwd)
[ $# -ge 1 ] || { echo "usage: $0 OUTDIR [STEP...]"; exit 2; }
mkdir -p "$1"; OUT=$(cd "$1" && pwd); shift
export PYTHONHASHSEED=0 PYTHONPATH="$R/lfm3_work"
REPO=LiquidAI/LFM3-Audio-2B; REV=9d49701c372d6011d900f6abb40c6b296fb706dd
FILES=(config.json generation_config.json tokenizer.json tokenizer_config.json special_tokens_map.json chat_template.jinja
  processor_config.json preprocessor_config.json configuration_lfm3.py modeling_lfm3.py lfm3_conv.py
  configuration_lfm3_audio.py modeling_lfm3_audio.py feature_extraction_lfm3_audio.py processing_lfm3_audio.py
  configuration_liquid_codec.py modeling_liquid_codec.py LICENSE model.safetensors audio_tokenizer/config.json
  audio_tokenizer/model.safetensors)
HF=$OUT/hf
SNAP=$OUT/hf_cache/models--${REPO//\//--}/snapshots/$REV
TEMPLATE=$OUT/lfm3audio_bundle.jinja
TEMPLATE_SHA=4f876bc5fb98ba3073def1f58ef11619bc70dac4cf1e845d29be900a74d37f26
FLOAT=$OUT/lm_f32_c2048
LM=$OUT/lm_int4_b128
PARTS=$OUT/parts
BUNDLE=lfm3-audio-2b_int4_b128_c2048_h30.litertlm
GRAPHS=(lfm3_audio_depth_proj_r4b.tflite lfm3_audio_depth_pass_r4b.tflite lfm3_audio_frame_embed_f32.tflite
  lfm3_audio_codec_decoder_f12_r4b2.tflite lfm3_audio_depth_pass_r5b_aot.tflite)
AOT_IMAGE=${AOT_IMAGE:-npu-aot-qc:2.2.0-full}
STEPS=("$@"); [ ${#STEPS[@]} -gt 0 ] || STEPS=(fetch template lm quant audio bundle parts npu sections)
mkdir -p "$OUT/.done" "$OUT/logs"
LOG=$OUT/recipe.log
# run <step> <command...>: the command line to LOG, its output to logs/<step>.log, its result lines to LOG too
run() {
  local step=$1; shift
  echo "### CMD: $*" >> "$LOG"
  local t0=$SECONDS rc=0
  "$@" >> "$OUT/logs/$step.log" 2>&1 || rc=$?
  grep -aE 'sha256|PACKED|PACK CHECK|floored|tensors loaded|rewrites traced| B \(|Partition|offloaded|IDENTICAL|DIFFERENT|^(OK|BAD) |PASS|FAIL|chars sha256' \
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
    lm)
      need PY
      rm -rf "$FLOAT"
      run lm "$PY" -m lfm3_ext.export_lfm3 --code_dir="$HF" --output_dir="$FLOAT" --weights="$HF/model.safetensors" \
        --key_prefix=lfm.model. --config_json="$HF/config.json" --config_key=lfm --tokenizer_dir="$HF" --emit_hidden \
        --export "quantization_recipe='none'" --export "keep_temporary_files=True" --export "cache_length=2048" \
        --export "prefill_lengths=[1024,256,64,16,4,1]" --export "externalize_embedder=True" \
        --export "single_token_embedder=True" --export "use_rope_composite=False" --export "use_jinja_template=True" \
        --export "jinja_chat_template_override='$TEMPLATE'" --export "tokenizer_path_override='$HF/tokenizer.json'" \
        --export "bundle_litert_lm=True"
      for f in model.tflite embedder.tflite llm_metadata.pb executor_metadata.pb; do
        [ -f "$FLOAT/$f" ] || { echo "no $f in $FLOAT" | tee -a "$LOG"; exit 1; }
      done
      rm -f "$FLOAT/model.litertlm"  # the float .litertlm is not published; its graphs feed the quant step
      ;;
    quant)
      need PY
      run quant "$PY" "$W/lm_quant.py" --src "$FLOAT" --recipe "$W/recipe_int4_b128.json" --out "$LM"
      [ -n "${KEEP_FLOAT:-}" ] || rm -f "$FLOAT/model.tflite" "$FLOAT/embedder.tflite"
      ;;
    audio)
      need PY_PARTS
      run audio "$PY_PARTS" "$W/scripts/export_parts.py" --part audio --hf "$HF" --out "$PARTS/lfm3_audio_section_T3000_r4b2.tflite"
      run audio "$PY_PARTS" "$W/scripts/quant_fp16w.py" "$PARTS/lfm3_audio_section_T3000_r4b2.tflite" \
        "$PARTS/lfm3_audio_section_T3000_r4b2_fp16w.tflite"
      ;;
    bundle)
      need PY
      rm -f "$OUT/$BUNDLE"
      run bundle "$PY" "$W/build_bundle.py" --hf "$HF" --export "$FLOAT" --lm "$LM" \
        --audio "$PARTS/lfm3_audio_section_T3000_r4b2_fp16w.tflite" --template "$TEMPLATE" --out "$OUT/$BUNDLE"
      ;;
    parts)
      need PY_PARTS
      run parts "$PY_PARTS" "$W/scripts/export_parts.py" --part frame --hf "$HF" --out "$OUT/lfm3_audio_frame_embed_f32.tflite"
      run parts "$PY_PARTS" "$W/scripts/export_parts.py" --part depth --hf "$HF" --out "$OUT/lfm3_audio_depth_proj_r4b.tflite" \
        --out2 "$OUT/lfm3_audio_depth_pass_r4b.tflite"
      run parts "$PY_PARTS" "$W/scripts/export_parts.py" --part codec12 --hf "$HF" --out "$OUT/lfm3_audio_codec_decoder_f12_r4b2.tflite"
      ;;
    npu)
      need PY_PARTS
      mkdir -p "$OUT/npu"
      run npu "$PY_PARTS" "$W/scripts/export_parts.py" --part depth_npu --hf "$HF" --out "$OUT/npu/lfm3_audio_depth_pass_r5b.tflite"
      rm -rf "$OUT/npu/aot"
      run npu docker run --rm --name "lfm3_audio_aot_$$" --platform linux/amd64 -v "$OUT/npu:/w" -v "$W/npu:/aot:ro" \
        "$AOT_IMAGE" python /aot/aot_compile_sm8850.py /w/lfm3_audio_depth_pass_r5b.tflite /w/aot
      cp "$OUT/npu/aot/lfm3_audio_depth_pass_r5b_Qualcomm_SM8850_apply_plugin.tflite" "$OUT/lfm3_audio_depth_pass_r5b_aot.tflite"
      ;;
    sections)
      need PY
      if [ -n "${REF_DIR:-}" ]; then
        run sections "$PY" "$R/nemotron_bolt_work/sections.py" "$OUT/$BUNDLE" "$REF_DIR/$BUNDLE" || true
      else
        run sections "$PY" "$R/nemotron_bolt_work/sections.py" "$OUT/$BUNDLE"
      fi
      for g in "${GRAPHS[@]}"; do
        line="$g $(wc -c < "$OUT/$g" | tr -d ' ') B sha256 $(sha "$OUT/$g")"
        if [ -n "${REF_DIR:-}" ]; then
          [ "$(sha "$OUT/$g")" = "$(sha "$REF_DIR/$g")" ] && line="$line IDENTICAL" || line="$line DIFFERENT (published $(sha "$REF_DIR/$g"))"
        fi
        echo "$line" | tee -a "$LOG"
      done
      ;;
    *) echo "unknown step $step"; exit 2 ;;
  esac
  touch "$OUT/.done/$step"
done
