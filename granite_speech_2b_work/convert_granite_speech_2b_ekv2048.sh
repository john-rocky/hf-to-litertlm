#!/usr/bin/env bash
# convert_granite_speech_2b_ekv2048.sh OUTDIR [STEP...] - ibm-granite/granite-speech-5.0-2b (pinned
# 981601698173ac1a71fe6838b127d8e54f8cf682, Apache-2.0) -> the second file of litert-community/granite-speech-5.0-2b:
#   OUTDIR/granite-speech-5.0-2b_ekv2048.litertlm   one .litertlm for LiteRT-LM's generic audio path: the language model
#                                          with int8 weights and a 2,048-token KV cache (litert-torch export_hf,
#                                          embedder in its own section, prefill 1,024 / 256 / 64), the audio encoder
#                                          written for the GPU and NPU delegates (port6.py: every tensor rank <= 4, no
#                                          GATHER_ND / BROADCAST_TO / DIV) with a 24-second window and fp16 weights
#                                          computed in fp32, the source's tokenizer.json; no activation type declared
# convert_granite_speech_2b.sh writes the other file (granite-speech-5.0-2b.litertlm). The two scripts can share one
# OUTDIR: the steps they have in common (fetch, shim, lm, clips, reference) write the same files under the same names
# and a finished one is skipped; every other step and file of this script carries `_ekv2048` / `6` in its name.
# Steps (in order; with no STEP named, every step runs and a finished one (OUTDIR/.done/<step>) is skipped; a named step
# always runs):
#   fetch           the 13 files the steps read, curl into OUTDIR/hf_cache (dory_work/fetch_pinned.sh), checked against
#                   the Hub's record of the commit (dory_work/verify_pinned.py: sha256 for LFS files, git blob sha1 for
#                   the others). With MODEL_DIR set to a local copy of the revision, its files are linked instead and
#                   checked the same way
#   shim            a stub `tensorflow` package in OUTDIR/tfstub: litert-torch 0.10.0 imports
#                   tensorflow.lite.python.schema_py_generated (model_ext/bonsai_flux2.py) without depending on
#                   TensorFlow; the stub re-exports ai_edge_litert's schema module under that name
#   lm              OUTDIR/lm: the language model as a standalone GraniteForCausalLM checkpoint + checks (build_lm.py)
#   lm_int8_kv2048  scripts/export_simple_template.py on OUTDIR/lm (dynamic_wi8_afp32, EXTERNALIZE_EMBEDDER=1,
#                   CACHE=2048, PREFILL=1024,256,64, NO_START_TOKEN=1, PYTHONHASHSEED=0), then collect_lm_export.py moves
#                   the two graphs and executor_metadata.pb into OUTDIR/lm_int8_kv2048. litert-torch 0.10.0's own
#                   packaging step stops on litert-lm-builder 0.19.1 (AttributeError, TfLiteModelType) after everything
#                   this bundle uses is written; the collector accepts that one stop and no other
#   encoder6        OUTDIR/encoder6: port6.py exported for a 24 s window, its batched matrix products made rank 4 again
#                   (tools/gpu_graph/tfl_rank4.py --rules bmm --unique-names), then its fp16-weight form; the operator and
#                   rank audit and the window-by-window comparison with the eager modules on the repo's
#                   multilingual_sample.wav (export_encoder6.py)
#   bundle_ekv2048  OUTDIR/granite-speech-5.0-2b_ekv2048.litertlm (build_bundle.py --cache 2048 --win-s 24: metadata,
#                   template and id checks, read-back of every section). No activation type is declared: LiteRT-LM's
#                   GPU executor then runs the language model with fp16 activations, the audio encoder with fp32
#   sections_ekv2048  type, header items, size and sha256 of every section (mnn_pocket_work/sections_items.py); with
#                   REF=<the published granite-speech-5.0-2b_ekv2048.litertlm> also the comparison section by section,
#                   header items included (exit 1 on any difference). The packer writes a new uuid and creation time
#                   into the header, so the file's own sha256 differs on every run
#   clips           OUTDIR/clips: the repo's multilingual_sample.wav and five LibriSpeech dev-clean clips,
#                   sha256-checked (fetch_clips.py)
#   transcribe_ekv2048  the clips through LiteRT-LM (transcribe.py; BACKEND=cpu|gpu for the language model,
#                   AUDIO_BACKEND=cpu|gpu for the audio encoder, THREADS; default cpu / cpu / 1 thread); with REF the
#                   published file too. The engine cache (about 3 GB) is deleted after each file
#   reference       the same clips through the source model in transformers, fp32 on the CPU (reference_hf.py)
#   compare_ekv2048 this file's transcripts against the source model's (compare.py); with REF also against the published
#                   file's, which must be identical
# Interpreters (env), the same two environments as convert_granite_speech_2b.sh:
#   PY    (required) the python of an environment made from granite_speech_2b_work/requirements.txt (Python 3.13;
#         litert-torch 0.10.0, ai-edge-quantizer 0.10.1, ai-edge-litert 2.3.0, litert-converter 0.5.0,
#         litert-lm-builder 0.19.1, torch 2.14.1, torchaudio 2.11.1, transformers 5.22.1):
#           uv venv -p 3.13 .venv-gs2b && uv pip install --python .venv-gs2b/bin/python -r granite_speech_2b_work/requirements.txt
#   RT_PY (step transcribe_ekv2048) the python of an environment with litert-lm 0.19.1 (Python API):
#           uv venv -p 3.13 .venv-gs2b-rt && uv pip install --python .venv-gs2b-rt/bin/python -r granite_speech_2b_work/requirements-runtime.txt
set -euo pipefail
R=$(cd "$(dirname "$0")/.." && pwd); W=$R/granite_speech_2b_work
[ $# -ge 1 ] || { echo "usage: $0 OUTDIR [STEP...]"; exit 2; }
mkdir -p "$1"; OUT=$(cd "$1" && pwd); shift
: "${PY:?set PY to the python of an environment made from granite_speech_2b_work/requirements.txt}"
export PYTHONHASHSEED=0 OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
REPO=ibm-granite/granite-speech-5.0-2b; REV=981601698173ac1a71fe6838b127d8e54f8cf682
FILES=(config.json generation_config.json model.safetensors tokenizer.json tokenizer_config.json special_tokens_map.json
  added_tokens.json vocab.json merges.txt chat_template.jinja preprocessor_config.json processor_config.json
  multilingual_sample.wav)
NAME=granite-speech-5.0-2b_ekv2048.litertlm
CACHE=2048; WIN_S=24; WIN_F=$((WIN_S * 100))
ENC=$OUT/encoder6/granite_speech_audio6_${WIN_F}f_fp16.tflite
SNAP=$OUT/hf_cache/models--${REPO//\//--}/snapshots/$REV
BACKEND=${BACKEND:-cpu}; AUDIO_BACKEND=${AUDIO_BACKEND:-cpu}; THREADS=${THREADS:-1}
TAG=ekv2048_${BACKEND}_a${AUDIO_BACKEND}_t${THREADS}
TR=$OUT/transcripts_$TAG.jsonl; TR_REF=$OUT/transcripts_ref_$TAG.jsonl
need_rt() {
  : "${RT_PY:?set RT_PY to the python of an environment made from granite_speech_2b_work/requirements-runtime.txt}"
}
EXPLICIT=0
STEPS=("$@")
if [ ${#STEPS[@]} -gt 0 ]; then EXPLICIT=1; else
  STEPS=(fetch shim lm lm_int8_kv2048 encoder6 bundle_ekv2048 sections_ekv2048 clips transcribe_ekv2048 reference
         compare_ekv2048); fi
mkdir -p "$OUT/.done"
LOG=$OUT/recipe.log
run() { echo "### CMD: $*" | tee -a "$LOG"; "$@" 2>&1 | tee -a "$LOG"; return "${PIPESTATUS[0]}"; }
transcribe() {  # transcribe <bundle> <out.jsonl>
  rm -rf "$OUT/engine_cache"
  run "$RT_PY" "$W/transcribe.py" --bundle "$1" --clips "$OUT/clips/clips.json" --backend "$BACKEND" \
    --audio-backend "$AUDIO_BACKEND" --threads "$THREADS" --cache-dir "$OUT/engine_cache" --out "$2"
  rm -rf "$OUT/engine_cache"
}

for step in "${STEPS[@]}"; do
  if [ $EXPLICIT = 0 ] && [ -f "$OUT/.done/$step" ]; then echo "skip $step (done)"; continue; fi
  echo "=== $step $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG"
  case $step in
    fetch)
      if [ -n "${MODEL_DIR:-}" ]; then
        M=$(cd "$MODEL_DIR" && pwd); mkdir -p "$SNAP"
        for f in "${FILES[@]}"; do ln -sf "$M/$f" "$SNAP/$f"; done
      else
        run bash "$R/dory_work/fetch_pinned.sh" "$REPO" "$REV" "$OUT/hf_cache" "${FILES[@]}"
      fi
      run "$PY" "$R/dory_work/verify_pinned.py" "$REPO" "$REV" "$OUT/hf_cache" "$OUT/fetch_manifest.json" "${FILES[@]}"
      ;;
    shim)
      mkdir -p "$OUT/tfstub/tensorflow/lite/python"
      : > "$OUT/tfstub/tensorflow/__init__.py"
      : > "$OUT/tfstub/tensorflow/lite/__init__.py"
      : > "$OUT/tfstub/tensorflow/lite/python/__init__.py"
      echo "from ai_edge_litert.schema_py_generated import *  # noqa: F401,F403" > "$OUT/tfstub/tensorflow/lite/python/schema_py_generated.py"
      ;;
    lm)
      run "$PY" "$W/build_lm.py" --src "$SNAP" --out "$OUT/lm"
      ;;
    lm_int8_kv2048)
      run "$PY" -c 'import importlib.metadata as m, sys
v = {p: m.version(p) for p in ("litert-torch", "litert-lm-builder", "ai-edge-quantizer", "ai-edge-litert", "litert-converter", "torch", "transformers")}
print(" ".join(f"{k}=={x}" for k, x in v.items()))
sys.exit(0 if (v["litert-torch"], v["litert-lm-builder"]) == ("0.10.0", "0.19.1") else "this step expects litert-torch 0.10.0 and litert-lm-builder 0.19.1 (requirements.txt)")'
      rm -rf "$OUT/lm_export_kv2048" "$OUT/lm_int8_kv2048"
      cmd=(env PYTHONPATH="$OUT/tfstub" EXTERNALIZE_EMBEDDER=1 CACHE=$CACHE PREFILL=1024,256,64 NO_START_TOKEN=1
           "$PY" scripts/export_simple_template.py "$OUT/lm" "$OUT/lm_export_kv2048" "$SNAP/chat_template.jinja" dynamic_wi8_afp32)
      echo "### CMD: (cd $R && ${cmd[*]} < /dev/null) > $OUT/lm_export_kv2048.log" | tee -a "$LOG"
      set +e
      (cd "$R" && "${cmd[@]}" < /dev/null) > "$OUT/lm_export_kv2048.log" 2>&1
      rc=$?
      set -e
      tail -n 4 "$OUT/lm_export_kv2048.log" | tee -a "$LOG"
      echo "export exit status $rc (litert-torch's own packaging step is expected to stop; checked next)" | tee -a "$LOG"
      run "$PY" "$W/collect_lm_export.py" "$OUT/lm_export_kv2048" "$OUT/lm_export_kv2048.log" "$OUT/lm_int8_kv2048"
      run shasum -a 256 "$OUT/lm_int8_kv2048/embedder_int8.tflite" "$OUT/lm_int8_kv2048/prefill_decode_int8.tflite" \
        "$OUT/lm_int8_kv2048/executor_metadata.pb"
      ;;
    encoder6)
      run "$PY" "$W/export_encoder6.py" --src "$SNAP" --out "$OUT/encoder6" --win-s $WIN_S \
        --check-wav "$SNAP/multilingual_sample.wav"
      ;;
    bundle_ekv2048)
      run "$PY" "$W/build_bundle.py" --src "$SNAP" --enc "$ENC" --lm "$OUT/lm_int8_kv2048" --cache $CACHE \
        --win-s $WIN_S --out "$OUT/$NAME"
      ls -l "$OUT/$NAME" | tee -a "$LOG"
      run shasum -a 256 "$OUT/$NAME"
      ;;
    sections_ekv2048)
      if [ -n "${REF:-}" ]; then
        run "$PY" "$R/mnn_pocket_work/sections_items.py" "$OUT/$NAME" "$REF"
      else
        run "$PY" "$R/mnn_pocket_work/sections_items.py" "$OUT/$NAME"
      fi
      ;;
    clips)
      run "$PY" "$W/fetch_clips.py" --src "$SNAP" --out "$OUT/clips"
      ;;
    transcribe_ekv2048)
      need_rt
      transcribe "$OUT/$NAME" "$TR"
      if [ -n "${REF:-}" ]; then
        transcribe "$REF" "$TR_REF"
      fi
      ;;
    reference)
      run "$PY" "$W/reference_hf.py" --src "$SNAP" --clips "$OUT/clips/clips.json" --out "$OUT/reference_hf.jsonl"
      ;;
    compare_ekv2048)
      run "$PY" "$W/compare.py" "$TR" "$OUT/reference_hf.jsonl"
      if [ -n "${REF:-}" ]; then
        run "$PY" "$W/compare.py" "$TR" "$TR_REF" --require-identical
      fi
      ;;
    *) echo "unknown step $step"; exit 2 ;;
  esac
  touch "$OUT/.done/$step"
done
