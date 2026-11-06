#!/usr/bin/env bash
# convert_granite_speech_2b.sh OUTDIR [STEP...] - ibm-granite/granite-speech-5.0-2b (pinned 981601698173ac1a71fe6838b127d8e54f8cf682,
# Apache-2.0) -> the file of litert-community/granite-speech-5.0-2b:
#   OUTDIR/granite-speech-5.0-2b.litertlm   one .litertlm for LiteRT-LM's generic audio path: the language model with int8
#                                          weights (litert-torch export_hf, embedder in its own section, KV cache 4,096,
#                                          prefill 1,024 / 256 / 64), the audio encoder (log-mel front end, conformer,
#                                          Q-Former and projector in one graph, port.py) with fp16 weights computed in
#                                          fp32, the source's tokenizer.json, fp32 activations declared for the GPU
# Steps (in order; with no STEP named, every step runs and a finished one (OUTDIR/.done/<step>) is skipped; a named step
# always runs):
#   fetch       the 13 files the steps read, curl into OUTDIR/hf_cache (dory_work/fetch_pinned.sh), checked against the
#               Hub's record of the commit (dory_work/verify_pinned.py: sha256 for LFS files, git blob sha1 for the
#               others). With MODEL_DIR set to a local copy of the revision, its files are linked instead and checked
#               the same way
#   shim        a stub `tensorflow` package in OUTDIR/tfstub: litert-torch 0.10.0 imports
#               tensorflow.lite.python.schema_py_generated (model_ext/bonsai_flux2.py) without depending on TensorFlow;
#               the stub re-exports ai_edge_litert's schema module under that name
#   lm          OUTDIR/lm: the language model as a standalone GraniteForCausalLM checkpoint + checks (build_lm.py)
#   lm_int8     scripts/export_simple_template.py on OUTDIR/lm (dynamic_wi8_afp32, EXTERNALIZE_EMBEDDER=1, CACHE=4096,
#               PREFILL=1024,256,64, NO_START_TOKEN=1, PYTHONHASHSEED=0), then collect_lm_export.py moves the two
#               graphs and executor_metadata.pb into OUTDIR/lm_int8. litert-torch 0.10.0's own packaging step stops on
#               litert-lm-builder 0.19.1 (AttributeError, TfLiteModelType) after everything this bundle uses is
#               written; the collector accepts that one stop and no other
#   encoder     OUTDIR/encoder: the fp32 and fp16 audio encoder files + a check of each against the eager port on the
#               repo's multilingual_sample.wav (export_encoder.py)
#   bundle      OUTDIR/granite-speech-5.0-2b_undeclared.litertlm (build_bundle.py: metadata, template and id checks,
#               read-back of every section)
#   act         fp32 declared on every TFLite section (scripts/set_activation_type.py with the litert-lm CLI named by
#               LITERT_LM; an unpack / pack, read back; the section bytes stay the same) -> OUTDIR/granite-speech-5.0-2b.litertlm
#   sections    type, header items, size and sha256 of every section (mnn_pocket_work/sections_items.py); with
#               REF=<the published granite-speech-5.0-2b.litertlm> also the comparison section by section, header items
#               included (exit 1 on any difference). The packer writes a new uuid and creation time into the header, so
#               the file's own sha256 differs on every run
#   clips       OUTDIR/clips: the repo's multilingual_sample.wav and five LibriSpeech dev-clean clips, sha256-checked
#               (fetch_clips.py)
#   transcribe  the clips through LiteRT-LM (transcribe.py, BACKEND=cpu|gpu, THREADS, default cpu / 1 thread); with REF
#               the published file too
#   reference   the same clips through the source model in transformers, fp32 on the CPU (reference_hf.py)
#   compare     this file's transcripts against the source model's (compare.py); with REF also against the published
#               file's, which must be identical
# Interpreters (env):
#   PY    (required) the python of an environment made from granite_speech_2b_work/requirements.txt (Python 3.13;
#         litert-torch 0.10.0, ai-edge-quantizer 0.10.1, ai-edge-litert 2.3.0, litert-converter 0.5.0,
#         litert-lm-builder 0.19.1, torch 2.14.1, torchaudio 2.11.1, transformers 5.22.1):
#           uv venv -p 3.13 .venv-gs2b && uv pip install --python .venv-gs2b/bin/python -r granite_speech_2b_work/requirements.txt
#   RT_PY (steps act, transcribe) the python of an environment with litert-lm 0.19.1 (CLI and Python API):
#           uv venv -p 3.13 .venv-gs2b-rt && uv pip install --python .venv-gs2b-rt/bin/python -r granite_speech_2b_work/requirements-runtime.txt
#         LITERT_LM defaults to the litert-lm CLI next to RT_PY.
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
NAME=granite-speech-5.0-2b.litertlm
UNDECL=granite-speech-5.0-2b_undeclared.litertlm
SNAP=$OUT/hf_cache/models--${REPO//\//--}/snapshots/$REV
BACKEND=${BACKEND:-cpu}; THREADS=${THREADS:-1}
TR=$OUT/transcripts_${BACKEND}_t${THREADS}.jsonl; TR_REF=$OUT/transcripts_ref_${BACKEND}_t${THREADS}.jsonl
need_rt() {
  : "${RT_PY:?set RT_PY to the python of an environment made from granite_speech_2b_work/requirements-runtime.txt}"
  LITERT_LM=${LITERT_LM:-$(dirname "$RT_PY")/litert-lm}
}
EXPLICIT=0
STEPS=("$@")
if [ ${#STEPS[@]} -gt 0 ]; then EXPLICIT=1; else
  STEPS=(fetch shim lm lm_int8 encoder bundle act sections clips transcribe reference compare); fi
mkdir -p "$OUT/.done"
LOG=$OUT/recipe.log
run() { echo "### CMD: $*" | tee -a "$LOG"; "$@" 2>&1 | tee -a "$LOG"; return "${PIPESTATUS[0]}"; }

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
    lm_int8)
      run "$PY" -c 'import importlib.metadata as m, sys
v = {p: m.version(p) for p in ("litert-torch", "litert-lm-builder", "ai-edge-quantizer", "ai-edge-litert", "litert-converter", "torch", "transformers")}
print(" ".join(f"{k}=={x}" for k, x in v.items()))
sys.exit(0 if (v["litert-torch"], v["litert-lm-builder"]) == ("0.10.0", "0.19.1") else "this step expects litert-torch 0.10.0 and litert-lm-builder 0.19.1 (requirements.txt)")'
      rm -rf "$OUT/lm_export" "$OUT/lm_int8"
      cmd=(env PYTHONPATH="$OUT/tfstub" EXTERNALIZE_EMBEDDER=1 CACHE=4096 PREFILL=1024,256,64 NO_START_TOKEN=1
           "$PY" scripts/export_simple_template.py "$OUT/lm" "$OUT/lm_export" "$SNAP/chat_template.jinja" dynamic_wi8_afp32)
      echo "### CMD: (cd $R && ${cmd[*]} < /dev/null) > $OUT/lm_export.log" | tee -a "$LOG"
      set +e
      (cd "$R" && "${cmd[@]}" < /dev/null) > "$OUT/lm_export.log" 2>&1
      rc=$?
      set -e
      tail -n 4 "$OUT/lm_export.log" | tee -a "$LOG"
      echo "export exit status $rc (litert-torch's own packaging step is expected to stop; checked next)" | tee -a "$LOG"
      run "$PY" "$W/collect_lm_export.py" "$OUT/lm_export" "$OUT/lm_export.log" "$OUT/lm_int8"
      run shasum -a 256 "$OUT/lm_int8/embedder_int8.tflite" "$OUT/lm_int8/prefill_decode_int8.tflite" "$OUT/lm_int8/executor_metadata.pb"
      ;;
    encoder)
      run "$PY" "$W/export_encoder.py" --src "$SNAP" --out "$OUT/encoder" --check-wav "$SNAP/multilingual_sample.wav"
      ;;
    bundle)
      run "$PY" "$W/build_bundle.py" --src "$SNAP" --enc "$OUT/encoder/granite_speech_audio_3600f_fp16.tflite" \
        --lm "$OUT/lm_int8" --out "$OUT/$UNDECL"
      ;;
    act)
      need_rt
      run "$PY" "$R/scripts/set_activation_type.py" "$OUT/$UNDECL" "$OUT/$NAME" --type fp32 --litert-lm "$LITERT_LM"
      rm -f "$OUT/$UNDECL"
      ls -l "$OUT/$NAME" | tee -a "$LOG"
      run shasum -a 256 "$OUT/$NAME"
      ;;
    sections)
      if [ -n "${REF:-}" ]; then
        run "$PY" "$R/mnn_pocket_work/sections_items.py" "$OUT/$NAME" "$REF"
      else
        run "$PY" "$R/mnn_pocket_work/sections_items.py" "$OUT/$NAME"
      fi
      ;;
    clips)
      run "$PY" "$W/fetch_clips.py" --src "$SNAP" --out "$OUT/clips"
      ;;
    transcribe)
      need_rt
      run "$RT_PY" "$W/transcribe.py" --bundle "$OUT/$NAME" --clips "$OUT/clips/clips.json" --backend "$BACKEND" \
        --threads "$THREADS" --out "$TR"
      if [ -n "${REF:-}" ]; then
        run "$RT_PY" "$W/transcribe.py" --bundle "$REF" --clips "$OUT/clips/clips.json" --backend "$BACKEND" \
          --threads "$THREADS" --out "$TR_REF"
      fi
      ;;
    reference)
      run "$PY" "$W/reference_hf.py" --src "$SNAP" --clips "$OUT/clips/clips.json" --out "$OUT/reference_hf.jsonl"
      ;;
    compare)
      run "$PY" "$W/compare.py" "$TR" "$OUT/reference_hf.jsonl"
      if [ -n "${REF:-}" ]; then
        run "$PY" "$W/compare.py" "$TR" "$TR_REF" --require-identical
      fi
      ;;
    *) echo "unknown step $step"; exit 2 ;;
  esac
  touch "$OUT/.done/$step"
done
