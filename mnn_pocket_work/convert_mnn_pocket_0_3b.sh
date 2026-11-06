#!/usr/bin/env bash
# convert_mnn_pocket_0_3b.sh OUTDIR [STEP...] - taobao-mnn/MNN-Pocket-0.3B (pinned 087a52fd9ff0dee9a2a0b2605172fd55d8af9e85,
# Apache-2.0) -> a file of litert-community/MNN-Pocket-0.3B:
#   OUTDIR/MNN-Pocket-0.3B_int8.litertlm   litert-torch's stock export_hf, int8 weights (one scale per output channel) with
#                                          float activations, GPU composites, prefill signatures of 128 and 1,024 tokens,
#                                          a fixed 4,096-token KV cache, the source's tokenizer.json, the chat template
#                                          templates/mnn_pocket_0_3b.jinja
# Without the options below the script writes the repo's first file (model type qwen2p5, uploaded 2026-11-06). The files
# of 2026-11-07 take ATTN=1 TYPE=generic_model, and KV=2048 for the second one (scripts/reproduce_llm.sh keys
# mnn-pocket-0.3b-attn-kv4096 and mnn-pocket-0.3b-attn-kv2048):
#   ATTN=1               the attention composites too (--use_qkv_norm_rope_composite=True --use_sdpa_composite=True;
#                        default False / False)
#   TYPE=generic_model   the LlmMetadata's llm_model_type (default qwen2p5; make_pbtext.py --model-type)
#   KV=2048              the KV cache length: --cache_length and the LlmMetadata's max_num_tokens (default 4096); any
#                        other value than 4096 names the output MNN-Pocket-0.3B_int8_ekv<KV>.litertlm
# Steps (in order; a finished step leaves OUTDIR/.done/<step> and is skipped on a re-run; name steps to run only those):
#   fetch     the 10 files the export reads, curl into OUTDIR/hf_cache (dory_work/fetch_pinned.sh), checked against the
#             Hub's record of the commit (dory_work/verify_pinned.py: sha256 for LFS files, git blob sha1 for the others).
#             With MODEL_DIR set to a local copy of the revision, its files are linked into OUTDIR/hf_cache instead and
#             checked the same way
#   shim      a stub `tensorflow` package in OUTDIR/tfstub: litert-torch 0.10.0 imports
#             tensorflow.lite.python.schema_py_generated (model_ext/bonsai_flux2.py) without depending on TensorFlow; the
#             stub re-exports ai_edge_litert's schema module under that name (a full TensorFlow install clashes with
#             litert-converter at process start)
#   pbtext    OUTDIR/LlmMetadataProto.pbtext from templates/mnn_pocket_0_3b.jinja (mnn_pocket_work/make_pbtext.py)
#   int8      `python -m litert_torch.generative.export_hf` with the flags below, PYTHONHASHSEED=0 (the exporter writes the
#             stop tokens in the iteration order of a set)
#   act       only with ACT=fp32 (or fp16): declare that GPU activation type on the TFLiteModel section
#             (scripts/set_activation_type.py: an unpack / pack with the litert-lm CLI named by LITERT_LM, read back; the
#             section bytes stay the same). Without ACT the file declares none and LiteRT-LM's GPU runs it in fp16
#   sections  type, header items, size and sha256 of every section of the file (mnn_pocket_work/sections_items.py); with
#             REF=<a published MNN-Pocket-0.3B_int8.litertlm> also the comparison section by section, header items
#             included (exit 1 on any difference). The packer writes a new uuid and creation time into the header, so
#             the file's own sha256 differs on every run
# Interpreter (env, required):
#   PY = the python of an environment made from mnn_pocket_work/requirements.txt (Python 3.13; litert-torch 0.10.0,
#        ai-edge-quantizer 0.10.1, ai-edge-litert 2.3.0, litert-converter 0.5.0, litert-lm-builder 0.18.1, torch 2.14.1,
#        torchao 0.18.0, transformers 5.22.1):
#          uv venv -p 3.13 .venv-mnn-pocket && uv pip install --python .venv-mnn-pocket/bin/python -r mnn_pocket_work/requirements.txt
#        litert-lm-builder 0.19.0 and 0.19.1 do not have litert_lm_builder.TfLiteModelType, which litert-torch 0.10.0
#        packs the bundle with (AttributeError at the "Package model" step).
#   LITERT_LM (step act only) = a litert-lm CLI from its own environment (litert-lm 0.19.1 needs litert-lm-builder
#        0.19.1, which the export environment cannot hold):
#          uv venv -p 3.13 .venv-litert-lm && uv pip install --python .venv-litert-lm/bin/python litert-lm==0.19.1
set -euo pipefail
R=$(cd "$(dirname "$0")/.." && pwd); W=$R/mnn_pocket_work
[ $# -ge 1 ] || { echo "usage: $0 OUTDIR [STEP...]"; exit 2; }
mkdir -p "$1"; OUT=$(cd "$1" && pwd); shift
: "${PY:?set PY to the python of an environment made from mnn_pocket_work/requirements.txt}"
export PYTHONHASHSEED=0
REPO=taobao-mnn/MNN-Pocket-0.3B; REV=087a52fd9ff0dee9a2a0b2605172fd55d8af9e85
FILES=(config.json generation_config.json model.safetensors tokenizer.json tokenizer_config.json special_tokens_map.json
  added_tokens.json vocab.json merges.txt chat_template.jinja)
ATTN=${ATTN:-}; TYPE=${TYPE:-qwen2p5}; KV=${KV:-4096}
case "$ATTN" in ""|0) ATTN_FLAG=False ;; 1) ATTN_FLAG=True ;; *) echo "ATTN must be 1 or empty"; exit 2 ;; esac
case "$TYPE" in qwen2p5|generic_model) ;; *) echo "TYPE must be qwen2p5 or generic_model"; exit 2 ;; esac
case "$KV" in ''|*[!0-9]*) echo "KV must be a number of tokens"; exit 2 ;; esac
NAME=MNN-Pocket-0.3B_int8.litertlm
[ "$KV" = 4096 ] || NAME=MNN-Pocket-0.3B_int8_ekv$KV.litertlm
SNAP=$OUT/hf_cache/models--${REPO//\//--}/snapshots/$REV
ACT=${ACT:-}
case "$ACT" in ""|fp32|fp16) ;; *) echo "ACT must be fp32, fp16 or empty"; exit 2 ;; esac
STEPS=("$@"); [ ${#STEPS[@]} -gt 0 ] || { STEPS=(fetch shim pbtext int8); [ -n "$ACT" ] && STEPS+=(act); STEPS+=(sections); }
mkdir -p "$OUT/.done"
LOG=$OUT/recipe.log
run() { echo "### CMD: $*" | tee -a "$LOG"; "$@" 2>&1 | tee -a "$LOG"; return "${PIPESTATUS[0]}"; }
done_or() { [ -f "$OUT/.done/$1" ] && { echo "skip $1 (done)"; return 0; }; return 1; }

for step in "${STEPS[@]}"; do
  done_or "$step" && continue
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
    pbtext)
      run "$PY" "$W/make_pbtext.py" --template "$R/templates/mnn_pocket_0_3b.jinja" --model-type "$TYPE" \
        --max-num-tokens "$KV" --out "$OUT/LlmMetadataProto.pbtext"
      run shasum -a 256 "$R/templates/mnn_pocket_0_3b.jinja" "$OUT/LlmMetadataProto.pbtext"
      ;;
    int8)
      work=$OUT/work_int8
      rm -rf "$work"; mkdir -p "$work"
      run "$PY" -c 'import importlib.metadata as m; print(" ".join(f"{p}=={m.version(p)}" for p in ("litert-torch", "ai-edge-quantizer", "ai-edge-litert", "litert-converter", "litert-lm-builder", "torch", "transformers")))'
      run env PYTHONPATH="$OUT/tfstub" "$PY" -m litert_torch.generative.export_hf \
        --model="$SNAP" --output_dir="$work" \
        --quantization_recipe=dynamic_wi8_afp32 --prefill_lengths=128,1024 --cache_length="$KV" \
        --enable_gpu_dynamic_cache=False --apply_gpu_composites=True --use_bool_mask=True \
        --fuse_qkv=True --fuse_gate_up=True --use_rope_composite=True --use_swiglu_composite=True \
        --use_qkv_norm_rope_composite="$ATTN_FLAG" --use_sdpa_composite="$ATTN_FLAG" \
        --tokenizer_path_override="$SNAP/tokenizer.json" \
        --bundle_litert_lm=True --litert_lm_llm_metadata_override="$OUT/LlmMetadataProto.pbtext"
      mv "$work/model.litertlm" "$OUT/$NAME"
      rm -rf "$work"
      ls -l "$OUT/$NAME" | tee -a "$LOG"
      run shasum -a 256 "$OUT/$NAME"
      ;;
    act)
      [ -n "$ACT" ] || { echo "step act needs ACT=fp32 or ACT=fp16"; exit 2; }
      run "$PY" "$R/scripts/set_activation_type.py" "$OUT/$NAME" "$OUT/act.tmp.litertlm" --type "$ACT" --litert-lm "${LITERT_LM:-litert-lm}"
      mv "$OUT/act.tmp.litertlm" "$OUT/$NAME"
      run shasum -a 256 "$OUT/$NAME"
      ;;
    sections)
      if [ -n "${REF:-}" ]; then
        run "$PY" "$W/sections_items.py" "$OUT/$NAME" "$REF"
      else
        run "$PY" "$W/sections_items.py" "$OUT/$NAME"
      fi
      ;;
    *) echo "unknown step $step"; exit 2 ;;
  esac
  touch "$OUT/.done/$step"
done
