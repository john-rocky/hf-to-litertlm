#!/usr/bin/env bash
# convert_mnn_pocket_0_3b_npu.sh OUTDIR [STEP...] - taobao-mnn/MNN-Pocket-0.3B (pinned 087a52fd9ff0dee9a2a0b2605172fd55d8af9e85,
# Apache-2.0) -> OUTDIR/MNN-Pocket-0.3B_qualcomm_npu_ekv2048.litertlm, the Qualcomm NPU file of
# litert-community/MNN-Pocket-0.3B: litert-torch's experimental npu_export for the HTP of the Snapdragon 8 Elite Gen 5
# (SM8850), static range quantization (int16 activations, int8 weights), one prefill signature of 128 tokens, a
# 2,048-token KV cache, no AOT step (LiteRT-LM compiles the graphs for the HTP each time an engine is created).
# Steps (in order; a finished step leaves OUTDIR/.done/<step> and is skipped on a re-run; name steps to run only those):
#   fetch      the 10 files of the source, as mnn_pocket_work/convert_mnn_pocket_0_3b.sh does (dory_work/fetch_pinned.sh
#              into OUTDIR/hf_cache, checked by dory_work/verify_pinned.py; MODEL_DIR=<a local copy of the revision>
#              links that copy instead and checks it the same way)
#   tree       litert-torch at commit 6d4c622c9a3aade4d411a1e859c6bbe571d38ee0 cloned into OUTDIR/litert-torch, with
#              litert_torch_6d4c622_split_cache_mask.diff applied (the prefill attention mask added to the logits by
#              broadcast instead of a 7-fold concatenation, so the int16 mask the ADD reads stays under 1 MiB:
#              557,056 B instead of 3,899,392 B; the HTP adds an operand above 1 MiB to the wrong rows,
#              https://github.com/google-ai-edge/litert-torch/issues/1184), the patched file's sha256 checked
#   pbtext     OUTDIR/LlmMetadataProto.pbtext (mnn_pocket_work/make_pbtext.py: model type qwen2p5, max_num_tokens 2048,
#              the chat template templates/mnn_pocket_0_3b.jinja)
#   calib      OUTDIR/calib/ALL.jsonl, the 48 calibration prompts (make_calib.py; checked against the published file's)
#   export     stage 1 -> OUTDIR/exported/model.litertlm (export_npu.py export)
#   calibrate  stage 2 -> OUTDIR/calib_bounds/ (export_npu.py calibrate; about 6 minutes on an M4 Max)
#   rope       OUTDIR/calib_bounds_rope/: the rope inputs' range widened to the whole cache (rope_calib_range.py)
#   quantize   stage 3 -> OUTDIR/MNN-Pocket-0.3B_qualcomm_npu_ekv2048.litertlm (export_npu.py quantize)
#   sections   type, header items, size and sha256 of every section (mnn_pocket_work/sections_items.py); with REF=<the
#              published MNN-Pocket-0.3B_qualcomm_npu_ekv2048.litertlm> the comparison section by section (exit 1 on
#              any difference). The packer writes a new uuid and creation time into the header, so the file's own
#              sha256 differs on every run
# Interpreter (env, required): PY = the python of an environment made from mnn_pocket_work/npu/requirements.txt
#   (Python 3.14; ai-edge-quantizer 0.9.0, ai-edge-litert 2.2.0, litert-converter 0.4.0, litert-lm-builder 0.17.0,
#   torch 2.13.0, transformers 5.14.1; litert-torch comes from the tree step):
#     uv venv -p 3.14 .venv-mnn-npu && uv pip install --python .venv-mnn-npu/bin/python -r mnn_pocket_work/npu/requirements.txt
set -euo pipefail
R=$(cd "$(dirname "$0")/../.." && pwd); W=$R/mnn_pocket_work; N=$W/npu
[ $# -ge 1 ] || { echo "usage: $0 OUTDIR [STEP...]"; exit 2; }
mkdir -p "$1"; OUT=$(cd "$1" && pwd); shift
: "${PY:?set PY to the python of an environment made from mnn_pocket_work/npu/requirements.txt}"
export PYTHONHASHSEED=0
REPO=taobao-mnn/MNN-Pocket-0.3B; REV=087a52fd9ff0dee9a2a0b2605172fd55d8af9e85
FILES=(config.json generation_config.json model.safetensors tokenizer.json tokenizer_config.json special_tokens_map.json
  added_tokens.json vocab.json merges.txt chat_template.jinja)
LT_URL=https://github.com/google-ai-edge/litert-torch.git; LT_REV=6d4c622c9a3aade4d411a1e859c6bbe571d38ee0
ATTN_PY=litert_torch/generative/export_hf/core/split_cache/attention.py
ATTN_SHA=08ba37d1ec9d8c5126f33f202497393150c9ad419e3a0a72b0276eadc15d87bb
NAME=MNN-Pocket-0.3B_qualcomm_npu_ekv2048.litertlm
SNAP=$OUT/hf_cache/models--${REPO//\//--}/snapshots/$REV
TREE=$OUT/litert-torch
STEPS=("$@"); [ ${#STEPS[@]} -gt 0 ] || STEPS=(fetch tree pbtext calib export calibrate rope quantize sections)
mkdir -p "$OUT/.done"
LOG=$OUT/recipe.log
run() { echo "### CMD: $*" | tee -a "$LOG"; "$@" 2>&1 | tee -a "$LOG"; return "${PIPESTATUS[0]}"; }
done_or() { [ -f "$OUT/.done/$1" ] && { echo "skip $1 (done)"; return 0; }; return 1; }
lt() { run env PYTHONPATH="$TREE" "$PY" "$@"; }   # litert-torch from the patched tree

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
    tree)
      rm -rf "$TREE"
      run git clone -q --filter=blob:none "$LT_URL" "$TREE"
      run git -C "$TREE" checkout -q "$LT_REV"
      run git -C "$TREE" apply "$N/litert_torch_6d4c622_split_cache_mask.diff"
      got=$(shasum -a 256 "$TREE/$ATTN_PY" | cut -d' ' -f1)
      echo "patched $ATTN_PY sha256 $got" | tee -a "$LOG"
      [ "$got" = "$ATTN_SHA" ] || { echo "the patched file differs from the one the published file was built with ($ATTN_SHA)" | tee -a "$LOG"; exit 1; }
      ;;
    pbtext)
      run "$PY" "$W/make_pbtext.py" --template "$R/templates/mnn_pocket_0_3b.jinja" --model-type qwen2p5 \
        --max-num-tokens 2048 --out "$OUT/LlmMetadataProto.pbtext"
      run shasum -a 256 "$R/templates/mnn_pocket_0_3b.jinja" "$OUT/LlmMetadataProto.pbtext"
      ;;
    calib)
      run "$PY" "$N/make_calib.py" "$SNAP" "$TREE" "$OUT/calib"
      ;;
    export)
      run "$PY" -c 'import importlib.metadata as m; print(" ".join(f"{p}=={m.version(p)}" for p in ("ai-edge-quantizer", "ai-edge-litert", "litert-converter", "litert-lm-builder", "torch", "transformers")))'
      rm -rf "$OUT/exported"
      lt "$N/export_npu.py" export "$SNAP" "$OUT/LlmMetadataProto.pbtext" "$OUT/exported"
      run shasum -a 256 "$OUT/exported/model.litertlm"
      ;;
    calibrate)
      rm -rf "$OUT/calib_bounds"
      lt "$N/export_npu.py" calibrate "$SNAP" "$OUT/exported/model.litertlm" "$OUT/calib" "$OUT/calib_bounds"
      (cd "$OUT/calib_bounds" && run shasum -a 256 ./*)
      ;;
    rope)
      run "$PY" "$N/rope_calib_range.py" "$OUT/calib_bounds" "$OUT/calib_bounds_rope"
      ;;
    quantize)
      lt "$N/export_npu.py" quantize "$SNAP" "$OUT/exported/model.litertlm" "$OUT/calib_bounds_rope" "$OUT/$NAME"
      ls -l "$OUT/$NAME" | tee -a "$LOG"
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
