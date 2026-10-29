#!/usr/bin/env bash
# convert_lfm3_17b.sh OUTDIR [STEP...] - LiquidAI/LFM3-1.7B (pinned 40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd, LFM Open
# License v1.0) -> the four files of litert-community/LFM3-1.7B-LiteRT, text in / text out, LiteRT-LM 0.18.1:
#   OUTDIR/LFM3-1.7B_mixed_int4_ekv2048.litertlm  MLP gate / up projections int4 blockwise-32, every other weight int8
#   OUTDIR/LFM3-1.7B_mixed_int4_ekv4096.litertlm  the same with a 4,096-token KV cache
#   OUTDIR/LFM3-1.7B_int8_ekv2048.litertlm        every weight int8 (dynamic_wi8_afp32)
#   OUTDIR/LFM3-1.7B_int8_ekv4096.litertlm        the same with a 4,096-token KV cache
# All four: prefill signatures 1024 / 256 / 64 / 16 / 4 / 1, the source tokenizer.json, the chat template of
# make_template.py, the sampler top-k 50 / top-p 1 / temperature 0.1, generic model type, no activation type declared.
# Steps (in order; a finished step leaves OUTDIR/.done/<step> and is skipped on a re-run; name steps to run only those):
#   fetch     the files the export reads, curl into OUTDIR/hf_cache (dory_work/fetch_pinned.sh), checked against the
#             Hub's record of the commit (dory_work/verify_pinned.py: sha256 for LFS files, git blob sha1 for the
#             others), then linked into OUTDIR/hf/LFM3-1.7B. With MODEL_DIR set to a local copy of the revision its
#             files are linked and checked the same way. The folder name matters: transformers names the remote code
#             module after it (transformers_modules.LFM3_hyphen_1_dot_7B.<hash of the code files>), and the graph's
#             tensor names carry that module path
#   template  OUTDIR/lfm3_bundle.jinja from the source chat_template.jinja (make_template.py)
#   exporter  lfm3_ext.tiny_parity_test: a random 6-layer LFM3 through the exporter's module against transformers, on
#             every path LiteRT-LM drives (seconds)
#   line1     the exporter's module against transformers float32 on the real weights (tools/line1_hf_oracle.py, then
#             tools/line1_module.py): 8 questions x thinking off / on + a 1,584-token text, each extended by the
#             bfloat16 model's greedy continuation; four paths (one prefill, token-by-token decode, 128-token prefill
#             calls with the last one padded, that prefill then decode). Not in the default steps (about an hour on a
#             busy Mac, 17 GB)
#   float2048 / float4096  the float32 export (lfm3_ext.export_lfm3, quantization_recipe 'none', the graph kept)
#   pack      the four files: quant_pack.py quantizes the float graphs (recipe_q1b.json, recipe_int8.json) and packs them
#   sections  sha256 of every section of every file (nemotron_bolt_work/sections.py); with REF_DIR set (a folder that
#             holds the published files under the same names), each file is compared with it section by section
# Interpreter (env, required):
#   PY = Python 3.12 with litert-torch 0.11.0 installed editable from a checkout at
#        f61d0b792013300ae7a1ca215d8c2f9de5ef4adf, torch 2.14.0, transformers 5.20.2, ai-edge-quantizer-nightly
#        0.10.0.dev20261023, ai-edge-litert-nightly 2.4.0.dev20261022, litert-converter 0.6.0.dev20261022,
#        litert-lm-builder 0.18.1
# Memory: a float export peaked at 23.8 GB resident, quantize + pack at 12 GB; line1 at 17 GB.
set -euo pipefail
W=$(cd "$(dirname "$0")" && pwd); R=$(cd "$W/.." && pwd)
[ $# -ge 1 ] || { echo "usage: $0 OUTDIR [STEP...]"; exit 2; }
mkdir -p "$1"; OUT=$(cd "$1" && pwd); shift
: "${PY:?set PY}"
export PYTHONHASHSEED=0 PYTHONPATH="$W"
REPO=LiquidAI/LFM3-1.7B; REV=40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd
FILES=(config.json generation_config.json tokenizer.json tokenizer_config.json special_tokens_map.json chat_template.jinja
  configuration_lfm3.py modeling_lfm3.py lfm3_conv.py model.safetensors LICENSE)
NAME=LFM3-1.7B
HF=$OUT/hf/$NAME
SNAP=$OUT/hf_cache/models--${REPO//\//--}/snapshots/$REV
TEMPLATE=$OUT/lfm3_bundle.jinja
TEMPLATE_SHA=15b329890303d5bbda354bee4e1c079b71cc1e3e5316c170012fe77405a6226c
PREFILL="[1024,256,64,16,4,1]"
STEPS=("$@"); [ ${#STEPS[@]} -gt 0 ] || STEPS=(fetch template exporter float2048 float4096 pack sections)
mkdir -p "$OUT/.done"
LOG=$OUT/recipe.log
run() { echo "### CMD: $*" | tee -a "$LOG"; "$@" 2>&1 | tee -a "$LOG"; return "${PIPESTATUS[0]}"; }
done_or() { [ -f "$OUT/.done/$1" ] && { echo "skip $1 (done)"; return 0; }; return 1; }

export_float() {  # export_float <ctx>
  local dir=$OUT/float_c$1
  rm -rf "$dir"
  run "$PY" -m lfm3_ext.export_lfm3 --code_dir="$HF" --output_dir="$dir" \
    --export "quantization_recipe='none'" --export "keep_temporary_files=True" \
    --export "cache_length=$1" --export "prefill_lengths=$PREFILL" \
    --export "use_jinja_template=True" --export "jinja_chat_template_override='$TEMPLATE'" \
    --export "tokenizer_path_override='$HF/tokenizer.json'" \
    --export "externalize_embedder=False" --export "single_token_embedder=False"
  [ -f "$dir/model.tflite" ] || { echo "no float graph in $dir"; exit 1; }
  rm -f "$dir/model.litertlm"  # the float .litertlm is not published; the graph feeds the pack step
}

pack() {  # pack <file name> <ctx> <recipe>
  run "$PY" "$W/quant_pack.py" --float "$OUT/float_c$2/model.tflite" --recipe "$W/$3" --ctx "$2" \
    --code-dir "$HF" --template "$TEMPLATE" --sampler-top-p 1.0 --out "$OUT/$1"
  ls -l "$OUT/$1" | tee -a "$LOG"
}

for step in "${STEPS[@]}"; do
  done_or "$step" && continue
  case $step in
    fetch)
      mkdir -p "$SNAP" "$HF"
      if [ -n "${MODEL_DIR:-}" ]; then
        for f in "${FILES[@]}"; do ln -sf "$(cd "$MODEL_DIR" && pwd)/$f" "$SNAP/$f"; done
      else
        run bash "$R/dory_work/fetch_pinned.sh" "$REPO" "$REV" "$OUT/hf_cache" "${FILES[@]}"
      fi
      run "$PY" "$R/dory_work/verify_pinned.py" "$REPO" "$REV" "$OUT/hf_cache" "$OUT/fetch_manifest.json" "${FILES[@]}"
      # the code and text files are copied, the weights linked: the remote code module's hash covers the code files'
      # names relative to this folder and their bytes
      for f in "${FILES[@]}"; do
        rm -f "$HF/$f"
        if [ "$f" = model.safetensors ]; then ln -s "$SNAP/$f" "$HF/$f"; else cp -L "$SNAP/$f" "$HF/$f"; fi
      done
      ;;
    template)
      run "$PY" "$W/make_template.py" "$HF/chat_template.jinja" "$TEMPLATE"
      got=$(shasum -a 256 "$TEMPLATE" | cut -d' ' -f1)
      [ "$got" = "$TEMPLATE_SHA" ] || { echo "template sha256 $got, expected $TEMPLATE_SHA"; exit 1; }
      ;;
    exporter)
      run "$PY" -m lfm3_ext.tiny_parity_test "$HF"
      ;;
    line1)
      run "$PY" "$W/tools/line1_hf_oracle.py" "$HF" "$OUT/line1_oracle"
      run "$PY" "$W/tools/line1_module.py" "$HF" "$OUT/line1_oracle" "$OUT/line1_module.json"
      run "$PY" "$W/tools/line1_table.py" "$OUT/line1_module.json" "$OUT/line1_oracle"
      ;;
    float2048) export_float 2048 ;;
    float4096) export_float 4096 ;;
    pack)
      pack "${NAME}_mixed_int4_ekv2048.litertlm" 2048 recipe_q1b.json
      pack "${NAME}_int8_ekv2048.litertlm" 2048 recipe_int8.json
      pack "${NAME}_mixed_int4_ekv4096.litertlm" 4096 recipe_q1b.json
      pack "${NAME}_int8_ekv4096.litertlm" 4096 recipe_int8.json
      [ -n "${KEEP_FLOAT:-}" ] || rm -rf "$OUT/float_c2048" "$OUT/float_c4096"
      ;;
    sections)
      for f in "${NAME}_mixed_int4_ekv2048" "${NAME}_mixed_int4_ekv4096" "${NAME}_int8_ekv2048" "${NAME}_int8_ekv4096"; do
        if [ -n "${REF_DIR:-}" ]; then
          run "$PY" "$R/nemotron_bolt_work/sections.py" "$OUT/$f.litertlm" "$REF_DIR/$f.litertlm"
        else
          run "$PY" "$R/nemotron_bolt_work/sections.py" "$OUT/$f.litertlm"
        fi
      done
      ;;
    *) echo "unknown step $step"; exit 2 ;;
  esac
  touch "$OUT/.done/$step"
done
