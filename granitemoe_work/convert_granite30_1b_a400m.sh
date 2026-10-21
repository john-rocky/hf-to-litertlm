#!/usr/bin/env bash
# convert_granite30_1b_a400m.sh OUTDIR - ibm-granite/granite-3.0-1b-a400m-instruct (pinned
# ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445, Apache-2.0; GraniteMoeForCausalLM: 24 layers, hidden 1024, 32 experts per
# layer with 8 active per token, SiLU experts, a 49,155-token vocabulary, tied embeddings)
# -> OUTDIR/granite-3.0-1b-a400m-instruct_int8.litertlm, the file of litert-community/granite-3.0-1b-a400m-instruct, in
# one command. Every step logs its command to OUTDIR/recipe.log.
#   0 checkpoint: MODEL_DIR, a local copy of the pinned commit, checked file by file against the Hub's record for the
#     commit (check_pinned.py: LFS sha256, git blob sha1, sizes; nothing is downloaded)
#   1 export: litert-torch export_hf on the litert_moe_sequential experts route (every expert computed for every
#     token with the model's SiLU; the routing weight is zero for the experts outside the token's top 8), through
#     pad_logits.py -> export_fused.py -> export_one.py (this directory):
#       export_fused.py + fused_experts.py  each layer's 32 expert FCs as two FCs: gate_up [32768, 1024] and down
#                                           [1024, 16384]; the routing weights of all 32 experts from one broadcast EQUAL
#       pad_logits.py                       one column of -1e4 appended to the decode signature's logits (49,155 -> 49,156)
#       export_one.py                       --tokenizer-verbatim packs MODEL_DIR/tokenizer.json byte for byte;
#                                           --jinja-override embeds granite30_tools.jinja
#     recipe dynamic_wi8_afp32 (channelwise int8 weights, float activations), prefill signatures 128 and 1024 + decode,
#     KV cache 2048, PYTHONHASHSEED=0 (the exporter collects the stop tokens in a Python set, so without a fixed seed
#     their order in the LlmMetadata section changes from run to run)
#   2 GPU activations: scripts/set_activation_type.py --type fp32 writes prefer_activation_type "fp32" on the TFLite
#     section (a header item; every section's bytes stay the same; the CPU path does not read it)
#   3 section table of the result (sections.py); with REFERENCE set, the comparison with that file
# The file sha256 never reproduces (the packer writes a new uuid and creation timestamp into the header); compare
# sections with the table in REPRODUCE.md.
# Environments:
#   EXPORT_PY  a Python 3.12 with granitemoe_work/requirements-export.txt installed (litert-torch at git commit
#              0d3869c6, transformers 5.20.1, torch 2.14.0 and the dated nightlies pinned there)
#   RUN_PY     a Python with litert-lm==0.18.0 (litert-lm-builder reads the bundle header)
#   LITERT_LM  the litert-lm 0.18.0 CLI (unpack / pack), default "$(dirname "$RUN_PY")/litert-lm"
#   MODEL_DIR  required: a local copy of the pinned commit, for example from
#              hf download ibm-granite/granite-3.0-1b-a400m-instruct --revision ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445 --local-dir DIR
#   REFERENCE  optional: the published .litertlm, compared section by section at the end (sections.py; exit 1 if any
#              section differs)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
[ $# -ge 1 ] || { echo "usage: MODEL_DIR=... $0 OUTDIR"; exit 2; }
[ -n "${MODEL_DIR:-}" ] || { echo "set MODEL_DIR to a local copy of the pinned commit (see the header of $0)"; exit 2; }
mkdir -p "$1"; OUT=$(cd "$1" && pwd)
EPY=${EXPORT_PY:-python}; RPY=${RUN_PY:-python}
CLI=${LITERT_LM:-$(dirname "$(command -v "$RPY")")/litert-lm}
NAME=granite-3.0-1b-a400m-instruct
log() { echo "$*" >> "$OUT/recipe.log"; }
run() { log "### CMD: $*"; "$@"; }
log "### start $(date '+%F %T %Z') MODEL_DIR=$MODEL_DIR"
run "$EPY" -I "$HERE/check_pinned.py" "$MODEL_DIR" "$OUT/checkpoint_manifest.json" > "$OUT/checkpoint.log"
# litert-torch at this commit imports tensorflow.lite.python.schema_py_generated on the export path
# (generative/layers/attention.py -> lora.py); without TensorFlow installed, this stand-in re-exports ai_edge_litert's copy.
mkdir -p "$OUT/tf_shim/tensorflow/lite/python"
: > "$OUT/tf_shim/tensorflow/__init__.py"
: > "$OUT/tf_shim/tensorflow/lite/__init__.py"
: > "$OUT/tf_shim/tensorflow/lite/python/__init__.py"
echo 'from ai_edge_litert.schema_py_generated import *  # noqa: F401,F403' > "$OUT/tf_shim/tensorflow/lite/python/schema_py_generated.py"
rm -rf "$OUT/export"
run env PYTHONHASHSEED=0 PYTHONPATH="$OUT/tf_shim" HF_HUB_OFFLINE=1 HF_HUB_CACHE="$OUT/hf_cache" HF_MODULES_CACHE="$OUT/hf_modules" \
  "$EPY" "$HERE/pad_logits.py" --routing bcast --down-groups 1 "$MODEL_DIR" "$OUT/export" litert_moe_sequential \
  --quant dynamic_wi8_afp32 --prefill 128,1024 --cache 2048 --tokenizer-verbatim --jinja-override "$HERE/granite30_tools.jinja" \
  > "$OUT/export.log" 2>&1
run "$RPY" -I "$ROOT/scripts/set_activation_type.py" "$OUT/export/model.litertlm" "$OUT/${NAME}_int8.litertlm" --type fp32 \
  --litert-lm "$CLI" > "$OUT/activation_type.log" 2>&1
shasum -a 256 "$OUT/${NAME}_int8.litertlm" | tee -a "$OUT/recipe.log"
rc=0
if [ -n "${REFERENCE:-}" ]; then
  run "$RPY" -I "$HERE/sections.py" "$OUT/${NAME}_int8.litertlm" "$REFERENCE" --json "$OUT/sections.json" \
    > "$OUT/sections.txt" || rc=$?
else
  run "$RPY" -I "$HERE/sections.py" "$OUT/${NAME}_int8.litertlm" --json "$OUT/sections.json" > "$OUT/sections.txt" || rc=$?
fi
cat "$OUT/sections.txt"
log "### done $(date '+%F %T %Z') exit $rc"
exit $rc
