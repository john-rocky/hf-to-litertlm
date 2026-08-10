#!/usr/bin/env bash
# Shieldstral-1.0-3B -> .litertlm  (text lane)
#
# Toolchain: litert-torch 0.9.2 / litert-converter 0.3.0 / ai-edge-quantizer 0.8.0 /
#            litert-lm-builder 0.15.0 / transformers 5.14.1
# Runtime  : litert-lm 0.15.0
#
# Shieldstral is Mistral3ForConditionalGeneration = a pixtral vision tower plus a
# Ministral3 text decoder. This recipe ships the TEXT lane: the vision tower is
# dropped and the standalone Ministral3ForCausalLM rides the dense export path,
# the same recipe family as litert-community/Ministral-3-3B-Instruct-2512 (whose
# base checkpoint this model is a finetune of).
#
# STRIP_SOFTMAX_COMPOSITE: litert-torch 0.9.2 emits an odml.softmax composite that
# litert-converter 0.3.0 cannot lower, so every GPU delegate rejects the graph.
# Stripping the marker leaves the math unchanged and the GPU gate passes.
set -euo pipefail
cd "$(dirname "$0")/.."

SRC=${SRC:-src_models/shieldstral-3b}
TEXT=${TEXT:-src_models/shieldstral-3b-text}
PY=${PY:-python}

# 1. Drop the vision tower -> standalone causal LM. Output-neutral for text input
#    (bit-identical yes/no logits on the floor set).
[ -d "$TEXT" ] || $PY scripts/extract_ministral3_text.py "$SRC" "$TEXT"

# 2. Export. cache 4096 leaves room for real moderation documents; decode is a
#    single token, so the usual cache-length decode tax does not apply. The full
#    prefill ladder keeps long documents off padded chunks.
export STRIP_SOFTMAX_COMPOSITE=1
export EXTERNALIZE_EMBEDDER=1
export CACHE=4096
export PREFILL=2048,1024,512,256,128,64,32,16,8,4,2,1

$PY scripts/export_simple_template.py "$TEXT" out_int8 \
    templates/shieldstral_simple.jinja dynamic_wi8_afp32

$PY scripts/export_simple_template.py "$TEXT" out_int4 \
    templates/shieldstral_simple.jinja BOCTAV4

echo "CONVERT_DONE"
ls -la out_int8/*.litertlm out_int4/*.litertlm
