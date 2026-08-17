#!/usr/bin/env bash
# ibm-granite/granite-4.1-3b (DENSE GraniteForCausalLM) -> .litertlm   (2026-08-17)
#
# granite-4.1-3b is IBM's dense 3.4B instruct model. It is converted here because a dense
# model in the 3B class is what a current flagship phone can actually hold on its GPU: the
# int4 build's weights are ~2.2 GB, and it runs fully delegated on both an iPhone 17 Pro
# (Metal) and a Pixel 8a (OpenCL) — see REPRODUCE.md for the measured numbers.
#
# Rail: the plain dense export path (scripts/export_simple_template.py). granite-4.1's
# chat template renders roles exactly like templates/granite_simple.jinja
#   <|start_of_role|>{role}<|end_of_role|>{content}<|end_of_text|>\n
# (verified against the upstream chat_template.jinja on 2026-08-17), so the structured
# prompt_templates the runtime applies come out right; the upstream file's tools/documents
# branches are what the simple template drops.
#
# Toolchain: .venv-vl093 = litert-torch 0.9.3 / litert-converter 0.3.1 /
#            ai-edge-quantizer 0.8.0 / litert-lm-builder 0.16.0 / transformers 5.14.1
#            (a PRISTINE released stack — no PYTHONPATH override onto the patched
#            litert-torch checkout, so the public repro stays a pip install).
#
# Two builds:
#   int4 (BOCTAV4)          = the ship candidate. GPU weight residency is ~4x FILE size,
#                             so 1.9 GB int4 -> ~7.6 GB fits a 12-16 GB flagship, while
#                             3.4 GB int8 -> ~13.6 GB does not.
#   int8 (dynamic_wi8_afp32)= quality reference / desktop build. Granite dense int4 damage
#                             is documented as diffuse ([[granite-3.3-2b-int4-sensitive]]),
#                             so the int4 file is not shippable until a task eval says so.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv-vl093/bin/python
SRC=${SRC:-src_models/granite-4.1-3b}
OUT=granite41_work

export CACHE=${CACHE:-4096}
# Full ladder, as on granite-4.0-h-1b: the engine picks the tightest chunk per prompt.
export PREFILL=${PREFILL:-1024,512,256,128,64,32,16,8,4,2,1}
# 3.4B: keep the (tied) 100352x2560 vocab table out of the main weights section so the
# iOS ~2 GiB single-section mmap ceiling is not the thing that decides this conversion.
export EXTERNALIZE_EMBEDDER=${EXTERNALIZE_EMBEDDER:-1}
# granite's tokenizer_config says add_bos_token: False and bos == eos == <|end_of_text|>.
# litert-lm-builder writes start_token from tokenizer.bos_token regardless, and the runtime
# prepends it on the first turn (ApplyPromptTemplates) — which this model reads as a finished
# document: it echoes the question instead of answering (8Q 5/8 vs 8/8; reproduced on bf16
# PyTorch, so it is not quantization). NO_START_TOKEN=1 keeps the field out of the metadata.
export NO_START_TOKEN=${NO_START_TOKEN:-1}

RECIPES=${RECIPES:-"int4 int8"}
for R in $RECIPES; do
  case "$R" in
    int4) Q=BOCTAV4 ;;
    int8) Q=dynamic_wi8_afp32 ;;
    *) echo "unknown recipe $R"; exit 2 ;;
  esac
  echo "=== $R ($Q) ==="
  $PY scripts/export_simple_template.py "$SRC" "$OUT/out_$R" \
      templates/granite_simple.jinja "$Q" 2>&1 | tee "$OUT/export_$R.log"
  ls -la "$OUT/out_$R"/*.litertlm
done
echo "CONVERT_DONE"
