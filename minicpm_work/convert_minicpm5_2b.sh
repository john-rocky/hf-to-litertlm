#!/usr/bin/env bash
# openbmb/MiniCPM5-2B (dense LlamaForCausalLM, hybrid thinking, 2.5B) -> .litertlm   (2026-09-08)
#
# Same packaging as litert-community/MiniCPM5-1B: the LlmMetadata carries the checkpoint's
# chat_template.jinja VERBATIM (jinja path), a `thought` channel (<think>/</think>) that the
# exporter auto-declares because the template contains the literal marker, start_token <s>
# (the template's own `{{ bos_token }}` renders empty at runtime and the engine prepends the
# metadata start token instead -> exactly one <s>, as upstream), stops = generation_config eos
# [1, 130073] (+ the exporter's punctuation-prefix string stops, see REPRODUCE.md).
# `enable_thinking` stays a runtime knob (ThinkingConfig or conversation extra_context):
# unset -> the model thinks by default; true -> '<think>\n' prefill; false -> empty think block.
#
# Env: the repo's standard dense env (python >=3.11; litert-torch 0.9.3, ai-edge-quantizer 0.9.0,
# litert-lm-builder 0.16.1, transformers 5.14.1) - NOT the 0.9.1/5.6.2 env of convert_minicpm.sh.
#
# Recipes: int4 = BOCTAV4 (blockwise-32 int4 OCTAV + int8 embedding) -> needs the zero-scale
# post-fix (layer-0 MLP has 13 all-zero rows; XNNPACK refuses scale 0, the GPU delegate does not);
# int8 = dynamic_wi8_afp32 (no zero scales, no post-fix; its main section is 2.33 GB, above the
# iOS single-section mmap limit -> desktop/Android file).
#
#   bash minicpm_work/convert_minicpm5_2b.sh            # int4 + int8 -> out/minicpm5-2b/
#   RECIPES=int4 bash minicpm_work/convert_minicpm5_2b.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python3}
SRC=${SRC:-openbmb/MiniCPM5-2B}
OUT=${OUT:-out/minicpm5-2b}
mkdir -p "$OUT"

# The template is taken from the checkpoint itself, so the bundle can never drift from upstream.
TEMPLATE=$OUT/chat_template.jinja
[ -s "$TEMPLATE" ] || $PY -c "
from huggingface_hub import hf_hub_download; import shutil
shutil.copyfile(hf_hub_download('$SRC', 'chat_template.jinja'), '$TEMPLATE')"

export CACHE=${CACHE:-4096}
export PREFILL=${PREFILL:-1024,256,64,16,4,1}   # 6-signature ladder (iPhone engine-init memory)
export EXTERNALIZE_EMBEDDER=${EXTERNALIZE_EMBEDDER:-1}
export USE_JINJA=1

RECIPES=${RECIPES:-"int4 int8"}
for R in $RECIPES; do
  case "$R" in
    int4) Q=BOCTAV4 ;;
    int4b128) Q=BOCTAV4_128 ;;
    int8) Q=dynamic_wi8_afp32 ;;
    *) echo "unknown recipe $R"; exit 2 ;;
  esac
  echo "=== $R ($Q) ==="
  $PY scripts/export_simple_template.py "$SRC" "$OUT/raw_$R" "$TEMPLATE" "$Q" 2>&1 | tee "$OUT/export_$R.log"
  case "$R" in
    int4|int4b128)
      # blockwise int4: replace the all-zero blocks' scale 0 by an epsilon, in place (dequant unchanged)
      $PY minicpm_work/fix_zero_scales_inplace.py "$OUT/raw_$R/model.litertlm" "$OUT/MiniCPM5-2B_$R.litertlm" ;;
    *)
      cp "$OUT/raw_$R/model.litertlm" "$OUT/MiniCPM5-2B_$R.litertlm" ;;
  esac
  ls -la "$OUT/MiniCPM5-2B_$R.litertlm"
done
echo "CONVERT_DONE"
