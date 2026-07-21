#!/bin/zsh
# MiniCPM family -> .litertlm via the new-style flow: full-jinja LlmMetadata override
# (chat template + "thought" channel) + post-hoc ai-edge-quantizer quantization.
# Matches the packaging of litert-community/MiniCPM5-1B.
#
# Env: python 3.11+, pip install litert-torch litert-lm "transformers==5.6.2"
#      (litert-torch 0.9.1 is incompatible with transformers 5.14+)
#
# Usage:
#   ./convert_minicpm.sh minicpm5-1b      # openbmb/MiniCPM5-1B  (thinking, wi8 recommended)
#   ./convert_minicpm.sh minicpm4-0.5b    # openbmb/MiniCPM4-0.5B (muP fold, OCTAV int4)
#   ./convert_minicpm.sh minicpm4.1-8b    # openbmb/MiniCPM4.1-8B (muP fold, thinking)
set -e
WORK="$(cd "$(dirname "$0")" && pwd)"
KEY=${1:?usage: convert_minicpm.sh <minicpm5-1b|minicpm4-0.5b|minicpm4.1-8b>}
CACHE_LEN=${CACHE_LEN:-4099}

case "$KEY" in
  minicpm5-1b)
    SRC=openbmb/MiniCPM5-1B; META=$WORK/LlmMetaProto.pbtext
    RECIPE=wi8; ALGO=minmax; FOLD=0 ;;
  minicpm4-0.5b)
    SRC=openbmb/MiniCPM4-0.5B; META=$WORK/mc4_LlmMetaProto.pbtext
    RECIPE=wi4b32_wi8; ALGO=octav; FOLD=1 ;;
  minicpm4.1-8b)
    SRC=openbmb/MiniCPM4.1-8B; META=$WORK/mc41_LlmMetaProto.pbtext
    RECIPE=wi4b32_wi8; ALGO=minmax; FOLD=1 ;;
  *) echo "unknown key $KEY"; exit 1 ;;
esac

OUT=$WORK/out_${KEY//[.\/]/_}
mkdir -p "$OUT"

if [ "$FOLD" = 1 ]; then
  # MiniCPM4/4.1: custom MiniCPMForCausalLM (muP scalings + longrope) -> stock llama
  python "$WORK/prep_minicpm4_as_llama.py" "$SRC" "$OUT/as_llama"
  # longrope long==short & factor==1 -> strip transformers' dynamic-rope update
  # (data-dependent branch that torch.export rejects)
  python "$WORK/export_static_longrope.py" "$OUT/as_llama" "$OUT/fp" \
    --experimental_lightweight_conversion \
    --litert_lm_llm_metadata_override="$META" \
    --cache_length=$CACHE_LEN \
    --quantization_recipe=""
else
  litert-torch export_hf "$SRC" "$OUT/fp" \
    --experimental_lightweight_conversion \
    --litert_lm_llm_metadata_override="$META" \
    --cache_length=$CACHE_LEN \
    --quantization_recipe=""
fi

python "$WORK/quantize_litertlm.py" apply "$OUT/fp/model.litertlm" \
  "$OUT/${KEY//[.\/]/_}_${RECIPE}_${ALGO}.litertlm" --recipe "$RECIPE" --algo "$ALGO"

if [ "$FOLD" = 1 ]; then
  # MiniCPM4/4.1 tokenizer: HF tokenizer.json loses spaces on decode; raw
  # tokenizer.model lacks the added-token <|im_end|> stop. Bundle a fixed SP model.
  python "$WORK/fix_sp_added_tokens.py" "$OUT/as_llama/tokenizer.model" \
    "$OUT/as_llama/added_tokens.json" "$OUT/tokenizer_fixed.spiece" 73448
  python - "$OUT" "$KEY" "$META" "$RECIPE" "$ALGO" <<'PYEOF'
import subprocess, sys, tempfile
out, key, meta, recipe, algo = sys.argv[1:6]
sys.path.insert(0, out.rsplit('/', 1)[0])
from quantize_litertlm import extract_sections
name = f"{key.replace('.', '_').replace('/', '_')}_{recipe}_{algo}"
parts = extract_sections(f"{out}/{name}.litertlm", tempfile.mkdtemp())
subprocess.run([
    "litert-lm-builder",
    "llm_metadata", "--path", meta,
    "sp_tokenizer", "--path", f"{out}/tokenizer_fixed.spiece",
    "tflite_model", "--path", parts["tflite"], "--model_type", "prefill_decode",
    "output", "--path", f"{out}/{name}.litertlm",
], check=True)
print("rebundled with fixed SP tokenizer:", f"{out}/{name}.litertlm")
PYEOF
fi
echo "DONE -> $OUT"
