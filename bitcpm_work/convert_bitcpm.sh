#!/bin/zsh
# BitCPM-CANN-1B (openbmb ternary / 1.58-bit LLM) -> .litertlm, int4-blockwise.
#
# Why this works without native 1-bit support: BitCPM-CANN stores ternary QAT
# weights materialized in bf16 — per 128-input-channel group, values are
# {-a, 0, +a}. min-max symmetric int4 BLOCKWISE_32 (blocks subdivide the 128
# groups) maps every value onto {-7, 0, +7} with zero rounding decisions; the
# only residual is fp16 rounding of the per-block scale (<= 4e-4 relative).
# int4 is a lossless container for ternary: 4 bits/weight vs the native ~1.6,
# but 4x smaller than f16, on stock CPU/GPU int4 kernels.
#
# The 1B is a stock LlamaForCausalLM (MiniCPM4 tokenizer family): reuses the
# MiniCPM4 longrope-static export + SP added-tokens fix from ../minicpm_work.
set -e
WORK="$(cd "$(dirname "$0")" && pwd)"
MC="$WORK/../minicpm_work"
SRC=${1:-openbmb/BitCPM-CANN-1B}
CACHE_LEN=${CACHE_LEN:-4099}

# 1. normalize: bin -> safetensors, minimal llama config, added_tokens.json
python "$WORK/prep_bitcpm_as_llama.py" "$SRC" "$WORK/as_llama"

# 2. (optional but recommended) verify the ternary structure + int4 exactness
python "$WORK/verify_ternary.py" "$WORK/as_llama/model.safetensors" || true

# 3. export fp32 tflite + metadata (longrope long==short -> static rope wrapper)
python "$MC/export_static_longrope.py" "$WORK/as_llama" "$WORK/out_fp" \
  --experimental_lightweight_conversion \
  --litert_lm_llm_metadata_override="$WORK/bitcpm_LlmMetaProto.pbtext" \
  --cache_length=$CACHE_LEN \
  --quantization_recipe=""

# 4. quantize: int4 b32 min-max linears (exact for ternary) + int8 cw emb/head
python "$MC/quantize_litertlm.py" apply "$WORK/out_fp/model.litertlm" \
  "$WORK/bitcpm-cann-1b_wi4b32_wi8_minmax.litertlm" --recipe wi4b32_wi8 --algo minmax

# 5. tokenizer: raw SP lacks <|im_end|>=73440 (stop never fires); HF json drops
#    spaces at decode. Bundle an SP model extended with the HF added tokens.
python "$MC/fix_sp_added_tokens.py" "$WORK/as_llama/tokenizer.model" \
  "$WORK/as_llama/added_tokens.json" "$WORK/tokenizer_fixed.spiece" 73448
python - "$WORK" <<'PYEOF'
import subprocess, sys, tempfile, os
work = sys.argv[1]
sys.path.insert(0, os.path.join(work, "..", "minicpm_work"))
from quantize_litertlm import extract_sections
parts = extract_sections(f"{work}/bitcpm-cann-1b_wi4b32_wi8_minmax.litertlm",
                         tempfile.mkdtemp(prefix="bitcpm_"))
subprocess.run([
    "litert-lm-builder",
    "llm_metadata", "--path", f"{work}/bitcpm_LlmMetaProto.pbtext",
    "sp_tokenizer", "--path", f"{work}/tokenizer_fixed.spiece",
    "tflite_model", "--path", parts["tflite"], "--model_type", "prefill_decode",
    "output", "--path", f"{work}/bitcpm-cann-1b_wi4b32_wi8.litertlm",
], check=True)
print("final:", f"{work}/bitcpm-cann-1b_wi4b32_wi8.litertlm")
PYEOF
echo "DONE -> $WORK/bitcpm-cann-1b_wi4b32_wi8.litertlm"
