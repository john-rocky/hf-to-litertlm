#!/bin/zsh
# PrismML Ternary-Bonsai-1.7B (ternary/1.58-bit Qwen3) -> .litertlm int4-blockwise.
#
# Ternary g128 with FP16 group scales (vendor-documented) = {-a, 0, +a} per 128
# input channels -> min-max symmetric int4 BLOCKWISE_32 holds it exactly (the
# same lossless-container argument as BitCPM-CANN; verify_ternary.py checks it
# on the actual weights — Bonsai keeps ~0.003% salient outliers in fp16, so a
# handful of blocks quantize normally instead of exactly).
#
# The model is a stock Qwen3ForCausalLM (yarn rope = static init, no wrapper
# needed; BPE tokenizer bundles as hf_tokenizer — no SP fix). Two extra steps
# vs a vanilla convert:
#   - metadata pbtext generated from the repo's chat_template.jinja verbatim
#     (always-empty <think> prefill = non-thinking 2507-style; stops 151645/151643;
#     no start_token — Qwen has no BOS)
#   - fix_zero_block_scales.py: ternary sparsity can make a 32-weight block
#     ALL-zero -> aeq min-max emits scale=0 -> XNNPACK refuses to prepare
#     ("unsupported scale value (0.000000)"). The patch swaps zero scales for
#     the tensor's min nonzero scale (blocks are all-zero, dequant unchanged).
set -e
WORK="$(cd "$(dirname "$0")" && pwd)"
MC="$WORK/../minicpm_work"
SRC=${1:-prism-ml/Ternary-Bonsai-1.7B-unpacked}
CACHE_LEN=${CACHE_LEN:-4099}

SNAP=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download('$SRC'))")

python "$WORK/../bitcpm_work/verify_ternary.py" "$SNAP/model.safetensors" 128 || true

python - "$SNAP/chat_template.jinja" "$WORK/bonsai17_LlmMetaProto.pbtext" <<'PYEOF'
import sys
# The runtime's minijinja implements startswith/endswith (and [::-1] slicing)
# but NOT .strip()/.lstrip()/.split(). The template's assistant-history branch
# (reasoning re-render, lines 34-51) calls .strip('\n') and IS executed by the
# engine's incremental assistant-turn render -> multi-turn dies on-device with
# "unknown method: string has no method named strip". Single-turn evals never
# hit it. Replace the branch with a plain verbatim emit — faithful for this
# non-thinking model (it always answers after an empty prefilled think block).
lines = open(sys.argv[1]).read().split("\n")
assert "set reasoning_content" in lines[33], lines[33]
assert "endif" in lines[50], lines[50]
patched = lines[:33] + ["        {{- '<|im_start|>' + message.role + '\\n' + content }}"] + lines[51:]
jinja = "\n".join(patched)
for bad in (".strip(", ".lstrip(", ".rstrip(", ".split("):
    assert bad not in jinja, bad
esc = (jinja.replace("\\", "\\\\").replace('"', '\\"').replace("'", "\\'")
       .replace("\n", "\\n").replace("\t", "\\t"))
pb = ('stop_tokens {\n  token_ids {\n    ids: 151645\n  }\n}\n'
      'stop_tokens {\n  token_ids {\n    ids: 151643\n  }\n}\n'
      'max_num_tokens: 4096\n'
      'llm_model_type {\n  generic_model {\n  }\n}\n'
      f'jinja_prompt_template: "{esc}"\n')
open(sys.argv[2], "w").write(pb)
print("wrote", sys.argv[2], "(minijinja-safe)")
PYEOF

litert-torch export_hf "$SNAP" "$WORK/out_fp" \
  --experimental_lightweight_conversion \
  --litert_lm_llm_metadata_override="$WORK/bonsai17_LlmMetaProto.pbtext" \
  --cache_length=$CACHE_LEN \
  --quantization_recipe=""

python "$MC/quantize_litertlm.py" apply "$WORK/out_fp/model.litertlm" \
  "$WORK/ternary-bonsai-1.7b_wi4b32_wi8_raw.litertlm" --recipe wi4b32_wi8 --algo minmax

python "$WORK/fix_zero_block_scales.py" \
  "$WORK/ternary-bonsai-1.7b_wi4b32_wi8_raw.litertlm" \
  "$WORK/ternary-bonsai-1.7b_wi4b32_wi8.litertlm"
rm -f "$WORK/ternary-bonsai-1.7b_wi4b32_wi8_raw.litertlm"
echo "DONE -> $WORK/ternary-bonsai-1.7b_wi4b32_wi8.litertlm"
