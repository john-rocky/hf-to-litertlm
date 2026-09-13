#!/usr/bin/env bash
# openbmb/MiniCPM5-2B -> .litertlm with the LiteRT-LM `models/minicpm5/README.md#model-conversion` command,
# run as written (2026-09-11). This is the recipe behind litert-community/MiniCPM5-2B's int8 file.
#
#   bash minicpm_work/export_minicpm5_2b_readme.sh dynamic_wi8_afp32      out/minicpm5-2b-readme-int8
#   bash minicpm_work/export_minicpm5_2b_readme.sh dynamic_wi4c_hr_afp32  out/minicpm5-2b-readme-int4hr
#
# Only three things differ from the README block: the model id (the README names the 1B), the metadata
# path, and the output dir. The metadata override is the repository's LlmMetadataProto.pbtext, PINNED to
# the commit the shipped file was built with (b5e34ab1, 2026-09-01; sha256 below) — on 2026-09-11 the file
# on main moved to a template that takes message content as multimodal parts only, so an unpinned copy
# gives a different bundle. The command needs nothing beyond the released wheels: litert-torch 0.9.4,
# litert-converter 0.4.0, ai-edge-quantizer 0.9.0, ai-edge-litert 2.2.0, litert-lm-builder 0.16.1,
# transformers 5.14.1, torch 2.13.0 (`pip install litert-torch==0.9.4`; python >= 3.11).
#
# What it produces (int8): one TFLiteModel section with prefill_128 + decode, int8 per-channel on every
# linear incl. the in-graph embedding and lm_head, `odml.cache_update` composites, a boolean mask; metadata
# = the pbtext (start <s>, stops </s> + <|im_end|>, max_num_tokens 4096, thought channel "<think>\n" /
# "</think>", the LiteRT-LM canonical chat template whose enable_thinking defaults to false).
# `--cache_length=32771` is the runtime's magic number: the KV cache is sized from the metadata's 4096
# at load. ~1 min on an M4 Max for int8, ~2.5 min for int4 HR (`--experimental_lightweight_conversion`).
set -euo pipefail
cd "$(dirname "$0")/.."
RECIPE=${1:?quantization recipe, e.g. dynamic_wi8_afp32}
OUT=${2:?output dir}
PY=${PY:-python3}
SRC=${SRC:-openbmb/MiniCPM5-2B}
PBTEXT_COMMIT=b5e34ab1
PBTEXT_SHA256=d95e3dc7ff4b71c69e54d0f3246a140bca070f6d3c4dcede70e8f5ae3b8a40b7
mkdir -p "$OUT"
META="$OUT/minicpm_metadata.textpb"
if [ ! -s "$META" ]; then
  curl -fsSL "https://raw.githubusercontent.com/google-ai-edge/LiteRT-LM/$PBTEXT_COMMIT/models/minicpm5/LlmMetadataProto.pbtext" -o "$META"
fi
echo "$PBTEXT_SHA256  $META" | shasum -a 256 -c -
"$PY" -c 'import litert_torch; print("litert_torch", getattr(litert_torch, "__version__", "?"))'
set -x
"$(dirname "$(command -v "$PY")")/litert-torch" export_hf \
  --model="$SRC" \
  --litert_lm_llm_metadata_override="$PWD/$META" \
  --quantization_recipe="$RECIPE" \
  --use_bool_mask=True \
  --apply_gpu_composites=True \
  --output_dir="$PWD/$OUT" \
  --cache_length=32771 \
  --experimental_lightweight_conversion
set +x
ls -la "$OUT"/*.litertlm
shasum -a 256 "$OUT"/*.litertlm
