#!/usr/bin/env bash
# Qwen/Qwen3-0.6B -> the dynamic INT4 (block-32) GPU bundle published as litert-community/Qwen3-0.6B's
# Qwen3-0.6B_dynamic_wi4b32_afp32.litertlm (refreshed 2026-09-20, on the repo's main since 2026-09-21):
# litert-torch's stock `export_hf`, run as written on the Hub PR, with LiteRT-LM's Qwen3 metadata (the chat
# template that takes message content as a string or as a list of parts) pinned to one commit.
#
#   bash qwen3_work/export_qwen3_06b_wi4b32.sh out/qwen3-0.6b-wi4b32     # -> out/qwen3-0.6b-wi4b32/model.litertlm
#   REF=Qwen3-0.6B_dynamic_wi4b32_afp32.litertlm bash qwen3_work/export_qwen3_06b_wi4b32.sh out/x
#       # also unpacks both bundles and compares every section (needs the `litert-lm` CLI, 0.17.1+)
#
# Toolchain the shipped file was built with (2026-09-20): litert-torch main 731ef0a816691e1ef2a7296206894dcca1fa4d76
# (2026-09-17; installs as 0.10.0 from source), ai-edge-quantizer 0.9.0, litert-converter 0.4.0, ai-edge-litert 2.2.0,
# litert-lm-builder 0.16.1, torch 2.13.0, torchao 0.18.0, transformers 5.14.1, Python 3.14. The released litert-torch
# 0.9.4 wheel has no `--use_swiglu_composite`, so this recipe needs a main checkout:
#   git clone https://github.com/google-ai-edge/litert-torch && git -C litert-torch checkout 731ef0a && pip install -e litert-torch
# The exporter is a Python Fire CLI that accepts unknown flags ("additional flags are accepted"), so a misspelled flag
# is silently ignored: when a composite is missing, check the graph, not the command line.
#
# What it produces: one TFLiteModel section with prefill_128 + prefill_1024 + decode (the prefill signatures return
# logits), INT4 block-32 weights on every linear and on the embedding table, fp32 activations, float KV, the
# `odml.rope`, `odml.swiglu`, `odml.rms_norm`, `odml.runtime_bmm` and `odml.cache_update` composites (916 / 916 / 830 ops),
# fused QKV and gate/up projections, a boolean mask;
# the HF tokenizer as a zlib section; metadata from the pinned pbtext (stop <|im_end|>, TOP_P k20 p0.95 t0.6,
# max_num_tokens 4096, thought channel "<think>\n" / "\n</think>", thinking on unless the app sets enable_thinking
# false). About one minute on an M4 Max once the checkpoint is cached.
#
# Two exports of one recipe never share a file hash (the bundle carries a uuid and a creation_timestamp). The proof of
# a reproduction is the section comparison at the end: every section byte-identical to the published file, model.toml
# equal apart from those two keys.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:?output dir, e.g. out/qwen3-0.6b-wi4b32}
PY=${PY:-python3}
SRC=${SRC:-Qwen/Qwen3-0.6B}
PBTEXT_COMMIT=a8178d7dca6ac4d472476c9d76e482d8c8cbee3c
PBTEXT_SHA256=330494aa83811ce6d6f483b377cadcea7ff1bcc75659270d9260dddb62a63d33
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}   # plain HTTP for the checkpoint download; xet transfers stall on some hosts
mkdir -p "$OUT"
META="$OUT/LlmMetadataProto.pbtext"
if [ ! -s "$META" ]; then
  curl -fsSL "https://raw.githubusercontent.com/google-ai-edge/LiteRT-LM/$PBTEXT_COMMIT/models/qwen3/LlmMetadataProto.pbtext" -o "$META"
fi
echo "$PBTEXT_SHA256  $META" | shasum -a 256 -c -
"$PY" -c 'import litert_torch; print("litert_torch", getattr(litert_torch, "__version__", "?"), litert_torch.__file__)'
help_text=$("$PY" -m litert_torch.generative.export_hf --help 2>&1 || true)
if ! grep -q -- 'use_swiglu_composite' <<<"$help_text"; then
  echo "this litert-torch has no --use_swiglu_composite (the released 0.9.x wheels): install litert-torch main at 731ef0a (see the header)" >&2
  exit 2
fi
set -x
"$PY" -m litert_torch.generative.export_hf --model="$SRC" --output_dir="$OUT" \
  --quantization_recipe=dynamic_wi4b32_afp32 --prefill_lengths=128,1024 --cache_length=32768 \
  --enable_gpu_dynamic_cache=True --apply_gpu_composites=True --use_bool_mask=True \
  --fuse_qkv=True --fuse_gate_up=True --use_swiglu_composite=True --use_rope_composite=True \
  --bundle_litert_lm=True --litert_lm_llm_metadata_override="$META"
set +x
ls -la "$OUT"/model.litertlm
shasum -a 256 "$OUT"/model.litertlm

# ---- optional: prove the reproduction section for section against the published file ----
if [ -z "${REF:-}" ]; then
  echo "export done; set REF=<published .litertlm> to compare every section against it"
  exit 0
fi
CLI=${LITERT_LM_CLI:-litert-lm}
if ! command -v "$CLI" >/dev/null 2>&1; then
  echo "REF given but '$CLI' is not on PATH: pip install 'litert-lm>=0.17.1' or set LITERT_LM_CLI=/path/to/litert-lm" >&2
  exit 2
fi
rm -rf "$OUT/unpack_new" "$OUT/unpack_ref"
"$CLI" unpack "$OUT/model.litertlm" --output-dir "$OUT/unpack_new" >/dev/null
"$CLI" unpack "$REF" --output-dir "$OUT/unpack_ref" >/dev/null
status=0
for f in $(cd "$OUT/unpack_ref" && ls | grep -v '^model\.toml$'); do
  b=$(shasum -a 256 "$OUT/unpack_ref/$f" | cut -c1-64)
  if [ -f "$OUT/unpack_new/$f" ]; then a=$(shasum -a 256 "$OUT/unpack_new/$f" | cut -c1-64); else a=missing; fi
  if [ "$a" = "$b" ]; then echo "SAME  $f  $(wc -c < "$OUT/unpack_ref/$f" | tr -d " ") B  $b"; else echo "DIFF  $f  new=$a ref=$b"; status=1; fi
done
for f in $(cd "$OUT/unpack_new" && ls | grep -v '^model\.toml$'); do
  [ -f "$OUT/unpack_ref/$f" ] || { echo "DIFF  $f  only in the new bundle"; status=1; }
done
if diff <(grep -v -E 'uuid|creation_timestamp' "$OUT/unpack_new/model.toml") <(grep -v -E 'uuid|creation_timestamp' "$OUT/unpack_ref/model.toml"); then
  echo "SAME  model.toml apart from uuid / creation_timestamp"
else
  echo "DIFF  model.toml"; status=1
fi
if [ $status -eq 0 ]; then echo "REPRODUCED section for section: $OUT/model.litertlm == $REF"; else echo "NOT IDENTICAL: see the DIFF lines above" >&2; fi
exit $status
