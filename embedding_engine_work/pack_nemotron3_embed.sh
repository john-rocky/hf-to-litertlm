#!/bin/bash
# Nemotron-3-Embed-1B -> EmbeddingEngine bundle. The model's tokenizer adds NO special tokens
# (post-processor = bare sequence; sentence-transformers mean-pools prompt + text), so the
# metadata declares no bos/eos and the engine's insert_special_tokens inserts nothing.
set -euo pipefail
OUT=$1; EMB=$2; ENC=$3; TOK=$4; BUNDLE=$5
PY=${BUILDER_PY:-python}   # a venv with litert-lm-builder >= 0.17.0, < 0.19.2
# litert-lm-builder 0.19.2 moves the weights into TFLiteWeights sections by default and writes
# min_runtime_version 0.19.0, so it packs a different file from the one REPRODUCE.md describes.
"$PY" - <<'PYV'
import importlib.metadata as m, re, sys
v = m.version("litert-lm-builder")
t = tuple(int(re.match(r"\d*", p).group() or 0) for p in v.split(".")[:3])
if not (0, 17, 0) <= t < (0, 19, 2):
    sys.exit(f"litert-lm-builder {v}: this pack needs >= 0.17.0, < 0.19.2 "
             "(pip install 'litert-lm-builder>=0.17.0,<0.19.2')")
PYV
TP=$OUT/embedding_metadata.textproto
printf 'embedding_model_type { generic_model {} }\n' > "$TP"
rm -f "$BUNDLE"
"$PY" -m litert_lm_builder.litertlm_builder_cli \
  embedding_metadata --path "$TP" \
  tflite_model --path "$EMB" --model_type embedder \
  tflite_model --path "$ENC" --model_type text_encoder \
  hf_tokenizer --path "$TOK" \
  output --path "$BUNDLE"
ls -la "$BUNDLE"
