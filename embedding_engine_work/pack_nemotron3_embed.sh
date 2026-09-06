#!/bin/bash
# Nemotron-3-Embed-1B -> EmbeddingEngine bundle. The model's tokenizer adds NO special tokens
# (post-processor = bare sequence; sentence-transformers mean-pools prompt + text), so the
# metadata declares no bos/eos and the engine's insert_special_tokens inserts nothing.
set -euo pipefail
OUT=$1; EMB=$2; ENC=$3; TOK=$4; BUNDLE=$5
PY=${BUILDER_PY:-python}   # any venv with litert-lm-builder >= 0.17.0
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
