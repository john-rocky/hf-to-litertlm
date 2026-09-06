#!/bin/bash
# LFM2.5-Embedding-350M -> EmbeddingEngine bundle. CLS = position 0 = <|startoftext|> (id 1),
# which the HF tokenizer's post-processor prepends and the runtime's tokenizer does NOT — so
# declare it as bos and let the engine's insert_special_tokens (default true) add it.
set -euo pipefail
OUT=$1; EMB=$2; ENC=$3; TOK=$4; BUNDLE=$5
PY=${BUILDER_PY:-python}   # any venv with litert-lm-builder >= 0.17.0
TP=$OUT/embedding_metadata.textproto
printf 'embedding_model_type { generic_model {} }\nbos_token { token_ids { ids: 1 } }\n' > "$TP"
rm -f "$BUNDLE"
"$PY" -m litert_lm_builder.litertlm_builder_cli \
  embedding_metadata --path "$TP" \
  tflite_model --path "$EMB" --model_type embedder \
  tflite_model --path "$ENC" --model_type text_encoder \
  hf_tokenizer --path "$TOK" \
  output --path "$BUNDLE"
ls -la "$BUNDLE"
