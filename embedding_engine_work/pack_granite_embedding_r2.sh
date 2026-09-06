#!/bin/bash
# Pack the split granite artifacts into an EmbeddingEngine bundle (litert-lm-builder >= 0.17.0).
#   granite_embed_work/engine/pack_bundle.sh <out_dir> <embedder.tflite> <encoder.tflite> <tokenizer.json> <bundle.litertlm> [textproto]
set -euo pipefail
OUT=$1; EMB=$2; ENC=$3; TOK=$4; BUNDLE=$5; TP=${6:-}
PY=${BUILDER_PY:-python}   # any venv with litert-lm-builder >= 0.17.0
if [ -z "$TP" ]; then
  TP=$OUT/embedding_metadata.textproto
  # CLS pooling reads position 0, which the HF tokenizer fills with <bos> (id 2,
  # add_bos_token: true). Declare it so the engine's insert_special_tokens can
  # prepend it when the bundled tokenizer does not.
  cat > "$TP" <<'TXT'
embedding_model_type { generic_model {} }
bos_token { token_ids { ids: 2 } }
TXT
fi
rm -f "$BUNDLE"
"$PY" -m litert_lm_builder.litertlm_builder_cli \
  embedding_metadata --path "$TP" \
  tflite_model --path "$EMB" --model_type embedder \
  tflite_model --path "$ENC" --model_type text_encoder \
  hf_tokenizer --path "$TOK" \
  output --path "$BUNDLE"
ls -la "$BUNDLE"
