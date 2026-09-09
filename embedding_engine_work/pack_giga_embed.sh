#!/bin/bash
# Giga-Embeddings-instruct-480M-0826 -> EmbeddingEngine bundle (litert-lm-builder >= 0.17.0).
#   embedding_engine_work/pack_giga_embed.sh <out_dir> <embedder.tflite> <encoder.tflite> <tokenizer.json> <bundle.litertlm>
# The model's tokenizer post-processor is "<s> A </s>" (bos 1, eos 2, both inside the mean) and
# the runtime does not run it, so the metadata declares both and the engine's
# insert_special_tokens (default true) prepends <s> and appends </s>.
set -euo pipefail
OUT=$1; EMB=$2; ENC=$3; TOK=$4; BUNDLE=$5
PY=${BUILDER_PY:-$HOME/venvs/lt0170run/bin/python}
TP=$OUT/embedding_metadata.textproto
cat > "$TP" <<'TXT'
embedding_model_type { generic_model {} }
bos_token { token_ids { ids: 1 } }
eos_token { token_ids { ids: 2 } }
TXT
rm -f "$BUNDLE"
"$PY" -m litert_lm_builder.litertlm_builder_cli \
  embedding_metadata --path "$TP" \
  tflite_model --path "$EMB" --model_type embedder \
  tflite_model --path "$ENC" --model_type text_encoder \
  hf_tokenizer --path "$TOK" \
  output --path "$BUNDLE"
ls -la "$BUNDLE"
