#!/bin/bash
# Pack the split granite artifacts into an EmbeddingEngine bundle (litert-lm-builder >= 0.17.0, < 0.19.2).
#   granite_embed_work/engine/pack_bundle.sh <out_dir> <embedder.tflite> <encoder.tflite> <tokenizer.json> <bundle.litertlm> [textproto]
set -euo pipefail
OUT=$1; EMB=$2; ENC=$3; TOK=$4; BUNDLE=$5; TP=${6:-}
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
