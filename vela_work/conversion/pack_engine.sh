#!/usr/bin/env bash
# Pack the EmbeddingEngine bundle (litert-lm-builder; the runtime contract of memory embedding-engine-bundle-contract):
#   scripts/pack_engine.sh <embedder.tflite> <encoder.tflite> <bundle.litertlm>
# TF_LITE_EMBEDDER = the one-token lookup (vela_lookup_*.tflite, signature embedder), TF_LITE_TEXT_ENCODER = the encoder_<L>
# signatures (a multi-signature file packed by scripts/merge_encoders.py, or one window), hf_tokenizer = the repo's
# tokenizer.json verbatim. The runtime does NOT run the tokenizer's post-processor, so the metadata declares BOTH specials the
# post-processor adds (<bos> A <eos>): bos 2 and eos 1; insert_special_tokens (engine default true) inserts them — the off
# control must FAIL (REPRODUCE.md).
set -euo pipefail
cd "$(dirname "$0")/.."
EMB=$1; ENC=$2; BUNDLE=$3
PY=${BUILDER_PY:-python3}
TP=exports/engine/embedding_metadata.textproto
mkdir -p exports/engine
cat > "$TP" <<'TXT'
embedding_model_type { generic_model {} }
bos_token { token_ids { ids: 2 } }
eos_token { token_ids { ids: 1 } }
TXT
rm -f "$BUNDLE"
"$PY" -m litert_lm_builder.litertlm_builder_cli \
  embedding_metadata --path "$TP" \
  tflite_model --path "$EMB" --model_type embedder \
  tflite_model --path "$ENC" --model_type text_encoder \
  hf_tokenizer --path hf/tokenizer.json \
  output --path "$BUNDLE"
ls -la "$BUNDLE"
shasum -a 256 "$BUNDLE"
