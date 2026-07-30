#!/bin/bash
# Copies the app's bundled resources (tokenizer tables + pipeline meta) and
# the LiteRT runtime dylib pair. Resources/ and Frameworks/ are not committed —
# run this before xcodegen + build.
#
# Runtime pair: ai-edge-litert 2.1.6 (libLiteRt.dylib +
# libLiteRtMetalAccelerator.dylib from the SAME wheel). Keep the pair
# same-generation: mixing generations makes the accelerator reject the
# serialized options, and the LiteRT-main 2026-07-31 prebuilt pair SIGSEGVs
# in xnn_x8_transposec during Metal delegate init on the 2.27 GiB DiT.
set -euo pipefail
cd "$(dirname "$0")"
SRC="${1:-$HOME/models/bonsai-image-4b-tflite/hf_upload}"
RUNTIME="${RUNTIME:-$HOME/models/litert-prebuilt/ai_edge_litert_216}"

mkdir -p Resources Frameworks
cp "$SRC/tokenizer/vocab.json" "$SRC/tokenizer/merges.txt" "$SRC/pipeline_meta.json" Resources/

if [ ! -f "$RUNTIME/libLiteRt.dylib" ]; then
  echo "Runtime pair not found at $RUNTIME — extracting from the ai-edge-litert wheel"
  TMP=$(mktemp -d)
  python3 -m pip download ai-edge-litert==2.1.6 --no-deps --only-binary :all: \
    --platform macosx_12_0_arm64 -d "$TMP"
  unzip -o -q "$TMP"/ai_edge_litert-*.whl -d "$TMP/wheel" \
    "ai_edge_litert/libLiteRt.dylib" "ai_edge_litert/libLiteRtMetalAccelerator.dylib"
  mkdir -p "$RUNTIME"
  cp "$TMP/wheel/ai_edge_litert/"*.dylib "$RUNTIME/"
  rm -rf "$TMP"
fi
cp "$RUNTIME/libLiteRt.dylib" "$RUNTIME/libLiteRtMetalAccelerator.dylib" Frameworks/
ls -la Resources Frameworks
