#!/bin/bash
# Fetches everything the build needs that is not committed here:
#   Resources/           tokenizer tables + pipeline meta (from the HF staging dir)
#   Frameworks/          the LiteRT runtime dylib pair
#   third_party/LiteRT/  the LiteRT C API headers
# Run this before xcodegen + build. Nothing from the LiteRT codebase is
# committed in this repo — headers are downloaded at the v2.1.6 release tag,
# the SAME release as the runtime binaries.
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
LITERT_TAG="${LITERT_TAG:-v2.1.6}"

mkdir -p Resources Frameworks
cp "$SRC/tokenizer/vocab.json" "$SRC/tokenizer/merges.txt" "$SRC/pipeline_meta.json" Resources/

# --- LiteRT C API headers, pinned to the runtime's release tag -------------
HDR_BASE="https://raw.githubusercontent.com/google-ai-edge/LiteRT/$LITERT_TAG"
HDRS=(
  litert/c/internal/litert_scheduling_info.h
  litert/c/litert_any.h
  litert/c/litert_common.h
  litert/c/litert_compiled_model.h
  litert/c/litert_custom_op_kernel.h
  litert/c/litert_custom_tensor_buffer.h
  litert/c/litert_environment.h
  litert/c/litert_environment_options.h
  litert/c/litert_gl_types.h
  litert/c/litert_layout.h
  litert/c/litert_model.h
  litert/c/litert_model_types.h
  litert/c/litert_op_code.h
  litert/c/litert_opaque_options.h
  litert/c/litert_opencl_types.h
  litert/c/litert_options.h
  litert/c/litert_tensor_buffer.h
  litert/c/litert_tensor_buffer_requirements.h
  litert/c/litert_tensor_buffer_types.h
  litert/c/litert_webgpu_types.h
)
if [ ! -f "third_party/LiteRT/.tag-$LITERT_TAG" ]; then
  echo "Fetching LiteRT C headers at $LITERT_TAG"
  rm -rf third_party/LiteRT
  for h in "${HDRS[@]}"; do
    mkdir -p "third_party/LiteRT/$(dirname "$h")"
    curl -sfL "$HDR_BASE/$h" -o "third_party/LiteRT/$h" &
  done
  wait
  for h in "${HDRS[@]}"; do
    [ -s "third_party/LiteRT/$h" ] || { echo "FAILED to fetch $h"; exit 1; }
  done
  # build_config.h is bazel-generated; the generated file is an empty guard.
  mkdir -p third_party/LiteRT/litert/build_common
  printf '#ifndef LITERT_BUILD_COMMON_BUILD_CONFIG_H_\n#define LITERT_BUILD_COMMON_BUILD_CONFIG_H_\n#endif\n' \
    > third_party/LiteRT/litert/build_common/build_config.h
  touch "third_party/LiteRT/.tag-$LITERT_TAG"
fi

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
