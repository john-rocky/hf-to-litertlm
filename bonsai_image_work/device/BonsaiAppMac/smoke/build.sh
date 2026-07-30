#!/bin/zsh
# Build the macOS smoke test against the prebuilt macos_arm64 LiteRT pair.
# Headers come from the LiteRT checkout (C API is append-only, so the 7/16 pin
# is ABI-compatible with the main-branch prebuilts).
set -e
cd "$(dirname "$0")"
LITERT_SRC="${LITERT_SRC:-$HOME/code/litert-tensor/LiteRT}"
PREBUILT="${PREBUILT:-$HOME/models/litert-prebuilt/macos_arm64}"
clang++ -std=c++17 -O2 bonsai_smoke.mm \
  -I "$LITERT_SRC" -I shim \
  -L "$PREBUILT" -lLiteRt \
  -Wl,-rpath,"$PREBUILT" \
  -framework Foundation -framework CoreGraphics -framework ImageIO \
  -o bonsai_smoke
echo "built ./bonsai_smoke  (usage: ./bonsai_smoke [all|text|dit|vae])"
