#!/bin/zsh
# Build the macOS smoke test. Headers come from ../third_party/LiteRT —
# fetched at the runtime's release tag by ../prep_resources.sh (run it first).
set -e
cd "$(dirname "$0")"
LITERT_SRC="${LITERT_SRC:-../third_party/LiteRT}"
PREBUILT="${PREBUILT:-$HOME/models/litert-prebuilt/ai_edge_litert_216}"
clang++ -std=c++17 -O2 bonsai_smoke.mm \
  -I "$LITERT_SRC" \
  -L "$PREBUILT" -lLiteRt \
  -Wl,-rpath,"$PREBUILT" \
  -framework Foundation -framework CoreGraphics -framework ImageIO \
  -o bonsai_smoke
echo "built ./bonsai_smoke  (usage: ./bonsai_smoke [all|text|dit|vae])"
