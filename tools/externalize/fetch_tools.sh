#!/bin/bash
# fetch_tools.sh — fetch LiteRT's two external-weight tools at a pinned commit and add the import shim they need.
#
#   bash tools/externalize/fetch_tools.sh [DIR]        default DIR = ~/.cache/litert-externalize/<commit>
#
# Writes DIR/litert/tools/externalize_tflite_flatbuffer.py and externalize_litertlm_sections.py (Apache-2.0, from
# github.com/google-ai-edge/LiteRT at the commit below, sha256 checked), the three empty package files, and
# DIR/litert/python/schema_py_generated.py = `from ai_edge_litert.schema_py_generated import *`: the tools import
# LiteRT's in-repo flatc output, and the ai-edge-litert wheel ships the same generated module, so no flatc is needed.
# Then checks that the Python in XR_PY (default python3) imports it with the ExternalBuffer types, and prints the
# line to export for externalize_repo.sh.
set -euo pipefail
COMMIT=c5c0d28f27abddec81bcc14bbd05bdb84eb5c556
PIN_FLATBUFFER=be0630c5d47325f36107247ac5e7c65380a38757b7a122df1534015f99fc6a9b
PIN_SECTIONS=0aa2b9e1eba0427a127a1b22916fb98ae440bd9a8376dc50ed15d65261b6c367
DIR=${1:-$HOME/.cache/litert-externalize/$COMMIT}
PY=${XR_PY:-python3}
RAW=https://raw.githubusercontent.com/google-ai-edge/LiteRT/$COMMIT/litert/tools
sha() { if command -v shasum > /dev/null; then shasum -a 256 "$1" | cut -d' ' -f1; else sha256sum "$1" | cut -d' ' -f1; fi; }
mkdir -p "$DIR/litert/tools" "$DIR/litert/python"
for f in externalize_tflite_flatbuffer.py externalize_litertlm_sections.py; do
  curl -fsSL "$RAW/$f" -o "$DIR/litert/tools/$f"
done
[ "$(sha "$DIR/litert/tools/externalize_tflite_flatbuffer.py")" = "$PIN_FLATBUFFER" ] \
  || { echo "sha256 differs from the pin: externalize_tflite_flatbuffer.py"; exit 1; }
[ "$(sha "$DIR/litert/tools/externalize_litertlm_sections.py")" = "$PIN_SECTIONS" ] \
  || { echo "sha256 differs from the pin: externalize_litertlm_sections.py"; exit 1; }
: > "$DIR/litert/__init__.py"
: > "$DIR/litert/tools/__init__.py"
: > "$DIR/litert/python/__init__.py"
echo 'from ai_edge_litert.schema_py_generated import *' > "$DIR/litert/python/schema_py_generated.py"
echo "$COMMIT" > "$DIR/LITERT_COMMIT"
"$PY" -I -c 'import sys; sys.path.insert(0, sys.argv[1]); from litert.python import schema_py_generated as s; s.ExternalBufferT; s.ExternalBufferGroupT; s.ModelT().externalBuffers; s.TensorT().externalBuffer' "$DIR" \
  || { echo "$PY cannot import the ExternalBuffer types: pip install ai-edge-litert (2.1.6 to 2.4.0.dev20261011 carry them)"; exit 1; }
echo "tools ready in $DIR (LiteRT $COMMIT)"
echo "export XR_TOOLS_PATH=$DIR"
