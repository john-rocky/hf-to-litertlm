#!/bin/bash
# make_envs.sh DIR - the three Python environments of scripts/ship_unlimited_ocr.sh, each from its frozen requirements
# file in this folder, then `pip check`:
#   DIR/export        Python 3.14 + requirements-export.txt        (PY_EXPORT: litert-torch 0.9.3, litert-converter 0.4.0,
#                                                                   ai-edge-litert 2.2.0, ai-edge-quantizer 0.9.0, torch 2.13.0)
#   DIR/bundle        Python 3.14 + requirements-bundle.txt        (PY_BUNDLE: litert-lm-builder 0.19.0, litert-lm 0.19.0)
#   DIR/externalize   Python 3.12 + requirements-externalize.txt   (PY_XR: ai-edge-litert 2.3.0, litert-lm 0.19.1,
#                                                                   litert-lm-builder 0.19.1; tools/externalize)
# PYTHON314 / PYTHON312 = the interpreters to make them from (default python3.14 / python3.12 on PATH; the published
# file was built with 3.14.6 and 3.12.11). An environment that already exists is checked, not made again.
# Prints the three PY_* lines to export.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
[ $# -eq 1 ] || { echo "usage: $0 DIR"; exit 2; }
mkdir -p "$1"; D=$(cd "$1" && pwd)
make_env() {   # name python requirements
  local name=$1 py=$2 req=$3 v=$D/$1
  if [ ! -x "$v/bin/python" ]; then
    "$py" -m venv "$v"
    "$v/bin/python" -m pip install -q --disable-pip-version-check -r "$req"
  fi
  "$v/bin/python" -m pip check --disable-pip-version-check
  "$v/bin/python" -c 'import sys; print(sys.argv[1], "Python", sys.version.split()[0])' "$name"
}
make_env export "${PYTHON314:-python3.14}" "$HERE/requirements-export.txt"
make_env bundle "${PYTHON314:-python3.14}" "$HERE/requirements-bundle.txt"
make_env externalize "${PYTHON312:-python3.12}" "$HERE/requirements-externalize.txt"
echo "export PY_EXPORT=$D/export/bin/python PY_BUNDLE=$D/bundle/bin/python PY_XR=$D/externalize/bin/python"
