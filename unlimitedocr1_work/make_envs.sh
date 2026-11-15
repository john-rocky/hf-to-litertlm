#!/bin/bash
# make_envs.sh DIR - the three Python environments of scripts/ship_unlimited_ocr.sh, each from the frozen requirements
# files in this folder, then `pip check`:
#   DIR/export       Python 3.14: requirements-export.txt, then requirements-export-nodeps.txt (litert-torch) with
#                    --no-deps; PY_EXPORT. pip check may list litert-torch 0.9.3's own version bounds (the published
#                    file was exported with newer torch, litert-converter, ai-edge-quantizer and ai-edge-litert; see
#                    requirements-export-nodeps.txt) and nothing else.
#   DIR/bundle       Python 3.14: requirements-bundle.txt; PY_BUNDLE.
#   DIR/externalize  Python 3.12: requirements-externalize.txt; PY_XR.
# PYTHON314 / PYTHON312 = the interpreters (default python3.14 / python3.12; the published file was built with 3.14.6
# and 3.12.11). An environment that installed completely is marked DIR/<name>/.complete and is not installed again.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
[ $# -eq 1 ] || { echo "usage: $0 DIR"; exit 2; }
mkdir -p "$1"; D=$(cd "$1" && pwd)
make_env() {   # name python requirements [requirements installed with --no-deps]
  local name=$1 py=$2 req=$3 nodeps=${4:-} v=$D/$1 out other
  if [ ! -e "$v/.complete" ]; then
    [ ! -e "$v" ] || { echo "$v: an environment that did not install completely; remove it and run again"; exit 1; }
    "$py" -m venv "$v"
    "$v/bin/python" -m pip install -q --disable-pip-version-check -r "$req"
    [ -z "$nodeps" ] || "$v/bin/python" -m pip install -q --disable-pip-version-check --no-deps -r "$nodeps"
    : > "$v/.complete"
  fi
  if out=$("$v/bin/python" -m pip check --disable-pip-version-check 2>&1); then
    echo "$name: pip check: $out"
  else
    other=$(printf '%s\n' "$out" | grep -v '^litert-torch 0\.9\.3 has requirement ' || true)
    if [ "$name" != export ] || [ -n "$other" ]; then
      printf '%s\n' "$out"; echo "$name: pip check failed"; exit 1
    fi
    printf '%s\n' "$out" | sed "s/^/$name: pip check, litert-torch's own bounds (expected): /"
  fi
  "$v/bin/python" -c 'import sys; print(sys.argv[1], "Python", sys.version.split()[0])' "$name"
}
make_env export "${PYTHON314:-python3.14}" "$HERE/requirements-export.txt" "$HERE/requirements-export-nodeps.txt"
make_env bundle "${PYTHON314:-python3.14}" "$HERE/requirements-bundle.txt"
make_env externalize "${PYTHON312:-python3.12}" "$HERE/requirements-externalize.txt"
echo "export PY_EXPORT=$D/export/bin/python PY_BUNDLE=$D/bundle/bin/python PY_XR=$D/externalize/bin/python"
