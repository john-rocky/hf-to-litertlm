#!/usr/bin/env bash
# Reproducible convert -> verify -> publish pipeline for litert-community models.
#
# Stage 2 (verify) is a HARD GATE: publish is refused unless the quality report
# shows passed=true. Stage 3 (publish) is DRY-RUN unless you pass --publish, and
# even then only pushes to the repo you name (default: your own HF namespace —
# publish to any namespace you have write access to).
#
#   scripts/pipeline.sh <model.litertlm> <repo_id> <card.md> [--publish]
#
# Stage 1 (convert) is model-specific and run separately — see PIPELINE.md:
#   dense        : python scripts/export_simple_template.py <hf_id> <out> <tmpl> dynamic_wi8_afp32
#   qwen3.5/hybrid: python scripts/convert_qwen35.py <hf_id> <out> dynamic_wi8_afp32
set -euo pipefail

MODEL="${1:?usage: pipeline.sh <model.litertlm> <repo_id> <card.md> [--publish]}"
REPO="${2:?missing repo_id (e.g. mlboydaisuke/SmolLM3-3B-LiteRT)}"
CARD="${3:?missing model card path}"
PUBLISH="${4:-}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
NAME="$(basename "$(dirname "$MODEL")")"
REPORT="$ROOT/reports/${NAME}.json"
mkdir -p "$ROOT/reports"

echo "### Stage 2 — verify (quality gate) ###"
if ! "$PY" "$ROOT/scripts/verify_quality.py" "$MODEL" --name "$NAME" --json "$REPORT"; then
  echo ">>> quality gate FAILED — stopping before publish." >&2
  exit 1
fi

echo
echo "### Stage 3 — publish ###"
ARGS=(--model "$MODEL" --card "$CARD" --report "$REPORT" --repo "$REPO")
if [[ "$PUBLISH" == "--publish" ]]; then
  ARGS+=(--confirm)
fi
"$PY" "$ROOT/scripts/publish_hf.py" "${ARGS[@]}"
