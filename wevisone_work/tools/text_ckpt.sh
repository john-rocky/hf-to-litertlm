#!/usr/bin/env bash
# The text tower as a plain Qwen3ForCausalLM checkpoint (fp32) with this repository's tools/vlm_text_ckpt/make_text_ckpt.py,
# the rope keys of the config reduced to Qwen3's form; PROVE=1 adds the 9-prompt logits proof (tools/prove_text_ckpt.py,
# loads the full VLM and the derived model in fp32).
#   PY_REF             Python with torch 2.11.0, transformers 5.12.1, safetensors (the reference environment)
#   WEVISONE_SNAPSHOT  the tencent/WeVisOne-VL-2B snapshot at 3dad9ce8 (default: the Hugging Face cache path)
set -euo pipefail
K="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY_REF:?set PY_REF}"
SNAP="${WEVISONE_SNAPSHOT:-$HOME/.cache/huggingface/hub/models--tencent--WeVisOne-VL-2B/snapshots/3dad9ce8bcb6fbe0aea945bf8860bd2cfc3663bf}"
TOOL="$K/../tools/vlm_text_ckpt/make_text_ckpt.py"
OUT="$K/out/text_ckpt"
mkdir -p "$K/ref"
if [ ! -f "$OUT/config.json" ]; then
  rm -rf "$OUT"
  HF_HUB_OFFLINE=1 "$PY" -I "$TOOL" "$SNAP" "$OUT" --prefix model.language_model. --model-type qwen3 \
    --architectures Qwen3ForCausalLM --drop model.visual. --report "$K/ref/text_ckpt_report.json"
  # Qwen3's rope: keep rope_theta / rope_type, drop the M-RoPE keys (mrope_section, mrope_interleaved).
  "$PY" -I - "$OUT/config.json" <<'PY'
import json, sys
path = sys.argv[1]
cfg = json.load(open(path))
rope = cfg.get("rope_parameters") or {}
dropped = sorted(k for k in rope if k.startswith("mrope"))
cfg["rope_parameters"] = {k: v for k, v in rope.items() if not k.startswith("mrope")}
json.dump(cfg, open(path, "w"), indent=2)
open(path, "a").write("\n")
print("config.json rope_parameters ->", cfg["rope_parameters"], "dropped", dropped)
PY
fi
if [ "${PROVE:-0}" = "1" ]; then
  WEVISONE_SNAPSHOT="$SNAP" "$PY" -I "$K/tools/prove_text_ckpt.py"
fi
