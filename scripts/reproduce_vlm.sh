#!/bin/bash
# One-command reproduction of the shipped fast_vlm VLMs (image+text -> .litertlm).
# Each key runs its ship_*.sh: download source -> convert vision (encoder+adapter) +
# decoder (int4) -> assemble the fast_vlm .litertlm bundle. Output lands under out/*-bundle/.
#
#   bash scripts/reproduce_vlm.sh <key>     # reproduce one VLM
#   bash scripts/reproduce_vlm.sh --list
#
# See cards/<name>-litert.md for each model's details, and README.md for the recipe.
set -euo pipefail
cd "$(dirname "$0")/.."

case "${1:-}" in
  internvl3-1b)          bash scripts/ship_internvl_1b.sh ;;      # InternViT + Qwen2.5-0.5B
  internvl3.5-1b)        bash scripts/ship_internvl3_5_1b.sh ;;   # InternViT + Qwen3-0.6B
  internvl3.5-2b)        bash scripts/ship_internvl3_5_2b.sh ;;   # InternViT + Qwen3-1.7B
  internvl3.5-4b)        bash scripts/ship_internvl3_5_4b.sh ;;   # InternViT + Qwen3-4B
  llava-onevision-0.5b)  bash scripts/ship_llavaov.sh ;;          # SigLIP + Qwen2-0.5B (730 tokens)
  ovis2.5-2b)            bash scripts/ship_ovis_2b.sh ;;          # static-NaViT + Qwen3-1.7B (structural embed)
  smolvlm2-500m)         bash scripts/ship_smolvlm2.sh ;;         # SigLIP + SmolLM2 (64 tokens)
  smolvlm2-2.2b)         bash scripts/ship_smolvlm2_22b.sh ;;     # SigLIP + SmolLM2-1.7B (81 tokens)
  --list|"")
    echo "VLM keys (fast_vlm single-image):"
    printf '  %s\n' internvl3-1b internvl3.5-1b internvl3.5-2b internvl3.5-4b \
      llava-onevision-0.5b ovis2.5-2b smolvlm2-500m smolvlm2-2.2b
    echo; echo "each -> out/*-bundle/<Model>.litertlm ; details in cards/*-litert.md" ;;
  *) echo "unknown VLM key: '$1' (run --list)"; exit 2 ;;
esac
