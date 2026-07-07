#!/usr/bin/env bash
# moe_kernel_probe.sh — Detect MoE (and its capabilities) in the shipped LiteRT-LM GPU/Metal delegates.
#
# Why: the ML Drift GPU kernel source is internal (third_party/odml/litert/ml_drift/...),
# so the only way to know what the shipped runtime supports is to symbol-scan the prebuilt
# delegate binaries that litert-lm vendors under prebuilt/ (git-LFS).
#
# As of litert-lm v0.14.0 (delegate rebuilt 2026-06-29) the `moe` kernel IS present on
# android GPU/OpenCL and iOS Metal, but it is GELU-only / int8-symmetric-or-fp32 / no-int4 /
# renormalized_top_weights=true. This probe re-checks the LATEST delegate so you learn the
# moment SiLU or int4 support lands (the real unlock for OLMoE / Qwen3-MoE / Granite-MoE).
#
# Usage: scripts/moe_kernel_probe.sh [path-to-litert-lm-clone]   (default: ~/code/litert-lm)
# Non-destructive: fetches LFS objects into the repo's LFS cache; never touches the working tree.

set -euo pipefail
REPO="${1:-$HOME/code/litert-lm}"
REF="refs/remotes/origin/main"

DELEGATES=(
  "prebuilt/android_arm64/libLiteRtGpuAccelerator.so"
  "prebuilt/android_arm64/libLiteRtOpenClAccelerator.so"
  "prebuilt/ios_arm64/libLiteRtMetalAccelerator.dylib"
)

cd "$REPO"
echo "== MoE kernel probe =="
echo "repo: $REPO"
git fetch --quiet origin 2>/dev/null || echo "  (warning: git fetch failed — scanning cached refs)"
echo "litert-lm origin/main: $(git log -1 --format='%h %ci' origin/main | cut -d' ' -f1-3)"
echo "bundled LITERT_REF:   $(git show origin/main:WORKSPACE 2>/dev/null | grep -m1 'LITERT_REF =' | grep -oE '[a-f0-9]{40}' | cut -c1-12)"
echo

overall_moe=0
for path in "${DELEGATES[@]}"; do
  oid=$(git cat-file -p "origin/main:$path" 2>/dev/null | sed -n 's/^oid sha256://p')
  [ -z "$oid" ] && { echo "  $path : (not tracked on origin/main)"; continue; }
  git lfs fetch origin "$REF" --include="$path" >/dev/null 2>&1 || true
  obj=".git/lfs/objects/${oid:0:2}/${oid:2:2}/$oid"
  [ -f "$obj" ] || { echo "  $path : (LFS object $oid not available)"; continue; }

  dump=$(strings -a "$obj")
  name=$(basename "$path")
  if ! grep -qiE "ml_drift.*Moe|moe_experts|moe expects" <<<"$dump"; then
    echo "  $name : NO moe kernel"
    continue
  fi
  overall_moe=1
  # Capability probe — scope to moe/expert-context lines ONLY (else e.g. hard_swish, a
  # different op, false-triggers the SiLU check). The authoritative signal is the kernel's
  # own guard string `moe only supports activation='...'`.
  moe_lines=$(grep -iE "moe|expert" <<<"$dump")
  actline=$(grep -iE "moe only supports activation=" <<<"$moe_lines" | head -1)
  case "$actline" in
    *"'gelu'"*) act="gelu-only" ;;
    "")         act="unknown (guard string gone — inspect!)" ;;
    *)          act="** NON-GELU: ${actline#*supports } (NEW!) **" ;;
  esac
  int4="no";         grep -qiE "moe.*int4|int4.*(expert|weight)" <<<"$moe_lines" && int4="** int4 (NEW!) **"
  renorm="required"; grep -qiE "renormalized_top_weights=(false| ?optional)|renorm.*optional" <<<"$moe_lines" && renorm="optional (NEW!)"
  echo "  $name : MoE PRESENT  | activation=$act | int4=$int4 | renorm_topk=$renorm"
done

echo
if [ "$overall_moe" = 1 ]; then
  echo ">> moe kernel is SHIPPED. Watch the activation/int4 flags above:"
  echo "   a 'NEW!' on activation=SiLU or int4 means real MoE targets (OLMoE/Qwen3-MoE/Granite) become shippable."
else
  echo ">> moe kernel NOT present in the latest delegates."
fi
