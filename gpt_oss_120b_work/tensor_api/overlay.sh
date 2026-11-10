#!/bin/bash
# overlay.sh LITERT_CHECKOUT — prepares a LiteRT checkout (dcb657e8, the LiteRT that LiteRT-LM 0.19.1 pins) for the
# gpt-oss writer: applies blockwise_named_scales.diff (BlockwiseQuantization with block_shape and an existing scales
# tensor, the form ai-edge-quantizer writes for MoE experts) unless already applied, and links gptoss/ in as
# tensor/examples/gptoss.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
LITERT=${1:?usage: overlay.sh LITERT_CHECKOUT}
cd "$LITERT"
if git apply --check "$HERE/blockwise_named_scales.diff" 2>/dev/null; then
  git apply "$HERE/blockwise_named_scales.diff" && echo "patch applied"
elif git apply --reverse --check "$HERE/blockwise_named_scales.diff" 2>/dev/null; then
  echo "patch already applied"
else
  echo "patch does not apply to $LITERT"; exit 1
fi
DST=$LITERT/tensor/examples/gptoss
if [ -L "$DST" ] && [ "$(readlink "$DST")" = "$HERE/gptoss" ]; then echo "already linked: $DST"; exit 0; fi
[ -e "$DST" ] && { echo "exists and is not our link: $DST"; exit 1; }
ln -s "$HERE/gptoss" "$DST" && echo "linked: $DST -> $HERE/gptoss"
