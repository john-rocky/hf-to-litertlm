#!/bin/bash
# overlay.sh LITERT_CHECKOUT — prepares a LiteRT checkout for the Qwen3-MoE writer:
#   1. applies patches/blockwise_named_scales.diff (the Tensor API's BlockwiseQuantization with a block_shape and an
#      existing scales tensor = the form ai-edge-quantizer writes for MoE experts, LiteRT #11309) unless it is applied;
#   2. links graph/ into the checkout as tensor/examples/qwen3moe.
# Tested on LiteRT dcb657e8 (the LiteRT that LiteRT-LM 0.19.0 pins). Then, in the checkout:
#   bazel build -c opt //tensor/examples/qwen3moe:qwen3moe_writer
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
[ $# -eq 1 ] || { echo "usage: $0 LITERT_CHECKOUT"; exit 2; }
LITERT=$(cd "$1" && pwd)
cd "$LITERT"
if git apply --check "$HERE/patches/blockwise_named_scales.diff" 2>/dev/null; then
  git apply "$HERE/patches/blockwise_named_scales.diff" && echo "patch applied"
elif git apply --reverse --check "$HERE/patches/blockwise_named_scales.diff" 2>/dev/null; then
  echo "patch already applied"
else
  echo "patch does not apply to $LITERT"; exit 1
fi
DST=$LITERT/tensor/examples/qwen3moe
if [ -L "$DST" ] && [ "$(readlink "$DST")" = "$HERE/graph" ]; then echo "already linked: $DST"; exit 0; fi
[ -e "$DST" ] && { echo "exists and is not this link: $DST"; exit 1; }
ln -s "$HERE/graph" "$DST" && echo "linked: $DST -> $HERE/graph"
