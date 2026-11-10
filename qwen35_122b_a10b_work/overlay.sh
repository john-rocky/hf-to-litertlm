#!/bin/bash
# overlay.sh LITERT_CHECKOUT — prepares a LiteRT checkout for the Qwen3.5-MoE writer:
#   1. applies, in this order (each on top of the one before), unless already applied:
#      patches/blockwise_named_scales.diff  the Tensor API's BlockwiseQuantization with a block_shape and an existing
#                                           scales tensor = the form ai-edge-quantizer writes for MoE experts (LiteRT
#                                           pull request #11309); the same file as ../qwen3_30b_a3b_work/patches/
#      patches/custom_opcode_by_name.diff   ModelFactory::Build keyed operator codes by the builtin code only, so every
#                                           CUSTOM op of a model took the first custom op's code (the moe ops came out
#                                           as gated_delta_update); this keys CUSTOM codes by (builtin, custom code), as
#                                           LiteRT main at d3b84ea9 does
#      patches/save_at_offset.diff          ModelFactory::SaveAt(path, offset): the model written into a file from a
#                                           byte offset on, the same bytes (buffer offsets stay relative to the model's
#                                           start), so the writer's --out_offset puts the graph straight into its
#                                           .litertlm section (make_bundle.py --plan-offset / --head-only)
#   2. links graph/ into the checkout as tensor/examples/qwen35moe.
# Tested on LiteRT dcb657e8 (the LiteRT that LiteRT-LM 0.19.0 pins). Then, in the checkout:
#   bazel build -c opt //tensor/examples/qwen35moe:qwen35moe_writer
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
[ $# -eq 1 ] || { echo "usage: $0 LITERT_CHECKOUT"; exit 2; }
LITERT=$(cd "$1" && pwd)
cd "$LITERT"
for P in blockwise_named_scales.diff custom_opcode_by_name.diff save_at_offset.diff; do
  if git apply --check "$HERE/patches/$P" 2>/dev/null; then
    git apply "$HERE/patches/$P" && echo "patch applied: $P"
  elif git apply --reverse --check "$HERE/patches/$P" 2>/dev/null; then
    echo "patch already applied: $P"
  else
    echo "patch does not apply to $LITERT: $P"; exit 1
  fi
done
DST=$LITERT/tensor/examples/qwen35moe
if [ -L "$DST" ] && [ "$(readlink "$DST")" = "$HERE/graph" ]; then echo "already linked: $DST"; exit 0; fi
[ -e "$DST" ] && { echo "exists and is not this link: $DST"; exit 1; }
ln -s "$HERE/graph" "$DST" && echo "linked: $DST -> $HERE/graph"
