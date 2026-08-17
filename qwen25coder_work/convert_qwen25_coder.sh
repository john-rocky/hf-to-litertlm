#!/usr/bin/env bash
# Qwen/Qwen2.5-Coder-1.5B-Instruct -> .litertlm   (2026-08-17)
#
# A 1.5B code model at 1.12 GB. The size is the point: decode on this runtime is
# memory-bandwidth-bound, so halving the weights is the most direct way to make a model
# feel fast on a phone — and code assistance, which returns short outputs constantly, is
# the use case where decode rate IS the experience.
#
# Two things here are not optional:
#
#   EXTERNALIZE_EMBEDDER=1 — this model ties its embedding and lm_head, so a recipe asking
#   for int4 linears and an int8 embedder describes one tensor two ways. ai-edge-quantizer
#   resolves that by copying the 151936x1536 table once per signature: seven copies,
#   1.63 GB, 65% of the file, and no error anywhere. Externalising it gives 1.12 GB.
#   Check any fresh bundle with `python check_bundle_sanity.py <file> --params 1543714304`.
#
#   qwen25_coder_simple.jinja — the upstream chat template inserts a default system prompt
#   ("You are Qwen, created by Alibaba Cloud...") whenever the caller sends no system
#   message. A plain ChatML template drops it and puts the model in a state it was not
#   tuned in, silently, on every default request. This one bakes it in and renders
#   byte-identical to upstream for a single turn.
set -euo pipefail
cd "$(dirname "$0")/.."

python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-Coder-1.5B-Instruct', local_dir='src_models/qwen25-coder-1.5b')"

CACHE=4096 PREFILL=1024,256,64,16,4,1 EXTERNALIZE_EMBEDDER=1 \
  python scripts/export_simple_template.py \
    src_models/qwen25-coder-1.5b out_qwen25_coder \
    qwen25coder_work/qwen25_coder_simple.jinja BOCTAV4

python qwen25coder_work/check_bundle_sanity.py out_qwen25_coder/model.litertlm --params 1543714304
