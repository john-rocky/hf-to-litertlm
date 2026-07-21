#!/usr/bin/env python3
"""export_hf wrapper that makes longrope static for torch.export.

transformers' `@dynamic_rope_update` re-selects long vs short inv_freq at
forward time (`if seq_len > original_max_position_embeddings`), which is a
data-dependent branch torch.export aborts on. For MiniCPM4/4.1 the long and
short factor lists are IDENTICAL and factor == 1, so the update is a no-op:
stripping the decorator (init already applied the short factors) is exact.

Usage: same CLI args as `litert-torch export_hf`, e.g.
  python export_static_longrope.py <model_dir> <out_dir> --cache_length=4099 ...
"""
import sys

import torch
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

# forward is wrapped as @torch.no_grad -> @dynamic_rope_update -> forward;
# unwrap fully (both layers), keep no_grad.
_f = LlamaRotaryEmbedding.forward
while hasattr(_f, "__wrapped__"):
    _f = _f.__wrapped__
assert "dynamic_rope" not in (_f.__module__ or "")
LlamaRotaryEmbedding.forward = torch.no_grad()(_f)

from litert_torch.cli import main

sys.argv = [sys.argv[0], "export_hf"] + sys.argv[1:]
main()
