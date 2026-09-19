#!/usr/bin/env python3
"""One-off: transpose linear_attn.conv1d.weight from the mlx layout [out, k, 1] to torch [out, 1, k] in an
already-written hf_bf16 checkpoint (dequant_mlx_pack.py now does this itself)."""
import json, os, sys
import torch
from safetensors.torch import load_file, save_file
d = sys.argv[1]
idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
shards = sorted({v for k, v in idx.items() if k.endswith("conv1d.weight")})
for sh in shards:
    p = os.path.join(d, sh); t = load_file(p); n = 0
    for k in list(t):
        if k.endswith("conv1d.weight") and t[k].shape[-1] == 1:
            t[k] = t[k].transpose(1, 2).contiguous(); n += 1
    save_file(t, p, metadata={"format": "pt"}); print(sh, "fixed", n)
