#!/usr/bin/env python3
"""One-off: subtract 1 from the Qwen3_5RMSNorm weights (input/post layernorms, final norm, q_norm, k_norm) in an
already-written hf_bf16 checkpoint. transformers computes x_norm * (1 + w); the mlx pack stores mlx's plain w' = 1 + w.
The gated norm (linear_attn.norm) is plain in both and is left alone. (dequant_mlx_pack.py now does this itself.)"""
import json, os, sys, torch
from safetensors.torch import load_file, save_file
d = sys.argv[1]
SUF = ("input_layernorm.weight", "post_attention_layernorm.weight", "language_model.norm.weight", "q_norm.weight", "k_norm.weight")
idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
for sh in sorted({v for k, v in idx.items() if k.endswith(SUF)}):
    p = os.path.join(d, sh); t = load_file(p); n = 0
    for k in list(t):
        if k.endswith(SUF):
            t[k] = (t[k].float() - 1.0).to(torch.bfloat16); n += 1
    save_file(t, p, metadata={"format": "pt"}); print(sh, "norm weights offset:", n)
