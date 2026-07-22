#!/usr/bin/env python3
"""Verify BitCPM-CANN ternary structure + int4-blockwise representability.

BitCPM-CANN stores ternary QAT weights materialized in bf16: per group of 128
input channels, values are {-a, 0, +a} with a = 1/absmean scale at train time
(SteTernaryQuantizer in the released qat-convert.py). If that holds, aeq
MIN_MAX symmetric int4 with BLOCKWISE_32 (blocks subdivide the 128 groups)
maps every linear weight onto {-7, 0, +7} with no rounding decisions — the
ternary levels survive exactly; the only residual error is fp16 rounding of
the per-block scale (absmax/7), which we measure here.

Usage: verify_ternary.py <pytorch_model.bin | model.safetensors>
"""
import sys

import torch

path = sys.argv[1]
if path.endswith(".safetensors"):
    from safetensors.torch import load_file
    sd = load_file(path)
else:
    sd = torch.load(path, map_location="cpu", weights_only=True)
print(f"{len(sd)} tensors, lm_head present: {'lm_head.weight' in sd}")

GROUP = 128
n_lin = 0
worst_scale_rel = 0.0
bad = []
for name, w in sd.items():
    if w.dim() == 2 and "layers" in name and "weight" in name:
        n_lin += 1
        rows = w.float().reshape(-1, GROUP)
        alpha = rows.abs().amax(dim=1, keepdim=True)
        off = ((rows != 0) & (rows.abs() != alpha)).sum().item()

        # aeq minmax sym int4 b32 emulation, fp16 scale storage
        b = w.float().reshape(-1, 32)
        s = (b.abs().amax(dim=1, keepdim=True) / 7.0).half().float()
        szero = s == 0
        s = torch.where(szero, torch.ones_like(s), s)
        q = torch.clamp(torch.round(b / s), -8, 7)
        levels = torch.unique(q).tolist()
        deq = q * s
        rel = ((deq - b).abs().max() / alpha.max()).item()
        worst_scale_rel = max(worst_scale_rel, rel)
        if off or not set(levels) <= {-7.0, 0.0, 7.0}:
            bad.append((name, off, levels[:10], rel))
    elif w.dim() == 2:
        print(f"  non-layer 2D (stays fp / int8-cw path): {name} {list(w.shape)}")

print(f"linears checked (group={GROUP}): {n_lin}")
if bad:
    print("NON-TERNARY / NON-EXACT tensors:")
    for name, off, lv, rel in bad[:20]:
        print(f"  {name}: off-ternary={off} int4 levels={lv} scale-rel-err={rel:.2e}")
else:
    print("ALL layer linears are ternary per 128-group; int4-b32 minmax maps every")
    print("value onto {-7, 0, +7} (no rounding decisions).")
print(f"max relative error from fp16 scale rounding: {worst_scale_rel:.2e}")
