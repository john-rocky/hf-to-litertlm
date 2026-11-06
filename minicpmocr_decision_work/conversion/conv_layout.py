#!/usr/bin/env python3
"""Kernel form token `cv`: the GatedDeltaNet's causal depthwise conv in the layout the conv runs in. The stock forward takes [1, L, C] to [1, C, L] for nn.Conv1d, and the converter goes on
to [1, 1, L + 6, C] and back: four TRANSPOSE, a RESHAPE and a SLICE of the largest activation of the layer (C = 6144), per layer. Here: [1, L, C] -> [1, 1, L, C] -> left pad K - 1 ->
conv -> [1, L, C]. The right pad of the stock conv only feeds the K - 1 outputs it cuts off, so the first L outputs are the same numbers.
The NCHW permutes around F.conv2d cancel against the converter's own NHWC transposes in its layout pass."""
import types

import torch
import torch.nn.functional as F_torch

import patch_q35 as P


def _conv_nhwc(self, x):
    """x [1, L, C]: the same first L outputs as self.conv1d(x.transpose(1, 2))[:, :, :L].transpose(1, 2)."""
    k = self._cv_weight.shape[-1]
    y = F_torch.pad(x.unsqueeze(1), (0, 0, k - 1, 0))             # [1, 1, L + k - 1, C]
    y = y.permute(0, 3, 1, 2)                                     # NCHW [1, C, 1, L + k - 1]
    y = F_torch.conv2d(y, self._cv_weight, self.conv1d.bias, groups=self.conv1d.groups)
    return y.permute(0, 2, 3, 1).squeeze(1)                       # [1, L, C]


def _forward():
    g = P.PatchedQwen3_5GatedDeltaNet.forward.__globals__
    src = g["_guarded_src"]
    swaps = (("    mixed_qkv = self.in_proj_qkv(hidden_states)\n    mixed_qkv = mixed_qkv.transpose(1, 2)\n", "    mixed_qkv = self.in_proj_qkv(hidden_states)\n"),
             ("mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, : mixed_qkv.shape[-1]])\n", "mixed_qkv = F.silu(self._cv_conv(mixed_qkv))\n"),
             ("    mixed_qkv = mixed_qkv.transpose(1, 2)\n    if valid is not None:\n        mixed_qkv = mixed_qkv * valid[..., None]\n", "    if valid is not None:\n        mixed_qkv = mixed_qkv * valid[..., None]\n"))
    for a, b in swaps:
        assert src.count(a) == 1, f"conv layout anchor not found: {a!r}"
        src = src.replace(a, b)
    keep = g["forward"]
    exec(compile(src, "<qwen3_5_conv_layout>", "exec"), g)  # pylint: disable=exec-used
    fn = g["forward"]
    g["forward"] = keep
    return fn


def apply(mods) -> int:
    fn = _forward()
    for m in mods:
        assert m.causal_conv1d_fn is None, "the fused conv path is in use: this rewrite is for the torch fallback"
        m._cv_weight = torch.nn.Parameter(m.conv1d.weight.unsqueeze(2).detach(), requires_grad=False)   # [C, 1, 1, k]
        m._cv_conv = types.MethodType(_conv_nhwc, m)
        m.forward = types.MethodType(fn, m)
    return len(mods)


def reset(mods) -> None:
    for m in mods:
        if "forward" in m.__dict__:
            del m.forward
            del m._cv_conv
            del m._cv_weight
