"""d1-2B picture tower, export form: the vision tower (14 layers) and the projector of LiquidAI/d1-2B on a fixed canvas
of GH x GW patches, the picture's own gh x gw patch grid placed at the top-left, the rest masked out exactly.

    D1Tower(Wv, Wp, vcfg, GH, GW)(patches [1, GH*GW, 768], valid [1, GH*GW, 1]) -> {"features": [1, GH*GW/4, 2048]}

`patches`: the processor's pixel values (rescale 1/255, normalize mean 0.5 std 0.5) of each 16x16 patch, the first of
its two identical temporal copies, channel-major (c, 16, 16) = the processor's patch vector [c*512 + t*256 + i*16 + j]
at t = 0; canvas raster order (row r, column c); cells outside the picture zero. `valid`: 1 on the picture's patches.
`features`: the projector's tokens in block-raster order of the canvas; block (i, j), i < gh/2, j < gw/2, row-major,
is the provider's token order for the picture.

Exact by construction against the provider (modeling_lfm3_vl.py of the snapshot):
- patchifier Linear(1536 -> 1024) on two equal temporal copies = Linear(768 -> 1024) with the two halves of each
  channel's weights summed (`patch_weight`);
- every layer's residual output is multiplied by `valid`, so the cells outside the picture stay exactly 0: the
  depthwise 3x3 convolution then reads zeros past the picture's right and bottom edges, as the provider's zero padding
  does, and the canvas edge is zero padding too;
- attention: keys outside the picture get -30000 (exp underflows to 0 in fp32); queries outside it are discarded;
- the provider orders patches by 2x2 merge block; attention, norms and MLPs are per token or permutation-equivariant,
  the convolution runs on the raster grid (the provider converts to raster for it), RoPE uses each patch's own (h, w)
  (transformers.vision_utils.get_vision_position_ids), so the whole tower runs in raster order with constant tables;
- the projector's 2x2 merge = the four strided slices (r0c0, r0c1, r1c0, r1c1) concatenated on channels = the
  provider's reshape(-1, 4 * 1024) of merge-block order (lfm3vl_work/tools/lfm3_vision.py, validated there).
GQA 16 / 8: query heads 2j, 2j+1 share kv head j; the two heads of a group are stacked on the query axis
([1, 8, 2N, 64] against [1, 8, N, 64]), no repeat. `qblock` splits the queries (scores memory on long canvases).
"""
from __future__ import annotations

import json
import os

import torch
import torch.nn.functional as F
from torch import nn

from d1_graph import NEG, RMSNorm, SNAP_DEFAULT, linear

TOWER = "model.vision_tower."
PROJ = "model.multi_modal_projector."


def read_vcfg(snap: str = SNAP_DEFAULT) -> dict:
    c = json.load(open(os.path.join(snap, "config.json")))
    v = dict(c["vision_config"])
    assert v["rope_parameters"]["rope_type"] == "axial" and v["temporal_patch_size"] == 2
    assert c["downsample_factor"] == v["spatial_merge_size"] == 2 and c["projector_hidden_act"] == "gelu"
    v.update(rope_theta=float(v["rope_parameters"]["rope_theta"]), projector_hidden_size=c["projector_hidden_size"],
             projector_layer_norm_eps=c["projector_layer_norm_eps"], projector_bias=bool(c["projector_bias"]),
             text_hidden_size=c["text_config"]["hidden_size"])
    return v


def load_vision_weights(snap: str = SNAP_DEFAULT, dtype=torch.float32) -> dict:
    from safetensors import safe_open

    out = {}
    with safe_open(os.path.join(snap, "model.safetensors"), framework="pt") as f:
        for k in f.keys():
            if k.startswith((TOWER, PROJ)):
                out[k] = f.get_tensor(k).to(dtype)
    return out


def patch_weight(w: torch.Tensor, vcfg: dict) -> torch.Tensor:
    """[1024, 3*2*16*16] -> [1024, 3*16*16]: the two temporal halves of each channel summed."""
    p, t, c = vcfg["patch_size"], vcfg["temporal_patch_size"], vcfg["in_channels"]
    return w.reshape(w.shape[0], c, t, p * p).sum(2).reshape(w.shape[0], c * p * p)


def rope_tables(vcfg: dict, GH: int, GW: int, dtype=torch.float32):
    """cos / sin [1, N, 1, head_dim] in canvas raster order: Lfm3VlVisionRotaryEmbedding on (h, w) per cell (fp32 as
    the provider computes it; fp64 the same functions in fp64)."""
    hd = vcfg["hidden_size"] // vcfg["num_attention_heads"]
    sd = hd // 2
    inv = 1.0 / (vcfg["rope_theta"] ** (torch.arange(0, sd, 2, dtype=torch.float) / sd))
    hh, ww = torch.meshgrid(torch.arange(GH), torch.arange(GW), indexing="ij")
    pos = torch.stack([hh.flatten(), ww.flatten()], dim=-1)          # [N, 2]
    if dtype == torch.float32:
        freqs = pos[..., None].float() * inv.float()
    else:
        freqs = pos[..., None].to(dtype) * inv.to(dtype)
    out = []
    for f in (freqs.cos(), freqs.sin()):
        fhw = torch.cat([f[:, 0], f[:, 1]], dim=-1)
        out.append(torch.cat([fhw, fhw], dim=-1).reshape(1, GH * GW, 1, hd).to(dtype))
    return out


class Mlp(nn.Module):
    def __init__(self, W, p):
        super().__init__()
        self.w1, self.w3, self.w2 = linear(W[p + "w1.weight"]), linear(W[p + "w3.weight"]), linear(W[p + "w2.weight"])

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class ConvMixer(nn.Module):
    def __init__(self, W, p, GH, GW, d):
        super().__init__()
        self.in_proj, self.out_proj = linear(W[p + "in_proj.weight"]), linear(W[p + "out_proj.weight"])
        w = W[p + "spatial_conv.weight"]                              # [d, 1, 3, 3]
        self.spatial_conv = nn.Conv2d(d, d, kernel_size=w.shape[-1], padding=w.shape[-1] // 2, groups=d, bias=False)
        self.spatial_conv.weight = nn.Parameter(w, requires_grad=False)
        self.GH, self.GW, self.d = GH, GW, d

    def forward(self, x):
        b, c, xx = self.in_proj(x).chunk(3, dim=-1)
        g = (b * xx).reshape(1, self.GH, self.GW, self.d).permute(0, 3, 1, 2)
        y = self.spatial_conv(g).permute(0, 2, 3, 1).reshape(1, self.GH * self.GW, self.d)
        return self.out_proj(c * y)


class Attention(nn.Module):
    def __init__(self, W, p, vcfg, N, qblock):
        super().__init__()
        d = vcfg["hidden_size"]
        self.h, self.kvh = vcfg["num_attention_heads"], vcfg["num_key_value_heads"]
        self.hd, self.rep, self.N, self.qblock = d // self.h, self.h // self.kvh, N, qblock
        self.q_proj, self.k_proj = linear(W[p + "q_proj.weight"]), linear(W[p + "k_proj.weight"])
        self.v_proj, self.out_proj = linear(W[p + "v_proj.weight"]), linear(W[p + "out_proj.weight"])
        self.q_layernorm = RMSNorm(W[p + "q_layernorm.weight"], vcfg["norm_eps"])
        self.k_layernorm = RMSNorm(W[p + "k_layernorm.weight"], vcfg["norm_eps"])
        self.scale = self.hd ** -0.5

    def _rope(self, x, cos, sin):
        half = self.hd // 2
        return x * cos + torch.cat([-x[..., half:], x[..., :half]], dim=-1) * sin

    def forward(self, x, cos, sin, key_bias):
        N, hd = self.N, self.hd
        q = self._rope(self.q_layernorm(self.q_proj(x).reshape(1, N, self.h, hd)), cos, sin)
        k = self._rope(self.k_layernorm(self.k_proj(x).reshape(1, N, self.kvh, hd)), cos, sin)
        v = self.v_proj(x).reshape(1, N, self.kvh, hd)
        q = q.permute(0, 2, 1, 3)                                       # [1, 16, N, 64]
        k, v = k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)             # [1, 8, N, 64]
        qb = self.qblock if self.qblock and self.qblock < N else N
        outs = []
        for s in range(0, N, qb):
            qq = q[:, :, s:s + qb].reshape(1, self.kvh, self.rep * min(qb, N - s), hd)   # heads 2j, 2j+1 stacked
            a = torch.softmax(torch.matmul(qq, k.transpose(-1, -2)) * self.scale + key_bias, dim=-1)
            outs.append(torch.matmul(a, v).reshape(1, self.h, min(qb, N - s), hd))
        o = outs[0] if len(outs) == 1 else torch.cat(outs, dim=2)
        return self.out_proj(o.permute(0, 2, 1, 3).reshape(1, N, self.h * hd))


class Layer(nn.Module):
    def __init__(self, W, i, vcfg, GH, GW, qblock):
        super().__init__()
        p = f"{TOWER}layers.{i}."
        d = vcfg["hidden_size"]
        self.kind = vcfg["layer_types"][i]
        if self.kind == "full_attention":
            self.self_attn = Attention(W, p + "self_attn.", vcfg, GH * GW, qblock)
        else:
            self.conv = ConvMixer(W, p + "conv.", GH, GW, d)
        self.feed_forward = Mlp(W, p + "feed_forward.")
        self.operator_norm = RMSNorm(W[p + "operator_norm.weight"], vcfg["norm_eps"])
        self.ffn_norm = RMSNorm(W[p + "ffn_norm.weight"], vcfg["norm_eps"])

    def forward(self, x, cos, sin, key_bias, valid):
        h = self.operator_norm(x)
        x = x + (self.self_attn(h, cos, sin, key_bias) if self.kind == "full_attention" else self.conv(h))
        x = x + self.feed_forward(self.ffn_norm(x))
        return x * valid


class D1Tower(nn.Module):
    """patches [1, N, 768] + valid [1, N, 1] -> {"features": [1, N/4, 2048]} on a GH x GW canvas."""

    def __init__(self, W: dict, vcfg: dict, GH: int, GW: int, qblock: int | None = None, dtype=torch.float32):
        super().__init__()
        assert GH % 2 == 0 and GW % 2 == 0
        self.GH, self.GW, self.N, self.d = GH, GW, GH * GW, vcfg["hidden_size"]
        self.patchifier = linear(patch_weight(W[TOWER + "patchifier.weight"], vcfg))
        self.embedding_norm = RMSNorm(W[TOWER + "embedding_norm.weight"], vcfg["norm_eps"])
        self.layers = nn.ModuleList([Layer(W, i, vcfg, GH, GW, qblock) for i in range(vcfg["num_hidden_layers"])])
        self.layer_norm = nn.LayerNorm(self.d, eps=vcfg["projector_layer_norm_eps"], bias=vcfg["projector_bias"])
        self.layer_norm.weight = nn.Parameter(W[PROJ + "layer_norm.weight"], requires_grad=False)
        self.layer_norm.bias = nn.Parameter(W[PROJ + "layer_norm.bias"], requires_grad=False)
        self.linear_1 = nn.Linear(4 * self.d, vcfg["projector_hidden_size"], bias=True)
        self.linear_1.weight = nn.Parameter(W[PROJ + "linear_1.weight"], requires_grad=False)
        self.linear_1.bias = nn.Parameter(W[PROJ + "linear_1.bias"], requires_grad=False)
        self.linear_2 = nn.Linear(vcfg["projector_hidden_size"], vcfg["text_hidden_size"], bias=True)
        self.linear_2.weight = nn.Parameter(W[PROJ + "linear_2.weight"], requires_grad=False)
        self.linear_2.bias = nn.Parameter(W[PROJ + "linear_2.bias"], requires_grad=False)
        cos, sin = rope_tables(vcfg, GH, GW, dtype)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.to(dtype)

    def forward(self, patches, valid):
        key_bias = ((valid - 1.0) * NEG).reshape(1, 1, 1, self.N)
        x = self.embedding_norm(self.patchifier(patches)) * valid
        for layer in self.layers:
            x = layer(x, self.cos, self.sin, key_bias, valid)
        x = self.layer_norm(x).reshape(1, self.GH, self.GW, self.d)
        m = torch.cat([x[:, 0::2, 0::2, :], x[:, 0::2, 1::2, :], x[:, 1::2, 0::2, :], x[:, 1::2, 1::2, :]], dim=-1)
        m = m.reshape(1, self.N // 4, 4 * self.d)
        return {"features": self.linear_2(F.gelu(self.linear_1(m)))}


from d1_layout import from_canvas, to_canvas  # noqa: E402,F401  (numpy only, shared with the host)
