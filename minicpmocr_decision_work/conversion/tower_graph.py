"""TowerRow: the vision tower and merger of openbmb/MiniCPM-OCR (`model.visual`, 24 blocks, hidden 1024, 16 heads x 64,
FFN 4096, out 2048) at a patch-count bucket N, written for the LiteRT export. Plain torch (no transformers), so the same
module runs in the reference environment (the check against the author's module) and in the export environment.

    forward(patches f32 [1, N, 1536], valid f32 [1, N], pos f32 [1, N, 1024], cos f32 [1, N, 64], sin f32 [1, N, 64])
        -> {"rows": f32 [1, N / 4, 2048]}

`valid` and the tables are graph inputs (one graph serves every grid of up to N patches, merge-block-major; the picture's patches first, the rest zero); the
query-block form for N >= 4096.
  key_bias = (valid - 1) * 1e4 as [1, 1, 1, N]: pad keys underflow to exactly 0 in the softmax, real keys add exact 0. Pad
  queries are computed and never read. The author runs one picture at a time (`Process each chunk separately`), so one
  graph call per picture is the author's computation.
  merger: LayerNorm(1024) -> reshape [1, N / 4, 4096] (4 consecutive patches = one merge block) -> fc1 -> GELU (exact erf) ->
  fc2 [-> 2048].
Every activation keeps the leading batch dim (rank 3, attention rank 4); no GATHER / SELECT_V2 / BROADCAST_TO / INT64."""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

HIDDEN, HEADS, HEAD_DIM, FFN, OUT, DEPTH = 1024, 16, 64, 4096, 2048, 24
PATCH_DIM = 3 * 2 * 16 * 16
MERGE_UNIT = 4
EPS = 1e-6
NEG = 1e4
PREFIX = "model.visual."


def load_visual_state(snapshot: str | Path, dtype=torch.float32) -> dict:
    """{key without 'model.visual.': tensor} of the checkpoint's vision tensors, bf16 widened to `dtype`."""
    from safetensors import safe_open

    out = {}
    with safe_open(str(Path(snapshot) / "model.safetensors"), framework="pt") as f:
        for k in f.keys():
            if k.startswith(PREFIX):
                out[k[len(PREFIX):]] = f.get_tensor(k).to(dtype)
    return out


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _linear(w: torch.Tensor, b: torch.Tensor | None) -> nn.Linear:
    m = nn.Linear(w.shape[1], w.shape[0], bias=b is not None, dtype=w.dtype)
    m.weight = nn.Parameter(w.detach().clone(), requires_grad=False)
    if b is not None:
        m.bias = nn.Parameter(b.detach().clone(), requires_grad=False)
    return m


def _norm(w: torch.Tensor, b: torch.Tensor) -> nn.LayerNorm:
    m = nn.LayerNorm(w.shape[0], eps=EPS, dtype=w.dtype)
    m.weight = nn.Parameter(w.detach().clone(), requires_grad=False)
    m.bias = nn.Parameter(b.detach().clone(), requires_grad=False)
    return m


class ScaledLayerNorm(nn.Module):
    """The fp16-safe variant: LayerNorm with x * 2^-k, eps * 4^-k and the rsqrt capped at its value for zero variance
    (eps * 4^-k flushes in fp16). The same number in fp32 up to rounding; in fp16 the squares and their SUM stay inside the range."""

    def __init__(self, w: torch.Tensor, b: torch.Tensor, k: int):
        super().__init__()
        self.weight = nn.Parameter(w.detach().clone(), requires_grad=False)
        self.bias = nn.Parameter(b.detach().clone(), requires_grad=False)
        self.s = float(2 ** -k)
        self.eps = float(torch.tensor(EPS, dtype=torch.float32) * (self.s * self.s))
        self.cap = float(torch.rsqrt(torch.tensor(self.eps, dtype=torch.float32)))

    def forward(self, x):
        h = x * self.s
        d = h - h.mean(-1, keepdim=True)
        r = torch.rsqrt((d * d).mean(-1, keepdim=True) + self.eps)
        return d * torch.minimum(r, torch.tensor(self.cap, dtype=r.dtype)) * self.weight + self.bias


F16S_BLOCK_LOG2, F16S_MERGER_LOG2 = 2, 6     # the probe's maxima: blocks |x - mean| 256.9, SUM 81,460; merger 6,341, SUM 4.5e7


def _norm_any(w, b, k):
    return ScaledLayerNorm(w, b, k) if k else _norm(w, b)


class Block(nn.Module):
    def __init__(self, sd: dict, i: int, hidden: int, heads: int, qblock: int, norm_log2: int = 0):
        super().__init__()
        p = f"blocks.{i}."
        self.hidden, self.heads, self.head_dim, self.qblock = hidden, heads, hidden // heads, qblock
        self.scaling = self.head_dim ** -0.5
        self.norm1 = _norm_any(sd[p + "norm1.weight"], sd[p + "norm1.bias"], norm_log2)
        self.norm2 = _norm_any(sd[p + "norm2.weight"], sd[p + "norm2.bias"], norm_log2)
        w, b = sd[p + "attn.qkv.weight"], sd[p + "attn.qkv.bias"]
        self.q = _linear(w[0:hidden], b[0:hidden])
        self.k = _linear(w[hidden:2 * hidden], b[hidden:2 * hidden])
        self.v = _linear(w[2 * hidden:3 * hidden], b[2 * hidden:3 * hidden])
        self.proj = _linear(sd[p + "attn.proj.weight"], sd[p + "attn.proj.bias"])
        self.fc1 = _linear(sd[p + "mlp.linear_fc1.weight"], sd[p + "mlp.linear_fc1.bias"])
        self.fc2 = _linear(sd[p + "mlp.linear_fc2.weight"], sd[p + "mlp.linear_fc2.bias"])

    def forward(self, h, cos, sin, key_bias):                       # h [1, N, C]; cos / sin [1, N, 1, d]; key_bias [1, 1, 1, N]
        n = h.shape[1]
        x = self.norm1(h)
        q = self.q(x).reshape(1, n, self.heads, self.head_dim)
        k = self.k(x).reshape(1, n, self.heads, self.head_dim)
        v = self.v(x).reshape(1, n, self.heads, self.head_dim)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)        # [1, H, N, d]
        k_t = k.transpose(2, 3)
        q = q * self.scaling
        if self.qblock and self.qblock < n:
            outs = []
            for s in range(0, n, self.qblock):
                a = torch.softmax(torch.matmul(q[:, :, s:s + self.qblock], k_t) + key_bias, dim=-1)
                outs.append(torch.matmul(a, v))
            o = torch.cat(outs, dim=2)
        else:
            o = torch.matmul(torch.softmax(torch.matmul(q, k_t) + key_bias, dim=-1), v)
        o = o.transpose(1, 2).reshape(1, n, self.hidden)
        h = h + self.proj(o)
        return h + self.fc2(F.gelu(self.fc1(self.norm2(h)), approximate="tanh"))


class TowerRow(nn.Module):
    def __init__(self, sd: dict, n: int, heads: int = HEADS, qblock: int = 0, f16s: bool = False):
        super().__init__()
        w = sd["patch_embed.proj.weight"]                                   # [C, 3, 2, 16, 16]
        hidden = w.shape[0]
        assert n % MERGE_UNIT == 0, n
        self.n, self.hidden, self.qblock = n, hidden, qblock
        self.patch = _linear(w.reshape(hidden, -1), sd["patch_embed.proj.bias"])
        depth = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks."))
        self.blocks = nn.ModuleList(Block(sd, i, hidden, heads, qblock, F16S_BLOCK_LOG2 if f16s else 0) for i in range(depth))
        self.norm = _norm_any(sd["merger.norm.weight"], sd["merger.norm.bias"], F16S_MERGER_LOG2 if f16s else 0)
        self.fc1 = _linear(sd["merger.linear_fc1.weight"], sd["merger.linear_fc1.bias"])
        self.fc2 = _linear(sd["merger.linear_fc2.weight"], sd["merger.linear_fc2.bias"])

    def forward(self, patches, valid, pos, cos, sin):
        n, d = self.n, cos.shape[-1]
        h = self.patch(patches) + pos
        c4, s4 = cos.reshape(1, n, 1, d), sin.reshape(1, n, 1, d)
        key_bias = (valid - 1.0).reshape(1, 1, 1, n) * NEG
        for blk in self.blocks:
            h = blk(h, c4, s4, key_bias)
        x = self.norm(h).reshape(1, n // MERGE_UNIT, MERGE_UNIT * self.hidden)
        return {"rows": self.fc2(F.gelu(self.fc1(x)))}


def graph(sd: dict, n: int, qblock: int = 0, f16s: bool = False) -> TowerRow:
    return TowerRow(sd, n, qblock=qblock, f16s=f16s).eval().requires_grad_(False)


def tag(n: int, qblock: int = 0) -> str:
    return f"N{n}" + (f"_qb{qblock}" if qblock and qblock < n else "")


def pad_inputs(patches: np.ndarray, pos: np.ndarray, cos: np.ndarray, sin: np.ndarray, n_bucket: int) -> dict:
    """The contract's padding: patches / pos zero, cos 1, sin 0, valid 0 after the picture's n patches."""
    n = patches.shape[0]
    assert n <= n_bucket and n % MERGE_UNIT == 0, (n, n_bucket)
    out = {"patches": np.zeros((1, n_bucket, patches.shape[1]), np.float32), "valid": np.zeros((1, n_bucket), np.float32),
           "pos": np.zeros((1, n_bucket, pos.shape[1]), np.float32), "cos": np.ones((1, n_bucket, cos.shape[1]), np.float32),
           "sin": np.zeros((1, n_bucket, sin.shape[1]), np.float32)}
    out["patches"][0, :n] = patches
    out["valid"][0, :n] = 1.0
    out["pos"][0, :n] = pos
    out["cos"][0, :n] = cos
    out["sin"][0, :n] = sin
    return out


def tiny_state(hidden=64, heads=4, ffn=96, out=80, depth=2, seed=0) -> dict:
    """A random state dict with the checkpoint's key layout at tiny sizes (the rehearsals; no checkpoint read)."""
    g = torch.Generator().manual_seed(seed)
    r = lambda *s, sc=0.05: torch.randn(*s, generator=g) * sc   # noqa: E731
    sd = {"patch_embed.proj.weight": r(hidden, 3, 2, 16, 16, sc=0.02), "patch_embed.proj.bias": r(hidden)}
    for i in range(depth):
        p = f"blocks.{i}."
        sd.update({p + "norm1.weight": 1 + r(hidden, sc=0.1), p + "norm1.bias": r(hidden),
                   p + "norm2.weight": 1 + r(hidden, sc=0.1), p + "norm2.bias": r(hidden),
                   p + "attn.qkv.weight": r(3 * hidden, hidden), p + "attn.qkv.bias": r(3 * hidden),
                   p + "attn.proj.weight": r(hidden, hidden), p + "attn.proj.bias": r(hidden),
                   p + "mlp.linear_fc1.weight": r(ffn, hidden), p + "mlp.linear_fc1.bias": r(ffn),
                   p + "mlp.linear_fc2.weight": r(hidden, ffn), p + "mlp.linear_fc2.bias": r(hidden)})
    sd.update({"merger.norm.weight": 1 + r(hidden, sc=0.1), "merger.norm.bias": r(hidden),
               "merger.linear_fc1.weight": r(4 * hidden, 4 * hidden), "merger.linear_fc1.bias": r(4 * hidden),
               "merger.linear_fc2.weight": r(out, 4 * hidden), "merger.linear_fc2.bias": r(out)})
    return sd


def contract(n: int, qblock: int = 0) -> dict:
    return {"graph": "TowerRow (vision tower + merger, patch-count bucket)", "N": n, "qblock": qblock,
            "inputs": [{"name": "patches", "shape": [1, n, PATCH_DIM], "dtype": "FLOAT32"},
                       {"name": "valid", "shape": [1, n], "dtype": "FLOAT32"},
                       {"name": "pos", "shape": [1, n, HIDDEN], "dtype": "FLOAT32"},
                       {"name": "cos", "shape": [1, n, HEAD_DIM], "dtype": "FLOAT32"},
                       {"name": "sin", "shape": [1, n, HEAD_DIM], "dtype": "FLOAT32"}],
            "outputs": [{"name": "rows", "shape": [1, n // MERGE_UNIT, OUT], "dtype": "FLOAT32"}]}
