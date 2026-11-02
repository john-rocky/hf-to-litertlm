"""d1-2B decision graph over a BATCH of B packed sequences of one bucket length L <= 256 (the card's
"64 states in one call" as one graph call per 8 sequences instead of one per sequence):

    D1DecideB(weights, cfg, B, L)(embeds [B, L, 2048], sees_q [B, L, 66], sees_k [B, L, 66]) -> {"hidden": [B, L, 2048]}

Row b is exactly the batch-1 D1Decide (d1_graph.py) on sequence b: the same weights, RMSNorm, RoPE tables, `set` mask
from the row's own sees_q / sees_k, causal sliding window (L <= 256 = one causal constant), short convolution and MLP.
What changes is only where a batch-1 form would put a batch > 1 tensor in a shape ML Drift (the GPU delegate) reads
wrongly (a runtime rank-3 tensor beside a constant of rank 4):

- short convolution: the batch-1 graph pads the time axis of a rank-3 [1, L, C] tensor; ML Drift's rank-3 PAD shifts a
  row when the batch is > 1, so here the product b * x is viewed as rank 4 [B, L, 1, C] before the pad of k - 1 = 6
  zeros on the time axis and the 7 shifted slices (the same sums), and back to [B, L, C] after the taps;
- GQA: the batch-1 graph repeats the kv heads through a rank-4 [kv, 1, L, hd] view (batch 1 dropped); here
  [B * kv, 1, L, hd], the 4 copies concatenated on axis 1, then [B, H, L, hd] (kv head j still serves query heads
  4j..4j+3; no rank 5);
- the sliding layers' causal mask: the batch-1 graph makes it a runtime rank-4 tensor because litert-torch folds the
  batch-1 rank-4 score matmul to rank 3; with B > 1 the scores stay rank 4 [B, H, L, L], so the constant [1, 1, L, L]
  is added as the RoPE tables are multiplied (runtime rank 4 + constant rank 4). `causal_runtime=True` builds the
  batch-1 graph's form instead (constant x a runtime one per row): the shipped B = 8 graphs use it.
The full layers' mask is made from the inputs as in the batch-1 graph ((sees_q . sees_k^T - 1) * 30000 per row,
[B, 1, L, L]).

Padding rows (a batch with fewer than B sequences): embeds 0, sees_q [0, 1, 0..], sees_k 0 on every position (the
host writes them, `make_sees_batch`); they see nothing real, stay finite (the full layers' scores are all -30000 =
a uniform softmax over zeros), and no real row reads them (attention is per row).
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "export"))
sys.path.insert(0, HERE)
import d1_graph as G  # noqa: E402  (the batch-1 graph module)
from d1_layout_b import make_sees_batch  # noqa: E402,F401  (numpy, shared with the host)

NEG = G.NEG


class ShortConvB(nn.Module):
    def __init__(self, W, p, cfg):
        super().__init__()
        self.in_proj, self.out_proj = G.linear(W[p + "in_proj.weight"]), G.linear(W[p + "out_proj.weight"])
        w = W[p + "conv.weight"]                       # [C, 1, k]
        self.k = w.shape[-1]
        self.taps = nn.Parameter(w[:, 0, :].t().contiguous()[:, None, None, :], requires_grad=False)  # [k, 1, 1, C]
        self.c = cfg.hidden

    def forward(self, x, ctx):
        B, L = x.shape[0], x.shape[1]
        bcx = self.in_proj(x)
        b, c, xx = bcx[..., : self.c], bcx[..., self.c: 2 * self.c], bcx[..., 2 * self.c:]
        bx = (b * xx).reshape(B, L, 1, self.c)
        xp = F.pad(bx, (0, 0, 0, 0, self.k - 1, 0))   # [B, L + k - 1, 1, C]: zeros before position 0, rank 4
        y = xp[:, 0:L] * self.taps[0]
        for i in range(1, self.k):
            y = y + xp[:, i:i + L] * self.taps[i]
        return self.out_proj(c * y.reshape(B, L, self.c))


class AttentionB(nn.Module):
    def __init__(self, W, p, cfg, kind):
        super().__init__()
        self.q_proj, self.k_proj = G.linear(W[p + "q_proj.weight"]), G.linear(W[p + "k_proj.weight"])
        self.v_proj, self.o_proj = G.linear(W[p + "v_proj.weight"]), G.linear(W[p + "o_proj.weight"])
        self.gate_proj = G.linear(W[p + "gate_proj.weight"])
        self.q_norm, self.k_norm = G.RMSNorm(W[p + "q_norm.weight"], cfg.eps), G.RMSNorm(W[p + "k_norm.weight"], cfg.eps)
        self.h, self.kv, self.hd, self.kind = cfg.heads, cfg.kv, cfg.hd, kind
        self.scale = cfg.hd ** -0.5

    def _gqa(self, t, B, L):
        rep = self.h // self.kv
        t = t.reshape(B * self.kv, 1, L, self.hd)
        return torch.cat([t] * rep, dim=1).reshape(B, self.h, L, self.hd)

    def forward(self, x, ctx):
        B, L = x.shape[0], x.shape[1]
        cos, sin = ctx[self.kind + "_rope"]
        q = self.q_norm(self.q_proj(x).view(B, L, self.h, self.hd)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(B, L, self.kv, self.hd)).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.kv, self.hd).transpose(1, 2)
        q = q * cos + G.rotate_half(q) * sin
        k = k * cos + G.rotate_half(k) * sin
        k, v = self._gqa(k, B, L), self._gqa(v, B, L)
        mask = ctx["full_mask"] if self.kind == "full" else ctx["causal"]
        s = torch.matmul(q, k.transpose(-1, -2)) * self.scale + mask
        y = torch.matmul(torch.softmax(s, dim=-1), v)                       # [B, H, L, hd]
        g = torch.sigmoid(self.gate_proj(x)).unsqueeze(-1)                  # [B, L, H, 1]
        y = (y.transpose(1, 2) * g).reshape(B, L, self.h * self.hd)
        return self.o_proj(y)


class LayerB(nn.Module):
    def __init__(self, W, i, cfg):
        super().__init__()
        p = f"layers.{i}."
        t = cfg.layer_types[i]
        self.op = (ShortConvB(W, p + "conv.", cfg) if t == "conv" else
                   AttentionB(W, p + "self_attn.", cfg, "full" if t == "full_attention" else "sliding"))
        self.ffn = G.Mlp(W, p + "feed_forward.")
        n = lambda name: G.RMSNorm(W[p + name + ".weight"], cfg.eps)
        self.operator_norm, self.post_operator_norm = n("operator_norm"), n("post_operator_norm")
        self.ffn_norm, self.post_ffn_norm = n("ffn_norm"), n("post_ffn_norm")

    def forward(self, h, ctx):
        h = h + self.post_operator_norm(self.op(self.operator_norm(h), ctx))
        return h + self.post_ffn_norm(self.ffn(self.ffn_norm(h)))


class D1DecideB(nn.Module):
    """embeds [B, L, d] + sees_q / sees_k [B, L, 66] -> {"hidden": [B, L, d]}, L <= the sliding window (256)."""

    def __init__(self, W: dict, cfg, B: int, L: int, dtype=torch.float32, causal_runtime: bool = False):
        super().__init__()
        assert L <= cfg.window, "the batch graph is for one short bucket (L <= the sliding window)"
        self.B, self.L, self.causal_runtime = B, L, causal_runtime
        self.layers = nn.ModuleList([LayerB(W, i, cfg) for i in range(len(cfg.layer_types))])
        self.embedding_norm = G.RMSNorm(W["embedding_norm.weight"], cfg.eps)
        for kind, key in (("full", "full_attention"), ("sliding", "sliding_attention")):
            cos, sin = G.rope_tables(cfg, key, L, dtype)
            self.register_buffer(kind + "_cos", cos, persistent=False)
            self.register_buffer(kind + "_sin", sin, persistent=False)
        self.register_buffer("causal", G.causal_constant(L, dtype), persistent=False)   # [1, 1, L, L]
        self.to(dtype)

    def forward(self, embeds, sees_q, sees_k):
        allowed = torch.matmul(sees_q, sees_k.transpose(1, 2))             # [B, L, L], exactly 0 / 1
        ctx = {"full_rope": (self.full_cos, self.full_sin), "sliding_rope": (self.sliding_cos, self.sliding_sin),
               "full_mask": ((allowed - 1.0) * NEG).unsqueeze(1)}
        if self.causal_runtime:   # the batch-1 form, per row: constant x the row's first key flag (1 on a real row)
            ctx["causal"] = self.causal * sees_k[:, :1, :1].reshape(self.B, 1, 1, 1)
        else:
            ctx["causal"] = self.causal
        h = embeds
        for layer in self.layers:
            h = layer(h, ctx)
        return {"hidden": self.embedding_norm(h)}


def build(B: int, L: int, W: dict, cfg, dtype=torch.float32, causal_runtime: bool = False) -> D1DecideB:
    return D1DecideB(W, cfg, B, L, dtype, causal_runtime).eval().requires_grad_(False)
