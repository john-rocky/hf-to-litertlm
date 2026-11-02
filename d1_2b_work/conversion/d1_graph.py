"""d1-2B decision graph, export form: LiquidAI/d1-2B's language model (LFM3, 24 layers) over one packed sequence of
`set` attention, right-padded to a bucket length L, returning the final-norm hidden state at every position.

    D1Decide(weights, cfg, L)(embeds [1, L, 2048], sees_q [1, L, 66], sees_k [1, L, 66]) -> {"hidden": [1, L, 2048]}

The provider's computation (modeling_lfm3.py of the snapshot, run by the provider under `attention.state_attention`
with `set` attention, attention.py), written out in plain torch so litert-torch sees only what LiteRT runs:

- Lfm3RMSNorm: weight * (x * rsqrt(mean(x^2) + eps)) (fp32 throughout in the fp32 graph).
- 13 short-convolution layers: in_proj -> B, C, x (channel thirds, in that order) -> B * x -> causal depthwise conv of
  width 7 (no bias) -> C * . -> out_proj. The conv runs as 7 shifted products summed (`depthwise.shifted_sum`'s form:
  a zero pad of 6 in front, then slices), so no layout transposes enter the graph.
- 11 attention layers: q/k/v projections, per-head q/k RMSNorm, RoPE (the provider's tables: full layers theta 1e6 on
  the first 32 of 64 frequencies (zero tail), sliding layers theta 1e4 on all 64; computed exactly as the provider
  computes them in fp32: position * inv_freq in fp32, then cos / sin), GQA 16 / 4 by concatenation (kv head j serves
  query heads 4j..4j+3, as repeat_kv), scores = q k^T * 128^-0.5 + mask, softmax, p v, then the per-head sigmoid output
  gate (gate_proj 2048 -> 16) and o_proj.
  * full layers (6): the `set` mask, built in the graph from two small inputs: allowed = sees_q . sees_k^T,
    mask = (allowed - 1) * 30000. sees_q[t] = [is_state(t), 1 - is_state(t), onehot_question(t)], sees_k[t] =
    real(t) * [1, is_state(t), onehot_question(t)] (question tags 1..64) reproduce `attention.Sees.rows`: a state token
    sees every real token, every other token the state and its own question, nothing sees padding. The three terms
    never overlap (a state token has no question tag), so the product is exactly 0 or 1: no clamp in the graph.
  * sliding layers (5, window 256): causal within the window over the packed row (a right pad never reaches a real
    token). L <= 256: one causal constant. L > 256: the provider's own blocked form (`attention._windowed`): blocks of
    256 queries over their block and the one before it, block 0's previous block zeros and masked.
- sandwich norms: h + post_operator_norm(op(operator_norm(h))), then h + post_ffn_norm(ffn(ffn_norm(h))); MLP
  w2(silu(w1 x) * w3 x); embedding_norm at the end.

Positions are 0..L-1 along the packed row (the provider's arange). Pad rows of the inputs: embeds 0, sees_q [0, 1, 0..],
sees_k 0 (the host writes them; `make_sees` below).

Options: `qblock` splits the full layers' queries into blocks (peak memory of the scores, for long buckets);
`norm_k` pre-scales chosen RMSNorm modules by 2^-k with eps * 4^-k (exact in fp32: the fp16-storage overflow
rewrite of d1-3B, applied only if measured necessary).
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

NEG = 30000.0
Q_MAX = 64
SEES = 2 + Q_MAX
PREFIX = "model.language_model."
SNAP_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                            "snapshot")


class Cfg:
    def __init__(self, snap: str = SNAP_DEFAULT):
        t = json.load(open(os.path.join(snap, "config.json")))["text_config"]
        self.hidden, self.inter = t["hidden_size"], t["intermediate_size"]
        self.heads, self.kv, self.hd = t["num_attention_heads"], t["num_key_value_heads"], t["head_dim"]
        self.eps, self.window, self.conv_k = t["norm_eps"], t["sliding_window"], t["conv_L_cache"]
        self.layer_types = list(t["layer_types"])
        self.rope = t["rope_parameters"]
        self.vocab = t["vocab_size"]
        assert t["use_sandwich_norm"] and not t["conv_bias"] and t["tie_word_embeddings"]


def load_text_weights(snap: str = SNAP_DEFAULT, dtype=torch.float32) -> dict:
    """The language model's tensors of model.safetensors (bf16), as `dtype`, names without the prefix."""
    from safetensors import safe_open

    out = {}
    with safe_open(os.path.join(snap, "model.safetensors"), framework="pt") as f:
        for k in f.keys():
            if k.startswith(PREFIX):
                out[k[len(PREFIX):]] = f.get_tensor(k).to(dtype)
    return out


def inv_freq(cfg: Cfg, kind: str) -> torch.Tensor:
    """Lfm3RotaryEmbedding.compute_default_rope_parameters, fp32 as the provider computes it."""
    p = cfg.rope[kind]
    base, partial, dim = p["rope_theta"], p.get("partial_rotary_factor", 1.0), cfg.hd
    angles = int(partial * dim) // 2
    f = 1.0 / (base ** (torch.arange(0, 2 * angles, 2, dtype=torch.float) / dim))
    return torch.cat((f, torch.zeros(dim // 2 - angles, dtype=torch.float)))


def rope_tables(cfg: Cfg, kind: str, L: int, dtype=torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    """cos / sin [1, 1, L, head_dim] at positions 0..L-1: in fp32 exactly the provider's computation
    (position.float() * inv_freq, cat, cos, sin); in fp64 the same product and functions in fp64."""
    f = inv_freq(cfg, kind)
    if dtype == torch.float32:
        freqs = torch.arange(L)[:, None].float() * f[None, :]
    else:
        freqs = torch.arange(L, dtype=dtype)[:, None] * f.to(dtype)[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype)[None, None], emb.sin().to(dtype)[None, None]


class RMSNorm(nn.Module):
    def __init__(self, weight: torch.Tensor, eps: float):
        super().__init__()
        self.weight = nn.Parameter(weight.clone(), requires_grad=False)
        self.eps = eps
        self.k = 0

    def forward(self, x):
        if self.k:
            s = 2.0 ** -self.k
            x = x * s
            return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps * s * s))
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps))


def linear(w: torch.Tensor) -> nn.Linear:
    """A bias-free linear over `w` itself (no copy: graphs of several buckets share one set of weights)."""
    m = nn.Linear(1, 1, bias=False)
    m.in_features, m.out_features = w.shape[1], w.shape[0]
    m.weight = nn.Parameter(w, requires_grad=False)
    return m


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


class Mlp(nn.Module):
    def __init__(self, W, p):
        super().__init__()
        self.w1, self.w3, self.w2 = linear(W[p + "w1.weight"]), linear(W[p + "w3.weight"]), linear(W[p + "w2.weight"])

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class ShortConv(nn.Module):
    def __init__(self, W, p, cfg: Cfg):
        super().__init__()
        self.in_proj, self.out_proj = linear(W[p + "in_proj.weight"]), linear(W[p + "out_proj.weight"])
        w = W[p + "conv.weight"]                       # [C, 1, k]
        self.k = w.shape[-1]
        self.taps = nn.Parameter(w[:, 0, :].t().contiguous()[:, None, None, :], requires_grad=False)  # [k, 1, 1, C]
        self.c = cfg.hidden

    def forward(self, x, ctx):
        L = x.shape[1]
        bcx = self.in_proj(x)
        b, c, xx = bcx[..., : self.c], bcx[..., self.c: 2 * self.c], bcx[..., 2 * self.c:]
        bx = b * xx
        xp = F.pad(bx, (0, 0, self.k - 1, 0))          # zeros before position 0
        y = xp[:, 0:L] * self.taps[0]
        for i in range(1, self.k):
            y = y + xp[:, i:i + L] * self.taps[i]
        return self.out_proj(c * y)


class Attention(nn.Module):
    def __init__(self, W, p, cfg: Cfg, kind: str):
        super().__init__()
        self.q_proj, self.k_proj = linear(W[p + "q_proj.weight"]), linear(W[p + "k_proj.weight"])
        self.v_proj, self.o_proj = linear(W[p + "v_proj.weight"]), linear(W[p + "o_proj.weight"])
        self.gate_proj = linear(W[p + "gate_proj.weight"])
        self.q_norm, self.k_norm = RMSNorm(W[p + "q_norm.weight"], cfg.eps), RMSNorm(W[p + "k_norm.weight"], cfg.eps)
        self.h, self.kv, self.hd, self.kind, self.window = cfg.heads, cfg.kv, cfg.hd, kind, cfg.window
        self.scale = cfg.hd ** -0.5

    def _gqa(self, t, L):
        rep = self.h // self.kv
        return torch.cat([t.reshape(self.kv, 1, L, self.hd)] * rep, dim=1).reshape(1, self.h, L, self.hd)

    def forward(self, x, ctx):
        L = x.shape[1]
        cos, sin = ctx[self.kind + "_rope"]
        q = self.q_norm(self.q_proj(x).view(1, L, self.h, self.hd)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(1, L, self.kv, self.hd)).transpose(1, 2)
        v = self.v_proj(x).view(1, L, self.kv, self.hd).transpose(1, 2)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        k, v = self._gqa(k, L), self._gqa(v, L)
        if self.kind == "full":
            y = self._full(q, k, v, ctx, L)
        else:
            y = self._sliding(q, k, v, ctx, L)
        g = torch.sigmoid(self.gate_proj(x)).unsqueeze(-1)      # [1, L, H, 1]
        y = (y.transpose(1, 2) * g).reshape(1, L, self.h * self.hd)
        return self.o_proj(y)

    def _attend(self, q, k, v, mask):
        s = torch.matmul(q, k.transpose(-1, -2)) * self.scale + mask
        return torch.matmul(torch.softmax(s, dim=-1), v)

    def _full(self, q, k, v, ctx, L):
        qb = ctx["qblock"]
        if not qb or qb >= L:
            return self._attend(q, k, v, ctx["full_mask"])
        # query blocks: each block's mask made here from the small inputs, so no [L, L] tensor is ever live
        outs = []
        for s in range(0, L, qb):
            allowed = torch.matmul(ctx["sees_q"][:, s:s + qb], ctx["sees_kT"])
            outs.append(self._attend(q[:, :, s:s + qb], k, v, ((allowed - 1.0) * NEG).unsqueeze(1)))
        return torch.cat(outs, dim=2)

    def _sliding(self, q, k, v, ctx, L):
        W = self.window
        if L <= W:
            return self._attend(q, k, v, ctx["causal"])
        nb = L // W
        qb = q.reshape(self.h, nb, W, self.hd).transpose(0, 1)                      # [nb, H, W, D]
        kb, vb = k.reshape(self.h, nb, W, self.hd), v.reshape(self.h, nb, W, self.hd)
        zeros = torch.zeros(self.h, 1, W, self.hd, dtype=q.dtype)
        kk = torch.cat([torch.cat([zeros, kb[:, : nb - 1]], dim=1), kb], dim=2).transpose(0, 1)   # [nb, H, 2W, D]
        vv = torch.cat([torch.cat([zeros, vb[:, : nb - 1]], dim=1), vb], dim=2).transpose(0, 1)
        y = self._attend(qb, kk, vv, ctx["band"])                               # [nb, H, W, D]
        return y.transpose(0, 1).reshape(1, self.h, L, self.hd)


class Layer(nn.Module):
    def __init__(self, W, i, cfg: Cfg):
        super().__init__()
        p = f"layers.{i}."
        t = cfg.layer_types[i]
        self.op = (ShortConv(W, p + "conv.", cfg) if t == "conv" else
                   Attention(W, p + "self_attn.", cfg, "full" if t == "full_attention" else "sliding"))
        self.ffn = Mlp(W, p + "feed_forward.")
        n = lambda name: RMSNorm(W[p + name + ".weight"], cfg.eps)
        self.operator_norm, self.post_operator_norm = n("operator_norm"), n("post_operator_norm")
        self.ffn_norm, self.post_ffn_norm = n("ffn_norm"), n("post_ffn_norm")

    def forward(self, h, ctx):
        h = h + self.post_operator_norm(self.op(self.operator_norm(h), ctx))
        return h + self.post_ffn_norm(self.ffn(self.ffn_norm(h)))


def causal_constant(L: int, dtype=torch.float32) -> torch.Tensor:
    i = torch.arange(L)
    return torch.where(i[None, :] <= i[:, None], 0.0, -NEG).to(dtype)[None, None]


def band_constants(W: int, nb: int, dtype=torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    """attention._windowed's mask, split: the band [1, 1, W, 2W] every block shares, and block 0's penalty on its
    (zero) previous block [nb, 1, 1, 2W]."""
    qpos = torch.arange(W)[:, None] + W
    kpos = torch.arange(2 * W)[None, :]
    band = torch.where((kpos <= qpos) & (kpos > qpos - W), 0.0, -NEG).to(dtype)[None, None]
    first = torch.zeros(nb, 1, 1, 2 * W, dtype=dtype)
    first[0, 0, 0, :W] = -NEG
    return band, first


class D1Decide(nn.Module):
    """embeds [1, L, d] + sees_q / sees_k [1, L, 66] -> {"hidden": [1, L, d]}."""

    def __init__(self, W: dict, cfg: Cfg, L: int, qblock: int | None = None, dtype=torch.float32):
        super().__init__()
        self.L, self.qblock = L, qblock
        self.layers = nn.ModuleList([Layer(W, i, cfg) for i in range(len(cfg.layer_types))])
        self.embedding_norm = RMSNorm(W["embedding_norm.weight"], cfg.eps)
        for kind, key in (("full", "full_attention"), ("sliding", "sliding_attention")):
            cos, sin = rope_tables(cfg, key, L, dtype)
            self.register_buffer(kind + "_cos", cos, persistent=False)
            self.register_buffer(kind + "_sin", sin, persistent=False)
        if L <= cfg.window:
            self.register_buffer("causal", causal_constant(L, dtype), persistent=False)
        else:
            assert L % cfg.window == 0, L
            band, first = band_constants(cfg.window, L // cfg.window, dtype)
            self.register_buffer("band", band, persistent=False)
            self.register_buffer("first_block", first, persistent=False)
        self.to(dtype)

    def forward(self, embeds, sees_q, sees_k):
        sees_kT = sees_k.transpose(1, 2)
        ctx = {"full_rope": (self.full_cos, self.full_sin), "sliding_rope": (self.sliding_cos, self.sliding_sin),
               "qblock": self.qblock, "sees_q": sees_q, "sees_kT": sees_kT}
        if not self.qblock or self.qblock >= self.L:
            allowed = torch.matmul(sees_q, sees_kT)
            ctx["full_mask"] = ((allowed - 1.0) * NEG).unsqueeze(1)
        # L <= W: the causal mask enters as a RUNTIME rank-4 tensor (constant x a runtime 1). Added as a constant to the
        # score matmul, which litert-torch folds to a rank-3 BATCH_MATMUL at batch 1, ML Drift reads it misaligned
        # (runtime rank 3 = BHWC(a,1,b,c), constant right-aligned HWC; Metal fp32 off by 8.4 at L128).
        # `one` = sees_k[0, 0, 0]: the first token is always a real state token.
        # L > W: the band stays a constant added to the rank-4 [nb, H, W, 2W] scores (same rank, the batch nb > 1 keeps
        # the matmul at rank 4): a runtime [1,1,1,1] x constant [nb,1,W,2W] product loses its batch on Metal.
        if hasattr(self, "causal"):
            ctx["causal"] = self.causal * sees_k[:, :1, :1].reshape(1, 1, 1, 1)
        else:
            ctx["band"] = self.band + self.first_block      # [nb, 1, W, 2W]; constant-folded by the converter
        h = embeds
        for layer in self.layers:
            h = layer(h, ctx)
        return {"hidden": self.embedding_norm(h)}


def build(L: int, W: dict | None = None, cfg: Cfg | None = None, qblock: int | None = None, dtype=torch.float32,
          snap: str = SNAP_DEFAULT) -> D1Decide:
    cfg = cfg or Cfg(snap)
    W = W if W is not None else load_text_weights(snap, dtype)
    return D1Decide(W, cfg, L, qblock, dtype).eval().requires_grad_(False)


def apply_norm_k(model: D1Decide, ks: dict) -> list[str]:
    """Pre-scale RMSNorm modules by name (`layers.4.operator_norm`, `layers.2.op.q_norm`, `embedding_norm`) or by the
    last part of the name (a site, every module of it): x * 2^-k, eps * 4^-k; exact in fp32 (d1-3B REPRODUCE)."""
    done = []
    for name, m in model.named_modules():
        if isinstance(m, RMSNorm):
            k = ks.get(name, ks.get(name.rsplit(".", 1)[-1], 0))
            if k:
                m.k = int(k)
                done.append(name)
    return done


from d1_layout import make_sees  # noqa: E402,F401  (numpy only; the host imports it from d1_layout)
