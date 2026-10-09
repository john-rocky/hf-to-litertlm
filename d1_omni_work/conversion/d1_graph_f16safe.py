"""D1DecisionF16Safe: d1_graph.D1Decision with the fp16-safe norm rewrite (round 8, launch
2026-10-08-d1c-d1-omni-litert-opus-r8.md). Pure torch: imports in the reference venv and in the exporter venv.

Why: the converter lowers RMSNorm's mean(x^2) to MUL (the square) + SUM + MUL (1 / dim) and LayerNorm to MEAN /
SQUARED_DIFFERENCE / MEAN; an fp16-storage GPU path (Mac Metal default precision, Android FP16_WITH_FP32_ACCUM) keeps
those results in fp16, and round 2 (results/norm_range.json) measured 36 of the 50 norm sites above 65,504 on the
fixture (trunk operator_norm / ffn_norm x 16 + embedding_norm, head L1 norm1 / norm2, scorer LN; the largest
trunk.L14.ffn_norm 5.53e6). Past 65,504 the sum is inf, rsqrt gives 0 and the norm returns exact zeros: rounds 3-5
saw every row collapse to uniform probabilities with no non-finite value.

The rewrite, per site with k > 0 (s = 2^-k; k = the smallest k with max sum of squares * 4^-k <= 65,504 / 4, read
from results/norm_range.json `sites.<name>.k`):
    RMSNorm    w * x * rsqrt(mean(x^2) + eps)               ->  w * (x s) * rsqrt(mean((x s)^2) + eps s^2)
    LayerNorm  layer_norm(x, w, b, eps)                     ->  layer_norm(x s, w, b, eps s^2)
Both are the same function: (x s)^2 = x^2 s^2 and every reduction / division commutes with a power-of-two scale,
eps s^2 is exact (eps * 4^-k in binary floating point), and rsqrt(v s^2) = rsqrt(v) / s. In fp32 the scaled form is
expected to give bit-identical results (power-of-two products are exact away from under- / overflow); that is
measured, not assumed (scripts/f16safe_check.py). The weights are the provider's, unchanged; sites with k = 0 (the
q / k RMSNorms of the attention layers, the head's layer-0 LayerNorms) keep the original module, so their part of the
graph is the same as d1_graph's. The forward of every other module is d1_graph's.

    import d1_graph_f16safe as F16
    m = F16.D1DecisionF16Safe(text_config, head_layers, L)      # k table from results/norm_range.json
    G.load_state_dict_from_provider(m, sd)                       # same parameter keys as D1Decision
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import d1_graph as G

K = Path(__file__).resolve().parents[1]
NORM_RANGE = K / "results/norm_range.json"
FP16_MAX = 65504.0
MARGIN = 4.0                       # the k rule: max sum of squares after the scale <= FP16_MAX / MARGIN


class ScaledRMSNorm(G.RMSNorm):
    """G.RMSNorm on x * 2^-k with eps * 4^-k (same parameter `weight`)."""

    def __init__(self, dim: int, eps: float, k: int):
        super().__init__(dim, eps)
        self.k = int(k)
        self.s = 2.0 ** -self.k
        self.eps_k = eps * 4.0 ** -self.k          # python float: exact (a power-of-two multiple of eps)

    def forward(self, x):
        xs = x * self.s
        return self.weight * (xs * torch.rsqrt(xs.pow(2).mean(-1, keepdim=True) + self.eps_k))


class ScaledRMSNormSum(ScaledRMSNorm):
    """Diagnostic variant (round 8, after the Metal default-precision run): the same function written without the
    mean, w * (x s) * (rsqrt(sum((x s)^2) + eps dim s^2) * sqrt(dim)), dim a power of 4 (1024 -> sqrt 32). In fp32 the
    two forms agree bit for bit when rsqrt commutes with a power-of-4 scale (dim s^2 = 4^(5-k)); measured, not assumed.
    Why: the mean form's MUL 1 / dim takes the fixture's small sums at trunk L00 below the fp16 normal range (2^-14)
    after the 2^-k scale (80,080 of 89,225 real positions at L00 operator_norm); the sum stays in the normal range."""

    def __init__(self, dim: int, eps: float, k: int):
        super().__init__(dim, eps, k)
        root = round(dim ** 0.5)
        assert root * root == dim and (dim & (dim - 1)) == 0 and (dim.bit_length() - 1) % 2 == 0, dim
        self.root_dim = float(root)
        self.eps_sum = eps * dim * 4.0 ** -self.k          # exact: eps times a power of four

    def forward(self, x):
        xs = x * self.s
        return self.weight * (xs * (torch.rsqrt(xs.pow(2).sum(-1, keepdim=True) + self.eps_sum) * self.root_dim))


class ScaledLayerNorm(nn.LayerNorm):
    """nn.LayerNorm on x * 2^-k with eps * 4^-k (same parameters `weight` / `bias`)."""

    def __init__(self, dim: int, eps: float, k: int):
        super().__init__(dim, eps=eps)
        self.k = int(k)
        self.s = 2.0 ** -self.k
        self.eps_k = eps * 4.0 ** -self.k

    def forward(self, x):
        return F.layer_norm(x * self.s, self.normalized_shape, self.weight, self.bias, self.eps_k)


def norm_sites(model) -> dict:
    """site name (results/norm_range.json naming) -> (parent module, attribute) for the 50 norms of D1Decision."""
    out = {}
    for i, layer in enumerate(model.encoder.layers):
        out[f"trunk.L{i:02d}.operator_norm"] = (layer, "operator_norm")
        out[f"trunk.L{i:02d}.ffn_norm"] = (layer, "ffn_norm")
        if layer.is_attention_layer:
            out[f"trunk.L{i:02d}.q_layernorm"] = (layer.self_attn, "q_layernorm")
            out[f"trunk.L{i:02d}.k_layernorm"] = (layer.self_attn, "k_layernorm")
    out["trunk.embedding_norm"] = (model.encoder, "embedding_norm")
    for j, hl in enumerate(model.head.layers):
        out[f"head.L{j}.norm1"] = (hl, "norm1")
        out[f"head.L{j}.norm2"] = (hl, "norm2")
    out["head.scorer.0"] = (model.head.scorer, "0")
    return out


def load_k_table(path=NORM_RANGE) -> dict:
    """{site: k} from round 2's norm range json (`sites.<name>.k`)."""
    doc = json.loads(Path(path).read_text())
    return {name: int(v["k"]) for name, v in doc["sites"].items()}


def apply_k_table(model, k_table: dict, sum_sites=()) -> dict:
    """Swap every norm with k > 0 for its scaled form (before the weights are loaded: the keys are unchanged).
    Every site of the model must be in the table and the reverse. `sum_sites` (diagnostic): RMSNorm sites written in
    the sum form (ScaledRMSNormSum) instead of the mean form."""
    sites = norm_sites(model)
    assert set(sites) == set(k_table), (sorted(set(sites) ^ set(k_table)))
    assert set(sum_sites) <= {n for n in sites if k_table[n] > 0}, sorted(sum_sites)
    scaled = {}
    for name, (parent, attr) in sites.items():
        k = int(k_table[name])
        old = getattr(parent, attr)
        assert k >= 0, (name, k)
        if k == 0:
            assert type(old) in (G.RMSNorm, nn.LayerNorm), (name, type(old))
            continue
        if type(old) is G.RMSNorm:
            cls = ScaledRMSNormSum if name in sum_sites else ScaledRMSNorm
            new = cls(old.weight.shape[0], old.eps, k)
        elif type(old) is nn.LayerNorm:
            assert old.elementwise_affine and old.bias is not None and len(old.normalized_shape) == 1, name
            new = ScaledLayerNorm(old.normalized_shape[0], old.eps, k)
        else:
            raise TypeError(f"{name}: {type(old)}")
        new.load_state_dict(old.state_dict())
        setattr(parent, attr, new)
        scaled[name] = k
    assert all(type(getattr(p, a)) in (ScaledRMSNorm, ScaledRMSNormSum, ScaledLayerNorm)
               for n, (p, a) in norm_sites(model).items() if n in scaled)
    return {"sites": len(sites), "scaled": len(scaled), "k_by_site": scaled,
            "sum_form_sites": sorted(sum_sites), "unchanged": sorted(n for n in sites if n not in scaled)}


# Diagnostic variants (graph_build.py --f16safe-variant <name>): RMSNorm sites in the sum form.
VARIANTS = {"l0sum": ("trunk.L00.operator_norm", "trunk.L00.ffn_norm")}


class D1DecisionF16Safe(G.D1Decision):
    """D1Decision with the per-site fp16-safe norm rewrite; same inputs, outputs and parameter keys."""

    def __init__(self, text_config: dict, head_layers: int, length: int, k_table: dict | None = None,
                 variant: str | None = None):
        super().__init__(text_config, head_layers, length)
        self.k_table = dict(load_k_table() if k_table is None else k_table)
        self.variant = variant
        self.f16safe_report = apply_k_table(self, self.k_table, VARIANTS[variant] if variant else ())
        self.f16safe_report["variant"] = variant
