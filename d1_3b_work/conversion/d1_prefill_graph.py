"""D1Prefill: the row-prefill graph of the d1-3B LiteRT conversion = one state-free causal row of the LFM2 text stack.

    forward(ids: int32 [1, L], valid: float32 [1, L]) -> {"hidden": float32 [1, L, d]}       (D1Prefill)
    forward(embeds: float32 [1, L, d], valid: float32 [1, L]) -> {"hidden": float32 [1, L, d]} (D1PrefillEmbeds)

`hidden` is the output of the final RMSNorm (`embedding_norm`) at every position; d = 2048 on d1-3B. A row is one
question's tokens (state + question) from position 0, right-padded to L; `valid` is 1.0 on real tokens and 0.0 on pads.

The stack is the provider's own `lfm2_vl.language_model(text_config)` (LiquidAI/d1-3B at REV, loaded from hf_small/),
so module and weight names are the checkpoint's (`model.language_model.` + name). Only the attention forward is
replaced (`RowAttention`, swapped onto the provider's Attention instances); every other module runs the provider's
code: `DecoderLayer.forward`, `ShortConv.forward` with `causal_conv(x, weight, tree=None)` (F.pad + depthwise conv1d:
a right pad never reaches a real token through a causal conv), the MLP and `RMSNorm`. The provider's
`LanguageModel.run` is replaced by the same loop with a context in place of `pos` (DecoderLayer passes `pos` through to
the operator untouched; ShortConv ignores it).

RowAttention (the export form; the provider's math, rearranged for LiteRT's GPU delegates):
  q/k/v projections -> per-head q/k RMSNorm (the provider's modules) -> [1, H, L, D] -> RoPE with constant tables
  cos/sin [1, 1, L, D] (float32, the provider's inv_freq and half-split `rotate`) -> GQA by concat (Kev-0.8B LiteRT's
  `kev_eager`: [1, Hkv, L, D] -> [Hkv, 1, L, D] -> n_rep copies on dim 1 -> [1, Hkv * n_rep, L, D], no BROADCAST_TO;
  kv head h serves query heads h * n_rep .. h * n_rep + n_rep - 1, the order of SDPA's enable_gqa) -> scores = rank-4
  BMM * scale + mask [1, 1, L, L] -> softmax in float32 -> BMM -> out_proj.
  mask = causal constant (0 on and below the diagonal, -1e4 above; a non-zero constant: an all-zero constant plus a
  row folds to BROADCAST_TO + INT64, Kev r10 trap 1) + (1 - valid)[:, None, None, :] * -1e4.
  The provider runs `F.scaled_dot_product_attention(is_causal=True, enable_gqa=True)`: same function, other rounding.

Weights: `--make-tiny` writes the round-2 tiny random model (`tiny_config()`, seed 0) to cache/tiny/; round 3 on,
`language_model_from_snapshot(<snapshot dir>)` loads the 266 `model.language_model.*` tensors of model.safetensors
(bfloat16 -> float32), strict.

Norm pre-scale (round 5, `apply_norm_scale(lm, k)`, `--norm-scale` of the scripts that build the graph): a fp16-storage
GPU path (Metal default precision, Android FP16_WITH_FP32_ACCUM) stores RMSNorm's sum of squares in fp16, and past
65,504 the row comes back exactly 0 (the fp16 norm note in REPRODUCE.md). Per site an integer k
(one k for every module of the site: operator_norm, ffn_norm, q_layernorm, k_layernorm, embedding_norm; k > 0 shrinks,
k < 0 grows a small-variance site), each module of the site runs the provider's forward on s x with s = 2^-k, eps s^2:
    x32 = x.float() * s;  y = (x32 * rsqrt(mean(x32^2) + eps * s^2)).type_as(x) * weight
= the provider's `(x32 * rsqrt(mean(x32^2) + eps)).type_as(x) * weight` with the same operations in the same order, so in
float32 it is bit-identical: a power-of-two factor commutes with every rounding while the values stay normal (x32 * s,
each square (x s)^2 = x^2 s^2, every partial sum of the mean and the division by n, and fp32(eps s^2) = fp32(eps) s^2);
s^2 = 4^-k is an even power of two, so rsqrt(s^2 v) = rsqrt(v) / s exactly for a correctly rounded 1 / sqrt; and
(x s) * (r / s) = x * r. Only values below 2^-126 / s (fp32 subnormal territory) can break it. The sum of squares the
reduction sees drops by 4^-k; the graph gains one MUL (x * s) per scaled module, except where the converter folds it into
the producing FC's weights (q / k layernorm: FC -> RESHAPE -> MUL by a constant becomes FC with weights x s, exact in
fp32; results/tiny_norm_scale_opscan.json); eps s^2 replaces eps in the ADD.
Round 6b: k may also be given per module (`layers.4.operator_norm`: k), which wins over the site's k; the derivation is
per module, so it holds the same way. On d1-3B one k per site had no solution at headroom 4 (layers 0-4: sums of squares
<= 1.8 with variances ~2e-5; layers 5-12: sums near 920), one k per module has (results/real_norm_scale_layer_r6b.json).

    $EXPORT scripts/d1_prefill_graph.py --make-tiny
    $EXPORT scripts/d1_prefill_graph.py --check-header   # real key map vs the header
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import types
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from d1_common import HF_SMALL, K, provider, sha256_file  # noqa: E402

NEG = -1e4
PREFIX = "model.language_model."
TINY_DIR = K / "cache/tiny"
TINY_SEED = 0
TINY_WEIGHTS = TINY_DIR / f"tiny_lfm2_seed{TINY_SEED}.safetensors"
TINY_PAD_ID = 255     # in-vocab stand-in for <|pad|> (124893) in the 256-token tiny vocab
TINY_OVERRIDES = {    # the tiny text_config of round 2
    "hidden_size": 64, "intermediate_size": 128, "num_attention_heads": 4, "num_key_value_heads": 2,
    "vocab_size": 256, "conv_L_cache": 3, "norm_eps": 1e-5,
    "layer_types": ["conv", "conv", "full_attention", "conv", "full_attention", "conv"],
    "rope_parameters": {"rope_theta": 1000000.0, "rope_type": "default"},
}
HEADER = K / "evidence/hub_safetensors_header_da1fe36a.json"


# --------------------------------------------------------------------------- #
# config and weights
# --------------------------------------------------------------------------- #


def real_text_config():
    """config.json's text_config as the Lfm2Config the provider's Lfm2VlForConditionalGeneration builds from."""
    from transformers import Lfm2Config

    return Lfm2Config.from_dict(json.loads((HF_SMALL / "config.json").read_text())["text_config"])


def tiny_config():
    """The real text_config with the tiny sizes of round 2 (every other flag, e.g. conv_bias / block_auto_adjust_ff_dim
    False, stays the checkpoint's)."""
    cfg = copy.deepcopy(real_text_config())
    for k, v in TINY_OVERRIDES.items():
        setattr(cfg, k, copy.deepcopy(v))
    cfg.num_hidden_layers = len(cfg.layer_types)
    for k in ("block_dim", "conv_dim"):     # unused by the provider's code; kept consistent with hidden_size
        if hasattr(cfg, k):
            setattr(cfg, k, cfg.hidden_size)
    if hasattr(cfg, "num_heads"):
        cfg.num_heads = cfg.num_attention_heads
    return cfg


def build(cfg) -> nn.Module:
    """The provider's language model (random init), float32, eval."""
    return provider("lfm2_vl").language_model(cfg).float().eval().requires_grad_(False)


def make_tiny() -> dict:
    """Seed-fixed weights for the tiny model, written once to TINY_WEIGHTS: norms 1 + 0.1 N(0,1) (so a weight that is
    not applied shows), the embedding N(0, 1), every other tensor N(0, 1 / fan_in) (fan_in = numel / out)."""
    from safetensors.torch import save_file

    lm = build(tiny_config())
    g = torch.Generator().manual_seed(TINY_SEED)
    sd = {}
    for name, t in sorted(lm.state_dict().items()):
        z = torch.randn(t.shape, generator=g, dtype=torch.float32)
        if name.endswith("norm.weight") or name.endswith("layernorm.weight"):
            sd[name] = 1.0 + 0.1 * z
        elif name == "embed_tokens.weight":
            sd[name] = z
        else:
            fan_in = t[0].numel()
            sd[name] = z / fan_in ** 0.5
    TINY_DIR.mkdir(parents=True, exist_ok=True)
    assert not TINY_WEIGHTS.exists(), f"refusing to overwrite {TINY_WEIGHTS}"
    save_file(sd, str(TINY_WEIGHTS), metadata={"what": "d1-3B LiteRT round 2 tiny random LFM2 text stack",
                                               "seed": str(TINY_SEED), "overrides": json.dumps(TINY_OVERRIDES)})
    return {"file": str(TINY_WEIGHTS.relative_to(K)), "sha256": sha256_file(TINY_WEIGHTS), "tensors": len(sd),
            "params": int(sum(t.numel() for t in sd.values()))}


def load_tiny() -> tuple[nn.Module, dict]:
    from safetensors.torch import load_file

    lm = build(tiny_config())
    lm.load_state_dict(load_file(str(TINY_WEIGHTS)), strict=True)
    return lm, {"source": "tiny", "file": str(TINY_WEIGHTS.relative_to(K)), "sha256": sha256_file(TINY_WEIGHTS)}


def real_key_map(lm: nn.Module) -> dict:
    """checkpoint name -> module state-dict name for the language model."""
    return {PREFIX + k: k for k in lm.state_dict()}


def language_model_from_snapshot(snapshot: str | Path) -> tuple[nn.Module, dict]:
    """Round 3 on: the 266 language-model tensors of <snapshot>/model.safetensors, bfloat16 -> float32, strict."""
    from safetensors import safe_open

    path = Path(snapshot) / "model.safetensors"
    lm = build(real_text_config())
    names = real_key_map(lm)
    sd = {}
    with safe_open(str(path), framework="pt") as f:
        present = {k for k in f.keys() if k.startswith(PREFIX)}
        assert present == set(names), (sorted(present - set(names))[:5], sorted(set(names) - present)[:5])
        for ck, mk in names.items():
            sd[mk] = f.get_tensor(ck).to(torch.float32)
    lm.load_state_dict(sd, strict=True)
    return lm, {"source": "snapshot", "file": str(path), "tensors": len(sd)}


def check_header() -> dict:
    """Without weights: the real model built on the meta device against the Hub's safetensors header (names, shapes)."""
    hdr = json.loads(HEADER.read_text())["header"]
    with torch.device("meta"):
        lm = provider("lfm2_vl").language_model(real_text_config())
    names = real_key_map(lm)
    sd = lm.state_dict()
    in_hdr = {k for k in hdr if k.startswith(PREFIX)}
    shape_bad = [k for k in names if k in hdr and list(sd[names[k]].shape) != hdr[k]["shape"]]
    return {"model_tensors": len(names), "header_language_model_tensors": len(in_hdr),
            "missing_in_header": sorted(set(names) - in_hdr), "extra_in_header": sorted(in_hdr - set(names)),
            "shape_mismatch": shape_bad, "dtypes": sorted({hdr[k]["dtype"] for k in in_hdr}),
            "pass": not shape_bad and set(names) == in_hdr}


# --------------------------------------------------------------------------- #
# the export form
# --------------------------------------------------------------------------- #


def causal_constant(L: int) -> torch.Tensor:
    rows = [[0.0 if j <= i else NEG for j in range(L)] for i in range(L)]
    return torch.tensor(rows, dtype=torch.float32).reshape(1, 1, L, L)


def rope_tables(L: int, head_dim: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin [1, 1, L, head_dim] float32, computed as the provider's `hybrid.rope` computes them at positions 0..L-1
    with rotated = head_dim // 2 (every pair rotated)."""
    hybrid = provider("hybrid")
    pos = torch.arange(L)
    inv = 1.0 / theta ** (torch.arange(0, head_dim, 2).float() / head_dim)
    inv[head_dim // 2:] = 0.0
    cos, sin = hybrid.cos_sin(pos.float()[:, None] * inv)
    return cos.reshape(1, 1, L, head_dim).contiguous(), sin.reshape(1, 1, L, head_dim).contiguous()


def _row_attention_class():
    lfm2_vl, hybrid = provider("lfm2_vl"), provider("hybrid")

    class RowAttention(lfm2_vl.Attention):
        """The provider's Attention with the export-form forward; `ctx` = (cos, sin, mask) in place of `pos`."""

        def forward(self, x, ctx, tree):
            assert tree is None, "row graphs run no tree"
            cos, sin, mask = ctx
            b, length, _ = x.shape
            hd, h, hkv = self.head_dim, self.heads, self.kv_heads
            q = self.q_layernorm(self.q_proj(x).view(b, length, h, hd)).transpose(1, 2)
            k = self.k_layernorm(self.k_proj(x).view(b, length, hkv, hd)).transpose(1, 2)
            v = self.v_proj(x).view(b, length, hkv, hd).transpose(1, 2)
            q, k = hybrid.rotate(q, cos, sin), hybrid.rotate(k, cos, sin)
            n_rep = h // hkv
            k = torch.cat([k.reshape(hkv, 1, length, hd)] * n_rep, dim=1).reshape(1, h, length, hd)
            v = torch.cat([v.reshape(hkv, 1, length, hd)] * n_rep, dim=1).reshape(1, h, length, hd)
            scores = torch.matmul(q, k.transpose(2, 3)) * self.scale + mask
            p = torch.softmax(scores, dim=-1, dtype=torch.float32)
            y = torch.matmul(p, v)
            return self.out_proj(y.transpose(1, 2).flatten(2))

    return RowAttention


def to_row_form(lm: nn.Module) -> nn.Module:
    """Swap RowAttention onto every attention instance (same object, same parameters and names)."""
    cls = _row_attention_class()
    base = provider("lfm2_vl").Attention
    n = 0
    for layer in lm.layers:
        if layer.operator_name == "self_attn":
            assert type(layer.self_attn) in (base, cls), type(layer.self_attn)
            layer.self_attn.__class__ = cls
            n += 1
    lm._row_attention_layers = n
    return lm


# --------------------------------------------------------------------------- #
# norm pre-scale (module docstring, "Norm pre-scale")
# --------------------------------------------------------------------------- #

NORM_SITES = ("operator_norm", "ffn_norm", "q_layernorm", "k_layernorm", "embedding_norm")
NORM_K_MAX = 15


def norm_site(name: str, mod) -> str | None:
    """The site of an RMSNorm module by the last part of its name (`post_operator_norm` / `post_ffn_norm` are
    nn.Identity on this checkpoint and never match)."""
    last = name.rsplit(".", 1)[-1]
    return last if last in NORM_SITES and type(mod).__name__ == "RMSNorm" else None


def _rmsnorm_prescaled(self, x):
    """hybrid.RMSNorm.forward on s x with eps s^2, s = 2^-k (set per module by apply_norm_scale)."""
    s = self._d1_norm_scale
    x32 = x.float() * s
    return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps * (s * s))).type_as(x) * self.weight


def parse_norm_scale(arg: str | dict | None) -> dict:
    """--norm-scale: inline JSON {"site": k, ...}, or a JSON file: the `k` of d1_norm_range.py --recommend's
    results/norm_scale_<tag>.json, or a plain {"site": k} file. Missing sites are 0.
    Round 6b: keys may also be module names (`layers.4.operator_norm`, `layers.5.self_attn.q_layernorm`; the `k` of
    d1_norm_range.py --recommend --per-module): a module's own k wins over its site's. The result is the five sites
    (always, in NORM_SITES order) followed by the module keys (sorted); a site-only input gives the round-5 dict."""
    if not arg:
        return {s: 0 for s in NORM_SITES}
    if isinstance(arg, dict):
        doc = arg
    else:
        p = Path(arg) if Path(arg).is_absolute() else K / arg
        doc = json.loads(p.read_text()) if p.exists() else json.loads(arg)
    ks = doc.get("k", doc) if isinstance(doc.get("k", None), dict) else doc
    bad = sorted(k for k in ks if k not in NORM_SITES and k.rsplit(".", 1)[-1] not in NORM_SITES)
    assert not bad, f"unknown norm sites / modules {bad[:5]}; sites are {NORM_SITES}"
    out = {s: int(ks.get(s, 0)) for s in NORM_SITES}
    mods = {m: int(ks[m]) for m in sorted(ks) if m not in NORM_SITES}
    assert all(-NORM_K_MAX <= v <= NORM_K_MAX and v == ks.get(s, 0) for s, v in out.items()), ks
    assert all(-NORM_K_MAX <= v <= NORM_K_MAX and v == ks[m] for m, v in mods.items()), ks
    out.update(mods)
    return out


def apply_norm_scale(lm: nn.Module, ks: str | dict | None) -> dict:
    """Pre-scale every RMSNorm module of each site with k != 0 (instance forward; the class stays RMSNorm, the weights
    and names are unchanged). k < 0 scales up (a small-variance site; Kev's gated norm took 2^7 = k -7). Returns the
    record written into the export / check json."""
    ks = parse_norm_scale(ks)
    per_module = {m: v for m, v in ks.items() if m not in NORM_SITES}
    modules, scaled, k_of, seen = {s: 0 for s in NORM_SITES}, [], {}, set()
    for name, mod in lm.named_modules():
        site = norm_site(name, mod)
        if site is None:
            continue
        modules[site] += 1
        assert not hasattr(mod, "_d1_norm_scale"), f"{name} is already pre-scaled"
        k = per_module.get(name, ks[site])   # round 6b: a module's own k wins over its site's
        seen.update({name} & set(per_module))
        if k:
            mod._d1_norm_scale = float(2.0 ** -k)
            mod.forward = types.MethodType(_rmsnorm_prescaled, mod)
            scaled.append(name)
            k_of[name] = k
    missing = sorted(set(per_module) - seen)
    assert not missing, f"norm-scale modules not in this model: {missing[:5]}"
    return {"k": ks, "modules_per_site": modules, "scaled_modules": len(scaled), "scaled": scaled,
            "k_per_scaled_module": k_of,
            "form": "x32 = x.float() * 2^-k; (x32 * rsqrt(mean(x32^2) + eps * 4^-k)).type_as(x) * weight"}


def norm_scale_tag(ks: dict) -> str:
    """A short name for a k set in NORM_SITES order, e.g. operator 3, ffn 2, q 0, k 1, embedding 4 -> 'ns32014'
    (k >= 10 -> '[k]', k < 0 -> 'm|k|')."""
    return "ns" + "".join(f"m{-ks[s]}" if ks[s] < 0 else (str(ks[s]) if ks[s] < 10 else f"[{ks[s]}]") for s in NORM_SITES)


class D1Prefill(nn.Module):
    """ids int32 [1, L] + valid float32 [1, L] -> {"hidden": float32 [1, L, d]}."""

    def __init__(self, lm: nn.Module, L: int):
        super().__init__()
        attn = next(layer.self_attn for layer in lm.layers if layer.operator_name == "self_attn")
        assert type(attn).__name__ == "RowAttention", "call to_row_form(lm) first"
        self.lm, self.L = lm, L
        cos, sin = rope_tables(L, attn.head_dim, attn.theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer("causal_const", causal_constant(L), persistent=False)

    def run(self, h, valid):
        mask = self.causal_const + (1.0 - valid)[:, None, None, :] * NEG
        ctx = (self.cos, self.sin, mask)
        for layer in self.lm.layers:
            h = layer(h, ctx, None)
        return {"hidden": self.lm.embedding_norm(h)}

    def forward(self, ids, valid):
        return self.run(self.lm.embed_tokens(ids), valid)


class D1PrefillEmbeds(D1Prefill):
    """embeds float32 [1, L, d] + valid float32 [1, L] -> {"hidden": float32 [1, L, d]} (pictures, host lookup)."""

    def forward(self, embeds, valid):
        return self.run(embeds, valid)


def row_inputs(row_ids, L: int, pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """One row -> (ids int32 [1, L] right-padded with pad_id, valid float32 [1, L])."""
    n = len(row_ids)
    assert 0 < n <= L, (n, L)
    ids = torch.full((1, L), pad_id, dtype=torch.int32)
    ids[0, :n] = torch.tensor(list(row_ids), dtype=torch.int32)
    valid = torch.zeros((1, L), dtype=torch.float32)
    valid[0, :n] = 1.0
    return ids, valid


def graph(lm: nn.Module, L: int, embeds: bool = False) -> nn.Module:
    cls = D1PrefillEmbeds if embeds else D1Prefill
    return cls(lm, L).eval().requires_grad_(False)


def load(source: str, row_form: bool = True) -> tuple[nn.Module, dict]:
    """`tiny` or a snapshot directory -> (language model, info); in row form unless row_form=False (the provider's
    own Attention, for the reference; `to_row_form` can be applied later to the same object)."""
    lm, info = load_tiny() if source == "tiny" else language_model_from_snapshot(source)
    attn = [layer.self_attn for layer in lm.layers if layer.operator_name == "self_attn"]
    info.update(hidden=lm.embedding_norm.weight.shape[0], layers=len(lm.layers), attention_layers=len(attn),
                heads=attn[0].heads, kv_heads=attn[0].kv_heads, head_dim=attn[0].head_dim,
                vocab=lm.embed_tokens.weight.shape[0], rope_theta=attn[0].theta,
                layer_types=[layer.operator_name for layer in lm.layers])
    if row_form:
        to_row_form(lm)
    return lm, info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-tiny", action="store_true", help="write cache/tiny/tiny_lfm2_seed0.safetensors")
    ap.add_argument("--check-header", action="store_true", help="real key map vs the Hub header (no weights)")
    a = ap.parse_args()
    if a.make_tiny:
        print(json.dumps(make_tiny()))
    if a.check_header:
        r = check_header()
        print(json.dumps(r))
        return 0 if r["pass"] else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
