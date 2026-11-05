"""DecideRow: the decision graph = one state-free causal row of the Qwen3.5 text stack of
openbmb/MiniCPM-OCR-Jev (revision a0ed278e), embeddings and position tables in, the final-norm hidden state of
every position out.

    forward(embeds f32 [1, L, 2048], valid f32 [1, L], cos f32 [1, L, 64], sin f32 [1, L, 64]) -> {"hidden": f32 [1, L, 2048]}

The form:
  - the model: 24 layers (linear_attention x 18, full_attention at 3, 7, 11, 15, 19, 23), hidden 2048, 8 heads / 2 kv /
    head_dim 256, GatedDeltaNet 16 k + 16 v heads x 128 (ratio 1: the head interleave is not traced), FFN 6144;
  - one safetensors file, read directly (SNAP = snapshot/);
  - the attention function takes the query-block form when `qblock` is set (L >= 4096: no [L, L] tensor in the graph; a
    block's keys stop at the block's last query, so its scores are [1, 8, Q, s + Q]; mask = a shared triangle on the last Q
    columns + the pad bias on the keys);
  - `mask_form`: the runtime-rank-4 forms the tiny rehearsal compares.
The mask of the plain form: causal_const + (1 - valid)[:, None, None, :] * -1e4.
No COS, SIN, INT64, GATHER, no embed_tokens, no score in the graph."""
from __future__ import annotations

import copy
import json
import os
import struct
import sys
import time
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(os.path.abspath(__file__)).parent))
import patch_q35 as P  # noqa: E402  (import runs the rank-4 kernel gate and the anchor asserts)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS  # noqa: E402
from transformers.models.qwen3_5 import modeling_qwen3_5 as M  # noqa: E402

SNAP = Path(os.path.abspath(__file__)).parents[1] / "snapshot"
PREFIX = "model.language_model."
NEG = -1e4
ATTN = "row_eager"
ATTN_QB = "row_eager_qb"
FORM = "R64+sp+ec+dd+vs6"
HIDDEN = 2048
SWAPS = ("Qwen3_5TextModel", "Qwen3_5GatedDeltaNet", "Qwen3_5TextRotaryEmbedding")
ORIG = {name: getattr(M, name) for name in SWAPS}      # the stock classes, before any swap
PATCHED = {"Qwen3_5TextModel": P.PatchedQwen3_5TextModel, "Qwen3_5GatedDeltaNet": P.PatchedQwen3_5GatedDeltaNet,
           "Qwen3_5TextRotaryEmbedding": P.PatchedQwen3_5TextRotaryEmbedding}
FULL_LAYERS = [3, 7, 11, 15, 19, 23]


def row_eager(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    """key / value [1, Hkv, L, D] -> [Hkv, 1, L, D] -> n_rep copies by concat on dim 1 ->
    [1, Hkv * n_rep, L, D] (repeat_kv's order), scores = q @ k^T * scaling + mask (rank-4 BMM), softmax in float32,
    out = p @ v -> [1, L, H, D]."""
    n_rep = module.num_key_value_groups
    b, hkv, length, dim = key.shape
    assert b == 1, "batch 1 only (the batch dim is folded to keep the copy axis inside rank 4)"
    key_states = torch.cat([key.reshape(hkv, 1, length, dim)] * n_rep, dim=1).reshape(1, hkv * n_rep, length, dim)
    value_states = torch.cat([value.reshape(hkv, 1, length, dim)] * n_rep, dim=1).reshape(1, hkv * n_rep, length, dim)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    attn_weights = attn_weights + attention_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, None


def row_eager_qb(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    """The same computation with the queries in blocks of Q. attention_mask = (bias [1, 1, 1, L], tri [1, 1, Q, Q], Q).
    A block's keys stop at its last query: scores [1, H, Q, s + Q]. Columns before the block see all queries (causal),
    only the pad bias applies; the last Q columns get the triangle + the pad bias. Same numbers as the plain form up to
    the softmax's rounding (the masked keys are exactly 0 in the sum either way)."""
    bias, tri, q_size = attention_mask
    n_rep = module.num_key_value_groups
    b, hkv, length, dim = key.shape
    assert b == 1 and length % q_size == 0, (b, length, q_size)
    key_states = torch.cat([key.reshape(hkv, 1, length, dim)] * n_rep, dim=1).reshape(1, hkv * n_rep, length, dim)
    value_states = torch.cat([value.reshape(hkv, 1, length, dim)] * n_rep, dim=1).reshape(1, hkv * n_rep, length, dim)
    outs = []
    for s in range(0, length, q_size):
        e = s + q_size
        q_b = query[:, :, s:e]
        k_b = key_states[:, :, :e].transpose(2, 3)
        scores = torch.matmul(q_b, k_b) * scaling
        if s:
            scores = torch.cat([scores[..., :s] + bias[..., :s], scores[..., s:] + (tri + bias[..., s:e])], dim=-1)
        else:
            scores = scores + (tri + bias[..., :e])
        p = nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        outs.append(torch.matmul(p, value_states[:, :, :e]))
    out = torch.cat(outs, dim=2).transpose(1, 2).contiguous()
    return out, None


def register_attention() -> None:
    ALL_ATTENTION_FUNCTIONS.register(ATTN, row_eager)
    ALL_ATTENTION_FUNCTIONS.register(ATTN_QB, row_eager_qb)


def _set_classes(classes: dict) -> None:
    pkg = sys.modules["transformers.models.qwen3_5"]
    for name, cls in classes.items():
        setattr(M, name, cls)
        if name in vars(pkg):
            setattr(pkg, name, cls)


def causal_constant(L: int) -> torch.Tensor:
    rows = [[0.0 if j <= i else NEG for j in range(L)] for i in range(L)]
    return torch.tensor(rows, dtype=torch.float32).reshape(1, 1, L, L)


def text_config(attn: str = ATTN):
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(str(SNAP)).text_config
    cfg._attn_implementation = attn
    return cfg


def read_header(path: Path) -> tuple[dict, int]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def _build(cfg, classes: dict) -> nn.Module:
    _set_classes(classes)
    try:
        with torch.device("meta"):
            model = M.Qwen3_5TextModel(cfg)
    finally:
        _set_classes(PATCHED)
    model.embed_tokens = None               # the host owns the table: the graph never builds it
    return model


def _finish(model: nn.Module, cfg, rotary_cls) -> nn.Module:
    model.rotary_emb = rotary_cls(config=cfg)   # off the meta device (the stock one reads inv_freq)
    meta = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.is_meta]
    assert not meta, meta[:5]
    return model.eval().requires_grad_(False)


def load_text_model(attn: str = ATTN, dtype=torch.float32):
    """-> (patched text model, `dtype`, CPU, embed_tokens None; info). Weights read once from the one safetensors file."""
    from safetensors import safe_open
    register_attention()
    cfg = text_config(attn)
    t0 = time.perf_counter()
    model = _build(cfg, PATCHED)
    path = SNAP / "model.safetensors"
    sd = {}
    with safe_open(str(path), framework="pt") as f:
        keys = [k[len(PREFIX):] for k in f.keys() if k.startswith(PREFIX)]
        assert "embed_tokens.weight" in keys
        keys.remove("embed_tokens.weight")
        names = set(model.state_dict())
        assert names == set(keys), (sorted(names - set(keys))[:5], sorted(set(keys) - names)[:5])
        for k in keys:
            sd[k] = f.get_tensor(PREFIX + k)
    src_dtypes = sorted({str(t.dtype) for t in sd.values()})
    sd = {k: v.to(dtype) for k, v in sd.items()}
    model.load_state_dict(sd, strict=True, assign=True)
    del sd
    model = _finish(model, cfg, P.PatchedQwen3_5TextRotaryEmbedding)
    load_s = time.perf_counter() - t0
    assert type(model) is P.PatchedQwen3_5TextModel, type(model)
    gdn = [m for m in model.modules() if isinstance(m, M.Qwen3_5GatedDeltaNet)]
    attn_mods = [m for m in model.modules() if isinstance(m, M.Qwen3_5Attention)]
    assert gdn and all(type(m) is P.PatchedQwen3_5GatedDeltaNet for m in gdn)
    assert all(m.chunk_gated_delta_rule is P._rank4_chunk_gated_delta_rule for m in gdn)
    layer_types = [layer.block_type for layer in model.layers]
    attn_idx = [i for i, t in enumerate(layer_types) if t == "full_attention"]
    assert len(model.layers) == 24 and len(gdn) == 18 and len(attn_mods) == 6, (len(model.layers), len(gdn), len(attn_mods))
    assert attn_idx == FULL_LAYERS, attn_idx
    assert (gdn[0].num_k_heads, gdn[0].num_v_heads, gdn[0].head_k_dim, gdn[0].head_v_dim) == (16, 16, 128, 128)
    assert (cfg.num_attention_heads, cfg.num_key_value_heads, attn_mods[0].head_dim) == (8, 2, 256)
    assert cfg.hidden_size == HIDDEN and isinstance(model.norm, M.Qwen3_5RMSNorm)
    assert all(m.config._attn_implementation == attn for m in attn_mods)
    n_params = sum(p.numel() for p in model.parameters())
    info = {"model_class": type(model).__name__, "layers": len(model.layers), "gdn_layers": len(gdn),
            "attention_layers": attn_idx, "attn_implementation": cfg._attn_implementation,
            "gdn_heads": {"k": gdn[0].num_k_heads, "v": gdn[0].num_v_heads, "ratio": gdn[0].num_v_heads // gdn[0].num_k_heads,
                          "k_dim": gdn[0].head_k_dim, "v_dim": gdn[0].head_v_dim},
            "attention_heads": {"q": cfg.num_attention_heads, "kv": cfg.num_key_value_heads, "head_dim": attn_mods[0].head_dim,
                                "num_key_value_groups": attn_mods[0].num_key_value_groups},
            "hidden_size": cfg.hidden_size, "intermediate_size": cfg.intermediate_size, "final_norm": type(model.norm).__name__,
            "params": n_params, "fp32_bytes": n_params * 4, "tensors": len(keys), "source_dtypes": src_dtypes,
            "embed_tokens": None, "load_seconds": round(load_s, 1), "rope_parameters": cfg.rope_parameters,
            "rms_norm_eps": cfg.rms_norm_eps}
    return model, info


def stock_twin(model: nn.Module) -> nn.Module:
    """The same weights (shared storage, no copy) in the stock classes of transformers 5.14.1 with sdpa attention."""
    cfg = copy.deepcopy(model.config)
    cfg._attn_implementation = "sdpa"
    stock = _build(cfg, ORIG)
    stock.load_state_dict(model.state_dict(), strict=True, assign=True)
    return _finish(stock, cfg, ORIG["Qwen3_5TextRotaryEmbedding"])


def apply_form(model: nn.Module, form: str = FORM) -> dict:
    import kernel_q35 as R
    return {k: v for k, v in R.apply(model, form).items() if k != "tokens"}


class DecideRow(nn.Module):
    """qblock = 0: the plain form (attention = row_eager, mask [1, 1, L, L]). qblock = Q > 0: attention = row_eager_qb, the
    text model's config must say so (set by the caller: set_attention)."""

    def __init__(self, text_model: nn.Module, L: int, qblock: int = 0):
        super().__init__()
        self.text_model = text_model
        self.L, self.qblock = L, qblock
        self.register_buffer("position_ids", torch.arange(L, dtype=torch.long).reshape(1, L), persistent=False)
        if qblock and qblock < L:
            assert L % qblock == 0, (L, qblock)
            self.register_buffer("tri", causal_constant(qblock), persistent=False)
        else:
            self.register_buffer("causal_const", causal_constant(L), persistent=False)

    def forward(self, embeds, valid, cos, sin):
        self.text_model._decide_valid = valid
        self.text_model.rotary_emb._decide_cos_sin = (cos, sin)
        bias = (1.0 - valid)[:, None, None, :] * NEG                 # [1, 1, 1, L]
        if self.qblock and self.qblock < self.L:
            full = (bias, self.tri, self.qblock)
        else:
            full = self.causal_const + bias
        masks = {"full_attention": full, "linear_attention": None}
        out = self.text_model(inputs_embeds=embeds, attention_mask=masks, position_ids=self.position_ids,
                              use_cache=False, past_key_values=None)
        return {"hidden": out.last_hidden_state}


class DecideHead(DecideRow):
    """Lever (c): the head in the graph, one row picked by a one-hot input: `pick` [1, L], `logits` [1, 2]. No GATHER: MUL + SUM over the rows."""

    def __init__(self, text_model: nn.Module, L: int, qblock: int, score_w: torch.Tensor):
        super().__init__(text_model, L, qblock)
        self.score = nn.Linear(score_w.shape[1], score_w.shape[0], bias=False, dtype=score_w.dtype)
        self.score.weight = nn.Parameter(score_w, requires_grad=False)

    def forward(self, embeds, valid, cos, sin, pick):
        hidden = super().forward(embeds, valid, cos, sin)["hidden"]
        row = (hidden * pick[:, :, None]).sum(dim=1, keepdim=True)        # [1, 1, 2048]
        return {"logits": self.score(row)[:, 0]}


def graph_head(model: nn.Module, L: int, qblock: int = 0) -> nn.Module:
    from safetensors import safe_open

    with safe_open(str(SNAP / "model.safetensors"), framework="pt") as f:
        w = f.get_tensor("score.weight").to(torch.float32)
    set_attention(model, ATTN_QB if qblock and qblock < L else ATTN)
    return DecideHead(model, L, qblock, w).eval().requires_grad_(False)


F16S_FINAL_LOG2 = 4


def apply_f16s(model: nn.Module, k: int = F16S_FINAL_LOG2) -> dict:
    """The fp16-safe variant: the only norm site past fp16's range in this graph is the final RMSNorm (input |x| 748.6 from the last layer's
    mlp, x^2 560,400, SUM 600,500 > 65,504: inf -> rsqrt 0 -> a row of zeros, which is what Metal's default precision returned). x * 2^-k, eps * 4^-k, rsqrt capped at its value for a zero input (eps * 4^-k flushes in fp16, a pad row would be 0 * inf). The same
    number in fp32 up to the rsqrt's rounding (powers of two); k = 4 = the probe's maximum with 4 x headroom on the SUM."""
    s = float(2 ** -k)
    eps = torch.tensor(model.norm.eps, dtype=torch.float32) * (s * s)
    cap = float(torch.rsqrt(eps))

    def scaled_norm(self, x):
        h = x * s
        r = torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
        return h * torch.minimum(r, torch.tensor(cap, dtype=r.dtype))

    import types

    model.norm._norm = types.MethodType(scaled_norm, model.norm)
    return {"site": "norm (final RMSNorm)", "k": k, "eps_scaled": float(eps), "rsqrt_cap": cap}


class _ZbaFused(nn.Module):
    def __init__(self, fc: nn.Linear):
        super().__init__()
        self.fc = fc
        self.out = None


class _ZbaPart(nn.Module):
    """Stands where in_proj_z / b / a stood. The first part (z, called first by the forward) runs the fused FC; every part returns its slice."""

    def __init__(self, fused: _ZbaFused, start: int, end: int, first: bool):
        super().__init__()
        self.fused, self.start, self.end, self.first = fused, start, end, first

    def forward(self, x):
        if self.first:
            self.fused.out = self.fused.fc(x)
        return self.fused.out[..., self.start:self.end]


def apply_zba(model: nn.Module) -> dict:
    """Lever (c): litert-torch main 329dc25a3 for this graph: in_proj_z, in_proj_b, in_proj_a fused into one FC + a split, in_proj_qkv left alone (splitting its
    rows copies the largest activation). The same dot products; FC 222 -> 186 at 18 GatedDeltaNet layers."""
    n = 0
    for m in model.modules():
        if isinstance(m, M.Qwen3_5GatedDeltaNet):
            z, b, a = m.in_proj_z, m.in_proj_b, m.in_proj_a
            assert z.bias is None and b.bias is None and a.bias is None
            w = torch.cat([z.weight, b.weight, a.weight], dim=0)
            fc = nn.Linear(w.shape[1], w.shape[0], bias=False, dtype=w.dtype)
            fc.weight = nn.Parameter(w, requires_grad=False)
            fused = _ZbaFused(fc)
            nz, nb = z.weight.shape[0], b.weight.shape[0]
            m.in_proj_z = _ZbaPart(fused, 0, nz, True)
            m.in_proj_b = _ZbaPart(fused, nz, nz + nb, False)
            m.in_proj_a = _ZbaPart(fused, nz + nb, w.shape[0], False)
            n += 1
    assert n, "no GatedDeltaNet layer"
    return {"fused_layers": n, "z_b_a_rows": [nz, nb, w.shape[0] - nz - nb]}


def set_attention(model: nn.Module, attn: str) -> None:
    model.config._attn_implementation = attn
    for m in model.modules():
        if isinstance(m, M.Qwen3_5Attention):
            m.config._attn_implementation = attn


def graph(model: nn.Module, L: int, qblock: int = 0) -> nn.Module:
    set_attention(model, ATTN_QB if qblock and qblock < L else ATTN)
    return DecideRow(model, L, qblock).eval().requires_grad_(False)


def tag(L: int, qblock: int = 0) -> str:
    return f"L{L}" + (f"_qb{qblock}" if qblock and qblock < L else "")


def contract(L: int, qblock: int = 0, head: bool = False) -> dict:
    return {"graph": "DecideRow (state-free causal row, fp32 activations)", "L": L, "qblock": qblock, "form": FORM,
            "inputs": [{"name": "embeds", "shape": [1, L, HIDDEN], "dtype": "FLOAT32"},
                       {"name": "valid", "shape": [1, L], "dtype": "FLOAT32"},
                       {"name": "cos", "shape": [1, L, 64], "dtype": "FLOAT32"},
                       {"name": "sin", "shape": [1, L, 64], "dtype": "FLOAT32"}] + ([{"name": "pick", "shape": [1, L], "dtype": "FLOAT32"}] if head else []),
            "outputs": [{"name": "logits", "shape": [1, 2], "dtype": "FLOAT32"}] if head else [{"name": "hidden", "shape": [1, L, HIDDEN], "dtype": "FLOAT32"}],
            "attention_mask": "causal_const + (1 - valid) * -1e4" + (" (blocks: triangle + pad bias)" if qblock and qblock < L else ""),
            "state_io": "none"}
