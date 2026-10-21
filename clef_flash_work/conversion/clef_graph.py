"""ClefRowPrefill: the row graph of this conversion = one state-free causal row of the Qwen3.5-9B text stack of
Cloudflare/clef-flash (revision f0b7d3cd), embeddings and position tables in, the final-norm hidden state of every
position out.

    forward(embeds f32 [1, L, 4096], valid f32 [1, L], cos f32 [1, L, 64], sin f32 [1, L, 64]) -> {"hidden": f32 [1, L, 4096]}

embeds = the checkpoint's bf16 `embed_tokens` rows widened to float32 by the host (pad rows 0); valid = 1.0 on the
record's tokens, 0.0 on the right pads; cos / sin = the host's M-RoPE tables (scripts/clef_host.py; a text record =
positions (i, i, i)); hidden = last_hidden_state after the final RMSNorm at every position (what the author's joint
schema head reads, with lm_head.weight on the host). No lm_head, no table, no state I/O; 1 call = 1 record.

Inside (Kev's row graph, kev_work/scripts/kev_graph.py, with embeds and position tables as inputs):
    mask4d = causal_const + (1 - valid)[:, None, None, :] * -1e4      (causal_const [1, 1, L, L] float32 from a python
                                                                          list: 0 on and below the diagonal, -1e4 above)
    text_model(inputs_embeds=embeds, attention_mask=mask4d, position_ids=<constant [1, L] buffer>, use_cache=False)
A 4-D float mask passes create_causal_mask as is and makes create_recurrent_attention_mask return None; the wrapper
sets text_model._clef_valid = valid (stashed on every GatedDeltaNet as `_litert_valid`: the pad guard) and
text_model.rotary_emb._clef_cos_sin = (cos, sin) (the patched rotary returns them; the position buffer feeds nothing).

Model = transformers 5.14.1's Qwen3_5TextModel built from config.json's text_config with the three classes of
scripts/clef_qwen35_patch.py swapped into the modeling module (Kev's order), constructed on the meta device; the
`model.language_model.*` tensors of the four shards except embed_tokens (the host's table) are read bf16 -> float32 and
assigned in place (`load_state_dict(strict=True, assign=True)`), so the float32 text stack (~27.7 GB) is the only copy.
attention = `kev_eager` (GQA by concat, rank-4 BMM, additive mask, float32 softmax) registered in
ALL_ATTENTION_FUNCTIONS. The GatedDeltaNet kernel form (`R64+sp+ec+dd+vs6`, scripts/clef_kernel.py) is applied by the
caller (`apply_form`). `count_interleave()` proves the ratio-2 head interleave runs (2 calls per GatedDeltaNet)."""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import torch
from torch import nn

K = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(K / "scripts"))
import clef_qwen35_patch as P  # noqa: E402  (import runs the rank-4 kernel gate and the anchor asserts)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS  # noqa: E402
from transformers.models.qwen3_5 import modeling_qwen3_5 as M  # noqa: E402

REV = "f0b7d3cd65296abfe459e6c228b618f5d8b26d2f"
SNAP = K / "hf/hub/models--Cloudflare--clef-flash/snapshots" / REV
PREFIX = "model.language_model."
NEG = -1e4
ATTN = "kev_eager"
FORM = "R64+sp+ec+dd+vs6"
HIDDEN = 4096
SWAPS = ("Qwen3_5TextModel", "Qwen3_5GatedDeltaNet", "Qwen3_5TextRotaryEmbedding")
ORIG = {name: getattr(M, name) for name in SWAPS}      # the stock classes, before any swap
PATCHED = {"Qwen3_5TextModel": P.PatchedQwen3_5TextModel, "Qwen3_5GatedDeltaNet": P.PatchedQwen3_5GatedDeltaNet,
           "Qwen3_5TextRotaryEmbedding": P.PatchedQwen3_5TextRotaryEmbedding}


def kev_eager(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    """kev_work/scripts/kev_graph.py's attention, unchanged: key/value [1, Hkv, L, D] -> [Hkv, 1, L, D] -> n_rep copies
    by concat on dim 1 -> [1, Hkv * n_rep, L, D] (repeat_kv's order), scores = q @ k^T * scaling + mask (rank-4 BMM),
    softmax in float32, out = p @ v -> [1, L, H, D]."""
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


def count_interleave():
    """kev_graph.count_interleave, unchanged: a call counter around the guarded forward's `_litert_interleave_heads`
    (same function underneath). -> {"calls", "ratios", "shapes"}."""
    g = P.PatchedQwen3_5GatedDeltaNet.forward.__globals__
    inner = getattr(g["_litert_interleave_heads"], "__wrapped__", g["_litert_interleave_heads"])
    counter = {"calls": 0, "ratios": [], "shapes": []}

    def counted(x, r):
        counter["calls"] += 1
        if len(counter["ratios"]) < 4:
            counter["ratios"].append(int(r))
            counter["shapes"].append(list(x.shape))
        return inner(x, r)

    counted.__wrapped__ = inner
    g["_litert_interleave_heads"] = counted
    return counter


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


def _shard_keys(model: nn.Module) -> dict:
    """{shard file: [module key, ...]} for every tensor of the model's state dict (checkpoint key = PREFIX + key)."""
    weight_map = json.loads((SNAP / "model.safetensors.index.json").read_text())["weight_map"]
    ckpt = {k[len(PREFIX):]: f for k, f in weight_map.items() if k.startswith(PREFIX)}
    assert ckpt.pop("embed_tokens.weight", None) is not None, "no embed_tokens in the checkpoint map"
    names = set(model.state_dict())
    assert names == set(ckpt), (sorted(names - set(ckpt))[:5], sorted(set(ckpt) - names)[:5])
    by_shard: dict = {}
    for k, f in sorted(ckpt.items()):
        by_shard.setdefault(f, []).append(k)
    return by_shard


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


def load_text_model():
    """-> (patched text model, float32, CPU, embed_tokens None; info). Weights read once from the shards."""
    from safetensors import safe_open
    ALL_ATTENTION_FUNCTIONS.register(ATTN, kev_eager)
    cfg = text_config()
    t0 = time.perf_counter()
    model = _build(cfg, PATCHED)
    by_shard = _shard_keys(model)
    sd = {}
    for shard, keys in by_shard.items():
        with safe_open(str(SNAP / shard), framework="pt") as f:
            for k in keys:
                t = f.get_tensor(PREFIX + k)
                sd[k] = t.to(torch.float32)
    src_dtypes = sorted({str(t.dtype) for t in sd.values()})
    model.load_state_dict(sd, strict=True, assign=True)
    del sd
    model = _finish(model, cfg, P.PatchedQwen3_5TextRotaryEmbedding)
    load_s = time.perf_counter() - t0
    assert type(model) is P.PatchedQwen3_5TextModel, type(model)
    gdn = [m for m in model.modules() if isinstance(m, M.Qwen3_5GatedDeltaNet)]
    attn = [m for m in model.modules() if isinstance(m, M.Qwen3_5Attention)]
    assert gdn and all(type(m) is P.PatchedQwen3_5GatedDeltaNet for m in gdn)
    assert all(m.chunk_gated_delta_rule is P._rank4_chunk_gated_delta_rule for m in gdn)
    layer_types = [layer.block_type for layer in model.layers]
    attn_idx = [i for i, t in enumerate(layer_types) if t == "full_attention"]
    assert len(model.layers) == 32 and len(gdn) == 24 and len(attn) == 8, (len(model.layers), len(gdn), len(attn))
    assert attn_idx == list(range(3, 32, 4)), attn_idx
    assert (gdn[0].num_k_heads, gdn[0].num_v_heads, gdn[0].head_k_dim, gdn[0].head_v_dim) == (16, 32, 128, 128)
    assert (cfg.num_attention_heads, cfg.num_key_value_heads, attn[0].head_dim) == (16, 4, 256)
    assert cfg.hidden_size == HIDDEN and isinstance(model.norm, M.Qwen3_5RMSNorm)
    assert all(m.config._attn_implementation == ATTN for m in attn)
    n_params = sum(p.numel() for p in model.parameters())
    info = {"model_class": type(model).__name__, "layers": len(model.layers), "gdn_layers": len(gdn),
            "attention_layers": attn_idx, "attn_implementation": cfg._attn_implementation,
            "gdn_heads": {"k": gdn[0].num_k_heads, "v": gdn[0].num_v_heads, "ratio": gdn[0].num_v_heads // gdn[0].num_k_heads,
                          "k_dim": gdn[0].head_k_dim, "v_dim": gdn[0].head_v_dim},
            "attention_heads": {"q": cfg.num_attention_heads, "kv": cfg.num_key_value_heads, "head_dim": attn[0].head_dim,
                                "num_key_value_groups": attn[0].num_key_value_groups},
            "hidden_size": cfg.hidden_size, "intermediate_size": cfg.intermediate_size, "final_norm": type(model.norm).__name__,
            "chunk_kernel": P._rank4_chunk_gated_delta_rule.__name__, "params": n_params, "fp32_bytes": n_params * 4,
            "tensors": sum(len(v) for v in by_shard.values()), "source_dtypes": src_dtypes, "embed_tokens": None,
            "load_seconds": round(load_s, 1), "rope_parameters": cfg.rope_parameters, "rms_norm_eps": cfg.rms_norm_eps}
    return model, info


def stock_twin(model: nn.Module) -> nn.Module:
    """The same weights (shared storage, no copy) in the stock classes of transformers 5.14.1 with sdpa attention: the
    environment row of the torch check (the stock forward computes its own positions and causal mask)."""
    cfg = copy.deepcopy(model.config)
    cfg._attn_implementation = "sdpa"
    stock = _build(cfg, ORIG)
    stock.load_state_dict(model.state_dict(), strict=True, assign=True)
    return _finish(stock, cfg, ORIG["Qwen3_5TextRotaryEmbedding"])


def apply_form(model: nn.Module, form: str = FORM) -> dict:
    import clef_kernel as R
    return {k: v for k, v in R.apply(model, form).items() if k != "tokens"}


class ClefRowPrefill(nn.Module):
    def __init__(self, text_model: nn.Module, L: int):
        super().__init__()
        self.text_model = text_model
        self.L = L
        self.register_buffer("position_ids", torch.arange(L, dtype=torch.long).reshape(1, L), persistent=False)
        self.register_buffer("causal_const", causal_constant(L), persistent=False)

    def forward(self, embeds, valid, cos, sin):
        self.text_model._clef_valid = valid
        self.text_model.rotary_emb._clef_cos_sin = (cos, sin)
        mask4d = self.causal_const + (1.0 - valid)[:, None, None, :] * NEG
        out = self.text_model(inputs_embeds=embeds, attention_mask=mask4d, position_ids=self.position_ids,
                              use_cache=False, past_key_values=None)
        return {"hidden": out.last_hidden_state}


def graph(model: nn.Module, L: int) -> nn.Module:
    return ClefRowPrefill(model, L).eval().requires_grad_(False)


def contract(L: int) -> dict:
    return {
        "graph": "ClefRowPrefill (state-free causal row prefill, fp32)", "L": L, "form": FORM,
        "inputs": [{"name": "embeds", "shape": [1, L, HIDDEN], "dtype": "float32",
                    "meaning": "the checkpoint's bf16 embed_tokens rows as float32; pad rows 0"},
                   {"name": "valid", "shape": [1, L], "dtype": "float32", "meaning": "1.0 = the record's token, 0.0 = right pad"},
                   {"name": "cos", "shape": [1, L, 64], "dtype": "float32", "meaning": "M-RoPE cos table (scripts/clef_host.py)"},
                   {"name": "sin", "shape": [1, L, 64], "dtype": "float32", "meaning": "M-RoPE sin table (scripts/clef_host.py)"}],
        "outputs": [{"name": "hidden", "shape": [1, L, HIDDEN], "dtype": "float32",
                     "meaning": "last_hidden_state (after the final RMSNorm), every position"}],
        "pad_rule": "pads only after the record's last token", "attention_mask": "causal_const + (1 - valid) * -1e4",
        "gated_deltanet_pad_guard": "valid zeroes pad hidden states before the projections, sets pad decay to identity "
                                    "(a -> -30 pre-softplus) and re-zeroes pad q/k/v after the causal conv",
        "state_io": "none (every record starts from zero conv / recurrent state)",
    }
