"""D1Decision: the d1-omni trunk + decision head as one static-L graph body (design 1-4, handoff
2026-10-08-d1-omni-litert.md). Pure torch, no transformers: it imports in the reference venv and in the exporter venv.

    forward(ids int32 [1, L], prefix f32 [1, L, 1024], media f32 [1, L], pad f32 [1, L], keep_right f32 [1, L],
            qtype_onehot f32 [1, 3]) -> {"scores": f32 [1, L]}

- h0 = embed(ids) * (1 - media) + prefix * media: text and media rows run the same graph; the host needs no embedding
  table (media positions take `prefix`, text positions take the table row of `ids`).
- trunk = the provider's encoder.Trunk (bidirectional LFM2, 10 ShortConv + 6 GQA attention layers, final RMSNorm),
  with the attention mask built in-graph by broadcasting from the inputs (no trace specialisation):
      mask [1, 1, L, L] = (1 - pad_k) * NEG + media_q * (1 - media_k) * NEG,  NEG = -1e4
  (exp(-1e4 + s) is exactly 0 in fp32 for any |s| < ~1e4 - 104: the same weights as the provider's -1e9.)
- GQA at rank 4: q [1, 16, L, 64] -> reshape [1, 8, 2L, 64] (query heads 2g and 2g+1 stacked on the row axis of kv
  group g = repeat_interleave's pairing), scores [1, 8, 2L, L] + cat([mask, mask], 2), -> reshape [1, 16, L, 64].
  No repeat_interleave / expand / repeat / broadcast_to, nothing above rank 4.
- RoPE cos / sin: per-L contiguous constants [1, 1, L, 64] built with the provider's rope() formula on the CPU in fp32.
- ShortConv: the provider's centred 3-tap, channels-last ([1, L, d]): in_proj(x * pad) -> b, c, u; bx = b * u;
  y = bx[t-1] * w0 + bx[t] * w1 + (bx[t+1] * keep_right[t]) * w2 (shifts by slice + concat with a zero row, no PAD);
  out_proj(c * y). The three taps are separate contiguous [1, 1, d] parameters (no strided views of one weight).
- head = the provider's DecisionHead written out: + qtype_onehot @ type_emb, two pre-norm encoder layers (16 heads,
  ReLU FF 4096, LayerNorm eps 1e-5, biases) whose keys are the text positions only (key mask pad * (1 - media)), then
  the scorer (LayerNorm -> Linear -> GELU(erf) -> Linear(d, 1)) at EVERY position. The head has no position
  information, so dropping the prefix from its keys gives the provider's text-only head input exactly.
- The host gathers scores at P + marker, applies the text temperature, softmaxes, and reverses a noul.

Weights: load_state_dict_from_provider() maps the provider's Trunk / DecisionHead state (or the checkpoint's
`encoder.*` / `head.*` keys) onto this module with KEY_RULES (the conv weight and the head's in_proj are split).
"""
from __future__ import annotations

import re

import torch
import torch.nn as nn
import torch.nn.functional as F

NEG = -1e4
D, HEADS, KV_HEADS, HEAD_DIM = 1024, 16, 8, 64
INPUT_NAMES = ("ids", "prefix", "media", "pad", "keep_right", "qtype_onehot")


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps))


def rope_tables(length: int, head_dim: int, theta: float):
    """The provider's Trunk.rope() (CPU, fp32) -> contiguous cos, sin [1, 1, L, head_dim]."""
    exponent = torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim
    inv_freq = 1.0 / (theta ** exponent)
    positions = torch.arange(length).float()
    freqs = (inv_freq[None, :, None] @ positions[None, None, :]).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos()[:, None].contiguous(), emb.sin()[:, None].contiguous()


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class Attention(nn.Module):
    """GQA 16q / 8kv, head_dim 64, per-head q/k RMSNorm, RoPE, rank-4 grouped matmuls."""

    def __init__(self, cfg):
        super().__init__()
        d = cfg["hidden_size"]
        self.heads, self.kv_heads = cfg["num_attention_heads"], cfg["num_key_value_heads"]
        self.head_dim = d // self.heads
        self.groups = self.heads // self.kv_heads
        self.q_proj = nn.Linear(d, self.heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d, self.kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d, self.kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.heads * self.head_dim, d, bias=False)
        self.q_layernorm = RMSNorm(self.head_dim, cfg["norm_eps"])
        self.k_layernorm = RMSNorm(self.head_dim, cfg["norm_eps"])
        self.scale = self.head_dim ** -0.5

    def forward(self, x, cos, sin, mask2):
        _, length, _ = x.shape
        q = self.q_layernorm(self.q_proj(x).reshape(1, length, self.heads, self.head_dim)).transpose(1, 2)
        k = self.k_layernorm(self.k_proj(x).reshape(1, length, self.kv_heads, self.head_dim)).transpose(1, 2)
        v = self.v_proj(x).reshape(1, length, self.kv_heads, self.head_dim).transpose(1, 2)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        qg = q.reshape(1, self.kv_heads, self.groups * length, self.head_dim)       # [1, 8, 2L, 64]
        s = torch.matmul(qg, k.transpose(2, 3)) * self.scale + mask2               # [1, 8, 2L, L]
        y = torch.matmul(torch.softmax(s, dim=-1), v)                                # [1, 8, 2L, 64]
        y = y.reshape(1, self.heads, length, self.head_dim).transpose(1, 2).reshape(1, length, -1)
        return self.out_proj(y)


class ShortConv(nn.Module):
    """out_proj(c * conv(b * u)) with the provider's centred 3-tap and the keep_right gate on the right tap."""

    def __init__(self, cfg):
        super().__init__()
        d = cfg["hidden_size"]
        self.d = d
        self.in_proj = nn.Linear(d, 3 * d, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)
        self.tap0 = nn.Parameter(torch.zeros(1, 1, d))
        self.tap1 = nn.Parameter(torch.zeros(1, 1, d))
        self.tap2 = nn.Parameter(torch.zeros(1, 1, d))
        self.register_buffer("zero_row", torch.zeros(1, 1, d), persistent=False)

    def forward(self, x, pad, keep_right):
        d = self.d
        bcu = self.in_proj(x * pad[:, :, None])
        b, c, u = bcu[:, :, 0:d], bcu[:, :, d:2 * d], bcu[:, :, 2 * d:3 * d]
        bx = b * u
        prev = torch.cat([self.zero_row, bx[:, :-1]], dim=1)     # bx[t-1]
        right = torch.cat([bx[:, 1:], self.zero_row], dim=1) * keep_right[:, :, None]   # bx[t+1], media never reads text
        y = prev * self.tap0
        y = y + bx * self.tap1
        y = y + right * self.tap2
        return self.out_proj(c * y)


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden = int(2 * cfg["intermediate_size"] / 3)
        hidden = int(cfg["block_ffn_dim_multiplier"] * hidden)
        hidden = cfg["block_multiple_of"] * ((hidden + cfg["block_multiple_of"] - 1) // cfg["block_multiple_of"])
        d = cfg["hidden_size"]
        self.w1 = nn.Linear(d, hidden, bias=False)
        self.w3 = nn.Linear(d, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Layer(nn.Module):
    def __init__(self, cfg, kind: str):
        super().__init__()
        self.is_attention_layer = kind == "full_attention"
        if self.is_attention_layer:
            self.self_attn = Attention(cfg)
        else:
            self.conv = ShortConv(cfg)
        self.feed_forward = MLP(cfg)
        self.operator_norm = RMSNorm(cfg["hidden_size"], cfg["norm_eps"])
        self.ffn_norm = RMSNorm(cfg["hidden_size"], cfg["norm_eps"])


class Trunk(nn.Module):
    def __init__(self, cfg, length: int):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])
        self.layers = nn.ModuleList(Layer(cfg, kind) for kind in cfg["layer_types"])
        self.embedding_norm = RMSNorm(cfg["hidden_size"], cfg["norm_eps"])
        cos, sin = rope_tables(length, cfg["hidden_size"] // cfg["num_attention_heads"], cfg["rope_theta"])
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, h, media, pad, keep_right):
        mask = (1.0 - pad)[:, None, None, :] * NEG + media[:, None, :, None] * (1.0 - media)[:, None, None, :] * NEG
        mask2 = torch.cat([mask, mask], dim=2)                                      # [1, 1, 2L, L]
        for layer in self.layers:
            x = layer.operator_norm(h)
            if layer.is_attention_layer:
                x = layer.self_attn(x, self.cos, self.sin, mask2)
            else:
                x = layer.conv(x, pad, keep_right)
            h = h + x
            h = h + layer.feed_forward(layer.ffn_norm(h))
        return self.embedding_norm(h)


class TypeEmb(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(3, d))

    def forward(self, onehot):
        return torch.matmul(onehot, self.weight)                                     # [1, 3] @ [3, d]


class HeadLayer(nn.Module):
    """nn.TransformerEncoderLayer(d, 16, 4d, dropout 0, batch_first, norm_first=True), written out."""

    def __init__(self, d: int, heads: int):
        super().__init__()
        self.heads, self.head_dim = heads, d // heads
        self.q_proj, self.k_proj, self.v_proj = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)
        self.linear1, self.linear2 = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)
        self.norm1, self.norm2 = nn.LayerNorm(d, eps=1e-5), nn.LayerNorm(d, eps=1e-5)
        self.scale = self.head_dim ** -0.5

    def forward(self, x, kmask):
        _, length, _ = x.shape
        y = self.norm1(x)
        q = self.q_proj(y).reshape(1, length, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(y).reshape(1, length, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(y).reshape(1, length, self.heads, self.head_dim).transpose(1, 2)
        s = torch.matmul(q, k.transpose(2, 3)) * self.scale + kmask                 # [1, 16, L, L] + [1, 1, 1, L]
        a = torch.matmul(torch.softmax(s, dim=-1), v).transpose(1, 2).reshape(1, length, -1)
        x = x + self.out_proj(a)
        return x + self.linear2(F.relu(self.linear1(self.norm2(x))))


class Head(nn.Module):
    def __init__(self, d: int, layers: int):
        super().__init__()
        self.type_emb = TypeEmb(d)
        self.layers = nn.ModuleList(HeadLayer(d, d // 64) for _ in range(layers))
        self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, h, media, pad, qtype_onehot):
        h = h + self.type_emb(qtype_onehot)[:, None, :]
        kmask = (1.0 - pad * (1.0 - media))[:, None, None, :] * NEG                 # keys = real text positions
        for layer in self.layers:
            h = layer(h, kmask)
        return self.scorer(h)[:, :, 0]                                               # [1, L]


class D1Decision(nn.Module):
    def __init__(self, text_config: dict, head_layers: int, length: int):
        super().__init__()
        self.length = length
        self.encoder = Trunk(text_config, length)
        self.head = Head(text_config["hidden_size"], head_layers)

    def embed(self, ids, prefix, media):
        return self.encoder.embed_tokens(ids) * (1.0 - media)[:, :, None] + prefix * media[:, :, None]

    def trunk(self, h0, media, pad, keep_right):
        return self.encoder(h0, media, pad, keep_right)

    def decide(self, h0, media, pad, keep_right, qtype_onehot):
        return self.head(self.trunk(h0, media, pad, keep_right), media, pad, qtype_onehot)

    def forward(self, ids, prefix, media, pad, keep_right, qtype_onehot):
        return {"scores": self.decide(self.embed(ids, prefix, media), media, pad, keep_right, qtype_onehot)}


# ---------------------------------------------------------------- weights

# provider / checkpoint key (encoder.* = Trunk, head.* = DecisionHead) -> this module's key(s)
KEY_RULES = [
    (r"encoder\.embed_tokens\.weight", "same"),
    (r"encoder\.embedding_norm\.weight", "same"),
    (r"encoder\.layers\.(\d+)\.(operator_norm|ffn_norm)\.weight", "same"),
    (r"encoder\.layers\.(\d+)\.feed_forward\.(w1|w2|w3)\.weight", "same"),
    (r"encoder\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|out_proj)\.weight", "same"),
    (r"encoder\.layers\.(\d+)\.self_attn\.(q_layernorm|k_layernorm)\.weight", "same"),
    (r"encoder\.layers\.(\d+)\.conv\.(in_proj|out_proj)\.weight", "same"),
    (r"encoder\.layers\.(\d+)\.conv\.conv\.weight", "conv_taps"),          # [d, 1, 3] -> tap0/1/2 [1, 1, d]
    (r"head\.type_emb\.weight", "same"),
    (r"head\.head\.layers\.(\d+)\.self_attn\.in_proj_weight", "in_proj_weight"),   # [3d, d] -> q/k/v_proj.weight
    (r"head\.head\.layers\.(\d+)\.self_attn\.in_proj_bias", "in_proj_bias"),       # [3d] -> q/k/v_proj.bias
    (r"head\.head\.layers\.(\d+)\.self_attn\.out_proj\.(weight|bias)", "head_out_proj"),
    (r"head\.head\.layers\.(\d+)\.(linear1|linear2|norm1|norm2)\.(weight|bias)", "head_layer"),
    (r"head\.scorer\.(0|1|3)\.(weight|bias)", "same"),
]
IGNORED_PREFIXES = ("vision.", "audio.")


def map_key(key: str, shape):
    """-> list of (our_key, how, out_shape) for one provider key, [] if ignored, None if no rule matches."""
    if key.startswith(IGNORED_PREFIXES):
        return []
    for pat, how in KEY_RULES:
        m = re.fullmatch(pat, key)
        if not m:
            continue
        if how == "same":
            return [(key, "copy", list(shape))]
        if how == "conv_taps":
            base = f"encoder.layers.{m.group(1)}.conv"
            d = shape[0]
            return [(f"{base}.tap{i}", f"weight[:, 0, {i}] -> [1, 1, d]", [1, 1, d]) for i in range(3)]
        if how in ("in_proj_weight", "in_proj_bias"):
            base = f"head.layers.{m.group(1)}"
            d = shape[0] // 3
            leaf = "weight" if how == "in_proj_weight" else "bias"
            rows = ["[0:d]", "[d:2d]", "[2d:3d]"]
            out = [d, shape[1]] if leaf == "weight" else [d]
            return [(f"{base}.{p}.{leaf}", f"{how}{rows[i]}", out) for i, p in enumerate(("q_proj", "k_proj", "v_proj"))]
        if how == "head_out_proj":
            return [(f"head.layers.{m.group(1)}.out_proj.{m.group(2)}", "copy", list(shape))]
        if how == "head_layer":
            return [(f"head.layers.{m.group(1)}.{m.group(2)}.{m.group(3)}", "copy", list(shape))]
    return None


def provider_to_ours(sd: dict) -> tuple[dict, dict]:
    """A provider state dict (keys `encoder.*` / `head.*`, optionally `vision.*` / `audio.*`) -> (our state dict,
    report). Every encoder / head key must match a rule."""
    out, unmatched, ignored = {}, [], []
    for key, t in sd.items():
        targets = map_key(key, list(t.shape))
        if targets is None:
            unmatched.append(key)
            continue
        if not targets:
            ignored.append(key)
            continue
        for ours, how, _ in targets:
            if how == "copy":
                out[ours] = t.detach().clone().contiguous()
            elif how.startswith("weight[:, 0, "):
                i = int(how[len("weight[:, 0, "):].split("]")[0])
                out[ours] = t.detach()[:, 0, i].clone().reshape(1, 1, -1).contiguous()
            else:  # in_proj split
                d = t.shape[0] // 3
                i = ("[0:d]", "[d:2d]", "[2d:3d]").index(how[how.index("["):])
                out[ours] = t.detach()[i * d:(i + 1) * d].clone().contiguous()
    return out, {"mapped_source_keys": len(sd) - len(unmatched) - len(ignored), "target_keys": len(out),
                 "unmatched": unmatched, "ignored": len(ignored)}


def load_state_dict_from_provider(model: D1Decision, provider) -> dict:
    """`provider` = a D1OmniModel (has .encoder / .head), a (Trunk, DecisionHead) pair, or a flat dict with the
    checkpoint's keys. Loads strictly (every parameter of `model` is set) and returns the mapping report."""
    if isinstance(provider, dict):
        sd = provider
    else:
        trunk, head = (provider.encoder, provider.head) if hasattr(provider, "encoder") else provider
        sd = {**{f"encoder.{k}": v for k, v in trunk.state_dict().items()},
              **{f"head.{k}": v for k, v in head.state_dict().items()}}
    ours, report = provider_to_ours(sd)
    assert not report["unmatched"], report["unmatched"]
    missing, unexpected = model.load_state_dict(ours, strict=False)
    assert not unexpected, unexpected
    assert not missing, missing   # buffers (cos / sin / zero_row) are non-persistent, so nothing else may be missing
    report["loaded_parameters"] = len(ours)
    return report


def checkpoint_state(path) -> dict:
    """The checkpoint's `encoder.*` / `head.*` tensors (fp32, as stored) from model.safetensors; vision / audio keys
    are not read. Feed the result to load_state_dict_from_provider()."""
    from safetensors import safe_open

    out = {}
    with safe_open(str(path), framework="pt") as f:
        for key in f.keys():
            if key.startswith(("encoder.", "head.")):
                out[key] = f.get_tensor(key)
    return out


def load_checkpoint(model: D1Decision, path) -> dict:
    """Load model.safetensors into D1Decision through KEY_RULES (strict: every parameter set, nothing unexpected).
    Every source tensor must be float32. Returns the mapping report plus the source key / dtype counts."""
    sd = checkpoint_state(path)
    dtypes = sorted({str(t.dtype) for t in sd.values()})
    assert dtypes == ["torch.float32"], dtypes
    report = load_state_dict_from_provider(model, sd)
    report["source_tensors"] = len(sd)
    report["source_dtypes"] = dtypes
    return report


def sample_inputs(length: int, n_text: int, n_prefix: int = 0, seed: int = 0, qtype: int = 0):
    """Random sample tensors with the graph's dtypes and shapes (for tracing and dry runs)."""
    g = torch.Generator().manual_seed(seed)
    ids = torch.zeros(1, length, dtype=torch.int32)
    ids[0, n_prefix:n_prefix + n_text] = torch.randint(22, 64000, (n_text,), generator=g, dtype=torch.int32)
    prefix = torch.zeros(1, length, D)
    prefix[0, :n_prefix] = torch.randn(n_prefix, D, generator=g)
    t = torch.arange(length)
    media = (t < n_prefix).float()[None]
    pad = (t < n_prefix + n_text).float()[None]
    keep_right = (t != n_prefix - 1).float()[None]
    onehot = F.one_hot(torch.tensor([qtype]), 3).float()
    return {"ids": ids, "prefix": prefix, "media": media, "pad": pad, "keep_right": keep_right,
            "qtype_onehot": onehot}
