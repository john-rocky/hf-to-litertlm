"""Shared pieces of the iris-3b text-encoder graphs (Qwen3-VL-4B language stack).

Layout of one prompt (upstream `iris3b/text/qwen3_vl.py`): 34 prefix tokens, a
caption of at most 295 tokens, 5 suffix tokens, right-padded with
<|endoftext|> to 334. The DiT reads hidden_states[k] for k = 2, 5, ..., 35
(1-based = the output of decoder layer k-1), rows 34..333, masked.

Graphs: one graph per tap, so every graph has exactly one output and that
output is a sole leaf (the Mali [1, N, C] fan-out class never applies):

    c00  layers  0..1   -> hidden_states[2]
    c01  layers  2..4   -> hidden_states[5]
    ...
    c11  layers 32..34  -> hidden_states[35]

Inputs of every graph (all float32): hidden [1, 334, 2560], mask [1, 32, 334, 334]
(additive: 0 or MASK_NEG, causal plus padded keys, the same plane on every head),
cos and sin [1, 1, 334, 128] (1D rotary tables: with no position_ids every MRoPE
axis gets the same arange, so the interleaved MRoPE equals plain RoPE).

Rewrites relative to transformers 5.14.1 `Qwen3VLTextDecoderLayer` (all exact in
real arithmetic):
  * GQA 32 -> 8 heads: q [1, 32, S, 128] is reshaped to [1, 8, 4S, 128] (the four
    query heads of a KV group are contiguous) and the mask [1, 32, S, S] to
    [1, 8, 4S, S]; no repeat_kv, no expand, no 5D tensor, same-shape mask ADD.
  * The two attention products are einsums (rank-4 BATCH_MATMUL).
  * The attention scale 128**-0.5 is folded into the q_norm weight
    (RoPE is linear, so rope(s * q) = s * rope(q)).
  * Optional fp16-safe RMSNorm (safe=True): with m = max(1, max|x|) per row,
    w * (x/m) * rsqrt(mean((x/m)^2) + eps/m^2).
"""

import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# The Qwen3-VL-4B-Instruct checkpoint directory (config.json, tokenizer files, safetensors shards and their
# index). The scripts point it at their --text-encoder directory before any weight is read.
TE_DIR = Path("weights") / "Qwen3-VL-4B-Instruct"

SEQ = 334
PREFIX_LEN = 34
WINDOW = 300                 # text_encoder.max_length
SUFFIX_LEN = 5
PAD_ID = 151643
HIDDEN = 2560
N_HEADS = 32
N_KV = 8
HEAD_DIM = 128
INTER = 9728
EPS = 1e-6
ROPE_THETA = 5_000_000
VOCAB_ROWS = 151936
TAPS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)   # 1-based hidden_states index
MASK_NEG = -30000.0          # finite in fp16; exp underflows to 0 in fp32 and fp16

# [start, end) 0-based decoder layers per graph; the end layer's output is tap k = end.
CHUNKS = [(0, 2)] + [(k - 3, k) for k in TAPS[1:]]
assert [end for _, end in CHUNKS] == list(TAPS)


def chunk_name(i, form):
    return f"iris_te_c{i:02d}_{form}.tflite"


# ----------------------------------------------------------------- weights
def _weight_map():
    idx = json.loads((TE_DIR / "model.safetensors.index.json").read_text())
    return idx["weight_map"]


def load_tensors(names):
    """Reads checkpoint tensors (bf16) by full name; returns {name: tensor}."""
    from safetensors import safe_open

    wm = _weight_map()
    by_shard = {}
    for n in names:
        by_shard.setdefault(wm[n], []).append(n)
    out = {}
    for shard, keys in by_shard.items():
        with safe_open(str(TE_DIR / shard), framework="pt") as f:
            for k in keys:
                out[k] = f.get_tensor(k)
    return out


_LAYER_KEYS = (
    "input_layernorm.weight", "post_attention_layernorm.weight",
    "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
    "self_attn.o_proj.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight",
    "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
)


def layer_weights(i):
    """fp32 copies of decoder layer i's weights, keyed by the short name."""
    pre = f"model.language_model.layers.{i}."
    t = load_tensors([pre + k for k in _LAYER_KEYS])
    return {k: t[pre + k].float() for k in _LAYER_KEYS}


# ----------------------------------------------------------------- modules
def rms(x, w, eps, safe):
    """RMSNorm over the last dim, keepdim everywhere (no rank-0 tensor)."""
    if not safe:
        var = (x * x).mean(-1, keepdim=True)
        return w * (x * torch.rsqrt(var + eps))
    m = x.abs().amax(-1, keepdim=True).clamp_min(1.0)
    xs = x / m
    var = (xs * xs).mean(-1, keepdim=True)
    return w * (xs * torch.rsqrt(var + eps / (m * m)))


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class TELayer(nn.Module):
    """One Qwen3-VL text decoder layer, re-authored for a fixed-shape GPU graph."""

    def __init__(self, w, safe=False, rank4=False):
        super().__init__()
        self.safe = safe
        self.rank4 = rank4
        p = lambda t: nn.Parameter(t.contiguous(), requires_grad=False)
        self.ln_in = p(w["input_layernorm.weight"])
        self.ln_post = p(w["post_attention_layernorm.weight"])
        self.q_proj = p(w["self_attn.q_proj.weight"])
        self.k_proj = p(w["self_attn.k_proj.weight"])
        self.v_proj = p(w["self_attn.v_proj.weight"])
        self.o_proj = p(w["self_attn.o_proj.weight"])
        # attention scale folded into the q_norm weight (exact: RoPE is linear)
        self.q_norm = p(w["self_attn.q_norm.weight"] * (HEAD_DIM ** -0.5))
        self.k_norm = p(w["self_attn.k_norm.weight"])
        self.gate = p(w["mlp.gate_proj.weight"])
        self.up = p(w["mlp.up_proj.weight"])
        self.down = p(w["mlp.down_proj.weight"])

    def forward(self, h, mask_g, cos, sin):
        if self.rank4:
            return self._forward_rank4(h, mask_g, cos, sin)
        s = h.shape[1]
        x = rms(h, self.ln_in, EPS, self.safe)
        q = F.linear(x, self.q_proj).view(1, s, N_HEADS, HEAD_DIM)
        k = F.linear(x, self.k_proj).view(1, s, N_KV, HEAD_DIM)
        v = F.linear(x, self.v_proj).view(1, s, N_KV, HEAD_DIM)
        if self.safe:
            # 3D view for the max-normalized form (litert-torch's NHWC pass has no rewriter for a 4D amax)
            q = rms(q.reshape(1, s * N_HEADS, HEAD_DIM), self.q_norm, EPS, True).reshape(1, s, N_HEADS, HEAD_DIM)
            k = rms(k.reshape(1, s * N_KV, HEAD_DIM), self.k_norm, EPS, True).reshape(1, s, N_KV, HEAD_DIM)
            q = q.transpose(1, 2)                                        # [1,32,S,128]
            k = k.transpose(1, 2)                                        # [1,8,S,128]
        else:
            # each norm, then its transpose: the op order of the published graphs (the same math in any order)
            q = rms(q, self.q_norm, EPS, False).transpose(1, 2)          # [1,32,S,128]
            k = rms(k, self.k_norm, EPS, False).transpose(1, 2)          # [1,8,S,128]
        v = v.transpose(1, 2)                                            # [1,8,S,128]
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        qg = q.reshape(1, N_KV, (N_HEADS // N_KV) * s, HEAD_DIM)         # [1,8,4S,128]
        w = torch.einsum("bgqd,bgkd->bgqk", qg, k) + mask_g              # [1,8,4S,S]
        a = torch.softmax(w, dim=-1)
        o = torch.einsum("bgqk,bgkd->bgqd", a, v)                        # [1,8,4S,128]
        o = o.reshape(1, N_HEADS, s, HEAD_DIM).transpose(1, 2).reshape(1, s, N_HEADS * HEAD_DIM)
        h = h + F.linear(o, self.o_proj)
        x = rms(h, self.ln_post, EPS, self.safe)
        h = h + F.linear(F.silu(F.linear(x, self.gate)) * F.linear(x, self.up), self.down)
        return h

    def _fc4(self, x, w):
        """FULLY_CONNECTED on a rank-4 activation [1, 1, S, C] (LiteRT #9277: rank-3 int8 FC -> ADD)."""
        return F.linear(x.reshape(1, 1, x.shape[-2], x.shape[-1]), w)

    def _forward_rank4(self, h, mask_g, cos, sin):
        """Same math as forward(); every FC input and both residual ADDs are rank 4."""
        s = h.shape[1]
        x = rms(h, self.ln_in, EPS, self.safe)
        q = self._fc4(x, self.q_proj).reshape(1, s, N_HEADS, HEAD_DIM)
        k = self._fc4(x, self.k_proj).reshape(1, s, N_KV, HEAD_DIM)
        v = self._fc4(x, self.v_proj).reshape(1, s, N_KV, HEAD_DIM)
        if self.safe:
            q = rms(q.reshape(1, s * N_HEADS, HEAD_DIM), self.q_norm, EPS, True).reshape(1, s, N_HEADS, HEAD_DIM)
            k = rms(k.reshape(1, s * N_KV, HEAD_DIM), self.k_norm, EPS, True).reshape(1, s, N_KV, HEAD_DIM)
        else:
            q = rms(q, self.q_norm, EPS, False)
            k = rms(k, self.k_norm, EPS, False)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        qg = q.reshape(1, N_KV, (N_HEADS // N_KV) * s, HEAD_DIM)
        w = torch.einsum("bgqd,bgkd->bgqk", qg, k) + mask_g
        a = torch.softmax(w, dim=-1)
        o = torch.einsum("bgqk,bgkd->bgqd", a, v)
        o = o.reshape(1, N_HEADS, s, HEAD_DIM).transpose(1, 2).reshape(1, s, N_HEADS * HEAD_DIM)
        h4 = h.reshape(1, 1, s, HIDDEN) + self._fc4(o, self.o_proj)
        x = rms(h4.reshape(1, s, HIDDEN), self.ln_post, EPS, self.safe)
        m = F.silu(self._fc4(x, self.gate)) * self._fc4(x, self.up)      # [1,1,S,9728]
        h4 = h4 + self._fc4(m, self.down)
        return h4.reshape(1, s, HIDDEN)


class TEChunk(nn.Module):
    """A run of decoder layers; output = the hidden state after the last one."""

    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, hidden, mask, cos, sin):
        s = hidden.shape[1]
        mask_g = mask.reshape(1, N_KV, (N_HEADS // N_KV) * s, s)
        for layer in self.layers:
            hidden = layer(hidden, mask_g, cos, sin)
        return hidden


def build_chunk(i, safe=False, rank4=False):
    start, end = CHUNKS[i]
    return TEChunk([TELayer(layer_weights(j), safe=safe, rank4=rank4) for j in range(start, end)]).eval()


# ------------------------------------------------------------- host inputs
def host_mask(real_len, s=SEQ, neg=MASK_NEG):
    """[1, 32, s, s] float32: 0 where key <= query and key < real_len, else neg."""
    q = np.arange(s)[:, None]
    k = np.arange(s)[None, :]
    plane = np.where((k <= q) & (k < real_len), 0.0, neg).astype(np.float32)
    return np.ascontiguousarray(np.broadcast_to(plane, (1, N_HEADS, s, s)))


def host_rope(s=SEQ):
    """1D rotary tables [1, 1, s, 128] float32 (rotate_half layout, theta 5e6)."""
    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.int64).float() / HEAD_DIM))
    pos = torch.arange(s, dtype=torch.float32)
    freqs = pos[:, None] * inv_freq[None, :]                              # [s, 64]
    emb = torch.cat((freqs, freqs), dim=-1)                               # [s, 128]
    return emb.cos()[None, None].contiguous(), emb.sin()[None, None].contiguous()


def embed_lookup(ids, path):
    """inputs_embeds [1, len(ids), 2560] float32 from the host bf16 table at `path` (exact)."""
    table = np.memmap(path, dtype="<u2", mode="r", shape=(VOCAB_ROWS, HIDDEN))
    rows = np.asarray(table[np.asarray(ids, dtype=np.int64)], dtype=np.uint32) << 16
    return rows.view(np.float32)[None].copy()


# ----------------------------------------------------------------- metrics
def real_rows(attention_mask):
    """Indices (0..333) of the 300-row window whose mask is 1 = caption + suffix."""
    m = np.asarray(attention_mask).reshape(-1)
    return np.array([r for r in range(PREFIX_LEN, SEQ) if m[r] == 1], dtype=np.int64)


def compare(ref, cand):
    """ref / cand: [rows, C] arrays. corr, max|d|, norm ratio, min row cosine, max|ref|."""
    a = np.asarray(ref, np.float64)
    b = np.asarray(cand, np.float64)
    d = b - a
    num = (a * b).sum(-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    cos = num / np.maximum(den, 1e-300)
    corr = float(np.corrcoef(a.ravel(), b.ravel())[0, 1]) if a.size > 1 else float("nan")
    return {
        "corr": corr,
        "max_abs": float(np.abs(d).max()),
        "norm_ratio": float(np.linalg.norm(b) / np.linalg.norm(a)),
        "min_row_cos": float(cos.min()),
        "ref_max_abs": float(np.abs(a).max()),
        "finite": bool(np.isfinite(b).all()),
        "rows": int(a.shape[0]),
    }


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False))
    os.replace(tmp, path)


def sha256(path, chunk=1 << 24):
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()
