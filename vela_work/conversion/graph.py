"""The static-window graph of Vela-1.0-Encoder-307M-Embedding (ModernBERT, mmBERT-base shape) for LiteRT.

    ids graph     input_ids int32 [1,L], attention_mask f32 [1,L]            -> embedding f32 [1,768]
    embeds graph  inputs_embeds f32 [1,L,768], attention_mask f32 [1,L]      -> embedding f32 [1,768]   (host table lookup)
    engine graph  embeddings f32 [1,L,768], input_mask f32 [1,L]             -> embedding f32 [1,768]   (LiteRT-LM EmbeddingEngine
                  TF_LITE_TEXT_ENCODER names; the runtime fills only the first n rows of `embeddings`, the rest is stale)

Built from the Hub checkpoint's modules as transformers loads them (ModernBertModel: embeddings.tok_embeddings / norm,
layers[i].attn_norm (Identity at 0) / attn.Wqkv / attn.Wo / mlp_norm / mlp.Wi / mlp.Wo, final_norm), fp32 throughout:
  * RoPE cos / sin are constants per window, read from oracle/rope_L512.npz = the tables the publisher's stack (transformers
    4.57.6) computes for this config (rope_scaling yarn, factor 4, original 8192, attention_factor 1.0; both layer types use
    theta 160000) — never recomputed here, so the graph carries the reference's positions whatever the export venv's
    transformers does with `rope_scaling`;
  * attention is q k^T / 8 + bias, softmax, v (SDPA written out); the bias is additive, NEG = -1e4 (fp16-safe) on padded
    keys and outside the +-64 band of the sliding layers (local_attention 128), 0 on the diagonal of every row so a padded
    query attends to itself instead of to nothing (no empty softmax row, no NaN: memory sliding-window-empty-row-nan;
    granite_embed_work had 45-442 fully-masked rows at S=128-512). Valid rows are unchanged: a valid query's own key is
    already allowed;
  * the embeds / engine forms replace every padded (stale) input row with the pad token's own embedding row (table row 0,
    a constant), so the padded positions carry exactly what the ids form computes for them (a zero row would make the first
    LayerNorm's variance 0, and in fp16 storage eps * 2^-2k rounds to 0: rsqrt(0) = inf, 0 * inf = NaN);
  * mean pooling over the valid positions (sentence-transformers Pooling mean: sum(h * mask) / clamp(sum(mask), 1e-9)),
    written as torch.where so a stale NaN row of the engine form cannot poison the sum, then L2 normalization (Normalize);
  * optional fp16 headroom rewrite (exact in fp32): LayerNorm on x * 2^-k with eps * 2^-2k (`ln_shift`), applied only to
    the norms the headroom scan says leave float16 (oracle/fp16_headroom.json; phocinae_work: mmBERT layers 12-21 + final).
"""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

NEG = -1.0e4
HIDDEN = 768
HEADS = 12
HEAD_DIM = 64
LOCAL = 64          # local_attention 128 = +-64


def rotate_half(x):
    d = x.shape[-1] // 2
    return torch.cat([-x[..., d:], x[..., :d]], dim=-1)


def rec(taps, key, value):
    v = float(value)
    if v > taps.get(key, float("-inf")):
        taps[key] = v


def layer_norm_shifted(x, norm, k, taps=None, name=None, valid=None):
    """LayerNorm(x) computed on x * 2^-k with eps * 2^-2k: the same value in fp32, smaller squares in fp16 storage."""
    if taps is not None:   # taps measure the values the fp16 kernel sees: the input after the 2^-k scale
        xv = x[0, : int(valid)] if valid is not None else x[0]
        xv = xv * (2.0 ** (-k)) if k else xv
        c = xv - xv.mean(-1, keepdim=True)
        rec(taps, f"ln:{name}:abs", xv.abs().max()); rec(taps, f"ln:{name}:sq", (c * c).max()); rec(taps, f"ln:{name}:sumsq", (c * c).sum(-1).max())
        rec(taps, f"ln:{name}:sumsq_raw", (xv * xv).sum(-1).max())
    if not k:
        return F.layer_norm(x, norm.normalized_shape, norm.weight, norm.bias, norm.eps)
    s = 2.0 ** (-k)
    return F.layer_norm(x * s, norm.normalized_shape, norm.weight, norm.bias, norm.eps * s * s)


def parse_shift(spec):
    """'attn12-21=6,mlp12-21=6,final=6,emb=2' -> {attn12: 6, ..., final: 6}."""
    out = {}
    for part in (spec or "").split(","):
        if not part.strip():
            continue
        name, k = part.split("=")
        name, k = name.strip(), int(k)
        m = __import__("re").fullmatch(r"([a-z]+)(\d+)-(\d+)", name)
        if m:
            for i in range(int(m.group(2)), int(m.group(3)) + 1):
                out[f"{m.group(1)}{i}"] = k
        else:
            out[name] = k
    return out


class VelaGraph(nn.Module):
    def __init__(self, body, L, rope, ln_shift=None, neg=NEG, pool_mode="where"):
        """body = transformers ModernBertModel (fp32, eval); rope = {layer_type: (cos [>=L,64], sin)} float32 tables."""
        super().__init__()
        cfg = body.config
        assert cfg.hidden_size == HIDDEN and cfg.num_attention_heads == HEADS and cfg.local_attention == 2 * LOCAL and cfg.num_hidden_layers == 22, cfg
        self.L = L
        self.neg = float(neg)
        self.pool_mode = pool_mode     # "where" (torch.where + F.normalize) | "arith" (MUL / SUM / MAXIMUM / RSQRT: the GPU delegate has no broadcast SELECT_V2)
        assert self.pool_mode in ("where", "arith"), pool_mode
        self.shift = dict(ln_shift or {})
        self.tok_embeddings = body.embeddings.tok_embeddings
        self.emb_norm = body.embeddings.norm
        self.layers = body.layers
        self.layer_types = list(cfg.layer_types)
        self.final_norm = body.final_norm
        self.taps = None
        self.valid = None
        with torch.no_grad():
            self.register_buffer("pad_row", self.tok_embeddings.weight[0].detach().float().reshape(1, 1, HIDDEN).contiguous().clone())
            for lt in ("full_attention", "sliding_attention"):
                cos, sin = rope[lt]
                assert cos.shape[0] >= L and cos.shape[1] == HEAD_DIM, cos.shape
                self.register_buffer(f"cos_{lt}", torch.as_tensor(np.ascontiguousarray(cos[:L]), dtype=torch.float32)[None, None].contiguous().clone())
                self.register_buffer(f"sin_{lt}", torch.as_tensor(np.ascontiguousarray(sin[:L]), dtype=torch.float32)[None, None].contiguous().clone())
            idx = torch.arange(L)
            eye = (idx[:, None] == idx[None, :]).float()
            self.register_buffer("not_eye", (1.0 - eye)[None, None].contiguous().clone())                                  # [1,1,L,L]
            band = (idx[:, None] - idx[None, :]).abs() <= LOCAL
            self.register_buffer("band_bias", torch.where(band, 0.0, self.neg)[None, None].contiguous().clone())          # [1,1,L,L]

    def attention(self, q, k, v, bias, name=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(HEAD_DIM)) + bias
        if self.taps is not None:
            rec(self.taps, f"scores:{name}", scores[0, :, : int(self.valid), : int(self.valid)].abs().max())
        return torch.matmul(torch.softmax(scores, dim=-1), v)

    def encoder_layer(self, i, layer, h, full_bias, slide_bias):
        lt = self.layer_types[i]
        if isinstance(layer.attn_norm, nn.Identity):
            x = h
        else:
            x = layer_norm_shifted(h, layer.attn_norm, self.shift.get(f"attn{i}", 0), self.taps, f"attn{i}", self.valid)
        qkv = layer.attn.Wqkv(x)
        q = qkv[:, :, :HIDDEN].reshape(1, self.L, HEADS, HEAD_DIM).transpose(1, 2)
        k = qkv[:, :, HIDDEN:2 * HIDDEN].reshape(1, self.L, HEADS, HEAD_DIM).transpose(1, 2)
        v = qkv[:, :, 2 * HIDDEN:].reshape(1, self.L, HEADS, HEAD_DIM).transpose(1, 2)
        cos, sin = getattr(self, f"cos_{lt}"), getattr(self, f"sin_{lt}")
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        bias = full_bias if lt == "full_attention" else slide_bias
        o = self.attention(q, k, v, bias, f"enc{i}").transpose(1, 2).reshape(1, self.L, HIDDEN)
        h = h + layer.attn.Wo(o)
        if self.taps is not None:
            rec(self.taps, f"residual:enc{i}.attn", h[0, : int(self.valid)].abs().max())
        x = layer_norm_shifted(h, layer.mlp_norm, self.shift.get(f"mlp{i}", 0), self.taps, f"mlp{i}", self.valid)
        inp, gate = layer.mlp.Wi(x).chunk(2, dim=-1)
        prod = F.gelu(inp) * gate
        if self.taps is not None:
            rec(self.taps, f"geglu:enc{i}", prod[0, : int(self.valid)].abs().max())
            rec(self.taps, f"wi:enc{i}", torch.maximum(inp.abs().max(), gate.abs().max()))
        h = h + layer.mlp.Wo(prod)
        if self.taps is not None:
            rec(self.taps, f"residual:enc{i}.mlp", h[0, : int(self.valid)].abs().max())
        return h

    def body(self, x, attention_mask):
        mask = attention_mask.reshape(1, 1, 1, self.L)
        key_bias = (1.0 - mask) * self.neg                       # NEG on padded keys
        full_bias = key_bias * self.not_eye                      # 0 on every diagonal
        slide_bias = full_bias + self.band_bias
        h = layer_norm_shifted(x, self.emb_norm, self.shift.get("emb", 0), self.taps, "emb", self.valid)
        for i, layer in enumerate(self.layers):
            h = self.encoder_layer(i, layer, h, full_bias, slide_bias)
        h = layer_norm_shifted(h, self.final_norm, self.shift.get("final", 0), self.taps, "final", self.valid)
        return h

    def pool(self, h, attention_mask):
        if self.pool_mode == "arith":
            # GPU-friendly: MUL / SUM / MAXIMUM / RSQRT only (no SELECT_V2, GREATER, POW, DIV). The mask is exactly 0 / 1, so
            # h * m == where(m, h, 0) bit for bit on finite h; x * rsqrt(max(sum x², eps²)) == F.normalize up to one rounding.
            m = attention_mask.reshape(1, self.L, 1)
            pooled = (h * m).sum(dim=1) * (1.0 / torch.clamp(m.sum(dim=1), min=1e-9))
            return pooled * torch.rsqrt(torch.clamp((pooled * pooled).sum(dim=-1, keepdim=True), min=1e-24))
        valid = attention_mask.reshape(1, self.L, 1) > 0.5
        hz = torch.where(valid, h, torch.zeros((), dtype=h.dtype))
        pooled = hz.sum(dim=1) / attention_mask.reshape(1, self.L).sum(dim=1, keepdim=True).clamp(min=1e-9)
        return F.normalize(pooled, p=2, dim=-1)

    def forward_ids(self, input_ids, attention_mask):
        return self.pool(self.body(self.tok_embeddings(input_ids), attention_mask), attention_mask)

    def forward_embeds(self, inputs_embeds, attention_mask):
        if self.pool_mode == "arith":
            m = attention_mask.reshape(1, self.L, 1)
            x = inputs_embeds * m + self.pad_row * (1.0 - m)      # pad rows -> the pad token's row; the host writes finite (zero) pad rows
        else:
            valid = attention_mask.reshape(1, self.L, 1) > 0.5
            x = torch.where(valid, inputs_embeds, self.pad_row)  # stale / garbage pad rows -> the pad token's row (= the ids form)
        return self.pool(self.body(x, attention_mask), attention_mask)


class IdsGraph(nn.Module):
    def __init__(self, g):
        super().__init__()
        self.g = g

    def forward(self, input_ids, attention_mask):
        return {"embedding": self.g.forward_ids(input_ids, attention_mask)}


class EmbedsGraph(nn.Module):
    def __init__(self, g):
        super().__init__()
        self.g = g

    def forward(self, inputs_embeds, attention_mask):
        return {"embedding": self.g.forward_embeds(inputs_embeds, attention_mask)}


class EngineEncoderGraph(nn.Module):
    """LiteRT-LM EmbeddingEngine TF_LITE_TEXT_ENCODER input names (`embeddings`, `input_mask`); output 0 = the vector."""

    def __init__(self, g):
        super().__init__()
        self.g = g

    def forward(self, embeddings, input_mask):
        return {"encodings": self.g.forward_embeds(embeddings, input_mask)}


class LookupGraph(nn.Module):
    """TF_LITE_EMBEDDER: token int32[1] -> raw table row f32 [1,1,768]."""

    def __init__(self, body):
        super().__init__()
        self.table = body.embeddings.tok_embeddings

    def forward(self, token):
        return {"embedding": self.table(token)[None]}


def load_body(hf_dir):
    from transformers import AutoModel
    m = AutoModel.from_pretrained(str(hf_dir), dtype=torch.float32, attn_implementation="sdpa").eval().requires_grad_(False)
    assert type(m).__name__ == "ModernBertModel", type(m).__name__
    return m


def load_rope(path):
    z = np.load(path)
    return {lt: (z[f"{lt}_cos"], z[f"{lt}_sin"]) for lt in ("full_attention", "sliding_attention")}


def row_inputs(ids, L, pad_id=0):
    n = len(ids)
    assert n <= L, (n, L)
    input_ids = torch.full((1, L), pad_id, dtype=torch.int32)
    input_ids[0, :n] = torch.tensor(ids, dtype=torch.int32)
    mask = torch.zeros((1, L), dtype=torch.float32)
    mask[0, :n] = 1.0
    return input_ids, mask


def embed_rows(table_f32, input_ids):
    """Host lookup: the table row of every position (padded positions read row pad_id = 0, as the SDK's TokenTable does)."""
    return table_f32[input_ids.long()]
