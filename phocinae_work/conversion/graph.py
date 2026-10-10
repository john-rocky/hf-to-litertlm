"""The static-window graph of Phocinae-Largha-150M-v1 for LiteRT, in the laya contract of hfmodels-android's
`EncoderDecisions` (family `laya`): mmBERT-small encoder + the 2-layer typed-decision head + the scorer at every
position; the host gathers the option markers, computes the four act features and runs the act head graph.

    main graph  input_ids [1,L] int32 (or inputs_embeds [1,L,384] float32 when host_lookup), attention_mask [1,L] float32,
                qtype_onehot [1,3] float32  ->  token_logits [1,L] float32, pooled_cls [1,384] float32
    act head    pooled_cls [1,384], feats [1,4]  ->  act_logits [1,2]

Built from the publisher's own modules (phocinae/model.py `DecisionModel`, loaded by `Engine`), fp32 throughout:
  * RoPE tables are the author's (inv_freq = theta^-(2i/64), cos/sin of outer(pos, inv_freq) twice) as constants;
  * attention is q k^T / 8 + bias, softmax, v (SDPA written out); the bias is additive: NEG (-1e4, fp16-safe) on
    padded keys and outside the +-64 band of the sliding layers, 0 on the diagonal of every row (a padded query row
    then attends to itself instead of to nothing: no empty softmax row, no NaN; valid rows are unchanged because a
    valid query's own key is already allowed) — memory sliding-window-empty-row-nan;
  * the head's nn.TransformerEncoderLayer (pre-norm, ReLU, 6 heads, ff 1536) is written out the same way
    (ExplicitHeadLayer), with the key padding as the same additive bias;
  * type embedding = qtype_onehot @ type_emb.weight (the SDK's one-hot input), added to every position;
  * optional fp16 headroom rewrites (exact in fp32): LayerNorm on x * 2^-k with eps * 2^-2k (`ln_shift`), applied
    only when the headroom scan (torch_parity.py --scan) says a norm input leaves the float16 range.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

NEG = -1.0e4
HIDDEN = 384
HEADS = 6
HEAD_DIM = 64


def rotate_half(x):
    d = x.shape[-1] // 2
    return torch.cat([-x[..., d:], x[..., :d]], dim=-1)


def layer_norm_shifted(x, norm, k, taps=None, name=None, valid=None):
    """LayerNorm(x) computed on x * 2^-k with eps * 2^-2k: the same value in fp32, smaller squares in fp16.
    taps (a dict) records the fp16-relevant maxima of this norm's input at the valid positions: |x|, (x-mean)^2 and
    the sum of squares the converter's reduction sees."""
    if taps is not None:
        xv = x[0, : int(valid)] if valid is not None else x[0]
        c = xv - xv.mean(-1, keepdim=True)
        rec(taps, f"ln:{name}:abs", xv.abs().max()); rec(taps, f"ln:{name}:sq", (c * c).max()); rec(taps, f"ln:{name}:sumsq", (c * c).sum(-1).max())
        rec(taps, f"ln:{name}:sumsq_raw", (xv * xv).sum(-1).max())
    if not k:
        return norm(x)
    s = 2.0 ** (-k)
    return F.layer_norm(x * s, norm.normalized_shape, norm.weight, norm.bias, norm.eps * s * s)


def rec(taps, key, value):
    v = float(value)
    if v > taps.get(key, float("-inf")):
        taps[key] = v


class ExplicitHeadLayer(nn.Module):
    """nn.TransformerEncoderLayer(norm_first=True, activation=relu, batch_first=True) without the fused eval path."""

    def __init__(self, layer, L, shift=0):
        super().__init__()
        assert layer.norm_first and layer.activation is F.relu
        sa = layer.self_attn
        assert sa.batch_first and sa._qkv_same_embed_dim and sa.embed_dim == HIDDEN and sa.num_heads == HEADS
        self.L = L
        self.shift = shift
        self.in_proj_weight = sa.in_proj_weight
        self.in_proj_bias = sa.in_proj_bias
        self.out_proj = sa.out_proj
        self.norm1, self.norm2 = layer.norm1, layer.norm2
        self.linear1, self.linear2 = layer.linear1, layer.linear2

    def forward(self, h, key_bias, taps=None, valid=None, name="head"):
        x = layer_norm_shifted(h, self.norm1, self.shift, taps, name + ".norm1", valid)
        qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias)
        q = qkv[:, :, :HIDDEN].reshape(1, self.L, HEADS, HEAD_DIM).transpose(1, 2)
        k = qkv[:, :, HIDDEN:2 * HIDDEN].reshape(1, self.L, HEADS, HEAD_DIM).transpose(1, 2)
        v = qkv[:, :, 2 * HIDDEN:].reshape(1, self.L, HEADS, HEAD_DIM).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(HEAD_DIM)) + key_bias
        if taps is not None:
            rec(taps, f"scores:{name}", scores[0, :, : int(valid), : int(valid)].abs().max())
        x = torch.matmul(torch.softmax(scores, dim=-1), v).transpose(1, 2).reshape(1, self.L, HIDDEN)
        h = h + self.out_proj(x)
        if taps is not None:
            rec(taps, f"residual:{name}", h[0, : int(valid)].abs().max())
        ff = F.relu(self.linear1(layer_norm_shifted(h, self.norm2, self.shift, taps, name + ".norm2", valid)))
        if taps is not None:
            rec(taps, f"ff:{name}", ff[0, : int(valid)].abs().max())
        return h + self.linear2(ff)


class PhocinaeGraph(nn.Module):
    def __init__(self, dm, L, host_lookup=False, neg=NEG, ln_shift=None):
        """dm = the publisher's DecisionModel (with .encoder); ln_shift = {name: k} for the fp16 rewrite (None = plain)."""
        super().__init__()
        enc = dm.encoder
        self.L = L
        self.host_lookup = host_lookup
        self.neg = float(neg)
        self.shift = dict(ln_shift or {})
        self.tok_embeddings = enc.embeddings.tok_embeddings
        self.emb_norm = enc.embeddings.norm
        self.layers = enc.layers
        self.full = [bool(l.full) for l in enc.layers]
        self.final_norm = enc.final_norm
        self.type_weight = dm.type_emb.weight
        self.head = nn.ModuleList([ExplicitHeadLayer(l, L, self.shift.get(f"head{i}", 0)) for i, l in enumerate(dm.head.layers)])
        self.scorer = dm.scorer
        self.taps = None          # set to a dict to record fp16 headroom maxima (never during export)
        self.valid = None         # number of valid positions of the current row (taps only)
        local = int(enc.local_window)
        assert local == 64 and len(enc.layers) == 22 and enc.hd == HEAD_DIM
        with torch.no_grad():
            pos = torch.arange(L, dtype=torch.float32)
            freqs = torch.outer(pos, enc.inv_freq.float())
            emb = torch.cat([freqs, freqs], dim=-1)
            self.register_buffer("cos", emb.cos()[None, None].contiguous().clone())      # [1,1,L,64]
            self.register_buffer("sin", emb.sin()[None, None].contiguous().clone())
            idx = torch.arange(L)
            eye = (idx[:, None] == idx[None, :]).float()
            self.register_buffer("not_eye", (1.0 - eye)[None, None].contiguous().clone())                      # [1,1,L,L]
            band = (idx[:, None] - idx[None, :]).abs() <= local
            self.register_buffer("band_bias", torch.where(band, 0.0, self.neg)[None, None].contiguous().clone())  # [1,1,L,L]

    def attention(self, q, k, v, bias, name=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(HEAD_DIM)) + bias
        if self.taps is not None:
            rec(self.taps, f"scores:{name}", scores[0, :, : int(self.valid), : int(self.valid)].abs().max())
        return torch.matmul(torch.softmax(scores, dim=-1), v)

    def encoder_layer(self, i, layer, h, full_bias, slide_bias):
        if isinstance(layer.attn_norm, nn.Identity):
            x = h
        else:
            x = layer_norm_shifted(h, layer.attn_norm, self.shift.get(f"attn{i}", 0), self.taps, f"attn{i}", self.valid)
        qkv = layer.attn.Wqkv(x)
        q = qkv[:, :, :HIDDEN].reshape(1, self.L, HEADS, HEAD_DIM).transpose(1, 2)
        k = qkv[:, :, HIDDEN:2 * HIDDEN].reshape(1, self.L, HEADS, HEAD_DIM).transpose(1, 2)
        v = qkv[:, :, 2 * HIDDEN:].reshape(1, self.L, HEADS, HEAD_DIM).transpose(1, 2)
        q = q * self.cos + rotate_half(q) * self.sin
        k = k * self.cos + rotate_half(k) * self.sin
        o = self.attention(q, k, v, full_bias if self.full[i] else slide_bias, f"enc{i}").transpose(1, 2).reshape(1, self.L, HIDDEN)
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

    def body(self, x, attention_mask, qtype_onehot):
        key_bias = (1.0 - attention_mask.reshape(1, 1, 1, self.L)) * self.neg       # NEG on padded keys
        full_bias = key_bias * self.not_eye                                           # 0 on every diagonal
        slide_bias = full_bias + self.band_bias
        h = layer_norm_shifted(x, self.emb_norm, self.shift.get("emb", 0), self.taps, "emb", self.valid)
        for i, layer in enumerate(self.layers):
            h = self.encoder_layer(i, layer, h, full_bias, slide_bias)
        h = layer_norm_shifted(h, self.final_norm, self.shift.get("final", 0), self.taps, "final", self.valid)
        h = h + torch.matmul(qtype_onehot, self.type_weight).reshape(1, 1, HIDDEN)
        for j, layer in enumerate(self.head):
            h = layer(h, full_bias, self.taps, self.valid, f"head{j}")
        return h

    def outputs(self, h):
        s = self.scorer
        x = layer_norm_shifted(h, s[0], self.shift.get("scorer", 0), self.taps, "scorer", self.valid)
        logits = s[3](s[2](s[1](x))).squeeze(-1)
        return {"token_logits": logits, "pooled_cls": h[:, 0, :]}

    def forward(self, *args, **kw):
        if self.host_lookup:
            return self.forward_embeds(*args, **kw)
        return self.forward_ids(*args, **kw)

    def forward_ids(self, input_ids, attention_mask, qtype_onehot):
        return self.outputs(self.body(self.tok_embeddings(input_ids), attention_mask, qtype_onehot))

    def forward_embeds(self, inputs_embeds, attention_mask, qtype_onehot):
        return self.outputs(self.body(inputs_embeds, attention_mask, qtype_onehot))


class IdsGraph(nn.Module):
    """Export wrapper with the laya input names (input_ids)."""

    def __init__(self, g):
        super().__init__()
        self.g = g

    def forward(self, input_ids, attention_mask, qtype_onehot):
        return self.g.forward_ids(input_ids, attention_mask, qtype_onehot)


class EmbedsGraph(nn.Module):
    """Export wrapper with the laya host-lookup input names (inputs_embeds)."""

    def __init__(self, g):
        super().__init__()
        self.g = g

    def forward(self, inputs_embeds, attention_mask, qtype_onehot):
        return self.g.forward_embeds(inputs_embeds, attention_mask, qtype_onehot)


class ActHeadGraph(nn.Module):
    def __init__(self, dm):
        super().__init__()
        self.act_head = dm.act_head

    def forward(self, pooled_cls, feats):
        return {"act_logits": self.act_head(torch.cat([pooled_cls, feats], dim=-1))}


def load_decision_model(hf_dir):
    """The publisher's Engine on the CPU in fp32 (its own loader, vocab check, weights check), model only."""
    from phocinae.engine import Engine
    eng = Engine(str(hf_dir), device="cpu", warm=False)
    return eng.model.eval().requires_grad_(False), eng


def row_inputs(ids, L, pad_id=0, qtype=0, markers=None):
    n = len(ids)
    assert n <= L, (n, L)
    input_ids = torch.full((1, L), pad_id, dtype=torch.int32)
    input_ids[0, :n] = torch.tensor(ids, dtype=torch.int32)
    mask = torch.zeros((1, L), dtype=torch.float32)
    mask[0, :n] = 1.0
    onehot = torch.zeros((1, 3), dtype=torch.float32)
    onehot[0, qtype] = 1.0
    return input_ids, mask, onehot


def embed_rows(table_fp32, input_ids):
    """Host lookup: table rows for every position (padded positions read row pad_id, as the SDK's TokenTable does)."""
    return table_fp32[input_ids.long()]
