"""GPU-clean re-authoring of the Iris-3B Turbo DiT for LiteRT CompiledModel.

The upstream `IrisDiT` (upstream/src/iris3b/models/dit.py, rev 13fcf6a4) is
rewritten as chunk modules whose ops are rank <= 4 tensor ops the LiteRT GPU
delegate (ML Drift) takes. Every rewrite is exact in real arithmetic:

  * RoPE: the interleaved (view_as_complex) rotation becomes a half-split
    rotation; the even/odd de-interleave is baked into the q/k projection rows
    and the q/k norm weights (q.k is invariant to a shared permutation).
  * GQA (20 query / 5 KV heads): the query heads are reordered (r, g) so each of
    the 4 chains is a plain 5-head attention against the 5 KV heads; the reorder
    is baked into the q rows, the gate rows and the output-projection columns.
    No K/V repeat, no BROADCAST_TO, no rank 5.
  * Attention products are einsums at rank 4 with a non-unit head axis; the
    1/sqrt(head_dim) scale is folded into the q norm weight (or into the q rows
    when there is no q norm).
  * RMSNorm is max-normalized (keeps fp16 sums of squares in range).
  * The adapter refiner's key mask is an additive [1, H, T, T] input.
  * unfold / fold / PixelEmbedder grouping are host permutations into one
    patch-major layout [1, L, p*p, 3]; the s_embedder columns are permuted to
    read that layout.
  * Everything that depends on t only (timestep MLP, the shared adaLN cores and
    the per-block biases, 1 + scale) is precomputed on the host per sampler
    step: the Turbo sigmas are fixed by distill.yaml.
  * RoPE cos / sin tables and the pixel sincos table are graph inputs.
"""

import torch
import torch.nn.functional as F
from torch import nn

MASK_NEG = -30000.0  # additive key mask, finite in fp16


# ----------------------------------------------------------------- primitives
def frozen(t):
    """A non-trainable parameter that shares storage with `t` when it can."""
    return nn.Parameter(t.detach(), requires_grad=False)


class Lin(nn.Module):
    """y = x W^T + b with a (possibly permuted) frozen weight."""

    def __init__(self, weight, bias=None):
        super().__init__()
        self.weight = frozen(weight)
        self.bias = None if bias is None else frozen(bias)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


def safe_rms(x, weight, eps):
    """RMSNorm over the last dim of a rank-3 tensor, max-normalized.

    x / sqrt(mean(x^2) + eps) == xs / sqrt(mean(xs^2) + eps / m^2) with
    xs = x / m, m = max(1, max|x|): exact, and every sum stays <= the row width.
    Rank 3 because litert-torch's NHWC pass has no rewriter for a rank-4 amax.
    """
    m = x.abs().amax(dim=-1, keepdim=True).clamp_min(1.0)
    xs = x / m
    ms = (xs * xs).mean(dim=-1, keepdim=True)
    return xs * torch.rsqrt(ms + eps / (m * m)) * weight


def safe_rms4(x, weight, eps):
    """safe_rms over the last dim of a rank-4 [1, A, B, C] tensor."""
    b, s, h, d = x.shape
    return safe_rms(x.reshape(b, s * h, d), weight, eps).reshape(b, s, h, d)


def rope_half(x, cos, sin):
    """Half-split rotation. x: [1, S, H, D] with each head stored [evens | odds];
    cos / sin: [1, S, 1, D/2]."""
    half = x.shape[-1] // 2
    a = x[..., :half]
    b = x[..., half:]
    return torch.cat([a * cos - b * sin, b * cos + a * sin], dim=-1)


def attention(q, k, v, chain, bias=None):
    """Softmax attention on [1, S, H, D] tensors (q already scaled).

    Query head r * Hkv + g reads KV head g, so q may carry `rep` times the KV
    heads; the heads run in chains of `chain` (a divisor of Hkv) so every
    product stays a rank-4 BATCH_MATMUL with a non-unit head axis.
    bias: optional additive [1, Hq, S, S] mask.
    Returns [1, S, Hq * D] with heads in q's order.
    """
    s, hq, d = q.shape[1], q.shape[2], q.shape[3]
    hkv = k.shape[2]
    qh = q.permute(0, 2, 1, 3)
    kh = k.permute(0, 2, 1, 3)
    vh = v.permute(0, 2, 1, 3)
    outs = []
    for h0 in range(0, hq, chain):
        g0 = h0 % hkv
        scores = torch.einsum("bhqd,bhkd->bhqk", qh[:, h0:h0 + chain], kh[:, g0:g0 + chain])
        if bias is not None:
            scores = scores + bias[:, h0:h0 + chain]
        p = torch.softmax(scores, dim=-1)
        outs.append(torch.einsum("bhqk,bhkd->bhqd", p, vh[:, g0:g0 + chain]))
    o = outs[0] if len(outs) == 1 else torch.cat(outs, dim=1)
    return o.permute(0, 2, 1, 3).reshape(1, s, hq * d)


def even_odd(d):
    return list(range(0, d, 2)) + list(range(1, d, 2))


def gqa_order(n_heads, n_kv):
    """Position r * n_kv + g holds the original query head g * rep + r."""
    rep = n_heads // n_kv
    return [g * rep + r for r in range(rep) for g in range(n_kv)]


def rows(head_order, within):
    """Row indices of a [heads * d, in] weight: head blocks in `head_order`,
    rows inside a head in `within` order."""
    d = len(within)
    return torch.tensor([h * d + i for h in head_order for i in within], dtype=torch.long)


class SwiGLU(nn.Module):
    def __init__(self, mlp):
        super().__init__()
        self.w1 = Lin(mlp.w1.weight)
        self.w3 = Lin(mlp.w3.weight)
        self.w2 = Lin(mlp.w2.weight)

    def forward(self, x):
        a = self.w1(x)
        return self.w2(a * torch.sigmoid(a) * self.w3(x))


# ------------------------------------------------------------- trunk blocks
class DualBlock(nn.Module):
    """MMDiTBlock (dual stream, joint attention, text first), GPU-clean."""

    def __init__(self, blk):
        super().__init__()
        a = blk.attn
        self.nh, self.nkv, self.hd = a.num_heads, a.num_kv_heads, a.head_dim
        self.eps = blk.norm_x1.eps
        eo = even_odd(self.hd)
        order = gqa_order(self.nh, self.nkv)
        q_rows = rows(order, eo)
        k_rows = rows(range(self.nkv), eo)
        o_cols = rows(order, range(self.hd))
        scale = self.hd ** -0.5
        eo_t = torch.tensor(eo)
        for s in ("x", "y"):
            setattr(self, f"q_{s}", Lin(getattr(a, f"q_proj_{s}").weight[q_rows]))
            setattr(self, f"k_{s}", Lin(getattr(a, f"k_proj_{s}").weight[k_rows]))
            setattr(self, f"v_{s}", Lin(getattr(a, f"v_proj_{s}").weight))
            setattr(self, f"qn_{s}", frozen(getattr(a, f"q_norm_{s}").weight[eo_t] * scale))
            setattr(self, f"kn_{s}", frozen(getattr(a, f"k_norm_{s}").weight[eo_t]))
            setattr(self, f"gate_{s}", Lin(getattr(blk, f"attn_gate_{s}").weight[o_cols]))
            proj = getattr(a, f"proj_{s}")
            setattr(self, f"proj_{s}", Lin(proj.weight[:, o_cols], proj.bias))
            setattr(self, f"mlp_{s}", SwiGLU(getattr(blk, f"mlp_{s}")))
            for n in ("1", "2"):
                setattr(self, f"n_{s}{n}", frozen(getattr(blk, f"norm_{s}{n}").weight))
            setattr(self, f"apn_{s}", frozen(getattr(blk, f"attn_post_norm_{s}").weight))
            setattr(self, f"mpn_{s}", frozen(getattr(blk, f"mlp_post_norm_{s}").weight))

    def forward(self, x, y, cos, sin, mod):
        """mod: [1, 12, D] = img (shift1, 1+scale1, gate1, shift2, 1+scale2, gate2)
        then the same six rows for text."""
        e = self.eps
        xs1, xc1, xg1, xs2, xc2, xg2 = (mod[:, i:i + 1] for i in range(6))
        ys1, yc1, yg1, ys2, yc2, yg2 = (mod[:, i:i + 1] for i in range(6, 12))
        nx, ny = x.shape[1], y.shape[1]
        hx = safe_rms(x, self.n_x1, e) * xc1 + xs1
        hy = safe_rms(y, self.n_y1, e) * yc1 + ys1
        qx = safe_rms4(self.q_x(hx).reshape(1, nx, self.nh, self.hd), self.qn_x, e)
        kx = safe_rms4(self.k_x(hx).reshape(1, nx, self.nkv, self.hd), self.kn_x, e)
        qy = safe_rms4(self.q_y(hy).reshape(1, ny, self.nh, self.hd), self.qn_y, e)
        ky = safe_rms4(self.k_y(hy).reshape(1, ny, self.nkv, self.hd), self.kn_y, e)
        q = rope_half(torch.cat([qy, qx], dim=1), cos, sin)
        k = rope_half(torch.cat([ky, kx], dim=1), cos, sin)
        v = torch.cat([self.v_y(hy).reshape(1, ny, self.nkv, self.hd),
                       self.v_x(hx).reshape(1, nx, self.nkv, self.hd)], dim=1)
        o = attention(q, k, v, chain=self.nkv)
        ox = o[:, ny:] * torch.sigmoid(self.gate_x(hx))
        oy = o[:, :ny] * torch.sigmoid(self.gate_y(hy))
        x = x + xg1 * safe_rms(self.proj_x(ox), self.apn_x, e)
        x = x + xg2 * safe_rms(self.mlp_x(safe_rms(x, self.n_x2, e) * xc2 + xs2), self.mpn_x, e)
        y = y + yg1 * safe_rms(self.proj_y(oy), self.apn_y, e)
        y = y + yg2 * safe_rms(self.mlp_y(safe_rms(y, self.n_y2, e) * yc2 + ys2), self.mpn_y, e)
        return x, y


class SingleBlock(nn.Module):
    """SingleStreamBlock over the joint [text; image] sequence, GPU-clean."""

    def __init__(self, blk):
        super().__init__()
        self.nh, self.nkv, self.hd = blk.num_heads, blk.num_kv_heads, blk.head_dim
        self.eps = blk.norm1.eps
        eo = even_odd(self.hd)
        order = gqa_order(self.nh, self.nkv)
        o_cols = rows(order, range(self.hd))
        eo_t = torch.tensor(eo)
        self.q = Lin(blk.q_proj.weight[rows(order, eo)])
        self.k = Lin(blk.k_proj.weight[rows(range(self.nkv), eo)])
        self.v = Lin(blk.v_proj.weight)
        self.qn = frozen(blk.q_norm.weight[eo_t] * self.hd ** -0.5)
        self.kn = frozen(blk.k_norm.weight[eo_t])
        self.gate = Lin(blk.attn_gate.weight[o_cols])
        self.proj = Lin(blk.attn_proj.weight[:, o_cols], blk.attn_proj.bias)
        self.mlp = SwiGLU(blk.mlp)
        self.n1, self.n2 = frozen(blk.norm1.weight), frozen(blk.norm2.weight)
        self.apn, self.mpn = frozen(blk.attn_post_norm.weight), frozen(blk.mlp_post_norm.weight)

    def forward(self, j, cos, sin, mod):
        """mod: [1, 6, D] = (shift1, 1+scale1, gate1, shift2, 1+scale2, gate2)."""
        e = self.eps
        s1, c1, g1, s2, c2, g2 = (mod[:, i:i + 1] for i in range(6))
        n = j.shape[1]
        h = safe_rms(j, self.n1, e) * c1 + s1
        q = rope_half(safe_rms4(self.q(h).reshape(1, n, self.nh, self.hd), self.qn, e), cos, sin)
        k = rope_half(safe_rms4(self.k(h).reshape(1, n, self.nkv, self.hd), self.kn, e), cos, sin)
        v = self.v(h).reshape(1, n, self.nkv, self.hd)
        o = attention(q, k, v, chain=self.nkv) * torch.sigmoid(self.gate(h))
        j = j + g1 * safe_rms(self.proj(o), self.apn, e)
        return j + g2 * safe_rms(self.mlp(safe_rms(j, self.n2, e) * c2 + s2), self.mpn, e)


class DualChunk(nn.Module):
    """A run of dual-stream blocks; the first chunk also embeds the patches."""

    def __init__(self, blocks, s_embedder=None, patch=16, channels=3):
        super().__init__()
        self.blocks = nn.ModuleList(DualBlock(b) for b in blocks)
        self.embed = None
        if s_embedder is not None:
            self.embed = Lin(s_embedder.proj.weight[:, s_embed_perm(patch, channels)],
                             s_embedder.proj.bias)

    def forward(self, x, y, cos, sin, mod):
        """x: [1, 1, L, p*p*3] patch-major pixels for the first chunk (a rank-4 graph
        input, so the patch FC stays rank 4), else [1, L, D]."""
        if self.embed is not None:
            x = self.embed(x)
            x = x.reshape(1, x.shape[2], x.shape[3])
        for i, b in enumerate(self.blocks):
            x, y = b(x, y, cos, sin, mod[:, 12 * i:12 * (i + 1)])
        return x, y


class SingleChunk(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList(SingleBlock(b) for b in blocks)

    def forward(self, j, cos, sin, mod):
        for i, b in enumerate(self.blocks):
            j = b(j, cos, sin, mod[:, 6 * i:6 * (i + 1)])
        return j


# ------------------------------------------------------------- pixel stage
class PixelBlock(nn.Module):
    """PiTBlock, modulation "post": per-pixel adaLN, compacted global attention."""

    def __init__(self, blk):
        super().__init__()
        assert blk.modulation == "post"
        self.pp = blk.pixels_per_patch
        self.ch = blk.norm1.weight.shape[0]
        self.eps = blk.norm1.eps
        a = blk.attn
        self.nh, self.hd = a.num_heads, a.head_dim
        self.chain = self.nh // 2 if self.nh % 2 == 0 else self.nh
        # adaLN rows f = p * 4C + k: k in [0,C) scale1, [C,2C) shift1, [2C,3C) scale2,
        # [3C,4C) shift2. Fold the +1 of (1 + scale) into the scale biases.
        bias = blk.adaln.bias.detach().clone().reshape(self.pp, 4 * self.ch)
        bias[:, 0:self.ch] += 1.0
        bias[:, 2 * self.ch:3 * self.ch] += 1.0
        self.adaln = Lin(blk.adaln.weight, bias.reshape(-1))
        self.compress = Lin(blk.compress.weight, blk.compress.bias)
        self.expand = Lin(blk.expand.weight, blk.expand.bias)
        eo = even_odd(self.hd)
        dim = self.nh * self.hd
        w = a.qkv.weight
        self.qkv = Lin(torch.cat([w[:dim][rows(range(self.nh), eo)],
                                  w[dim:2 * dim][rows(range(self.nh), eo)],
                                  w[2 * dim:]], dim=0))
        eo_t = torch.tensor(eo)
        self.qn = frozen(a.q_norm.weight[eo_t] * self.hd ** -0.5)
        self.kn = frozen(a.k_norm.weight[eo_t])
        self.proj = Lin(a.proj.weight, a.proj.bias)
        self.fc1 = Lin(blk.mlp.fc1.weight, blk.mlp.fc1.bias)
        self.fc2 = Lin(blk.mlp.fc2.weight, blk.mlp.fc2.bias)
        self.n1, self.n2 = frozen(blk.norm1.weight), frozen(blk.norm2.weight)

    def forward(self, px, s, cos, sin):
        """px: [1, L, pp, C] pixels, s: [1, L, D] patch condition."""
        e, c, pp = self.eps, self.ch, self.pp
        n = px.shape[1]
        mods = self.adaln(s).reshape(1, n, pp, 4 * c)
        sc1, sh1 = mods[..., 0:c], mods[..., c:2 * c]
        sc2, sh2 = mods[..., 2 * c:3 * c], mods[..., 3 * c:4 * c]
        h = safe_rms(px.reshape(1, n * pp, c), self.n1, e)
        comp = self.compress(h.reshape(1, n, pp * c))
        qkv = self.qkv(comp)
        dim = self.nh * self.hd
        q = rope_half(safe_rms4(qkv[..., :dim].reshape(1, n, self.nh, self.hd), self.qn, e), cos, sin)
        k = rope_half(safe_rms4(qkv[..., dim:2 * dim].reshape(1, n, self.nh, self.hd), self.kn, e),
                      cos, sin)
        v = qkv[..., 2 * dim:].reshape(1, n, self.nh, self.hd)
        att = self.expand(self.proj(attention(q, k, v, chain=self.chain)))
        px = px + att.reshape(1, n, pp, c) * sc1 + sh1
        hm = safe_rms(px.reshape(1, n * pp, c), self.n2, e)
        m = self.fc2(F.gelu(self.fc1(hm)))
        return px + m.reshape(1, n, pp, c) * sc2 + sh2


class PixelGraph(nn.Module):
    """silu(t_emb + s) -> pixel embedder -> PiT blocks -> final layer."""

    def __init__(self, model):
        super().__init__()
        pe = model.pixel_embedder
        self.pe = Lin(pe.proj.weight, pe.proj.bias)
        self.blocks = nn.ModuleList(PixelBlock(b) for b in model.pixel_blocks)
        self.fn = frozen(model.final_layer.norm.weight)
        self.eps = model.final_layer.norm.eps
        self.fl = Lin(model.final_layer.linear.weight, model.final_layer.linear.bias)
        self.pp = model.pixel_blocks[0].pixels_per_patch
        self.ch = pe.proj.out_features
        self.cin = pe.proj.in_features

    def forward(self, s_img, t_emb, x_pix, pix_pos, cos, sin):
        """s_img [1, L, D]; t_emb [1, 1, D]; x_pix [1, L, pp, 3]; pix_pos [1, L, pp, C];
        cos / sin [1, L, 1, hd/2] -> v [1, L, pp, 3]."""
        n, pp, c = s_img.shape[1], self.pp, self.ch
        u = t_emb + s_img
        s = u * torch.sigmoid(u)
        px = self.pe(x_pix.reshape(1, n * pp, self.cin)).reshape(1, n, pp, c) + pix_pos
        for b in self.blocks:
            px = b(px, s, cos, sin)
        out = self.fl(safe_rms(px.reshape(1, n * pp, c), self.fn, self.eps))
        return out.reshape(1, n, pp, self.cin)


# ------------------------------------------------------------- text adapter
class LayerBlock(nn.Module):
    """LayerwiseAttentionBlock: attention over one token's encoder-layer taps.

    The residual stream is rank 4 [1, 1, T*Lyr, D], and an FC whose output
    feeds a residual ADD carries no bias (its bias is added to the residual
    first: (x + b) + W o): litert-torch emits a biased FC with a single ADD
    consumer as a rank-2 FC plus a mixed-rank ADD, which the int8 FC kernel of
    the GPU delegate refuses (LiteRT #9277 / #10445 class).
    """

    def __init__(self, blk, n_layers):
        super().__init__()
        a = blk.attn
        self.nh, self.hd, self.nl = a.num_heads, a.head_dim, n_layers
        self.eps = blk.norm1.eps
        dim = self.nh * self.hd
        w = a.qkv.weight.detach().clone()
        w[:dim] *= self.hd ** -0.5  # no q norm here: fold the softmax scale into the q rows
        self.qkv = Lin(w)
        self.proj = Lin(a.proj.weight)
        self.proj_b = frozen(a.proj.bias)
        self.fc1 = Lin(blk.mlp[0].weight, blk.mlp[0].bias)
        self.fc2 = Lin(blk.mlp[2].weight)
        self.fc2_b = frozen(blk.mlp[2].bias)
        self.n1, self.n2 = frozen(blk.norm1.weight), frozen(blk.norm2.weight)

    def _heads(self, t, n_tok):
        """[1, 1, T*Lyr, H*d] -> [1, T*H, d, Lyr]."""
        dim = self.nh * self.hd
        return (t.reshape(1, n_tok, self.nl, dim).permute(0, 1, 3, 2)
                .reshape(1, n_tok * self.nh, self.hd, self.nl))

    def forward(self, x):
        """x: [1, 1, T * Lyr, D] (token-major, Lyr taps per token)."""
        e = self.eps
        dim = self.nh * self.hd
        n = x.shape[2]
        n_tok = n // self.nl
        qkv = self.qkv(safe_rms(x.reshape(1, n, dim), self.n1, e).reshape(1, 1, n, dim))
        q = self._heads(qkv[..., :dim], n_tok)
        k = self._heads(qkv[..., dim:2 * dim], n_tok)
        v = self._heads(qkv[..., 2 * dim:], n_tok)
        p = torch.softmax(torch.einsum("bxdi,bxdj->bxij", q, k), dim=-1)
        o = torch.einsum("bxij,bxdj->bxdi", p, v)
        o = o.reshape(1, n_tok, dim, self.nl).permute(0, 1, 3, 2).reshape(1, 1, n, dim)
        x = (x + self.proj_b) + self.proj(o)
        h = self.fc1(safe_rms(x.reshape(1, n, dim), self.n2, e).reshape(1, 1, n, dim))
        return (x + self.fc2_b) + self.fc2(h * torch.sigmoid(h))


class RefinerBlock(nn.Module):
    """TextAdapterBlock: pre-norm self-attention (qk norm, key mask) + SwiGLU.
    Rank-4 residual [1, 1, T, D]; the output projection's bias goes to the
    residual (see LayerBlock)."""

    def __init__(self, blk):
        super().__init__()
        a = blk.attn
        self.nh, self.hd = a.num_heads, a.head_dim
        self.eps = blk.norm1.eps
        self.qkv = Lin(a.qkv.weight)
        self.qn = frozen(a.q_norm.weight * self.hd ** -0.5)
        self.kn = frozen(a.k_norm.weight)
        self.proj = Lin(a.proj.weight)
        self.proj_b = frozen(a.proj.bias)
        self.mlp = SwiGLU(blk.mlp)
        self.n1, self.n2 = frozen(blk.norm1.weight), frozen(blk.norm2.weight)

    def forward(self, y, bias):
        e = self.eps
        n = y.shape[2]
        dim = self.nh * self.hd
        qkv = self.qkv(safe_rms(y.reshape(1, n, dim), self.n1, e).reshape(1, 1, n, dim))
        q = safe_rms4(qkv[..., :dim].reshape(1, n, self.nh, self.hd), self.qn, e)
        k = safe_rms4(qkv[..., dim:2 * dim].reshape(1, n, self.nh, self.hd), self.kn, e)
        v = qkv[..., 2 * dim:].reshape(1, n, self.nh, self.hd)
        o = attention(q, k, v, chain=self.nh, bias=bias).reshape(1, 1, n, dim)
        y = (y + self.proj_b) + self.proj(o)
        return y + self.mlp(safe_rms(y.reshape(1, n, dim), self.n2, e).reshape(1, 1, n, dim))


class AdapterGraph(nn.Module):
    """LayerwiseTextEmbedder (lap_blocks2) + y_pos_embedding. Once per prompt."""

    def __init__(self, model):
        super().__init__()
        ye = model.y_embedder
        self.nl = ye.num_layers
        self.layer_blocks = nn.ModuleList(LayerBlock(b, self.nl) for b in ye.layer_blocks)
        self.pool_w = frozen(ye.layer_pool.weight.reshape(1, 1, self.nl, 1))
        self.pool_b = frozen(ye.layer_pool.bias)
        r = ye.refiner
        self.r_proj = Lin(r.proj.weight, r.proj.bias)
        self.r_blocks = nn.ModuleList(RefinerBlock(b) for b in r.blocks)
        self.r_norm = frozen(r.norm.weight)
        self.eps = r.norm.eps

    def forward(self, te, mask_bias, y_pos):
        """te [1, 1, T * Lyr, Dt] (token-major), mask_bias [1, H, T, T], y_pos [1, T, D]
        -> y0 [1, T, D]."""
        n_tok = y_pos.shape[1]
        x = te
        for b in self.layer_blocks:
            x = b(x)
        dt = x.shape[-1]
        y = (x.reshape(1, n_tok, self.nl, dt) * self.pool_w).sum(dim=2, keepdim=True)
        y = self.r_proj(y.reshape(1, 1, n_tok, dt) + self.pool_b)
        for b in self.r_blocks:
            y = b(y, mask_bias)
        return safe_rms(y.reshape(1, n_tok, -1), self.r_norm, self.eps) + y_pos


# ------------------------------------------------------------- chunk plan
def build_graphs(model, dual_split=(3, 3, 2), single_split=(6, 6, 4)):
    """-> ordered {name: module} for the adapter, the trunk chunks and the pixel stage."""
    cfg = model.cfg
    n_dual = cfg.dual_depth
    assert sum(dual_split) == n_dual and sum(single_split) == cfg.depth - n_dual
    graphs = {"dit_adapter": AdapterGraph(model).eval()}
    start = 0
    for i, n in enumerate(dual_split):
        graphs[f"dit_dual{i}"] = DualChunk(model.blocks[start:start + n],
                                           model.s_embedder if i == 0 else None,
                                           cfg.patch_size, cfg.in_channels).eval()
        start += n
    for i, n in enumerate(single_split):
        graphs[f"dit_single{i}"] = SingleChunk(model.blocks[start:start + n]).eval()
        start += n
    graphs["dit_pixel"] = PixelGraph(model).eval()
    return graphs


def mod_rows(dual_split, single_split):
    """Row ranges of the per-step modulation table for each trunk chunk."""
    out, r = {}, 0
    for i, n in enumerate(dual_split):
        out[f"dit_dual{i}"] = (r, r + 12 * n)
        r += 12 * n
    for i, n in enumerate(single_split):
        out[f"dit_single{i}"] = (r, r + 6 * n)
        r += 6 * n
    return out


def run_step(graphs, rows_of, h, x_pix, y0, mods, t_emb, trace=None):
    """One DiT evaluation from the chunk modules plus the host glue.

    h: host tables (cos_j, sin_j, cos_i, sin_i, pix_pos); x_pix [1, L, pp, 3];
    y0 [1, T, D] adapter output; mods [1, R, D]; t_emb [1, 1, D].
    trace: optional dict that receives every chunk's inputs and outputs.
    """
    n_txt = y0.shape[1]
    n = x_pix.shape[1]
    x = x_pix.reshape(1, 1, n, -1)
    y = y0
    names = list(graphs)
    for name in [g for g in names if g.startswith("dit_dual")]:
        lo, hi = rows_of[name]
        args = (x, y, h["cos_j"], h["sin_j"], mods[:, lo:hi])
        x, y = graphs[name](*args)
        if trace is not None:
            trace[name] = {"in": args, "out": (x, y)}
    j = torch.cat([y, x], dim=1)  # host concat, text first
    for name in [g for g in names if g.startswith("dit_single")]:
        lo, hi = rows_of[name]
        args = (j, h["cos_j"], h["sin_j"], mods[:, lo:hi])
        j = graphs[name](*args)
        if trace is not None:
            trace[name] = {"in": args, "out": (j,)}
    args = (j[:, n_txt:], t_emb, x_pix, h["pix_pos"], h["cos_i"], h["sin_i"])
    v = graphs["dit_pixel"](*args)
    if trace is not None:
        trace["dit_pixel"] = {"in": args, "out": (v,)}
    return v


# ------------------------------------------------------------- host side
def s_embed_perm(p, c):
    """Column order that lets s_embedder read a patch flattened as (i, j, c)
    instead of F.unfold's (c, i, j)."""
    return torch.tensor([ch * p * p + (q // c) for q in range(p * p * c) for ch in [q % c]],
                        dtype=torch.long)


def patchify(x, p):
    """[B=1, C, H, W] -> [1, L, p*p, C], L row-major over the patch grid, pixels
    row-major inside a patch (= F.unfold's L order and PixelEmbedder's grouping)."""
    _, c, hh, ww = x.shape
    t = x.permute(0, 2, 3, 1).reshape(1, hh // p, p, ww // p, p, c)
    return t.permute(0, 1, 3, 2, 4, 5).reshape(1, (hh // p) * (ww // p), p * p, c)


def unpatchify(v, p, hh, ww):
    """Inverse of patchify: [1, L, p*p, C] -> [1, C, H, W]."""
    c = v.shape[-1]
    t = v.reshape(1, hh // p, ww // p, p, p, c).permute(0, 1, 3, 2, 4, 5)
    return t.reshape(1, hh, ww, c).permute(0, 3, 1, 2)


def host_tables(model, height, width):
    """RoPE cos / sin (joint text+image for the trunk, image for the pixel
    stage) and the patch-major pixel sincos table, all from upstream code."""
    from iris3b.nn.embeddings import sincos_pos_embed_2d

    cfg = model.cfg
    p = cfg.patch_size
    grid = (height // p, width // p)
    img = model._fetch_rope_img(grid, torch.device("cpu"))
    txt = model._fetch_rope_txt(cfg.text_len, torch.device("cpu"))
    pix = model._fetch_rope_pix(grid, torch.device("cpu"))
    joint = torch.cat([txt, img], dim=0)

    def cs(cis):
        return (cis.real.float()[None, :, None, :].contiguous(),
                cis.imag.float()[None, :, None, :].contiguous())

    cos_j, sin_j = cs(joint)
    cos_i, sin_i = cs(pix)
    pos = sincos_pos_embed_2d(model.pixel_embedder.proj.out_features, height, width)
    pos = pos.reshape(1, height, width, -1).permute(0, 3, 1, 2)
    pix_pos = patchify(pos, p).contiguous()
    return {"cos_j": cos_j, "sin_j": sin_j, "cos_i": cos_i, "sin_i": sin_i, "pix_pos": pix_pos}


@torch.no_grad()
def step_tensors(model, t_model):
    """t_emb [1, 1, D] and the modulation table [1, R, D] for one model time:
    every block's adaLN (shared core + block bias) as six rows, 1 + scale on
    rows 1 and 4, dual blocks img then txt."""
    t_emb = model.t_embedder(t_model)
    cond = F.silu(t_emb)
    groups = []
    for i, blk in enumerate(model.blocks):
        if i < model.cfg.dual_depth:
            groups += [blk.adaln_img(cond), blk.adaln_txt(cond)]
        else:
            groups.append(blk.adaln(cond))
    rows_ = []
    for g in groups:
        r = g.reshape(6, -1).clone()
        r[1] += 1.0
        r[4] += 1.0
        rows_.append(r)
    return t_emb.float(), torch.cat(rows_, dim=0)[None].float()


def mask_bias(mask, n_heads):
    """[1, T] (1 = real token) -> additive [1, H, T, T]: a query attends real
    keys and itself (TransformerTextEmbedder's keep | eye)."""
    t = mask.shape[-1]
    keep = mask.bool()[:, None, None, :] | torch.eye(t, dtype=torch.bool)[None, None]
    bias = torch.zeros(1, 1, t, t)
    bias = bias.masked_fill(~keep, MASK_NEG)
    return bias.expand(1, n_heads, t, t).contiguous()
