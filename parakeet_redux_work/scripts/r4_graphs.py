"""r4 torch modules (shared by the r4 build in .venv-lt and the torch diagnosis in .venv-ref): r2_graphs.py with the
window length as a parameter (the 10 / 20 / 30 s buckets of CONTRACT v3) and the fp16-safe LayerNorm rescale.

MelB(mel_filters, t_mel)   r2 Mel30 with T_MEL = t_mel and N_AUDIO = 160 * t_mel: 10 s = 1,000 frames / 160,000
                           samples, 20 s = 2,000 / 320,000, 30 s = 3,000 / 480,000 (= Mel30). Inputs, outputs and
                           arithmetic as Mel30; frame_mask = mel_mask[::8] (t_mel / 8 encoder frames).
EncB(model, t_mel, ln_k=None, res_m=None, taps=False)
    r2 Enc30 (pw_linear, ln_rank3) with T_MEL = t_mel: features [1,128,t_mel] + mel_mask [1,t_mel] ->
    enc [1,t_mel/8,640]; the relative-position table (2T-1 rows) is built for T = t_mel / 8.
    ln_k {(block, site): k}, site in SITES: that LayerNorm takes x * 2^-k and eps * 4^-k. LN(x 2^-k; eps 4^-k) =
      LN(x; eps) in real arithmetic and in fp32 bit for bit while the values stay normal (a power of two commutes
      with every rounding); it keeps the squared difference (x - mean)^2 of an fp16-storage path under 65,504.
    res_m {block: m}: the block's residual stream is carried at 2^-m (the block input and the four branch outputs
      scaled; every LayerNorm of the block takes eps * 4^-(m + k)); norm_out returns it to scale 1. Default 0.
    taps=True: extra outputs "sub" (subsampling output [1,T,1024]) and "b00".."b23" (each block's output) for the
      per-layer GPU comparison.
    forward(..., q=None): q = an optional per-op hook q(name, x) -> x with q.c(constant) -> constant (r4_diag.py
      records every op output and / or rounds it to fp16); with a hook, LayerNorm, GLU, SiLU and the attention core
      run as the explicit ops the converter emits (MEAN, SQUARED_DIFFERENCE, MEAN, ADD, RSQRT, SUB, MUL; LOGISTIC +
      MUL; BATCH_MATMUL, ADD, SOFTMAX, BATCH_MATMUL). Without a hook (the export) the graph is r2's op for op when
      every k and m is 0.
DecPredict / DecJoint and the constants come from r2_graphs (unchanged).
"""
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r2_graphs import (HOP, KEY_BIAS, LOG_GUARD, MIN_SAMPLES, NBIN, NFFT, NORM_EPS, PREEMPH, WIN,  # noqa: E402,F401
                       DecJoint, DecPredict, _linear_from_conv)

BUCKETS = {10: 1000, 20: 2000, 30: 3000}
W30 = 479840
SITES = ("ff1", "att", "conv", "ff2", "out")


def n_audio(t_mel):
    return t_mel * HOP


def bucket_of(n_samples, buckets=(10, 20, 30)):
    """CONTRACT v3 host rule: the smallest bucket whose audio input holds the window (10 s: n <= 160,000,
    20 s: n <= 320,000, else 30 s); with buckets=(30,) every window goes to 30."""
    for b in sorted(buckets):
        if b != 30 and n_samples <= n_audio(BUCKETS[b]):
            return b
    return 30


def enc_valid_frames(n_valid):
    return ((n_valid // HOP) + 7) // 8


def _id(name, x):
    return x


class MelB(nn.Module):
    def __init__(self, mel_filters, t_mel):
        super().__init__()
        self.t_mel = int(t_mel)
        self.n_audio = n_audio(self.t_mel)
        n = torch.arange(WIN, dtype=torch.float64)
        k = torch.arange(NBIN, dtype=torch.float64)
        w = torch.hann_window(WIN, periodic=False, dtype=torch.float64)
        ph = 2.0 * math.pi * torch.outer(k, n + (NFFT - WIN) // 2) / NFFT
        basis = torch.cat([torch.cos(ph) * w, -torch.sin(ph) * w], 0)
        self.register_buffer("basis", basis.to(torch.float32))
        self.register_buffer("melfb", mel_filters.to(torch.float32).clone())
        self.register_buffer("sample_idx", torch.arange(self.n_audio, dtype=torch.float32)[None])
        self.register_buffer("frame_idx", torch.arange(self.t_mel, dtype=torch.float32))

    def forward(self, audio, n_valid):
        t, n = self.t_mel, self.n_audio
        nv = n_valid.to(torch.float32)
        smask = (self.sample_idx < nv).to(torch.float32)
        prev = F.pad(audio[:, :-1], (1, 0))
        y = (audio - PREEMPH * prev) * smask
        yp = F.pad(y, (WIN // 2, (t + 2) * HOP - n - WIN // 2))
        ch = yp.reshape(t + 2, HOP)
        frames = torch.cat([ch[0:t], ch[1:t + 1], ch[2:t + 2, :WIN - 2 * HOP]], dim=1)
        spec = F.linear(frames, self.basis)
        re = spec[:, :NBIN]
        im = spec[:, NBIN:]
        mag = torch.sqrt(re * re + im * im)
        power = mag * mag
        mel = F.linear(power, self.melfb)
        logmel = torch.log(mel + LOG_GUARD)
        lf = torch.floor(nv / HOP)
        fm = (self.frame_idx < lf).to(torch.float32)
        m = fm[:, None]
        xm = logmel * m
        mean = xm.sum(0, keepdim=True) / lf
        var = (((xm - mean) ** 2) * m).sum(0, keepdim=True) / (lf - 1.0)
        std = torch.sqrt(var)
        feat = (logmel - mean) / (std + NORM_EPS) * m
        mel_mask = fm[None]
        return {"features": feat.t()[None], "mel_mask": mel_mask, "frame_mask": mel_mask[:, ::8]}


class LNS(nn.Module):
    """r2 LN3 (affine-free layer_norm, then weight / bias as [1, 1, C] constants) with the input scaled by 2^-k and eps by
    4^-(k + m) (m = the residual scale of the block the norm reads)."""

    def __init__(self, ln, k=0, m=0):
        super().__init__()
        self.shape = tuple(ln.normalized_shape)
        self.k = int(k)
        self.eps = float(ln.eps) * 4.0 ** -(self.k + int(m))
        self.register_buffer("w", ln.weight.detach().clone().view(1, 1, -1))
        self.register_buffer("b", ln.bias.detach().clone().view(1, 1, -1))

    def forward(self, x, q=None, name=""):
        if q is None:
            if self.k:
                x = x * (2.0 ** -self.k)
            return F.layer_norm(x, self.shape, None, None, self.eps) * self.w + self.b
        x = q(name + ".in", x)
        if self.k:
            x = q(name + ".scaled", x * (2.0 ** -self.k))
        mu = q(name + ".mean", x.mean(-1, keepdim=True))
        d2 = q(name + ".sqdiff", (x - mu) ** 2)
        var = q(name + ".var", d2.mean(-1, keepdim=True))
        r = q(name + ".rsqrt", torch.rsqrt(q(name + ".var_eps", var + q.c(self.eps))))
        y = q(name + ".norm", q(name + ".sub", x - mu) * r)
        return q(name + ".out", q(name + ".w", y * q.c(self.w)) + q.c(self.b))


class BlockB(nn.Module):
    def __init__(self, layer, idx, ln_k=None, m=0):
        super().__init__()
        lk = ln_k or {}
        self.idx = int(idx)
        self.m = int(m)
        self.ff1 = layer.feed_forward1
        self.att = layer.self_attn
        self.conv = layer.conv
        self.ff2 = layer.feed_forward2
        self.n_ff1 = LNS(layer.norm_feed_forward1, lk.get((idx, "ff1"), 0), self.m)
        self.n_att = LNS(layer.norm_self_att, lk.get((idx, "att"), 0), self.m)
        self.n_conv = LNS(layer.norm_conv, lk.get((idx, "conv"), 0), self.m)
        self.n_ff2 = LNS(layer.norm_feed_forward2, lk.get((idx, "ff2"), 0), self.m)
        self.n_out = LNS(layer.norm_out, lk.get((idx, "out"), 0), self.m)
        self.pw1 = _linear_from_conv(layer.conv.pointwise_conv1)
        self.pw2 = _linear_from_conv(layer.conv.pointwise_conv2)

    def ffn(self, f, x, q, p):
        if q is None:
            return f(x)
        a = q(p + "l1", f.linear1(x))
        a = q(p + "silu", a * q(p + "sig", torch.sigmoid(a)))
        return q(p + "l2", f.linear2(a))

    def attn(self, x, pos, kbias, q, p):
        a = self.att
        qq = q or _id
        t = x.shape[1]
        qp = qq(p + "q", a.q_proj(x)).view(1, t, -1, a.head_dim).transpose(1, 2)
        k = qq(p + "k", a.k_proj(x)).view(1, t, -1, a.head_dim).transpose(1, 2)
        v = qq(p + "v", a.v_proj(x)).view(1, t, -1, a.head_dim).transpose(1, 2)
        nh = qp.shape[1]
        bu, bv = a.bias_u.view(1, nh, 1, a.head_dim), a.bias_v.view(1, nh, 1, a.head_dim)
        rk = a.relative_k_proj(pos).view(1, -1, nh, a.head_dim)
        if q is not None:
            bu, bv, rk = q.c(bu), q.c(bv), q.c(rk)
        qu = qq(p + "qu", qp + bu)
        qv = qq(p + "qv", qp + bv)
        bd = qq(p + "bd", qv @ rk.permute(0, 2, 3, 1))
        bd = qq(p + "bd_scale", a._rel_shift(bd)[..., :t] * a.scaling)
        bd = qq(p + "bd_mask", bd + kbias)
        if q is None:
            o = F.scaled_dot_product_attention(qu, k, v, attn_mask=bd, dropout_p=0.0, scale=a.scaling, is_causal=False)
        else:
            s = q(p + "ac", (qu @ k.transpose(-1, -2)) * a.scaling)
            s = q(p + "logits", s + bd)
            o = q(p + "ctx", q(p + "softmax", torch.softmax(s, -1)) @ v)
        o = o.transpose(1, 2).contiguous().reshape(1, t, -1)
        return qq(p + "o", a.o_proj(o))

    def convm(self, x, cmask, q, p):
        c = self.conv
        if q is None:
            h = F.glu(self.pw1(x), dim=-1) * cmask
            return self.pw2(c.activation(c.norm(c.depthwise_conv(h.transpose(1, 2)))).transpose(1, 2))
        a = q(p + "pw1", self.pw1(x))
        a1, a2 = a.chunk(2, dim=-1)
        h = q(p + "mask", q(p + "glu", a1 * q(p + "glu_sig", torch.sigmoid(a2))) * cmask)
        h = q(p + "dw_bn", c.norm(c.depthwise_conv(h.transpose(1, 2))))
        h = q(p + "silu", h * q(p + "sig", torch.sigmoid(h)))
        return q(p + "pw2", self.pw2(h.transpose(1, 2)))

    def forward(self, h, pos, kbias, cmask, q=None):
        qq = q or _id
        p = f"b{self.idx:02d}."
        if self.m:
            h = qq(p + "res_in", h * (2.0 ** -self.m))
        s = 2.0 ** -self.m
        h = qq(p + "r_ff1", h + (0.5 * s) * self.ffn(self.ff1, self.n_ff1(h, q, p + "n_ff1"), q, p + "ff1."))
        o = self.attn(self.n_att(h, q, p + "n_att"), pos, kbias, q, p + "att.")
        h = qq(p + "r_att", h + (o * s if self.m else o))
        o = self.convm(self.n_conv(h, q, p + "n_conv"), cmask, q, p + "conv.")
        h = qq(p + "r_conv", h + (o * s if self.m else o))
        h = qq(p + "r_ff2", h + (0.5 * s) * self.ffn(self.ff2, self.n_ff2(h, q, p + "n_ff2"), q, p + "ff2."))
        return self.n_out(h, q, p + "n_out")


class EncB(nn.Module):
    def __init__(self, model, t_mel, ln_k=None, res_m=None, taps=False, key_bias=KEY_BIAS):
        super().__init__()
        assert t_mel % 8 == 0, t_mel
        self.t_mel = int(t_mel)
        self.t_enc = self.t_mel // 8
        enc = model.encoder
        sub = enc.subsampling
        kinds = [type(m).__name__ for m in sub.layers]
        assert kinds == ["Conv2d", "ReLU", "Conv2d", "Conv2d", "ReLU", "Conv2d", "Conv2d", "ReLU"], kinds
        self.c0, self.dw1, self.pw1, self.dw2, self.pw2 = (sub.layers[i] for i in (0, 2, 3, 5, 6))
        self.sub_out = sub.linear
        assert enc.input_scale == 1.0, enc.input_scale
        rm = res_m or {}
        self.blocks = nn.ModuleList([BlockB(lay, i, ln_k, rm.get(i, 0)) for i, lay in enumerate(enc.layers)])
        self.proj_out = model.encoder_projector
        with torch.no_grad():
            pos = enc.encode_positions(torch.zeros(1, self.t_enc, enc.config.hidden_size))
        self.register_buffer("pos", pos.detach().clone())
        self.key_bias = float(key_bias)
        self.taps = bool(taps)

    def forward(self, features, mel_mask, q=None):
        qq = q or _id
        x = features.transpose(1, 2)[:, None]
        # the masks are sliced at rank 4 ([1, 1, T, 1], the time axis of the NCHW conv tensors) as in r2
        m0 = mel_mask.reshape(1, 1, -1, 1)
        m1 = m0[:, :, ::2]
        m2 = m1[:, :, ::2]
        m3 = m2[:, :, ::2]
        x = F.relu(qq("sub.c0m", qq("sub.c0", self.c0(x)) * m1))
        x = qq("sub.dw1m", qq("sub.dw1", self.dw1(x)) * m2)
        x = F.relu(qq("sub.pw1m", qq("sub.pw1", self.pw1(x)) * m2))
        x = qq("sub.dw2m", qq("sub.dw2", self.dw2(x)) * m3)
        x = F.relu(qq("sub.pw2m", qq("sub.pw2", self.pw2(x)) * m3))
        x = x.transpose(1, 2).reshape(1, x.shape[2], -1)
        h = qq("sub.out", self.sub_out(x))
        kbias = ((m3 - 1.0) * self.key_bias).reshape(1, 1, 1, -1)
        cmask = m3.reshape(1, -1, 1)
        outs = {}
        if self.taps:
            outs["sub"] = h
        for b in self.blocks:
            h = b(h, self.pos, kbias, cmask, q)
            if self.taps:
                outs[f"b{b.idx:02d}"] = h
        outs["enc"] = qq("proj", self.proj_out(h))
        return outs


def fix_from_json(d):
    """fix.json (r4_diag.py) -> (ln_k {(block, site): k}, res_m {block: m})"""
    ln_k = {(int(r["block"]), r["site"]): int(r["k"]) for r in d.get("ln_k", []) if int(r["k"]) > 0}
    res_m = {int(b): int(m) for b, m in d.get("res_m", {}).items() if int(m) > 0}
    return ln_k, res_m
