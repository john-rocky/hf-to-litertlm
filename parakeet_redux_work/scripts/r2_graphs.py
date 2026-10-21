"""r2 torch modules for the LiteRT files (shared by the export in .venv-lt and the torch checks in .venv-ref).

Mel30   audio [1, 480000] f32 (valid samples first, zeros after) + n_valid [1] int32
        -> features [1, 128, 3000] f32, mel_mask [1, 3000] f32, frame_mask [1, 375] f32.
        transformers ParakeetFeatureExtractor in one graph: preemphasis 0.97 (zeroed from n_valid on), STFT with
        n_fft 512 / win 400 symmetric Hann / hop 160 / center zero pad written as a constant DFT matmul over 400-sample
        frames, |X| then squared (the extractor's sqrt-then-pow), Slaney mel 128 (the extractor's own filter bank),
        log(x + 2^-24), per-feature mean / unbiased std over the L0 = n_valid // 160 valid frames, (x - mean) /
        (std + 1e-5), padded frames 0. mel_mask[t] = t < L0; frame_mask = mel_mask[::8] (the valid encoder frames,
        L3 = ceil(L0 / 8)). n_valid is cast to float32 once; the index compares and floor(n_valid / 160) are float
        ops on exact integers (all below 2^24).
        n_valid must be >= 320 (two mel frames): below that the unbiased std divides by zero, as in the extractor.

Enc30   features [1, 128, 3000] f32 + mel_mask [1, 3000] f32 -> enc [1, 375, 640] f32 (after encoder_projector).
        Reference B's encoder rewritten so that every mask is float arithmetic (no SELECT_V2 / where / masked_fill, no
        integer ops): the subsampling masks after each conv are mel_mask[::2] / [::4] / [::8] (L_{r+1} = ceil(L_r / 2)
        makes m_{r+1}[t] = m_r[2t] exact) applied by MUL; attention adds a finite key bias (m3 - 1) * 30000 to the
        relative-position term (fp16-safe; valid rows equal the reference's finfo.min fill, padded rows are never read
        by a valid row); the conv module zeroes padded frames by MUL. Options: pw_linear writes the 48 pointwise convs
        as nn.Linear (FULLY_CONNECTED, which the int4 recipe can take; Conv1d k=1 lowers to CONV_2D, which it leaves
        float); ln_rank3 applies every LayerNorm's weight / bias as [1, 1, 1024] constants after an affine-free
        layer_norm (the same arithmetic as nn.LayerNorm; the export then has no rank-3 x rank-1 elementwise site).

DecPredict  token [1, 1] int32 + h [2, 1, 640] + c [2, 1, 640] -> dec_out [1, 640] (decoder_projector output),
            h_out, c_out: the 2-layer LSTM step written as explicit gates (torch's LSTMCell arithmetic).
DecJoint    enc_frame [1, 640] + dec_out [1, 640] -> logits [1, 8198] = head(relu(enc_frame + dec_out)) (8,193 token
            logits then 5 duration logits).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

N_AUDIO = 480000
T_MEL = 3000
T_ENC = 375
HOP = 160
WIN = 400
NFFT = 512
NBIN = NFFT // 2 + 1
LOG_GUARD = 2.0 ** -24
NORM_EPS = 1e-5
PREEMPH = 0.97
KEY_BIAS = 30000.0
MIN_SAMPLES = 2 * HOP


class Mel30(nn.Module):
    def __init__(self, mel_filters):
        super().__init__()
        n = torch.arange(WIN, dtype=torch.float64)
        k = torch.arange(NBIN, dtype=torch.float64)
        w = torch.hann_window(WIN, periodic=False, dtype=torch.float64)
        ph = 2.0 * math.pi * torch.outer(k, n + (NFFT - WIN) // 2) / NFFT
        basis = torch.cat([torch.cos(ph) * w, -torch.sin(ph) * w], 0)
        self.register_buffer("basis", basis.to(torch.float32))
        self.register_buffer("melfb", mel_filters.to(torch.float32).clone())
        # sample / frame indices as float32: exact for integers below 2^24 (480,000 here), so the compares and the
        # floor below are exact and the only integer op of the graph is the cast of n_valid
        self.register_buffer("sample_idx", torch.arange(N_AUDIO, dtype=torch.float32)[None])
        self.register_buffer("frame_idx", torch.arange(T_MEL, dtype=torch.float32))

    def forward(self, audio, n_valid):
        nv = n_valid.to(torch.float32)
        smask = (self.sample_idx < nv).to(torch.float32)
        prev = F.pad(audio[:, :-1], (1, 0))
        y = (audio - PREEMPH * prev) * smask
        # frame t covers padded samples [160 t, 160 t + 400) = samples [160 t - 200, 160 t + 200): the 512-point
        # centered frame of torch.stft restricted to the 400 window taps (offset 56)
        yp = F.pad(y, (WIN // 2, (T_MEL + 2) * HOP - N_AUDIO - WIN // 2))
        ch = yp.reshape(T_MEL + 2, HOP)
        frames = torch.cat([ch[0:T_MEL], ch[1:T_MEL + 1], ch[2:T_MEL + 2, :WIN - 2 * HOP]], dim=1)
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


class LN3(nn.Module):
    """nn.LayerNorm with its weight / bias as [1, 1, C] constants (same arithmetic: normalize, * w, + b)."""

    def __init__(self, ln):
        super().__init__()
        self.shape = ln.normalized_shape
        self.eps = ln.eps
        self.register_buffer("w", ln.weight.detach().clone().view(1, 1, -1))
        self.register_buffer("b", ln.bias.detach().clone().view(1, 1, -1))

    def forward(self, x):
        return F.layer_norm(x, self.shape, None, None, self.eps) * self.w + self.b


def _linear_from_conv(conv):
    lin = nn.Linear(conv.in_channels, conv.out_channels, bias=conv.bias is not None)
    lin.weight = nn.Parameter(conv.weight.detach()[:, :, 0].clone())
    if conv.bias is not None:
        lin.bias = nn.Parameter(conv.bias.detach().clone())
    return lin


class Block(nn.Module):
    def __init__(self, layer, pw_linear=True, ln_rank3=True):
        super().__init__()
        self.ff1 = layer.feed_forward1
        self.att = layer.self_attn
        self.conv = layer.conv
        self.ff2 = layer.feed_forward2
        mk = LN3 if ln_rank3 else (lambda ln: ln)
        self.n_ff1 = mk(layer.norm_feed_forward1)
        self.n_att = mk(layer.norm_self_att)
        self.n_conv = mk(layer.norm_conv)
        self.n_ff2 = mk(layer.norm_feed_forward2)
        self.n_out = mk(layer.norm_out)
        self.pw_linear = pw_linear
        if pw_linear:
            self.pw1 = _linear_from_conv(layer.conv.pointwise_conv1)
            self.pw2 = _linear_from_conv(layer.conv.pointwise_conv2)

    def attn(self, x, pos, kbias):
        a = self.att
        t = x.shape[1]
        q = a.q_proj(x).view(1, t, -1, a.head_dim).transpose(1, 2)
        k = a.k_proj(x).view(1, t, -1, a.head_dim).transpose(1, 2)
        v = a.v_proj(x).view(1, t, -1, a.head_dim).transpose(1, 2)
        nh = q.shape[1]
        qu = q + a.bias_u.view(1, nh, 1, a.head_dim)
        qv = q + a.bias_v.view(1, nh, 1, a.head_dim)
        rk = a.relative_k_proj(pos).view(1, -1, nh, a.head_dim)
        bd = qv @ rk.permute(0, 2, 3, 1)
        bd = a._rel_shift(bd)[..., :t] * a.scaling
        bd = bd + kbias
        o = F.scaled_dot_product_attention(qu, k, v, attn_mask=bd, dropout_p=0.0, scale=a.scaling, is_causal=False)
        o = o.transpose(1, 2).contiguous().reshape(1, t, -1)
        return a.o_proj(o)

    def convm(self, x, cmask):
        c = self.conv
        if self.pw_linear:
            h = F.glu(self.pw1(x), dim=-1) * cmask
            h = h.transpose(1, 2)
        else:
            h = F.glu(c.pointwise_conv1(x.transpose(1, 2)), dim=1) * cmask.transpose(1, 2)
        h = c.activation(c.norm(c.depthwise_conv(h)))
        if self.pw_linear:
            return self.pw2(h.transpose(1, 2))
        return c.pointwise_conv2(h).transpose(1, 2)

    def forward(self, h, pos, kbias, cmask):
        h = h + 0.5 * self.ff1(self.n_ff1(h))
        h = h + self.attn(self.n_att(h), pos, kbias)
        h = h + self.convm(self.n_conv(h), cmask)
        h = h + 0.5 * self.ff2(self.n_ff2(h))
        return self.n_out(h)


class Enc30(nn.Module):
    def __init__(self, model, pw_linear=True, ln_rank3=True, key_bias=KEY_BIAS):
        super().__init__()
        enc = model.encoder
        sub = enc.subsampling
        kinds = [type(m).__name__ for m in sub.layers]
        assert kinds == ["Conv2d", "ReLU", "Conv2d", "Conv2d", "ReLU", "Conv2d", "Conv2d", "ReLU"], kinds
        self.c0, self.dw1, self.pw1, self.dw2, self.pw2 = (sub.layers[i] for i in (0, 2, 3, 5, 6))
        self.sub_out = sub.linear
        assert enc.input_scale == 1.0, enc.input_scale
        self.blocks = nn.ModuleList([Block(lay, pw_linear, ln_rank3) for lay in enc.layers])
        self.proj_out = model.encoder_projector
        with torch.no_grad():
            pos = enc.encode_positions(torch.zeros(1, T_ENC, enc.config.hidden_size))
        self.register_buffer("pos", pos.detach().clone())
        self.key_bias = float(key_bias)

    def forward(self, features, mel_mask):
        x = features.transpose(1, 2)[:, None]
        # the masks are sliced at rank 4 ([1, 1, T, 1], the time axis of the NCHW conv tensors) so that no rank-2
        # strided slice reaches a GPU delegate
        m0 = mel_mask.reshape(1, 1, -1, 1)
        m1 = m0[:, :, ::2]
        m2 = m1[:, :, ::2]
        m3 = m2[:, :, ::2]
        x = F.relu(self.c0(x) * m1)
        x = self.dw1(x) * m2
        x = F.relu(self.pw1(x) * m2)
        x = self.dw2(x) * m3
        x = F.relu(self.pw2(x) * m3)
        x = x.transpose(1, 2).reshape(1, x.shape[2], -1)
        h = self.sub_out(x)
        kbias = ((m3 - 1.0) * self.key_bias).reshape(1, 1, 1, -1)
        cmask = m3.reshape(1, -1, 1)
        for b in self.blocks:
            h = b(h, self.pos, kbias, cmask)
        return {"enc": self.proj_out(h)}


class DecPredict(nn.Module):
    def __init__(self, model):
        super().__init__()
        d = model.decoder
        self.emb = d.embedding
        lstm = d.lstm
        assert lstm.batch_first and lstm.num_layers == 2 and lstm.bias and not lstm.bidirectional
        self.nl = lstm.num_layers
        self.w_ih = nn.ParameterList([getattr(lstm, f"weight_ih_l{i}") for i in range(self.nl)])
        self.w_hh = nn.ParameterList([getattr(lstm, f"weight_hh_l{i}") for i in range(self.nl)])
        self.b_ih = nn.ParameterList([getattr(lstm, f"bias_ih_l{i}") for i in range(self.nl)])
        self.b_hh = nn.ParameterList([getattr(lstm, f"bias_hh_l{i}") for i in range(self.nl)])
        self.proj = d.decoder_projector

    def forward(self, token, h, c):
        x = self.emb(token)[:, 0]
        hs, cs = [], []
        for i in range(self.nl):
            g = F.linear(x, self.w_ih[i], self.b_ih[i]) + F.linear(h[i], self.w_hh[i], self.b_hh[i])
            gi, gf, gg, go = g.chunk(4, dim=-1)
            ci = torch.sigmoid(gf) * c[i] + torch.sigmoid(gi) * torch.tanh(gg)
            hi = torch.sigmoid(go) * torch.tanh(ci)
            hs.append(hi)
            cs.append(ci)
            x = hi
        return {"dec_out": self.proj(x), "h_out": torch.stack(hs, 0), "c_out": torch.stack(cs, 0)}


class DecJoint(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.act = model.joint.activation
        self.head = model.joint.head

    def forward(self, enc_frame, dec_out):
        return {"logits": self.head(self.act(enc_frame + dec_out))}


def enc_valid_frames(n_valid):
    """valid encoder frames of a window with n_valid samples: L3 = ceil(L0 / 8), L0 = n_valid // 160."""
    l0 = n_valid // HOP
    return (l0 + 7) // 8


def windows(n_samples, size=479840, min_samples=MIN_SAMPLES):
    """[(start, length)] of the 30 s windows of an utterance (r1's cut: consecutive `size`-sample windows from the
    start); a window shorter than min_samples (two mel frames) is dropped."""
    out = []
    for s in range(0, max(n_samples, 1), size):
        n = min(size, n_samples - s)
        if n >= min_samples:
            out.append((s, n))
    return out
