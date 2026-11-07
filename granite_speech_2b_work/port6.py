# Portions adapted from transformers 5.22.1 (Apache License 2.0, Copyright 2025 The HuggingFace Inc. team):
#   models/granite_speech/{feature_extraction,modeling}_granite_speech.py, models/blip_2/modeling_blip_2.py.
"""granite-speech-5.0-2b audio path for the GPU and NPU delegates: the computation of port.py, written again.

  GraniteSpeechAudioGraph6(config, win_frames=W).forward(audio f32 [1, W, 160])
      -> {"features" f32 [1, W / 10, 2048], "mask" uint8 [1, W / 10]}       (W = 2400 for a 24 s window)

The same computation as port.py (transformers on the UNPADDED clip, length = last non-zero sample + 1), written
without the ops port.py's graph needs and the GPU delegates refuse: GATHER_ND, BROADCAST_TO, rank-5 CONCATENATION /
RESHAPE, SELECT_V2 with a broadcast condition, MEAN over a constant. Every tensor is rank 4 or lower:

  * length, frame and token masks in BLOCK units (the input's own 160-sample rows): per block "has a non-zero
    sample" and "last non-zero offset" (1..160), a suffix-any by 12 shift-and-max steps, the one-hot of the last
    non-zero block K and its offset o; n = 160 K + o is never formed as a number. Frame t is valid iff t <= K, or
    t = K + 1 when o = 160 (= t < n // 160 + 1 = M); encoder frame e is valid iff frame 2e + 1 is (= e < M // 2 = E);
    token t is valid iff 15 (t // 3) < E. Every value stays an integer <= 1,800 or a 0 / 1 flag, exact in fp16 too.
    No division, no gather.
  * the STFT's centre reflection without an index gather: frames are cut from the zero-padded input by static slices;
    the start reflection (frames 0 and 1) is a constant linear map of samples 0..319; the end reflection touches at
    most frames n // 160 and n // 160 - 1: those two rows of the window shifted one sample left are picked by a
    one-hot matmul, mapped by the reflection matrix P(u) built from comparisons on the in-frame offset u (0..199),
    and added back by the transposed one-hot matmul.
  * block attention with blocks x heads on one axis ([1, blocks x heads, 200, 128]); Shaw relative positions by the
    Transformer-XL skew instead of a gather: q times the row-flipped table with one leading zero row ([1, N, 200,
    400]), reshape [1, N, 400, 200], drop row 0, reshape [1, N, 200, 399], keep columns 0..199
    (= q[c] . table[c - r + 199]); the key mask from a constant folded index compared with E.
  * the depthwise convolution as a 2-D convolution on [1, C, 1, T], so its zero padding is on axis 2 of a rank-4
    tensor (a rank-3 PAD on a middle axis returns rows shifted by one on the GPU, LiteRT#9272).
  * zeroing by MUL with a 0 / 1 float mask (no where / SELECT_V2).
  * Q-Former: the learned query's LayerNorm and layer 0's self-attention depend on no input: computed once at load
    (fp32, from the checkpoint's BF16 -> fp32 weights) and kept as constants; layer 0's cross-attention scores are an
    FC of the keys with the constant query folded into a block-diagonal weight; every runtime tensor carries the
    window batch (no broadcast of the query).
`features` = the projector output (feature_scale 1.0).

The converter names every tensor after its module path (`port6.GraniteSpeechAudioGraph6/...`), so a renamed file,
class or attribute gives a TFLite file with other bytes.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

SR, HOP, N_FFT, WIN_LEN, N_MELS = 16000, 160, 512, 400, 80
CTX = 200            # encoder block attention context
WINDOW = 15          # projector window (encoder frames)
QUERIES = 3          # projector queries per window
NEG = -1.0e4         # additive key mask (exp underflows to 0 in fp32; representable in fp16)


def torchaudio_constants():
    """The hann window and mel filterbank exactly as transformers' feature extractor gets them from torchaudio."""
    import torchaudio
    ms = torchaudio.transforms.MelSpectrogram(sample_rate=SR, n_fft=N_FFT, win_length=WIN_LEN, hop_length=HOP,
                                              n_mels=N_MELS)
    window = ms.spectrogram.window.detach().clone().to(torch.float32)      # [400] hann (periodic)
    fb = ms.mel_scale.fb.detach().clone().to(torch.float32)                # [257, 80]
    assert ms.spectrogram.center and ms.spectrogram.pad_mode == "reflect" and ms.spectrogram.power == 2.0
    assert ms.spectrogram.normalized is False and ms.spectrogram.onesided
    return window, fb


def _k4(v):
    return torch.tensor(v, dtype=torch.float32).reshape(1, 1, 1, 1)


class LogMel6(nn.Module):
    """[1, B, 160] PCM blocks in [-1, 1) -> (stacked log-mel [1, F2, 160], frame valid [1, F, 1], encoder-frame valid
    [1, F2, 1], E [1, 1, 1]); F = B + 1 mel frames, F2 = (F - 1) // 2 encoder frames."""

    def __init__(self, n_blocks):
        super().__init__()
        self.B = n_blocks
        self.F = n_blocks + 1                             # mel frames of a full window (n = 160 B): 3601
        self.F2 = (self.F - 1) // 2                       # encoder frames the static graph keeps: 1800
        window, fb = torchaudio_constants()
        f32 = torch.float32
        # frame t covers x_reflect[160 t - 200 : 160 t + 200]: the 400 non-zero taps of the window torch.stft centres
        # inside n_fft = 512 (left pad (512 - 400) // 2 = 56: tap 0 sits at frame offset 56)
        n = torch.arange(56, 56 + WIN_LEN, dtype=torch.float64).reshape(WIN_LEN, 1)
        k = torch.arange(N_FFT // 2 + 1, dtype=torch.float64).reshape(1, N_FFT // 2 + 1)
        ang = 2.0 * math.pi * torch.remainder(n * k, N_FFT) / N_FFT
        self.register_buffer("dft_cos", (torch.cos(ang) * window.double().reshape(WIN_LEN, 1)).to(f32).contiguous(),
                             persistent=False)
        self.register_buffer("dft_sin", (torch.sin(ang) * window.double().reshape(WIN_LEN, 1)).to(f32).contiguous(),
                             persistent=False)
        self.register_buffer("mel_fb", fb.contiguous(), persistent=False)
        # in-block offsets 1..160 (the last non-zero offset of a block = max over its non-zero samples)
        self.register_buffer("jgrid", torch.arange(1, HOP + 1, dtype=f32).reshape(1, 1, 1, HOP), persistent=False)
        # frame 0 is always valid (M = n // 160 + 1 >= 1; an all-zero window = port.py's n = 1)
        e0 = torch.zeros(1, 1, self.F, 1, dtype=f32)
        e0[0, 0, 0, 0] = 1.0
        self.register_buffer("e0", e0, persistent=False)
        # start reflection: x_r[-i] = x[i]; frame 0 offsets 0..199 = x[200 - j], frame 1 offsets 0..39 = x[40 - j];
        # read from samples 0..319 (blocks 0 and 1) -> [1, 1, 800] = the two 400-tap rows to add
        W01 = torch.zeros(2 * HOP, 2 * WIN_LEN, dtype=f32)
        for j in range(200):
            W01[200 - j, j] = 1.0
        for j in range(40):
            W01[40 - j, WIN_LEN + j] = 1.0
        self.register_buffer("W01", W01, persistent=False)
        # end reflection matrix P(u)[i, j] = [i + j == 2u + 399] * [j >= u + 200]: i = offset in the window shifted
        # one sample left (x[160 t - 201 + i]), j = frame offset (x_r[160 t - 200 + j]), u = n - 160 t in 0..199
        self.register_buffer("ij_sum", (torch.arange(WIN_LEN, dtype=f32).reshape(1, 1, WIN_LEN, 1)
                                        + torch.arange(WIN_LEN, dtype=f32).reshape(1, 1, 1, WIN_LEN)).contiguous(),
                             persistent=False)                                  # [1, 1, 400, 400]
        self.register_buffer("jcol", torch.arange(WIN_LEN, dtype=f32).reshape(1, 1, 1, WIN_LEN), persistent=False)
        self.register_buffer("c160", _k4(160.0), persistent=False)
        self.register_buffer("c39", _k4(39.0), persistent=False)
        self.register_buffer("c399", _k4(399.0), persistent=False)
        self.register_buffer("c200", _k4(200.0), persistent=False)

    @staticmethod
    def _shift_up(g, s):
        """g[..., k + s, :] with zeros past the end; g is rank 4 [1, 1, L, 1] (a rank-3 PAD on a middle axis returns
        rows shifted by one on ML Drift, LiteRT #9272: every pad here is on axis 2 of a rank-4 tensor)."""
        return F.pad(g[:, :, s:, :], (0, 0, 0, s))

    def lengths(self, x):
        """Length facts in block units (x [1, 1, B, 160]): frame valid v [1, 1, F, 1], one-hot rows of the frames
        n // 160 - 1 (h0) and n // 160 (h1), n % 160 (r), each [1, 1, F, 1] / [1, 1, 1, 1]."""
        nz = (x != 0.0).to(torch.float32)                                              # [1, 1, B, 160]
        bnz = torch.amax(nz, dim=3, keepdim=True)                                      # [1, 1, B, 1]
        lastoff = torch.amax(nz * self.jgrid, dim=3, keepdim=True)                     # [1, 1, B, 1] in 0..160
        g = bnz
        s = 1
        while s < self.B:                                                              # suffix-any (inclusive)
            g = torch.maximum(g, self._shift_up(g, s))
            s *= 2
        L = g - self._shift_up(g, 1)                                                   # one-hot at K
        o = torch.sum(L * lastoff, dim=2, keepdim=True)                                # [1, 1, 1, 1] in 0..160
        full = torch.eq(o, self.c160).to(torch.float32)                                # n % 160 == 0
        r = o * (1.0 - full)                                                           # n % 160
        L_ext = F.pad(L, (0, 0, 0, 1))                                                 # [1, 1, F, 1] L[t]
        L_prev = F.pad(L, (0, 0, 1, 0))                                                # [1, 1, F, 1] L[t - 1]
        v = F.pad(g, (0, 0, 0, 1)) + L_prev * full                                     # t < M
        v = torch.maximum(v, self.e0)
        h1 = L_ext * (1.0 - full) + L_prev * full                                      # t == n // 160
        h0 = self._shift_up(h1, 1) * torch.le(r, self.c39).to(torch.float32)           # t == n // 160 - 1, r <= 39
        return v, h0, h1, r

    def frames(self, x, h0, h1, r):
        """The reflect-padded clip's 400-tap frames [1, 1, F, 400] (= port.py's gather of the reflect index); x [1, 1, B, 160]."""
        F_ = self.F
        xp = F.pad(x, (0, 0, 2, 2))                                                    # [1, 1, B + 4, 160]
        fz = torch.cat([xp[:, :, 0:F_, 120:HOP], xp[:, :, 1:F_ + 1, :], xp[:, :, 2:F_ + 2, :],
                        xp[:, :, 3:F_ + 3, 0:40]], dim=3)                              # x[160t-200 : 160t+200]
        sh = torch.cat([xp[:, :, 0:F_, 119:HOP], xp[:, :, 1:F_ + 1, :], xp[:, :, 2:F_ + 2, :],
                        xp[:, :, 3:F_ + 3, 0:39]], dim=3)                              # x[160t-201 : 160t+199]
        # end reflection on the two frames that reach past n
        hc = torch.cat([h0, h1], dim=3)                                                # [1, 1, F, 2]
        sel = torch.matmul(hc.transpose(2, 3), sh)                                     # [1, 1, 2, 400]
        u = torch.cat([r + self.c160, r], dim=2)                                       # [1, 1, 2, 1] = n - 160 t
        a = self.ij_sum - (2.0 * u + self.c399).reshape(1, 2, 1, 1)                    # [1, 2, 400, 400]
        eq = torch.eq(a, 0.0).to(torch.float32)
        ge = torch.ge(self.jcol, u.reshape(1, 2, 1, 1) + self.c200).to(torch.float32)  # [1, 2, 1, 400]
        corr = torch.matmul(sel.reshape(1, 2, 1, WIN_LEN), eq * ge)                    # [1, 2, 1, 400]
        fz = fz + torch.matmul(hc, corr.reshape(1, 1, 2, WIN_LEN))                     # [1, 1, F, 400]
        # start reflection on frames 0 and 1 (x_r[-i] = x[i])
        start = torch.matmul(x[:, :, 0:2, :].reshape(1, 1, 1, 2 * HOP), self.W01).reshape(1, 1, 2, WIN_LEN)
        return torch.cat([fz[:, :, 0:2, :] + start, fz[:, :, 2:, :]], dim=2)

    def forward(self, x):
        """x [1, B, 160] (the runtime's frames) -> stacked [1, F2, 160], v [1, 1, F, 1], ve [1, F2, 1], E [1, 1, 1, 1]."""
        x = x.reshape(1, 1, self.B, HOP)
        v, h0, h1, r = self.lengths(x)
        fr = self.frames(x, h0, h1, r)
        re = torch.matmul(fr, self.dft_cos)
        im = torch.matmul(fr, self.dft_sin)
        power = re * re + im * im                                                      # [1, 1, F, 257]
        mel = torch.matmul(power, self.mel_fb)                                         # [1, 1, F, 80]
        logmel = torch.log10(torch.clamp(mel, min=1e-10))
        mx = torch.amax(logmel + (v - 1.0) * 1.0e4, dim=(2, 3), keepdim=True)         # max over the M valid frames
        logmel = torch.maximum(logmel, mx - 8.0) * 0.25 + 1.0
        stacked = logmel[:, :, 0:2 * self.F2, :].reshape(1, self.F2, 2 * N_MELS)       # [1, 1800, 160]
        ve4 = v[:, :, 1:2 * self.F2:2, :]                                              # [1, 1, F2, 1] e < M // 2
        E = torch.sum(ve4, dim=2, keepdim=True)                                        # [1, 1, 1, 1]
        return stacked, v, ve4.reshape(1, self.F2, 1), E


class FeedForward(nn.Module):
    def __init__(self, d, mult):
        super().__init__()
        self.pre_norm = nn.LayerNorm(d)
        self.up_proj = nn.Linear(d, d * mult)
        self.down_proj = nn.Linear(d * mult, d)

    def forward(self, x):
        return self.down_proj(F.silu(self.up_proj(self.pre_norm(x))))


def rel_shift(qw_pad):
    """[1, N, C, 2C] (column 0 = the zero row's product) -> [1, N, C, C] with out[c, r] = qw_pad[c, C - c + r]."""
    B, N, C, C2 = qw_pad.shape
    y = qw_pad.reshape(B, N, C2, C)[:, :, 1:, :]                                       # [1, N, 2C - 1, C]
    return y.reshape(B, N, C, C2 - 1)[:, :, :, 0:C]


class BlockAttention(nn.Module):
    """GraniteSpeechConformerAttention on a static number of full 200-frame blocks, blocks x heads on one axis."""

    def __init__(self, d, heads, dim_head, max_pos_emb, ctx, n_blocks):
        super().__init__()
        self.h, self.dh, self.ctx, self.nb = heads, dim_head, ctx, n_blocks
        self.scale = dim_head ** -0.5
        self.pre_norm = nn.LayerNorm(d)
        self.to_q = nn.Linear(d, heads * dim_head, bias=False)
        self.to_kv = nn.Linear(d, 2 * heads * dim_head, bias=False)
        self.to_out = nn.Linear(heads * dim_head, d)
        self.rel_pos_emb = nn.Embedding(2 * max_pos_emb + 1, dim_head)
        # attention_dists[c, r] = clamp(c - r, -ctx, ctx) + max_pos_emb: rows lo .. lo + 2 ctx - 2 in use
        self.lo, self.nrel = max_pos_emb - (ctx - 1), 2 * ctx - 1
        self.register_buffer("table_pad", torch.zeros(2 * ctx, dim_head), persistent=False)   # set by finalize()

    def finalize(self):
        """table_pad[0] = 0, table_pad[j] = table[2 ctx - 1 - j] (j >= 1): the skew's padded, row-flipped table."""
        t = self.rel_pos_emb.weight.detach()[self.lo:self.lo + self.nrel].to(torch.float32)   # [399, 128]
        tp = torch.zeros(2 * self.ctx, self.dh, dtype=torch.float32)
        tp[1:] = torch.flip(t, dims=[0])
        self.table_pad = tp.contiguous()

    def fold(self, t):
        """[1, T, H * dh] -> [1, nb * H, ctx, dh] (blocks x heads on one axis, rank 4)."""
        return t.reshape(self.nb, self.ctx, self.h, self.dh).permute(0, 2, 1, 3).reshape(1, self.nb * self.h,
                                                                                         self.ctx, self.dh)

    def forward(self, x, key_bias):
        B, T, _ = x.shape
        h = self.pre_norm(x)
        q = self.to_q(h)
        k, v = self.to_kv(h).chunk(2, dim=-1)
        q, k, v = self.fold(q), self.fold(k), self.fold(v)                             # [1, N, 200, 128]
        qw = torch.matmul(q, self.table_pad.transpose(0, 1))                           # [1, N, 200, 400]
        pos = rel_shift(qw) * self.scale                                               # [1, N, 200, 200]
        s = math.sqrt(self.scale)                                                      # torch math SDPA scaling
        att = torch.matmul(q * s, (k * s).transpose(2, 3)) + pos + key_bias            # key_bias [1, N, 1, 200]
        att = torch.softmax(att, dim=-1)
        o = torch.matmul(att, v)                                                       # [1, N, 200, 128]
        o = o.reshape(self.nb, self.h, self.ctx, self.dh).permute(0, 2, 1, 3).reshape(B, T, self.h * self.dh)
        return self.to_out(o)


class DepthConv(nn.Module):
    """Depthwise conv1d (k 15, zero pad 7 / 7) as a rank-4 conv2d on [1, C, 1, T]: the converter's PAD then sits on
    axis 2 (W) of an NHWC [1, 1, T, C] tensor, not on axis 1 of a rank-3 one (LiteRT #9272)."""

    def __init__(self, ch, k):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, k, groups=ch, bias=False)
        self.pad = (k // 2, k // 2 - (k + 1) % 2)

    def forward(self, x):                                                              # [1, C, T]
        C_, T_ = x.shape[1], x.shape[2]
        w = self.conv.weight.unsqueeze(2)                                              # [C, 1, 1, k]
        y = F.conv2d(F.pad(x.reshape(1, C_, 1, T_), self.pad), w, groups=C_)
        return y.reshape(1, C_, T_)


class ConvModule(nn.Module):
    def __init__(self, d, expansion, k):
        super().__init__()
        inner = d * expansion
        self.norm = nn.LayerNorm(d)
        self.up_conv = nn.Conv1d(d, inner * 2, 1)
        self.depth_conv = DepthConv(inner, k)
        self.batch_norm = nn.BatchNorm1d(inner)
        self.down_conv = nn.Conv1d(inner, d, 1)

    def forward(self, x, valid_f):
        h = self.norm(x)                                                               # [1, T, d]
        h = F.linear(h, self.up_conv.weight.squeeze(-1), self.up_conv.bias)            # 1x1 conv as a linear
        a, g = h.chunk(2, dim=-1)
        h = a * torch.sigmoid(g) * valid_f                                             # GLU; zeros after the clip
        h = self.depth_conv(h.transpose(1, 2))                                         # [1, inner, T]
        h = F.batch_norm(h, self.batch_norm.running_mean, self.batch_norm.running_var, self.batch_norm.weight,
                         self.batch_norm.bias, False, 0.0, self.batch_norm.eps)
        h = F.silu(h).transpose(1, 2)
        return F.linear(h, self.down_conv.weight.squeeze(-1), self.down_conv.bias)


class ConformerBlock(nn.Module):
    def __init__(self, c, n_blocks):
        super().__init__()
        d = c["hidden_dim"]
        self.ff1 = FeedForward(d, c["feedforward_mult"])
        self.attn = BlockAttention(d, c["num_heads"], c["dim_head"], c["max_pos_emb"], c["context_size"], n_blocks)
        self.conv = ConvModule(d, c["conv_expansion_factor"], c["conv_kernel_size"])
        self.ff2 = FeedForward(d, c["feedforward_mult"])
        self.post_norm = nn.LayerNorm(d)

    def forward(self, x, key_bias, valid_f):
        x = 0.5 * self.ff1(x) + x
        x = self.attn(x, key_bias) + x
        x = self.conv(x, valid_f) + x
        x = 0.5 * self.ff2(x) + x
        return self.post_norm(x)


class Encoder(nn.Module):
    def __init__(self, c, n_blocks):
        super().__init__()
        self.input_linear = nn.Linear(c["input_dim"], c["hidden_dim"])
        self.layers = nn.ModuleList([ConformerBlock(c, n_blocks) for _ in range(c["num_layers"])])
        self.out = nn.Linear(c["hidden_dim"], c["output_dim"])
        self.out_mid = nn.Linear(c["output_dim"], c["hidden_dim"])
        self.mid = c["num_layers"] // 2

    def forward(self, x, key_bias, valid_f):
        x = self.input_linear(x)
        for i, layer in enumerate(self.layers, start=1):
            x = layer(x, key_bias, valid_f)
            if i == self.mid:
                x = x + self.out_mid(torch.softmax(self.out(x), dim=-1))
        return x


class QAttention(nn.Module):
    """Blip2QFormerAttention: BERT multi-head attention + output dense / residual LayerNorm (runtime query)."""

    def __init__(self, d, heads, eps):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.attention = nn.Module()
        self.attention.query = nn.Linear(d, d)
        self.attention.key = nn.Linear(d, d)
        self.attention.value = nn.Linear(d, d)
        self.output = nn.Module()
        self.output.dense = nn.Linear(d, d)
        self.output.LayerNorm = nn.LayerNorm(d, eps=eps)

    def heads(self, t):
        """[nw, L, d] -> [1, nw * H, L, dh]."""
        nw, L, _ = t.shape
        return t.reshape(nw, L, self.h, self.dh).permute(0, 2, 1, 3).reshape(1, nw * self.h, L, self.dh)

    def merge(self, o, nw, L):
        return o.reshape(nw, self.h, L, self.dh).permute(0, 2, 1, 3).reshape(nw, L, self.h * self.dh)

    def forward(self, x, kv):
        nw, Lq, d = x.shape
        Lk = kv.shape[1]
        q = self.heads(self.attention.query(x))
        k = self.heads(self.attention.key(kv))
        v = self.heads(self.attention.value(kv))
        att = torch.softmax(torch.matmul(q, k.transpose(2, 3)) * (self.dh ** -0.5), dim=-1)
        o = self.merge(torch.matmul(att, v), nw, Lq)
        return self.output.LayerNorm(self.output.dense(o) + x)

    def eager(self, x, kv):
        """port.py's computation on a batch-1 query (used once at load for the constant layer-0 self-attention)."""
        Bq, Lq, d = x.shape
        Bk, Lk, _ = kv.shape
        q = self.attention.query(x).reshape(Bq, Lq, self.h, self.dh).transpose(1, 2)
        k = self.attention.key(kv).reshape(Bk, Lk, self.h, self.dh).transpose(1, 2)
        v = self.attention.value(kv).reshape(Bk, Lk, self.h, self.dh).transpose(1, 2)
        att = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * (self.dh ** -0.5), dim=-1)
        o = torch.matmul(att, v).transpose(1, 2)
        o = o.reshape(o.shape[0], Lq, d)
        return self.output.LayerNorm(self.output.dense(o) + x)


class QCrossConstQuery(nn.Module):
    """Layer 0's cross-attention with a constant query x1 [1, 3, d]: scores = keys x block-diagonal (scaled) query."""

    def __init__(self, att: QAttention):
        super().__init__()
        self._att = [att]                                                              # not a submodule (one owner)
        self.register_buffer("wq_block", torch.zeros(1, 1), persistent=False)          # set by finalize()
        self.register_buffer("x1_tiled", torch.zeros(1, 1, 1), persistent=False)

    def finalize(self, x1, nw):
        a = self._att[0]
        q = a.attention.query(x1)[0]                                                   # [3, d]
        Hh, dh, Lq = a.h, a.dh, q.shape[0]
        qh = q.reshape(Lq, Hh, dh).permute(1, 0, 2) * (dh ** -0.5)                     # [H, 3, dh]
        W = torch.zeros(Hh * dh, Hh * Lq, dtype=torch.float32)
        for h in range(Hh):
            W[h * dh:(h + 1) * dh, h * Lq:(h + 1) * Lq] = qh[h].transpose(0, 1)
        self.wq_block = W.contiguous()                                                 # [d, H * 3]
        self.x1_tiled = x1.contiguous()                  # [1, 3, d] constant (leading 1: broadcast over the windows)
        self.Lq = Lq

    def forward(self, kv):
        a = self._att[0]
        nw, Lk, d = kv.shape
        k = a.attention.key(kv)                                                        # [nw, 15, d]
        sc = torch.matmul(k, self.wq_block)                                            # [nw, 15, H * 3]
        sc = sc.reshape(nw, Lk, a.h, self.Lq).permute(0, 2, 3, 1).reshape(1, nw * a.h, self.Lq, Lk)  # [1, nw H, 3, 15]
        att = torch.softmax(sc, dim=-1)
        v = a.heads(a.attention.value(kv))                                             # [1, nw H, 15, dh]
        o = a.merge(torch.matmul(att, v), nw, self.Lq)                                 # [nw, 3, d]
        return a.output.LayerNorm(a.output.dense(o) + self.x1_tiled)


class QFFN(nn.Module):
    def __init__(self, d, inner, eps):
        super().__init__()
        self.dense = nn.Linear(d, inner)

    def forward(self, x):
        return F.gelu(self.dense(x))


class QOut(nn.Module):
    def __init__(self, inner, d, eps):
        super().__init__()
        self.dense = nn.Linear(inner, d)
        self.LayerNorm = nn.LayerNorm(d, eps=eps)

    def forward(self, h, x):
        return self.LayerNorm(self.dense(h) + x)


class QLayer(nn.Module):
    def __init__(self, d, heads, inner, eps):
        super().__init__()
        self.attention = QAttention(d, heads, eps)
        self.crossattention = QAttention(d, heads, eps)
        self.intermediate_query = QFFN(d, inner, eps)
        self.output_query = QOut(inner, d, eps)

    def forward(self, x, enc):
        x = self.attention(x, x)
        x = self.crossattention(x, enc)
        return self.output_query(self.intermediate_query(x), x)


class Projector(nn.Module):
    def __init__(self, pc, text_hidden, nw):
        super().__init__()
        d, eps = pc["hidden_size"], pc["layer_norm_eps"]
        self.nw = nw
        self.query = nn.Parameter(torch.zeros(1, QUERIES, d))
        self.qformer = nn.Module()
        self.qformer.layernorm = nn.LayerNorm(d, eps=eps)
        self.qformer.encoder = nn.Module()
        self.qformer.encoder.layer = nn.ModuleList(
            [QLayer(d, pc["num_attention_heads"], pc["intermediate_size"], eps) for _ in range(pc["num_hidden_layers"])])
        self.linear = nn.Linear(d, text_hidden)
        self.cross0 = QCrossConstQuery(self.qformer.encoder.layer[0].crossattention)

    @torch.no_grad()
    def finalize(self):
        x0 = self.qformer.layernorm(self.query)                                        # [1, 3, d]
        x1 = self.qformer.encoder.layer[0].attention.eager(x0, x0)                     # [1, 3, d]
        self.cross0.finalize(x1, self.nw)

    def forward(self, enc):
        """enc [1, T, 1024] (T a multiple of 15, rows after the clip already zero) -> [1, T // 5, 2048]."""
        B, T, d = enc.shape
        nw = T // WINDOW
        kv = enc.reshape(nw, WINDOW, d)
        l0, l1 = self.qformer.encoder.layer
        x = self.cross0(kv)                                                            # layer 0 (self-attn constant)
        x = l0.output_query(l0.intermediate_query(x), x)
        x = l1(x, kv)
        return self.linear(x.reshape(1, nw * QUERIES, d))


class GraniteSpeechAudioGraph6(nn.Module):
    def __init__(self, config, win_frames=3600, feature_scale=1.0):
        super().__init__()
        ec, pc, tc = config["encoder_config"], config["projector_config"], config["text_config"]
        assert config["window_size"] == WINDOW and config["window_size"] // config["downsample_rate"] == QUERIES
        assert ec["input_dim"] == 2 * N_MELS and ec["context_size"] == CTX
        self.win_frames = win_frames
        self.frontend = LogMel6(win_frames)
        self.T = self.frontend.F2                                                      # 1800
        assert self.T % CTX == 0 and self.T % WINDOW == 0, self.T
        self.nb, self.H = self.T // CTX, ec["num_heads"]
        self.n_tok = self.T // WINDOW * QUERIES                                        # 360
        assert win_frames % self.n_tok == 0                                            # shrink = 10 exact
        self.encoder = Encoder(ec, self.nb)
        self.projector = Projector(pc, tc["hidden_size"], self.T // WINDOW)
        self.feature_scale = float(feature_scale)
        f32 = torch.float32
        # folded key index: block b, head h, key r -> encoder frame b * 200 + r (constant [1, nb * H, 1, 200])
        fi = (torch.arange(self.nb).reshape(self.nb, 1, 1, 1) * CTX + torch.arange(CTX).reshape(1, 1, 1, CTX))
        fi = fi.expand(self.nb, self.H, 1, CTX).reshape(1, self.nb * self.H, 1, CTX).to(f32)
        self.register_buffer("fold_idx", fi.contiguous(), persistent=False)
        # first encoder frame of each token's projector window: 15 * (t // 3)
        self.register_buffer("tok_start", (torch.arange(self.n_tok) // QUERIES * WINDOW).to(f32).reshape(
            1, 1, 1, self.n_tok), persistent=False)

    def finalize(self):
        for layer in self.encoder.layers:
            layer.attn.finalize()
        self.projector.finalize()
        return self

    def _run(self, audio):
        stacked, v, ve, E = self.frontend(audio)
        key_bias = torch.ge(self.fold_idx, E).to(torch.float32) * NEG                  # [1, nb H, 1, 200]
        enc = self.encoder(stacked, key_bias, ve)
        enc = enc * ve
        proj = self.projector(enc)                                                     # [1, 360, 2048]
        tok_valid = torch.lt(self.tok_start, E).reshape(1, self.n_tok)                 # [1, 360] bool
        return stacked, v, ve, E, enc, proj, tok_valid

    def forward(self, audio):
        _, _, _, _, _, proj, tok_valid = self._run(audio)
        feats = proj * self.feature_scale if self.feature_scale != 1.0 else proj
        return {"features": feats, "mask": tok_valid.to(torch.uint8)}

    def forward_debug(self, audio):
        stacked, v, ve, E, enc, proj, tok_valid = self._run(audio)
        M = torch.sum(v, dim=2, keepdim=True)
        return {"stacked": stacked, "M": M, "E": E, "enc": enc, "proj": proj,
                "n_tok": torch.sum(tok_valid.to(torch.float32), dim=1, keepdim=True)}


def load_graph6(src_dir, win_frames=3600, feature_scale=1.0):
    """Strict-load encoder.* / projector.* from model.safetensors (BF16 -> fp32), then build the constants."""
    import json
    from safetensors import safe_open
    cfg = json.load(open(f"{src_dir}/config.json"))
    m = GraniteSpeechAudioGraph6(cfg, win_frames=win_frames, feature_scale=feature_scale)
    sd = {}
    with safe_open(f"{src_dir}/model.safetensors", framework="pt") as f:
        for k in f.keys():
            if k.startswith("encoder.") or k.startswith("projector."):
                t = f.get_tensor(k)
                sd[k] = t.to(torch.float32) if t.is_floating_point() else t
    res = m.load_state_dict(sd, strict=True)
    assert not res.missing_keys and not res.unexpected_keys, res
    assert len(sd) == len(m.state_dict()), (len(sd), len(m.state_dict()))
    return m.eval().finalize(), len(sd)


def frame_window(wave, win_frames=3600):
    """Runtime framing: int16 / 32768 (or float samples) -> [1, win_frames, 160], the rest of the window zero."""
    import numpy as np
    n = len(wave)
    assert n <= win_frames * HOP, (n, win_frames * HOP)
    buf = np.zeros(win_frames * HOP, np.float32)
    buf[:n] = wave.astype(np.float32) / 32768.0 if wave.dtype == np.int16 else wave.astype(np.float32)
    return torch.from_numpy(buf.reshape(1, win_frames, HOP))
