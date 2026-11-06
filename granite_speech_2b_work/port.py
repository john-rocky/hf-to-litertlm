# Portions adapted from transformers 5.22.1 (Apache License 2.0, Copyright 2025 The HuggingFace Inc. team):
#   models/granite_speech/{feature_extraction,modeling}_granite_speech.py, models/blip_2/modeling_blip_2.py.
"""granite-speech-5.0-2b audio path as ONE torch module for LiteRT-LM's generic audio encoder contract.

  GraniteSpeechAudioGraph.forward(audio f32 [1, 3600, 160]) -> {"features" f32 [1, 360, 2048], "mask" uint8 [1, 360]}

The runtime hands the encoder raw 16 kHz PCM frames (frame = hop = 160 samples, a 36 s window zero-padded) and no
length, so the graph reproduces what transformers computes on the UNPADDED clip:
  * clip length n = index of the last non-zero sample + 1;
  * log-mel = torchaudio MelSpectrogram(16 kHz, n_fft 512, win 400, hop 160, 80 mels, center=True, reflect pad,
    power 2) with the STFT as a DFT matmul (no FFT op); the hann window and the mel filterbank are the tensors
    torchaudio builds (copied as constants); the center pad reflects the CLIP's samples at both ends (a gather on
    an index computed from n), never the window's zeros;
  * log10(clamp 1e-10), max over the clip's M = n // 160 + 1 frames only, floor at max - 8, /4 + 1, drop the odd
    last frame, stack frame pairs -> E = M // 2 encoder frames of 160;
  * 16 conformer blocks: block attention in 200-frame blocks (keys >= E masked, as HF masks its zero-padded last
    block), Shaw rel-pos as q @ table^T + a static gather (shaw_patch.py re-association), GLU output rows >= E
    zeroed before the depthwise conv (HF's conv sees zeros after the clip end), BatchNorm eval, mid-layer softmax
    re-injection after block 8;
  * encoder rows >= E zeroed before the Q-Former (HF zero-pads the last 15-frame window), 2-layer BLIP-2 Q-Former
    with 3 queries per window, linear 1024 -> 2048;
  * mask = 1 for the first ceil(E / 15) * 3 tokens (= GraniteSpeechFeatureExtractor._get_num_audio_features),
    computed without a division: token t is valid iff its window's first frame 15 * (t // 3) < E. The
    converter turns `x / c` into `x * f32(1 / c)` (no DIV op in the tflite); f32(1/15) = 0.0666666701 > 1/15,
    so ceil(E * f32(1/15)) = E / 15 + 1 for 97 of the 120 multiples of 15 in 0..1800 (an earlier export with a
    division gave 5 of 45 test clips 3 extra audio tokens). M likewise uses floor((n + 0.5) / 160), exact under
    either rounding of the reciprocal.
`features` = projector output x `feature_scale`, 1.0 in the published file: the LM export applies
embedding_multiplier 12 inside its prefill / decode graph to every input row, audio rows included, as transformers
does after it scatters the projector output into inputs_embeds.

The class and attribute names are part of the exported file: the converter names every tensor after its module path
(`port.GraniteSpeechAudioGraph/port.Encoder_encoder/...`), so renaming this file, a class or an attribute gives a
TFLite file with other bytes (the same computation).
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

SR, HOP, N_FFT, WIN_LEN, N_MELS = 16000, 160, 512, 400, 80
CTX = 200            # encoder block attention context
WINDOW = 15          # projector window (encoder frames)
QUERIES = 3          # projector queries per window
NEG = -1.0e4         # additive key mask (exp underflows to exactly 0 in fp32; representable in fp16)


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


class LogMelGraph(nn.Module):
    """[1, N] PCM in [-1, 1) -> (stacked log-mel [1, F2, 160], M [1, 1] float, E [1, 1] float)."""

    def __init__(self, n_samples):
        super().__init__()
        assert n_samples % HOP == 0
        self.N = n_samples
        self.F = n_samples // HOP + 1                     # max mel frames (n = N) = 3601
        self.F2 = (self.F - 1) // 2                       # encoder frames kept by the static graph = 1800
        self.L = (self.F + 2) * HOP                       # gathered signal length (cat of 3 blocks of 160)
        window, fb = torchaudio_constants()
        f32 = torch.float32
        # frame t covers x_reflect[t*160 - 200 : t*160 + 200]: the 400 non-zero taps of the window that torch.stft
        # centres inside n_fft = 512 (left pad (512 - 400) // 2 = 56, so window tap 0 sits at frame offset 56)
        n = torch.arange(56, 56 + WIN_LEN, dtype=torch.float64).reshape(WIN_LEN, 1)
        k = torch.arange(N_FFT // 2 + 1, dtype=torch.float64).reshape(1, N_FFT // 2 + 1)
        ang = 2.0 * math.pi * torch.remainder(n * k, N_FFT) / N_FFT
        self.register_buffer("dft_cos", (torch.cos(ang) * window.double().reshape(WIN_LEN, 1)).to(f32).contiguous(),
                             persistent=False)
        self.register_buffer("dft_sin", (torch.sin(ang) * window.double().reshape(WIN_LEN, 1)).to(f32).contiguous(),
                             persistent=False)
        self.register_buffer("mel_fb", fb.contiguous(), persistent=False)
        self.register_buffer("sample_idx", torch.arange(n_samples, dtype=f32).reshape(1, n_samples), persistent=False)
        self.register_buffer("gpos", (torch.arange(self.L, dtype=f32) - 200.0).reshape(1, self.L), persistent=False)
        self.register_buffer("frame_idx", torch.arange(self.F, dtype=f32).reshape(self.F, 1), persistent=False)

    def forward(self, x, n_valid=None):
        if n_valid is None:
            nz = (x != 0.0).to(x.dtype)
            n_valid = torch.amax(self.sample_idx * nz, dim=1, keepdim=True) + 1.0          # [1, 1]
        i = self.gpos                                                                    # x index of each tap
        src = torch.where(i < 0.0, -i, torch.where(i < n_valid, i, 2.0 * n_valid - 2.0 - i))
        src = torch.clamp(src, 0.0, float(self.N - 1)).to(torch.int32).reshape(self.L)
        y = torch.index_select(x.reshape(self.N), 0, src)                                # [L] reflect-padded clip
        b = y.reshape(self.F + 2, HOP)
        frames = torch.cat([b[0:self.F], b[1:self.F + 1], b[2:self.F + 2, 0:WIN_LEN - 2 * HOP]], dim=1)   # [F, 400]
        re = torch.matmul(frames, self.dft_cos)
        im = torch.matmul(frames, self.dft_sin)
        power = re * re + im * im                                                        # [F, 257]
        mel = torch.matmul(power, self.mel_fb)                                           # [F, 80]
        logmel = torch.log10(torch.clamp(mel, min=1e-10))
        M = torch.floor((n_valid + 0.5) / float(HOP)) + 1.0                              # [1, 1] = n // 160 + 1
        valid = self.frame_idx < M                                                       # [F, 1]
        mx = torch.amax(torch.where(valid, logmel, torch.full_like(logmel, -10.0)), dim=(0, 1), keepdim=True)
        logmel = torch.maximum(logmel, mx - 8.0) / 4.0 + 1.0
        stacked = logmel[0:2 * self.F2].reshape(1, self.F2, 2 * N_MELS)                  # [1, 1800, 160]
        E = torch.floor(M / 2.0)
        return stacked, M, E


class FeedForward(nn.Module):
    def __init__(self, d, mult):
        super().__init__()
        self.pre_norm = nn.LayerNorm(d)
        self.up_proj = nn.Linear(d, d * mult)
        self.down_proj = nn.Linear(d * mult, d)

    def forward(self, x):
        return self.down_proj(F.silu(self.up_proj(self.pre_norm(x))))


class BlockAttention(nn.Module):
    """GraniteSpeechConformerAttention on a static number of full 200-frame blocks with a key mask."""

    def __init__(self, d, heads, dim_head, max_pos_emb, ctx):
        super().__init__()
        self.h, self.dh, self.ctx = heads, dim_head, ctx
        self.scale = dim_head ** -0.5
        self.pre_norm = nn.LayerNorm(d)
        self.to_q = nn.Linear(d, heads * dim_head, bias=False)
        self.to_kv = nn.Linear(d, 2 * heads * dim_head, bias=False)
        self.to_out = nn.Linear(heads * dim_head, d)
        self.rel_pos_emb = nn.Embedding(2 * max_pos_emb + 1, dim_head)
        # attention_dists[c, r] = clamp(c - r, -ctx, ctx) + max_pos_emb in [max_pos_emb - ctx + 1, max_pos_emb + ctx - 1]
        lo = max_pos_emb - (ctx - 1)
        self.lo, self.nrel = lo, 2 * ctx - 1
        c = torch.arange(ctx).reshape(ctx, 1)
        r = torch.arange(ctx).reshape(1, ctx)
        self.register_buffer("rel_idx", (c - r + (ctx - 1)).reshape(1, 1, ctx, ctx).contiguous(), persistent=False)

    def forward(self, x, key_bias):
        B, T, _ = x.shape
        nb = T // self.ctx
        h = self.pre_norm(x)
        q = self.to_q(h)
        k, v = self.to_kv(h).chunk(2, dim=-1)
        q = q.reshape(nb, self.ctx, self.h, self.dh).transpose(1, 2)                     # [nb, H, 200, 128]
        k = k.reshape(nb, self.ctx, self.h, self.dh).transpose(1, 2)
        v = v.reshape(nb, self.ctx, self.h, self.dh).transpose(1, 2)
        table = self.rel_pos_emb.weight[self.lo:self.lo + self.nrel]                       # [399, 128] rows in use
        qw = torch.matmul(q, table.transpose(0, 1))                                      # [nb, H, 200, 399]
        pos = torch.gather(qw, 3, self.rel_idx.expand(nb, self.h, self.ctx, self.ctx)) * self.scale
        s = math.sqrt(self.scale)                                                        # torch math SDPA scaling
        att = torch.matmul(q * s, (k * s).transpose(-2, -1)) + pos + key_bias            # key_bias [nb, 1, 1, 200]
        att = torch.softmax(att, dim=-1)
        o = torch.matmul(att, v).transpose(1, 2).reshape(B, T, self.h * self.dh)
        return self.to_out(o)


class DepthConv(nn.Module):
    def __init__(self, ch, k):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, k, groups=ch, bias=False)
        self.pad = (k // 2, k // 2 - (k + 1) % 2)

    def forward(self, x):
        return self.conv(F.pad(x, self.pad))


class ConvModule(nn.Module):
    def __init__(self, d, expansion, k):
        super().__init__()
        inner = d * expansion
        self.norm = nn.LayerNorm(d)
        self.up_conv = nn.Conv1d(d, inner * 2, 1)
        self.depth_conv = DepthConv(inner, k)
        self.batch_norm = nn.BatchNorm1d(inner)
        self.down_conv = nn.Conv1d(inner, d, 1)

    def forward(self, x, valid_t):
        h = self.norm(x)                                                                 # [1, T, d]
        h = F.linear(h, self.up_conv.weight.squeeze(-1), self.up_conv.bias)              # 1x1 conv as a linear
        a, g = h.chunk(2, dim=-1)
        h = a * torch.sigmoid(g)                                                         # GLU over channels
        h = torch.where(valid_t, h, torch.zeros_like(h))                                 # zeros after the clip end
        h = self.depth_conv(h.transpose(1, 2))                                           # [1, inner, T]
        h = F.batch_norm(h, self.batch_norm.running_mean, self.batch_norm.running_var, self.batch_norm.weight,
                         self.batch_norm.bias, False, 0.0, self.batch_norm.eps)
        h = F.silu(h).transpose(1, 2)
        return F.linear(h, self.down_conv.weight.squeeze(-1), self.down_conv.bias)


class ConformerBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        d = c["hidden_dim"]
        self.ff1 = FeedForward(d, c["feedforward_mult"])
        self.attn = BlockAttention(d, c["num_heads"], c["dim_head"], c["max_pos_emb"], c["context_size"])
        self.conv = ConvModule(d, c["conv_expansion_factor"], c["conv_kernel_size"])
        self.ff2 = FeedForward(d, c["feedforward_mult"])
        self.post_norm = nn.LayerNorm(d)

    def forward(self, x, key_bias, valid_t):
        x = 0.5 * self.ff1(x) + x
        x = self.attn(x, key_bias) + x
        x = self.conv(x, valid_t) + x
        x = 0.5 * self.ff2(x) + x
        return self.post_norm(x)


class Encoder(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.input_linear = nn.Linear(c["input_dim"], c["hidden_dim"])
        self.layers = nn.ModuleList([ConformerBlock(c) for _ in range(c["num_layers"])])
        self.out = nn.Linear(c["hidden_dim"], c["output_dim"])
        self.out_mid = nn.Linear(c["output_dim"], c["hidden_dim"])
        self.mid = c["num_layers"] // 2

    def forward(self, x, key_bias, valid_t):
        x = self.input_linear(x)
        for i, layer in enumerate(self.layers, start=1):
            x = layer(x, key_bias, valid_t)
            if i == self.mid:
                x = x + self.out_mid(torch.softmax(self.out(x), dim=-1))
        return x


class QAttention(nn.Module):
    """Blip2QFormerAttention: BERT multi-head attention + output dense / residual LayerNorm."""

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

    def forward(self, x, kv):
        Bq, Lq, d = x.shape
        Bk, Lk, _ = kv.shape
        q = self.attention.query(x).reshape(Bq, Lq, self.h, self.dh).transpose(1, 2)
        k = self.attention.key(kv).reshape(Bk, Lk, self.h, self.dh).transpose(1, 2)
        v = self.attention.value(kv).reshape(Bk, Lk, self.h, self.dh).transpose(1, 2)
        att = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * (self.dh ** -0.5), dim=-1)
        o = torch.matmul(att, v).transpose(1, 2)
        o = o.reshape(o.shape[0], Lq, d)
        return self.output.LayerNorm(self.output.dense(o) + x)


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
    def __init__(self, pc, text_hidden):
        super().__init__()
        d, eps = pc["hidden_size"], pc["layer_norm_eps"]
        self.query = nn.Parameter(torch.zeros(1, QUERIES, d))
        self.qformer = nn.Module()
        self.qformer.layernorm = nn.LayerNorm(d, eps=eps)
        self.qformer.encoder = nn.Module()
        self.qformer.encoder.layer = nn.ModuleList(
            [QLayer(d, pc["num_attention_heads"], pc["intermediate_size"], eps) for _ in range(pc["num_hidden_layers"])])
        self.linear = nn.Linear(d, text_hidden)

    def forward(self, enc):
        """enc [1, T, 1024] (T a multiple of 15, rows after the clip already zero) -> [1, T // 5, 2048]."""
        B, T, d = enc.shape
        nw = T // WINDOW
        kv = enc.reshape(nw, WINDOW, d)
        x = self.qformer.layernorm(self.query)                                           # [1, 3, d], broadcast
        for layer in self.qformer.encoder.layer:
            x = layer(x, kv)
        return self.linear(x.reshape(1, nw * QUERIES, d))


class GraniteSpeechAudioGraph(nn.Module):
    def __init__(self, config, win_frames=3600, feature_scale=1.0):
        super().__init__()
        ec, pc, tc = config["encoder_config"], config["projector_config"], config["text_config"]
        assert config["window_size"] == WINDOW and config["window_size"] // config["downsample_rate"] == QUERIES
        assert ec["input_dim"] == 2 * N_MELS and ec["context_size"] == CTX
        self.win_frames = win_frames
        self.n_samples = win_frames * HOP
        self.frontend = LogMelGraph(self.n_samples)
        self.T = self.frontend.F2                                                        # 1800
        assert self.T % CTX == 0 and self.T % WINDOW == 0, self.T
        self.n_tok = self.T // WINDOW * QUERIES                                          # 360
        assert win_frames % self.n_tok == 0                                              # shrink = 10 exact
        self.encoder = Encoder(ec)
        self.projector = Projector(pc, tc["hidden_size"])
        self.feature_scale = float(feature_scale)
        f32 = torch.float32
        self.register_buffer("t_idx", torch.arange(self.T, dtype=f32).reshape(1, self.T, 1), persistent=False)
        self.register_buffer("tok_idx", torch.arange(self.n_tok, dtype=f32).reshape(1, self.n_tok), persistent=False)
        # first encoder frame of each token's projector window: 15 * (t // 3)
        self.register_buffer("tok_start", (torch.arange(self.n_tok) // QUERIES * WINDOW).to(f32).reshape(1, self.n_tok),
                             persistent=False)

    def _run(self, audio, n_valid=None):
        x = audio.reshape(1, self.n_samples)
        stacked, M, E = self.frontend(x, n_valid)
        valid_t = self.t_idx < E                                                         # [1, T, 1] bool
        key_bias = ((self.t_idx >= E).to(torch.float32) * NEG).reshape(self.T // CTX, 1, 1, CTX)
        enc = self.encoder(stacked, key_bias, valid_t)
        enc = torch.where(valid_t, enc, torch.zeros_like(enc))
        proj = self.projector(enc)                                                       # [1, 360, 2048]
        tok_valid = self.tok_start < E                                                   # [1, 360] bool
        n_tok = torch.sum(tok_valid.to(torch.float32), dim=1, keepdim=True)              # = ceil(E / 15) * 3
        return stacked, M, E, enc, proj, n_tok, tok_valid

    def forward(self, audio):
        _, _, _, _, proj, _, tok_valid = self._run(audio)
        feats = proj * self.feature_scale if self.feature_scale != 1.0 else proj
        return {"features": feats, "mask": tok_valid.to(torch.uint8)}

    def forward_debug(self, audio, n_valid=None):
        stacked, M, E, enc, proj, n_tok, _ = self._run(audio, n_valid)
        return {"stacked": stacked, "M": M, "E": E, "enc": enc, "proj": proj, "n_tok": n_tok}


def load_graph(src_dir, win_frames=3600, feature_scale=1.0):
    """Strict-load encoder.* / projector.* from model.safetensors (BF16 -> fp32)."""
    import json
    from safetensors import safe_open
    cfg = json.load(open(f"{src_dir}/config.json"))
    m = GraniteSpeechAudioGraph(cfg, win_frames=win_frames, feature_scale=feature_scale)
    sd = {}
    with safe_open(f"{src_dir}/model.safetensors", framework="pt") as f:
        for k in f.keys():
            if k.startswith("encoder.") or k.startswith("projector."):
                t = f.get_tensor(k)
                sd[k] = t.to(torch.float32) if t.is_floating_point() else t
    res = m.load_state_dict(sd, strict=True)
    assert not res.missing_keys and not res.unexpected_keys, res
    assert len(sd) == len(m.state_dict()), (len(sd), len(m.state_dict()))
    return m.eval(), len(sd)


def frame_window(wave_i16, win_frames=3600):
    """Runtime framing: int16 / 32768 -> [1, win_frames, 160], the rest of the window zero."""
    import numpy as np
    n = len(wave_i16)
    assert n <= win_frames * HOP, (n, win_frames * HOP)
    buf = np.zeros(win_frames * HOP, np.float32)
    buf[:n] = wave_i16.astype(np.float32) / 32768.0
    return torch.from_numpy(buf.reshape(1, win_frames, HOP))
