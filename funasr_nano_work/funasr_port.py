# Copyright (c) Alibaba, Inc. and its affiliates (FunASR, https://github.com/modelscope/FunASR).
# Speech Lab of DAMO Academy, Alibaba Group (SenseVoiceEncoderSmall / SAN-M).
# Licensed under the Apache License, Version 2.0.
#   funasr/models/sense_voice/model.py, funasr/models/llm_asr/adaptor.py,
#   funasr/frontends/wav_frontend.py (apply_lfr), funasr 1.4.16.
# Portions Copyright 2019 Shigeki Karita (ESPnet), Apache License 2.0:
#   funasr/models/transformer/{encoder,attention,positionwise_feed_forward,layer_norm}.py.
# Kaldi fbank reference adapted from torchaudio.compliance.kaldi (torchaudio 2.11.0), BSD 2-Clause:
#   Copyright (c) 2017 Facebook Inc. (Soumith Chintala). All rights reserved.
#
# Self-contained PyTorch port of the Fun-ASR-Nano-2512 audio path (frontend + SAN-M encoder +
# adaptor). Does NOT import funasr. Two frontends:
#   * reference (RefFrontend*): the exact funasr path on the true-length waveform (torchaudio kaldi
#     fbank + funasr apply_lfr), used to check the vendored modules against the oracle dumps;
#   * graph (KaldiFbankLfrGraph): the fixed 30.24 s zero-padded window the LiteRT-LM runtime feeds
#     (design steps 1-5: static framing, DFT as constant matmuls, length inferred from the last
#     non-zero sample, tail frames replaced by the last valid frame, LFR by reshapes).
# FunAsrNanoAudioEncoder.forward(audio [1,504,960]) -> {"features" [1,63,1024], "mask" uint8 [1,63]}.
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

SR = 16000
WIN_LEN, HOP = 400, 160           # 25 ms / 10 ms
N_FFT = 512                       # round_to_power_of_two
N_MELS = 80
LFR_M, LFR_N = 7, 6
EPS = float(torch.finfo(torch.float32).eps)  # 1.1920928955078125e-07 (kaldi log floor)
FRAME = 960                       # runtime frame = hop = one LFR step
WIN_FRAMES = 504                  # 30.24 s window
WIN_SAMPLES = FRAME * WIN_FRAMES  # 483840
OUT_TOKENS = WIN_FRAMES // 8      # 63
NEG = -1.0e4                      # additive attention mask (finite; softmax result identical to -inf)


# ----------------------------------------------------------------------------------------------
# Reference frontend: torchaudio.compliance.kaldi.fbank (the backend funasr.utils.fbank dispatches
# to when torchaudio imports) restricted to the WavFrontend options, + funasr apply_lfr verbatim.
# ----------------------------------------------------------------------------------------------
def _next_power_of_2(x: int) -> int:
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def mel_scale_scalar(freq: float) -> float:
    return 1127.0 * math.log(1.0 + freq / 700.0)


def mel_scale(freq):
    return 1127.0 * (1.0 + freq / 700.0).log()


def get_mel_banks(num_bins, window_length_padded, sample_freq, low_freq, high_freq):
    """torchaudio get_mel_banks with vtln_warp_factor == 1.0. Returns [num_bins, window_length_padded/2]."""
    assert num_bins > 3 and window_length_padded % 2 == 0
    num_fft_bins = window_length_padded / 2
    nyquist = 0.5 * sample_freq
    if high_freq <= 0.0:
        high_freq += nyquist
    assert (0.0 <= low_freq < nyquist) and (0.0 < high_freq <= nyquist) and (low_freq < high_freq)
    fft_bin_width = sample_freq / window_length_padded
    mel_low_freq = mel_scale_scalar(low_freq)
    mel_high_freq = mel_scale_scalar(high_freq)
    mel_freq_delta = (mel_high_freq - mel_low_freq) / (num_bins + 1)
    bin = torch.arange(num_bins).unsqueeze(1)
    left_mel = mel_low_freq + bin * mel_freq_delta
    center_mel = mel_low_freq + (bin + 1.0) * mel_freq_delta
    right_mel = mel_low_freq + (bin + 2.0) * mel_freq_delta
    mel = mel_scale(fft_bin_width * torch.arange(num_fft_bins)).unsqueeze(0)
    up_slope = (mel - left_mel) / (center_mel - left_mel)
    down_slope = (right_mel - mel) / (right_mel - center_mel)
    return torch.max(torch.zeros(1), torch.min(up_slope, down_slope))


def kaldi_mel_matrix():
    """[257, 80] float32: the matrix torchaudio's fbank multiplies the power spectrum with."""
    m = get_mel_banks(N_MELS, N_FFT, float(SR), 20.0, 0.0)
    m = F.pad(m, (0, 1), mode="constant", value=0)  # Nyquist column = 0
    return m.T.contiguous()


def kaldi_fbank_ref(waveform, frame_length=25.0):
    """torchaudio.compliance.kaldi.fbank(waveform[1,n], num_mel_bins=80, frame_length, frame_shift=10,
    dither=0, energy_floor=0, window_type='hamming', sample_frequency=16000, snip_edges=True)."""
    wav = waveform[0]
    window_shift = int(SR * 10.0 * 0.001)
    window_size = int(SR * frame_length * 0.001)
    padded = _next_power_of_2(window_size)
    assert 2 <= window_size <= len(wav)
    m = 1 + (wav.size(0) - window_size) // window_shift
    strided = wav.as_strided((m, window_size), (window_shift * wav.stride(0), wav.stride(0)))
    strided = strided - torch.mean(strided, dim=1).unsqueeze(1)                     # remove_dc_offset
    off = F.pad(strided.unsqueeze(0), (1, 0), mode="replicate").squeeze(0)          # preemphasis 0.97
    strided = strided - 0.97 * off[:, :-1]
    win = torch.hamming_window(window_size, periodic=False, alpha=0.54, beta=0.46, dtype=wav.dtype)
    strided = strided * win.unsqueeze(0)
    if padded != window_size:
        strided = F.pad(strided.unsqueeze(0), (0, padded - window_size), mode="constant", value=0).squeeze(0)
    spectrum = torch.fft.rfft(strided).abs().pow(2.0)
    mel = get_mel_banks(N_MELS, padded, float(SR), 20.0, 0.0).to(wav.dtype)
    mel = F.pad(mel, (0, 1), mode="constant", value=0)
    e = torch.mm(spectrum, mel.T)
    return torch.max(e, torch.tensor(EPS, dtype=wav.dtype)).log()


def apply_lfr_ref(inputs, lfr_m=LFR_M, lfr_n=LFR_N):
    """funasr/frontends/wav_frontend.py apply_lfr (verbatim logic)."""
    T = inputs.shape[0]
    T_lfr = int(np.ceil(T / lfr_n))
    left_padding = inputs[0].repeat((lfr_m - 1) // 2, 1)
    inputs = torch.vstack((left_padding, inputs))
    T = T + (lfr_m - 1) // 2
    feat_dim = inputs.shape[-1]
    strides = (lfr_n * feat_dim, 1)
    sizes = (T_lfr, lfr_m * feat_dim)
    last_idx = (T - lfr_m) // lfr_n + 1
    num_padding = lfr_m - (T - last_idx * lfr_n)
    if num_padding > 0:
        num_padding = (2 * lfr_m - 2 * T + (T_lfr - 1 + last_idx) * lfr_n) / 2 * (T_lfr - last_idx)
        inputs = torch.vstack([inputs] + [inputs[-1:]] * int(num_padding))
    return inputs.as_strided(sizes, strides).clone().type(torch.float32)


def ref_frontend(wave):
    """funasr WavFrontend.forward for one float waveform in [-1, 1) (torchaudio/soundfile load):
    x * 2**15 -> kaldi fbank (frame_length = min(25, ms)) -> LFR 7/6, no CMVN. Returns [1, L, 560]."""
    n = wave.shape[-1]
    x = (wave.reshape(1, n) * (1 << 15)).float()
    fb = kaldi_fbank_ref(x, frame_length=min(25, n / SR * 1000))
    return apply_lfr_ref(fb)[None]


# ----------------------------------------------------------------------------------------------
# Graph frontend for the fixed runtime window (design steps 1-5).
# ----------------------------------------------------------------------------------------------
class KaldiFbankLfrGraph(nn.Module):
    def __init__(self, n_samples=WIN_SAMPLES):
        super().__init__()
        assert n_samples % HOP == 0
        self.N = n_samples
        self.NB = n_samples // HOP                        # 3024 blocks of 160 samples
        self.F = 1 + (n_samples - WIN_LEN) // HOP         # 3022 kaldi frames (snip_edges)
        assert self.F == self.NB - 2
        self.T = math.ceil(self.F / LFR_N)                # 504 LFR rows
        assert 3 + self.F == LFR_N * (self.T - 1) + LFR_M  # 3025 rows cover exactly T LFR rows
        f32 = torch.float32
        self.register_buffer("sample_idx", torch.arange(n_samples, dtype=f32).reshape(1, n_samples).contiguous(), persistent=False)
        self.register_buffer("frame_idx_col", torch.arange(self.F, dtype=f32).reshape(self.F, 1).contiguous(), persistent=False)
        self.register_buffer("frame_idx_row", torch.arange(self.F, dtype=f32).reshape(1, self.F).contiguous(), persistent=False)
        win = torch.hamming_window(WIN_LEN, periodic=False, alpha=0.54, beta=0.46, dtype=f32)
        self.register_buffer("window", win.reshape(1, WIN_LEN).contiguous(), persistent=False)
        n = torch.arange(N_FFT, dtype=torch.float64).reshape(N_FFT, 1)
        k = torch.arange(N_FFT // 2 + 1, dtype=torch.float64).reshape(1, N_FFT // 2 + 1)
        ang = 2.0 * math.pi * torch.remainder(n * k, N_FFT) / N_FFT
        self.register_buffer("dft_cos", torch.cos(ang).to(f32).contiguous(), persistent=False)
        self.register_buffer("dft_sin", torch.sin(ang).to(f32).contiguous(), persistent=False)
        self.register_buffer("mel_t", kaldi_mel_matrix().to(f32).contiguous(), persistent=False)

    def forward(self, audio, n_valid=None):
        """audio [1, 504, 960] f32 in [-1, 1) -> (lfr [1, 504, 560], L [1, 1] float).
        n_valid ([1, 1] float) overrides the in-graph length (measurement only; not exported)."""
        x = audio.reshape(1, self.N) * 32768.0
        if n_valid is None:
            nz = (x != 0.0).to(x.dtype)
            n_valid = torch.amax(self.sample_idx * nz, dim=1, keepdim=True) + 1.0      # [1, 1]
        b = x.reshape(self.NB, HOP)
        frames = torch.cat([b[0:self.F], b[1:self.F + 1], b[2:self.F + 2, 0:WIN_LEN - 2 * HOP]], dim=1)  # [3022, 400]
        frames = frames - frames.mean(dim=1, keepdim=True)                           # DC removal
        prev = torch.cat([frames[:, 0:1], frames[:, 0:WIN_LEN - 1]], dim=1)
        frames = frames - 0.97 * prev                                                 # pre-emphasis
        frames = frames * self.window                                                 # hamming
        frames = F.pad(frames, (0, N_FFT - WIN_LEN))                                  # [3022, 512]
        re = torch.matmul(frames, self.dft_cos)
        im = torch.matmul(frames, self.dft_sin)
        power = re * re + im * im                                                     # [3022, 257]
        fb = torch.log(torch.clamp(torch.matmul(power, self.mel_t), min=EPS))         # [3022, 80]
        L_fb = torch.clamp(torch.floor((n_valid - float(WIN_LEN)) / float(HOP)) + 1.0, 1.0, float(self.F))  # [1, 1]
        keep = self.frame_idx_col < L_fb                                              # [3022, 1]
        onehot = (self.frame_idx_row == (L_fb - 1.0)).to(fb.dtype)                    # [1, 3022]
        last = torch.matmul(onehot, fb)                                               # [1, 80]
        fb = torch.where(keep, fb, last)                                              # tail = last valid frame
        f0 = fb[0:1]
        padded = torch.cat([f0, f0, f0, fb], dim=0)                                   # [3025, 80]
        six = padded[0:LFR_N * self.T].reshape(self.T, LFR_N * N_MELS)                # rows 6j..6j+5
        seventh = padded[1:LFR_N * self.T + 1].reshape(self.T, LFR_N, N_MELS)[:, LFR_N - 1, :]  # row 6j+6
        lfr = torch.cat([six, seventh], dim=1).reshape(1, self.T, LFR_M * N_MELS)
        L = torch.ceil(L_fb / float(LFR_N))                                           # [1, 1]
        return lfr, L


# ----------------------------------------------------------------------------------------------
# SenseVoiceEncoderSmall (funasr/models/sense_voice/model.py), mask passed as tensors.
# ----------------------------------------------------------------------------------------------
def sinusoidal_pe(timesteps, depth, dtype=torch.float32):
    """SinusoidalPositionEncoder.encode for positions 1..timesteps (verbatim arithmetic)."""
    positions = torch.arange(1, timesteps + 1)[None, :].type(dtype)
    log_timescale_increment = torch.log(torch.tensor([10000], dtype=dtype)) / (depth / 2 - 1)
    inv_timescales = torch.exp(torch.arange(depth / 2).type(dtype) * (-log_timescale_increment))
    inv_timescales = torch.reshape(inv_timescales, [1, -1])
    scaled_time = torch.reshape(positions, [1, -1, 1]) * torch.reshape(inv_timescales, [1, 1, -1])
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=2).type(dtype)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, idim, hidden_units):
        super().__init__()
        self.w_1 = nn.Linear(idim, hidden_units)
        self.w_2 = nn.Linear(hidden_units, idim)

    def forward(self, x):
        return self.w_2(torch.relu(self.w_1(x)))


class MultiHeadedAttentionSANM(nn.Module):
    def __init__(self, n_head, in_feat, n_feat, kernel_size=11, sanm_shfit=0):
        super().__init__()
        assert n_feat % n_head == 0
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.linear_q_k_v = nn.Linear(in_feat, n_feat * 3)
        self.fsmn_block = nn.Conv1d(n_feat, n_feat, kernel_size, stride=1, padding=0, groups=n_feat, bias=False)
        left = (kernel_size - 1) // 2 + sanm_shfit
        self.pad = (left, kernel_size - 1 - left)

    def forward(self, x, mask_t, mask_add):
        # mask_t [1, T, 1] (1 valid / 0 pad) or None; mask_add [1, 1, 1, T] (0 / NEG) or None
        b, t, _ = x.shape
        q, k, v = torch.split(self.linear_q_k_v(x), self.h * self.d_k, dim=-1)
        q_h = q.reshape(b, t, self.h, self.d_k).transpose(1, 2)
        k_h = k.reshape(b, t, self.h, self.d_k).transpose(1, 2)
        v_h = v.reshape(b, t, self.h, self.d_k).transpose(1, 2)
        inputs = v if mask_t is None else v * mask_t                                   # forward_fsmn
        y = self.fsmn_block(F.pad(inputs.transpose(1, 2), self.pad)).transpose(1, 2)
        y = y + inputs
        if mask_t is not None:
            y = y * mask_t
        q_h = q_h * self.d_k ** (-0.5)
        scores = torch.matmul(q_h, k_h.transpose(-2, -1))
        if mask_add is not None:
            scores = scores + mask_add
        attn = torch.softmax(scores, dim=-1)
        o = torch.matmul(attn, v_h).transpose(1, 2).reshape(b, t, self.h * self.d_k)
        return self.linear_out(o) + y


class EncoderLayerSANM(nn.Module):
    def __init__(self, in_size, size, self_attn, feed_forward):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = nn.LayerNorm(in_size, eps=1e-5)   # sense_voice LayerNorm == nn.LayerNorm defaults
        self.norm2 = nn.LayerNorm(size, eps=1e-5)
        self.in_size, self.size = in_size, size

    def forward(self, x, mask_t, mask_add):
        residual = x
        a = self.self_attn(self.norm1(x), mask_t, mask_add)
        x = residual + a if self.in_size == self.size else a
        return x + self.feed_forward(self.norm2(x))


class SenseVoiceEncoderSmall(nn.Module):
    def __init__(self, input_size=560, output_size=512, attention_heads=4, linear_units=2048, num_blocks=50,
                 tp_blocks=20, kernel_size=11, sanm_shfit=0, max_len=WIN_FRAMES):
        super().__init__()
        self._output_size = output_size
        mk = lambda i: EncoderLayerSANM(  # noqa: E731
            i, output_size, MultiHeadedAttentionSANM(attention_heads, i, output_size, kernel_size, sanm_shfit),
            PositionwiseFeedForward(output_size, linear_units))
        self.encoders0 = nn.ModuleList([mk(input_size)])
        self.encoders = nn.ModuleList([mk(output_size) for _ in range(num_blocks - 1)])
        self.tp_encoders = nn.ModuleList([mk(output_size) for _ in range(tp_blocks)])
        self.after_norm = nn.LayerNorm(output_size, eps=1e-5)
        self.tp_norm = nn.LayerNorm(output_size, eps=1e-5)
        self.register_buffer("pe", sinusoidal_pe(max_len, input_size).contiguous(), persistent=False)

    def forward(self, xs, mask_t=None, mask_add=None):
        t = xs.shape[1]
        xs = xs * (self._output_size ** 0.5)
        pe = self.pe if t == self.pe.shape[1] else sinusoidal_pe(t, xs.shape[2])
        xs = xs + pe
        for layer in self.encoders0:
            xs = layer(xs, mask_t, mask_add)
        for layer in self.encoders:
            xs = layer(xs, mask_t, mask_add)
        xs = self.after_norm(xs)
        for layer in self.tp_encoders:
            xs = layer(xs, mask_t, mask_add)
        return self.tp_norm(xs)


# ----------------------------------------------------------------------------------------------
# Adaptor: funasr/models/llm_asr/adaptor.py Transformer (k=1) with ESPnet EncoderLayer blocks.
# ----------------------------------------------------------------------------------------------
class MultiHeadedAttention(nn.Module):
    def __init__(self, n_head, n_feat):
        super().__init__()
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)

    def forward(self, x, mask_add):
        b, t, _ = x.shape
        q = self.linear_q(x).reshape(b, t, self.h, self.d_k).transpose(1, 2)
        k = self.linear_k(x).reshape(b, t, self.h, self.d_k).transpose(1, 2)
        v = self.linear_v(x).reshape(b, t, self.h, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask_add is not None:
            scores = scores + mask_add
        attn = torch.softmax(scores, dim=-1)
        o = torch.matmul(attn, v).transpose(1, 2).reshape(b, t, self.h * self.d_k)
        return self.linear_out(o)


class EncoderLayer(nn.Module):
    def __init__(self, size, n_head):
        super().__init__()
        self.self_attn = MultiHeadedAttention(n_head, size)
        self.feed_forward = PositionwiseFeedForward(size, size // 4)
        self.norm1 = nn.LayerNorm(size, eps=1e-12)     # funasr.models.transformer.layer_norm.LayerNorm
        self.norm2 = nn.LayerNorm(size, eps=1e-12)

    def forward(self, x, mask_add):
        x = x + self.self_attn(self.norm1(x), mask_add)
        return x + self.feed_forward(self.norm2(x))


class AudioAdaptor(nn.Module):
    def __init__(self, encoder_dim=512, llm_dim=1024, ffn_dim=2048, n_layer=2, attention_heads=8):
        super().__init__()
        self.linear1 = nn.Linear(encoder_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, llm_dim)
        self.blocks = nn.ModuleList([EncoderLayer(llm_dim, attention_heads) for _ in range(n_layer)])

    def forward(self, x, mask_add=None):
        x = self.linear2(torch.relu(self.linear1(x)))
        for blk in self.blocks:
            x = blk(x, mask_add)
        return x


# ----------------------------------------------------------------------------------------------
# The exported module.
# ----------------------------------------------------------------------------------------------
class FunAsrNanoAudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.frontend = KaldiFbankLfrGraph(WIN_SAMPLES)
        self.audio_encoder = SenseVoiceEncoderSmall()
        self.audio_adaptor = AudioAdaptor()
        self.register_buffer("row_idx", torch.arange(WIN_FRAMES, dtype=torch.float32).reshape(1, WIN_FRAMES).contiguous(), persistent=False)
        self.register_buffer("tok_idx", torch.arange(OUT_TOKENS, dtype=torch.float32).reshape(1, OUT_TOKENS).contiguous(), persistent=False)

    def _run(self, audio, n_valid=None):
        lfr, L = self.frontend(audio, n_valid)
        valid = (self.row_idx < L).to(lfr.dtype)                                      # [1, 504]
        mask_t = valid.reshape(1, WIN_FRAMES, 1)
        mask_add = ((valid - 1.0) * (-NEG)).reshape(1, 1, 1, WIN_FRAMES)              # 0 / -1e4
        enc = self.audio_encoder(lfr, mask_t, mask_add)
        adp = self.audio_adaptor(enc, mask_add)
        ftl = torch.ceil(L / 8.0)                                                     # fake_token_len
        return lfr, L, enc, adp, ftl

    def forward(self, audio):
        _, _, _, adp, ftl = self._run(audio)
        return {"features": adp[:, 0:OUT_TOKENS, :],
                "mask": (self.tok_idx < ftl).to(torch.uint8)}

    def forward_debug(self, audio, n_valid=None):
        lfr, L, enc, adp, ftl = self._run(audio, n_valid)
        return {"lfr": lfr, "L": L, "enc": enc, "adp": adp, "ftl": ftl,
                "features": adp[:, 0:OUT_TOKENS, :], "mask": (self.tok_idx < ftl).to(torch.uint8)}

    def forward_ref(self, lfr):
        """funasr path on an unpadded [1, L, 560] LFR (no masking needed: every row is valid)."""
        enc = self.audio_encoder(lfr)
        return enc, self.audio_adaptor(enc)


def load_encoder(st_path):
    """Strict-load audio_encoder.* / audio_adaptor.* from the -vllm model.safetensors (BF16 -> fp32)."""
    from safetensors import safe_open
    m = FunAsrNanoAudioEncoder()
    sd = {}
    with safe_open(st_path, framework="pt") as f:
        for k in f.keys():
            if k.startswith("audio_encoder.") or k.startswith("audio_adaptor."):
                sd[k] = f.get_tensor(k).to(torch.float32)
    res = m.load_state_dict(sd, strict=True)
    assert not res.missing_keys and not res.unexpected_keys, res
    assert len(sd) == len(m.state_dict()), (len(sd), len(m.state_dict()))
    return m.eval(), len(sd)


def frame_window(wave_i16, n_frames=WIN_FRAMES):
    """Runtime framing: int16 PCM / 32768 -> [1, n_frames, 960], the last partial frame and the rest
    of the window zero-padded (LiteRT-LM GetFramedSegments + window padding)."""
    n = len(wave_i16)
    assert n <= n_frames * FRAME, (n, n_frames * FRAME)
    buf = np.zeros(n_frames * FRAME, np.float32)
    buf[:n] = wave_i16.astype(np.float32) / 32768.0
    return torch.from_numpy(buf.reshape(1, n_frames, FRAME))


def read_wav_i16(path):
    import soundfile as sf
    x, sr = sf.read(path, dtype="int16")
    assert sr == SR and x.ndim == 1, (path, sr, x.shape)
    return x


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    m, n = load_encoder(os.path.join(here, "out", "hf_vllm", "model.safetensors"))
    print("loaded tensors", n, "params", sum(p.numel() for p in m.parameters()))
