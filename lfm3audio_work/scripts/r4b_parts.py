#!/usr/bin/env python3
"""GPU / NPU-friendly forms of the LFM3-Audio audio encoder section, codec decoder and depth decoder (the module name
`r4b_parts` and the class names reach the exported graphs' tensor names, so they stay as they were when the published
files were exported). The forms re-author r1a_audio_section.AudioSection and r1b_graphs.CodecChunk / DepthStep so that
the LiteRT GPU delegate takes every op and fp16 activations (GPU fp16 precision; the Qualcomm HTP runs float graphs in
fp16) keep their values. Same weights and parameter names (strict loads), fp32 graphs.

  AudioSectionR4b  encode: audio [1,3000,160] f32 -> features [1,375,2048] f32 + mask [1,375] uint8 (the contract of
                   r1a_audio_section.AudioSection)
    - GQA: the query heads that share a kv head are folded into the matmul rows ([1, kv, rep*N, hd]); repeat_kv
      (unsqueeze + expand + reshape) becomes BROADCAST_TO + a rank-5 RESHAPE, which the GPU delegate refuses.
    - lengths without indices: AudioSection finds the valid sample count as max(sample_index * (x != 0)) + 1 (indices up
      to 479,999; fp16 holds integers exactly only up to 2,048, its max is 65,504) and derives the frame / token counts
      with ceil / floor. This form computes the frame validity vector directly: frame t >= 1 is valid iff a non-zero
      sample lies at index >= 160 t + 240 (= t < 1 + ceil((n_valid - 400) / 160), the feature extractor's count), i.e.
      in the second half of block t + 1 or in any block >= t + 2: a per-block any (REDUCE_MAX over 80-sample halves) and
      a suffix max by doubling (12 shift-and-max steps); frame 0 is always valid (the count is clamped at 1). Every
      downsampled length is a prefix too, so the masks after each stride-2 stage are the strided samples v1[0::2],
      v1[0::4], v1[0::8] (u < ceil(F / 2) <=> 2u < F). All values are 0 / 1.
    - the attention mask constant is -30000 (fp16 holds it; -1e9 is -inf in fp16); exp() of either is 0 in fp32.
  CodecChunkR4b    decode: codes [1,12,F] int32, start [1] int32, k0 v0 k1 v1 k2 v2 [1,2,29,64], c0..c4 [1,6,512]
                   -> frames [1,6F,1280] + the state under the same names + `_out` (r1b_graphs.CodecChunk's names and
                   shapes)
    - RoPE relative to the chunk: the cached keys are kept UNROTATED (k* = keys after k_norm, before RoPE) and every
      call rotates the 29 cached + 6F new keys at positions -29..6F-1 and the queries at 0..6F-1 with constant
      tables. q_m . k_n under RoPE depends on m - n only, so the scores equal CodecChunk's absolute-position scores in
      exact arithmetic; CodecChunk computes cos / sin(position x inv_freq) in the graph from `start` (positions grow
      with the stream: fp16 rounds a position past 2,048). CodecChunk's k* state (rotated keys) and this form's
      (unrotated) are not interchangeable: a stream starts from zeros with one form.
    - the empty-slot term (cache slots before position 0) = relu(x) - relu(x - 1) with x = -(key position - start)
      (= clamp(x, 0, 1) for every real x; torch.clamp lowers to RELU_0_TO_1, which the GPU delegate refuses) times
      -10000 (-1e9 is -inf in fp16, and 0 x -inf = NaN where a slot is not empty).
  DepthStepR4b     depth_step: r1b_graphs.DepthStep unchanged (GQA already folded, no mask, no index past 2,016);
                   DepthProjR4b / DepthPassR4b split it into the per-frame projection and the per-pass rest.
The torch modules evaluate in fp32.
"""
import importlib.util
import math
import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

K = pathlib.Path(__file__).resolve().parents[1]


def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


S = _load("r1a_audio_section", K / "scripts" / "r1a_audio_section.py")
G = _load("r1b_graphs", K / "scripts" / "r1b_graphs.py")

MASK_NEG_AUDIO = -30000.0
MASK_NEG_CODEC = -10000.0


# ---------------------------------------------------------------- audio section
def frame_validity(audio, n_frames, first):
    """audio [1, T, 160] -> v1 [1, T, 1, 1] float 0 / 1: frame t valid iff t == 0 or a non-zero sample lies at index
    >= 160 t + 240 (AudioSection's count 1 + ceil((last non-zero + 1 - 400) / 160), clamped at 1, as a prefix mask).
    first = the constant [1, T, 1, 1] (1, 0, 0, ...). Every tensor is rank 4: the GPU delegate places lower-rank runtime
    tensors batch-first and constants right-aligned, and the same computation on rank-1 / rank-3 tensors returned
    wrong masks on the Mac GPU (Metal)."""
    z = audio.dtype
    nz = (audio != 0.0).to(z).reshape(1, n_frames, 2, 80)
    halves = torch.amax(nz, dim=3, keepdim=True)                      # [1, T, 2, 1]: any in samples 0-79 / 80-159
    block = torch.amax(halves, dim=2, keepdim=True)                   # [1, T, 1, 1]
    second = halves[:, :, 1:2, :]                                     # [1, T, 1, 1]
    suffix = block
    shift = 1
    while shift < n_frames:                                           # suffix[j] = max(block[j:])
        suffix = torch.maximum(suffix, torch.cat([suffix[:, shift:], torch.zeros(1, shift, 1, 1, dtype=z)], dim=1))
        shift *= 2
    nxt = torch.cat([second[:, 1:], torch.zeros(1, 1, 1, 1, dtype=z)], dim=1)   # second half of block t + 1
    nxt2 = torch.cat([suffix[:, 2:], torch.zeros(1, 2, 1, 1, dtype=z)], dim=1)  # any block >= t + 2
    return torch.maximum(torch.maximum(nxt, nxt2), first)


PCM_STEP = 2.0 ** -15  # one step of 16-bit PCM; the per-frame scale's floor


class LogMelR4b(S.LogMel):
    """r1a_audio_section.LogMel with the frame validity vector and a per-frame scale: each 400-sample frame is divided by
    m = max(max|x|, 2^-15) before the DFT and log10(m^2) is added back after the log (the DFT, |X|^2 and the filter
    bank are homogeneous: mel(x / m) = mel(x) / m^2). In fp32 it is the same value up to rounding; in fp16 it keeps
    the quiet frames' power spectrum out of the subnormal / flush range (unscaled, 16 % of the non-zero |X|^2 values
    of 20 English test clips lie below 3e-8) and the largest power at <= 200^2."""

    def __init__(self, spec, n_frames, mel_filters=None):
        super().__init__(spec, n_frames, mel_filters)
        first = torch.zeros(1, n_frames, 1, 1, dtype=torch.float32)
        first[0, 0] = 1.0
        self.register_buffer("first", first, persistent=False)

    def forward(self, audio):  # audio [1, T, 160] -> mel [1, 128, T], v1 [1, T, 1, 1]
        hop = self.spec.hop_length
        blocks = torch.cat([audio.reshape(self.T, hop), torch.zeros(2, hop, dtype=audio.dtype)], dim=0)
        frames = torch.cat([blocks[0:self.T], blocks[1:self.T + 1], blocks[2:self.T + 2, 0:self.spec.n_fft - 2 * hop]],
                           dim=1)                                     # [T, 400]
        m = torch.clamp(torch.amax(torch.abs(frames), dim=1, keepdim=True), min=PCM_STEP)  # [T, 1]
        xs = frames / m
        re = torch.matmul(xs, self.dft_cos)
        im = torch.matmul(xs, self.dft_sin)
        power = re * re + im * im
        mel = torch.matmul(power, self.mel_filters)                   # = mel(x) / m^2
        log_mel = torch.log10(mel) + 2.0 * torch.log10(m)
        log_mel = torch.clamp(log_mel, min=self.spec.log_floor)
        log_mel = (log_mel + self.spec.norm_offset) / self.spec.norm_divisor
        return log_mel.transpose(0, 1).unsqueeze(0), frame_validity(audio, self.T, self.first)


class DownsamplerR4b(S.Downsampler):
    def forward(self, mel, v1):  # mel [1, 128, T], v1 [1, T, 1, 1] -> h [1, T/8, hidden], v8 [1, T/8]
        T = self.T
        v2 = v1.reshape(1, T // 2, 2, 1)[:, :, 0:1, :].reshape(1, 1, 1, T // 2)
        v4 = v1.reshape(1, T // 4, 4, 1)[:, :, 0:1, :].reshape(1, 1, 1, T // 4)
        v8 = v1.reshape(1, T // 8, 8, 1)[:, :, 0:1, :].reshape(1, T // 8)
        h = mel.unsqueeze(1) * v1.reshape(1, 1, 1, T)
        h = F.silu(self.conv_in(h)) * v2
        h = F.silu(self.lin_mid(self.conv_mid(h))) * v4
        h = F.silu(self.lin_end(self.conv_end(h)))
        b, c, f, t = h.shape
        h = h.permute(0, 3, 1, 2).reshape(b, t, c * f)
        h = F.linear(h, self.lin_out.weight.squeeze(-1), self.lin_out.bias)
        return h, v8


class AttentionR4b(S.Attention):
    def forward(self, x, cos, sin, mask):  # mask = runtime [1, 1, rep * N, N] (rows r * N + position)
        n = x.shape[1]
        q = self.q_layernorm(self.q_proj(x).reshape(1, n, self.nh, self.hd)).transpose(1, 2)
        k = self.k_layernorm(self.k_proj(x).reshape(1, n, self.nkv, self.hd)).transpose(1, 2)
        v = self.v_proj(x).reshape(1, n, self.nkv, self.hd).transpose(1, 2)
        q, k = self.rope(q, cos, sin), self.rope(k, cos, sin)
        rep = self.nh // self.nkv          # query head j * rep + r reads kv head j (repeat_kv)
        qg = q.reshape(1, self.nkv, rep * n, self.hd)
        w = torch.softmax(torch.matmul(qg, k.transpose(2, 3)) * (self.hd ** -0.5) + mask, dim=-1)
        o = torch.matmul(w, v).reshape(1, self.nh, n, self.hd).transpose(1, 2).reshape(1, n, self.nh * self.hd)
        return self.out_proj(o)


class Lfm2EncoderR4b(S.Lfm2Encoder):
    """The window mask enters the attention as a RUNTIME rank-4 tensor (the constant [1, 1, rep N, N] times a [1, 1, 1, 1]
    runtime 1 = frame 0's validity, always 1). Adding the constant itself lets litert-torch fold the batch-1 rank-4
    score matmul to a rank-3 BATCH_MATMUL [4, 750, 375] that feeds ADD(.., const [1, 1, 750, 375]) -> [1, 4, 750, 375]
    = a mixed-rank constant operand; with a runtime operand the converter reshapes the product back to rank 4 first."""

    def forward(self, h, one):  # one [1, 1, 1, 1] runtime
        mask = self.mask4 * one
        for layer in self.layers:
            h = layer(h, self.cos, self.sin, mask)
        return self.embedding_norm(h)


class AudioSectionR4b(nn.Module):
    """r1a_audio_section.AudioSection's weights under this module's forward (see the module docstring)."""

    def __init__(self, spec=S.EncoderSpec(), n_frames=3000, mel_filters=None):
        super().__init__()
        assert n_frames % 8 == 0
        self.spec, self.T, self.N = spec, n_frames, n_frames // 8
        self.downsample = S.Downsampler(spec, n_frames)
        self.downsample.__class__ = DownsamplerR4b
        self.lfm = S.Lfm2Encoder(spec, self.N)
        self.lfm.__class__ = Lfm2EncoderR4b
        rep = spec.num_attention_heads // spec.num_key_value_heads
        for layer in self.lfm.layers:
            if layer.is_attn:
                layer.self_attn.__class__ = AttentionR4b
        i = torch.arange(self.N)[:, None]
        j = torch.arange(self.N)[None, :]
        allowed = (j <= i) & (j > i - spec.sliding_window)
        mask = torch.where(allowed, 0.0, MASK_NEG_AUDIO).to(torch.float32)
        self.lfm.register_buffer("mask4", mask.repeat(rep, 1).reshape(1, 1, rep * self.N, self.N).contiguous(),
                                 persistent=False)
        self.projector = S.Projector(spec)
        self.logmel = LogMelR4b(spec, n_frames, mel_filters)

    def forward_debug(self, audio):
        mel, v1 = self.logmel(audio)
        h, v8 = self.downsample(mel, v1)
        enc = self.lfm(h, v1[:, 0:1])
        return {"mel": mel, "v1": v1, "enc": enc, "features": self.projector(enc), "v8": v8}

    def forward(self, audio):
        mel, v1 = self.logmel(audio)
        h, v8 = self.downsample(mel, v1)
        return {"features": self.projector(self.lfm(h, v1[:, 0:1])), "mask": v8.to(torch.uint8)}


def build_audio_section(model_dir, *, keymap=None, n_frames=3000):
    """AudioSectionR4b with the checkpoint's weights (r1a_audio_section's config reading and strict load);
    (section, spec, n)."""
    import json
    import os
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    fe = None
    for name in ("processor_config.json", "preprocessor_config.json"):
        p = os.path.join(model_dir, name)
        if os.path.exists(p):
            fe = json.load(open(p))
            break
    spec = S.spec_from_config(cfg, fe)
    sec = AudioSectionR4b(spec, n_frames)
    n = S.load_section(sec, safetensors_path=model_dir, keymap=keymap or S.AUDIO2B_KEYMAP)
    return sec.eval(), spec, n


# ---------------------------------------------------------------- codec decoder
NORM_FLOOR = 2.0 ** -14


def rms_norm_scaled(x, w, eps):
    """RMSNorm with a per-row scale: x / m, m = max(max|x|, 2^-14), and eps / m^2 (the same value in exact arithmetic:
    rsqrt(mean((x/m)^2) + eps/m^2) * (x/m) = rsqrt(mean(x^2) + eps) * x). The plain form squares x in the activation
    dtype: the codec's residual reaches |x| ~ 3,000 (x^2 9.3e6 at layer 0 on 20 test sequences) = inf in fp16 (max
    65,504). After the scale (x/m)^2 <= 1 and the row mean >= 1/d."""
    m = torch.clamp(torch.amax(torch.abs(x), dim=-1, keepdim=True), min=NORM_FLOOR)
    xs = x / m
    r = math.sqrt(eps) / m                                           # eps / m^2 as (sqrt(eps) / m)^2: m^2 overflows fp16
    return xs * torch.rsqrt(xs.pow(2).mean(-1, keepdim=True) + r * r) * w


class CodecLayerR4b(G.Layer):
    """r1b_graphs' codec Layer (conv layers; sandwich norms) with the scaled RMSNorm."""

    def forward(self, x, **kw):
        h = rms_norm_scaled(x, self.op_norm, self.eps)
        o, c = self.conv(h, kw["conv_state"])
        o = rms_norm_scaled(o, self.post_op_norm, self.eps)
        x = x + o
        f = rms_norm_scaled(self.ffn(rms_norm_scaled(x, self.ffn_norm, self.eps)), self.post_ffn_norm, self.eps)
        return x + f, (c,)


class CodecAttentionR4b(G.Attention):
    """r1b_graphs' codec attention with the keys cached before RoPE; rotation by constant tables (positions relative to the
    chunk start): queries at 0..t-1 (cos_q / sin_q [t, hd]), the 29 cached + t new keys at -29..t-1 (cos_k / sin_k)."""

    def forward(self, h, cos_q, sin_q, cos_k, sin_k, bias, k_past, v_past):
        b, t, _ = h.shape
        q = rms_norm_scaled(F.linear(h, self.q_proj).view(b, t, self.h, self.hd), self.q_norm, self.eps).transpose(1, 2)
        k = rms_norm_scaled(F.linear(h, self.k_proj).view(b, t, self.kv, self.hd), self.k_norm, self.eps).transpose(1, 2)
        v = F.linear(h, self.v_proj).view(b, t, self.kv, self.hd).transpose(1, 2)
        k_raw = torch.cat((k_past, k), dim=2)                      # [b, kv, 29 + t, hd], unrotated
        v_all = torch.cat((v_past, v), dim=2)
        q = q * cos_q + G.rot_half(q) * sin_q
        kr = k_raw * cos_k + G.rot_half(k_raw) * sin_k
        rep = self.h // self.kv
        qg = q.reshape(b, self.kv, rep * t, self.hd)
        att = torch.matmul(qg, kr.transpose(2, 3)) * (self.hd ** -0.5) + bias
        att = torch.softmax(att, dim=-1)
        o = torch.matmul(att, v_all).reshape(b, self.h, t, self.hd).transpose(1, 2)
        gate = torch.sigmoid(F.linear(h, self.gate_proj)).unsqueeze(-1)
        o = (o * gate).reshape(b, t, self.h * self.hd)
        return F.linear(o, self.o_proj), k_raw, v_all


class CodecChunkR4b(G.CodecChunk):
    def __init__(self, sd, frames, window=None):
        super().__init__(sd, frames, window)
        c = G.CODEC_CFG
        for i in range(c["layers"]):
            if i in c["attn_layers"]:
                self.layers[i].attn.__class__ = CodecAttentionR4b
            else:
                self.layers[i].__class__ = CodecLayerR4b
        L = G.CODEC_KV_LEN + self.t
        cq, sq = G.rope_cos_sin(torch.arange(self.t), self.inv_freq)
        ck, sk = G.rope_cos_sin(torch.arange(-G.CODEC_KV_LEN, self.t), self.inv_freq)
        # the tables in float64 then fp32 (exact small-integer positions; CodecChunk computes fp32 position x inv_freq)
        pos_q = torch.arange(self.t, dtype=torch.float64)
        pos_k = torch.arange(-G.CODEC_KV_LEN, self.t, dtype=torch.float64)
        inv = self.inv_freq.double()
        for name, pos in (("q", pos_q), ("k", pos_k)):
            fr = pos[:, None] * inv[None, :]
            emb = torch.cat((fr, fr), dim=-1)
            self.register_buffer(f"cos_{name}", emb.cos().float().contiguous())
            self.register_buffer(f"sin_{name}", emb.sin().float().contiguous())
        rep = c["heads"] // c["kv_heads"]
        static = self.static_bias / -1e9 * MASK_NEG_CODEC          # CodecChunk's [rep * t, L] window mask, 0 / -1e9 -> 0 / -1e4
        assert static.shape == (rep * self.t, L)
        # rank 4: ADD(runtime [L], const [rep t, L]) is a mixed-rank constant operand, which gave wrong values on a GPU
        self.register_buffer("static_bias_r4b", static.reshape(1, 1, rep * self.t, L).contiguous())
        self.register_buffer("krel4", self.krel.reshape(1, 1, 1, L).contiguous())

    def forward(self, codes, start, k0, v0, k1, v1, k2, v2, c0, c1, c2, c3, c4):
        c = G.CODEC_CFG
        lat = F.embedding(torch.clamp(codes, 0, G.CB_SIZE - 1), self.lat_table)
        lat = lat.permute(0, 2, 1, 3).reshape(1, self.f, G.NUM_CB * len(c["fsq_levels"]))
        emb = F.linear(F.linear(lat, self.po_w, self.po_b), self.fq_w, self.fq_b)
        x = torch.cat([emb] * c["frame_stack"], dim=-1).reshape(1, self.t, c["dim"])
        s = start.float().reshape(1, 1, 1, 1)
        xe = -(self.krel4 + s)                                      # > 0 exactly for the slots before position 0
        empty = F.relu(xe) - F.relu(xe - 1.0)                       # = clamp(xe, 0, 1), [1, 1, 1, L]
        bias = self.static_bias_r4b + empty * MASK_NEG_CODEC        # [1, 1, rep t, L] runtime
        kv = [(k0, v0), (k1, v1), (k2, v2)]
        convs = [c0, c1, c2, c3, c4]
        kv_out, conv_out = [], []
        ia = ic = 0
        for i, layer in enumerate(self.layers):
            if i in c["attn_layers"]:
                h = rms_norm_scaled(x, layer.op_norm, layer.eps)
                o, k_raw, v_all = layer.attn(h, self.cos_q, self.sin_q, self.cos_k, self.sin_k, bias,
                                             kv[ia][0], kv[ia][1])
                o = rms_norm_scaled(o, layer.post_op_norm, layer.eps)
                x = x + o
                f = rms_norm_scaled(layer.ffn(rms_norm_scaled(x, layer.ffn_norm, layer.eps)), layer.post_ffn_norm,
                                    layer.eps)
                x = x + f
                kv_out += [k_raw[:, :, -G.CODEC_KV_LEN:], v_all[:, :, -G.CODEC_KV_LEN:]]
                ia += 1
            else:
                x, (cs,) = layer(x, conv_state=convs[ic])
                conv_out.append(cs)
                ic += 1
        x = rms_norm_scaled(x, self.final_norm, self.eps)
        y = F.linear(x, self.lin_w, self.lin_b)
        mag = torch.exp(y[..., :self.nbins])
        ang = y[..., self.nbins:]
        z = torch.cat((mag * torch.cos(ang), mag * torch.sin(ang)), dim=-1)
        frames = F.linear(z, self.basis_t)
        return (frames, *kv_out, *conv_out)


def codec_zero_state_r4b():
    return G.codec_zero_state()


# ---------------------------------------------------------------- depth step
class DepthStepR4b(G.DepthStep):
    """= r1b_graphs.DepthStep (no fp16 overflow site: max |x| 10,557 on 20 test sequences)."""


class DepthProjR4b(nn.Module):
    """depth_proj: hidden [1,2048] -> depth_in [1,12,1024] = depth_linear (DepthStep.project), once per frame."""

    def __init__(self, step):
        super().__init__()
        self.step = step

    def forward(self, hidden):
        return {"depth_in": self.step.project(hidden)}


class DepthPassR4b(nn.Module):
    """depth_pass: depth_in [1,12,1024] (depth_proj's output), codes [1,12] int32, masked [1,12] f32 -> logits
    [1,12,2017] = DepthStep.passes (the split form: the frame's 3 passes reuse one projection; DepthStep recomputes
    the 25.2 M-parameter depth_linear in every pass)."""

    def __init__(self, step):
        super().__init__()
        self.step = step

    def forward(self, depth_in, codes, masked):
        return {"logits": self.step.passes(depth_in, codes, masked)}


class DepthFrameGreedyR4b(nn.Module):
    """depth_frame: the frame's 3 maskgit passes in one graph, greedy (temperature 0 = argmax after the repetition
    penalty, the vendor's _sample), the codebook order from the host: hidden [1,2048], reveal [1,3,12] f32 (row p = 1
    for the codebooks pass p reveals = order[4p:4p+4]), penalty [1,12,2017] f32 (1 = the code appeared in that
    codebook's window, the host's history) -> codes [1,12] int32. The penalty in arithmetic only (no SELECT):
    l' = l + m (0.1 min(l, 0) - max(l, 0) / 11) = l / 1.1 where l > 0, l * 1.1 where l < 0 (transformers'
    RepetitionPenaltyLogitsProcessor at 1.1). The sampled path (temperature 1, top-k 128) is not in it: this form is the
    timing probe of a folded form (one call per frame instead of three); its codes are compared frame by frame with
    the HF greedy oracle (same hidden, order and history). The passes are sequential (pass p + 1 reads pass p's codes):
    every pass reads the trunk and head weights again, only depth_linear runs once per frame."""

    def __init__(self, step, penalty=1.1):
        super().__init__()
        self.step = step
        self.neg = penalty - 1.0          # l * 1.1 = l + 0.1 l
        self.pos = 1.0 - 1.0 / penalty    # l / 1.1 = l - l / 11

    def forward(self, hidden, reveal, penalty):
        d_in = self.step.project(hidden)
        codes = torch.zeros(1, G.NUM_CB, dtype=torch.int32)
        known = torch.zeros(1, G.NUM_CB)
        for p in range(3):
            logits = self.step.passes(d_in, codes, 1.0 - known)
            pos = F.relu(logits)
            logits = logits + penalty * ((logits - pos) * self.neg - pos * self.pos)
            am = torch.argmax(logits, dim=-1).to(torch.int32)          # [1, 12]
            pick = reveal[:, p]                                         # [1, 12] 0 / 1
            pick_i = pick.to(torch.int32)
            codes = codes * (1 - pick_i) + am * pick_i
            known = known + pick
        return {"codes": codes}


def export_depth_fold(sd, out_path):
    import litert_torch
    step = DepthStepR4b(sd).eval()
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    kw = {"hidden": torch.zeros(1, G.LM_HIDDEN), "reveal": torch.zeros(1, 3, G.NUM_CB),
          "penalty": torch.zeros(1, G.NUM_CB, G.CB_SIZE)}
    litert_torch.signature("depth_frame", DepthFrameGreedyR4b(step).eval(), sample_kwargs=kw).convert().export(str(out_path))
    return out_path


def export_depth_split(sd, out_proj, out_pass):
    import litert_torch
    step = DepthStepR4b(sd).eval()
    for p in (out_proj, out_pass):
        pathlib.Path(p).parent.mkdir(parents=True, exist_ok=True)
    litert_torch.signature("depth_proj", DepthProjR4b(step).eval(),
                           sample_kwargs={"hidden": torch.zeros(1, G.LM_HIDDEN)}).convert().export(str(out_proj))
    kw = {"depth_in": torch.zeros(1, G.NUM_CB, G.DEPTH_HIDDEN), "codes": torch.zeros(1, G.NUM_CB, dtype=torch.int32),
          "masked": torch.ones(1, G.NUM_CB)}
    litert_torch.signature("depth_pass", DepthPassR4b(step).eval(), sample_kwargs=kw).convert().export(str(out_pass))
    return out_proj, out_pass


# ---------------------------------------------------------------- export (litert-torch 0.10.0, the same toolchain as r1b_graphs)
class _Sig(nn.Module):
    def __init__(self, m, names):
        super().__init__()
        self.m, self.names = m, names

    def forward(self, *args):
        out = self.m(*args)
        if isinstance(out, dict):
            return out
        return dict(zip(self.names, out if isinstance(out, tuple) else (out,)))


class AudioSig(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, audio):
        return self.m(audio)


class CodecSig(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, codes, start, k0, v0, k1, v1, k2, v2, c0, c1, c2, c3, c4):
        out = self.m(codes, start, k0, v0, k1, v1, k2, v2, c0, c1, c2, c3, c4)
        return dict(zip(["frames"] + [n + "_out" for n in G.CODEC_STATE_NAMES], out))


def export_audio(sec, out_path, signature="encode"):
    import litert_torch
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kw = {"audio": torch.zeros(1, sec.T, sec.spec.hop_length)}
    litert_torch.signature(signature, AudioSig(sec).eval(), sample_kwargs=kw).convert().export(str(out_path))
    return out_path


def export_codec(sd, frames, out_path):
    import litert_torch
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m = CodecSig(CodecChunkR4b(sd, frames)).eval()
    litert_torch.signature("decode", m, sample_kwargs=G.codec_chunk_sample(frames)).convert().export(str(out_path))
    return out_path
