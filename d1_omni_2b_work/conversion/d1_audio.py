"""d1-Omni-2B audio encoder, export form: the downsampler, the 17-layer LFM2 encoder and the audio projector of
LiquidAI/d1-Omni-2B over one clip's log-mel frames, right-padded to a fixed frame count T (a length bucket).

    D1Audio(weights, spec, T)(mel [1, T, 128], valid [1, T, 1, 1]) -> {"features": [1, T / 8, 2048]}

The provider's computation (modeling_lfm3_omni.py of the snapshot: Lfm3OmniDownsampler, Lfm3OmniEncoder,
Lfm3OmniProjector, `get_audio_features`; the encoder's LFM2 = transformers 5.19.0 models/lfm2/modeling_lfm2.py, its
convolutions through d1's `_lfm2_conv` = the same causal depthwise sums), in plain torch:

- downsampler: Conv2d 1 -> 256 (3x3, stride 2, pad 1) -> SiLU, depthwise 3x3 stride 2 -> 1x1 -> SiLU, depthwise 3x3
  stride 2 -> 1x1 -> SiLU over the (mel, time) plane, then the 1x1 Conv1d over channel x mel (c * 16 + f) -> 768 per
  token (one token per 8 frames). The provider's `mask_padding` (zero the frames past the clip's length at the input,
  after the first stage and after the second) = products with `valid` and its stride-2 samples: v2 = valid[0::2],
  v4 = valid[0::4] (u < ceil(F / 2) <=> 2u < F; ceil(ceil(F / 2) / 2) = ceil(F / 4)), all rank 4 (memory
  mldrift-rank-promotion-class: the GPU delegate places lower-rank runtime tensors batch-first).
- 17 LFM2 layers (conv, conv, attention) x 5 + attention x 2, hidden 768: RMSNorm (eps 1e-5) before the operator and
  before the MLP (no sandwich norms); conv layers: in_proj -> B, C, x (thirds) -> B * x -> causal depthwise conv of
  width 3 (no bias) as 3 shifted products of a front pad of 2 -> C * . -> out_proj; attention layers: q / k / v (8 / 4
  heads, head_dim 96), per-head q / k RMSNorm, RoPE theta 1e6 at positions 0..N-1 (the provider's float32 tables),
  GQA folded into the matmul rows ([1, 4, 2N, 96]: query heads 2j, 2j + 1 read kv head j = repeat_kv), the causal
  window of 32 (keys i - 32 < j <= i) as -30000 entering as a runtime rank-4 tensor (constant [1, 1, 2N, N] times
  frame 0's validity [1, 1, 1, 1], always 1, in place of a constant mask),
  softmax, out_proj; MLP w2(silu(w1 x) * w3 x) (1792); embedding_norm at the end.
- projector: LayerNorm(768, eps 1e-5) -> Linear 768 -> 2048 -> GELU (erf) -> Linear 2048 -> 2048.

Every token reads only frames before its window's end through the causal layers, and the downsampler's look-ahead
reads zeros past the clip exactly as the provider's masked input does, so the first ceil(F / 8) tokens equal the
provider's tokens of that clip alone; the rest is padding output the host drops.
"""
from __future__ import annotations

import json
import os

import torch
import torch.nn.functional as F
from torch import nn

NEG = 30000.0
BUCKETS = (504, 1000, 2000, 3000)  # mel frames: ~5.05 / 10 / 20 / 30 s at 100 frames a second; all multiples of 8
ENC = "model.audio_encoder."
PROJ = "model.audio_projector."
SNAP_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                            "snapshot")


class Spec:
    def __init__(self, snap: str = SNAP_DEFAULT):
        c = json.load(open(os.path.join(snap, "config.json")))
        a = c["audio_encoder_config"]
        t = a["text_config"]
        self.mels, self.channels, self.hidden = a["num_mel_bins"], a["conv_channels"], a["hidden_size"]
        self.window = a["sliding_window"]
        self.layer_types = list(t["layer_types"])
        self.heads, self.kv = t["num_attention_heads"], t["num_key_value_heads"]
        self.hd = t.get("head_dim") or t["hidden_size"] // t["num_attention_heads"]
        self.eps, self.conv_k = t["norm_eps"], t["conv_L_cache"]
        self.theta = t["rope_parameters"]["rope_theta"]
        assert t["rope_parameters"].get("rope_type", "default") == "default"
        assert not t["conv_bias"] and t["hidden_size"] == self.hidden
        d = t["intermediate_size"]
        if t.get("block_auto_adjust_ff_dim", True):
            d = int(2 * d / 3)
            if t.get("block_ffn_dim_multiplier") is not None:
                d = int(t["block_ffn_dim_multiplier"] * d)
                m = t.get("block_multiple_of", 256)
                d = m * ((d + m - 1) // m)
        self.ffn = d
        self.out = c["text_config"]["hidden_size"]


def load_audio_weights(snap: str = SNAP_DEFAULT, dtype=torch.float32) -> dict:
    """The audio encoder's and projector's tensors (bf16 in the file) as `dtype`, full names."""
    from safetensors import safe_open

    out = {}
    with safe_open(os.path.join(snap, "model.safetensors"), framework="pt") as f:
        for k in f.keys():
            if k.startswith(ENC) or k.startswith(PROJ):
                out[k] = f.get_tensor(k).to(dtype)
    return out


def rope_tables(spec: Spec, n: int, dtype=torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    """Lfm2RotaryEmbedding (default rope): inv_freq in float32 as its __init__ computes it, positions 0..n-1;
    cos / sin [1, 1, n, hd]. fp32: position.float() * inv_freq, cat, cos, sin (the provider's ops); fp64: the same
    product and functions in fp64 on the same float32 inverse frequencies."""
    inv = 1.0 / (spec.theta ** (torch.arange(0, spec.hd, 2, dtype=torch.float) / spec.hd))
    if dtype == torch.float32:
        freqs = torch.arange(n)[:, None].float() * inv[None, :]
    else:
        freqs = torch.arange(n, dtype=dtype)[:, None] * inv.to(dtype)[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype)[None, None], emb.sin().to(dtype)[None, None]


def _p(w: torch.Tensor) -> nn.Parameter:
    return nn.Parameter(w, requires_grad=False)


class RMSNorm(nn.Module):
    def __init__(self, weight, eps):
        super().__init__()
        self.weight, self.eps = _p(weight), eps

    def forward(self, x):
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps))


class D1Audio(nn.Module):
    def __init__(self, w: dict, spec: Spec, T: int, dtype=torch.float32):
        super().__init__()
        assert T % 8 == 0, T
        self.spec, self.T, self.N = spec, T, T // 8
        g = lambda k: w[k].to(dtype)  # noqa: E731
        ds = ENC + "downsample."
        self.conv_in_w, self.conv_in_b = _p(g(ds + "conv_in.weight")), _p(g(ds + "conv_in.bias"))
        self.conv_mid_w, self.conv_mid_b = _p(g(ds + "conv_mid.weight")), _p(g(ds + "conv_mid.bias"))
        self.lin_mid_w, self.lin_mid_b = _p(g(ds + "lin_mid.weight")), _p(g(ds + "lin_mid.bias"))
        self.conv_end_w, self.conv_end_b = _p(g(ds + "conv_end.weight")), _p(g(ds + "conv_end.bias"))
        self.lin_end_w, self.lin_end_b = _p(g(ds + "lin_end.weight")), _p(g(ds + "lin_end.bias"))
        self.lin_out_w, self.lin_out_b = _p(g(ds + "lin_out.weight")[:, :, 0]), _p(g(ds + "lin_out.bias"))
        self.layers = nn.ModuleList()
        for i, kind in enumerate(spec.layer_types):
            p = f"{ENC}lfm.layers.{i}."
            m = nn.Module()
            m.kind = kind
            m.operator_norm = RMSNorm(g(p + "operator_norm.weight"), spec.eps)
            m.ffn_norm = RMSNorm(g(p + "ffn_norm.weight"), spec.eps)
            m.w1, m.w2, m.w3 = (_p(g(p + f"feed_forward.{n}.weight")) for n in ("w1", "w2", "w3"))
            if kind == "full_attention":
                for n in ("q_proj", "k_proj", "v_proj", "out_proj"):
                    setattr(m, n, _p(g(p + f"self_attn.{n}.weight")))
                m.q_norm = RMSNorm(g(p + "self_attn.q_layernorm.weight"), spec.eps)
                m.k_norm = RMSNorm(g(p + "self_attn.k_layernorm.weight"), spec.eps)
            else:
                assert kind == "conv", kind
                m.in_proj, m.out_proj = _p(g(p + "conv.in_proj.weight")), _p(g(p + "conv.out_proj.weight"))
                # [C, 1, k] -> k rows [1, 1, C]: y[t] = sum_i w[:, i] * x[t - (k - 1) + i]
                m.taps = _p(g(p + "conv.conv.weight")[:, 0, :].t().contiguous().reshape(spec.conv_k, 1, 1, -1))
            self.layers.append(m)
        self.final_norm = RMSNorm(g(ENC + "lfm.embedding_norm.weight"), spec.eps)
        self.ln_w, self.ln_b = _p(g(PROJ + "layer_norm.weight")), _p(g(PROJ + "layer_norm.bias"))
        self.l1_w, self.l1_b = _p(g(PROJ + "linear_1.weight")), _p(g(PROJ + "linear_1.bias"))
        self.l2_w, self.l2_b = _p(g(PROJ + "linear_2.weight")), _p(g(PROJ + "linear_2.bias"))
        cos, sin = rope_tables(spec, self.N, dtype)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        i = torch.arange(self.N)[:, None]
        j = torch.arange(self.N)[None, :]
        allowed = (j <= i) & (j > i - spec.window)
        mask = torch.where(allowed, 0.0, -NEG).to(dtype)
        rep = spec.heads // spec.kv
        self.register_buffer("mask4", mask.repeat(rep, 1).reshape(1, 1, rep * self.N, self.N).contiguous(),
                             persistent=False)
        self.dtype = dtype

    # ------------------------------------------------------------------ parts
    def downsample(self, mel, valid):
        T = self.T
        v1 = valid.reshape(1, 1, 1, T)
        v2 = valid.reshape(1, T // 2, 2, 1)[:, :, 0:1, :].reshape(1, 1, 1, T // 2)
        v4 = valid.reshape(1, T // 4, 4, 1)[:, :, 0:1, :].reshape(1, 1, 1, T // 4)
        h = mel.transpose(1, 2).reshape(1, 1, self.spec.mels, T) * v1          # [1, 1, 128, T]
        h = F.silu(F.conv2d(h, self.conv_in_w, self.conv_in_b, stride=2, padding=1)) * v2
        c = self.spec.channels
        h = F.conv2d(h, self.conv_mid_w, self.conv_mid_b, stride=2, padding=1, groups=c)
        h = F.silu(F.conv2d(h, self.lin_mid_w, self.lin_mid_b)) * v4
        h = F.conv2d(h, self.conv_end_w, self.conv_end_b, stride=2, padding=1, groups=c)
        h = F.silu(F.conv2d(h, self.lin_end_w, self.lin_end_b))                # [1, 256, 16, T / 8]
        b, ch, f, t = h.shape
        h = h.permute(0, 3, 1, 2).reshape(1, t, ch * f)                          # flatten(1, 2) order: c * 16 + f
        return F.linear(h, self.lin_out_w, self.lin_out_b)                       # [1, N, 768]

    def attention(self, m, x, mask):
        s, n = self.spec, self.N
        q = m.q_norm(F.linear(x, m.q_proj).reshape(1, n, s.heads, s.hd)).transpose(1, 2)
        k = m.k_norm(F.linear(x, m.k_proj).reshape(1, n, s.kv, s.hd)).transpose(1, 2)
        v = F.linear(x, m.v_proj).reshape(1, n, s.kv, s.hd).transpose(1, 2)
        half = s.hd // 2
        q = q * self.cos + torch.cat((-q[..., half:], q[..., :half]), dim=-1) * self.sin
        k = k * self.cos + torch.cat((-k[..., half:], k[..., :half]), dim=-1) * self.sin
        rep = s.heads // s.kv
        qg = q.reshape(1, s.kv, rep * n, s.hd)
        w = torch.softmax(torch.matmul(qg, k.transpose(2, 3)) * (s.hd ** -0.5) + mask, dim=-1)
        o = torch.matmul(w, v).reshape(1, s.heads, n, s.hd).transpose(1, 2).reshape(1, n, s.heads * s.hd)
        return F.linear(o, m.out_proj)

    def conv(self, m, x):
        d, kk = self.spec.hidden, self.spec.conv_k
        bcx = F.linear(x, m.in_proj)                                             # [1, N, 3d]: B, C, x
        b, c, xx = bcx[..., :d], bcx[..., d:2 * d], bcx[..., 2 * d:]
        u = b * xx
        up = torch.cat((torch.zeros(1, kk - 1, d, dtype=u.dtype), u), dim=1)     # causal front pad
        y = up[:, 0:self.N] * m.taps[0]
        for i in range(1, kk):
            y = y + up[:, i:i + self.N] * m.taps[i]
        return F.linear(c * y, m.out_proj)

    def encoder(self, h, one):
        mask = self.mask4 * one                                                  # runtime rank 4
        for m in self.layers:
            x = m.operator_norm(h)
            h = h + (self.attention(m, x, mask) if m.kind == "full_attention" else self.conv(m, x))
            f = m.ffn_norm(h)
            h = h + F.linear(F.silu(F.linear(f, m.w1)) * F.linear(f, m.w3), m.w2)
        return self.final_norm(h)

    def project(self, h):
        h = F.layer_norm(h, (self.spec.hidden,), self.ln_w, self.ln_b, 1e-5)
        return F.linear(F.gelu(F.linear(h, self.l1_w, self.l1_b)), self.l2_w, self.l2_b)

    def forward(self, mel, valid):
        h = self.downsample(mel, valid)
        h = self.encoder(h, valid[:, 0:1, :, :])
        return {"features": self.project(h)}


def bucket_for(frames: int) -> int | None:
    """The smallest bucket that holds `frames` mel frames, or None (longer than the largest)."""
    for t in BUCKETS:
        if frames <= t:
            return t
    return None


def make_inputs(features, T: int):
    """The provider's input_features [128, F] (numpy or torch) -> mel [1, T, 128] (zeros past F) and valid
    [1, T, 1, 1] (1 on the clip's frames), float32 numpy."""
    import numpy as np

    x = np.asarray(features, dtype=np.float32)
    f = x.shape[1]
    assert x.shape[0] == 128 and 1 <= f <= T, (x.shape, T)
    mel = np.zeros((1, T, 128), np.float32)
    mel[0, :f] = x.T
    valid = np.zeros((1, T, 1, 1), np.float32)
    valid[0, :f] = 1.0
    return mel, valid


def tokens_for(frames: int) -> int:
    return -(frames // -8)


def build(T: int, snap: str = SNAP_DEFAULT, dtype=torch.float32) -> D1Audio:
    return D1Audio(load_audio_weights(snap, torch.float32), Spec(snap), T, dtype).eval().requires_grad_(False)

