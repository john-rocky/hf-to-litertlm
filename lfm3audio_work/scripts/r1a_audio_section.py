"""LFM3-Audio speech encoder as one LiteRT-LM audio section: PCM -> log-mel (in graph) -> Downsampler -> LFM2 encoder
-> projector -> `features` + uint8 `mask`.

The modules re-author the checkpoint's code (LiquidAI/LFM3-Audio-2B @ 9d49701c, modeling_lfm3_audio.py l.76-110 and
l.130-158 / 217-228, feature_extraction_lfm3_audio.py l.159-185, transformers 5.20.2 models/lfm2/modeling_lfm2.py)
with the same parameter names, so a state dict loads strictly; nothing is traced from the HF classes.

  AudioSection(spec, n_frames=T).forward(audio [1, T, 160] f32 PCM in [-1, 1))
      -> {"features": [1, T/8, out_dim] f32, "mask": [1, T/8] uint8}
  The runtime passes no clip length: the valid sample count is the last non-zero sample + 1, the valid mel frame
  count is the feature extractor's num_frames() of it, and each Downsampler stage zeroes the frames past its length
  (HF mask_padding); mask = 1 for the first ceil(frames / 8) tokens (the encoder is causal: the tokens before that
  never read the padding).
  EncoderFromMel: the same Downsampler + encoder + projector with the mel and its frame count as inputs (a measurement
  graph).

Weights: load_section(...) takes a safetensors file (+ a key map), a state dict (+ a key map) or the HF modules.
  AUDIO2B_KEYMAP  LiquidAI/LFM3-Audio-2B released names (encoder.downsample.*, encoder.lfm.lfm.*, mlp.model.{0,1,3}.*)
  OMNI_KEYMAP     model.audio_encoder.{downsample,lfm}.* + model.audio_projector.* (LFM3-Omni, and a loaded
                  Lfm3AudioForConditionalGeneration's state_dict())
"""
import dataclasses
import json
import math
import re

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

AUDIO2B_KEYMAP = (
    (r"^encoder\.downsample\.", "downsample."),
    (r"^encoder\.lfm\.lfm\.", "lfm."),
    (r"^mlp\.model\.0\.", "projector.layer_norm."),
    (r"^mlp\.model\.1\.", "projector.linear_1."),
    (r"^mlp\.model\.3\.", "projector.linear_2."),
)
OMNI_KEYMAP = (
    (r"^model\.audio_encoder\.downsample\.", "downsample."),
    (r"^model\.audio_encoder\.lfm\.", "lfm."),
    (r"^model\.audio_projector\.", "projector."),
)
MASK_NEG = -1e9


@dataclasses.dataclass(frozen=True)
class EncoderSpec:
    """Shapes of the LFM3-Audio speech encoder; defaults = LFM3-Audio-2B config.json (`encoder`, `preprocessor`)."""
    sample_rate: int = 16000
    n_fft: int = 400
    hop_length: int = 160
    num_mel_bins: int = 128
    log_floor: float = -12.0
    norm_offset: float = 4.0
    norm_divisor: float = 4.0
    conv_channels: int = 256
    hidden_size: int = 768
    layer_types: tuple = ("conv", "conv", "full_attention") * 5 + ("full_attention", "full_attention")
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    intermediate_size: int = 2560
    block_multiple_of: int = 256
    block_ffn_dim_multiplier: float = 1.0
    block_auto_adjust_ff_dim: bool = True
    conv_L_cache: int = 3
    norm_eps: float = 1e-5
    rope_theta: float = 1_000_000.0
    sliding_window: int = 32
    out_dim: int = 2048
    projector_eps: float = 1e-5

    @property
    def head_dim(self):
        return self.hidden_size // self.num_attention_heads

    @property
    def ffn_dim(self):
        """Lfm2MLP's intermediate size (transformers models/lfm2/modeling_lfm2.py l.114-122)."""
        d = self.intermediate_size
        if self.block_auto_adjust_ff_dim:
            d = int(2 * d / 3)
            if self.block_ffn_dim_multiplier is not None:
                d = int(self.block_ffn_dim_multiplier * d)
                d = self.block_multiple_of * ((d + self.block_multiple_of - 1) // self.block_multiple_of)
        return d


def spec_from_config(cfg, fe_config=None):
    """EncoderSpec from a config.json dict: the released legacy layout (`encoder`, `preprocessor`, `lfm`; LFM3-Audio-2B)
    or the transformers layout (`audio_encoder_config`, `text_config`; LFM3-Omni-2B). The front-end values come from
    the config's `preprocessor` block, else from `fe_config` (a feature extractor dict: preprocessor_config.json or
    processor_config.json["feature_extractor"]), else the defaults (= both checkpoints' values).
    The audio projector's LayerNorm is nn.LayerNorm's default eps 1e-5 in both checkpoints
    (modeling_lfm3_audio.py l.222, modeling_lfm3_omni.py l.503); Omni's `projector_layer_norm_eps` (1e-6) belongs to
    its vision projector and is not read here."""
    if isinstance(cfg, str):
        cfg = json.load(open(cfg))
    enc = cfg.get("encoder") or cfg.get("audio_encoder_config")
    lfm = enc.get("lfm") or enc.get("text_config")
    pre = cfg.get("preprocessor")
    if pre is None and fe_config is not None:
        fe = fe_config.get("feature_extractor", fe_config)
        pre = {"sample_rate": fe.get("sampling_rate", 16000), "n_fft": fe.get("n_fft", 400),
               "hop_length": fe.get("hop_length", 160), "logmel_floor": fe.get("log_floor", -12.0),
               "normalization": {"offset": fe.get("normalization_offset", 4.0),
                                 "divisor": fe.get("normalization_divisor", 4.0)}}
        assert fe.get("feature_size", 128) == enc.get("dim_in", enc.get("num_mel_bins", 128)), fe
        assert fe.get("win_length", 400) == fe.get("n_fft", 400), fe
    pre = pre or {}
    norm = pre.get("normalization", {})
    text = cfg.get("lfm") or cfg.get("text_config")
    rope = lfm.get("rope_parameters") or {}
    rope = rope.get("full_attention", rope)
    assert rope.get("rope_type", "default") == "default", rope
    return EncoderSpec(
        sample_rate=pre.get("sample_rate", 16000), n_fft=pre.get("n_fft", 400), hop_length=pre.get("hop_length", 160),
        num_mel_bins=enc.get("dim_in", enc.get("num_mel_bins", 128)), log_floor=pre.get("logmel_floor", -12.0),
        norm_offset=norm.get("offset", 4.0), norm_divisor=norm.get("divisor", 4.0),
        conv_channels=enc.get("dim_conv", enc.get("conv_channels", 256)),
        hidden_size=enc.get("dim_out", enc.get("hidden_size", 768)), layer_types=tuple(lfm["layer_types"]),
        num_attention_heads=lfm["num_attention_heads"], num_key_value_heads=lfm["num_key_value_heads"],
        intermediate_size=lfm["intermediate_size"], block_multiple_of=lfm.get("block_multiple_of", 256),
        block_ffn_dim_multiplier=lfm.get("block_ffn_dim_multiplier", 1.0),
        block_auto_adjust_ff_dim=lfm.get("block_auto_adjust_ff_dim", True), conv_L_cache=lfm.get("conv_L_cache", 3),
        norm_eps=lfm.get("norm_eps", 1e-5), rope_theta=rope.get("rope_theta", 1_000_000.0),
        sliding_window=enc.get("sliding_window", 32), out_dim=text["hidden_size"])


def hf_mel_filters(spec):
    """The feature extractor's filter bank (feature_extraction_lfm3_audio.py l.95-103), float64 [n_fft/2+1, mels]."""
    from transformers.audio_utils import mel_filter_bank
    return mel_filter_bank(num_frequency_bins=spec.n_fft // 2 + 1, num_mel_filters=spec.num_mel_bins, min_frequency=0.0,
                           max_frequency=spec.sample_rate / 2, sampling_rate=spec.sample_rate, norm="slaney",
                           mel_scale="slaney")


class LogMel(nn.Module):
    """Lfm3AudioFeatureExtractor._log_mel over a zero-padded window of T PCM frames, plus the valid frame count.

    Frame t covers samples [160 t, 160 t + 400): three consecutive 160-sample blocks, the third cut to 80, over the
    window followed by two zero blocks (the extractor right-pads the clip with zeros; the window's tail is the
    runtime's zero padding). The Hann window (torch.hann_window(400), periodic) is folded into a fixed DFT basis
    (fp32 matmul for the one-sided 201 bins), then |X|^2, the slaney filter bank, log10, clamp, affine."""

    def __init__(self, spec, n_frames, mel_filters=None):
        super().__init__()
        assert spec.n_fft == 400 and spec.hop_length == 160, "frame assembly assumes 400 / 160"
        self.spec, self.T = spec, n_frames
        self.n_samples = n_frames * spec.hop_length
        n = torch.arange(spec.n_fft, dtype=torch.float64).reshape(-1, 1)
        k = torch.arange(spec.n_fft // 2 + 1, dtype=torch.float64).reshape(1, -1)
        ang = 2.0 * math.pi * torch.remainder(n * k, spec.n_fft) / spec.n_fft
        win = torch.hann_window(spec.n_fft, periodic=True, dtype=torch.float64).reshape(-1, 1)
        self.register_buffer("dft_cos", (win * torch.cos(ang)).to(torch.float32), persistent=False)  # [400, 201]
        self.register_buffer("dft_sin", (win * torch.sin(ang)).to(torch.float32), persistent=False)
        mel = hf_mel_filters(spec) if mel_filters is None else np.asarray(mel_filters)
        self.register_buffer("mel_filters", torch.from_numpy(mel.astype(np.float32)), persistent=False)  # [201, 128]
        self.register_buffer("sample_idx", torch.arange(self.n_samples, dtype=torch.float32).reshape(1, -1),
                             persistent=False)

    def valid_frames(self, x):
        """num_frames(last non-zero sample + 1): 1 + ceil((n - 400) / 160), at least 1 (l.159-166)."""
        nz = (x != 0.0).to(x.dtype)
        n_valid = torch.amax(self.sample_idx * nz, dim=1, keepdim=True) + 1.0  # [1, 1]
        frames = 1.0 + torch.ceil((n_valid - float(self.spec.n_fft)) / float(self.spec.hop_length))
        return torch.clamp(frames, min=1.0)

    def forward(self, audio):  # audio [1, T, 160]
        hop = self.spec.hop_length
        x = audio.reshape(1, self.n_samples)
        blocks = torch.cat([audio.reshape(self.T, hop), torch.zeros(2, hop, dtype=audio.dtype)], dim=0)
        frames = torch.cat([blocks[0:self.T], blocks[1:self.T + 1], blocks[2:self.T + 2, 0:self.spec.n_fft - 2 * hop]],
                           dim=1)  # [T, 400]
        re = torch.matmul(frames, self.dft_cos)
        im = torch.matmul(frames, self.dft_sin)
        power = re * re + im * im  # [T, 201]
        mel = torch.matmul(power, self.mel_filters)  # [T, 128]
        log_mel = torch.clamp(torch.log10(mel), min=self.spec.log_floor)
        log_mel = (log_mel + self.spec.norm_offset) / self.spec.norm_divisor
        return log_mel.transpose(0, 1).unsqueeze(0), self.valid_frames(x)  # [1, 128, T], [1, 1]


class Downsampler(nn.Module):
    """Lfm3AudioDownsampler (l.76-110) with the lengths as a [1, 1] float tensor."""

    def __init__(self, spec, n_frames):
        super().__init__()
        c = spec.conv_channels
        conv = {"kernel_size": 3, "stride": 2, "padding": 1}
        self.conv_in = nn.Conv2d(1, c, **conv)
        self.conv_mid = nn.Conv2d(c, c, groups=c, **conv)
        self.lin_mid = nn.Conv2d(c, c, kernel_size=1)
        self.conv_end = nn.Conv2d(c, c, groups=c, **conv)
        self.lin_end = nn.Conv2d(c, c, kernel_size=1)
        self.lin_out = nn.Conv1d(spec.num_mel_bins * c // 8, spec.hidden_size, kernel_size=1)
        assert n_frames % 8 == 0
        self.T = n_frames
        self.register_buffer("t_idx", torch.arange(n_frames, dtype=torch.float32).reshape(1, -1), persistent=False)

    def mask(self, h, length):  # h [1, C, H, W]; zero the time columns >= length
        w = h.shape[-1]
        return h * (self.t_idx[:, :w] < length).to(h.dtype).reshape(1, 1, 1, w)

    def forward(self, mel, frames):  # mel [1, 128, T], frames [1, 1] float
        h = self.mask(mel.unsqueeze(1), frames)
        h = F.silu(self.conv_in(h))
        l2 = torch.floor((frames + 1.0) / 2.0)
        h = self.mask(h, l2)
        h = F.silu(self.lin_mid(self.conv_mid(h)))
        l4 = torch.floor((l2 + 1.0) / 2.0)
        h = self.mask(h, l4)
        h = F.silu(self.lin_end(self.conv_end(h)))  # [1, C, mels/8, T/8]
        b, c, f, t = h.shape
        h = h.permute(0, 3, 1, 2).reshape(b, t, c * f)  # flatten(1, 2) order: c * f_dim + f
        h = F.linear(h, self.lin_out.weight.squeeze(-1), self.lin_out.bias)
        return h, torch.floor((l4 + 1.0) / 2.0)  # [1, T/8, hidden], tokens = ceil(frames / 8)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps))


class Attention(nn.Module):
    def __init__(self, spec):
        super().__init__()
        d, hd = spec.hidden_size, spec.head_dim
        self.nh, self.nkv, self.hd = spec.num_attention_heads, spec.num_key_value_heads, hd
        self.q_proj = nn.Linear(d, self.nh * hd, bias=False)
        self.k_proj = nn.Linear(d, self.nkv * hd, bias=False)
        self.v_proj = nn.Linear(d, self.nkv * hd, bias=False)
        self.out_proj = nn.Linear(self.nh * hd, d, bias=False)
        self.q_layernorm = RMSNorm(hd, spec.norm_eps)
        self.k_layernorm = RMSNorm(hd, spec.norm_eps)

    @staticmethod
    def rope(x, cos, sin):
        half = x.shape[-1] // 2
        rot = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
        return x * cos + rot * sin

    def forward(self, x, cos, sin, mask):  # x [1, N, d]
        n = x.shape[1]
        q = self.q_layernorm(self.q_proj(x).reshape(1, n, self.nh, self.hd)).transpose(1, 2)
        k = self.k_layernorm(self.k_proj(x).reshape(1, n, self.nkv, self.hd)).transpose(1, 2)
        v = self.v_proj(x).reshape(1, n, self.nkv, self.hd).transpose(1, 2)
        q, k = self.rope(q, cos, sin), self.rope(k, cos, sin)
        rep = self.nh // self.nkv  # repeat_kv: head h reads kv head h // rep
        k = k.unsqueeze(2).expand(1, self.nkv, rep, n, self.hd).reshape(1, self.nh, n, self.hd)
        v = v.unsqueeze(2).expand(1, self.nkv, rep, n, self.hd).reshape(1, self.nh, n, self.hd)
        w = torch.softmax(torch.matmul(q, k.transpose(2, 3)) * (self.hd ** -0.5) + mask, dim=-1)
        o = torch.matmul(w, v).transpose(1, 2).reshape(1, n, self.nh * self.hd)
        return self.out_proj(o)


class ShortConv(nn.Module):
    """Lfm2ShortConv without a cache: in_proj -> B * x -> causal depthwise conv -> C * . -> out_proj."""

    def __init__(self, spec):
        super().__init__()
        d, kk = spec.hidden_size, spec.conv_L_cache
        self.k = kk
        self.conv = nn.Conv1d(d, d, kernel_size=kk, groups=d, bias=False, padding=kk - 1)
        self.in_proj = nn.Linear(d, 3 * d, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)

    def forward(self, x):  # x [1, N, d]
        bcx = self.in_proj(x).transpose(1, 2)
        b, c, xx = bcx.chunk(3, dim=1)
        y = F.conv1d(F.pad(b * xx, (self.k - 1, 0)), self.conv.weight, None, groups=self.conv.weight.shape[0])
        return self.out_proj((c * y).transpose(1, 2))


class MLP(nn.Module):
    def __init__(self, spec):
        super().__init__()
        d, f = spec.hidden_size, spec.ffn_dim
        self.w1 = nn.Linear(d, f, bias=False)
        self.w3 = nn.Linear(d, f, bias=False)
        self.w2 = nn.Linear(f, d, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Layer(nn.Module):
    def __init__(self, spec, kind):
        super().__init__()
        self.is_attn = kind == "full_attention"
        if self.is_attn:
            self.self_attn = Attention(spec)
        else:
            self.conv = ShortConv(spec)
        self.feed_forward = MLP(spec)
        self.operator_norm = RMSNorm(spec.hidden_size, spec.norm_eps)
        self.ffn_norm = RMSNorm(spec.hidden_size, spec.norm_eps)

    def forward(self, h, cos, sin, mask):
        x = self.operator_norm(h)
        h = h + (self.self_attn(x, cos, sin, mask) if self.is_attn else self.conv(x))
        return h + self.feed_forward(self.ffn_norm(h))


class Lfm2Encoder(nn.Module):
    """Lfm2Model over embeddings (no embed_tokens): rope at positions 0..N-1, every attention layer under the
    sliding-window causal mask (window counts the token itself), conv layers causal from a zero state, final
    `embedding_norm`."""

    def __init__(self, spec, n_tokens):
        super().__init__()
        self.layers = nn.ModuleList([Layer(spec, kind) for kind in spec.layer_types])
        self.embedding_norm = RMSNorm(spec.hidden_size, spec.norm_eps)
        hd = spec.head_dim
        # Lfm2RotaryEmbedding (default rope) in float32, the same ops as the reference
        inv_freq = 1.0 / (spec.rope_theta ** (torch.arange(0, hd, 2, dtype=torch.float) / hd))
        freqs = torch.arange(n_tokens, dtype=torch.float32)[:, None] * inv_freq[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos(), persistent=False)  # [N, hd]
        self.register_buffer("sin", emb.sin(), persistent=False)
        i = torch.arange(n_tokens)[:, None]
        j = torch.arange(n_tokens)[None, :]
        allowed = (j <= i) & (j > i - spec.sliding_window)
        self.register_buffer("mask", torch.where(allowed, 0.0, MASK_NEG).to(torch.float32), persistent=False)

    def forward(self, h):
        for layer in self.layers:
            h = layer(h, self.cos, self.sin, self.mask)
        return self.embedding_norm(h)


class Projector(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.layer_norm = nn.LayerNorm(spec.hidden_size, eps=spec.projector_eps)
        self.linear_1 = nn.Linear(spec.hidden_size, spec.out_dim)
        self.linear_2 = nn.Linear(spec.out_dim, spec.out_dim)

    def forward(self, x):
        return self.linear_2(F.gelu(self.linear_1(self.layer_norm(x))))


class EncoderFromMel(nn.Module):
    """Downsampler + LFM2 encoder + projector on a mel window [1, 128, T] with its valid frame count [1, 1]."""

    def __init__(self, spec=EncoderSpec(), n_frames=3000):
        super().__init__()
        assert n_frames % 8 == 0
        self.spec, self.T, self.N = spec, n_frames, n_frames // 8
        self.downsample = Downsampler(spec, n_frames)
        self.lfm = Lfm2Encoder(spec, self.N)
        self.projector = Projector(spec)
        self.register_buffer("tok_idx", torch.arange(self.N, dtype=torch.float32).reshape(1, -1), persistent=False)

    def encode(self, mel, frames):
        h, n_tok = self.downsample(mel, frames)
        enc = self.lfm(h)
        return enc, self.projector(enc), n_tok

    def forward(self, mel, frames):
        _, feats, n_tok = self.encode(mel, frames)
        return {"features": feats, "mask": (self.tok_idx < n_tok).to(torch.uint8)}


class AudioSection(EncoderFromMel):
    """The LiteRT-LM audio section: `audio` [1, T, 160] -> {"features", "mask"}."""

    def __init__(self, spec=EncoderSpec(), n_frames=3000, mel_filters=None):
        super().__init__(spec, n_frames)
        self.logmel = LogMel(spec, n_frames, mel_filters)

    def forward(self, audio):
        mel, frames = self.logmel(audio)
        return super().forward(mel, frames)

    def forward_debug(self, audio):
        mel, frames = self.logmel(audio)
        enc, feats, n_tok = self.encode(mel, frames)
        return {"mel": mel, "frames": frames, "enc": enc, "features": feats, "n_tokens": n_tok,
                "mask": (self.tok_idx < n_tok).to(torch.uint8)}


# ---- weights ---------------------------------------------------------------------------------------------------
def remap(state, keymap):
    """{source key: tensor} -> {section key: tensor} for the keys any rule of `keymap` matches (others dropped)."""
    out = {}
    for key, value in state.items():
        for pattern, repl in keymap:
            if re.match(pattern, key):
                out[re.sub(pattern, repl, key)] = value
                break
    return out


def safetensors_files(path):
    """A .safetensors file, a list of them, or a checkpoint dir (model.safetensors or the shards of its index)."""
    import os
    if isinstance(path, (list, tuple)):
        return list(path)
    if os.path.isdir(path):
        index = os.path.join(path, "model.safetensors.index.json")
        if os.path.exists(index):
            shards = sorted(set(json.load(open(index))["weight_map"].values()))
            return [os.path.join(path, s) for s in shards]
        return [os.path.join(path, "model.safetensors")]
    return [path]


def load_section(section, *, safetensors_path=None, state_dict=None, keymap=AUDIO2B_KEYMAP, hf_audio_encoder=None,
                 hf_audio_projector=None):
    """Strict-load the section's weights (cast to fp32) from one of: safetensors (a file, a list of shards or a
    checkpoint dir) + key map, a state dict + key map, or the HF modules (Lfm3AudioModel.audio_encoder /
    .audio_projector, or Omni's equivalents). Returns the number of tensors loaded."""
    if hf_audio_encoder is not None:
        enc = {k: v for k, v in hf_audio_encoder.state_dict().items() if not k.startswith("lfm.rotary_emb.")}
        state = {**enc, **{f"projector.{k}": v for k, v in hf_audio_projector.state_dict().items()}}
    elif safetensors_path is not None:
        from safetensors import safe_open
        state = {}
        for file in safetensors_files(safetensors_path):
            with safe_open(file, framework="pt") as f:
                for k in f.keys():
                    if any(re.match(p, k) for p, _ in keymap):
                        state[k] = f.get_tensor(k)
        state = remap(state, keymap)
    else:
        state = remap(state_dict, keymap)
    state = {k: v.to(torch.float32) for k, v in state.items()}
    res = section.load_state_dict(state, strict=True)
    assert not res.missing_keys and not res.unexpected_keys, res
    return len(state)


def build_section(model_dir=None, *, safetensors_path=None, config=None, fe_config=None, keymap=AUDIO2B_KEYMAP,
                  n_frames=3000, mel_input=False):
    """AudioSection (or EncoderFromMel with mel_input=True) for a checkpoint dir (config.json + safetensors; the
    feature extractor config is read from processor_config.json / preprocessor_config.json when the config has no
    `preprocessor` block) or an explicit safetensors path + config dict; eval mode, fp32. Returns
    (section, spec, number of tensors loaded)."""
    import os
    cfg = config if config is not None else json.load(open(os.path.join(model_dir, "config.json")))
    if fe_config is None and model_dir is not None:
        for name in ("processor_config.json", "preprocessor_config.json"):
            p = os.path.join(model_dir, name)
            if os.path.exists(p):
                fe_config = json.load(open(p))
                break
    spec = spec_from_config(cfg, fe_config)
    section = (EncoderFromMel if mel_input else AudioSection)(spec, n_frames)
    n = load_section(section, safetensors_path=safetensors_path or model_dir, keymap=keymap)
    return section.eval(), spec, n


def export_section(section, out_path, signature="encode"):
    """One-signature fp32 .tflite of an AudioSection (`audio` [1, T, 160]) or an EncoderFromMel (`mel` [1, 128, T] +
    `frames` [1, 1]) with litert-torch's generic converter; outputs `features` + `mask`."""
    import litert_torch
    if isinstance(section, AudioSection):
        kw = {"audio": torch.zeros(1, section.T, section.spec.hop_length)}
    else:
        kw = {"mel": torch.zeros(1, section.spec.num_mel_bins, section.T), "frames": torch.ones(1, 1)}
    edge = litert_torch.signature(signature, section.eval(), sample_kwargs=kw).convert()
    edge.export(out_path)
    return out_path


def frame_window(pcm_f32, n_frames=3000, hop=160):
    """Runtime framing of one clip (LiteRT-LM GetFramedSegments + window padding): [1, n_frames, hop], zero tail."""
    n = len(pcm_f32)
    assert n <= n_frames * hop, (n, n_frames * hop)
    buf = np.zeros(n_frames * hop, np.float32)
    buf[:n] = pcm_f32
    return buf.reshape(1, n_frames, hop)
