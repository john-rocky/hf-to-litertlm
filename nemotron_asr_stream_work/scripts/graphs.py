"""r1 torch modules for the LiteRT graphs of nvidia/nemotron-3.5-asr-streaming-0.6b (320 ms cache-aware streaming).

Shared by the export (.venv-lt) and the torch checks (.venv-ref). Weights are read from hf/nemotron/model.safetensors
(the pinned revision ea30d66d). The conformer / pre-encode / decoder arithmetic is the PyTorch re-author from our
earlier conversion of this model for another on-device runtime (token-exact there against the HF streaming
oracle), copied here and rewritten for one 24-layer graph and the GPU delegates:

MelFirst  pcm [1, 4040] f32 -> mel [1, 25, 128]   (processor, is_first_audio_chunk=True: center=True, first 25 frames)
MelNext   pcm [1, 5520] f32 -> mel [1, 32, 128]   (processor, is_first_audio_chunk=False: center=False)
          preemphasis 0.97 inside the chunk (sample 0 kept), STFT n_fft 512 / win 400 symmetric Hann / hop 160 as a
          constant DFT matmul over the 400 window taps, |X| then squared (the extractor's sqrt-then-pow), Slaney mel 128
          (the extractor's own filter bank, out/graphs/melfb.npy), log(x + 2^-24), no normalization.
PreFirst  mel [1, 25, 128] -> embeds [1, 4, 1024] + cache0 [1,1,1,131], cache1 [1,256,1,68], cache2 [1,256,1,36]
Pre       mel [1, 32, 128] + cache0/1/2 -> embeds [1, 4, 1024] + cache0/1/2 out   (causal subsampling, conv2d caches)
Conformer x [1, 4, 1024], neg_mask [1, 1, 4, 60] (0 / -30000 on the not-yet-filled cache slots), k_cache / v_cache
          [24, 8, 56, 128] (right-aligned rolling window), conv_cache [24, 1024, 8], one_hot [1, 128]
          -> enc_proj [1, 4, 640], k_out, v_out, conv_out (same shapes as the caches).
          Rewrites against that earlier module (same arithmetic up to fp32 summation order where noted):
          * one graph for the 24 layers (the earlier one split them 12 + 12 for an iOS loader limit);
          * the language one-hot enters as F.linear(one_hot, W1[:, 1024:], b1) added to F.linear(x, W1[:, :1024])
            (the concat + expand of the one-hot would lower to BROADCAST_TO + CONCATENATION; the split sums the
            1,152-term dot product as 1,024 + 128 terms);
          * cache slices stay rank 4 / rank 3 (k_cache[i:i+1], no squeeze / unsqueeze);
          * pointwise convs as nn.Linear on the channel axis (FULLY_CONNECTED; Conv1d k=1 lowers to CONV_2D, which the
            int4 blockwise recipe leaves float32);
          * optional per-LayerNorm input scale 2^-k with eps * 4^-k (fp32 bit-equal; fp16 storage headroom), ln_k.
Predict   token [1, 1] int32 + h [2, 1, 640] + c [2, 1, 640] -> dec_out [1, 640] + h_out + c_out (2 LSTM cells as gates)
Joint     dec_out [1, 640] + enc_frame [1, 640] -> logits [1, 13088] = head(relu(enc_frame + dec_out))
"""
import math

import numpy as np  # noqa: F401
import torch
import torch.nn as nn
import torch.nn.functional as F

from contract import *  # noqa: F401,F403  (constants and host rules)


# --------------------------------------------------------------------------- mel
class _MelBase(nn.Module):
    def __init__(self, melfb):
        super().__init__()
        n = torch.arange(WIN, dtype=torch.float64)
        k = torch.arange(NBIN, dtype=torch.float64)
        w = torch.hann_window(WIN, periodic=False, dtype=torch.float64)
        ph = 2.0 * math.pi * torch.outer(k, n + (NFFT - WIN) // 2) / NFFT
        basis = torch.cat([torch.cos(ph) * w, -torch.sin(ph) * w], 0)        # [514, 400]
        self.register_buffer("basis", basis.to(torch.float32))
        self.register_buffer("melfb", torch.as_tensor(melfb, dtype=torch.float32).clone())  # [128, 257]

    def _logmel(self, frames):                                                # frames [T, 400]
        spec = F.linear(frames, self.basis)
        re = spec[:, :NBIN]
        im = spec[:, NBIN:]
        mag = torch.sqrt(re * re + im * im)
        power = mag * mag
        mel = F.linear(power, self.melfb)
        return torch.log(mel + LOG_GUARD)[None]                               # [1, T, 128]

    @staticmethod
    def _preemph(pcm):
        prev = F.pad(pcm[:, :-1], (1, 0))
        return pcm - PREEMPH * prev


class MelFirst(_MelBase):
    """center=True: frame t covers padded samples [160 t, 160 t + 400) of y padded by 200 zeros on the left (the
    512-point centered frame restricted to its 400 window taps); 25 frames reach sample 4,039."""

    def forward(self, pcm):                                                   # [1, 4040]
        y = self._preemph(pcm)
        yp = F.pad(y, (WIN // 2, (T_FIRST + 2) * HOP - N_FIRST - WIN // 2))  # [1, 4320]
        ch = yp.reshape(T_FIRST + 2, HOP)
        frames = torch.cat([ch[0:T_FIRST], ch[1:T_FIRST + 1], ch[2:T_FIRST + 2, :WIN - 2 * HOP]], dim=1)
        return self._logmel(frames)


class MelNext(_MelBase):
    """center=False: frame t covers chunk samples [160 t, 160 t + 512); its window taps are [160 t + 56, 160 t + 456)."""

    def forward(self, pcm):                                                   # [1, 5520]
        y = self._preemph(pcm)
        off = (NFFT - WIN) // 2
        yp = y[:, off:off + (T_NEXT + 2) * HOP]                               # [1, 5440]
        ch = yp.reshape(T_NEXT + 2, HOP)
        frames = torch.cat([ch[0:T_NEXT], ch[1:T_NEXT + 1], ch[2:T_NEXT + 2, :WIN - 2 * HOP]], dim=1)
        return self._logmel(frames)


# --------------------------------------------------------------------------- subsampling (pre-encode)
class Subsampling(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv2d(1, SUB_CH, 3, stride=2)
        self.layers = nn.ModuleList()
        for _ in range(2):
            stage = nn.Module()
            stage.depthwise_conv = nn.Conv2d(SUB_CH, SUB_CH, 3, stride=2, groups=SUB_CH)
            stage.pointwise_conv = nn.Conv2d(SUB_CH, SUB_CH, 1)
            self.layers.append(stage)
        f = MEL
        for _ in range(3):
            f = f // 2 + 1                                                    # 128 -> 65 -> 33 -> 17
        self.linear = nn.Linear(SUB_CH * f, HID, bias=True)


def _freq_pad(x):
    return F.pad(x, (2, 1))                                                   # causal freq pad (k-1, s-1)


class PreFirst(nn.Module):
    def __init__(self, sub):
        super().__init__()
        self.sub = sub

    def forward(self, mel):                                                   # [1, 25, 128]
        xf = _freq_pad(mel.unsqueeze(1))                                      # [1, 1, 25, 131]
        c0 = xf[:, :, -1:, :]
        h = torch.relu(self.sub.conv_in(F.pad(xf, (0, 0, 2, 0))))             # [1, 256, 13, 65]
        s1, s2 = self.sub.layers
        h1f = _freq_pad(h)
        c1 = h1f[:, :, -1:, :]
        h = torch.relu(s1.pointwise_conv(s1.depthwise_conv(F.pad(h1f, (0, 0, 2, 0)))))   # [1, 256, 7, 33]
        h2f = _freq_pad(h)
        c2 = h2f[:, :, -1:, :]
        h = torch.relu(s2.pointwise_conv(s2.depthwise_conv(F.pad(h2f, (0, 0, 2, 0)))))   # [1, 256, 4, 17]
        h = h.transpose(1, 2).reshape(1, h.shape[2], -1)
        return self.sub.linear(h), c0, c1, c2


class Pre(nn.Module):
    def __init__(self, sub):
        super().__init__()
        self.sub = sub

    def forward(self, mel, cache0, cache1, cache2):                           # [1, 32, 128] + caches
        xf = _freq_pad(mel.unsqueeze(1))                                      # [1, 1, 32, 131]
        c0 = xf[:, :, -1:, :]
        h = torch.relu(self.sub.conv_in(torch.cat([cache0, xf], dim=2)))      # [1, 256, 16, 65]
        s1, s2 = self.sub.layers
        h1f = _freq_pad(h)
        c1 = h1f[:, :, -1:, :]
        h = torch.relu(s1.pointwise_conv(s1.depthwise_conv(torch.cat([cache1, h1f], dim=2))))  # [1, 256, 8, 33]
        h2f = _freq_pad(h)
        c2 = h2f[:, :, -1:, :]
        h = torch.relu(s2.pointwise_conv(s2.depthwise_conv(torch.cat([cache2, h2f], dim=2))))  # [1, 256, 4, 17]
        h = h.transpose(1, 2).reshape(1, h.shape[2], -1)
        return self.sub.linear(h), c0, c1, c2


# --------------------------------------------------------------------------- conformer
class LNs(nn.Module):
    """nn.LayerNorm(x * 2^-k) with eps * 4^-k: bit-equal to LayerNorm(x) in fp32 (powers of two scale exactly)."""

    def __init__(self, ln, k=0):
        super().__init__()
        self.weight = nn.Parameter(ln.weight.detach().clone())
        self.bias = nn.Parameter(ln.bias.detach().clone())
        self.k = int(k)
        self.eps = ln.eps * (4.0 ** -self.k)

    def forward(self, x):
        if self.k:
            x = x * (2.0 ** -self.k)
        return F.layer_norm(x, (HID,), self.weight, self.bias, self.eps)


def build_pos_emb(T=KV):
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, HID, 2, dtype=torch.float) / HID))
    pos = torch.arange(T - 1, -T, -1, dtype=torch.float)
    freqs = pos[:, None] * inv_freq[None, :]
    pe = torch.stack([freqs.sin(), freqs.cos()], dim=-1).reshape(2 * T - 1, HID)
    return pe[None]                                                           # [1, 119, 1024]


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ff1_l1 = nn.Linear(HID, FF, bias=False)
        self.ff1_l2 = nn.Linear(FF, HID, bias=False)
        self.ff2_l1 = nn.Linear(HID, FF, bias=False)
        self.ff2_l2 = nn.Linear(FF, HID, bias=False)
        self.q_proj = nn.Linear(HID, HID, bias=False)
        self.k_proj = nn.Linear(HID, HID, bias=False)
        self.v_proj = nn.Linear(HID, HID, bias=False)
        self.o_proj = nn.Linear(HID, HID, bias=False)
        self.relative_k_proj = nn.Linear(HID, HID, bias=False)
        self.bias_u = nn.Parameter(torch.zeros(HEADS, HDIM))
        self.bias_v = nn.Parameter(torch.zeros(HEADS, HDIM))
        self.pw1 = nn.Linear(HID, 2 * HID, bias=False)                       # pointwise_conv1 (k=1) as Linear
        self.dw = nn.Conv1d(HID, HID, CONV_K, groups=HID, bias=False)        # causal depthwise conv
        self.pw2 = nn.Linear(HID, HID, bias=False)                           # pointwise_conv2 (k=1) as Linear
        self.n_ff1 = nn.LayerNorm(HID)
        self.n_att = nn.LayerNorm(HID)
        self.n_conv = nn.LayerNorm(HID)
        self.n_cm = nn.LayerNorm(HID)                                         # conv module's own norm
        self.n_ff2 = nn.LayerNorm(HID)
        self.n_out = nn.LayerNorm(HID)
        self.scaling = HDIM ** -0.5

    @staticmethod
    def _rel_shift(x):                                                        # [1, H, Q, 2KV-1]
        b, h, t, p = x.shape
        x = F.pad(x, (1, 0))
        x = x.view(b, h, p + 1, t)
        return x[:, :, 1:].view(b, h, t, p)


LN_SITES = ("n_ff1", "n_att", "n_conv", "n_cm", "n_ff2", "n_out")


class Conformer(nn.Module):
    def __init__(self, ln_k=None):
        super().__init__()
        self.layers = nn.ModuleList([Block() for _ in range(NLAYERS)])
        self.prompt_linear_1 = nn.Linear(HID + NUM_PROMPTS, PROMPT_FF)
        self.prompt_linear_2 = nn.Linear(PROMPT_FF, HID)
        self.projector = nn.Linear(HID, PROJ, bias=True)
        self.register_buffer("pos_emb", build_pos_emb(KV), persistent=False)
        self.ln_k = ln_k
        self.collect = None                                                   # list -> per-layer hidden (checks)
        self.ln_probe = None                                                  # dict -> per-site |x - mean| / var max
        self.stats = None                                                     # dict -> per-point max|x| (range pass)

    def _st(self, name, t, masked=False):
        """Range pass only (stats is None at export): max|x| of an intermediate; masked = also the max over the
        entries above -15,000 (the attention terms that carry the -30,000 key bias)."""
        if self.stats is None:
            return
        a = float(t.detach().abs().max())
        o = self.stats.get(name)
        self.stats[name] = max(a, o) if o is not None else a
        if masked:
            u = t.detach()
            u = u[u > -15000.0]
            if u.numel():
                a2 = float(u.abs().max())
                o2 = self.stats.get(name + ":unmasked")
                self.stats[name + ":unmasked"] = max(a2, o2) if o2 is not None else a2

    def apply_ln_k(self, ln_k):
        """Swap every LayerNorm for LNs with its k (dict 'L<i>.<site>' -> k). Call after load_weights."""
        for i, blk in enumerate(self.layers):
            for s in LN_SITES:
                setattr(blk, s, LNs(getattr(blk, s), (ln_k or {}).get(f"L{i}.{s}", 0)))
        self.ln_k = ln_k

    def _ln(self, i, site, mod, x):
        if self.ln_probe is not None:
            m = x.mean(-1, keepdim=True)
            d = x - m
            key = f"L{i}.{site}"
            amax = float(d.abs().max())
            vmax = float((d * d).mean(-1).max())
            o = self.ln_probe.get(key)
            self.ln_probe[key] = (max(amax, o[0]), max(vmax, o[1])) if o else (amax, vmax)
        return mod(x)

    def forward(self, x, neg_mask, k_cache, v_cache, conv_cache, one_hot):
        k_outs, v_outs, c_outs = [], [], []
        st = self._st
        for i, blk in enumerate(self.layers):
            L = f"L{i}."
            st(L + "in", x)
            h = self._ln(i, "n_ff1", blk.n_ff1, x)
            a1 = blk.ff1_l1(h)
            st(L + "ff1_h", a1)
            f1 = blk.ff1_l2(F.silu(a1))
            st(L + "ff1_o", f1)
            x = x + 0.5 * f1
            st(L + "r_ff1", x)

            hn = self._ln(i, "n_att", blk.n_att, x)
            q = blk.q_proj(hn).view(1, Q, HEADS, HDIM).transpose(1, 2)            # [1, H, Q, D]
            k_new = blk.k_proj(hn).view(1, Q, HEADS, HDIM).transpose(1, 2)
            v_new = blk.v_proj(hn).view(1, Q, HEADS, HDIM).transpose(1, 2)
            st(L + "q", q)
            st(L + "k", k_new)
            st(L + "v", v_new)
            k = torch.cat([k_cache[i:i + 1], k_new], dim=2)                       # [1, H, KV, D]
            v = torch.cat([v_cache[i:i + 1], v_new], dim=2)
            k_outs.append(k[:, :, Q:, :])
            v_outs.append(v[:, :, Q:, :])
            q_u = q + blk.bias_u.view(1, HEADS, 1, HDIM)
            q_v = q + blk.bias_v.view(1, HEADS, 1, HDIM)
            rel_k = blk.relative_k_proj(self.pos_emb).view(1, 2 * KV - 1, HEADS, HDIM)
            bd = q_v @ rel_k.permute(0, 2, 3, 1)                                  # [1, H, Q, 2KV-1]
            st(L + "bd_raw", bd)
            bd = blk._rel_shift(bd)[..., :KV] * blk.scaling + neg_mask
            ac = q_u @ k.transpose(-2, -1) * blk.scaling
            st(L + "ac", ac)
            logits = ac + bd
            st(L + "att_logits", logits, masked=True)
            w = torch.softmax(logits, dim=-1)
            attn = (w @ v).transpose(1, 2).reshape(1, Q, HID)
            st(L + "attn", attn)
            o = blk.o_proj(attn)
            st(L + "o", o)
            x = x + o
            st(L + "r_att", x)

            hc = self._ln(i, "n_conv", blk.n_conv, x)
            g = blk.pw1(hc)                                                       # [1, Q, 2 HID]
            st(L + "pw1", g)
            g = g[..., :HID] * torch.sigmoid(g[..., HID:])                        # GLU
            gp = torch.cat([conv_cache[i:i + 1], g.transpose(1, 2)], dim=-1)      # [1, HID, K-1+Q]
            c_outs.append(gp[:, :, Q:])
            y = blk.dw(gp).transpose(1, 2)                                        # [1, Q, HID]
            st(L + "dw", y)
            y = F.silu(self._ln(i, "n_cm", blk.n_cm, y))
            p2 = blk.pw2(y)
            st(L + "pw2", p2)
            x = x + p2
            st(L + "r_conv", x)

            h = self._ln(i, "n_ff2", blk.n_ff2, x)
            a2 = blk.ff2_l1(h)
            st(L + "ff2_h", a2)
            f2 = blk.ff2_l2(F.silu(a2))
            st(L + "ff2_o", f2)
            x = x + 0.5 * f2
            st(L + "r_ff2", x)
            x = self._ln(i, "n_out", blk.n_out, x)
            st(L + "out", x)
            if self.collect is not None:
                self.collect.append(x.detach().clone())

        w1 = self.prompt_linear_1.weight
        p = F.linear(one_hot, w1[:, HID:], self.prompt_linear_1.bias)          # [1, 2048]
        hx = F.linear(x, w1[:, :HID]) + p[:, None, :]
        st("head.prompt_pre", hx)
        hf = torch.relu(hx)
        h2 = self.prompt_linear_2(hf)
        st("head.prompt_o", h2)
        enc = self.projector(h2)                                                # [1, Q, 640]
        st("head.enc", enc)
        return enc, torch.cat(k_outs, 0), torch.cat(v_outs, 0), torch.cat(c_outs, 0)


# --------------------------------------------------------------------------- decoder
class Predict(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB, DEC_HID)
        self.decoder_projector = nn.Linear(DEC_HID, DEC_HID, bias=True)
        for l in range(DEC_NL):
            self.register_buffer(f"w_ih{l}", torch.zeros(4 * DEC_HID, DEC_HID))
            self.register_buffer(f"w_hh{l}", torch.zeros(4 * DEC_HID, DEC_HID))
            self.register_buffer(f"b_ih{l}", torch.zeros(4 * DEC_HID))
            self.register_buffer(f"b_hh{l}", torch.zeros(4 * DEC_HID))

    def _cell(self, x, h, c, l):
        g = F.linear(x, getattr(self, f"w_ih{l}"), getattr(self, f"b_ih{l}")) + \
            F.linear(h, getattr(self, f"w_hh{l}"), getattr(self, f"b_hh{l}"))
        i, f, gg, o = g.chunk(4, dim=-1)
        c2 = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(gg)
        h2 = torch.sigmoid(o) * torch.tanh(c2)
        return h2, c2

    def forward(self, token, h, c):                                           # token [1, 1] int32, h / c [2, 1, 640]
        x = self.embedding(token)[:, 0, :]                                   # [1, 640]
        h0, c0 = self._cell(x, h[0], c[0], 0)
        h1, c1 = self._cell(h0, h[1], c[1], 1)
        return self.decoder_projector(h1), torch.stack([h0, h1], 0), torch.stack([c0, c1], 0)


class Joint(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(DEC_HID, VOCAB, bias=True)

    def forward(self, dec_out, enc_frame):                                    # [1, 640] each
        return self.head(torch.relu(enc_frame + dec_out))


# --------------------------------------------------------------------------- weights
def load_all(path=WEIGHTS):
    """Return (Subsampling, Conformer, Predict, Joint) with the checkpoint's weights (strict name map)."""
    from safetensors import safe_open
    sub, conf, pred, joint = Subsampling(), Conformer(), Predict(), Joint()
    used = set()
    with safe_open(path, framework="pt") as f:
        keys = set(f.keys())

        def g(k):
            used.add(k)
            return f.get_tensor(k).to(torch.float32)

        sd = {}
        for k in keys:
            if k.startswith("encoder.subsampling."):
                sd[k[len("encoder.subsampling."):]] = g(k)
        sub.load_state_dict(sd, strict=True)
        for i, blk in enumerate(conf.layers):
            p = f"encoder.layers.{i}."
            m = {"ff1_l1": "feed_forward1.linear1", "ff1_l2": "feed_forward1.linear2",
                 "ff2_l1": "feed_forward2.linear1", "ff2_l2": "feed_forward2.linear2",
                 "q_proj": "self_attn.q_proj", "k_proj": "self_attn.k_proj", "v_proj": "self_attn.v_proj",
                 "o_proj": "self_attn.o_proj", "relative_k_proj": "self_attn.relative_k_proj"}
            for a, b in m.items():
                getattr(blk, a).weight.data.copy_(g(p + b + ".weight"))
            blk.bias_u.data.copy_(g(p + "self_attn.bias_u"))
            blk.bias_v.data.copy_(g(p + "self_attn.bias_v"))
            blk.pw1.weight.data.copy_(g(p + "conv.pointwise_conv1.weight")[:, :, 0])
            blk.dw.weight.data.copy_(g(p + "conv.depthwise_conv.weight"))
            blk.pw2.weight.data.copy_(g(p + "conv.pointwise_conv2.weight")[:, :, 0])
            for a, b in (("n_ff1", "norm_feed_forward1"), ("n_att", "norm_self_att"), ("n_conv", "norm_conv"),
                         ("n_cm", "conv.norm"), ("n_ff2", "norm_feed_forward2"), ("n_out", "norm_out")):
                getattr(blk, a).weight.data.copy_(g(p + b + ".weight"))
                getattr(blk, a).bias.data.copy_(g(p + b + ".bias"))
        for a, b in (("prompt_linear_1", "prompt_projector.linear_1"), ("prompt_linear_2", "prompt_projector.linear_2"),
                     ("projector", "encoder_projector")):
            getattr(conf, a).weight.data.copy_(g(b + ".weight"))
            getattr(conf, a).bias.data.copy_(g(b + ".bias"))
        pred.embedding.weight.data.copy_(g("decoder.embedding.weight"))
        pred.decoder_projector.weight.data.copy_(g("decoder.decoder_projector.weight"))
        pred.decoder_projector.bias.data.copy_(g("decoder.decoder_projector.bias"))
        for l in range(DEC_NL):
            for a, b in (("w_ih", "weight_ih"), ("w_hh", "weight_hh"), ("b_ih", "bias_ih"), ("b_hh", "bias_hh")):
                getattr(pred, f"{a}{l}").copy_(g(f"decoder.lstm.{b}_l{l}"))
        joint.head.weight.data.copy_(g("joint.head.weight"))
        joint.head.bias.data.copy_(g("joint.head.bias"))
        unused = sorted(keys - used)
    assert not unused, f"checkpoint tensors not mapped: {unused[:10]}"
    for m in (sub, conf, pred, joint):
        m.eval()
    return sub, conf, pred, joint



# --------------------------------------------------------------------------- export wrappers (named outputs)
class ExMelFirst(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, pcm):
        return {"mel": self.m(pcm)}


class ExMel(ExMelFirst):
    pass


class ExPreFirst(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, mel):
        e, c0, c1, c2 = self.m(mel)
        return {"embeds": e, "cache0_out": c0, "cache1_out": c1, "cache2_out": c2}


class ExPre(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, mel, cache0, cache1, cache2):
        e, c0, c1, c2 = self.m(mel, cache0, cache1, cache2)
        return {"embeds": e, "cache0_out": c0, "cache1_out": c1, "cache2_out": c2}


class ExConformer(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x, neg_mask, k_cache, v_cache, conv_cache, one_hot):
        e, k, v, c = self.m(x, neg_mask, k_cache, v_cache, conv_cache, one_hot)
        return {"enc_proj": e, "k_out": k, "v_out": v, "conv_out": c}


class ExPredict(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, token, h, c):
        d, h2, c2 = self.m(token, h, c)
        return {"dec_out": d, "h_out": h2, "c_out": c2}


class ExJoint(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, dec_out, enc_frame):
        return {"logits": self.m(dec_out, enc_frame)}
