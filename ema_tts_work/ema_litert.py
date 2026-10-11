"""EMA Lightning as three static-shape graphs (text / sound / decoder) plus the host bookkeeping between them.

The graphs hold only float math: embeddings, convolutions, matmuls, LayerNorm, softmax, cos/sin. Every integer or
boolean step of the package (word indices, the frame timeline, the aligner's window, the decoder's padding masks)
runs on the host (numpy here; the same ~80 lines in Kotlin or Swift on a phone) and enters the graphs as float
inputs. Shapes are fixed per bucket so the graphs compile for GPU / NPU delegates.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ema_lightning.model import apply_rope, masked_group_norm, rope

NEG = -1e4
L_BUCKETS = (32, 64, 128, 256)
LT_BUCKETS = ((32, 48), (64, 96), (128, 192), (256, 384), (256, 512))  # (letters, frames); frames ~ 1.4-1.5 x letters at speed 1
P_BUCKETS = (48, 120)  # decoder windows: first window 25 + 8 context -> 48; later 100 + 2 x 8 -> 120 (engine.py)
CONTEXT, FIRST_WINDOW, WINDOW, MAX_WORD_FRAMES, MAX_FRAMES = 8, 25, 100, 250, 3000


# ----------------------------------------------------------------------------------------------------- graphs
class TextGraph(nn.Module):
    """ids i32 [1,L], mask f32 [1,L] -> h f32 [1,L,224], dur f32 [1,L] (frames per letter at speed 1)."""

    def __init__(self, m):
        super().__init__()
        self.m = m
        assert len(m.text.attn) == 0, "the shipped checkpoint has text_attn = 0; add the attention path before exporting"

    def forward(self, ids, mask):
        return text_stage(self.m, self.m.text.emb(ids), mask)


def text_stage(m, x, mask):
    """ConvNeXt blocks + LayerNorm -> h, then the duration predictor -> dur (frames per letter at speed 1).

    The ConvNeXt layer scale gamma is folded into pw2's weight and bias: the exported `gamma * pw2(...)` is a MUL of a
    rank-2 runtime tensor with a rank-3 [1,1,C] constant, the form the ML Drift GPU backend miscomputes silently
    (LiteRT #10231; S26 round p1: the text graph's h off by 1.5, corr 0.37, while every other graph was exact).
    """
    te = m.text
    for block in te.conv:
        y = block.norm(block.dw((x * mask[..., None]).transpose(1, 2)).transpose(1, 2))
        x = x + F.linear(F.gelu(block.pw1(y)), block.pw2.weight * block.gamma[:, None], block.pw2.bias * block.gamma)
    h = te.norm(x)
    dp, mm = m.chardur, mask[:, None]
    x = masked_group_norm(F.silu(dp.net[0](h.transpose(1, 2) * mm)), dp.net[2], mm) * mm
    x = masked_group_norm(F.silu(dp.net[4](x)), dp.net[6], mm)
    log_d = dp.out(x.transpose(1, 2)).squeeze(-1) * mask
    dur = (torch.exp(log_d.clamp(max=6.0)) - 1.0).clamp(min=1e-3) * mask  # expm1 written as exp - 1
    return h, dur


class TextGraphEmb(TextGraph):
    """TextGraph with the letter embedding done on the host: emb f32 [1,L,224] (the 49 x 224 table rows), mask f32 [1,L]
    -> h, dur. Removes EMBEDDING_LOOKUP (int32 ids) from the graph for GPU delegates."""

    def forward(self, emb, mask):
        return text_stage(self.m, emb, mask)


def embedding_table(m):
    """The 49 x 224 float32 letter table for the host lookup (TextGraphEmb)."""
    return m.text.emb.weight.detach().numpy().astype(np.float32)


def aligner_cond(a, h, cg, fg, allow):
    """The windowed Gaussian aligner (model.Aligner.forward, letter_pos path) with the positions cg / fg and the window
    `allow` from the host. The frame query is one learned vector: q = W_q frame_q is a [1,1,D] constant and the RoPE
    multiply broadcasts it over the T frames (the package's `frame_q.expand` would export as BROADCAST_TO, which the
    GPU delegate refuses: S26 round p1, 'Failed to compile model')."""
    B, L, D = h.shape
    T = fg.shape[1]
    q = a.q(a.frame_q).view(1, 1, a.h, a.dh).transpose(1, 2)  # [1, h, 1, dh]
    k = a.k(h).view(B, L, a.h, a.dh).transpose(1, 2)
    v = a.v(h).view(B, L, a.h, a.dh).transpose(1, 2)
    q = apply_rope(q, *(t[:, None] for t in rope(fg * a.pos_scale, a.dh)))  # -> [B, h, T, dh] by broadcast
    k = apply_rope(k, *(t[:, None] for t in rope(cg * a.pos_scale, a.dh)))
    logits = (q @ k.transpose(-2, -1)) / math.sqrt(a.dh) * a.log_temp.exp()
    sig2 = (a.log_sigma.exp() ** 2).view(1, a.h, 1, 1)
    dist2 = ((fg[:, :, None] - cg[:, None, :]) ** 2)[:, None]
    logits = logits - a.bias_w.view(1, a.h, 1, 1) * dist2 / (2 * sig2)
    logits = logits * allow + NEG * (1.0 - allow)  # masked_fill(~allow, NEG) in float
    attn = logits.softmax(-1) * allow
    return a.o((attn @ v).transpose(1, 2).reshape(B, T, D))


class SoundGraph(nn.Module):
    """h [1,L,224], cg [1,L], fg [1,T], allow f32 [1,1,T,L], fbias f32 [1,1,1,T], noise [1,4,T,64] -> latents [1,T,64].

    allow = the aligner's window (1 where frame t may read letter l), fbias = 0 on real frames / NEG on padding.
    """

    def __init__(self, m):
        super().__init__()
        self.m = m
        assert m.aligner.letter_pos and m.shared

    def forward(self, h, cg, fg, allow, fbias, noise):
        m = self.m
        T = fg.shape[1]
        cond = aligner_cond(m.aligner, h, cg, fg, allow)
        cos, sin = rope(torch.arange(T), m.dh)
        x = noise[:, 0]
        for kk, t in enumerate(m.times):
            x1 = x + (1 - t) * self.backbone(x, cond, torch.full((1,), t), fbias, cos, sin)
            if kk + 1 < len(m.times):
                x = (1 - m.times[kk + 1]) * noise[:, kk + 1] + m.times[kk + 1] * x1
        return x1

    def backbone(self, x, cond, t, fbias, cos, sin):
        m = self.m
        c = m.t_embed(t)
        bc = m.ada_shared(c).view(-1, 6, m.d)
        x = m.in_proj(x) + cond
        for block in m.blocks:
            sa, ga, aa, sf, gf, af = (p.unsqueeze(1) for p in (bc + block.ada_offset[None]).unbind(1))
            x = x + aa * self.attention(block.attn, block.n1(x) * (1 + ga) + sa, cos, sin, fbias)
            x = x + af * block.ff(block.n2(x) * (1 + gf) + sf)
        s, g = m.ada_out(c).chunk(2, -1)
        return m.out_proj(m.norm_out(x) * (1 + g.unsqueeze(1)) + s.unsqueeze(1))

    @staticmethod
    def attention(at, x, cos, sin, fbias):
        B, T, D = x.shape
        q, k, v = (t.view(B, T, at.h, at.dh).transpose(1, 2) for t in at.qkv(x).chunk(3, -1))
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        s = (q @ k.transpose(-2, -1)) / math.sqrt(at.dh) + fbias
        return at.proj((s.softmax(-1) @ v).transpose(1, 2).reshape(B, T, D))


class AlignGraph(nn.Module):
    """h [1,L,224], cg [1,L], fg [1,T], allow [1,1,T,L] -> cond [1,T,224] (the aligner alone; split form)."""

    def __init__(self, m):
        super().__init__()
        self.m = m
        assert m.aligner.letter_pos

    def forward(self, h, cg, fg, allow):
        return aligner_cond(self.m.aligner, h, cg, fg, allow)


class StepGraph(nn.Module):
    """x [1,T,64], cond [1,T,224], bc [1,6,224], sg [1,2,224], fbias [1,1,1,T] -> v [1,T,64]: one DiT velocity pass.

    bc / sg are the step's modulation vectors (ada_shared(t_embed(t)) and ada_out(t_embed(t))), a 4-row host table
    (`step_table`); the Euler mixing with the noise runs on the host, so the DiT weights appear once in the file.
    """

    def __init__(self, m):
        super().__init__()
        self.m = m
        assert m.shared

    def forward(self, x, cond, bc, sg, fbias):
        m = self.m
        cos, sin = rope(torch.arange(x.shape[1]), m.dh)
        x = m.in_proj(x) + cond
        for block in m.blocks:
            sa, ga, aa, sf, gf, af = (p.unsqueeze(1) for p in (bc + block.ada_offset[None]).unbind(1))
            x = x + aa * SoundGraph.attention(block.attn, block.n1(x) * (1 + ga) + sa, cos, sin, fbias)
            x = x + af * block.ff(block.n2(x) * (1 + gf) + sf)
        s, g = sg[:, 0], sg[:, 1]
        return m.out_proj(m.norm_out(x) * (1 + g.unsqueeze(1)) + s.unsqueeze(1))


def step_table(m):
    """[(t, bc [1,6,224], sg [1,2,224])] for the model's distilled times, from the checkpoint (host-side constants)."""
    out = []
    with torch.no_grad():
        for t in m.times:
            c = m.t_embed(torch.full((1,), t))
            bc = m.ada_shared(c).view(1, 6, m.d)
            s, g = m.ada_out(c).chunk(2, -1)
            out.append((float(t), bc.numpy().astype(np.float32), torch.stack([s, g], 1).numpy().astype(np.float32)))
    return out


def euler(run_step, x0_noise, cond, table, fbias, times):
    """The package's sampling loop on the host: x1 = x + (1 - t) v; x = (1 - t') n' + t' x1 (model.py sound_stage)."""
    x = x0_noise[0]
    for k, (t, bc, sg) in enumerate(table):
        v = run_step(x=x, cond=cond, bc=bc, sg=sg, fbias=fbias)
        x1 = x + (1 - t) * v
        if k + 1 < len(times):
            x = (1 - times[k + 1]) * x0_noise[k + 1] + times[k + 1] * x1
    return x1


def stage_lengths(d, n):
    """Real length after every decoder stage for n real input frames (decoder.py's `length` recurrence)."""
    out, length = [n], n
    for up in d.ups:
        length = (length - 1) * up.stride[0] - 2 * up.padding[0] + up.kernel_size[0]
        out.append(length)
    return out


class DecoderGraph(nn.Module):
    """z [1,64,P] + seven masks [1,1,len_i] (1 on real samples, 0 past the window's real length) -> audio [1, P*1920]."""

    def __init__(self, d):
        super().__init__()
        self.d = d

    def forward(self, z, m0, m1, m2, m3, m4, m5, m6):
        d, masks = self.d, (m0, m1, m2, m3, m4, m5, m6)
        P = z.shape[-1]
        x = d.pre(z * m0) * m0
        for i, up in enumerate(d.ups):
            x = self.up(up, F.leaky_relu(x, 0.1)) * masks[i + 1]
            x = sum(self.res(b, x, masks[i + 1]) for b in d.blocks[i * d.nk:(i + 1) * d.nk]) / d.nk
        return torch.tanh(d.post(F.leaky_relu(x)))[:, 0, :P * d.hop]

    @staticmethod
    def up(up, x):
        # ConvTranspose1d(k, s, padding=p) = the VALID transposed convolution cropped by p on each side. The converter
        # legalizes only VALID / SAME transposed convolutions, and stage 3 (k 10, s 5, p 2 -> 5n + 1 outputs) is neither.
        p = up.padding[0]
        y = F.conv_transpose1d(x, up.weight, up.bias, stride=up.stride[0])
        return y[..., p:y.shape[-1] - p] if p else y

    @staticmethod
    def res(b, x, m):
        for c1, c2 in zip(b.convs1, b.convs2):
            y = F.leaky_relu(c1(F.leaky_relu(x, 0.1)), 0.1) * m
            x = (x + c2(y)) * m
        return x


# ------------------------------------------------------------------------------------------- host bookkeeping
def piece_ids(text, stoi):
    """engine.Engine.piece: letters -> ids, the word of each letter, the first letter of that word."""
    ids = [stoi.get(ch, 1) for ch in text]
    starts = [i for i, ch in enumerate(text) if ch != " " and (i == 0 or text[i - 1] == " ")] or [0]
    bounds = [0] + starts[1:] + [len(text)]
    cw, wstart = [], []
    for w, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        cw += [w] * (b - a)
        wstart += [a] * (b - a)
    return np.array(ids, np.int32), np.array(cw, np.int32), np.array(wstart, np.int32)


def timeline(dur, cw, wstart, speed=1.0):
    """engine.plan + model.sound_stage + Aligner(letter_pos): fw, fp, cg, fg for one piece (speed 1 = as trained)."""
    dur = dur.astype(np.float64) / speed
    n_words = int(cw[-1]) + 1
    counts = np.zeros(n_words)
    np.add.at(counts, cw, dur)
    counts = np.clip(np.round(counts), 1, MAX_WORD_FRAMES).astype(np.int64)
    frames = min(int(counts.sum()), MAX_FRAMES)
    fw = np.repeat(np.arange(n_words), counts)[:frames]
    starts = np.cumsum(counts) - counts
    fp = ((np.arange(frames) - starts[fw]) / counts[fw]).astype(np.float32)
    c = np.maximum(dur, 1e-4).astype(np.float32)
    done = np.cumsum(c)
    before = done - c
    total = np.zeros(n_words, np.float32)
    np.add.at(total, cw, c)
    cp = np.clip((done - before[wstart] - 0.5 * c) / np.maximum(total[cw], 1e-8), 0, 1).astype(np.float32)
    wlen = np.maximum(np.bincount(cw, minlength=n_words).astype(np.float32), 1)
    woff = np.cumsum(wlen) - wlen
    cg = (woff[cw] + cp * wlen[cw]).astype(np.float32)
    fg = (woff[fw] + fp * wlen[fw]).astype(np.float32)
    return fw.astype(np.int32), fp, cg, fg


def bucket_for(L, T):
    for lb, tb in LT_BUCKETS:
        if L <= lb and T <= tb:
            return lb, tb
    raise ValueError(f"piece too long for the buckets: L={L} T={T}")


def text_bucket(L):
    return next(b for b in L_BUCKETS if L >= 0 and L <= b)


def sound_inputs(h, cg, fg, cw, fw, Lb, Tb, noise, lookback=1, lookahead=1):
    """Pad one piece to (Lb, Tb) and build allow [1,1,Tb,Lb] and fbias [1,1,1,Tb]."""
    L, T = len(cg), len(fg)
    H = np.zeros((1, Lb, h.shape[-1]), np.float32); H[0, :L] = h
    CG = np.zeros((1, Lb), np.float32); CG[0, :L] = cg
    FG = np.zeros((1, Tb), np.float32); FG[0, :T] = fg
    CW = np.full(Lb, -1, np.int64); CW[:L] = cw
    FW = np.full(Tb, -1, np.int64); FW[:T] = fw
    rel = CW[None, :] - FW[:, None]
    allow = ((rel >= -lookback) & (rel <= lookahead) & (CW[None, :] >= 0) & (FW[:, None] >= 0)).astype(np.float32)[None, None]
    fbias = np.full((1, 1, 1, Tb), NEG, np.float32); fbias[..., :T] = 0.0
    N = np.zeros((1, noise.shape[0], Tb, noise.shape[-1]), np.float32); N[0, :, :T] = noise
    return dict(h=H, cg=CG, fg=FG, allow=allow, fbias=fbias, noise=N)


def decoder_masks(d, n, P):
    """Seven float masks for a window of n real frames decoded at bucket P."""
    full, real = stage_lengths(d, P), stage_lengths(d, n)
    out = {}
    for i, (f, r) in enumerate(zip(full, real)):
        m = np.zeros((1, 1, f), np.float32); m[..., :r] = 1.0
        out[f"m{i}"] = m
    return out


def decode_windows(frames):
    """(a, b, s, e): the decoder window [a, b) with context, the audio cut [s, e)."""
    spans, s = [], 0
    while s < frames:
        e = min(frames, s + (FIRST_WINDOW if s == 0 else WINDOW))
        spans.append((max(0, s - CONTEXT), min(frames, e + CONTEXT), s, e))
        s = e
    return spans


def p_bucket(n):
    return next(p for p in P_BUCKETS if n <= p)


def noise_for(seed, frames, steps=4, latent=64):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn((steps, frames, latent), generator=g).numpy()


# ------------------------------------------------------------------------------------------------- tflite run
class Runner:
    """One .tflite (single signature) through ai_edge_litert's Interpreter."""

    def __init__(self, path, threads=4, signature=None):
        from ai_edge_litert.interpreter import Interpreter
        self.it = Interpreter(model_path=str(path), num_threads=threads)
        sigs = list(self.it.get_signature_list().keys())  # litert-torch names them after lt.signature(name)
        self.signature = signature or sigs[0]
        assert self.signature in sigs, (self.signature, sigs)
        self.sig = self.it.get_signature_runner(self.signature)
        self.inputs = self.sig.get_input_details()

    def __call__(self, **kw):
        feed = {}
        for k, v in kw.items():
            d = self.inputs[k]
            feed[k] = np.ascontiguousarray(v, dtype=d["dtype"])
        return self.sig(**feed)
