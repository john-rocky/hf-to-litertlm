"""DecGraph: the exported graph = the merged MNN-Pocket-0.3B-Dec backbone (Qwen3, 28 layers) written out in plain torch.

    forward(embeds f32 [1, L, 896], cos f32 [1, L, 64], sin f32 [1, L, 64], mask f32 [1, 1, L, L])
        -> {"hidden": f32 [1, L, 896]}   (the final RMSNorm output, every position)

Per layer, as transformers 4.57.1 Qwen3DecoderLayer / Qwen3Attention / Qwen3MLP / Qwen3RMSNorm compute it:
  x = RMSNorm_in(h); q = q_norm(q_proj(x) as [1, L, 14, 64]); k = k_norm(k_proj(x) as [1, L, 2, 64]); v = v_proj(x)
  q, k <- q * cos + rotate_half(q) * sin (cos / sin broadcast over heads); GQA 14 / 2: each key / value head copied 7
  times by CONCATENATION (no BROADCAST_TO; head h serves query heads 7h .. 7h + 6, the order of repeat_kv);
  scores = q k^T * 0.125 + mask, softmax over keys in fp32, o = p v -> o_proj; h += o
  h += down_proj(silu(gate_proj(x2)) * up_proj(x2)), x2 = RMSNorm_post(h)
  hidden = RMSNorm_final(h). RMSNorm(x) = w * (x * rsqrt(mean(x^2) + 1e-6)).
No biases (attention_bias false), no lm_head. Weights = the file merge_lora.py writes (the LoRA folded in fp32).

Forms of the published files (`--form` of export_dec.py). Each one computes the same thing in exact arithmetic; float16 activations and
the NPUs are why they exist:
  k4           RMSNorm on x * 2^-4 with eps * 4^-4 at the input and post-attention norms of layers 6 to 27 and at the final norm:
               rsqrt(4^-4 (v + eps)) = 2^4 rsqrt(v + eps) and 2^-4 x * 2^4 r = x r. The CPU and GPU files.
  k4_exp_silu  k4, with SiLU as h * exp(min(h, 0)) / (1 + exp(-|h|)): every EXP argument is <= 0 and there is no LOGISTIC op. The Qualcomm
               NPU files.
  mean_norm    RMSNorm on y = x / (4 a), a = mean|x| of the row, with eps as (1e-3 / (4 a))^2, at every norm:
               x rsqrt(mean(x^2) + eps) = y rsqrt(mean(y^2) + eps / (4 a)^2). A row of width W has max|x| / mean|x| <= W, so at width 896 |y| <= 224
               and sum(y^2) <= 50,176, inside float16; mean(y^2) >= 1/16, so the squares a float16 unit loses below 2^-14 are at most 1e-3 of the
               sum. The Google Tensor NPU files.
A recorder (set_recorder) keeps, per site and layer, max |x| and the largest and smallest mean(x^2) over real positions
and over all positions: the float16 headroom scan that chose these forms."""
import math

import torch
from torch import nn

HIDDEN, LAYERS, HEADS, KV_HEADS, HEAD_DIM, FFN = 896, 28, 14, 2, 64, 2560
EPS = 1e-6


FORMS = {"k4": {"shift": 4, "exp_silu": False, "mean_norm": False},
         "k4_exp_silu": {"shift": 4, "exp_silu": True, "mean_norm": False},
         "mean_norm": {"shift": 0, "exp_silu": False, "mean_norm": True}}
SHIFTED_LAYERS = range(6, 28)


def shift_for(form, site, layer):
    """The 2^-k shift of a norm site in a layer (0 = none)."""
    if site == "final" or (site in ("in", "post") and layer in SHIFTED_LAYERS):
        return form["shift"]
    return 0


class Rec:
    """Float16 headroom recorder (use: graph.set_recorder(rec); rec.n, rec.L = request tokens, L): per site key, [max |x| real, max mean(x^2) real, max sum(x^2) real,
    max |x| all, max mean(x^2) all, min mean(x^2) real] (the last = the small side: a shifted norm's mean must stay
    a normal float16, >= 6.1e-5)."""

    def __init__(self):
        self.n = None
        self.sites = {}

    def note(self, key, x):
        if self.n is None:
            return
        x = x.detach().float()
        flat = x.reshape(-1, x.shape[-1]) if x.dim() > 1 else x[None]
        # rows are positions (for [1, L, H, D] the positions are axis 1)
        if x.dim() == 4 and x.shape[1] != self.L and x.shape[2] == self.L:   # [1, H, L, D]
            xr = x[:, :, : self.n]
        elif x.dim() >= 3:
            xr = x[:, : self.n]
        else:
            xr = x
        ms_all = (x * x).mean(-1)
        ss_real = (xr * xr).sum(-1)
        ms_real = (xr * xr).mean(-1)
        r = self.sites.setdefault(key, [0.0, 0.0, 0.0, 0.0, 0.0, math.inf])
        r[5] = min(r[5], float(ms_real.min()))
        r[0] = max(r[0], float(xr.abs().max()))
        r[1] = max(r[1], float(ms_real.max()))
        r[2] = max(r[2], float(ss_real.max()))
        r[3] = max(r[3], float(x.abs().max()))
        r[4] = max(r[4], float(ms_all.max()))


class RMSNorm(nn.Module):
    def __init__(self, dim, shift=0, mean_norm=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.shift = shift
        self.eps = EPS * (4.0 ** -shift)
        self.s = 2.0 ** -shift
        self.mean_norm = mean_norm

    def forward(self, x):
        if self.mean_norm:
            ia = 0.25 / x.abs().mean(-1, keepdim=True)
            y = x * ia
            e = (EPS ** 0.5) * ia
            return self.weight * (y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + e * e))
        if self.shift:
            x = x * self.s
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps))


def rotate_half(x):
    x1, x2 = x[..., : HEAD_DIM // 2], x[..., HEAD_DIM // 2:]
    return torch.cat((-x2, x1), dim=-1)


class Layer(nn.Module):
    def __init__(self, i, form):
        super().__init__()
        self.i = i
        self.input_layernorm = RMSNorm(HIDDEN, shift_for(form, "in", i), form["mean_norm"])
        self.post_attention_layernorm = RMSNorm(HIDDEN, shift_for(form, "post", i), form["mean_norm"])
        self.q_proj = nn.Linear(HIDDEN, HEADS * HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(HIDDEN, KV_HEADS * HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(HIDDEN, KV_HEADS * HEAD_DIM, bias=False)
        self.o_proj = nn.Linear(HEADS * HEAD_DIM, HIDDEN, bias=False)
        self.q_norm = RMSNorm(HEAD_DIM, shift_for(form, "qn", i), form["mean_norm"])
        self.k_norm = RMSNorm(HEAD_DIM, shift_for(form, "kn", i), form["mean_norm"])
        self.gate_proj = nn.Linear(HIDDEN, FFN, bias=False)
        self.up_proj = nn.Linear(HIDDEN, FFN, bias=False)
        self.down_proj = nn.Linear(FFN, HIDDEN, bias=False)
        self.exp_silu = form["exp_silu"]
        self.rec = None

    def silu(self, g):
        if self.exp_silu:
            return g * torch.exp(torch.clamp(g, max=0.0)) / (1.0 + torch.exp(-torch.abs(g)))
        return g * torch.sigmoid(g)

    def forward(self, h, cos, sin, mask):
        rec, i = self.rec, self.i
        L = h.shape[1]
        if rec is not None:
            rec.note(f"in/{i}", h)
        x = self.input_layernorm(h)
        q = self.q_proj(x).reshape(1, L, HEADS, HEAD_DIM)
        k = self.k_proj(x).reshape(1, L, KV_HEADS, HEAD_DIM)
        v = self.v_proj(x).reshape(1, L, KV_HEADS, HEAD_DIM).transpose(1, 2)
        if rec is not None:
            rec.note(f"qn/{i}", q)
            rec.note(f"kn/{i}", k)
        q = self.q_norm(q).transpose(1, 2)                       # [1, 14, L, 64]
        k = self.k_norm(k).transpose(1, 2)                       # [1, 2, L, 64]
        c, s = cos.unsqueeze(1), sin.unsqueeze(1)                # [1, 1, L, 64]
        q = q * c + rotate_half(q) * s
        k = k * c + rotate_half(k) * s
        rep = HEADS // KV_HEADS
        k = torch.cat([k.reshape(KV_HEADS, 1, L, HEAD_DIM)] * rep, dim=1).reshape(1, HEADS, L, HEAD_DIM)
        v = torch.cat([v.reshape(KV_HEADS, 1, L, HEAD_DIM)] * rep, dim=1).reshape(1, HEADS, L, HEAD_DIM)
        scores = torch.matmul(q, k.transpose(2, 3)) * (HEAD_DIM ** -0.5) + mask
        if rec is not None:
            rec.note(f"scores/{i}", scores.amax(-1, keepdim=True))
        p = torch.softmax(scores, dim=-1, dtype=torch.float32)
        o = torch.matmul(p, v).transpose(1, 2).reshape(1, L, HEADS * HEAD_DIM)
        o = self.o_proj(o)
        if rec is not None:
            rec.note(f"o/{i}", o)
        h = h + o
        if rec is not None:
            rec.note(f"post/{i}", h)
        x = self.post_attention_layernorm(h)
        g = self.gate_proj(x)
        u = self.up_proj(x)
        m = self.silu(g) * u
        if rec is not None:
            rec.note(f"gate/{i}", g)
            rec.note(f"mlpmid/{i}", m)
        d = self.down_proj(m)
        if rec is not None:
            rec.note(f"down/{i}", d)
        return h + d


class DecGraph(nn.Module):
    def __init__(self, form):
        super().__init__()
        self.form = FORMS[form]
        self.layers = nn.ModuleList([Layer(i, self.form) for i in range(LAYERS)])
        self.norm = RMSNorm(HIDDEN, shift_for(self.form, "final", 0), self.form["mean_norm"])
        self.rec = None

    def set_recorder(self, rec):
        self.rec = rec
        for l in self.layers:
            l.rec = rec

    def forward(self, embeds, cos, sin, mask):
        h = embeds
        for layer in self.layers:
            h = layer(h, cos, sin, mask)
        if self.rec is not None:
            self.rec.note("final/0", h)
        return {"hidden": self.norm(h)}


def load_graph(form, weights):
    """DecGraph with the merged fp32 weights (strict: every tensor of the module, nothing else but the table)."""
    from safetensors.torch import load_file
    sd = load_file(str(weights))
    g = DecGraph(form)
    mapped = {}
    for k, v in sd.items():
        if k == "model.embed_tokens.weight":
            continue
        assert k.startswith("model."), k
        mapped[k[len("model."):].replace("self_attn.", "").replace("mlp.", "")] = v
    missing, unexpected = g.load_state_dict(mapped, strict=False)
    assert not missing and not unexpected, (missing[:5], unexpected[:5])
    return g.eval().requires_grad_(False)
