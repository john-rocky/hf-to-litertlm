"""D1Audio: the d1-omni audio prefix graph — ConvSubsampling + FastConformer 17 layers + Adapter + Residual — as one
static-T graph body (the mel front end is on the host: host/d1_audio_host.py). Pure torch at import: it loads in the
reference venv (venv-ref) and in the exporter venv (lt094dev).

    forward(mel f32 [1, 128, T], mel_valid f32 [1, T], v1 f32 [1, T1], v2 f32 [1, T2], v3 f32 [1, T3])
        -> {"prefix": f32 [1, T3, 1024]}     T1 = (T - 1) // 2 + 1, T2, T3 likewise; the host keeps rows [:P]

The provider's audio.py (Audio.forward = residual(adapter(encoder(mel, frames)[:lengths]))) written for one static T:
- ConvSubsampling: Conv2d(1->256, 3, s2, p1) -> ReLU -> dw Conv2d(3, s2, p1) -> pw 1x1 -> ReLU -> dw -> pw -> ReLU ->
  Linear(256 * 16 -> 512). The provider multiplies by the time mask before every layer and once at the end; here the
  mask multiplies sit where they change a value: on the mel input (mel_valid), and after each ReLU (v1 / v2 / v3).
  The five the provider has besides those multiply a tensor that is already 0 on the invalid frames (the ReLU of a
  masked tensor, the masked tensor itself) or sit before a pointwise op whose invalid rows the next mask zeroes;
  relu(x) * m == relu(x * m) for m in {0, 1}. Every row equals the provider's (checked bit for bit, eager gate).
- 17 ConformerLayer: x += FF1(LN x) * 0.5; x += RelPosMHA(LN x); x += ConvModule(LN x); x += FF2(LN x) * 0.5;
  x = LN_out(x).
  RelPosMHA (Transformer-XL): q / k / v / out linears with bias; ac = (q + u) k^T; bd = (q + v) p^T with p =
  linear_pos(pos_emb) FOLDED into a per-bucket contiguous constant p_t [1, 8, 64, 2 T3 - 1] (pos_emb = the provider's
  Conformer.pos_emb(T3), built on the CPU in fp32, times the checkpoint's linear_pos in the module's dtype); the
  rel-shift is the provider's PAD (1, 0) -> RESHAPE [1, 8, 2 T3, T3] -> SLICE [:, :, 1:] -> RESHAPE [1, 8, T3,
  2 T3 - 1] -> SLICE [..., :T3]; scores = (ac + bd) * 0.125 (= / sqrt(64), exact); the provider's
  masked_fill(mask, -1e4) becomes an additive mask (1 - v3_q v3_k) * -1e4 (exp of it is exactly 0 in fp32 next to
  any finite score), softmax, then * v3_q (the provider's masked_fill(mask, 0) zeroes the pad query rows).
  ConvModule channels-last: pointwise_conv1 as a Linear (k = 1) -> GLU (a * sigmoid(b)) -> * v3 (the provider's
  masked_fill(pad, 0)) -> depthwise Conv1d k 9 with padding 4 (the provider pads (4, 4) by hand) -> BatchNorm1d eval
  as an affine (scale = weight / sqrt(var + 1e-5), shift = bias - mean * scale, computed as ATen's inference kernel
  does: invstd = 1 / sqrt(var + eps), alpha = invstd * weight, beta = bias - mean * alpha) -> SiLU ->
  pointwise_conv2 as a Linear.
- Adapter: LN(512) -> Linear 512 -> 1024 -> GELU(erf) -> Linear 1024 -> 1024. Residual: x + up(GELU(down(LN x))).
- Every tensor rank <= 4; no repeat_interleave / expand / broadcast_to; no gather; no int64.

Weights: load_checkpoint() maps the checkpoint's `audio.*` tensors (KEY_RULES; num_batches_tracked is not read) and
fold() computes the constants (p_t, BN scale / shift) in the module's dtype.

CLI (from K; the exporter venv for --build / --fp16 / --smoke, either venv for --keymap):
    venv-ref/bin/python scripts/audio_graph.py --keymap                       # step 0 -> results/audio_keymap.json
    $Q/quiet_wait.py -- ~/venvs/lt094dev/bin/python scripts/audio_graph.py --build 1001   # step 3, fp32 file + scan
    $Q/quiet_wait.py -- ~/venvs/lt094dev/bin/python scripts/audio_graph.py --fp16 1001    # FC-only fp16 form

Round 12 (additions; every call above behaves as before, the default model is unchanged):
    ~/venvs/lt094dev/bin/python scripts/audio_graph.py --k-table           # -> results/audio_f16safe_k_table.json
    $Q/quiet_wait.py -- ~/venvs/lt094dev/bin/python scripts/audio_graph.py --build 1001 --variant f16safe   (or clast)
    $Q/quiet_wait.py -- ~/venvs/lt094dev/bin/python scripts/audio_graph.py --fp16 1001 --variant f16safe
  --variant f16safe: every LayerNorm site whose squared deviation (x - mean)^2 can pass fp16's 65,504 on the fixture
    takes its input x 2^-k and eps x 4^-k (d1_graph_f16safe.ScaledLayerNorm, the round 8 form: the same function in
    fp32), k per site = the smallest k >= 0 with max (x - mean)^2 x 4^-k <= 65,504 / 4 (margin >= 4x), the maxima read
    from round 7's results/audio_fp16_range.json (7 clips, host mel); sites already within the margin keep the
    original module (k = 0). The k table: results/audio_f16safe_k_table.json (--k-table).
  --variant clast: the depthwise conv of every ConvModule computed from the channels-last activation [1, T3, 512]
    viewed as NHWC [1, 1, T3, 512]: torch's conv2d needs NCHW, so the module permutes to [1, 512, 1, T3] right before
    the conv and back right after it; that permute sits next to the converter's own NCHW <-> NHWC transposes, which the
    converter may fold away (round 7: 4 TRANSPOSE around each of the 17 convs, 68 of 155). The op scan says what it did.
  Files: out/d1omni_audio_T<T>_<variant>_{fp32,fp16}.tflite, results/audio_{opscan,signature}_T<T>_<variant>*.json,
  results/audio_quant_<variant>.json (round 7's result files are never written by a variant run).
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
K = HERE.parent

D_MODEL, HEADS, D_K, FF, N_LAYERS, CONV_K, SUB_CH, FEAT, OUT, RES_W = 512, 8, 64, 2048, 17, 9, 256, 128, 1024, 512
NEG = -1e4                       # the provider's masked_fill value (-10000.0)
BN_EPS = 1e-5
T_BUCKETS = (501, 1001, 2001, 3001)
INPUT_NAMES = ("mel", "mel_valid", "v1", "v2", "v3")


def sub_len(n: int) -> int:
    return (n + 2 - 3) // 2 + 1


def dims(T: int) -> tuple[int, int, int]:
    t1 = sub_len(T)
    t2 = sub_len(t1)
    return t1, t2, sub_len(t2)


def pos_table(t: int, d: int = D_MODEL) -> torch.Tensor:
    """The provider's Conformer.pos_emb(t) on the CPU in fp32: relative positions t-1 .. -(t-1) -> [1, 2t-1, d]."""
    positions = torch.arange(t - 1, -t, -1, dtype=torch.float32)[:, None]
    div = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * -(math.log(10000.0) / d))
    pe = torch.zeros(len(positions), d)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(positions * div), torch.cos(positions * div)
    return pe[None]


class Subsampling(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(1, SUB_CH, 3, 2, 1)
        self.conv2 = nn.Conv2d(SUB_CH, SUB_CH, 3, 2, 1, groups=SUB_CH)
        self.conv3 = nn.Conv2d(SUB_CH, SUB_CH, 1)
        self.conv5 = nn.Conv2d(SUB_CH, SUB_CH, 3, 2, 1, groups=SUB_CH)
        self.conv6 = nn.Conv2d(SUB_CH, SUB_CH, 1)
        self.out = nn.Linear(SUB_CH * (FEAT // 8), D_MODEL)

    def forward(self, mel, mel_valid, v1, v2, v3):
        x = mel.transpose(1, 2)[:, None] * mel_valid[:, None, :, None]          # [1, 1, T, 128]
        x = F.relu(self.conv0(x)) * v1[:, None, :, None]                      # [1, 256, T1, 64]
        x = F.relu(self.conv3(self.conv2(x))) * v2[:, None, :, None]          # [1, 256, T2, 32]
        x = F.relu(self.conv6(self.conv5(x))) * v3[:, None, :, None]          # [1, 256, T3, 16]
        b, c, t, f = x.shape
        return self.out(x.transpose(1, 2).reshape(b, t, c * f))               # [1, T3, 512]


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1, self.linear2 = nn.Linear(D_MODEL, FF), nn.Linear(FF, D_MODEL)

    def forward(self, x):
        return self.linear2(F.silu(self.linear1(x)))


class RelPosAttention(nn.Module):
    def __init__(self, t3: int):
        super().__init__()
        self.t3 = t3
        self.linear_q, self.linear_k = nn.Linear(D_MODEL, D_MODEL), nn.Linear(D_MODEL, D_MODEL)
        self.linear_v, self.linear_out = nn.Linear(D_MODEL, D_MODEL), nn.Linear(D_MODEL, D_MODEL)
        self.pos_bias_u = nn.Parameter(torch.zeros(1, 1, HEADS, D_K))
        self.pos_bias_v = nn.Parameter(torch.zeros(1, 1, HEADS, D_K))
        self.register_buffer("p_t", torch.zeros(1, HEADS, D_K, 2 * t3 - 1), persistent=False)

    def forward(self, x, addmask, vq):
        t = self.t3
        q = self.linear_q(x).reshape(1, t, HEADS, D_K)
        k = self.linear_k(x).reshape(1, t, HEADS, D_K).transpose(1, 2)
        v = self.linear_v(x).reshape(1, t, HEADS, D_K).transpose(1, 2)
        ac = torch.matmul((q + self.pos_bias_u).transpose(1, 2), k.transpose(2, 3))     # [1, 8, t, t]
        bd = torch.matmul((q + self.pos_bias_v).transpose(1, 2), self.p_t)               # [1, 8, t, 2t-1]
        bd = F.pad(bd, (1, 0)).reshape(1, HEADS, 2 * t, t)[:, :, 1:].reshape(1, HEADS, t, 2 * t - 1)[:, :, :, :t]
        s = (ac + bd) * 0.125 + addmask
        a = torch.softmax(s, dim=-1) * vq
        return self.linear_out(torch.matmul(a, v).transpose(1, 2).reshape(1, t, D_MODEL))


class ConvModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.pointwise_conv1 = nn.Linear(D_MODEL, 2 * D_MODEL)
        self.depthwise_conv = nn.Conv1d(D_MODEL, D_MODEL, CONV_K, padding=(CONV_K - 1) // 2, groups=D_MODEL)
        self.register_buffer("bn_scale", torch.ones(D_MODEL), persistent=False)
        self.register_buffer("bn_shift", torch.zeros(D_MODEL), persistent=False)
        self.pointwise_conv2 = nn.Linear(D_MODEL, D_MODEL)

    def forward(self, x, vt):
        y = self.pointwise_conv1(x)
        y = y[:, :, :D_MODEL] * torch.sigmoid(y[:, :, D_MODEL:]) * vt                   # GLU, then the pad rows -> 0
        y = self.depthwise_conv(y.transpose(1, 2)).transpose(1, 2)
        y = y * self.bn_scale + self.bn_shift
        return self.pointwise_conv2(F.silu(y))


class ConvModuleCL(ConvModule):
    """Round 12 (--variant clast): ConvModule with the depthwise conv taken from the channels-last activation. The
    activation [1, T3, 512] is viewed as NHWC [1, 1, T3, 512]; torch's conv2d takes NCHW, so it is permuted to
    [1, 512, 1, T3] right before the conv (kernel [512, 1, 1, 9], padding (0, 4), groups 512 = the same depthwise conv
    as the Conv1d) and back right after. Same parameters and keys as ConvModule; the function is the same, the fp32
    op order may differ (measured, not assumed)."""

    def forward(self, x, vt):
        y = self.pointwise_conv1(x)
        y = y[:, :, :D_MODEL] * torch.sigmoid(y[:, :, D_MODEL:]) * vt                   # GLU, then the pad rows -> 0
        t = y.shape[1]
        w = self.depthwise_conv.weight.unsqueeze(2)                                      # [512, 1, 9] -> [512, 1, 1, 9]
        z = F.conv2d(y.reshape(1, 1, t, D_MODEL).permute(0, 3, 1, 2), w, self.depthwise_conv.bias,
                     padding=(0, (CONV_K - 1) // 2), groups=D_MODEL)                    # [1, 512, 1, t]
        y = z.permute(0, 2, 3, 1).reshape(1, t, D_MODEL)
        y = y * self.bn_scale + self.bn_shift
        return self.pointwise_conv2(F.silu(y))


class ConformerLayer(nn.Module):
    def __init__(self, t3: int, conv_cls=ConvModule):
        super().__init__()
        self.norm_feed_forward1, self.feed_forward1 = nn.LayerNorm(D_MODEL), FeedForward()
        self.norm_self_att, self.self_attn = nn.LayerNorm(D_MODEL), RelPosAttention(t3)
        self.norm_conv, self.conv = nn.LayerNorm(D_MODEL), conv_cls()
        self.norm_feed_forward2, self.feed_forward2 = nn.LayerNorm(D_MODEL), FeedForward()
        self.norm_out = nn.LayerNorm(D_MODEL)

    def forward(self, x, addmask, vq, vt):
        x = x + self.feed_forward1(self.norm_feed_forward1(x)) * 0.5
        x = x + self.self_attn(self.norm_self_att(x), addmask, vq)
        x = x + self.conv(self.norm_conv(x), vt)
        x = x + self.feed_forward2(self.norm_feed_forward2(x)) * 0.5
        return self.norm_out(x)


class Adapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm, self.linear_1, self.linear_2 = nn.LayerNorm(D_MODEL), nn.Linear(D_MODEL, OUT), nn.Linear(OUT, OUT)

    def forward(self, x):
        return self.linear_2(F.gelu(self.linear_1(self.norm(x))))


class Residual(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln, self.down, self.up = nn.LayerNorm(OUT), nn.Linear(OUT, RES_W), nn.Linear(RES_W, OUT)

    def forward(self, x):
        return x + self.up(F.gelu(self.down(self.ln(x))))


class D1Audio(nn.Module):
    def __init__(self, T: int, variant: str | None = None):
        super().__init__()
        assert variant in VARIANTS, variant
        self.T = T
        self.T1, self.T2, self.T3 = dims(T)
        self.variant = variant
        self.pre_encode = Subsampling()
        conv_cls = ConvModuleCL if variant == "clast" else ConvModule
        self.layers = nn.ModuleList(ConformerLayer(self.T3, conv_cls) for _ in range(N_LAYERS))
        self.adapter = Adapter()
        self.residual = Residual()
        self.raw = {}            # checkpoint tensors behind the folded constants (linear_pos, BatchNorm), not modules
        self.f16safe_report = apply_f16safe(self, load_k_table()) if variant == "f16safe" else None

    def encode(self, mel, mel_valid, v1, v2, v3):
        x = self.pre_encode(mel, mel_valid, v1, v2, v3)
        vt = v3[:, :, None]                                                              # [1, T3, 1]
        vq = v3[:, None, :, None]                                                        # [1, 1, T3, 1]
        addmask = (1.0 - vq * v3[:, None, None, :]) * NEG                               # [1, 1, T3, T3]
        for layer in self.layers:
            x = layer(x, addmask, vq, vt)
        return x

    def forward(self, mel, mel_valid, v1, v2, v3):
        return {"prefix": self.residual(self.adapter(self.encode(mel, mel_valid, v1, v2, v3)))}

    @torch.no_grad()
    def fold(self):
        """p_t and the BatchNorm affine from self.raw, in the module's current dtype."""
        dt = next(self.parameters()).dtype
        pe = pos_table(self.T3).to(dt)                                                   # the provider: .to(x.dtype)
        for i, layer in enumerate(self.layers):
            w_pos = self.raw[f"layers.{i}.linear_pos"].to(dt)
            p = F.linear(pe, w_pos)                                                      # [1, 2t-1, 512]
            layer.self_attn.p_t = p.view(1, -1, HEADS, D_K).permute(0, 2, 3, 1).contiguous()
            bn = {k: self.raw[f"layers.{i}.bn.{k}"].to(dt) for k in ("weight", "bias", "running_mean", "running_var")}
            invstd = 1 / torch.sqrt(bn["running_var"] + torch.tensor(BN_EPS, dtype=dt))
            alpha = invstd * bn["weight"]
            layer.conv.bn_scale = alpha.contiguous()
            layer.conv.bn_shift = (bn["bias"] - bn["running_mean"] * alpha).contiguous()
        return self


# ---------------------------------------------------------------- round 12 variants

VARIANTS = (None, "f16safe", "clast")
FP16_MAX = 65504.0
K_MARGIN = 4.0                  # the k rule: max (x - mean)^2 x 4^-k <= FP16_MAX / K_MARGIN
AUDIO_RANGE = K / "results/audio_fp16_range.json"
AUDIO_K_TABLE = K / "results/audio_f16safe_k_table.json"
NORM_NAMES = ("norm_feed_forward1", "norm_self_att", "norm_conv", "norm_feed_forward2", "norm_out")


def audio_norm_sites(model) -> dict:
    """site name (results/audio_fp16_range.json naming) -> (parent module, attribute): the 87 LayerNorms (17 layers x 5,
    the adapter's, the residual's)."""
    out = {}
    for i, layer in enumerate(model.layers):
        for n in NORM_NAMES:
            out[f"L{i:02d}.{n}"] = (layer, n)
    out["adapter.norm"] = (model.adapter, "norm")
    out["residual.ln"] = (model.residual, "ln")
    return out


def k_for(max_sq: float) -> int:
    k = 0
    while max_sq * 4.0 ** -k > FP16_MAX / K_MARGIN:
        k += 1
    return k


def load_k_table(path=AUDIO_K_TABLE) -> dict:
    doc = __import__("json").loads(Path(path).read_text())
    return {name: int(v["k"]) for name, v in doc["sites"].items()}


def apply_f16safe(model, k_table: dict) -> dict:
    """Swap every LayerNorm with k > 0 for d1_graph_f16safe.ScaledLayerNorm (before the weights are loaded: the
    parameter keys are unchanged). Every site must be in the table and the reverse."""
    import d1_graph_f16safe as F16

    sites = audio_norm_sites(model)
    assert set(sites) == set(k_table), sorted(set(sites) ^ set(k_table))
    scaled = {}
    for name, (parent, attr) in sites.items():
        old = getattr(parent, attr)
        k = int(k_table[name])
        assert type(old) is nn.LayerNorm and old.elementwise_affine and len(old.normalized_shape) == 1, name
        if k == 0:
            continue
        new = F16.ScaledLayerNorm(old.normalized_shape[0], old.eps, k)
        new.load_state_dict(old.state_dict())
        setattr(parent, attr, new)
        scaled[name] = k
    return {"sites": len(sites), "scaled": len(scaled), "k_by_site": scaled,
            "unchanged": sorted(n for n in sites if n not in scaled)}


def k_table_doc() -> dict:
    """results/audio_f16safe_k_table.json from round 7's range probe (eager fp32, host mel, the 7 clips' valid rows)."""
    import hashlib
    import json
    import time

    rng = json.loads(AUDIO_RANGE.read_text())
    allv = rng["all"]
    model = D1Audio(1001)
    sites = audio_norm_sites(model)
    out, hist = {}, {}
    for name, (parent, attr) in sites.items():
        m = getattr(parent, attr)
        dim, eps = int(m.normalized_shape[0]), float(m.eps)
        sq = allv[f"{name}.sq_dev_max"]
        var = allv.get(f"{name}.sq_dev_mean_max")
        mx = float(sq["max"])
        k = k_for(mx)
        hist[k] = hist.get(k, 0) + 1
        vmax = float(var["max"]) if isinstance(var, dict) else (float(var) if var is not None else None)
        e = {"kind": "layernorm", "dim": dim, "eps": eps, "k": k, "scale_s": 2.0 ** -k, "eps_k": eps * 4.0 ** -k,
             "sq_dev_max": mx, "sq_dev_max_clip": sq.get("clip"), "overflows_fp16_before": mx > FP16_MAX,
             "sq_dev_max_after_k": mx * 4.0 ** -k, "margin_after_k": FP16_MAX / (mx * 4.0 ** -k),
             "variance_max": vmax}
        if vmax is not None:
            # observation only (not the rule): the row's sum of squared deviations = variance x dim; an fp16
            # accumulator inside the MEAN would hold it (FP16_WITH_FP32_ACCUM accumulates in fp32: unverified here)
            e.update(sum_sq_dev_max_est=vmax * dim, sum_sq_dev_max_est_after_k=vmax * dim * 4.0 ** -k,
                     sum_after_k_over_fp16_max=vmax * dim * 4.0 ** -k > FP16_MAX)
        out[name] = e
    mins = min(out.items(), key=lambda kv: kv[1]["margin_after_k"])
    doc = {"step": "round 12 step 1: the per-site k of the audio graph's fp16-safe LayerNorm (input x 2^-k, eps x 4^-k)",
           "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "source": {"file": str(AUDIO_RANGE.relative_to(K)), "sha256": hashlib.sha256(AUDIO_RANGE.read_bytes()).hexdigest(),
                      "written": rng.get("written"), "step": rng.get("step")},
           "rule": "k = the smallest k >= 0 with max (x - mean)^2 (the element, fixture valid rows) x 4^-k <= 65,504 / 4 "
                   "(margin >= 4x); a site already within the margin keeps k = 0 and its original module (launch r12)",
           "fp16": {"max": FP16_MAX, "min_normal": 2.0 ** -14, "min_subnormal": 2.0 ** -24},
           "summary": {"sites": len(out), "sites_k_gt_0": sum(v["k"] > 0 for v in out.values()),
                       "k_histogram": {str(k): n for k, n in sorted(hist.items())},
                       "sites_overflowing_before": sorted(n for n, v in out.items() if v["overflows_fp16_before"]),
                       "min_margin_after_k": mins[1]["margin_after_k"], "min_margin_site": mins[0],
                       "all_margins_ge_4_after_k": all(v["margin_after_k"] >= K_MARGIN for v in out.values()),
                       "sites_sum_after_k_over_fp16_max_observation": sorted(n for n, v in out.items()
                                                                            if v.get("sum_after_k_over_fp16_max"))},
           "sites": out}
    return doc


# ---------------------------------------------------------------- weights

# checkpoint key (audio.*) -> this module's key; how: copy | pw (Conv1d k=1 weight [o, i, 1] -> Linear [o, i]) |
# uv ([8, 64] -> [1, 1, 8, 64]) | raw:<name> (kept for fold(): linear_pos, BatchNorm) | skip (num_batches_tracked)
KEY_RULES = [
    (r"audio\.encoder\.pre_encode\.conv\.(0|2|3|5|6)\.(weight|bias)", "pre_encode.conv{0}.{1}", "copy"),
    (r"audio\.encoder\.pre_encode\.out\.(weight|bias)", "pre_encode.out.{0}", "copy"),
    (r"audio\.encoder\.layers\.(\d+)\.(norm_feed_forward1|norm_self_att|norm_conv|norm_feed_forward2|norm_out)\."
     r"(weight|bias)", "layers.{0}.{1}.{2}", "copy"),
    (r"audio\.encoder\.layers\.(\d+)\.(feed_forward1|feed_forward2)\.(linear1|linear2)\.(weight|bias)",
     "layers.{0}.{1}.{2}.{3}", "copy"),
    (r"audio\.encoder\.layers\.(\d+)\.self_attn\.(linear_q|linear_k|linear_v|linear_out)\.(weight|bias)",
     "layers.{0}.self_attn.{1}.{2}", "copy"),
    (r"audio\.encoder\.layers\.(\d+)\.self_attn\.(pos_bias_u|pos_bias_v)", "layers.{0}.self_attn.{1}", "uv"),
    (r"audio\.encoder\.layers\.(\d+)\.self_attn\.linear_pos\.weight", "layers.{0}.linear_pos", "raw"),
    (r"audio\.encoder\.layers\.(\d+)\.conv\.(pointwise_conv1|pointwise_conv2)\.weight", "layers.{0}.conv.{1}.weight",
     "pw"),
    (r"audio\.encoder\.layers\.(\d+)\.conv\.(pointwise_conv1|pointwise_conv2)\.bias", "layers.{0}.conv.{1}.bias",
     "copy"),
    (r"audio\.encoder\.layers\.(\d+)\.conv\.depthwise_conv\.(weight|bias)", "layers.{0}.conv.depthwise_conv.{1}",
     "copy"),
    (r"audio\.encoder\.layers\.(\d+)\.conv\.batch_norm\.(weight|bias|running_mean|running_var)",
     "layers.{0}.bn.{1}", "raw"),
    (r"audio\.encoder\.layers\.(\d+)\.conv\.batch_norm\.num_batches_tracked", None, "skip"),
    (r"audio\.adapter\.(norm|linear_1|linear_2)\.(weight|bias)", "adapter.{0}.{1}", "copy"),
    (r"audio\.residual\.(ln|down|up)\.(weight|bias)", "residual.{0}.{1}", "copy"),
]


def map_key(key: str):
    """-> (our key or None, how) for one checkpoint key; None if no rule matches."""
    for pat, tgt, how in KEY_RULES:
        m = re.fullmatch(pat, key)
        if m:
            return (tgt.format(*m.groups()) if tgt else None), how
    return None


def checkpoint_audio(path) -> dict:
    from safetensors import safe_open

    out = {}
    with safe_open(str(path), framework="pt") as f:
        for key in f.keys():
            if key.startswith("audio."):
                out[key] = f.get_tensor(key)
    return out


def load_state(model: D1Audio, sd: dict) -> dict:
    """Checkpoint `audio.*` tensors -> model (strict: every parameter set, nothing unmatched), then fold()."""
    ours, unmatched, skipped, rows = {}, [], {}, []
    params = dict(model.named_parameters())
    for key, t in sd.items():
        r = map_key(key)
        if r is None:
            unmatched.append(key)
            continue
        tgt, how = r
        if how == "skip":
            skipped[key] = int(t.item()) if t.numel() == 1 else list(t.shape)
            rows.append({"key": key, "how": "skip (int64 counter, not read)", "shape": list(t.shape),
                         "dtype": str(t.dtype)})
            continue
        assert t.dtype == torch.float32, (key, t.dtype)
        if how == "raw":
            model.raw[tgt] = t.detach().clone().contiguous()
            rows.append({"key": key, "target": f"raw[{tgt}] -> fold()", "how": how, "shape": list(t.shape)})
            continue
        src = t.detach()
        if how == "pw":
            assert src.dim() == 3 and src.shape[2] == 1, (key, src.shape)
            src = src[:, :, 0]
        elif how == "uv":
            src = src.reshape(1, 1, HEADS, D_K)
        assert tgt in params, (key, tgt)
        assert tuple(params[tgt].shape) == tuple(src.shape), (key, tgt, params[tgt].shape, src.shape)
        ours[tgt] = src.clone().contiguous()
        rows.append({"key": key, "target": tgt, "how": how, "shape": list(t.shape), "target_shape": list(src.shape)})
    missing, unexpected = model.load_state_dict(ours, strict=False)
    raw_expected = {f"layers.{i}.linear_pos" for i in range(N_LAYERS)} | {
        f"layers.{i}.bn.{k}" for i in range(N_LAYERS) for k in ("weight", "bias", "running_mean", "running_var")}
    report = {"source_tensors": len(sd), "mapped_parameters": len(ours), "raw_for_fold": len(model.raw),
              "skipped": skipped, "unmatched": unmatched, "missing_parameters": list(missing),
              "unexpected": list(unexpected), "raw_missing": sorted(raw_expected - set(model.raw)),
              "model_parameters": len(params), "rows": rows}
    assert not unmatched and not missing and not unexpected and not report["raw_missing"], {
        k: report[k] for k in ("unmatched", "missing_parameters", "unexpected", "raw_missing")}
    model.fold()
    return report


def build(T: int, weights_path=None, dtype=torch.float32, variant=None):
    """D1Audio(T) with the checkpoint (default: the pinned snapshot's model.safetensors), eval, folded."""
    sys.path.insert(0, str(HERE))
    import d1_src as S

    model = D1Audio(T, variant).eval()
    sd = checkpoint_audio(weights_path or S.WEIGHTS)
    report = load_state(model, sd)
    if dtype != torch.float32:
        model = model.to(dtype)
        model.fold()
    return model, report


# ---------------------------------------------------------------- samples (host inputs from the fixture clips)

def clip_samples(record_id: str, seconds: float | None = None):
    """int16 samples of a fixture clip (wav via the stdlib, flac via soundfile), optionally cut to `seconds`."""
    import json

    import numpy as np

    doc = json.loads((K / "fixtures/requests.json").read_text())
    rec = next(r for r in doc["records"] if r["id"] == record_id)
    path = K / rec["media"]["ref"]
    if path.suffix == ".wav":
        import wave

        with wave.open(str(path), "rb") as w:
            assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2), path
            x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.int16)
    else:
        import soundfile as sf

        x, rate = sf.read(str(path), dtype="int16")
        assert rate == 16000 and x.ndim == 1, (path, rate, x.shape)
    if seconds is not None:
        x = x[: int(round(seconds * 16000))]
    return x, str(path.relative_to(K))


def host_inputs(record_id: str, T: int | None = None, seconds: float | None = None, precision: str = "float32"):
    """-> (inputs dict of numpy, info) through host/d1_audio_host.prepare."""
    sys.path.insert(0, str(K / "host"))
    import d1_audio_host as AH

    x, path = clip_samples(record_id, seconds)
    inp, info = AH.prepare(x, precision=precision, bucket=T)
    info.update(record=record_id, file=path, seconds_cut=seconds)
    return inp, info


LONG_CLIP = ("aud_01", "aud_02", "aud_03")      # concatenated: 27.2 s, the T3001 sample (no oracle row of its own)


def long_samples():
    import numpy as np

    parts = [clip_samples(r)[0] for r in LONG_CLIP]
    return np.concatenate(parts), "+".join(LONG_CLIP)


def sample_for(T: int):
    """The trace / smoke sample of bucket T: a real clip that lands in it (5 s cut of aud_01, aud_01, card_topic, the
    27.2 s concatenation of aud_01..03)."""
    sys.path.insert(0, str(K / "host"))
    import d1_audio_host as AH

    if T == 501:
        return host_inputs("aud_01", seconds=5.0)
    if T == 1001:
        return host_inputs("aud_01")
    if T == 2001:
        return host_inputs("card_topic")
    x, name = long_samples()
    inp, info = AH.prepare(x)
    info.update(record=name)
    assert info["T_b"] == 3001, info
    return inp, info


class AudioRunner:
    """CompiledModel buffers by signature name; __call__(inputs dict of numpy) -> prefix float32 [T3, 1024]."""

    def __init__(self, cm, signature):
        self.cm, self.sig = cm, signature
        self.in_det = cm.get_input_tensor_details(signature)
        self.out_det = cm.get_output_tensor_details(signature)
        assert sorted(self.in_det) == sorted(INPUT_NAMES), self.in_det
        assert list(self.out_det) == ["prefix"], self.out_det
        shp = [int(v) for v in self.out_det["prefix"]["shape"]]
        assert shp[0] == 1 and shp[2] == OUT, shp
        self.T3 = shp[1]
        self.ins = {n: cm.create_input_buffer_by_name(signature, n) for n in self.in_det}
        self.outs = {"prefix": cm.create_output_buffer_by_name(signature, "prefix")}

    def write(self, x):
        import numpy as np

        for n in INPUT_NAMES:
            v = x[n]
            assert v.dtype == np.float32, (n, v.dtype)
            self.ins[n].write(np.ascontiguousarray(v))

    def read(self):
        import numpy as np

        return np.asarray(self.outs["prefix"].read(self.T3 * OUT, np.float32), np.float32).reshape(self.T3, OUT)

    def __call__(self, x):
        self.write(x)
        self.cm.run_by_name(self.sig, self.ins, self.outs)
        return self.read().copy()

    def timed(self, x):
        """-> (prefix, wall clock ms at start, ms of write + run + read back, ms of run only); inputs contiguous."""
        import time

        wall = time.time() * 1000.0
        t = time.perf_counter()
        for n in INPUT_NAMES:
            self.ins[n].write(x[n])
        tr = time.perf_counter()
        self.cm.run_by_name(self.sig, self.ins, self.outs)
        run_ms = (time.perf_counter() - tr) * 1000.0
        out = self.read()
        return out, wall, (time.perf_counter() - t) * 1000.0, run_ms

    def close(self):
        for b in list(self.ins.values()) + list(self.outs.values()):
            try:
                b.destroy()
            except Exception:
                pass


def _build(a):
    """Step 3: out/d1omni_audio_T<T>_fp32.tflite (signature audio_<T>), the op scan, the signature, a CPU smoke run."""
    import importlib.metadata as md
    import json
    import resource
    import time
    import traceback

    import numpy as np

    sys.path.insert(0, str(HERE))
    import litert_run as R

    import litert_torch

    torch.set_num_threads(8)
    T = a.build
    v = getattr(a, "variant", None)
    if v:   # round 12: the variant's own files; round 7's are never touched
        path = K / f"out/d1omni_audio_T{T}_{v}_fp32.tflite"
        res_scan = K / f"results/audio_opscan_T{T}_{v}_fp32.json"
        res_sig = K / f"results/audio_signature_T{T}_{v}.json"
    else:
        path = K / f"out/d1omni_audio_T{T}_fp32.tflite"
        res_scan = K / f"results/opscan_audio_T{T}_fp32.json"
        res_sig = K / f"results/signature_audio_T{T}.json"
    for p in (path, res_scan, res_sig):
        assert not p.exists(), f"refusing to overwrite {p}"
    t0 = time.time()
    model, rep = build(T, variant=v)
    kw_np, info = sample_for(T)
    kw = {k: torch.from_numpy(kw_np[k]) for k in INPUT_NAMES}
    with torch.no_grad():
        eager = model(**kw)["prefix"][0].numpy()
    sig = f"audio_{T}"
    rec = {"signature": sig, "T": T, "T123": list(dims(T)), "file": str(path.relative_to(K)), "sample": info,
           "inputs": {k: {"dtype": str(v.dtype), "shape": list(v.shape)} for k, v in kw.items()},
           "parameters": sum(p.numel() for p in model.parameters()),
           "weights_report": {k: v for k, v in rep.items() if k != "rows"},
           "eager_finite": bool(np.isfinite(eager).all()), "python": sys.version.split()[0], "torch": torch.__version__,
           "litert_torch": md.version("litert-torch"), "litert_converter": md.version("litert-converter"),
           "variant": v, "f16safe_report": model.f16safe_report}
    if v:
        with torch.no_grad():   # the variant's function vs the default module on the trace sample (fp32 eager)
            base, _ = build(T)
            ref0 = base(**kw)["prefix"][0].numpy()
            del base
        rec["eager_vs_default_module"] = {"max_abs_all_rows": float(np.abs(eager.astype(np.float64) - ref0).max()),
                                          "bit_equal_all_rows": bool(np.array_equal(eager, ref0))}
    try:
        t1 = time.time()
        edge = litert_torch.signature(sig, model, sample_kwargs=kw).convert()
        rec["convert_seconds"] = round(time.time() - t1, 1)
        t2 = time.time()
        edge.export(str(path))
        rec["export_seconds"] = round(time.time() - t2, 1)
        del edge
    except BaseException:
        (K / (f"logs/r12_audio_build_T{T}_{v}.traceback.txt" if v else f"logs/r7_audio_build_T{T}.traceback.txt")
         ).write_text(traceback.format_exc())
        raise
    rec["peak_rss_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    print(f"wrote {path} ({path.stat().st_size} B) convert {rec['convert_seconds']} s", flush=True)
    sc = R.scan(path)
    h = sc["op_histogram"]
    fc_expected = N_LAYERS * 10 + 1 + 2 + 2
    checks = {
        "CUSTOM_0": sc["custom_op_count"] == 0,
        "INT64_tensors_0": sc["int64_tensor_count"] == 0,
        "BROADCAST_TO_0": h.get("BROADCAST_TO", 0) == 0,
        "GATHER_ND_0": h.get("GATHER_ND", 0) == 0,
        "GATHER_0": h.get("GATHER", 0) == 0,
        "RESIZE_0": not any(k.startswith("RESIZE") for k in h),
        "max_tensor_rank_le_4": sc["max_tensor_rank"] <= 4,
        "one_signature_audio_T": [s["key"] for s in sc["signatures"]] == [sig],
        f"fully_connected_{fc_expected}": sc["fully_connected_count"] == fc_expected,
    }
    counts = {k: h.get(k, 0) for k in ("CONV_2D", "DEPTHWISE_CONV_2D", "CONV_1D", "PAD", "PADV2", "RESHAPE", "SLICE",
                                         "STRIDED_SLICE", "TRANSPOSE", "BATCH_MATMUL", "FULLY_CONNECTED", "MEAN",
                                         "SQUARED_DIFFERENCE", "RSQRT", "SOFTMAX", "LOGISTIC", "GELU", "MUL", "ADD",
                                         "SUB", "CONCATENATION", "SPLIT", "RELU", "EXPAND_DIMS", "SQUEEZE")}
    scan_doc = {"step": (f"round 12: op scan of the fp32 audio graph, variant {v}, T={T}" if v else
                         f"round 7 step 3: op scan of the fp32 audio graph, T={T}"),"written": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "export": rec, "checks": checks, "checks_pass": all(checks.values()), "op_counts": counts,
                "fc_expected": {"conformer (ff 4 + qkvo 4 + pointwise 2) x 17": N_LAYERS * 10, "subsampling out": 1,
                                "adapter": 2, "residual": 2}, "scan": sc}
    cm, desc = R.open_compiled(path, "cpu", threads=8)
    in_det, out_det = cm.get_input_tensor_details(sig), cm.get_output_tensor_details(sig)
    run = AudioRunner(cm, sig)
    got = run(kw_np)
    run.close()
    P = info["P"]
    smoke = {"backend": desc, "P": P, "T3": run.T3,
             "max_abs_valid_rows_vs_eager": float(np.abs(got[:P].astype(np.float64) - eager[:P]).max()),
             "max_abs_all_rows_vs_eager": float(np.abs(got.astype(np.float64) - eager).max()),
             "finite": bool(np.isfinite(got).all())}
    scan_doc["smoke_cpu_vs_eager"] = smoke
    sig_doc = {"step": f"round {12 if v else 7} step 3: signature of {path.relative_to(K)}","signature": sig,
               "flatbuffer_signatures": sc["signatures"],
               "compiled_model_inputs": {n: {k: str(v) for k, v in d.items()} for n, d in in_det.items()},
               "compiled_model_outputs": {n: {k: str(v) for k, v in d.items()} for n, d in out_det.items()},
               "inputs_contract": {"mel": "float32 [1, 128, T]: the host's normalised log-mel, frames 0 .. T_clip-1, "
                                          "0 after (any value is ignored where mel_valid = 0)",
                                   "mel_valid": "float32 [1, T]: 1 at t < frames (= n // 160), else 0",
                                   "v1": "float32 [1, T1]: 1 at t < L1", "v2": "float32 [1, T2]: 1 at t < L2",
                                   "v3": "float32 [1, T3]: 1 at t < L3 = P"},
               "output_contract": {"prefix": "float32 [1, T3, 1024]: rows 0 .. P-1 are the prefix; rows >= P are "
                                             "not prefix rows (finite, ignored)"},
               "file_bytes": sc["bytes"], "sha256": sc["sha256"]}
    rec["seconds_total"] = round(time.time() - t0, 1)
    R.dump_json(res_scan, scan_doc)
    R.dump_json(res_sig, sig_doc)
    print(json.dumps({"T": T, "bytes": sc["bytes"], "ops": sc["operator_count"], "checks": checks, "counts": counts,
                      "op_histogram": h, "smoke": smoke, "seconds": rec["seconds_total"]}, indent=1), flush=True)
    return 0 if all(checks.values()) else 1


# ---------------------------------------------------------------- CLI

def _keymap(a):
    import json
    import time

    import numpy as np

    sys.path.insert(0, str(HERE))
    import d1_src as S
    import litert_run as R

    torch.set_num_threads(8)
    model, report = build(1001)
    from safetensors import safe_open

    with safe_open(str(S.WEIGHTS), framework="pt") as f:
        n_audio = sum(1 for k in f.keys() if k.startswith("audio."))
        dtypes = {}
        for k in f.keys():
            if k.startswith("audio."):
                dtypes[f.get_slice(k).get_dtype()] = dtypes.get(f.get_slice(k).get_dtype(), 0) + 1
    bn = []
    for i, layer in enumerate(model.layers):
        rv = model.raw[f"layers.{i}.bn.running_var"]
        bn.append({"layer": i, "running_var_min": float(rv.min()), "running_var_max": float(rv.max()),
                   "running_mean_absmax": float(model.raw[f"layers.{i}.bn.running_mean"].abs().max()),
                   "scale_min": float(layer.conv.bn_scale.min()), "scale_max": float(layer.conv.bn_scale.max()),
                   "shift_absmax": float(layer.conv.bn_shift.abs().max())})
    doc = {"step": "round 7 step 0: checkpoint audio.* -> D1Audio (scripts/audio_graph.py KEY_RULES)",
           "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "weights": str(S.WEIGHTS),
           "checkpoint_audio_tensors": n_audio, "checkpoint_audio_dtypes": dtypes,
           "rules": [{"pattern": p, "target": t, "how": h} for p, t, h in KEY_RULES],
           "report": {k: v for k, v in report.items() if k != "rows"},
           "missing_total": len(report["unmatched"]) + len(report["missing_parameters"]) + len(report["raw_missing"]),
           "batch_norm_affine": {"eps": BN_EPS, "formula": "invstd = 1 / sqrt(running_var + eps); scale = invstd * "
                                 "weight; shift = bias - running_mean * scale (ATen batch_norm inference, fp32)",
                                 "per_layer": bn},
           "pos_table": {"rule": "the provider's Conformer.pos_emb(T3) (fp32, CPU) @ linear_pos.weight^T, viewed "
                                 "[1, 2T3-1, 8, 64] -> permuted to p_t [1, 8, 64, 2T3-1] per layer",
                         "T3_by_bucket": {str(T): dims(T)[2] for T in T_BUCKETS},
                         "p_t_bytes_fp32_all_layers_by_bucket": {str(T): N_LAYERS * HEADS * D_K * (2 * dims(T)[2] - 1) * 4
                                                                 for T in T_BUCKETS}},
           "parameters": sum(p.numel() for p in model.parameters()),
           "rows": report["rows"], "torch": torch.__version__, "python": sys.version.split()[0],
           "numpy": np.__version__, "executable": sys.executable}
    out = K / "results/audio_keymap.json"
    R.dump_json(out, doc, overwrite=a.overwrite)
    print(json.dumps({k: doc[k] for k in ("checkpoint_audio_tensors", "checkpoint_audio_dtypes", "missing_total",
                                          "parameters")} | {"report": doc["report"]}, indent=1, default=str))
    return 0


def weight_sources(path):
    """-> per conv / FC / BATCH_MATMUL op: the weight (input 1) dtype, directly or through a DEQUANTIZE producer."""
    import collections
    import mmap

    from ai_edge_litert import schema_py_generated as schema

    f = open(path, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    model = schema.Model.GetRootAs(mm, 0)
    op_names = {v: k for k, v in vars(schema.BuiltinOperator).items() if isinstance(v, int)}
    type_names = {v: k for k, v in vars(schema.TensorType).items() if isinstance(v, int)}
    codes = [op_names.get(max(model.OperatorCodes(i).BuiltinCode(), model.OperatorCodes(i).DeprecatedBuiltinCode()))
             for i in range(model.OperatorCodesLength())]
    g = model.Subgraphs(0)
    ops = [g.Operators(o) for o in range(g.OperatorsLength())]
    producer = {int(t): oi for oi, op in enumerate(ops) for t in op.OutputsAsNumpy()}
    out = collections.Counter()
    deq_consumers = collections.Counter()
    for op in ops:
        name = codes[op.OpcodeIndex()]
        ins = [int(x) for x in op.InputsAsNumpy()]
        if name == "DEQUANTIZE":
            continue
        for t in ins:
            if t in producer and codes[ops[producer[t]].OpcodeIndex()] == "DEQUANTIZE":
                deq_consumers[name] += 1
        if name in ("CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED", "BATCH_MATMUL") and len(ins) > 1:
            w = ins[1]
            if w in producer and codes[ops[producer[w]].OpcodeIndex()] == "DEQUANTIZE":
                src = int(ops[producer[w]].InputsAsNumpy()[0])
                out[f"{name}:DEQUANTIZE({type_names[g.Tensors(src).Type()]})"] += 1
            elif w in producer:
                out[f"{name}:activation"] += 1
            else:
                out[f"{name}:{type_names[g.Tensors(w).Type()]}"] += 1
    mm.close()
    f.close()
    return {"weight_source_by_op": dict(sorted(out.items())), "dequantize_consumers": dict(deq_consumers)}


def _fp16(a):
    """FC-only fp16 (ai-edge-quantizer FLOAT_CASTING on FULLY_CONNECTED; conv / depthwise conv / the folded position
    constants stay float32, fun-asr-nano's rule) -> out/d1omni_audio_T<T>_fp16.tflite + results/quant_audio.json."""
    import importlib.metadata as md
    import json
    import time

    sys.path.insert(0, str(HERE))
    import litert_run as R
    import quant_forms as QF
    from ai_edge_quantizer import quantizer

    T = a.fp16
    v = getattr(a, "variant", None)
    tag = f"_{v}" if v else ""
    src = K / f"out/d1omni_audio_T{T}{tag}_fp32.tflite"
    out = K / f"out/d1omni_audio_T{T}{tag}_fp16.tflite"
    res = K / (f"results/audio_quant_{v}.json" if v else "results/quant_audio.json")
    assert src.exists() and not out.exists(), (src, out)
    doc = json.loads(res.read_text()) if res.exists() else {"step": (f"round 12: FC-only fp16 form of the audio graph "
                                                                      f"variant {v} (FLOAT_CASTING, FULLY_CONNECTED only)"
                                                                      if v else "round 7 step 3: FC-only fp16 form of "
                                                                      "the audio graph (FLOAT_CASTING, FULLY_CONNECTED only)"),
                                                             "by_T": {}}
    assert str(T) not in doc["by_T"], f"T{T} already in {res.name}"
    sc_src = R.scan(src, with_sha=False)
    t0 = time.time()
    recipe, need_cal = QF.recipe_for("fp16")
    qt = quantizer.Quantizer(str(src), recipe)
    assert not (qt.need_calibration or need_cal)
    result = qt.quantize()
    result.export_model(str(out))
    del result, qt
    sec = round(time.time() - t0, 1)
    sc = R.scan(out)
    ws = weight_sources(out)
    n_fc = sc_src["fully_connected_count"]
    h0 = dict(sc_src["op_histogram"])
    h1 = dict(sc["op_histogram"])
    deq = h1.pop("DEQUANTIZE", 0)
    checks = {
        "ops_unchanged_except_DEQUANTIZE": h0 == h1,
        f"DEQUANTIZE_{n_fc}": deq == n_fc,
        f"fc_weight_all_FLOAT16_{n_fc}": sc["fc_weight_source_dtype"] == {"FLOAT16": n_fc},
        "dequantize_feeds_only_FC": ws["dequantize_consumers"] == {"FULLY_CONNECTED": n_fc},
        "conv_weights_FLOAT32": all(k.endswith("FLOAT32") for k in ws["weight_source_by_op"]
                                    if k.startswith(("CONV_2D", "DEPTHWISE_CONV_2D"))),
        "bmm_constant_FLOAT32": all(not k.startswith("BATCH_MATMUL:DEQUANTIZE") for k in ws["weight_source_by_op"]),
        "INT64_0": sc["int64_tensor_count"] == 0, "CUSTOM_0": sc["custom_op_count"] == 0,
    }
    doc["by_T"][str(T)] = {"input": {"file": str(src.relative_to(K)), "bytes": src.stat().st_size},
                           "output": str(out.relative_to(K)), "bytes": sc["bytes"], "sha256": sc["sha256"],
                           "seconds": sec, "recipe": recipe, "checks": checks, "checks_pass": all(checks.values()),
                           "op_histogram": sc["op_histogram"], "constant_bytes_by_dtype": sc["constant_bytes_by_dtype"],
                           "weight_sources": ws, "written": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    doc["versions"] = {p: md.version(p) for p in ("ai-edge-quantizer", "ai-edge-litert", "litert-torch")}
    R.dump_json(res, doc, overwrite=True)
    print(json.dumps({"T": T, "bytes": sc["bytes"], "checks": checks, "weight_sources": ws,
                      "constant_bytes_by_dtype": sc["constant_bytes_by_dtype"]}, indent=1))
    return 0 if all(checks.values()) else 1


def _convs(a):
    """Where the convolutions, pads and transposes of the fp32 audio graph landed (flatbuffer read only, no run) ->
    results/opscan_audio_T<T>_sites.json: every CONV_2D / DEPTHWISE_CONV_2D (input / filter / output shapes, stride,
    padding, fused activation, depth multiplier), every PAD (shape, paddings), the TRANSPOSE shape / perm groups."""
    import collections
    import json
    import mmap
    import time

    import numpy as np
    from ai_edge_litert import schema_py_generated as schema

    sys.path.insert(0, str(HERE))
    import litert_run as R

    T = a.convs
    v = getattr(a, "variant", None)
    path = K / (f"out/d1omni_audio_T{T}_{v}_fp32.tflite" if v else f"out/d1omni_audio_T{T}_fp32.tflite")
    f = open(path, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    model = schema.Model.GetRootAs(mm, 0)
    op_names = {v: k for k, v in vars(schema.BuiltinOperator).items() if isinstance(v, int)}
    pad_names = {v: k for k, v in vars(schema.Padding).items() if isinstance(v, int)}
    act_names = {v: k for k, v in vars(schema.ActivationFunctionType).items() if isinstance(v, int)}
    codes = [op_names.get(max(model.OperatorCodes(i).BuiltinCode(), model.OperatorCodes(i).DeprecatedBuiltinCode()))
             for i in range(model.OperatorCodesLength())]
    g = model.Subgraphs(0)
    shape = lambda t: g.Tensors(t).ShapeAsNumpy().tolist() if g.Tensors(t).ShapeLength() else []
    producer = {}
    for oi in range(g.OperatorsLength()):
        for t in g.Operators(oi).OutputsAsNumpy():
            producer[int(t)] = oi

    def const_values(t):
        tt = g.Tensors(t)
        b = model.Buffers(tt.Buffer())
        if b.DataLength() > 0:
            return np.frombuffer(b.DataAsNumpy().tobytes(), dtype=np.int32).tolist()
        if b.Offset() > 1:
            return np.frombuffer(mm[b.Offset(): b.Offset() + b.Size()], dtype=np.int32).tolist()
        return None

    convs, pads, transposes = [], [], collections.Counter()
    for oi in range(g.OperatorsLength()):
        op = g.Operators(oi)
        name = codes[op.OpcodeIndex()]
        ins = [int(x) for x in op.InputsAsNumpy()]
        outs = [int(x) for x in op.OutputsAsNumpy()]
        if name in ("CONV_2D", "DEPTHWISE_CONV_2D"):
            opt_t = schema.Conv2DOptions() if name == "CONV_2D" else schema.DepthwiseConv2DOptions()
            o = op.BuiltinOptions()
            opt_t.Init(o.Bytes, o.Pos)
            e = {"op_index": oi, "op": name, "input": shape(ins[0]), "filter": shape(ins[1]), "output": shape(outs[0]),
                 "stride": [opt_t.StrideH(), opt_t.StrideW()], "padding": pad_names.get(opt_t.Padding()),
                 "fused_activation": act_names.get(opt_t.FusedActivationFunction()),
                 "input_from": codes[g.Operators(producer[ins[0]]).OpcodeIndex()] if ins[0] in producer else "input"}
            if name == "DEPTHWISE_CONV_2D":
                e["depth_multiplier"] = opt_t.DepthMultiplier()
            convs.append(e)
        elif name in ("PAD", "PADV2"):
            pads.append({"op_index": oi, "input": shape(ins[0]), "paddings": const_values(ins[1]),
                         "consumer_shape": shape(outs[0])})
        elif name == "TRANSPOSE":
            transposes[json.dumps([shape(ins[0]), const_values(ins[1])])] += 1
    mm.close()
    f.close()
    groups = collections.Counter(json.dumps({k: v for k, v in c.items() if k != "op_index"}) for c in convs)
    doc = {"step": f"round {12 if v else 7}: conv / pad / transpose sites of {path.relative_to(K)} (flatbuffer read)",
           "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "conv_groups": [{"count": n, **json.loads(k)} for k, n in groups.most_common()],
           "pad_groups": [{"count": n, **json.loads(k)} for k, n in collections.Counter(
               json.dumps({k: v for k, v in p.items() if k != "op_index"}) for p in pads).most_common()],
           "transpose_groups": [{"count": n, "input_shape": json.loads(k)[0], "perm": json.loads(k)[1]}
                                for k, n in transposes.most_common()],
           "transpose_total": sum(transposes.values())}
    R.dump_json(K / (f"results/audio_opscan_T{T}_{v}_sites.json" if v else f"results/opscan_audio_T{T}_sites.json"),
                doc, overwrite=a.overwrite)
    print(json.dumps(doc, indent=1)[:5000])
    return 0


def _contract(a):
    """Step 6: the `audio` section of results/contract_draft.json (the file is read again right before the write and
    only the key `audio` is added or replaced; every other key is kept as it is)."""
    import json
    import os
    import time

    sys.path.insert(0, str(K / "host"))
    import d1_audio_host as AH

    sigs, files = {}, {}
    q = json.loads((K / "results/quant_audio.json").read_text())
    for T in T_BUCKETS:
        sp = K / f"results/signature_audio_T{T}.json"
        if not sp.exists():
            continue
        sd = json.loads(sp.read_text())
        fb = sd["flatbuffer_signatures"][0]
        sigs[str(T)] = {"signature": sd["signature"], "T123": list(dims(T)),
                        "inputs_by_tensor_index": sorted([{"name": i["name"], "dtype": i["dtype"], "shape": i["shape"],
                                                           "tensor_index": i["tensor_index"]} for i in fb["inputs"]],
                                                         key=lambda i: i["tensor_index"]),
                        "outputs": [{"name": o["name"], "dtype": o["dtype"], "shape": o["shape"]} for o in fb["outputs"]]}
        e16 = q["by_T"].get(str(T), {})
        files[str(T)] = {"fp32": {"file": f"out/d1omni_audio_T{T}_fp32.tflite", "bytes": sd["file_bytes"],
                                  "sha256": sd["sha256"]},
                         "fp16": {"file": e16.get("output"), "bytes": e16.get("bytes"), "sha256": e16.get("sha256")}}
    gates = {}
    for p in sorted((K / "results").glob("audio_parity_*.json")):
        d = json.loads(p.read_text())
        s = d.get("summary") or {}
        gates[p.name] = {k: s.get(k) for k in ("prefix_host_mel_max_abs", "prefix_host_mel_max_rel_to_absmax",
                                               "e2e_host_mel_max_abs_dp", "e2e_argmax", "e2e_bar_pass",
                                               "T3001_prefix_host_mel_max_abs", "replacing")}
    rec = None
    if a.mac_recommendation:
        rec = json.loads(Path(a.mac_recommendation).read_text())
    buckets = []
    for T in T_BUCKETS:
        t1, t2, t3 = dims(T)
        n_max = min((T - 1) * AH.HOP + AH.HOP - 1, AH.MAX_SECONDS * AH.SAMPLE_RATE)   # waveform() cuts at 30 s
        buckets.append({"T": T, "T1": t1, "T2": t2, "T3": t3, "clip_samples_max": n_max,
                        "clip_seconds_max": round(n_max / AH.SAMPLE_RATE, 4),
                        "P_max": AH.lengths(n_max // AH.HOP)[2],
                        "fp32_bytes": (files.get(str(T)) or {}).get("fp32", {}).get("bytes"),
                        "fp16_bytes": (files.get(str(T)) or {}).get("fp16", {}).get("bytes")})
    section = {
        "what": "d1-omni-600M audio prefix graph (D1Audio: ConvSubsampling + FastConformer 17 + Adapter + Residual) "
                "<-> host contract, round 7 draft (the mel front end is on the host)",
        "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "graph": {"one_file_per_bucket": True, "signature_name": "audio_<T>", "by_T": sigs, "weight_forms": files,
                  "inputs": {"mel": "float32 [1, 128, T]: the host's normalised log-mel (frames 0 .. n // 160), zero "
                                    "after; any value is ignored where mel_valid = 0",
                             "mel_valid": "float32 [1, T]: 1.0 at t < frames (= n // 160), else 0",
                             "v1": "float32 [1, T1]: 1.0 at t < L1", "v2": "float32 [1, T2]: 1.0 at t < L2",
                             "v3": "float32 [1, T3]: 1.0 at t < L3 = P"},
                  "output": {"prefix": "float32 [1, T3, 1024]: rows 0 .. P-1 are the audio prefix; rows >= P are "
                                       "finite and not used"},
                  "input_order_note": "bind by name (CompiledModel lists the inputs by name; the flatbuffer order is "
                                      "mel, mel_valid, v1, v2, v3)"},
        "buckets": buckets,
        "bucket_rule": "T = n // 160 + 1 STFT frames (n = samples after waveform()); the smallest T_b in "
                       "(501, 1001, 2001, 3001) with T <= T_b; a 30 s clip (480,000 samples) is T 3001, P 375",
        "host_steps": [
            "samples: 16 kHz mono; int16 -> float32 / 32768, float -> float32; cut to 30 s (480,000 samples); "
            "zero-pad to 8,000 samples (the provider's waveform())",
            "mel (float32, host/d1_audio_host.py): frames = n // 160; preemphasis y[0] = x[0], y[t] = x[t] - 0.97 x[t-1]; "
            "STFT n_fft 512, hop 160, Hann 400 (periodic=False, centred: 56 zeros each side), center=True with 256 zero "
            "samples each side -> n // 160 + 1 frames; |X|^2 (sqrt(re^2 + im^2), squared); Slaney mel 128 "
            "(slaney_filterbank, float32 [128, 257]); log(mel + 2^-24); per mel bin mean and std over the frames "
            "t < frames (std divides by count - 1, then + 1e-5; a NaN std is 0); frames t >= frames are 0",
            "lengths: L1 = (frames + 2 - 3) // 2 + 1, L2 = (L1 + 2 - 3) // 2 + 1, L3 = (L2 + 2 - 3) // 2 + 1 = P",
            "inputs: mel zero-padded on the right to [1, 128, T_b]; mel_valid / v1 / v2 / v3 = float32 1 at t < frames "
            "/ L1 / L2 / L3 (shapes [1, T_b], [1, T1], [1, T2], [1, T3])",
            "prefix = audio_<T_b>(**inputs)['prefix'][0, :P] (float32 [P, 1024])",
            "decision graph (contract above): the prefix rows at positions 0 .. P-1 (media = 1, keep_right = 0 at P-1), "
            "the question's ids after them; L = the smallest decide bucket >= P + n (the fixture's audio rows: L256); "
            "audio mode: state None -> {}, no temperature (raw softmax), noul default {false: no, true: yes}"],
        "host_mel_vs_provider": "float32 numpy front end vs the provider's MelFrontend: max |dmel| 1.5e-4 over the 7 "
                                "fixture clips (the provider's own float32 vs float64 run: 2.4e-4); the same steps in "
                                "float64 on both sides: 4e-15 (results/audio_mel_check.json)",
        "gates_at_writing": gates,
        "notes": ["Metal default precision (fp16 activations) breaks the audio graph: prefix max |d| 5.9, end to end "
                  "max |dp| 0.999 (25 LayerNorm squared-deviation sites exceed fp16's 65,504, up to 1.96e6, "
                  "results/audio_fp16_range.json) -> GpuOptions(enforce_f32=True) on Metal",
                  "the fp16 form = FULLY_CONNECTED weights in fp16 (FLOAT_CASTING); the convolutions, the BatchNorm "
                  "affine and the folded position constants stay float32"],
        **({"mac_recommendation": rec} if rec is not None else {}),
    }
    path = K / "results/contract_draft.json"
    doc = json.loads(path.read_text())                  # read right before the write: other rounds add their keys
    prev = doc.get("audio")
    if prev is not None:
        section["replaces"] = {"written": prev.get("written")}
    doc["audio"] = section
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, default=str) + "\n")
    os.replace(tmp, path)
    print(json.dumps({"keys": list(doc.keys()), "audio_buckets": buckets}, indent=1))
    return 0


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--keymap", action="store_true")
    ap.add_argument("--build", type=int, choices=T_BUCKETS)
    ap.add_argument("--fp16", type=int, choices=T_BUCKETS)
    ap.add_argument("--contract", action="store_true")
    ap.add_argument("--convs", type=int, choices=T_BUCKETS)
    ap.add_argument("--mac-recommendation")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--variant", choices=("f16safe", "clast"), help="round 12: --build / --fp16 / --convs of a variant")
    ap.add_argument("--k-table", action="store_true", help="round 12: results/audio_f16safe_k_table.json")
    a = ap.parse_args()
    if a.k_table:
        import json

        sys.path.insert(0, str(HERE))
        import litert_run as R

        doc = k_table_doc()
        R.dump_json(AUDIO_K_TABLE, doc)
        print(json.dumps(doc["summary"], indent=1))
        return 0 if doc["summary"]["all_margins_ge_4_after_k"] else 1
    if a.keymap:
        return _keymap(a)
    if a.build:
        return _build(a)
    if a.fp16:
        return _fp16(a)
    if a.contract:
        return _contract(a)
    if a.convs:
        return _convs(a)
    print(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main())
