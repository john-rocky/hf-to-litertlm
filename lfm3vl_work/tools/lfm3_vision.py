"""Plain-PyTorch re-author of the LFM3-VL vision tower + projector for ONE fixed G x G image (no transformers
import), split for the LiteRT-LM fast_vlm / generic image contract:

  VisionEncoder  images NHWC [1, G, G, 3] in [0, 1] -> [1, (G/16)^2, 1024]   (tower last hidden, RASTER order)
  VisionAdapter  [1, (G/16)^2, 1024] -> [1, (G/32)^2, 2048]                  (projector; HF's token order)

Entry points (a checkpoint with the same tower can load this file by path with its own weights and prefixes):
  build_tower(weights, config, g, tower_prefix=..., projector_prefix=..., dtype=torch.float32, opts=None)
      -> (VisionEncoder, VisionAdapter, info)
      weights = a safetensors file, a directory with model.safetensors (or a *.safetensors.index.json), or a
      mapping name -> tensor; config = config.json path or dict (reads vision_config, projector_*,
      downsample_factor, text_config.hidden_size); g = image side, a multiple of 32.
  export_tower(enc, adp, out_dir, forms=("fp32", "fp16", "int8"), tag="")       (needs litert_torch + quantizer)
      -> {form: {"encoder": path, "adapter": path}}

1:1 with LiquidAI/LFM3-VL-2B modeling_lfm3_vl.py @ f83d2b0a (Lfm3VlVisionModel L460-512, projector L515-537):
  preprocessing  the processor's rescale + normalize and patchify are in the graph. The runtime hands the encoder
      float(u8) * (1/255) (LiteRT-LM v0.18.1 stb_image_preprocessor.cc L203-210); the graph takes the exact u8
      back, u = floor(x * 255 + 0.5) (|x * 255 - u8| <= 1.5e-5 for every u8), and computes (u - 127.5) / 127.5 =
      the HF processor's fused rescale + normalize, its pixel_values bit for bit in torch (the converter turns the
      division into MUL by fl32(1 / 127.5): <= 6e-8 from it; FLOOR builds GPU-only on Metal).
      Opts(input_norm="unit") keeps
      the first form (x - 0.5) * 2: subtracting 0.5 from u8 / 255 cancels its leading bits (relative error 1.5e-5
      at u8 = 128), which failed a float64 test on flat grey images. The
      patchifier Linear(3*2*16*16 -> 1024) reads a temporal pair of identical frames, so it equals
      Conv2d(3 -> 1024, k16, s16) with weight W.reshape(1024, 3, 2, 16, 16).sum(2) on the NCHW image.
  token order    HF orders patches by 2x2 merge block (block raster, then (mh, mw) inside); the conv layer
      converts to raster, convolves and converts back (L330-340); attention has no mask inside one image and
      the RoPE position of every token is its own (h, w) (transformers vision_utils.get_vision_position_ids),
      MLP and norms are per token -> the whole tower runs in RASTER order with constant (h, w) cos / sin
      tables; the projector's 2x2 merge = 4 strided slices (r0c0, r0c1, r1c0, r1c1) concatenated on channels
      = HF's reshape(-1, 4 * 1024) of merge-block order, and its output is in HF's block-raster order.
  conv layer     in_proj 1024 -> 3 * 1024 (B, C, x) -> B * x -> depthwise Conv2d 3x3 pad 1 on the [G/16, G/16]
      grid -> out_proj(C * conv)
  attention      q 16 / kv 8 heads x 64, head RMSNorm on q and k, axial RoPE (h on dims 0-15 and 32-47, w on
      16-31 and 48-63, theta 1e6), softmax(q k^T / 8) v. GQA without repeat_kv / expand (BROADCAST_TO, which
      the Metal delegate refuses): q heads 2j and 2j+1 share kv head j, so q [1, 16, n, 64] is reshaped to
      [1, 8, 2n, 64] (the two heads of a group stacked on the query axis) and attends to k, v [1, 8, n, 64].
      The scale 64^-0.5 = 2^-3 is folded into the q RMSNorm weight (a power of two: exact).
  projector      LayerNorm(1024, eps 1e-6) -> 2x2 merge -> Linear 4096 -> 4096 -> GELU (erf; ACT2FN["gelu"])
      -> Linear 4096 -> 2048.
Every activation keeps a leading batch axis; the elementwise ops run on rank >= 3 tensors.

Opts: `input_norm` = "u8" (default, above) or "unit" (the first form); fp16 GPU safety, exact in real
arithmetic: `rms_s` = {norm key: S}, a power-of-two pre-scale 1/S of a RMSNorm input with eps / S^2
(x / S * rsqrt(mean((x / S)^2) + eps / S^2) = x * rsqrt(mean(x^2) + eps)); `calib` (dict) records the absmax of
every norm input during a forward.
fp16 precision: the GPU at fp16 precision breaks the default graph at its first op. The
converter folds the input step's constants into the patch conv (weights W / 127.5, the mean moved into the bias), so
the conv sums W / 127.5 * u8 and subtracts a bias of the same size: a flat grey patch (u8 128) keeps 1 / 256 of the
terms' size, below fp16's rounding of the sum (Metal fp16 vs fp32 token cos 0.36 at the patch conv).
`input_norm="u8c"` centres before the FLOOR, w = floor(x * 255 - 127.5) = u8 - 128 (exact for every u8), then
(w + 0.5) / 127.5 = the same pixel values bit for bit in torch fp32: the folded bias is then 0.5 * sum(W) / 127.5 and
the conv sums the centred integers, no cancellation. `ln_s` = S, the same power-of-two pre-scale for the projector
LayerNorm (x / S with eps / S^2: exact). `fp16_safe_scales(sites)` picks S per norm from measured ranges:
the smallest power of two that keeps the largest row sum of x^2 and the largest x^2 under
65504 / headroom, so a norm whose inputs are tiny (here 93-99.8 % of x^2 below fp16's smallest normal in layers 0-3,
43-92 % in layers 4-7) is scaled up into fp16's normal range and a norm that would overflow (the projector LayerNorm's
row sum of squares, 96,000) is scaled down (not used by the published files: they keep the default graph).
"""
import json
import math
import os
from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

TOWER_PREFIX = "model.vision_tower."
PROJECTOR_PREFIX = "model.multi_modal_projector."


class Opts:
  def __init__(self, rms_s=None, input_norm="u8", ln_s=1.0):
    assert input_norm in ("u8", "unit", "u8c"), input_norm
    self.rms_s = dict(rms_s or {})
    self.input_norm = input_norm
    self.ln_s = float(ln_s)
    for key, v in [*self.rms_s.items(), ("ln_s", self.ln_s)]:
      assert v > 0 and math.log2(v).is_integer(), (key, v)     # powers of two only: exact in float rounding

  def s(self, key):
    return float(self.rms_s.get(key, 1.0))


def fp16_safe_scales(sites, headroom=4.0, fp16_max=65504.0):
  """-> ({RMSNorm key: S}, S of the projector LayerNorm) from measured ranges ({key: {"max_row_sum_sq", "max_x_sq"}},
  one entry per norm input; LayerNorm values centred). S = 2^e with the smallest integer e (negative = scale
  up) such that (row sum of x^2) / S^2 and x^2 / S^2 both stay <= fp16_max / headroom over the measured set."""
  lim = fp16_max / headroom
  rms, ln = {}, 1.0
  for key, v in sites.items():
    if "max_row_sum_sq" not in v:
      continue
    e = max(math.ceil(0.5 * math.log2(v["max_row_sum_sq"] / lim)), math.ceil(0.5 * math.log2(v["max_x_sq"] / lim)))
    if key == "projector.layer_norm":
      ln = 2.0 ** e
    else:
      rms[key] = 2.0 ** e
  return rms, ln


def _note(calib, key, x):
  if calib is not None:
    calib[key] = max(calib.get(key, 0.0), float(x.detach().abs().max()))


class RMSNorm(nn.Module):
  def __init__(self, dim, eps):
    super().__init__()
    self.weight = nn.Parameter(torch.ones(dim))
    self.eps = eps

  def forward(self, x, s=1.0):
    if s != 1.0:
      x = x * (1.0 / s)
      eps = self.eps / (s * s)
    else:
      eps = self.eps
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * self.weight


class MLP(nn.Module):
  def __init__(self, d, inter):
    super().__init__()
    self.w1 = nn.Linear(d, inter, bias=False)
    self.w3 = nn.Linear(d, inter, bias=False)
    self.w2 = nn.Linear(inter, d, bias=False)

  def forward(self, x):
    return self.w2(F.silu(self.w1(x)) * self.w3(x))


class ConvMixer(nn.Module):
  """Gated depthwise 2D conv on the raster patch grid."""

  def __init__(self, d, k, gh, gw):
    super().__init__()
    self.d, self.gh, self.gw = d, gh, gw
    self.in_proj = nn.Linear(d, 3 * d, bias=False)
    self.spatial_conv = nn.Conv2d(d, d, kernel_size=k, padding=k // 2, groups=d, bias=False)
    self.out_proj = nn.Linear(d, d, bias=False)

  def forward(self, x):                                  # x [1, n, d] raster
    b, c, xx = self.in_proj(x).chunk(3, dim=-1)
    g = (b * xx).reshape(1, self.gh, self.gw, self.d).permute(0, 3, 1, 2)
    y = self.spatial_conv(g).permute(0, 2, 3, 1).reshape(1, self.gh * self.gw, self.d)
    return self.out_proj(c * y)


class Attention(nn.Module):
  def __init__(self, d, heads, kv_heads, eps, n):
    super().__init__()
    self.h, self.kvh, self.hd, self.n = heads, kv_heads, d // heads, n
    self.rep = heads // kv_heads
    self.q_proj = nn.Linear(d, heads * self.hd, bias=False)
    self.k_proj = nn.Linear(d, kv_heads * self.hd, bias=False)
    self.v_proj = nn.Linear(d, kv_heads * self.hd, bias=False)
    self.out_proj = nn.Linear(heads * self.hd, d, bias=False)
    self.q_layernorm = RMSNorm(self.hd, eps)
    self.k_layernorm = RMSNorm(self.hd, eps)
    self.register_buffer("cos", torch.zeros(1, n, 1, self.hd), persistent=False)
    self.register_buffer("sin", torch.zeros(1, n, 1, self.hd), persistent=False)

  def _rope(self, x):                                    # x [1, n, heads, hd]
    half = self.hd // 2
    rot = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    return x * self.cos + rot * self.sin

  def forward(self, x, opts=None, key="", calib=None):
    n, hd = self.n, self.hd
    q = self.q_proj(x).reshape(1, n, self.h, hd)
    k = self.k_proj(x).reshape(1, n, self.kvh, hd)
    v = self.v_proj(x).reshape(1, n, self.kvh, hd)
    _note(calib, key + ".q_layernorm", q)
    _note(calib, key + ".k_layernorm", k)
    s = (lambda kk: opts.s(kk)) if opts is not None else (lambda kk: 1.0)
    q = self._rope(self.q_layernorm(q, s(key + ".q_layernorm")))     # q norm weight carries the 1/8 scale
    k = self._rope(self.k_layernorm(k, s(key + ".k_layernorm")))
    q = q.permute(0, 2, 1, 3).reshape(1, self.kvh, self.rep * n, hd)   # [1, 8, 2n, 64]
    k = k.permute(0, 2, 1, 3)                                         # [1, 8, n, 64]
    v = v.permute(0, 2, 1, 3)
    a = torch.matmul(q, k.transpose(-2, -1)).softmax(dim=-1)          # [1, 8, 2n, n]
    o = torch.matmul(a, v).reshape(1, self.h, n, hd)                  # [1, 16, n, 64]
    return self.out_proj(o.permute(0, 2, 1, 3).reshape(1, n, self.h * hd))


class Layer(nn.Module):
  def __init__(self, cfg, kind, gh, gw):
    super().__init__()
    d = cfg["hidden_size"]
    self.kind = kind
    if kind == "full_attention":
      self.self_attn = Attention(d, cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["norm_eps"], gh * gw)
    else:
      self.conv = ConvMixer(d, cfg["conv_kernel_size"], gh, gw)
    self.feed_forward = MLP(d, cfg["intermediate_size"])
    self.operator_norm = RMSNorm(d, cfg["norm_eps"])
    self.ffn_norm = RMSNorm(d, cfg["norm_eps"])

  def forward(self, x, opts, key, calib=None):
    _note(calib, key + ".operator_norm", x)
    h = self.operator_norm(x, opts.s(key + ".operator_norm"))
    if self.kind == "full_attention":
      x = x + self.self_attn(h, opts, key + ".self_attn", calib)
    else:
      x = x + self.conv(h)
    _note(calib, key + ".ffn_norm", x)
    return x + self.feed_forward(self.ffn_norm(x, opts.s(key + ".ffn_norm")))


class VisionEncoder(nn.Module):
  def __init__(self, cfg, g, opts=None):
    super().__init__()
    p = cfg["patch_size"]
    assert g % (p * cfg["spatial_merge_size"]) == 0, (g, p)
    self.g, self.gh, self.gw = g, g // p, g // p
    self.n = self.gh * self.gw
    self.d = cfg["hidden_size"]
    self.opts = opts or Opts()
    self.patch_conv = nn.Conv2d(cfg["in_channels"], self.d, kernel_size=p, stride=p, bias=False)
    self.embedding_norm = RMSNorm(self.d, cfg["norm_eps"])
    self.layers = nn.ModuleList([Layer(cfg, t, self.gh, self.gw) for t in cfg["layer_types"]])

  def embed(self, images):                               # [1, G, G, 3] in [0, 1] -> [1, n, d] raster
    if self.opts.input_norm == "u8":
      u = torch.floor(images * 255.0 + 0.5)              # the runtime's float(u8) * (1/255) -> the exact u8
      x = ((u - 127.5) / 127.5).permute(0, 3, 1, 2)      # HF's fused rescale + normalize, NCHW
    elif self.opts.input_norm == "u8c":
      w = torch.floor(images * 255.0 - 127.5)            # u8 - 128, centred before the FLOOR (the converter cannot fold it)
      x = ((w + 0.5) / 127.5).permute(0, 3, 1, 2)        # (u8 - 127.5) / 127.5, the same fp32 values as "u8"
    else:
      x = ((images - 0.5) * 2.0).permute(0, 3, 1, 2)     # (x - 0.5) / 0.5 on u8 / 255 (the first form)
    return self.patch_conv(x).permute(0, 2, 3, 1).reshape(1, self.n, self.d)

  def forward(self, images, calib=None):
    x = self.embed(images)
    _note(calib, "embedding_norm", x)
    x = self.embedding_norm(x, self.opts.s("embedding_norm"))
    for i, layer in enumerate(self.layers):
      x = layer(x, self.opts, f"layers.{i}", calib)
    return x


class VisionAdapter(nn.Module):
  def __init__(self, cfg, g, opts=None):
    super().__init__()
    d = cfg["hidden_size"]
    self.d, self.gh = d, g // cfg["patch_size"]
    self.m = self.gh // 2
    self.ln_s = opts.ln_s if opts is not None else 1.0
    self.layer_norm = nn.LayerNorm(d, eps=cfg["projector_layer_norm_eps"], bias=cfg["projector_bias"])
    self.linear_1 = nn.Linear(4 * d, cfg["projector_hidden_size"], bias=cfg["projector_bias"])
    self.linear_2 = nn.Linear(cfg["projector_hidden_size"], cfg["text_hidden_size"], bias=cfg["projector_bias"])

  def forward(self, x, calib=None):                      # [1, n, d] raster -> [1, (gh/2)^2, text_hidden]
    _note(calib, "projector.layer_norm", x)
    if self.ln_s != 1.0:                                 # (x / S - mean / S) * rsqrt(var / S^2 + eps / S^2): exact
      ln, s = self.layer_norm, self.ln_s
      x = F.layer_norm(x * (1.0 / s), ln.normalized_shape, ln.weight, ln.bias, ln.eps / (s * s))
    else:
      x = self.layer_norm(x)
    x = x.reshape(1, self.gh, self.gh, self.d)
    m = torch.cat([x[:, 0::2, 0::2, :], x[:, 0::2, 1::2, :], x[:, 1::2, 0::2, :], x[:, 1::2, 1::2, :]], dim=-1)
    m = m.reshape(1, self.m * self.m, 4 * self.d)
    return self.linear_2(F.gelu(self.linear_1(m)))


# ----------------------------------------------------------------------------------------------- loading
def read_config(config):
  c = json.load(open(config)) if not isinstance(config, Mapping) else dict(config)
  v = dict(c["vision_config"])
  assert v["rope_parameters"]["rope_type"] == "axial", v["rope_parameters"]
  assert c.get("projector_hidden_act", "gelu") == "gelu", c.get("projector_hidden_act")
  assert c.get("downsample_factor", 2) == v["spatial_merge_size"] == 2, (c.get("downsample_factor"), v["spatial_merge_size"])
  assert v["temporal_patch_size"] == 2 and v["hidden_size"] % v["num_attention_heads"] == 0
  v["rope_theta"] = float(v["rope_parameters"]["rope_theta"])
  v["projector_hidden_size"] = c["projector_hidden_size"]
  v["projector_layer_norm_eps"] = c["projector_layer_norm_eps"]
  v["projector_bias"] = bool(c["projector_bias"])
  v["text_hidden_size"] = c["text_config"]["hidden_size"]
  return v


def rope_tables(cfg, gh, gw):
  """cos / sin [1, gh*gw, 1, head_dim] in raster order, computed with the exact ops of
  Lfm3VlVisionRotaryEmbedding (fp32, L106-134) on position_ids (h, w)."""
  hd = cfg["hidden_size"] // cfg["num_attention_heads"]
  spatial_dim = hd // 2
  inv_freq = 1.0 / (cfg["rope_theta"] ** (torch.arange(0, spatial_dim, 2, dtype=torch.float) / spatial_dim))
  hh, ww = torch.meshgrid(torch.arange(gh), torch.arange(gw), indexing="ij")
  pos = torch.stack([hh.flatten(), ww.flatten()], dim=-1)            # [n, 2] raster
  freqs = pos[..., None].float() * inv_freq.float()                   # [n, 2, hd/4]
  out = []
  for f in (freqs.cos(), freqs.sin()):
    fhw = torch.cat([f[:, 0], f[:, 1]], dim=-1)
    out.append(torch.cat([fhw, fhw], dim=-1).reshape(1, gh * gw, 1, hd))
  return out


def _open_weights(weights, prefixes):
  if isinstance(weights, Mapping):
    return {k: v for k, v in weights.items() if k.startswith(prefixes)}
  from safetensors import safe_open
  path = os.fspath(weights)
  files = [path]
  if os.path.isdir(path):
    idx = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(idx):
      wm = json.load(open(idx))["weight_map"]
      files = sorted({os.path.join(path, f) for k, f in wm.items() if k.startswith(prefixes)})
    else:
      files = [os.path.join(path, "model.safetensors")]
  out = {}
  for f in files:
    with safe_open(f, framework="pt") as h:
      for k in h.keys():
        if k.startswith(prefixes):
          out[k] = h.get_tensor(k)
  return out


def build_tower(weights, config, g, tower_prefix=TOWER_PREFIX, projector_prefix=PROJECTOR_PREFIX,
                dtype=torch.float32, opts=None):
  """-> (VisionEncoder, VisionAdapter, info). Strict: every tensor under the two prefixes is used once."""
  cfg = read_config(config)
  sd = _open_weights(weights, (tower_prefix, projector_prefix))
  enc, adp = VisionEncoder(cfg, g, opts), VisionAdapter(cfg, g, opts)
  hd = cfg["hidden_size"] // cfg["num_attention_heads"]
  qscale = hd ** -0.5
  assert math.log2(1.0 / qscale).is_integer(), qscale
  e_sd, a_sd = {}, {}
  for k, v in sd.items():
    v = v.to(torch.float32)
    if k.startswith(tower_prefix):
      kk = k[len(tower_prefix):]
      if kk == "patchifier.weight":
        c, t, p = cfg["in_channels"], cfg["temporal_patch_size"], cfg["patch_size"]
        e_sd["patch_conv.weight"] = v.reshape(v.shape[0], c, t, p, p).sum(2)
      elif kk.endswith("self_attn.q_layernorm.weight"):
        e_sd[kk] = v * qscale
      else:
        e_sd[kk] = v
    else:
      a_sd[k[len(projector_prefix):]] = v
  enc.load_state_dict(e_sd, strict=True)
  adp.load_state_dict(a_sd, strict=True)
  cos, sin = rope_tables(cfg, enc.gh, enc.gw)
  for layer in enc.layers:
    if layer.kind == "full_attention":
      layer.self_attn.cos.copy_(cos)
      layer.self_attn.sin.copy_(sin)
  enc, adp = enc.to(dtype).eval(), adp.to(dtype).eval()
  info = {"g": g, "grid": [enc.gh, enc.gw], "n_patches": enc.n, "n_tokens": adp.m * adp.m,
          "n_src_tensors": len(sd), "n_enc_keys": len(e_sd), "n_adp_keys": len(a_sd),
          "enc_params": sum(p.numel() for p in enc.parameters()),
          "adp_params": sum(p.numel() for p in adp.parameters()),
          "layer_types": cfg["layer_types"], "tower_prefix": tower_prefix, "projector_prefix": projector_prefix}
  return enc, adp, info


def raster_to_merge(x, gh, gw):
  """[..., gh*gw, C] raster -> HF merge-block order (block raster, then (mh, mw))."""
  c = x.shape[-1]
  y = x.reshape(-1, gh // 2, 2, gw // 2, 2, c).permute(0, 1, 3, 2, 4, 5)
  return y.reshape(*x.shape[:-2], gh * gw, c)


def merge_to_raster(x, gh, gw):
  c = x.shape[-1]
  y = x.reshape(-1, gh // 2, gw // 2, 2, 2, c).permute(0, 1, 3, 2, 4, 5)
  return y.reshape(*x.shape[:-2], gh * gw, c)


# ----------------------------------------------------------------------------------------------- export
def quantize_tflite(src, dst, form):
  """int8 = ai_edge_quantizer dynamic_wi8_afp32 (int8 weights, run-time int8 activations on CPU);
  fp16 = float_casting of FULLY_CONNECTED + CONV_2D weights."""
  import copy

  from ai_edge_quantizer import quantizer
  import ai_edge_quantizer.recipe as r
  if form == "int8":
    q = quantizer.Quantizer(src, r.dynamic_wi8_afp32())
  elif form == "fp16":
    base = copy.deepcopy(r.dynamic_wi8_afp32()[0])
    base["algorithm_key"] = "float_casting"
    base["op_config"]["weight_tensor_config"] = {"num_bits": 16, "symmetric": True, "granularity": "TENSORWISE",
                                                 "dtype": "FLOAT"}
    base["op_config"]["compute_precision"] = "FLOAT"
    base["op_config"]["explicit_dequantize"] = False
    recipes = []
    for op in ("FULLY_CONNECTED", "CONV_2D"):
      rr = copy.deepcopy(base)
      rr["operation"] = op
      recipes.append(rr)
    q = quantizer.Quantizer(src, recipes)
  else:
    raise ValueError(form)
  q.quantize().export_model(dst)


def export_tower(enc, adp, out_dir, forms=("fp32", "fp16", "int8"), tag=""):
  """Convert both modules with litert_torch (fp32 graph) and derive the quantized forms from it.
  File names: vision_{encoder,adapter}{_tag}{"" | _fp16 | _int8}.tflite."""
  import litert_torch
  os.makedirs(out_dir, exist_ok=True)
  t = f"_{tag}" if tag else ""
  sample = torch.rand(1, enc.g, enc.g, 3, dtype=torch.float32)
  with torch.no_grad():
    feat = enc(sample)
  paths = {"fp32": {"encoder": os.path.join(out_dir, f"vision_encoder{t}.tflite"),
                    "adapter": os.path.join(out_dir, f"vision_adapter{t}.tflite")}}
  for key, mod, x in (("encoder", enc, sample), ("adapter", adp, feat)):
    if not os.path.exists(paths["fp32"][key]):
      litert_torch.convert(mod, (x,)).export(paths["fp32"][key])
  for form in forms:
    if form == "fp32":
      continue
    paths[form] = {}
    for key in ("encoder", "adapter"):
      dst = os.path.join(out_dir, f"vision_{key}{t}_{form}.tflite")
      if not os.path.exists(dst):
        quantize_tflite(paths["fp32"][key], dst, form)
      paths[form][key] = dst
  return {f: paths[f] for f in forms}
