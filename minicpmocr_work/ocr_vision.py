"""Pure-torch re-author of the MiniCPM-OCR-1.0 vision path ("4x" mode) for ONE static GxG image
(no transformers import), split for the LiteRT-LM fast_vlm contract:

  VisionEncoder  images NHWC [1,G,G,3] in [0,1] -> [1,(G/14)^2,1152]   (post_layernorm output, raster order)
  VisionAdapter  [1,(G/14)^2,1152] -> [1,(G/28)^2,1024]                (merger.mlp[0]: 2x2 merge + DownsampleMLP;
                                                                         raster order of the merged grid = the row
                                                                         order of HF get_image_features(...).pooler_output[0])
  G = 448: 32x32 patches -> 1024 encoder tokens -> 256 soft tokens.

1:1 with transformers main @2df500f3 models/minicpm_ocr/modeling_minicpm_ocr.py with downsample_mode "4x"
(the checkpoint's default): MiniCPMOcrModel.get_image_features sets use_vit_merger=False, so the vision tower is
  embeddings  Conv2d(3->1152, k14, s14) on the GxG image instead of the processor's [1,3,14,N*14] strip (the same
      patches in raster order), flattened NHWC-style (permute(0,2,3,1).reshape: the converter lowers
      flatten(2).transpose to a TRANSPOSE->RESHAPE that Metal miscomputes, qwen35vl precedent) + the
      position_embedding rows get_vision_nearest_position_ids([[g,g]], 70) gathered in Python into a constant
      buffer (no GATHER in the graph).
  normalization (x-0.5)/0.5 (preprocessor_config mean / std 0.5) and NHWC->NCHW in the graph.
  27 x MiniCPMOcrVisionEncoderLayer: LN1 -> attention (q/k/v/out 1152x1152 + bias, 16 heads x 72, scale 72^-0.5,
      softmax fp32; one image = one cu_seqlens chunk = full attention, no mask) -> residual -> LN2 -> fc1 4304 ->
      gelu(tanh) -> fc2 -> residual.
  post_layernorm                                                                          = VisionEncoder
  MiniCPMOcrMerger (merger_times 1): view(g/2,2,g/2,2,C).permute(0,2,1,3,4) = four strided slices
      [0::2,0::2] [0::2,1::2] [1::2,0::2] [1::2,1::2] concatenated on channels -> pre_norm 4608 ->
      linear_1 4608 -> GELU(erf, nn.GELU()) -> linear_2 1024                              = VisionAdapter
The vit_merger tensors (16) are in the checkpoint but unused in 4x mode; build() refuses any other unused key.

Every activation keeps a leading batch dim (rank >= 3). LN->FC barrier (clamp +-65504, neutral) and an fp16-safe
LayerNorm with calibrated power-of-two pre-scales S (exact in real arithmetic) are switchable through Opts
(the MiniCPM-V-4.7-1B tower pattern: minicpmv47_work/mv47_vision.py and docs/vision-tower-reauthor.md in hf-to-litertlm).
"""
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

PATCH = 14
HID = 1152
HEADS = 16
HDIM = HID // HEADS            # 72
INTER = 4304
MERGED = HID * 4               # 4608
LLM_HID = 1024
EPS = 1e-6
POS_SIDE = 70                  # image_size 980 / patch 14
NLAYERS = 27
MEAN = STD = 0.5

PREFIX_VT = "model.vision_tower."
PREFIX_MG = "model.merger.mlp.0."
UNUSED_4X = "model.vision_tower.vit_merger."


class Opts:
  """barrier: LN->FC clamp; fp16safe: use ln_s pre-scales (S <= 1 means plain layer_norm)."""

  def __init__(self, barrier=True, fp16safe=True, ln_s=None):
    self.barrier = barrier
    self.fp16safe = fp16safe
    self.ln_s = dict(ln_s or {})

  def s(self, key):
    return float(self.ln_s.get(key, 1.0))


def nearest_pos_ids(h, w, side=POS_SIDE):
  """transformers vision_utils.get_vision_nearest_position_ids for one (h, w) grid."""
  boundaries = torch.arange(1 / side, 1.0, 1 / side)
  bh = torch.bucketize(torch.arange(h) / h, boundaries, right=True)
  bw = torch.bucketize(torch.arange(w) / w, boundaries, right=True)
  return (bh[:, None] * side + bw).flatten()


def merge2x2(f, side):
  """[1, side*side, C] raster -> [1, (side/2)^2, 4C] (r0c0, r0c1, r1c0, r1c1) per 2x2 block."""
  g = f.reshape(1, side, side, f.shape[-1])
  m = torch.cat([g[:, 0::2, 0::2, :], g[:, 0::2, 1::2, :],
                 g[:, 1::2, 0::2, :], g[:, 1::2, 1::2, :]], dim=-1)
  return m.reshape(1, (side // 2) * (side // 2), 4 * f.shape[-1])


def note(calib, key, x):
  if calib is not None:
    calib[key] = max(calib.get(key, 0.0), float(x.detach().abs().max()))


def tap(taps, key, x):
  if taps is not None:
    taps[key] = x.detach().clone()


def layer_norm(x, ln, S, fp16safe):
  if not fp16safe or S <= 1.0:
    return F.layer_norm(x, (x.shape[-1],), ln.weight, ln.bias, ln.eps)
  xs = x * (1.0 / S)                                     # power of two: exact
  d = xs - xs.mean(-1, keepdim=True)
  var = (d * d).mean(-1, keepdim=True)
  y = d * torch.rsqrt(var + ln.eps / (S * S))
  return y * ln.weight + ln.bias


class Attention(nn.Module):
  def __init__(self):
    super().__init__()
    self.q_proj = nn.Linear(HID, HID)
    self.k_proj = nn.Linear(HID, HID)
    self.v_proj = nn.Linear(HID, HID)
    self.out_proj = nn.Linear(HID, HID)
    self.scale = HDIM ** -0.5

  def forward(self, x):                                  # x [1,L,C]
    L = x.shape[1]
    q = self.q_proj(x).reshape(1, L, HEADS, HDIM).permute(0, 2, 1, 3)   # [1,H,L,d]
    k = self.k_proj(x).reshape(1, L, HEADS, HDIM).permute(0, 2, 1, 3)
    v = self.v_proj(x).reshape(1, L, HEADS, HDIM).permute(0, 2, 1, 3)
    a = ((q * self.scale) @ k.transpose(-2, -1)).softmax(dim=-1)        # [1,H,L,L]
    o = (a @ v).permute(0, 2, 1, 3).reshape(1, L, HID)
    return self.out_proj(o)


class MLP(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(HID, INTER)
    self.fc2 = nn.Linear(INTER, HID)


class EncoderLayer(nn.Module):
  def __init__(self):
    super().__init__()
    self.layer_norm1 = nn.LayerNorm(HID, eps=EPS)
    self.self_attn = Attention()
    self.layer_norm2 = nn.LayerNorm(HID, eps=EPS)
    self.mlp = MLP()


class VisionEncoder(nn.Module):
  def __init__(self, opts, g=448):
    super().__init__()
    assert g % (2 * PATCH) == 0, g
    self.opts, self.g, self.grid = opts, g, g // PATCH
    self.n_patch = self.grid * self.grid
    self.patch_embedding = nn.Conv2d(3, HID, PATCH, stride=PATCH, bias=True)
    self.register_buffer("pos_embed", torch.zeros(1, self.n_patch, HID))
    self.layers = nn.ModuleList([EncoderLayer() for _ in range(NLAYERS)])
    self.post_layernorm = nn.LayerNorm(HID, eps=EPS)

  def _barrier(self, x):
    return torch.clamp(x, -65504.0, 65504.0) if self.opts.barrier else x

  def _ln(self, key, x, ln, calib):
    note(calib, key, x)
    return layer_norm(x, ln, self.opts.s(key), self.opts.fp16safe)

  def forward(self, images, calib=None, taps=None):     # [1,G,G,3] in [0,1]
    x = ((images - MEAN) / STD).permute(0, 3, 1, 2)      # NCHW, normalized
    p = self.patch_embedding(x)                          # [1,1152,g,g]
    h = p.permute(0, 2, 3, 1).reshape(1, self.n_patch, HID) + self.pos_embed
    tap(taps, "embeddings", h)
    for i, blk in enumerate(self.layers):
      n1 = self._ln(f"layers.{i}.layer_norm1", h, blk.layer_norm1, calib)
      h = h + blk.self_attn(self._barrier(n1))
      n2 = self._ln(f"layers.{i}.layer_norm2", h, blk.layer_norm2, calib)
      h = h + blk.mlp.fc2(F.gelu(blk.mlp.fc1(self._barrier(n2)), approximate="tanh"))
      tap(taps, f"layer{i:02d}", h)
    return self._ln("post_layernorm", h, self.post_layernorm, calib)   # [1,g*g,1152]


class VisionAdapter(nn.Module):
  def __init__(self, opts, g=448):
    super().__init__()
    self.opts, self.grid = opts, g // PATCH
    self.pre_norm = nn.LayerNorm(MERGED, eps=EPS)
    self.linear_1 = nn.Linear(MERGED, MERGED)
    self.linear_2 = nn.Linear(MERGED, LLM_HID)

  def forward(self, feats, calib=None):                  # [1,g*g,1152]
    m = merge2x2(feats, self.grid)                       # [1,(g/2)^2,4608]
    note(calib, "merger.pre_norm", m)
    y = layer_norm(m, self.pre_norm, self.opts.s("merger.pre_norm"), self.opts.fp16safe)
    if self.opts.barrier:
      y = torch.clamp(y, -65504.0, 65504.0)
    return self.linear_2(F.gelu(self.linear_1(y)))       # erf GELU -> [1,(g/2)^2,1024]


def load_vision_state(src_dir):
  from safetensors import safe_open
  out = {}
  with safe_open(os.path.join(src_dir, "model.safetensors"), framework="pt") as f:
    for k in f.keys():
      if k.startswith(PREFIX_VT) or k.startswith(PREFIX_MG):
        out[k] = f.get_tensor(k).to(torch.float32)
  return out


def build(src_dir, opts, g=448):
  """-> (VisionEncoder, VisionAdapter, info). Strict: every vision / merger tensor is loaded once, except the
  vit_merger ones (unused in 4x mode), which are counted and must be exactly 16."""
  sd = load_vision_state(src_dir)
  grid = g // PATCH
  pos_ids = nearest_pos_ids(grid, grid)
  enc_sd, adp_sd, unused = {}, {}, []
  for k, v in sd.items():
    if k.startswith(UNUSED_4X):
      unused.append(k)
    elif k.startswith(PREFIX_VT):
      kk = k[len(PREFIX_VT):]
      if kk == "embeddings.position_embedding.weight":
        assert tuple(v.shape) == (POS_SIDE * POS_SIDE, HID), v.shape
        enc_sd["pos_embed"] = v[pos_ids].reshape(1, grid * grid, HID).contiguous()
      elif kk.startswith("embeddings.patch_embedding."):
        enc_sd["patch_embedding." + kk.rsplit(".", 1)[1]] = v
      elif kk.startswith("encoder.layers."):
        enc_sd[kk[len("encoder."):]] = v
      elif kk.startswith("post_layernorm."):
        enc_sd[kk] = v
      else:
        raise KeyError(f"unmapped vision key {k}")
    else:
      adp_sd[k[len(PREFIX_MG):]] = v
  assert len(unused) == 16, len(unused)
  enc, adp = VisionEncoder(opts, g), VisionAdapter(opts, g)
  enc.load_state_dict(enc_sd, strict=True)
  adp.load_state_dict(adp_sd, strict=True)
  info = {"g": g, "grid": grid, "n_src_tensors": len(sd), "n_enc_keys": len(enc_sd), "n_adp_keys": len(adp_sd),
          "n_unused_vit_merger": len(unused),
          "enc_params": sum(p.numel() for p in enc.parameters()),
          "adp_params": sum(p.numel() for p in adp.parameters())}
  return enc.eval(), adp.eval(), info


def strip_from_image(img01_nhwc):
  """The processor's reshape_by_patch strip [1,3,14,N*14] built from a [1,G,G,3] [0,1] image."""
  x = ((img01_nhwc - MEAN) / STD).permute(0, 3, 1, 2)[0]          # [3,G,G]
  p = F.unfold(x.unsqueeze(0), (PATCH, PATCH), stride=(PATCH, PATCH))
  p = p.reshape(3, PATCH, PATCH, -1).permute(0, 1, 3, 2).reshape(3, PATCH, -1)
  return p.unsqueeze(0)


def calib_S(absmax):
  """Power-of-two pre-scale: absmax / S <= 8 (qwen35vl_work recipe)."""
  return float(2 ** max(0, int(math.ceil(math.log2(max(absmax, 1e-6) / 8.0)))))
