"""Pure-torch re-author of the MiniCPM-V-4.7 vision path for ONE static 448x448 image
(no transformers import), split for the LiteRT-LM fast_vlm contract:

  VisionEncoder  images NHWC [1,448,448,3] in [0,1] -> [1,256,1152]
                 (post_layernorm output; 16x16 tokens after the layer-6 window merger, raster order)
  VisionAdapter  [1,256,1152] -> [1,64,1024]
                 (merger.mlp[0]: 2x2 merge + DownsampleMLP; 8x8 tokens, raster order
                  = the row order of HF get_image_features(...).pooler_output[0])

1:1 with transformers 5.19.0 models/minicpmv4_7/modeling_minicpmv4_7.py:
  MiniCPMV4_7VisionEmbeddings  Conv2d(3->1152, k14, s14) on the 448x448 image instead of the
      processor's [1,3,14,N*14] strip (same patches; the strip is reshape_by_patch of the raster
      patch grid), flattened NHWC-style (permute(0,2,3,1).reshape -- NOT flatten(2).transpose,
      which the converter lowers to a TRANSPOSE->RESHAPE[C,N] that Metal miscomputes; qwen35vl
      precedent) + position_embedding rows get_vision_nearest_position_ids([[32,32]], 70),
      gathered in python into a constant buffer (no GATHER in the graph).
  normalization (x-0.5)/0.5 (preprocessor_config mean/std 0.5) and NHWC->NCHW are in the graph.
  MiniCPMV4_7VisionEncoderLayer x27  LN1 -> attn (q/k/v/out 1152x1152 + bias, 16 heads x 72,
      scale 72^-0.5, softmax fp32) -> residual -> LN2 -> fc1 4304 -> gelu(tanh) -> fc2 -> residual.
      cu_seqlens holds one image, so every layer is full attention (1024 tokens before the
      merger, 256 after).
  MiniCPMV4_7ViTWindowAttentionMerger, after layer 6 (insert_layer_id):
      layer_norm1 -> self_attn restricted to each 2x2 window. HF gathers tokens into window order
      (get_vision_window_index) and attends per window (cu_seqlens of 4); here the tokens stay
      in RASTER order and a constant block-diagonal additive mask (0 inside the window, MASK_NEG
      elsewhere) does the same (attention inside a window is permutation-equivariant) -> residual
      -> 2x2 merge = 4 strided slices [0::2,0::2] [0::2,1::2] [1::2,0::2] [1::2,1::2] of the
      [1,32,32,C] grid concatenated on channels -> [1,256,4608] (= HF window order: 16x16 window
      grid row-major, (r0c0, r0c1, r1c0, r1c1) inside) -> pre_norm 4608 -> linear_1 17216 ->
      gelu(tanh) -> linear_2 1152 -> + mean of the 4 patches (patch_residual).
  layers 7..26 on 256 tokens -> post_layernorm                                  = VisionEncoder
  MiniCPMV4_7Merger / DownsampleMLP (merger_times 1): the same 4-slice 2x2 merge 16x16 -> 8x8
      (= HF view(8,2,8,2,C).permute(0,2,1,3,4)) -> pre_norm 4608 -> linear_1 4608 ->
      GELU(erf, nn.GELU()) -> linear_2 1024                                     = VisionAdapter

Every activation keeps a leading batch dim (rank >= 3; Metal computes rank-2 elementwise x
rank-2 const silently wrong). LN->FC fold barrier (clamp +-65504, semantically neutral) and an
fp16-safe LayerNorm with calibrated power-of-two pre-scales S (exact in real arithmetic;
qwen35vl_work/convert_qwen35_vision.py recipe) are switchable through Opts.

Key map (safetensors -> module; 459 tensors, bf16 read as fp32):
  model.vision_tower.embeddings.patch_embedding.{weight,bias}          -> enc.patch_embedding.*
  model.vision_tower.embeddings.position_embedding.weight [4900,1152]  -> enc.pos_embed [1,1024,1152]
                                                                          (rows nearest_pos_ids(32,32))
  model.vision_tower.encoder.layers.{0..26}.{layer_norm1,layer_norm2,self_attn.{q,k,v,out}_proj,
      mlp.fc1,mlp.fc2}.{weight,bias}                                   -> enc.layers.{i}.*
  model.vision_tower.vit_merger.{layer_norm1,self_attn.{q,k,v,out}_proj,pre_norm,linear_1,
      linear_2}.{weight,bias}                                          -> enc.vit_merger.*
  model.vision_tower.post_layernorm.{weight,bias}                      -> enc.post_layernorm.*
  model.merger.mlp.0.{pre_norm,linear_1,linear_2}.{weight,bias}        -> adp.*
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

IMG = 448
PATCH = 14
GRID = IMG // PATCH            # 32
N_PATCH = GRID * GRID          # 1024
WGRID = GRID // 2              # 16 (after the window merger)
N_WIN = WGRID * WGRID          # 256
MGRID = WGRID // 2             # 8 (after the merger)
N_TOK = MGRID * MGRID          # 64 soft tokens
HID = 1152
HEADS = 16
HDIM = HID // HEADS            # 72
INTER = 4304
WIN_HID = HID * 4              # 4608
WIN_INTER = INTER * 4          # 17216
LLM_HID = 1024
EPS = 1e-6
POS_SIDE = 70                  # image_size 980 / patch 14
INSERT = 6                     # insert_layer_id
NLAYERS = 27
MEAN = STD = 0.5
# Additive mask outside the 2x2 window. Finite in fp16 (|x| <= 65504): -1e9 would become -inf
# on an fp16 GPU path, and a softmax that meets an all -inf prefix can produce NaN. In fp32 the
# result is identical to -1e9 (exp underflows to exactly 0 for any realistic logit).
MASK_NEG = -1.0e4

PREFIX_VT = "model.vision_tower."
PREFIX_MG = "model.merger.mlp.0."


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


def window_mask(grid=GRID, neg=MASK_NEG):
  """[1,1,N,N] additive mask, raster token order: 0 iff both tokens are in the same 2x2 window."""
  r = torch.arange(grid * grid)
  win = (r // grid // 2) * (grid // 2) + (r % grid) // 2
  same = win[:, None] == win[None, :]
  return torch.where(same, torch.tensor(0.0), torch.tensor(float(neg))).reshape(1, 1, grid * grid, grid * grid)


def merge2x2(f, side):
  """[1, side*side, C] raster -> [1, (side/2)^2, 4C] window order (r0c0, r0c1, r1c0, r1c1)."""
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
  # fp16-safe: pre-scale by a power of two (exact) so (x-m)^2 stays inside fp16 range.
  xs = x * (1.0 / S)
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

  def forward(self, x, mask=None):                       # x [1,L,C]
    L = x.shape[1]
    q = self.q_proj(x).reshape(1, L, HEADS, HDIM).permute(0, 2, 1, 3)   # [1,H,L,d]
    k = self.k_proj(x).reshape(1, L, HEADS, HDIM).permute(0, 2, 1, 3)
    v = self.v_proj(x).reshape(1, L, HEADS, HDIM).permute(0, 2, 1, 3)
    a = (q * self.scale) @ k.transpose(-2, -1)                           # [1,H,L,L]
    if mask is not None:
      a = a + mask
    a = a.softmax(dim=-1)
    o = (a @ v).permute(0, 2, 1, 3).reshape(1, L, HID)                  # [1,L,C]
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


class VitMerger(nn.Module):
  def __init__(self):
    super().__init__()
    self.layer_norm1 = nn.LayerNorm(HID, eps=EPS)
    self.self_attn = Attention()
    self.pre_norm = nn.LayerNorm(WIN_HID, eps=EPS)
    self.linear_1 = nn.Linear(WIN_HID, WIN_INTER)
    self.linear_2 = nn.Linear(WIN_INTER, HID)


class VisionEncoder(nn.Module):
  def __init__(self, opts):
    super().__init__()
    self.opts = opts
    self.patch_embedding = nn.Conv2d(3, HID, PATCH, stride=PATCH, bias=True)
    self.register_buffer("pos_embed", torch.zeros(1, N_PATCH, HID))
    self.layers = nn.ModuleList([EncoderLayer() for _ in range(NLAYERS)])
    self.vit_merger = VitMerger()
    self.post_layernorm = nn.LayerNorm(HID, eps=EPS)
    self.register_buffer("win_mask", window_mask(), persistent=False)

  def _barrier(self, x):
    return torch.clamp(x, -65504.0, 65504.0) if self.opts.barrier else x

  def _ln(self, key, x, ln, calib):
    note(calib, key, x)
    return layer_norm(x, ln, self.opts.s(key), self.opts.fp16safe)

  def _layer(self, i, h, calib):
    blk = self.layers[i]
    n1 = self._ln(f"layers.{i}.layer_norm1", h, blk.layer_norm1, calib)
    h = h + blk.self_attn(self._barrier(n1))
    n2 = self._ln(f"layers.{i}.layer_norm2", h, blk.layer_norm2, calib)
    return h + blk.mlp.fc2(F.gelu(blk.mlp.fc1(self._barrier(n2)), approximate="tanh"))

  def _vit_merger(self, h, calib):                      # [1,1024,C] raster -> [1,256,C] raster
    vm = self.vit_merger
    n = self._ln("vit_merger.layer_norm1", h, vm.layer_norm1, calib)
    h = h + vm.self_attn(self._barrier(n), self.win_mask)
    g = h.reshape(1, GRID, GRID, HID)
    s00, s01 = g[:, 0::2, 0::2, :], g[:, 0::2, 1::2, :]
    s10, s11 = g[:, 1::2, 0::2, :], g[:, 1::2, 1::2, :]
    flat = torch.cat([s00, s01, s10, s11], dim=-1).reshape(1, N_WIN, WIN_HID)
    patch_residual = ((s00 + s01 + s10 + s11) * 0.25).reshape(1, N_WIN, HID)
    y = self._ln("vit_merger.pre_norm", flat, vm.pre_norm, calib)
    y = vm.linear_2(F.gelu(vm.linear_1(self._barrier(y)), approximate="tanh"))
    return y + patch_residual

  def forward(self, images, calib=None, taps=None):     # [1,448,448,3] in [0,1]
    x = ((images - MEAN) / STD).permute(0, 3, 1, 2)      # NCHW, normalized
    p = self.patch_embedding(x)                          # [1,1152,32,32]
    h = p.permute(0, 2, 3, 1).reshape(1, N_PATCH, HID) + self.pos_embed
    tap(taps, "embeddings", h)
    for i in range(INSERT + 1):
      h = self._layer(i, h, calib)
      tap(taps, f"layer{i:02d}", h)
    h = self._vit_merger(h, calib)
    tap(taps, "vit_merger", h)
    for i in range(INSERT + 1, NLAYERS):
      h = self._layer(i, h, calib)
      tap(taps, f"layer{i:02d}", h)
    return self._ln("post_layernorm", h, self.post_layernorm, calib)   # [1,256,1152]


class VisionAdapter(nn.Module):
  def __init__(self, opts):
    super().__init__()
    self.opts = opts
    self.pre_norm = nn.LayerNorm(WIN_HID, eps=EPS)
    self.linear_1 = nn.Linear(WIN_HID, WIN_HID)
    self.linear_2 = nn.Linear(WIN_HID, LLM_HID)

  def forward(self, feats, calib=None):                  # [1,256,1152]
    m = merge2x2(feats, WGRID)                           # [1,64,4608]
    note(calib, "merger.pre_norm", m)
    y = layer_norm(m, self.pre_norm, self.opts.s("merger.pre_norm"), self.opts.fp16safe)
    if self.opts.barrier:
      y = torch.clamp(y, -65504.0, 65504.0)
    return self.linear_2(F.gelu(self.linear_1(y)))       # erf GELU -> [1,64,1024]


def load_vision_state(src_dir):
  from safetensors import safe_open
  out = {}
  with safe_open(os.path.join(src_dir, "model.safetensors"), framework="pt") as f:
    for k in f.keys():
      if k.startswith(PREFIX_VT) or k.startswith(PREFIX_MG):
        out[k] = f.get_tensor(k).to(torch.float32)
  return out


def build(src_dir, opts):
  """-> (VisionEncoder, VisionAdapter, info). Strict load: every vision/merger tensor used once."""
  sd = load_vision_state(src_dir)
  pos_ids = nearest_pos_ids(GRID, GRID)
  enc_sd, adp_sd = {}, {}
  for k, v in sd.items():
    if k.startswith(PREFIX_VT):
      kk = k[len(PREFIX_VT):]
      if kk == "embeddings.position_embedding.weight":
        assert tuple(v.shape) == (POS_SIDE * POS_SIDE, HID), v.shape
        enc_sd["pos_embed"] = v[pos_ids].reshape(1, N_PATCH, HID).contiguous()
      elif kk.startswith("embeddings.patch_embedding."):
        enc_sd["patch_embedding." + kk.rsplit(".", 1)[1]] = v
      elif kk.startswith("encoder.layers."):
        enc_sd[kk[len("encoder."):]] = v
      else:                                              # vit_merger.*, post_layernorm.*
        enc_sd[kk] = v
    else:
      adp_sd[k[len(PREFIX_MG):]] = v
  enc, adp = VisionEncoder(opts), VisionAdapter(opts)
  enc.load_state_dict(enc_sd, strict=True)
  adp.load_state_dict(adp_sd, strict=True)
  info = {"n_src_tensors": len(sd), "n_enc_keys": len(enc_sd), "n_adp_keys": len(adp_sd),
          "enc_params": sum(p.numel() for p in enc.parameters()),
          "adp_params": sum(p.numel() for p in adp.parameters()),
          "pos_ids": pos_ids}
  return enc.eval(), adp.eval(), info


def strip_from_image(img01_nhwc):
  """The processor's reshape_by_patch strip [1,3,14,N*14] built from a [1,448,448,3] [0,1] image."""
  x = ((img01_nhwc - MEAN) / STD).permute(0, 3, 1, 2)[0]          # [3,448,448]
  p = F.unfold(x.unsqueeze(0), (PATCH, PATCH), stride=(PATCH, PATCH))
  p = p.reshape(3, PATCH, PATCH, -1).permute(0, 1, 3, 2).reshape(3, PATCH, -1)
  return p.unsqueeze(0)


def calib_S(absmax):
  """Power-of-two pre-scale: absmax / S <= 8 (qwen35vl_work recipe)."""
  import math
  return float(2 ** max(0, int(math.ceil(math.log2(max(absmax, 1e-6) / 8.0)))))
