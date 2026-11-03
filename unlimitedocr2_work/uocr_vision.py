"""Pure-torch re-author of the Unlimited-OCR-2 DeepEncoder for ONE base-mode global view (1024x1024), split for the
LiteRT-LM fast_vlm contract, and the HF reference built from the repo's own deepencoder.py classes.

  VisionEncoder  image NHWC [1,1024,1024,3] in [0,1] (the runtime hands float(u8) * (1/255))
                 -> u8 taken back in the graph: w = floor(x*255 - 127.5) = u8 - 128, then (w + 0.5) / 127.5
                    = (u8 - 127.5) / 127.5 = ToTensor + Normalize(mean 0.5, std 0.5) of BasicImageTransform
                    (centre before the FLOOR so the converter cannot fold the step into the patch conv)
                 -> NCHW -> SAM ViT-B: patch 16 -> 64x64 grid, + pos_embed [1,64,64,768] (64 = the checkpoint's grid:
                    get_abs_pos_sam interpolates nothing at 1024), 12 blocks (window 14 with the pad 64 -> 70 except
                    the global blocks 2/5/8/11; decomposed relative positions, the gathered tables Rh / Rw are constant
                    buffers: get_rel_pos interpolates nothing, 2*64-1 = 127 and 2*14-1 = 27 = the checkpoint's rows),
                    neck (1x1 conv, LayerNorm2d, 3x3 conv, LayerNorm2d), net_2 (3x3 s2 -> 512x32x32),
                    net_3 (3x3 s2 -> 1024x16x16)
                 -> CLIP-L conditioned on SAM: patch embeds = the SAM output (CLIP's own patch_embedding is unused by
                    the model), cls + 256 tokens + position_embedding rows 0..256 (257 = the checkpoint's rows:
                    get_abs_pos interpolates nothing), pre_layrnorm, 24 blocks (LN eps 1e-5, 16 heads x 64, no mask,
                    quick_gelu MLP 4096)
                 -> cat(clip[:, 1:], sam flattened in raster order) = [1,256,2048]
  VisionAdapter  [1,256,2048] -> projector Linear 2048 -> 1280 -> view 16x16, image_newline appended to every row
                 (16x17 = 272), view_seperator appended = [1,273,1280]
                 = the base-mode (no crop) branch of UnlimitedOCRModel.forward, the rows masked_scatter_ writes into
                 the 273 <image> slots of one page.

Attention is written out (q*scale @ k^T + bias, softmax, @ v), not SDPA. The relative-position bias of SAM is added
after the scale exactly like SDPA's attn_mask in the source. `rel_form`:
  "bias"  rel_h [.., kh] + rel_w [.., kw] broadcast to the [L, L] bias (the source's form)
  "aug"   the same scores from one matmul of augmented q / k: [q*scale | rel_h | rel_w] @ [k | onehot(kh) | onehot(kw)]^T
          (exact selection by one-hot columns; no separate [L, L] bias tensor)
`qsplit` = how many query chunks the global attention is computed in (1 = whole).

fp16 safety (switchable, exact in real arithmetic): LayerNorm with power-of-two pre-scales S per norm (ln_s table).
Every activation keeps a leading batch dim. Flattening NCHW -> tokens is written permute(0,2,3,1).reshape (the
qwen35vl / MiniCPM-OCR precedent: flatten(2).transpose lowered to a TRANSPOSE->RESHAPE that Metal miscomputed).
"""
import importlib.util
import json
import math
import os
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

HF = os.environ.get("MODEL", "hf")            # the source checkpoint: baidu/Unlimited-OCR-2 at revision 04809669

IMG = 1024
SAM_PATCH = 16
SAM_GRID = IMG // SAM_PATCH          # 64
SAM_DIM = 768
SAM_HEADS = 12
SAM_HDIM = SAM_DIM // SAM_HEADS      # 64
SAM_DEPTH = 12
SAM_WIN = 14
SAM_GLOBAL = (2, 5, 8, 11)
SAM_EPS = 1e-6
NECK = 256
CLIP_DIM = 1024
CLIP_HEADS = 16
CLIP_HDIM = 64
CLIP_DEPTH = 24
CLIP_FFN = 4096
CLIP_EPS = 1e-5
FEAT_GRID = 16                       # 1024 / 16 / 4
N_FEAT = FEAT_GRID * FEAT_GRID       # 256
LLM_DIM = 1280
N_SLOTS = FEAT_GRID * (FEAT_GRID + 1) + 1   # 273

PREFIX = {"sam": "model.sam_model.", "clip": "model.vision_model.", "proj": "model.projector.layers."}
UNUSED_CLIP = ("embeddings.patch_embedding.weight", "embeddings.position_ids")


class Opts:
  def __init__(self, fp16safe=False, ln_s=None, rel_form="bias", qsplit=1, barrier=False):
    self.fp16safe = fp16safe
    self.ln_s = dict(ln_s or {})
    self.rel_form = rel_form
    self.qsplit = int(qsplit)
    self.barrier = barrier

  def s(self, key):
    return float(self.ln_s.get(key, 1.0))

  def as_dict(self):
    return {"fp16safe": self.fp16safe, "ln_s_gt1": {k: v for k, v in sorted(self.ln_s.items()) if v > 1.0},
            "rel_form": self.rel_form, "qsplit": self.qsplit, "barrier": self.barrier}


def note(calib, key, x):
  if calib is not None:
    calib[key] = max(calib.get(key, 0.0), float(x.detach().abs().max()))


def layer_norm(x, w, b, eps, S, fp16safe):
  """LayerNorm over the last dim; with fp16safe and S > 1 the input is divided by the power of two S first and eps
  by S^2 (the same value in real arithmetic, the sum of squares stays inside fp16)."""
  if not fp16safe or S <= 1.0:
    mu = x.mean(-1, keepdim=True)
    d = x - mu
    var = (d * d).mean(-1, keepdim=True)
    return d * torch.rsqrt(var + eps) * w + b
  xs = x * (1.0 / S)
  d = xs - xs.mean(-1, keepdim=True)
  var = (d * d).mean(-1, keepdim=True)
  return d * torch.rsqrt(var + eps / (S * S)) * w + b


def rel_table(rel_pos, size):
  """get_rel_pos(q_size=k_size=size) of deepencoder.py when the table needs no interpolation: [size, size, C]."""
  assert rel_pos.shape[0] == 2 * size - 1, (rel_pos.shape, size)
  q = torch.arange(size)[:, None]
  k = torch.arange(size)[None, :]
  return rel_pos[(q - k) + (size - 1)]


class SamBlock(nn.Module):
  def __init__(self, idx):
    super().__init__()
    self.idx = idx
    self.win = 0 if idx in SAM_GLOBAL else SAM_WIN
    self.norm1 = nn.LayerNorm(SAM_DIM, eps=SAM_EPS)
    self.norm2 = nn.LayerNorm(SAM_DIM, eps=SAM_EPS)
    self.qkv = nn.Linear(SAM_DIM, 3 * SAM_DIM)
    self.proj = nn.Linear(SAM_DIM, SAM_DIM)
    self.lin1 = nn.Linear(SAM_DIM, 4 * SAM_DIM)
    self.lin2 = nn.Linear(4 * SAM_DIM, SAM_DIM)
    side = SAM_GRID if self.win == 0 else self.win
    self.side = side
    self.register_buffer("Rh", torch.zeros(side, side, SAM_HDIM), persistent=False)
    self.register_buffer("Rw", torch.zeros(side, side, SAM_HDIM), persistent=False)
    # one-hot key coordinates for rel_form "aug": key j = (kh, kw) in raster order
    kk = torch.arange(side * side)
    self.register_buffer("Eh", F.one_hot(kk // side, side).float(), persistent=False)   # [L, side]
    self.register_buffer("Ew", F.one_hot(kk % side, side).float(), persistent=False)    # [L, side]

  def set_rel(self, rel_pos_h, rel_pos_w):
    self.Rh = rel_table(rel_pos_h.float(), self.side).to(self.Rh.dtype)
    self.Rw = rel_table(rel_pos_w.float(), self.side).to(self.Rw.dtype)

  def attn(self, x, opts):
    """x [B', side, side, C] (B' = windows or 1) -> [B', side, side, C]. Every tensor stays at rank <= 4: q / k / v
    are the three C-wide slices of the qkv output (= reshape(B', L, 3, nH, d) of the source)."""
    Bp, h, w, C = x.shape
    L = h * w
    qkv = self.qkv(x.reshape(Bp, L, C))                                                # [B', L, 3C]

    def heads(t):
      return t.reshape(Bp, L, SAM_HEADS, SAM_HDIM).permute(0, 2, 1, 3).reshape(Bp * SAM_HEADS, L, SAM_HDIM)

    q, k, v = heads(qkv[:, :, :C]), heads(qkv[:, :, C:2 * C]), heads(qkv[:, :, 2 * C:])
    scale = SAM_HDIM ** -0.5
    BH = Bp * SAM_HEADS
    # decomposed rel pos (add_decomposed_rel_pos): rel_h[b, i, kh] = sum_c q[b, i, c] Rh[hi, kh, c]
    rq = q.reshape(BH, h, w, SAM_HDIM)
    rel_h = (rq.permute(1, 0, 2, 3).reshape(h, BH * w, SAM_HDIM) @ self.Rh.transpose(1, 2))     # [h, BH*w, kh]
    rel_h = rel_h.reshape(h, BH, w, h).permute(1, 0, 2, 3).reshape(BH, L, h)
    rel_w = (rq.permute(2, 0, 1, 3).reshape(w, BH * h, SAM_HDIM) @ self.Rw.transpose(1, 2))     # [w, BH*h, kw]
    rel_w = rel_w.reshape(w, BH, h, w).permute(1, 2, 0, 3).reshape(BH, L, w)
    nq = opts.qsplit if self.win == 0 else 1
    step = L // nq
    assert step * nq == L
    outs = []
    for c in range(nq):
      sl = slice(c * step, (c + 1) * step)
      qc = q[:, sl]
      if opts.rel_form == "aug":
        qa = torch.cat([qc * scale, rel_h[:, sl], rel_w[:, sl]], dim=-1)                          # [BH, l, d+h+w]
        ka = torch.cat([k, self.Eh.expand(BH, L, h), self.Ew.expand(BH, L, w)], dim=-1)           # [BH, L, d+h+w]
        s = qa @ ka.transpose(1, 2)
      else:
        s = (qc * scale) @ k.transpose(1, 2)                                                       # [BH, l, L]
        bias = (rel_h[:, sl].reshape(BH, step, h, 1) + rel_w[:, sl].reshape(BH, step, 1, w)).reshape(BH, step, L)
        s = s + bias
      a = s.softmax(dim=-1)
      outs.append(a @ v)
    o = outs[0] if nq == 1 else torch.cat(outs, dim=1)                                             # [BH, L, d]
    o = o.reshape(Bp, SAM_HEADS, L, SAM_HDIM).permute(0, 2, 1, 3).reshape(Bp, L, C)
    return self.proj(o).reshape(Bp, h, w, C)

  def forward(self, x, opts, calib=None):
    note(calib, f"sam.{self.idx}.norm1", x)
    sc = x
    n = layer_norm(x, self.norm1.weight, self.norm1.bias, SAM_EPS, opts.s(f"sam.{self.idx}.norm1"), opts.fp16safe)
    if self.win > 0:
      H = W = SAM_GRID
      pad = (self.win - H % self.win) % self.win
      n = F.pad(n, (0, 0, 0, pad, 0, pad))
      Hp = H + pad
      nw = Hp // self.win
      wc = self.win * SAM_DIM
      # window_partition at rank 4: (a, r, b, (c, C)) -> (a, b, r, (c, C))
      n = n.reshape(nw, self.win, nw, wc).permute(0, 2, 1, 3).reshape(nw * nw, self.win, self.win, SAM_DIM)
      a = self.attn(n, opts)
      # window_unpartition at rank 4: (a, b, r, (c, C)) -> (a, r, b, (c, C))
      a = a.reshape(nw, nw, self.win, wc).permute(0, 2, 1, 3).reshape(1, Hp, Hp, SAM_DIM)
      a = a[:, :H, :W, :]
    else:
      a = self.attn(n, opts)
    x = sc + a
    note(calib, f"sam.{self.idx}.norm2", x)
    n2 = layer_norm(x, self.norm2.weight, self.norm2.bias, SAM_EPS, opts.s(f"sam.{self.idx}.norm2"), opts.fp16safe)
    return x + self.lin2(F.gelu(self.lin1(n2)))


class ClipBlock(nn.Module):
  def __init__(self, idx):
    super().__init__()
    self.idx = idx
    self.layer_norm1 = nn.LayerNorm(CLIP_DIM, eps=CLIP_EPS)
    self.layer_norm2 = nn.LayerNorm(CLIP_DIM, eps=CLIP_EPS)
    self.qkv_proj = nn.Linear(CLIP_DIM, 3 * CLIP_DIM)
    self.out_proj = nn.Linear(CLIP_DIM, CLIP_DIM)
    self.fc1 = nn.Linear(CLIP_DIM, CLIP_FFN)
    self.fc2 = nn.Linear(CLIP_FFN, CLIP_DIM)

  def forward(self, x, opts, calib=None):
    L = x.shape[1]
    note(calib, f"clip.{self.idx}.ln1", x)
    n = layer_norm(x, self.layer_norm1.weight, self.layer_norm1.bias, CLIP_EPS, opts.s(f"clip.{self.idx}.ln1"),
                   opts.fp16safe)
    qkv = self.qkv_proj(n)                                     # [1, L, 3C] = view(1, L, 3, nH, d) in the source

    def heads(t):
      return t.reshape(1, L, CLIP_HEADS, CLIP_HDIM).permute(0, 2, 1, 3)

    q, k, v = heads(qkv[:, :, :CLIP_DIM]), heads(qkv[:, :, CLIP_DIM:2 * CLIP_DIM]), heads(qkv[:, :, 2 * CLIP_DIM:])
    a = ((q * (CLIP_HDIM ** -0.5)) @ k.transpose(-2, -1)).softmax(dim=-1)
    o = (a @ v).permute(0, 2, 1, 3).reshape(1, L, CLIP_DIM)
    h = x + self.out_proj(o)
    note(calib, f"clip.{self.idx}.ln2", h)
    n2 = layer_norm(h, self.layer_norm2.weight, self.layer_norm2.bias, CLIP_EPS, opts.s(f"clip.{self.idx}.ln2"),
                    opts.fp16safe)
    f = self.fc1(n2)
    return h + self.fc2(f * torch.sigmoid(1.702 * f))


class LN2d(nn.Module):
  def __init__(self, c):
    super().__init__()
    self.weight = nn.Parameter(torch.ones(c))
    self.bias = nn.Parameter(torch.zeros(c))


class VisionEncoder(nn.Module):
  def __init__(self, opts):
    super().__init__()
    self.opts = opts
    self.patch = nn.Conv2d(3, SAM_DIM, SAM_PATCH, SAM_PATCH)
    self.pos_embed = nn.Parameter(torch.zeros(1, SAM_GRID, SAM_GRID, SAM_DIM))
    self.blocks = nn.ModuleList([SamBlock(i) for i in range(SAM_DEPTH)])
    self.neck0 = nn.Conv2d(SAM_DIM, NECK, 1, bias=False)
    self.neck1 = LN2d(NECK)
    self.neck2 = nn.Conv2d(NECK, NECK, 3, padding=1, bias=False)
    self.neck3 = LN2d(NECK)
    self.net_2 = nn.Conv2d(NECK, 512, 3, stride=2, padding=1, bias=False)
    self.net_3 = nn.Conv2d(512, 1024, 3, stride=2, padding=1, bias=False)
    self.cls = nn.Parameter(torch.zeros(1, 1, CLIP_DIM))
    self.clip_pos = nn.Parameter(torch.zeros(1, N_FEAT + 1, CLIP_DIM))
    self.pre_ln = nn.LayerNorm(CLIP_DIM, eps=CLIP_EPS)
    self.clip = nn.ModuleList([ClipBlock(i) for i in range(CLIP_DEPTH)])

  def _ln2d(self, key, x_nhwc, mod, calib):
    """LayerNorm2d of deepencoder.py over channels, written on NHWC (last dim) = the same reduction."""
    note(calib, key, x_nhwc)
    return layer_norm(x_nhwc, mod.weight, mod.bias, 1e-6, self.opts.s(key), self.opts.fp16safe)

  def normalize(self, img01):
    w = torch.floor(img01 * 255.0 - 127.5)          # = u8 - 128 for the runtime's float(u8) * (1/255)
    return (w + 0.5) * (1.0 / 127.5)                 # = (u8 - 127.5) / 127.5

  def forward(self, img01, calib=None, taps=None, pre_normalized=None):
    x = self.normalize(img01) if pre_normalized is None else pre_normalized   # [1,1024,1024,3]
    x = x.permute(0, 3, 1, 2)                                                  # NCHW
    x = self.patch(x).permute(0, 2, 3, 1)                                      # [1,64,64,768]
    x = x + self.pos_embed
    for blk in self.blocks:
      x = blk(x, self.opts, calib)
    if taps is not None:
      taps["sam_blocks"] = x.detach().clone()
    y = self.neck0(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)                 # [1,64,64,256]
    y = self._ln2d("sam.neck1", y, self.neck1, calib)
    y = self.neck2(y.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    y = self._ln2d("sam.neck3", y, self.neck3, calib)
    y = self.net_2(y.permute(0, 3, 1, 2))                                      # [1,512,32,32]
    s = self.net_3(y)                                                          # [1,1024,16,16]
    sam_tok = s.permute(0, 2, 3, 1).reshape(1, N_FEAT, CLIP_DIM)               # raster order = flatten(2).permute
    if taps is not None:
      taps["sam_out"] = sam_tok.detach().clone()
    h = torch.cat([self.cls, sam_tok], dim=1) + self.clip_pos                  # [1,257,1024]
    note(calib, "clip.pre_ln", h)
    h = layer_norm(h, self.pre_ln.weight, self.pre_ln.bias, CLIP_EPS, self.opts.s("clip.pre_ln"), self.opts.fp16safe)
    for blk in self.clip:
      h = blk(h, self.opts, calib)
    if taps is not None:
      taps["clip_out"] = h.detach().clone()
    return torch.cat([h[:, 1:], sam_tok], dim=-1)                              # [1,256,2048]


class VisionAdapter(nn.Module):
  def __init__(self):
    super().__init__()
    self.proj = nn.Linear(2 * CLIP_DIM, LLM_DIM)
    self.image_newline = nn.Parameter(torch.zeros(1, 1, 1, LLM_DIM))
    self.view_seperator = nn.Parameter(torch.zeros(1, 1, LLM_DIM))

  def forward(self, feats):                                                    # [1,256,2048]
    p = self.proj(feats).reshape(1, FEAT_GRID, FEAT_GRID, LLM_DIM)
    p = torch.cat([p, self.image_newline.expand(1, FEAT_GRID, 1, LLM_DIM)], dim=2)   # [1,16,17,1280]
    p = p.reshape(1, FEAT_GRID * (FEAT_GRID + 1), LLM_DIM)
    return torch.cat([p, self.view_seperator], dim=1)                          # [1,273,1280]


# ----------------------------------------------------------------------------------------------- weights


def load_vision_tensors(src_dir=HF):
  from safetensors import safe_open
  idx = json.load(open(os.path.join(src_dir, "model.safetensors.index.json")))["weight_map"]
  want = [k for k in idx if any(k.startswith(p) for p in PREFIX.values())
          or k in ("model.image_newline", "model.view_seperator")]
  by_shard = {}
  for k in want:
    by_shard.setdefault(idx[k], []).append(k)
  out = {}
  for shard, ks in sorted(by_shard.items()):
    with safe_open(os.path.join(src_dir, shard), "pt") as f:
      for k in ks:
        out[k] = f.get_tensor(k)
  return out


def build(opts, dtype=torch.float32, src_dir=HF, tensors=None):
  t = tensors if tensors is not None else load_vision_tensors(src_dir)
  cast = lambda x: x.to(dtype) if x.is_floating_point() else x   # noqa: E731
  enc, adp = VisionEncoder(opts), VisionAdapter()
  used = set()

  def take(key):
    used.add(key)
    return cast(t[key])

  sam, clp = PREFIX["sam"], PREFIX["clip"]
  sd = {"patch.weight": take(sam + "patch_embed.proj.weight"), "patch.bias": take(sam + "patch_embed.proj.bias"),
        "pos_embed": take(sam + "pos_embed")}
  for i in range(SAM_DEPTH):
    b = f"{sam}blocks.{i}."
    for dst, src in (("norm1", "norm1"), ("norm2", "norm2"), ("qkv", "attn.qkv"), ("proj", "attn.proj"),
                     ("lin1", "mlp.lin1"), ("lin2", "mlp.lin2")):
      for p in ("weight", "bias"):
        sd[f"blocks.{i}.{dst}.{p}"] = take(f"{b}{src}.{p}")
  for dst, src in (("neck0.weight", "neck.0.weight"), ("neck1.weight", "neck.1.weight"), ("neck1.bias", "neck.1.bias"),
                   ("neck2.weight", "neck.2.weight"), ("neck3.weight", "neck.3.weight"), ("neck3.bias", "neck.3.bias"),
                   ("net_2.weight", "net_2.weight"), ("net_3.weight", "net_3.weight")):
    sd[dst] = take(sam + src)
  sd["cls"] = take(clp + "embeddings.class_embedding").reshape(1, 1, CLIP_DIM)
  pos_ids = t[clp + "embeddings.position_ids"]
  used.add(clp + "embeddings.position_ids")
  assert pos_ids.tolist() == [list(range(N_FEAT + 1))], "position_ids is not arange(257)"
  sd["clip_pos"] = take(clp + "embeddings.position_embedding.weight").reshape(1, N_FEAT + 1, CLIP_DIM)
  sd["pre_ln.weight"] = take(clp + "pre_layrnorm.weight")
  sd["pre_ln.bias"] = take(clp + "pre_layrnorm.bias")
  for i in range(CLIP_DEPTH):
    b = f"{clp}transformer.layers.{i}."
    for dst, src in (("layer_norm1", "layer_norm1"), ("layer_norm2", "layer_norm2"),
                     ("qkv_proj", "self_attn.qkv_proj"), ("out_proj", "self_attn.out_proj"),
                     ("fc1", "mlp.fc1"), ("fc2", "mlp.fc2")):
      for p in ("weight", "bias"):
        sd[f"clip.{i}.{dst}.{p}"] = take(f"{b}{src}.{p}")
  missing, unexpected = enc.load_state_dict(sd, strict=False)
  assert not unexpected, unexpected
  assert not missing, missing
  for i, blk in enumerate(enc.blocks):
    b = f"{sam}blocks.{i}.attn."
    blk.set_rel(take(b + "rel_pos_h"), take(b + "rel_pos_w"))
  adp_sd = {"proj.weight": take(PREFIX["proj"] + "weight"), "proj.bias": take(PREFIX["proj"] + "bias"),
            "image_newline": take("model.image_newline").reshape(1, 1, 1, LLM_DIM),
            "view_seperator": take("model.view_seperator").reshape(1, 1, LLM_DIM)}
  adp.load_state_dict(adp_sd, strict=True)
  unused = sorted(set(t) - used)
  assert unused == [clp + "embeddings.patch_embedding.weight"], unused
  enc = enc.to(dtype).eval()
  adp = adp.to(dtype).eval()
  info = {"n_src_tensors": len(t), "n_used": len(used), "unused": unused,
          "enc_params": sum(p.numel() for p in enc.parameters()),
          "adp_params": sum(p.numel() for p in adp.parameters()), "opts": opts.as_dict()}
  return enc, adp, info


# ----------------------------------------------------------------------------------------------- HF reference


def _shim_modules():
  """deepencoder.py imports einops.rearrange (never used) and easydict.EasyDict (an attribute dict for the CLIP
  config). Neither package is in the conversion venv; these stand-ins are registered only while it is imported."""
  class AttrDict(dict):
    def __getattr__(self, k):
      try:
        return self[k]
      except KeyError as e:
        raise AttributeError(k) from e

    def __setattr__(self, k, v):
      self[k] = v

  einops = types.ModuleType("einops")
  einops.rearrange = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("rearrange is not expected to run"))
  easydict = types.ModuleType("easydict")
  easydict.EasyDict = AttrDict
  return {"einops": einops, "easydict": easydict}, AttrDict


def load_hf_deepencoder(src_dir=HF):
  shims, AttrDict = _shim_modules()
  # a venv that has the real packages (the author's pinned versions) imports them; only missing ones are shimmed
  shims = {k: v for k, v in shims.items() if importlib.util.find_spec(k) is None}
  saved = {k: sys.modules.get(k) for k in shims}
  sys.modules.update(shims)
  try:
    spec = importlib.util.spec_from_file_location("uocr2_deepencoder", os.path.join(src_dir, "deepencoder.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
  finally:
    for k, v in saved.items():
      if v is None:
        sys.modules.pop(k, None)
      else:
        sys.modules[k] = v
  return mod, AttrDict


class HFReference(nn.Module):
  """sam_model / vision_model / projector built by deepencoder.py's own builders + the base-mode branch of
  UnlimitedOCRModel.forward copied verbatim (no crop: one global view of one page)."""

  def __init__(self, dtype=torch.float32, src_dir=HF, tensors=None):
    super().__init__()
    mod, AttrDict = load_hf_deepencoder(src_dir)
    self.sam_model = mod.build_sam_vit_b()
    self.vision_model = mod.build_clip_l()
    self.projector = mod.MlpProjector(AttrDict(projector_type="linear", input_dim=2048, n_embed=1280))
    self.image_newline = nn.Parameter(torch.zeros(LLM_DIM))
    self.view_seperator = nn.Parameter(torch.zeros(LLM_DIM))
    t = tensors if tensors is not None else load_vision_tensors(src_dir)
    for name, m in (("sam", self.sam_model), ("clip", self.vision_model), ("proj", self.projector)):
      p = PREFIX[name] if name != "proj" else "model.projector."
      sd = {k[len(p):]: v for k, v in t.items() if k.startswith(p)}
      m.load_state_dict(sd, strict=True)
    self.image_newline.data = t["model.image_newline"].clone()
    self.view_seperator.data = t["model.view_seperator"].clone()
    self.to(dtype).eval()

  def forward(self, image_ori_p):                                              # [1,3,1024,1024] normalized
    sam_model, vision_model = self.sam_model, self.vision_model
    global_features_1 = sam_model(image_ori_p)
    global_features_2 = vision_model(image_ori_p, global_features_1)
    global_features = torch.cat((global_features_2[:, 1:], global_features_1.flatten(2).permute(0, 2, 1)), dim=-1)
    feats = global_features
    global_features = self.projector(global_features)
    _, hw, n_dim = global_features.shape
    h = w = int(hw ** 0.5)
    global_features = global_features.view(h, w, n_dim)
    global_features = torch.cat(
        [global_features, self.image_newline[None, None, :].expand(h, 1, n_dim)], dim=1
    )
    global_features = global_features.view(-1, n_dim)
    global_local_features = torch.cat([global_features, self.view_seperator[None, :]], dim=0)
    return feats, global_local_features


def hf_pixels(img_pil):
  """BasicImageTransform of modeling_unlimitedocr.py on an already padded 1024x1024 RGB page (ToTensor, then
  Normalize(0.5, 0.5)), fp32, NCHW [1,3,1024,1024]. The HF path then casts to bf16; the fp32 reference keeps fp32."""
  from torchvision import transforms
  tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
  return tf(img_pil).unsqueeze(0)


def author_pad(img_pil, size=IMG):
  """_uocr_build_inputs base view: ImageOps.pad(image, (size, size), color=(127, 127, 127)) (PIL default BICUBIC,
  centred) = the pad colour tuple(int(x * 255) for x in mean 0.5)."""
  from PIL import ImageOps
  return ImageOps.pad(img_pil.convert("RGB"), (size, size), color=(127, 127, 127))


def calib_S(absmax, target=8.0):
  """Power-of-two pre-scale: absmax / S <= target (the qwen35vl / MiniCPM-OCR recipe)."""
  return float(2 ** max(0, int(math.ceil(math.log2(max(absmax, 1e-6) / target)))))
