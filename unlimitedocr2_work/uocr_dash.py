"""The Unlimited-OCR-2 decoder with the model's DASH attention written statically for one page, in the export form of
uocr_decoder.py.

The math is the source's inference path (modeling_deepseekv2.py configure_dash_inference / _infer_route_select, the
page2d rotary embedding of _sparse_base_impl.py) on the fixed one-page layout, where every query has at most 9
candidate blocks (fewer than the 12 the router keeps), so the top-k selects nothing and no TOPK / GATHER is needed.
Written for the GPU (four rewrites of a first version that the Metal delegate refused or computed wrongly):
  (1) entmax-1.5 on rank-4 tensors ([1, T, 9, 9] pairwise, [1, T, 9, 1] per block), every reduce with keepdim;
  (2) no BROADCAST_TO: the block bias reaches its 32 tokens by five doubling CONCATENATIONs (a MUL with a ones
      constant [1, 1, 9, 32] before a RESHAPE is folded into BROADCAST_TO by the converter); the pairwise operands are
      MULs with ones constants (folded into plain broadcasting of the comparison);
  (3) in-graph constants inside the fp16 range: "not visible" is NEG = -3e4 (exp underflows to 0 in an fp32 softmax
      exactly as -0.7*FLT_MAX does); the runtime mask is only compared (> -1), never used in arithmetic; the entmax
      input is clamped at X >= -16 (tau* >= max X - 1 = -1, so every X < -1 is outside the support: the clamp
      changes no output in exact arithmetic and keeps the X^2 sums inside fp16);
  (4) no rank-3 operand pattern that the GPU delegates miscompute: 0 rows of a rank-3 lint on the exported file.
Layout (export_hf): kv_cache_k_<i> [1, 11, C, 128] = heads 0-9 the RoPE'd K, head 10 the
router key (dash_router.k_proj + Gemma norm + the same page2d RoPE); kv_cache_v_<i> [1, 10, 128, C] (transposed V).
The scores run over all 11 K heads with a zero 11th query head (no copy of the K cache), and the 11th probability
row is sliced off before the V product.
Per signature, once for the 12 layers (DashShared): cos / sin of f(p) by a one-hot [T, C] FULLY_CONNECTED on a
constant [C, 256] table (class RopeTable, kept float by quant_decoder.py), the candidate blocks (p >= 32 b + 128 is
b <= floor(p / 32) - 4), and the token-mask terms: local window 128 and the runtime gate. Per layer: the router q / k
(classes DashQ / DashK, kept float), 9 head_cls summaries of the router keys at positions 0..287, 9 block scores,
entmax-1.5, the log-prior bias, the [T, 288] image part of the mask next to the shared [T, C - 288] rest.
One page, the fixed layout: BOS + <PAGE> + 273 image + 4 instruction ids, the reply starts with <PAGE>; then the
source's 2-D (tok, page) position of every token is f(p), a function of the 1-D position p only (f_coords_1p).
"""
import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import uocr_decoder as ud

BS, NIMG, WL, DH = 32, 9, 128, 128
IMG = BS * NIMG                    # 288 positions of the 9 image blocks
NEG = -3.0e4                       # "not visible" inside the graph (fp16 range)
ENT_NEG = -1.0e4                   # HF _ENTMAX_NEG for non-candidate blocks
X_MIN = -16.0                      # entmax input clamp (see the module docstring)
ACT = 1.0e4                        # HF _ACT_CLAMP
PROMPT_LEN = 279


def f_coords_1p(C):
  """(tok, page) of every position p < C on the 1-page layout."""
  p = torch.arange(C)
  tok = torch.zeros(C, dtype=torch.long)
  page = torch.zeros(C, dtype=torch.long)
  m1 = (p >= 1) & (p <= 274)
  tok[m1], page[m1] = p[m1] - 1, 1
  m2 = (p >= 275) & (p < PROMPT_LEN)
  tok[m2], page[m2] = p[m2] - 274, 0
  m3 = p >= PROMPT_LEN
  tok[m3], page[m3] = p[m3] - PROMPT_LEN + 274, 1
  return tok, page


def page2d_tables(C, config=os.path.join(ud.HF, "config.json")):
  """cos / sin [C, 128] of f(p): hf/_sparse_base_impl.py DeepseekV2IndexerRotary2D.forward, the same float ops."""
  sp = json.load(open(config))["sparse_attention_config"]
  assert sp["sparse_main_rope_mode"] == "page2d" and sp["sparse_rope_mode"] == "page2d"
  assert sp["sparse_rope_tok_freq_mode"] == "resample" and sp["sparse_main_rope_tok_freq_mode"] == "resample"
  base_tok, base_page = float(sp["sparse_rope_tok_theta"]), float(sp["sparse_rope_page_theta"])
  page_pairs, page_mod, dim = int(sp["sparse_rope_page_pairs"]), int(sp["sparse_rope_page_mod"]), DH
  half = dim // 2
  n_tok = half - page_pairs
  ar = torch.arange(0, n_tok, dtype=torch.int64).float()
  inv_tok = 1.0 / (base_tok ** (ar / n_tok))
  inv_page = 1.0 / (base_page ** (torch.arange(0, page_pairs, dtype=torch.int64).float() / page_pairs))
  tok, page = f_coords_1p(C)
  t = tok.float()[None, :, None]
  pg = torch.remainder(page.to(torch.int64), page_mod).float()[None, :, None]
  ang = torch.cat((t * inv_tok[None, None, :], pg * inv_page[None, None, :]), dim=-1)
  emb = torch.cat((ang, ang), dim=-1)
  return emb.cos()[0], emb.sin()[0]


def dash_consts(config=os.path.join(ud.HF, "config.json")):
  sp = json.load(open(config))["sparse_attention_config"]
  sc, fl = sp["dash_scale_per_layer"], sp["dash_floor_per_layer"]
  assert all(float(v) == float(sc[0]) for v in sc) and all(float(v) == float(fl[0]) for v in fl)
  assert sp["dash_stage0"] == "head_cls" and int(sp["dash_topk_blocks"]) >= NIMG
  assert int(sp["sparse_block_size"]) == BS and int(sp["sparse_local_block"]) * BS == WL
  return {"idx_scale": float(sc[0]) / math.sqrt(DH), "floor": float(fl[0]), "sigma": float(sp["dash_sigma"])}


class RopeTable(nn.Linear):
  """one-hot [.., C] -> cat(cos, sin) [.., 256]; weight = the [C, 256] table transposed (kept float)."""


class DashQ(nn.Linear):
  pass


class DashK(nn.Linear):
  pass


def gemma_norm(x, w, eps=1e-6):
  y = torch.clamp(x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps), -ACT, ACT)
  return torch.clamp(y * (1.0 + w), -ACT, ACT)


class DashShared(nn.Module):
  """Per-signature tensors shared by the 12 layers."""

  def __init__(self, cfg):
    super().__init__()
    C = cfg.cache
    assert C > IMG + WL
    self.C = C
    cos, sin = page2d_tables(C)
    self.table = RopeTable(C, 2 * DH, bias=False)
    with torch.no_grad():
      self.table.weight.copy_(torch.cat([cos, sin], dim=1).t())
    self.register_buffer("jrow", torch.arange(C, dtype=torch.float32).reshape(1, 1, 1, C), persistent=False)
    self.register_buffer("jcol", torch.arange(C, dtype=torch.float32).reshape(1, C), persistent=False)
    self.register_buffer("thr", (BS * torch.arange(NIMG, dtype=torch.float32) + WL).reshape(1, 1, 1, NIMG),
                         persistent=False)

  def forward(self, input_pos, mask):
    T = input_pos.shape[0]
    posf = input_pos.to(torch.float32).reshape(1, 1, T, 1)
    # the one-hot is built at rank 2 from rank-2 operands: a Linear fed through a reshape of
    # a rank-4 one-hot is folded by the converter into a rank-4 tfl.batch_matmul that fails verification
    # ("found invalid output rank, expected 3 but got 4", litert-torch 0.9.3, rank-4 and rank-3 views both)
    onehot = (self.jcol == input_pos.to(torch.float32).reshape(T, 1)).to(torch.float32)   # [T, C]
    cs = self.table(onehot).reshape(1, 1, T, 2 * DH)                     # [1, 1, T, 256]
    cos, sin = cs[..., :DH], cs[..., DH:]
    causal = (self.jrow <= posf).to(torch.float32)
    local = causal * (self.jrow > (posf - float(WL))).to(torch.float32)  # (p - 128, p]
    keep = (mask > -1.0).to(torch.float32)                               # the runtime's causal gate
    a = keep * (1.0 - local)                                              # token mask = a * full + b
    b = (1.0 - keep) * NEG
    rest = (a * NEG + b)[..., IMG:]                                       # positions >= 288: never a candidate
    cand = (posf >= self.thr).to(torch.float32)                           # [1, 1, T, 9]: b <= p // 32 - 4
    return {"cos": cos, "sin": sin, "a_img": a[..., :IMG], "b_img": b[..., :IMG], "rest": rest, "cand": cand}


def amax_last(x):
  """max over the last axis of a rank-4 x [a, b, c, n] -> [a, b, c, 1], reduced on a rank-3 view: litert-torch
  0.9.3's layout pass takes amax of a rank-4 input for a pooling op and stops ("NHWC node rewriter not found: amax")."""
  a, b, c, n = x.shape
  return torch.amax(x.reshape(a * b, c, n), dim=-1, keepdim=True).reshape(a, b, c, 1)


def entmax15_r4(x):
  """entmax-1.5 over the last axis of x [1, 1, T, 9] -> prior [1, T, 9, 1] (pairwise form, rank 4)."""
  T, n = x.shape[2], x.shape[3]
  X = torch.clamp((x - amax_last(x)) * 0.5, min=X_MIN)
  Xc = X.reshape(1, T, n, 1)
  Xr = X.reshape(1, T, 1, n)
  Xr_f = Xr * torch.ones((1, 1, n, 1), dtype=x.dtype)                   # [1, T, n, n]: X_j along the last axis
  Xc_f = Xc * torch.ones((1, 1, 1, n), dtype=x.dtype)                   # X_i
  G = (Xr_f >= Xc_f).to(x.dtype)                                        # G[.., i, j] = X_j >= X_i
  k = G.sum(-1, keepdim=True)                                           # [1, T, n, 1]
  S1 = (G * Xr_f).sum(-1, keepdim=True)
  S2 = (G * (Xr_f * Xr_f)).sum(-1, keepdim=True)
  mean = S1 / k
  meansq = S2 / k
  delta = (1.0 - k * (meansq - mean * mean)) / k
  tau = mean - torch.sqrt(F.relu(delta))
  ok = (tau <= Xc).to(x.dtype)
  kstar = torch.clamp(amax_last((ok * k).reshape(1, 1, T, n)).reshape(1, T, 1, 1), min=1.0)
  pick = ok * (k == kstar).to(x.dtype)
  tau_star = (tau * pick).sum(2, keepdim=True) / torch.clamp(pick.sum(2, keepdim=True), min=1.0)
  r = F.relu(Xc - tau_star)
  return r * r


class DashAttention(nn.Module):
  def __init__(self, cfg, consts):
    super().__init__()
    self.cfg = cfg
    H, n, d = cfg.hidden, cfg.heads, cfg.head_dim
    assert d == DH and cfg.extra_k_heads == 1 and cfg.kv_heads == n
    self.q_proj = nn.Linear(H, n * d, bias=False)
    self.k_proj = nn.Linear(H, n * d, bias=False)
    self.v_proj = nn.Linear(H, n * d, bias=False)
    self.o_proj = nn.Linear(n * d, H, bias=False)
    self.rq = DashQ(H, DH, bias=False)
    self.rk = DashK(H, DH, bias=False)
    self.rq_norm = nn.Parameter(torch.zeros(DH))
    self.rk_norm = nn.Parameter(torch.zeros(DH))
    self.head_cls = nn.Parameter(torch.zeros(DH))
    self.idx_scale, self.floor, self.inv_sigma = consts["idx_scale"], consts["floor"], 1.0 / consts["sigma"]

  def forward(self, xn, sh, input_pos, kc, vc):
    cfg = self.cfg
    T = xn.shape[1]
    n, d = cfg.heads, cfg.head_dim
    inv = 1.0 / math.sqrt(d)
    cos, sin = sh["cos"], sh["sin"]                                       # [1, 1, T, 128]
    q = self.q_proj(xn).reshape(1, T, n, d).permute(0, 2, 1, 3)
    k = self.k_proj(xn).reshape(1, T, n, d).permute(0, 2, 1, 3)
    v = self.v_proj(xn).reshape(1, T, n, d).permute(0, 2, 1, 3)
    q, k = ud.apply_rope(q, cos, sin), ud.apply_rope(k, cos, sin)
    iq = ud.apply_rope(gemma_norm(self.rq(xn), self.rq_norm).reshape(1, 1, T, DH), cos, sin)
    ik = ud.apply_rope(gemma_norm(self.rk(xn), self.rk_norm).reshape(1, 1, T, DH), cos, sin)
    zero = torch.zeros((1,), dtype=torch.int32)
    p0 = input_pos[:1].to(torch.int32)
    k_idx = torch.cat([zero, zero, p0, zero], dim=0)
    v_idx = torch.cat([zero, zero, zero, p0], dim=0)
    kc = ud.dus(kc, torch.cat([k, ik], dim=1), [x for x in k_idx])     # [1, 11, C, 128]
    vc = ud.dus(vc, v.permute(0, 1, 3, 2), [x for x in v_idx])          # [1, 10, 128, C]
    # head_cls summaries of the 9 image blocks (router keys at positions 0..287, written before any candidate use)
    kk = kc[:, n:n + 1, :IMG, :].reshape(1, NIMG, BS, DH)
    aw = torch.clamp((kk * self.head_cls.reshape(1, 1, 1, DH)).sum(-1, keepdim=True) * inv, -ACT, ACT)
    e = torch.exp(aw - amax_last(aw.reshape(1, 1, NIMG, BS)).reshape(1, NIMG, 1, 1))   # [1, 9, 32, 1]
    w = e / torch.clamp(e.sum(2, keepdim=True), min=1e-6)
    kbar = (w * kk).sum(2, keepdim=True).reshape(1, 1, NIMG, DH)
    z = torch.clamp(ud.bmm4(iq, kbar) * inv, -ACT, ACT)                  # [1, 1, T, 9]
    cand = sh["cand"]
    xs = cand * (z * self.idx_scale) + (1.0 - cand) * ENT_NEG
    prior = entmax15_r4(xs)                                               # [1, T, 9, 1]
    logp = torch.log(torch.clamp(prior, min=self.floor))
    cc = cand.reshape(1, T, NIMG, 1)
    cnt = torch.clamp(cc.sum(2, keepdim=True), min=1.0)
    mu = (cc * logp).sum(2, keepdim=True) / cnt
    bias = cc * ((logp - mu) * self.inv_sigma) + (1.0 - cc) * NEG        # [1, T, 9, 1]
    full = bias                                                           # [1, T, 9, 1] -> 32 copies of each:
    for _ in range(5):                                                    # doubling concatenations (the converter
      full = torch.cat([full, full], dim=-1)                              # folds a MUL with a ones constant into
    full = full.reshape(1, 1, T, IMG)                                     # BROADCAST_TO before a RESHAPE)
    m = torch.cat([sh["a_img"] * full + sh["b_img"], sh["rest"]], dim=-1)  # [1, 1, T, C]
    q11 = torch.cat([q * inv, torch.zeros((1, 1, T, d), dtype=xn.dtype)], dim=1)
    s = ud.bmm4(q11, kc) + m                                              # [1, 11, T, C]
    p = ud.softmax_last(s)
    o = ud.bmm4(p[:, :n], vc)                                             # [1, 10, T, 128]
    o = o.permute(0, 2, 1, 3).reshape(1, T, n * d)
    return self.o_proj(o), kc, vc


class DashLayer(nn.Module):
  def __init__(self, cfg, idx, consts):
    super().__init__()
    self.idx = idx
    self.input_layernorm = nn.Parameter(torch.ones(cfg.hidden))
    self.post_attention_layernorm = nn.Parameter(torch.ones(cfg.hidden))
    self.attn = DashAttention(cfg, consts)
    self.mlp = ud.DenseMLP(cfg.hidden, cfg.dense_inter) if idx < cfg.first_dense else ud.FusedMoE(cfg)
    self.eps = cfg.eps

  def forward(self, h, sh, input_pos, kc, vc):
    a, kc, vc = self.attn(ud.rms_norm(h, self.input_layernorm, self.eps), sh, input_pos, kc, vc)
    h = h + a
    h = h + self.mlp(ud.rms_norm(h, self.post_attention_layernorm, self.eps))
    return h, kc, vc


class DashDecoder(nn.Module):
  def __init__(self, cfg, n_layers=None):
    super().__init__()
    self.cfg = cfg
    consts = dash_consts()
    self.shared = DashShared(cfg)
    self.layers = nn.ModuleList([DashLayer(cfg, i, consts) for i in range(n_layers or cfg.n_layers)])
    self.norm = nn.Parameter(torch.ones(cfg.hidden))
    self.lm_head = ud.LMHead(cfg.hidden, cfg.vocab, bias=False)

  def forward(self, emb, input_pos, mask, kcs, vcs, with_logits, extra_k_fn=None):
    sh = self.shared(input_pos, mask)
    h = emb
    nk, nv = [], []
    for i, layer in enumerate(self.layers):
      h, kc, vc = layer(h, sh, input_pos, kcs[i], vcs[i])
      nk.append(kc)
      nv.append(vc)
    logits = self.lm_head(ud.rms_norm(h, self.norm, self.cfg.eps)) if with_logits else None
    return logits, nk, nv


def load_dash_layer(layer, cfg, i, get, dtype=torch.float32):
  """uocr_decoder.load_layer (norms, q/k/v/o, MLP / MoE) + the DASH router of layer i. Returns the keys read."""
  used = ud.load_layer(layer, cfg, i, get, dtype)
  p = f"model.layers.{i}.self_attn.dash_router."
  a = layer.attn
  with torch.no_grad():
    for dst, key in ((a.rq.weight, "q_proj.weight"), (a.rk.weight, "k_proj.weight"),
                     (a.rq_norm, "q_norm.weight"), (a.rk_norm, "k_norm.weight"), (a.head_cls, "head_cls")):
      t = get(p + key).to(dtype)
      assert tuple(t.shape) == tuple(dst.shape), (key, t.shape, dst.shape)
      dst.copy_(t)
      used.append(p + key)
  return used
