"""kimi_ext.py - a static Kimi-Linear writer for litert-torch: a LiteRT-LM decoder graph with signatures
prefill_<T> and decode, written from Transformers 5.22.2's native `kimi_linear` modeling (the reference it was
checked against) in plain TFLite ops plus the runtime's `moe` custom op.

Graph contract (the LiteRT-LM executor at main 02d7e8d8):
  inputs   tokens int32 [1, T], input_pos int32 [T], mask float32 [1, 1, T, C] (0 = attend, -0.7 * FLT_MAX elsewhere),
           kv_cache_* (below); decode adds nothing else. Output names: kv_cache_* (same names), decode also logits
           float32 [1, 1, V].
  states   KDA layer L: kv_cache_c_L float32 [1, 3, 12288] (the last 3 inputs of the short conv, time-major, channels
             = q | k | v projections) and kv_cache_r_L float32 [1, 32, 128, 128] (the delta-rule state S[h, k, v]);
           MLA layer L: kv_cache_k_L float32 [1, 1, C, 512] (the normalized compressed latent, kv_a_layernorm applied)
             and kv_cache_v_L float32 [1, 1, C, 64] (the shared k_rot part, no norm, no rotary: NoPE). Written at
             input_pos[0] by DYNAMIC_UPDATE_SLICE; pads past the real rows write slots that the mask hides and that
             the next call overwrites (the usual LiteRT-LM cache contract).
  pads     prefill rows past the real ids: the executor writes the pad id into tokens and 0 into input_pos; a test
           harness may write the pad id with continuing positions. A row is real when input_pos[i] == input_pos[0] + i AND
           tokens[i] != PAD_ID (163839 "[PAD]"): either fill marks the pads. Pads get beta = 0 and g = 0 in KDA (the
           state passes through unchanged) and the conv state is gathered at the last real row.

Layers:
  KDA (prefill)  q/k/v projections -> depthwise causal conv (4 taps, by 4 shifted products) -> SiLU -> l2norm(q, k),
                 q * 128^-0.5; g = -exp(A_log) * softplus(f_b(f_a(x)) + dt_bias) (per channel, stable softplus);
                 beta = sigmoid(b_proj(x)); the chunked delta rule (chunk 16): per chunk the pairwise decay
                 exp(min(G_i - G_j, 0)) [c, c, 128] (no exp(-G) factor, which overflows: g reaches -1.7e3 per token),
                 W = (I + diag(beta) A)^-1 by recursive block doubling (masks, no index_put), then a sequential pass
                 over the chunks for the state; RMSNormGated(o, sigmoid(g_b(g_a(x)))) -> o_proj.
  KDA (decode)   the same projections, conv over [state | x], the exact recurrent step.
  MLA            absorbed form: q_lat = q_nope @ W_UK (per head), scores = q_lat . latent + q_rot . k_rot over the
                 cache, o = (softmax @ latent) @ W_UV^T; scale 192^-0.5.
  MoE            router = plain ops: sigmoid(x W_r^T), top-8 chosen on sigmoid + e_score_correction_bias, weights =
                 the UNBIASED sigmoid scores at those experts / their sum (+1e-20); routed_scaling_factor (2.446) rides
                 as the op's per_expert_scale; routed experts = one `moe` op (silu, fp32 experts [I, E, 1, D] /
                 [D, E, 1, I]); shared expert = plain SwiGLU; output = shared + ((r * 0.5) + (r - r * 0.5))
                 (fusion guard 2: the GPU moe kernel must not get an elementwise op fused in).
Parameters are created on the meta device and served by `CheckpointLoader` (Converter V2's weights_loader); the
small constants are buffers with real values (served through weights_loader.buffers_on_meta / with_tensors).
"""
import json
import math
import os

from flatbuffers import flexbuffers
from litert_converter.mlir import ir
from litert_torch.backend import lowerings
from litert_torch.backend.lowerings import utils as lowering_utils
from litert_torch.generative.custom_ops import dynamic_update_slice as tfl_dus
import safetensors
import torch
from torch import nn
import torch.nn.functional as F

PAD_ID = 163839
CHUNK = int(os.environ.get("KIMI_CHUNK", "16"))
FUSION_GUARD = os.environ.get("KIMI_FUSION_GUARD", "2")
# The router weights at the chosen experts. arith (default): a one-hot written as float arithmetic (MoE.route);
# onehot = sum(one_hot(idx) * scores) = EQUAL, CAST, MUL, SUM; gather = torch.gather, which lowers to GATHER_ND
# (+ SELECT_V2 / LESS / SELECT for its indices), which LiteRT's WebGPU delegate refuses ("GATHER_ND: Operation is
# not supported").
ROUTER_GATHER = os.environ.get("KIMI_ROUTER_GATHER", "arith")
# Debugging only: KIMI_DBG = comma list of taps ("attn0", "moe0", "shr0", "w0", "x1": layer 0's attention output,
# routed MoE output, shared-expert output, router weights, layer 1's masked normed input) written as extra
# kv_cache_dbg_<tap> states (prefill: the whole window; decode: its row into the window's last row), for a harness
# that dumps states. Off (empty) for every real export.
# KIMI_EXPERTS=graft: every routed-expert weight of the `moe` op is a 4-element fp32 placeholder holding
# 1000 * layer + 10 * slot (slot 0 = gate = w1, 1 = ff1 = up = w3, 2 = linear = down = w2), the layout
# graft_kimi.py looks for: the export never materializes the experts (47.1B parameters), graft_kimi.py swaps the
# quantized experts in. fp32 (default) = the real experts in fp32 (for small cuts of the model).
EXPERTS = os.environ.get("KIMI_EXPERTS", "fp32")
assert EXPERTS in ("fp32", "graft"), EXPERTS
DBG_TAPS = [x for x in os.environ.get("KIMI_DBG", "").split(",") if x]
DBG_T = int(os.environ.get("KIMI_DBG_T", "128"))  # onehot | gather
assert FUSION_GUARD in ("0", "2") and ROUTER_GATHER in ("gather", "onehot", "arith")


# ----------------------------------------------------------------------------------------------------- the moe op
def moe_custom_options(num_experts, num_active, model_dim, hidden_dim, weight_type):
  return bytes(flexbuffers.Dumps({
      "num_experts": int(num_experts), "num_active_experts": int(num_active), "model_dim": int(model_dim),
      "hidden_dim": int(hidden_dim), "activation": "silu", "weight_type": weight_type,
      "renormalized_top_weights": True}))


def moe_reference(src, top_w, top_i, w_gate, w_ff1, w_lin, per_expert_scale, num_experts, num_active, model_dim,
                  hidden_dim):
  """Eager reference of the op: sum_k pes[e_k] * w_k * (silu(x gate_e^T) * (x ff1_e^T)) lin_e^T."""
  x = src.reshape(-1, model_dim).to(torch.float32)
  w = top_w.reshape(-1, num_active).to(torch.float32)
  idx = top_i.reshape(-1, num_active).to(torch.long)
  out = torch.zeros_like(x)
  if w_gate.numel() != hidden_dim * num_experts * model_dim:
    return out.reshape(src.shape).to(src.dtype)
  g = w_gate.reshape(hidden_dim, num_experts, model_dim)
  u = w_ff1.reshape(hidden_dim, num_experts, model_dim)
  d = w_lin.reshape(model_dim, num_experts, hidden_dim)
  pes = per_expert_scale.reshape(num_experts).to(torch.float32)
  for e in torch.unique(idx).tolist():
    tok, slot = torch.where(idx == e)
    xe = x[tok]
    h = F.silu(xe @ g[:, e, :].T.float()) * (xe @ u[:, e, :].T.float())
    out.index_add_(0, tok, (h @ d[:, e, :].T.float()) * (w[tok, slot] * pes[e]).unsqueeze(-1))
  return out.reshape(src.shape).to(src.dtype)


if not hasattr(torch.ops, "kimi") or not hasattr(torch.ops.kimi, "moe_silu"):

  @torch.library.custom_op("kimi::moe_silu", mutates_args=())
  def moe_silu(src: torch.Tensor, top_w: torch.Tensor, top_i: torch.Tensor, w_gate: torch.Tensor,
               w_ff1: torch.Tensor, w_lin: torch.Tensor, per_expert_scale: torch.Tensor, num_experts: int,
               num_active: int, model_dim: int, hidden_dim: int) -> torch.Tensor:
    return moe_reference(src, top_w, top_i, w_gate, w_ff1, w_lin, per_expert_scale, num_experts, num_active,
                         model_dim, hidden_dim)

  @moe_silu.register_fake
  def _moe_silu_fake(src, top_w, top_i, w_gate, w_ff1, w_lin, per_expert_scale, num_experts, num_active,
                     model_dim, hidden_dim):
    del top_w, top_i, w_gate, w_ff1, w_lin, per_expert_scale, num_experts, num_active, model_dim, hidden_dim
    return torch.empty_like(src)

  @lowerings.lower(torch.ops.kimi.moe_silu)
  def _moe_silu_lower(lctx, src, top_w, top_i, w_gate, w_ff1, w_lin, per_expert_scale, num_experts, num_active,
                      model_dim, hidden_dim):
    opts = moe_custom_options(num_experts, num_active, model_dim, hidden_dim, "fp32")
    op = ir.Operation.create(
        "tfl.custom", results=lowering_utils.node_meta_to_ir_types(lctx.node),
        operands=[src, top_w, top_i, w_gate, w_ff1, w_lin, per_expert_scale],
        attributes={"custom_code": ir.StringAttr.get("moe"),
                    "custom_option": ir.Attribute.parse(f'#tfl<const_bytes: "0x{opts.hex()}">')})
    return op.results[0]


# ------------------------------------------------------------------------------------------------------- helpers
def _param(shape, device):
  return nn.Parameter(torch.empty(*shape, device=device), requires_grad=False)


class Lin(nn.Module):
  """y = x W^T, W [out, in] (no bias anywhere in Kimi-Linear)."""

  def __init__(self, n_in, n_out, device):
    super().__init__()
    self.weight = _param((n_out, n_in), device)

  def forward(self, x):
    return F.linear(x, self.weight)


class RMSNorm(nn.Module):
  def __init__(self, n, eps, device):
    super().__init__()
    self.weight = _param((n,), device)
    self.eps = eps

  def forward(self, x):
    return self.weight * (x * torch.rsqrt((x * x).mean(-1, keepdim=True) + self.eps))


class MLP(nn.Module):
  def __init__(self, d, i, device):
    super().__init__()
    self.gate_proj, self.up_proj, self.down_proj = Lin(d, i, device), Lin(d, i, device), Lin(i, d, device)

  def forward(self, x):
    return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def _neg(x):
  return torch.neg(x)


def _tri(c, strict):
  return [[1.0 if (j < i if strict else j <= i) else 0.0 for j in range(c)] for i in range(c)]


def _doubling_mask(c, s):
  """1 at (i, j) when i is in the lower half and j in the upper half of the same 2s-block."""
  return [[1.0 if (i // (2 * s) == j // (2 * s) and (i // s) % 2 == 1 and (j // s) % 2 == 0) else 0.0
           for j in range(c)] for i in range(c)]


class Consts(nn.Module):
  """Small constants as buffers with real values (row-major python lists, never iota / eye ops)."""

  def __init__(self, prefill_lens, c):
    super().__init__()
    self.c = c
    self.register_buffer("tril_incl", torch.tensor(_tri(c, False)), persistent=False)
    self.register_buffer("tril_strict", torch.tensor(_tri(c, True)), persistent=False)
    self.register_buffer("eye", torch.tensor([[1.0 if i == j else 0.0 for j in range(c)] for i in range(c)]),
                         persistent=False)
    self.levels = []
    s = 1
    while s < c:
      self.register_buffer(f"dbl_{s}", torch.tensor(_doubling_mask(c, s)), persistent=False)
      self.levels.append(s)
      s *= 2
    self.register_buffer("r3", torch.tensor([0.0, 1.0, 2.0]), persistent=False)
    for t in prefill_lens:
      self.register_buffer(f"ar_{t}", torch.tensor(list(range(t)), dtype=torch.int32), persistent=False)
      self.register_buffer(f"p_{t}", torch.tensor([float(i) for i in range(t + 3)]), persistent=False)


# --------------------------------------------------------------------------------------------------------- KDA
def l2norm(x, eps=1e-6):
  return x / torch.sqrt((x * x).sum(-1, keepdim=True) + eps)


def kda_recurrent_step(q, k, v, g, beta, s):
  """q, k, v, g [1, H, 1, D]; beta [1, H, 1, 1]; s [1, H, Dk, Dv] -> o [1, H, 1, D], s'. Rank 4 with a leading 1
  everywhere (ML Drift's BHWC placement)."""
  s = s * torch.exp(g).transpose(2, 3)
  delta = (v - torch.matmul(k, s)) * beta
  s = s + torch.matmul(k.transpose(2, 3), delta)
  return torch.matmul(q, s), s


def kda_chunked(q, k, v, g, beta, s, consts):
  """q, k, v, g [1, H, T, D] (q already scaled); beta [1, H, T, 1]; s [1, H, Dk, Dv] -> o [1, H, T, D], s'.
  T % c == 0. The chunks are laid out chunk-major (m = n * H rows: row r = chunk r // H, head r % H). The intra-chunk
  matrices (the pairwise decay sums A and Bm, the triangular inverse) are rank 3 [m, c, c] and every constant mask
  they meet is rank 2 [c, c], so no elementwise op with a constant changes rank: ML Drift computes a rank-3 runtime
  tensor times a rank-4 constant (the form the converter makes of `sum -> reshape to rank 4 -> * mask`) wrong at fp32
  (found by comparing each op's output with the reference). The sequential
  pass over the chunks slices rank-4 [1, H, c, .] blocks out of [1, m, c, .] (axis 1, never the batch axis)."""
  c = consts.c
  _, h, t, d = q.shape
  n = t // c
  m = n * h

  def chunks(x, last):
    return x.reshape(h, n, c, last).transpose(0, 1).reshape(1, m, c, last)

  qc, kc, vc, gc = chunks(q, d), chunks(k, d), chunks(v, d), chunks(g, d)
  bc = chunks(beta, 1)  # [1, m, c, 1]
  big_g = F.linear(gc.transpose(2, 3), consts.tril_incl).transpose(2, 3)  # cumulative log decay, <= 0
  gam = torch.exp(big_g)
  g0, k0, q0 = big_g.reshape(m, c, d), kc.reshape(m, c, d), qc.reshape(m, c, d)
  diff = g0.unsqueeze(2) - g0.unsqueeze(1)  # [m, c(i), c(j), d] = G_i - G_j
  dec = torch.exp(_neg(F.relu(_neg(diff))))  # exp(min(G_i - G_j, 0))
  a = (k0.unsqueeze(2) * k0.unsqueeze(1) * dec).sum(-1) * consts.tril_strict  # [m, c, c]
  bm = (q0.unsqueeze(2) * k0.unsqueeze(1) * dec).sum(-1) * consts.tril_incl  # [m, c, c]
  b3 = bc.reshape(m, c, 1)
  nmat = _neg(b3 * a)  # strictly lower
  x = consts.eye + getattr(consts, "dbl_1") * nmat
  for lvl in consts.levels[1:]:
    x = x + getattr(consts, f"dbl_{lvl}") * torch.matmul(x, torch.matmul(nmat, x))
  ub = torch.matmul(x, vc.reshape(m, c, d) * b3).reshape(1, m, c, d)
  wk = torch.matmul(x, (kc * bc * gam).reshape(m, c, d)).reshape(1, m, c, d)
  bm = bm.reshape(1, m, c, c)
  qg = qc * gam
  g_last = big_g[:, :, c - 1:c, :]  # [1, m, 1, d]
  kd = (kc * torch.exp(g_last - big_g)).transpose(2, 3)  # [1, m, d, c]
  d_last = torch.exp(g_last).transpose(2, 3)  # [1, m, d, 1]
  outs = []
  for i in range(n):
    sl = slice(i * h, (i + 1) * h)
    u = ub[:, sl] - torch.matmul(wk[:, sl], s)
    outs.append(torch.matmul(qg[:, sl], s) + torch.matmul(bm[:, sl], u))
    s = s * d_last[:, sl] + torch.matmul(kd[:, sl], u)
  return torch.cat(outs, dim=2), s


class KDA(nn.Module):

  def __init__(self, cfg, device):
    super().__init__()
    hs, self.h, self.d = cfg.hidden_size, cfg.linear_num_heads, cfg.linear_head_dim
    qd = self.h * self.d
    self.qd = qd
    self.eps = cfg.rms_norm_eps
    self.q_proj, self.k_proj, self.v_proj = Lin(hs, qd, device), Lin(hs, qd, device), Lin(hs, qd, device)
    self.conv_w = _param((4, 3 * qd), device)  # tap j multiplies input t - 3 + j
    self.f_a_proj, self.f_b_proj = Lin(hs, self.d, device), Lin(self.d, qd, device)
    self.dt_bias = _param((qd,), device)
    # -exp(A_log[h]) repeated over head h's 128 channels, flat [H * D]: a rank-1 per-channel constant on the flat
    # [1, T, H * D] gate. The first form (a [H, 1, D] constant times the runtime [H, T, D], with the rest of KDA at
    # rank 3) gave r_1 cos 0.81 after one decode step on WebGPU; this form with KDA at rank 4 matches. The two
    # changes went in together, so which one fixed decode is not isolated.
    self.neg_decay = _param((qd,), device)
    self.b_proj = Lin(hs, self.h, device)
    self.g_a_proj, self.g_b_proj = Lin(hs, self.d, device), Lin(self.d, qd, device)
    self.o_norm_weight = _param((self.d,), device)
    self.o_proj = Lin(qd, hs, device)

  def _heads(self, x, t):
    return x.reshape(1, t, self.h, self.d).transpose(1, 2)  # [1, T, H*D] -> [1, H, T, D]

  def forward(self, x, conv_state, rec_state, consts, valid=None, n_valid=None):
    t = x.shape[1]
    qkv = torch.cat([self.q_proj(x), self.k_proj(x), self.v_proj(x)], dim=-1)  # [1, T, 3QD]
    xc = torch.cat([conv_state, qkv], dim=1)  # [1, 3 + T, 3QD]
    w = self.conv_w
    conv = (xc[:, 0:t] * w[0] + xc[:, 1:t + 1] * w[1]) + (xc[:, 2:t + 2] * w[2] + xc[:, 3:t + 3] * w[3])
    conv = F.silu(conv)
    if valid is None:
      new_conv = xc[:, 1:4]
    else:
      tgt = n_valid.reshape(1, 1, 1, 1) + consts.r3.reshape(1, 1, 3, 1)  # [1, 1, 3, 1]
      sel = (getattr(consts, f"p_{t}").reshape(1, 1, 1, t + 3) == tgt).to(torch.float32)  # [1, 1, 3, T + 3]
      new_conv = torch.matmul(sel, xc.reshape(1, 1, t + 3, 3 * self.qd)).reshape(1, 3, 3 * self.qd)
    q, k, v = conv[..., :self.qd], conv[..., self.qd:2 * self.qd], conv[..., 2 * self.qd:]
    q = l2norm(self._heads(q, t)) * (self.d ** -0.5)
    k = l2norm(self._heads(k, t))
    v = self._heads(v, t)
    gr = self.f_b_proj(self.f_a_proj(x)) + self.dt_bias  # [1, T, QD]
    sp = F.relu(gr) + torch.log(1.0 + torch.exp(_neg(torch.abs(gr))))  # softplus, stable form
    g = self._heads(self.neg_decay * sp, t)  # [1, H, T, D], <= 0
    beta = torch.sigmoid(self.b_proj(x)).reshape(1, t, self.h).transpose(1, 2).reshape(1, self.h, t, 1)
    if valid is None:
      o, s = kda_recurrent_step(q, k, v, g, beta, rec_state)
    else:
      vm = valid.reshape(1, 1, t, 1)
      o, s = kda_chunked(q, k, v, g * vm, beta * vm, rec_state, consts)
    gate = self._heads(self.g_b_proj(self.g_a_proj(x)), t)
    o = o * torch.rsqrt((o * o).mean(-1, keepdim=True) + self.eps) * self.o_norm_weight * torch.sigmoid(gate)
    o = o.transpose(1, 2).reshape(1, t, self.qd)
    return self.o_proj(o), new_conv, s


# --------------------------------------------------------------------------------------------------------- MLA
class MLA(nn.Module):

  def __init__(self, cfg, device):
    super().__init__()
    hs, self.h = cfg.hidden_size, cfg.num_attention_heads
    self.dn, self.dr, self.dv, self.r = cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.v_head_dim, cfg.kv_lora_rank
    self.scale = (self.dn + self.dr) ** -0.5
    self.q_nope_proj = Lin(hs, self.h * self.dn, device)
    self.q_rot_proj = Lin(hs, self.h * self.dr, device)
    self.kv_c_proj = Lin(hs, self.r, device)
    self.kv_r_proj = Lin(hs, self.dr, device)
    self.kv_norm = RMSNorm(self.r, 1e-6, device)  # KimiLinearRMSNorm(kv_lora_rank): default eps 1e-6
    self.w_uk = _param((1, self.h, self.dn, self.r), device)
    self.w_uv_t = _param((1, self.h, self.r, self.dv), device)
    self.o_proj = Lin(self.h * self.dv, hs, device)

  def forward(self, x, k_cache, v_cache, input_pos, mask):
    t = x.shape[1]
    cache_len = k_cache.shape[2]
    q_nope = self.q_nope_proj(x).reshape(1, t, self.h, self.dn).transpose(1, 2)  # [1, H, T, 128]
    q_rot = self.q_rot_proj(x).reshape(1, t, self.h, self.dr).transpose(1, 2)  # [1, H, T, 64]
    q_lat = torch.matmul(q_nope, self.w_uk)  # [1, H, T, 512]
    lat = self.kv_norm(self.kv_c_proj(x)).reshape(1, 1, t, self.r)
    rot = self.kv_r_proj(x).reshape(1, 1, t, self.dr)
    zero = torch.zeros_like(input_pos[:1])
    idx = [zero[0], zero[0], input_pos[0], zero[0]]
    k_cache = tfl_dus.dynamic_update_slice(k_cache, lat, idx)
    v_cache = tfl_dus.dynamic_update_slice(v_cache, rot, idx)
    s = (torch.matmul(q_lat.reshape(1, 1, self.h * t, self.r), k_cache.transpose(2, 3)) +
         torch.matmul(q_rot.reshape(1, 1, self.h * t, self.dr), v_cache.transpose(2, 3)))
    s = s.reshape(1, self.h, t, cache_len) * self.scale + mask
    a = torch.softmax(s, dim=-1)
    o_lat = torch.matmul(a.reshape(1, 1, self.h * t, cache_len), k_cache).reshape(1, self.h, t, self.r)
    o = torch.matmul(o_lat, self.w_uv_t)  # [1, H, T, 128]
    return self.o_proj(o.transpose(1, 2).reshape(1, t, self.h * self.dv)), k_cache, v_cache


# --------------------------------------------------------------------------------------------------------- MoE
class MoE(nn.Module):

  def __init__(self, cfg, device):
    super().__init__()
    d, i, e = cfg.hidden_size, cfg.moe_intermediate_size, cfg.num_local_experts
    self.d, self.i, self.e, self.k = d, i, e, cfg.num_experts_per_tok
    self.gate_weight = _param((e, d), device)
    self.e_bias = _param((e,), device)
    if EXPERTS == "graft":
      self.w_gate, self.w_ff1, self.w_lin = _param((4,), device), _param((4,), device), _param((4,), device)
    else:
      self.w_gate = _param((i, e, 1, d), device)
      self.w_ff1 = _param((i, e, 1, d), device)
      self.w_lin = _param((d, e, 1, i), device)
    self.per_expert_scale = _param((1, 1, 1, e), device)  # = routed_scaling_factor, every expert
    self.shared = MLP(d, i * cfg.n_shared_experts, device)
    if ROUTER_GATHER != "gather":
      self.register_buffer("e_ar", torch.tensor(list(range(e)), dtype=torch.int32), persistent=False)

  def route(self, x):
    """x [1, T, D] -> weights [1, T, k] (unbiased sigmoid scores at the chosen experts / their sum), idx [T, k] int32.
    The scores stay rank 4 [1, 1, T, E] around the bias add (a rank-2 activation plus a rank-2 constant is a
    measured ML Drift miscompute); TOPK_V2 reads them as [T, E]."""
    t = x.shape[1]
    s = torch.sigmoid(F.linear(x, self.gate_weight)).reshape(1, 1, t, self.e)
    choice = s + self.e_bias.reshape(1, 1, 1, self.e)
    _, idx = torch.topk(choice.reshape(t, self.e), self.k, dim=-1)
    idx = idx.to(torch.int32)
    if ROUTER_GATHER == "gather":
      w = torch.gather(s.reshape(t, self.e), 1, idx.to(torch.long)).reshape(1, t, self.k, 1)
    elif ROUTER_GATHER == "onehot":
      # the converter makes this a ONE_HOT op: ML Drift computes it right for one row only
      hot = (idx.reshape(1, t, self.k, 1) == self.e_ar.reshape(1, 1, 1, self.e)).to(torch.float32)  # [1, T, k, E]
      w = (hot * s.reshape(1, t, 1, self.e)).sum(-1, keepdim=True)  # [1, T, k, 1]
    else:
      # "arith": the one-hot as float arithmetic, 1 - min(|i - e|, 1) (exact: i and e are small integers)
      d = torch.abs(idx.to(torch.float32).reshape(1, t, self.k, 1) - self.e_ar.to(torch.float32).reshape(1, 1, 1, self.e))
      hot = 1.0 - torch.clamp(d, max=1.0)  # [1, T, k, E]
      w = (hot * s.reshape(1, t, 1, self.e)).sum(-1, keepdim=True)  # [1, T, k, 1]
    w = w / (w.sum(2, keepdim=True) + 1e-20)
    return w.reshape(1, t, self.k), idx

  def forward(self, x):
    t = x.shape[1]
    w, idx = self.route(x)
    routed = torch.ops.kimi.moe_silu(x.reshape(1, t, self.d), w.reshape(1, t, self.k), idx.reshape(1, t, self.k),
                                     self.w_gate, self.w_ff1, self.w_lin, self.per_expert_scale,
                                     self.e, self.k, self.d, self.i)
    shared = self.shared(x)
    if DBG_TAPS:
      self.dbg = {"moe": routed, "shr": shared, "w": w}
    if FUSION_GUARD == "2":
      half = routed * 0.5
      return shared + (half + (routed - half))
    return shared + routed


# ------------------------------------------------------------------------------------------------------- model
class Layer(nn.Module):

  def __init__(self, cfg, idx, device):
    super().__init__()
    self.kind = "kda" if cfg.layer_types[idx] == "linear_attention" else "mla"
    self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, device)
    self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, device)
    self.attn = KDA(cfg, device) if self.kind == "kda" else MLA(cfg, device)
    self.sparse = cfg.mlp_layer_types[idx] == "sparse"
    if self.sparse:
      self.moe = MoE(cfg, device)
    else:
      self.mlp = MLP(cfg.hidden_size, cfg.intermediate_size, device)


def prefill_valid(tokens, input_pos, consts):
  """The prefill window's real rows: position continues the first row's (input_pos[i] == input_pos[0] + i; LiteRT-LM's
  executor pads with input_pos 0) and the token is not the pad id (a test harness may pad with continuing positions).
  Returns valid [T] float and n_valid [1] (rows are a prefix: the padding is at the end)."""
  t = tokens.shape[1]
  ar = getattr(consts, f"ar_{t}")
  ok_pos = (input_pos - ar) == input_pos[0:1]
  ok_tok = tokens[0] != PAD_ID
  valid = torch.logical_and(ok_pos, ok_tok).to(torch.float32)  # [T]
  return valid, valid.sum(0, keepdim=True)  # [1]


class KimiStatic(nn.Module):

  def __init__(self, cfg, prefill_lens, device="meta"):
    super().__init__()
    self.cfg = cfg
    self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size, device=device)
    self.embed_tokens.weight.requires_grad_(False)
    self.layers = nn.ModuleList([Layer(cfg, i, device) for i in range(cfg.num_hidden_layers)])
    self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, device)
    self.lm_head = Lin(cfg.hidden_size, cfg.vocab_size, device)
    self.consts = Consts(prefill_lens, CHUNK)

  def state_shapes(self, cache_len):
    out = {}
    for i, layer in enumerate(self.layers):
      if layer.kind == "kda":
        a = layer.attn
        out[f"c_{i}"] = (1, 3, 3 * a.qd)
        out[f"r_{i}"] = (1, a.h, a.d, a.d)
      else:
        out[f"k_{i}"] = (1, 1, cache_len, layer.attn.r)
        out[f"v_{i}"] = (1, 1, cache_len, layer.attn.dr)
    for tap in DBG_TAPS:
      out[f"dbg_{tap}"] = (1, DBG_T, self.cfg.num_experts_per_tok if tap.startswith("w") else self.cfg.hidden_size)
    return out

  def run(self, tokens, input_pos, mask, kv_cache, decode):
    t = tokens.shape[1]
    h = self.embed_tokens(tokens)
    valid = n_valid = None
    if not decode:
      valid, n_valid = prefill_valid(tokens, input_pos, self.consts)
    new = {}
    taps = {}
    for i, layer in enumerate(self.layers):
      x = layer.input_layernorm(h)
      if layer.kind == "kda":
        if valid is not None:
          x = x * valid.reshape(1, t, 1)
        taps[f"x{i}"] = x
        y, new[f"c_{i}"], new[f"r_{i}"] = layer.attn(x, kv_cache[f"c_{i}"], kv_cache[f"r_{i}"], self.consts,
                                                      valid, n_valid)
      else:
        y, new[f"k_{i}"], new[f"v_{i}"] = layer.attn(x, kv_cache[f"k_{i}"], kv_cache[f"v_{i}"], input_pos, mask)
      h = h + y
      taps[f"attn{i}"] = y
      x = layer.post_attention_layernorm(h)
      h = h + (layer.moe(x) if layer.sparse else layer.mlp(x))
      if DBG_TAPS and layer.sparse:
        taps[f"moe{i}"], taps[f"shr{i}"], taps[f"w{i}"] = (layer.moe.dbg["moe"], layer.moe.dbg["shr"],
                                                         layer.moe.dbg["w"])
    zero = torch.zeros_like(input_pos[:1])[0]
    for tap in DBG_TAPS:
      src = kv_cache[f"dbg_{tap}"]
      row = zero if not decode else zero + (DBG_T - 1)
      new[f"dbg_{tap}"] = tfl_dus.dynamic_update_slice(src, taps[tap], [zero, row, zero])
    out = {"kv_cache": new}
    if decode:
      out["logits"] = self.lm_head(self.norm(h))
    return out


class Prefill(nn.Module):

  def __init__(self, model):
    super().__init__()
    self.model = model

  def forward(self, tokens, input_pos, mask, kv_cache):
    return self.model.run(tokens, input_pos, mask, kv_cache, decode=False)


class Decode(nn.Module):

  def __init__(self, model):
    super().__init__()
    self.model = model

  def forward(self, tokens, input_pos, mask, kv_cache):
    out = self.model.run(tokens, input_pos, mask, kv_cache, decode=True)
    return {"logits": out["logits"], "kv_cache": out["kv_cache"]}


# ------------------------------------------------------------------------------------------------------ weights
class CheckpointLoader:
  """FQN (relative to the Prefill / Decode wrapper: `model.<...>`) -> the fp32 tensor in this writer's layout, read
  from a HF Kimi-Linear checkpoint dir (model.safetensors.index.json or one model.safetensors)."""

  def __init__(self, ckpt_dir, cfg):
    self.dir, self.cfg = ckpt_dir, cfg
    idx = os.path.join(ckpt_dir, "model.safetensors.index.json")
    if os.path.exists(idx):
      self.wmap = json.load(open(idx))["weight_map"]
    else:
      with safetensors.safe_open(os.path.join(ckpt_dir, "model.safetensors"), "pt") as f:
        self.wmap = {k: "model.safetensors" for k in f.keys()}
    self.files = {}
    self.served = {}

  def raw(self, name):
    fn = self.wmap[name]
    if fn not in self.files:
      self.files[fn] = safetensors.safe_open(os.path.join(self.dir, fn), "pt")
    return self.files[fn].get_tensor(name)

  def close(self):
    self.files.clear()

  def __call__(self, fqn):
    t = self._load(fqn)
    self.served[fqn] = tuple(t.shape)
    return t

  def _load(self, fqn):
    cfg = self.cfg
    name = fqn[len("model."):] if fqn.startswith("model.") else fqn
    f32 = lambda n: self.raw(n).to(torch.float32)  # noqa: E731
    if name in ("embed_tokens.weight", "norm.weight"):
      return f32("model." + name)
    if name == "lm_head.weight":
      return f32("lm_head.weight")
    parts = name.split(".")
    if parts[0] != "layers":
      raise KeyError(fqn)
    li, rest = int(parts[1]), ".".join(parts[2:])
    p = f"model.layers.{li}."
    if rest in ("input_layernorm.weight", "post_attention_layernorm.weight"):
      return f32(p + rest)
    if rest.startswith("mlp."):
      return f32(p + rest)
    if rest.startswith("attn."):
      a = rest[len("attn."):]
      sa = p + "self_attn."
      direct = {"q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight", "b_proj.weight",
                "f_a_proj.weight", "f_b_proj.weight", "g_a_proj.weight", "g_b_proj.weight", "dt_bias"}
      if cfg.layer_types[li] == "linear_attention":
        if a in direct:
          return f32(sa + a)
        if a == "o_norm_weight":
          return f32(sa + "o_norm.weight")
        if a == "conv_w":
          w = torch.cat([f32(sa + f"{x}_conv1d.weight") for x in "qkv"], dim=0)  # [3QD, 1, 4]
          return w.reshape(w.shape[0], w.shape[-1]).transpose(0, 1).contiguous()  # [4, 3QD]
        if a == "neg_decay":
          al = f32(sa + "A_log").reshape(-1)
          return (-torch.exp(al)).reshape(-1, 1).expand(-1, cfg.linear_head_dim).reshape(-1).contiguous()
      else:
        h, dn, dr, dv, r = (cfg.num_attention_heads, cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.v_head_dim,
                            cfg.kv_lora_rank)
        if a == "q_nope_proj.weight":
          return f32(sa + "q_proj.weight").reshape(h, dn + dr, -1)[:, :dn].reshape(h * dn, -1).contiguous()
        if a == "q_rot_proj.weight":
          return f32(sa + "q_proj.weight").reshape(h, dn + dr, -1)[:, dn:].reshape(h * dr, -1).contiguous()
        if a == "kv_c_proj.weight":
          return f32(sa + "kv_a_proj_with_mqa.weight")[:r].contiguous()
        if a == "kv_r_proj.weight":
          return f32(sa + "kv_a_proj_with_mqa.weight")[r:].contiguous()
        if a == "kv_norm.weight":
          return f32(sa + "kv_a_layernorm.weight")
        if a == "w_uk":
          return f32(sa + "kv_b_proj.weight").reshape(h, dn + dv, r)[:, :dn].contiguous().unsqueeze(0)
        if a == "w_uv_t":
          return f32(sa + "kv_b_proj.weight").reshape(h, dn + dv, r)[:, dn:].transpose(1, 2).contiguous().unsqueeze(0)
        if a == "o_proj.weight":
          return f32(sa + "o_proj.weight")
    if rest.startswith("moe."):
      m = rest[len("moe."):]
      bs = p + "block_sparse_moe."
      e = cfg.num_local_experts
      if m == "gate_weight":
        return f32(bs + "gate.weight")
      if m == "e_bias":
        return f32(bs + "gate.e_score_correction_bias")
      if m == "per_expert_scale":
        return torch.full((1, 1, 1, e), float(cfg.routed_scaling_factor))
      if m.startswith("shared."):
        return f32(bs + "shared_experts." + m[len("shared."):])
      src = {"w_gate": "w1", "w_ff1": "w3", "w_lin": "w2"}.get(m)
      if src and EXPERTS == "graft":
        return torch.full((4,), float(1000 * li + 10 * ("w_gate", "w_ff1", "w_lin").index(m)))
      if src:
        w = torch.stack([self.raw(bs + f"experts.{j}.{src}.weight") for j in range(e)], dim=0)  # [E, out, in]
        o, n = w.shape[1], w.shape[2]
        return w.permute(1, 0, 2).to(torch.float32).contiguous().reshape(o, e, 1, n)
    raise KeyError(fqn)


def materialize(model, loader):
  """Eager use: replace every meta parameter by the loader's tensor (FQN as the wrappers see it: model.<name>)."""
  for mod_name, mod in model.named_modules():
    for pn, p in list(mod._parameters.items()):  # pylint: disable=protected-access
      if p is None:
        continue
      fqn = "model." + (f"{mod_name}.{pn}" if mod_name else pn)
      mod._parameters[pn] = nn.Parameter(loader(fqn), requires_grad=False)  # pylint: disable=protected-access
  return model


def sample_inputs(model, t, cache_len, device):
  shapes = model.state_shapes(cache_len)
  return {
      "tokens": torch.zeros((1, t), dtype=torch.int32, device=device),
      "input_pos": torch.zeros((t,), dtype=torch.int32, device=device),
      "mask": torch.zeros((1, 1, t, cache_len), dtype=torch.float32, device=device),
      "kv_cache": {k: torch.zeros(v, dtype=torch.float32, device=device) for k, v in shapes.items()},
  }


def load_config(ckpt_dir):
  from transformers.models.kimi_linear.configuration_kimi_linear import KimiLinearConfig  # pylint: disable=g-import-not-at-top
  return KimiLinearConfig.from_pretrained(ckpt_dir)
