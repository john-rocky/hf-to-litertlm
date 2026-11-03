"""The Unlimited-OCR-2 text decoder (DeepSeek-V2 type, 12 layers) as a pure-torch module in the LiteRT-LM signature
contract, MoE in form (ii): each MoE layer's 64 routed experts as two fully-connected ops (gate_up, down) with the
routing weights spread over all 64 by one broadcast EQUAL (hf-to-litertlm granitemoe_work/fused_experts.py "bcast").

The attention in this file is a stand-in (plain causal MHA, 10 heads x 128, 1-D RoPE, the runtime's additive mask)
that uocr_dash.py replaces with the model's DASH attention; the MLP / MoE, norms, embedder and lm_head are the model's
own.

Signatures (names and layouts copied from litert-torch 0.9.3 export_hf, external-embedder route, transposed attention;
LiteRT-LM v0.19.0 runtime/executor/litert_compiled_model_executor_utils.cc and litert/state.cc):
  prefill_<T>  embeddings [1,T,H] f32, input_pos [T] i32, mask [1,1,T,C] f32 (0 = allowed, -0.7*FLT_MAX = not),
               kv_cache_k_<i> [1,Hk,C,d], kv_cache_v_<i> [1,Hv,d,C]       -> kv_cache_k_<i>, kv_cache_v_<i>
  decode       the same with T = 1                                         -> logits [1,1,V], kv_cache_k_<i>, kv_cache_v_<i>
  embedder     token_ids [1,1] i32 -> embeddings [1,1,H]  (single_token_embedder)
The runtime prefills all prompt tokens but the last and runs decode on the last one, so prefill has no logits.
Hk = Hv + extra_k_heads: extra heads ride in the K cache (same position axis, same KEY state buffer) for per-token
state such as the DASH router keys; extra_k_heads = 0 is the plain model.

MoE (modeling_deepseekv2 / _sparse_base_impl.py MoEGate + DeepseekV2MoE, eval path):
  scores = softmax(x W_r^T) in fp32 over 64; top-6 (greedy); weights = the top-6 scores (norm_topk_prob false,
  routed_scaling_factor 1.0); y = sum_k w_k expert_k(x) + shared(x), expert = down(silu(gate x) * up x) (896),
  shared = the same MLP at 896 * n_shared = 1792. Layer 0 = dense MLP 6848 (first_k_dense_replace 1).
Fused: W_gu [E*2I, H] = per expert [gate_e; up_e]; W_d [H, E*I] = per expert down_e as a column block.
"""
import dataclasses
import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

HF = os.environ.get("MODEL", "hf")            # the source checkpoint: baidu/Unlimited-OCR-2 at revision 04809669
MASK_NEG = -0.7 * 3.4028234663852886e38      # LiteRT-LM fills a float mask with -0.7 * FLT_MAX


@dataclasses.dataclass
class Cfg:
  n_layers: int = 12
  hidden: int = 1280
  heads: int = 10
  kv_heads: int = 10
  head_dim: int = 128
  vocab: int = 129280
  dense_inter: int = 6848
  moe_inter: int = 896
  n_experts: int = 64
  top_k: int = 6
  n_shared: int = 2
  first_dense: int = 1
  eps: float = 1e-6
  rope_theta: float = 500000.0
  cache: int = 4096
  extra_k_heads: int = 0
  shared_in_experts: bool = False   # lever: the shared expert as two always-on experts of the fused FCs

  @classmethod
  def from_hf(cls, path=os.path.join(HF, "config.json"), **kw):
    c = json.load(open(path))
    base = dict(n_layers=c["num_hidden_layers"], hidden=c["hidden_size"], heads=c["num_attention_heads"],
                kv_heads=c["num_key_value_heads"], head_dim=c["hidden_size"] // c["num_attention_heads"],
                vocab=c["vocab_size"], dense_inter=c["intermediate_size"], moe_inter=c["moe_intermediate_size"],
                n_experts=c["n_routed_experts"], top_k=c["num_experts_per_tok"], n_shared=c["n_shared_experts"],
                first_dense=c["first_k_dense_replace"], eps=c["rms_norm_eps"], rope_theta=c["rope_theta"])
    assert c["scoring_func"] == "softmax" and c["topk_method"] == "greedy" and not c["norm_topk_prob"]
    assert c["routed_scaling_factor"] == 1.0 and c["hidden_act"] == "silu" and c["moe_layer_freq"] == 1
    base.update(kw)
    return cls(**base)


def rms_norm(x, w, eps):
  return w * (x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps))


def softmax_last(x):
  """F.softmax over the last axis, marked as the odml.softmax composite the way export_hf's attention does."""
  from litert_torch.backend import composite
  b = composite.StableHLOCompositeBuilder(name="odml.softmax", attr={"axis": -1})
  x = b.mark_inputs(x)
  return b.mark_outputs(F.softmax(x, dim=-1))


def bmm4(a, b):
  from litert_torch.generative.custom_ops import bmm_4d as bmm_lib
  return bmm_lib.bmm_4d(a, b)


def dus(state, update, idx):
  import litert_torch.generative.custom_ops.dynamic_update_slice as tfl_dus
  return tfl_dus.dynamic_update_slice(state, update, idx)


class Rope(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    d = cfg.head_dim
    inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
    self.register_buffer("inv_freq", inv.reshape(1, d // 2), persistent=False)

  def forward(self, pos):                                     # pos [T] int32
    ang = pos.to(torch.float32).reshape(-1, 1) * self.inv_freq  # [T, d/2]
    emb = torch.cat([ang, ang], dim=-1)
    return torch.cos(emb).reshape(1, 1, -1, emb.shape[-1]), torch.sin(emb).reshape(1, 1, -1, emb.shape[-1])


def apply_rope(x, cos, sin):                                 # x [1, n, T, d]
  h = x.shape[-1] // 2
  rot = torch.cat([-x[..., h:], x[..., :h]], dim=-1)
  return x * cos + rot * sin


class StandInAttention(nn.Module):
  """Stand-in: causal MHA with the runtime's mask, KV written at input_pos[0] by DYNAMIC_UPDATE_SLICE."""

  def __init__(self, cfg):
    super().__init__()
    self.cfg = cfg
    H, n, d = cfg.hidden, cfg.heads, cfg.head_dim
    self.q_proj = nn.Linear(H, n * d, bias=False)
    self.k_proj = nn.Linear(H, cfg.kv_heads * d, bias=False)
    self.v_proj = nn.Linear(H, cfg.kv_heads * d, bias=False)
    self.o_proj = nn.Linear(n * d, H, bias=False)

  def forward(self, x, cos, sin, input_pos, mask, kc, vc, extra_k=None):
    cfg = self.cfg
    T = x.shape[1]
    n, nk, d = cfg.heads, cfg.kv_heads, cfg.head_dim
    q = self.q_proj(x).reshape(1, T, n, d).permute(0, 2, 1, 3)
    k = self.k_proj(x).reshape(1, T, nk, d).permute(0, 2, 1, 3)
    v = self.v_proj(x).reshape(1, T, nk, d).permute(0, 2, 1, 3)
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    if extra_k is not None:                                   # [1, extra, T, d]
      k = torch.cat([k, extra_k], dim=1)
    zero = torch.zeros((1,), dtype=torch.int32)
    p0 = input_pos[:1].to(torch.int32)
    k_idx = torch.cat([zero, zero, p0, zero], dim=0)          # export_hf _get_slice_indices: a [4] tensor whose
    v_idx = torch.cat([zero, zero, zero, p0], dim=0)          # elements go to the op as 0-D scalars
    kc = dus(kc, k, [x for x in k_idx])                       # K [1, Hk, C, d]: time axis 2
    vc = dus(vc, v.permute(0, 1, 3, 2), [x for x in v_idx])   # V [1, Hv, d, C]: time axis 3
    kk = kc if cfg.extra_k_heads == 0 else kc[:, :nk]
    s = bmm4(q * (d ** -0.5), kk) + mask                      # [1, n, T, C]
    p = softmax_last(s)
    o = bmm4(p, vc)                                           # [1, n, T, d]
    o = o.permute(0, 2, 1, 3).reshape(1, T, n * d)
    return self.o_proj(o), kc, vc


class GateUp(nn.Linear):
  pass


class Down(nn.Linear):
  pass


class DenseMLP(nn.Module):
  """down(silu(gate x) * up x) with gate and up as one FC [2I, H]."""

  def __init__(self, hidden, inter):
    super().__init__()
    self.inter = inter
    self.gate_up = GateUp(hidden, 2 * inter, bias=False)
    self.down = Down(inter, hidden, bias=False)

  def forward(self, x):
    gu = self.gate_up(x)
    g, u = gu[..., :self.inter], gu[..., self.inter:]
    return self.down(F.silu(g) * u)


class Router(nn.Linear):
  pass


class ExpertsGateUp(nn.Linear):
  pass


class ExpertsDown(nn.Linear):
  pass


class FusedMoE(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.cfg = cfg
    E, I, H = cfg.n_experts, cfg.moe_inter, cfg.hidden
    self.router = Router(H, E, bias=False)
    self.n_slots = E + (cfg.n_shared if cfg.shared_in_experts else 0)
    self.experts_gu = ExpertsGateUp(H, self.n_slots * 2 * I, bias=False)
    self.experts_down = ExpertsDown(self.n_slots * I, H, bias=False)
    self.register_buffer("expert_ids", torch.arange(E, dtype=torch.int32).reshape(1, E, 1), persistent=False)
    self.shared = None if cfg.shared_in_experts else DenseMLP(H, I * cfg.n_shared)

  def routing(self, x2):                                      # x2 [N, H] -> w [N, E, 1]
    cfg = self.cfg
    scores = F.softmax(self.router(x2), dim=-1)               # fp32 softmax over the 64 experts
    topw, topi = torch.topk(scores, cfg.top_k, dim=-1)
    m = (topi.to(torch.int32).unsqueeze(1) == self.expert_ids).to(x2.dtype)   # [N, 1, K] vs [1, E, 1]
    return (m * topw.unsqueeze(1)).sum(dim=-1, keepdim=True)

  def forward(self, x):                                       # x [1, T, H]
    cfg = self.cfg
    T = x.shape[1]
    x2 = x.reshape(T, cfg.hidden)
    w = self.routing(x2)                                      # [T, E, 1]
    if cfg.shared_in_experts:
      w = torch.cat([w, torch.ones((T, cfg.n_shared, 1), dtype=x2.dtype)], dim=1)
    gu = self.experts_gu(x2).reshape(T, self.n_slots, 2 * cfg.moe_inter)
    g, u = gu[..., :cfg.moe_inter], gu[..., cfg.moe_inter:]
    h = F.silu(g) * u * w                                     # [T, slots, I]
    y = self.experts_down(h.reshape(T, self.n_slots * cfg.moe_inter))
    if self.shared is not None:
      y = y + self.shared(x2)
    return y.reshape(1, T, cfg.hidden)


class Layer(nn.Module):
  def __init__(self, cfg, idx):
    super().__init__()
    self.idx = idx
    self.input_layernorm = nn.Parameter(torch.ones(cfg.hidden))
    self.post_attention_layernorm = nn.Parameter(torch.ones(cfg.hidden))
    self.attn = StandInAttention(cfg)
    self.mlp = DenseMLP(cfg.hidden, cfg.dense_inter) if idx < cfg.first_dense else FusedMoE(cfg)
    self.eps = cfg.eps

  def forward(self, h, cos, sin, input_pos, mask, kc, vc, extra_k=None):
    a, kc, vc = self.attn(rms_norm(h, self.input_layernorm, self.eps), cos, sin, input_pos, mask, kc, vc, extra_k)
    h = h + a
    h = h + self.mlp(rms_norm(h, self.post_attention_layernorm, self.eps))
    return h, kc, vc


class LMHead(nn.Linear):
  pass


class Decoder(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.cfg = cfg
    self.layers = nn.ModuleList([Layer(cfg, i) for i in range(cfg.n_layers)])
    self.norm = nn.Parameter(torch.ones(cfg.hidden))
    self.lm_head = LMHead(cfg.hidden, cfg.vocab, bias=False)
    self.rope = Rope(cfg)

  def forward(self, emb, input_pos, mask, kcs, vcs, with_logits, extra_k_fn=None):
    cos, sin = self.rope(input_pos)
    h = emb
    nk, nv = [], []
    for i, layer in enumerate(self.layers):
      ek = extra_k_fn(i, h, input_pos) if extra_k_fn is not None else None
      h, kc, vc = layer(h, cos, sin, input_pos, mask, kcs[i], vcs[i], ek)
      nk.append(kc)
      nv.append(vc)
    logits = self.lm_head(rms_norm(h, self.norm, self.cfg.eps)) if with_logits else None
    return logits, nk, nv


class Embedder(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.embed_tokens = nn.Embedding(cfg.vocab, cfg.hidden)

  def forward(self, token_ids):
    return {"embeddings": self.embed_tokens(torch.maximum(token_ids, torch.tensor(0, dtype=torch.int32)))}


class Signature(nn.Module):
  """prefill_<T> / decode wrapper with the runtime's flat input and output names."""

  def __init__(self, dec, with_logits, extra_k_fn=None):
    super().__init__()
    self.dec = dec
    self.with_logits = with_logits
    self.extra_k_fn = extra_k_fn

  def forward(self, embeddings, input_pos, mask, **kv):
    L = self.dec.cfg.n_layers
    kcs = [kv[f"kv_cache_k_{i}"] for i in range(L)]
    vcs = [kv[f"kv_cache_v_{i}"] for i in range(L)]
    logits, nk, nv = self.dec(embeddings, input_pos, mask, kcs, vcs, self.with_logits, self.extra_k_fn)
    out = {}
    if logits is not None:
      out["logits"] = logits
    for i in range(L):
      out[f"kv_cache_k_{i}"] = nk[i]
      out[f"kv_cache_v_{i}"] = nv[i]
    return out


def sample_kwargs(cfg, T):
  kv = {}
  for i in range(cfg.n_layers):
    kv[f"kv_cache_k_{i}"] = torch.zeros(1, cfg.kv_heads + cfg.extra_k_heads, cfg.cache, cfg.head_dim)
    kv[f"kv_cache_v_{i}"] = torch.zeros(1, cfg.kv_heads, cfg.head_dim, cfg.cache)
  return {"embeddings": torch.zeros(1, T, cfg.hidden), "input_pos": torch.arange(T, dtype=torch.int32),
          "mask": torch.zeros(1, 1, T, cfg.cache), **kv}


def causal_mask(cfg, pos0, T):
  """The runtime's mask for T tokens at positions pos0..pos0+T-1: 0 where key position <= query position."""
  q = torch.arange(pos0, pos0 + T).reshape(T, 1)
  k = torch.arange(cfg.cache).reshape(1, cfg.cache)
  return torch.where(k <= q, 0.0, MASK_NEG).to(torch.float32).reshape(1, 1, T, cfg.cache)


def export_tflite(dec, emb_mod, cfg, prefills, out_path, emb_path=None, extra_k_fn=None):
  """One TFLite with prefill_<T> for T in prefills + decode; optionally the embedder TFLite."""

  import litert_torch
  conv = None
  for T in prefills:
    sig = Signature(dec, False, extra_k_fn).eval()
    conv = (litert_torch.signature(f"prefill_{T}", sig, sample_kwargs=sample_kwargs(cfg, T)) if conv is None
            else conv.signature(f"prefill_{T}", sig, sample_kwargs=sample_kwargs(cfg, T)))
  sig = Signature(dec, True, extra_k_fn).eval()
  conv = conv.signature("decode", sig, sample_kwargs=sample_kwargs(cfg, 1))
  conv.convert().export(out_path)
  if emb_path is not None:
    litert_torch.signature("embedder", emb_mod.eval(),
                           sample_kwargs={"token_ids": torch.zeros(1, 1, dtype=torch.int32)}).convert().export(emb_path)


# ----------------------------------------------------------------------------------------------- real weights


def hf_tensor_reader(src_dir=HF):
  from safetensors import safe_open
  idx = json.load(open(os.path.join(src_dir, "model.safetensors.index.json")))["weight_map"]
  handles = {}

  def get(key):
    shard = idx[key]
    if shard not in handles:
      handles[shard] = safe_open(os.path.join(src_dir, shard), "pt")
    return handles[shard].get_tensor(key)

  return get, idx


def load_layer(layer, cfg, i, get, dtype=torch.float32):
  """Copies HF layer i into Layer (MLP / MoE, norms, the q/k/v/o of the stand-in attention). Returns the keys read."""
  p = f"model.layers.{i}."
  used = []

  def g(k):
    used.append(p + k)
    return get(p + k).to(dtype)

  with torch.no_grad():
    layer.input_layernorm.copy_(g("input_layernorm.weight"))
    layer.post_attention_layernorm.copy_(g("post_attention_layernorm.weight"))
    for nm in ("q_proj", "k_proj", "v_proj", "o_proj"):
      getattr(layer.attn, nm).weight.copy_(g(f"self_attn.{nm}.weight"))
    if i < cfg.first_dense:
      layer.mlp.gate_up.weight.copy_(torch.cat([g("mlp.gate_proj.weight"), g("mlp.up_proj.weight")], 0))
      layer.mlp.down.weight.copy_(g("mlp.down_proj.weight"))
    else:
      m = layer.mlp
      m.router.weight.copy_(g("mlp.gate.weight"))
      I = cfg.moe_inter
      gu, dn = [], []
      for e in range(cfg.n_experts):
        gu += [g(f"mlp.experts.{e}.gate_proj.weight"), g(f"mlp.experts.{e}.up_proj.weight")]
        dn.append(g(f"mlp.experts.{e}.down_proj.weight"))           # [H, I]
      sg, su, sd = g("mlp.shared_experts.gate_proj.weight"), g("mlp.shared_experts.up_proj.weight"), \
          g("mlp.shared_experts.down_proj.weight")
      if cfg.shared_in_experts:
        for s in range(cfg.n_shared):
          gu += [sg[s * I:(s + 1) * I], su[s * I:(s + 1) * I]]
          dn.append(sd[:, s * I:(s + 1) * I])
      else:
        m.shared.gate_up.weight.copy_(torch.cat([sg, su], 0))
        m.shared.down.weight.copy_(sd)
      m.experts_gu.weight.copy_(torch.cat(gu, 0))
      m.experts_down.weight.copy_(torch.cat(dn, 1))
  return used
