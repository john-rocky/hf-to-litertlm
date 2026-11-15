"""The Unlimited-OCR (v1) text decoder in the LiteRT-LM signature contract, with the model's own decode written
statically for one page: plain multi-head attention (10 heads x 128, no MLA, no GQA), 1-D RoPE, and a KV cache of
P pinned slots + a ring of 128.

The source (baidu/Unlimited-OCR, modeling_deepseekv2.py SlidingWindowLlamaAttention, `mha_eager`; infer() sets
config._ring_window = sliding_window_size = 128 and config.sliding_window = None, modeling_unlimitedocr.py 993-995):
  prefill   full causal attention over the prompt; the cache length after it is P (the prompt length)
  decode    appends while the cache is shorter than P + 128, then overwrites the oldest generated slot (P + ring_pos)
            and attends over the whole cache with no mask
so the query at position p >= P reads the keys 0..P-1 and max(P, p - 127)..p, and every RoPE angle is the true
position p (position_ids = attention_mask.cumsum - 1, modeling_unlimitedocr.py 731-737).
Here the K / V caches hold SLOTS = P + 128 slots:
  slot(p) = p                       for p < P    (the prompt: pinned)
  slot(p) = P + (p - P) mod 128     for p >= P   (the ring)
computed in the graph from input_pos[0] (p - 128 * floor(relu(p - P) / 128) in float, exact below 2^24, cast to int32
= the DYNAMIC_UPDATE_SLICE index). Visible: pinned slot j iff j <= p and the runtime's causal gate (mask[..., :P] > -1);
ring slot j (= P + r) iff j <= p. A ring slot holds the latest position q <= p with q = j (mod 128), so the visible
ring slots hold exactly max(P, p - 127)..p. The keys are in the source's own cache order (pinned, then ring slots),
so every query sees the same keys as the source's cache in the same order.
RoPE: cos / sin of p by a one-hot [T, CTX] FULLY_CONNECTED on a constant [CTX, 256] table (class RopeTable, kept
float by quant_decoder.py): no SIN / COS with large arguments in the graph (XNNPACK's sin / cos lose accuracy there).
The table is transformers' LlamaRotaryEmbedding (default rope type, attention_scaling 1.0) with the same float ops.
Scores = (q . k) / sqrt(128) + mask, softmax in fp32, as the source's _attn_forward. "Not visible" inside the graph
is NEG = -3e4 (exp underflows to 0 in an fp32 softmax as the source's finfo.min does; inside the fp16 range); the
runtime mask is only compared (> -1).
The MLP / MoE (form (ii): the 64 routed experts as two FCs), norms, embedder and lm_head are those of
../unlimitedocr2_work/uocr_decoder.py, loaded from that folder (the v1 MoE is the same MoEGate + DeepseekV2MoE eval
path: softmax, greedy top-6, norm_topk_prob false, routed_scaling_factor 1.0, 2 shared experts).
The prefill signature: LiteRT-LM (read in v0.19.1: GetOptimizedPrefillWorkGroups, PrefillInternal) splits the P prompt
ids into prefill calls over the signatures it has, and in each call it holds the call's last id back for the next call
or for decode. With a signature of P + 1 = 278 rows the one-page prompt is one call at step 0: P - 1 = 276 rows of the
prompt and 2 padding rows (input_pos 0, mask closed; they land in slots 276 and 277, which decode writes before it
reads them), the last id goes to decode at position 276. A signature of P - 1 = 276 rows splits the prompt into two
calls, and the graph writes all T rows of a call from slot(input_pos[0]): the second call (position 275 plus 275
padding rows) runs past the 405 slots, and the reply came out garbled from the first token. A prefill that starts at
p0 >= P or crosses P is outside this contract (a second turn is outside the fixed layout already).
"""
import importlib.util
import json
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.join(os.path.dirname(HERE), "unlimitedocr2_work")   # the shared uocr_decoder.py


def _load(name):
  # The module keeps its own name: the exported tensor names carry it (uocr_decoder.Embedder, ...).
  if name in sys.modules:
    return sys.modules[name]
  spec = importlib.util.spec_from_file_location(name, os.path.join(V2, f"{name}.py"))
  mod = importlib.util.module_from_spec(spec)
  sys.modules[name] = mod
  spec.loader.exec_module(mod)
  return mod


ud = _load("uocr_decoder")

PROMPT_LEN = 277        # P: BOS + 273 <image> + 'document parsing.' (3 ids: 34030, 76466, 16)
RING = 128              # infer()'s _ring_window = config.sliding_window_size
NEG = -3.0e4            # "not visible" inside the graph (fp16 range)

# DeepseekV2Config defaults for the keys the v1 config.json does not carry (configuration_deepseek_v2.py 118-158)
V1_DEFAULTS = {"rope_theta": 10000.0, "rms_norm_eps": 1e-6, "scoring_func": "softmax", "norm_topk_prob": False,
               "routed_scaling_factor": 1.0, "hidden_act": "silu", "moe_layer_freq": 1, "rope_scaling": None,
               "attention_bias": False}


def cfg_from_hf(path=None, **kw):
  """ud.Cfg from the v1 config.json (the language_config block; the top level repeats it): keys absent there take
  the DeepseekV2Config defaults. Asserts the MoE / attention facts this module relies on."""
  path = path or os.path.join(ud.HF, "config.json")
  top = json.load(open(path))
  c = dict(V1_DEFAULTS)
  c.update(top.get("language_config", top))
  assert c["use_mla"] is False and c["num_key_value_heads"] == c["num_attention_heads"]
  assert c["scoring_func"] == "softmax" and c["topk_method"] == "greedy" and not c["norm_topk_prob"]
  assert float(c["routed_scaling_factor"]) == 1.0 and c["hidden_act"] == "silu" and c["moe_layer_freq"] == 1
  assert c["rope_scaling"] is None and not c["attention_bias"] and int(c["sliding_window_size"]) == RING
  base = dict(n_layers=c["num_hidden_layers"], hidden=c["hidden_size"], heads=c["num_attention_heads"],
              kv_heads=c["num_key_value_heads"], head_dim=c["hidden_size"] // c["num_attention_heads"],
              vocab=c["vocab_size"], dense_inter=c["intermediate_size"], moe_inter=c["moe_intermediate_size"],
              n_experts=c["n_routed_experts"], top_k=c["num_experts_per_tok"], n_shared=c["n_shared_experts"],
              first_dense=c["first_k_dense_replace"], eps=float(c["rms_norm_eps"]),
              rope_theta=float(c["rope_theta"]))
  base.update(kw)
  return ud.Cfg(**base)


def rope_tables(C, theta, d=128):
  """cos / sin [C, d] for the positions 0..C-1: transformers LlamaRotaryEmbedding, default rope type
  (_compute_default_rope_parameters + forward, the same float ops: fp32 inv_freq, a K = 1 matmul for the angles,
  cat(freqs, freqs), cos / sin in fp32, attention_scaling 1.0)."""
  inv_freq = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.int64).to(dtype=torch.float) / d))
  pos = torch.arange(C, dtype=torch.long)[None, None, :].float()
  freqs = (inv_freq[None, :, None].float() @ pos).transpose(1, 2)          # [1, C, d/2]
  emb = torch.cat((freqs, freqs), dim=-1)
  return emb.cos()[0], emb.sin()[0]


def ring_slot_f(input_pos, P, ring=RING):
  """[1] float32: slot of input_pos[0]."""
  p0 = input_pos[:1].to(torch.float32)
  return p0 - float(ring) * torch.floor(F.relu(p0 - float(P)) * (1.0 / ring))


class RopeTable(nn.Linear):
  """one-hot [.., C] -> cat(cos, sin) [.., 256]; weight = the [C, 256] table transposed (kept float)."""


class V1Shared(nn.Module):
  """Per-signature tensors shared by the 12 layers: cos / sin, the [T, P + 128] mask, the slot index."""

  def __init__(self, cfg, P=PROMPT_LEN, ring=RING):
    super().__init__()
    C, d = cfg.cache, cfg.head_dim
    assert C > P + ring
    self.P, self.ring, self.d = P, ring, d
    cos, sin = rope_tables(C, cfg.rope_theta, d)
    self.table = RopeTable(C, 2 * d, bias=False)
    with torch.no_grad():
      self.table.weight.copy_(torch.cat([cos, sin], dim=1).t())
    self.register_buffer("jcol", torch.arange(C, dtype=torch.float32).reshape(1, C), persistent=False)
    self.register_buffer("jpin", torch.arange(P, dtype=torch.float32).reshape(1, 1, 1, P), persistent=False)
    self.register_buffer("jring", torch.arange(ring, dtype=torch.float32).reshape(1, 1, 1, ring) + float(P),
                         persistent=False)

  def forward(self, input_pos, mask):
    T = input_pos.shape[0]
    P, d = self.P, self.d
    posf = input_pos.to(torch.float32).reshape(1, 1, T, 1)
    # the one-hot is built at rank 2 from rank-2 operands (a rank-4 one-hot fed to a Linear through a reshape is
    # folded by litert-torch 0.9.3 into a rank-4 tfl.batch_matmul that fails verification)
    onehot = (self.jcol == input_pos.to(torch.float32).reshape(T, 1)).to(torch.float32)   # [T, C]
    cs = self.table(onehot).reshape(1, 1, T, 2 * d)
    cos, sin = cs[..., :d], cs[..., d:]
    causal = (self.jpin <= posf).to(torch.float32)                       # [1, 1, T, P]
    keep = (mask[..., :P] > -1.0).to(torch.float32)                      # the runtime's causal gate, pinned part
    vis_pin = causal * keep
    vis_ring = (self.jring <= posf).to(torch.float32)                    # [1, 1, T, 128]
    m = torch.cat([(1.0 - vis_pin) * NEG, (1.0 - vis_ring) * NEG], dim=-1)   # [1, 1, T, P + 128]
    slot = ring_slot_f(input_pos, P, self.ring).to(torch.int32)          # [1]
    return {"cos": cos, "sin": sin, "mask": m, "slot": slot}


class V1Attention(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.cfg = cfg
    H, n, d = cfg.hidden, cfg.heads, cfg.head_dim
    assert cfg.kv_heads == n and cfg.extra_k_heads == 0
    self.q_proj = nn.Linear(H, n * d, bias=False)
    self.k_proj = nn.Linear(H, n * d, bias=False)
    self.v_proj = nn.Linear(H, n * d, bias=False)
    self.o_proj = nn.Linear(n * d, H, bias=False)
    self.scale = math.sqrt(d)

  def forward(self, xn, sh, kc, vc):
    cfg = self.cfg
    T = xn.shape[1]
    n, d = cfg.heads, cfg.head_dim
    cos, sin = sh["cos"], sh["sin"]                                      # [1, 1, T, 128]
    q = self.q_proj(xn).reshape(1, T, n, d).permute(0, 2, 1, 3)
    k = self.k_proj(xn).reshape(1, T, n, d).permute(0, 2, 1, 3)
    v = self.v_proj(xn).reshape(1, T, n, d).permute(0, 2, 1, 3)
    q, k = ud.apply_rope(q, cos, sin), ud.apply_rope(k, cos, sin)
    zero = torch.zeros((1,), dtype=torch.int32)
    s0 = sh["slot"]
    k_idx = torch.cat([zero, zero, s0, zero], dim=0)                     # export_hf _get_slice_indices: a [4] tensor
    v_idx = torch.cat([zero, zero, zero, s0], dim=0)                     # whose elements go to the op as 0-D scalars
    kc = ud.dus(kc, k, [x for x in k_idx])                               # K [1, 10, P + 128, 128]: slot axis 2
    vc = ud.dus(vc, v.permute(0, 1, 3, 2), [x for x in v_idx])           # V [1, 10, 128, P + 128]: slot axis 3
    s = ud.bmm4(q, kc) / self.scale + sh["mask"]                         # [1, 10, T, P + 128]
    p = ud.softmax_last(s)
    o = ud.bmm4(p, vc)                                                   # [1, 10, T, 128]
    o = o.permute(0, 2, 1, 3).reshape(1, T, n * d)
    return self.o_proj(o), kc, vc


class V1Layer(nn.Module):
  def __init__(self, cfg, idx):
    super().__init__()
    self.idx = idx
    self.input_layernorm = nn.Parameter(torch.ones(cfg.hidden))
    self.post_attention_layernorm = nn.Parameter(torch.ones(cfg.hidden))
    self.attn = V1Attention(cfg)
    self.mlp = ud.DenseMLP(cfg.hidden, cfg.dense_inter) if idx < cfg.first_dense else ud.FusedMoE(cfg)
    self.eps = cfg.eps

  def forward(self, h, sh, kc, vc):
    a, kc, vc = self.attn(ud.rms_norm(h, self.input_layernorm, self.eps), sh, kc, vc)
    h = h + a
    h = h + self.mlp(ud.rms_norm(h, self.post_attention_layernorm, self.eps))
    return h, kc, vc


class V1Decoder(nn.Module):
  def __init__(self, cfg, n_layers=None, P=PROMPT_LEN, ring=RING):
    super().__init__()
    self.cfg, self.P, self.ring = cfg, P, ring
    self.slots = P + ring
    self.shared = V1Shared(cfg, P, ring)
    self.layers = nn.ModuleList([V1Layer(cfg, i) for i in range(n_layers or cfg.n_layers)])
    self.norm = nn.Parameter(torch.ones(cfg.hidden))
    self.lm_head = ud.LMHead(cfg.hidden, cfg.vocab, bias=False)

  def forward(self, emb, input_pos, mask, kcs, vcs, with_logits):
    sh = self.shared(input_pos, mask)
    h = emb
    nk, nv = [], []
    for i, layer in enumerate(self.layers):
      h, kc, vc = layer(h, sh, kcs[i], vcs[i])
      nk.append(kc)
      nv.append(vc)
    logits = self.lm_head(ud.rms_norm(h, self.norm, self.cfg.eps)) if with_logits else None
    return logits, nk, nv


def sample_kwargs(cfg, T, slots, n_layers=None):
  """Signature inputs: K / V of `slots` slots, the mask of cfg.cache (= CTX) columns."""
  kv = {}
  for i in range(n_layers or cfg.n_layers):
    kv[f"kv_cache_k_{i}"] = torch.zeros(1, cfg.kv_heads, slots, cfg.head_dim)
    kv[f"kv_cache_v_{i}"] = torch.zeros(1, cfg.kv_heads, cfg.head_dim, slots)
  return {"embeddings": torch.zeros(1, T, cfg.hidden), "input_pos": torch.arange(T, dtype=torch.int32),
          "mask": torch.zeros(1, 1, T, cfg.cache), **kv}


class Sig(nn.Module):
  """prefill_<T> / decode wrapper with the runtime's flat input and output names."""

  def __init__(self, dec, with_logits, n_layers):
    super().__init__()
    self.dec, self.with_logits, self.L = dec, with_logits, n_layers

  def forward(self, embeddings, input_pos, mask, **kv):
    kcs = [kv[f"kv_cache_k_{i}"] for i in range(self.L)]
    vcs = [kv[f"kv_cache_v_{i}"] for i in range(self.L)]
    logits, nk, nv = self.dec(embeddings, input_pos, mask, kcs, vcs, self.with_logits)
    out = {} if logits is None else {"logits": logits}
    for i in range(self.L):
      out[f"kv_cache_k_{i}"] = nk[i]
      out[f"kv_cache_v_{i}"] = nv[i]
    return out


def export_tflite(dec, emb_mod, cfg, prefills, out_path, emb_path=None):
  import litert_torch
  L = len(dec.layers)
  conv = None
  for T in prefills:
    sig = Sig(dec, False, L).eval()
    kw = sample_kwargs(cfg, T, dec.slots, L)
    conv = (litert_torch.signature(f"prefill_{T}", sig, sample_kwargs=kw) if conv is None
            else conv.signature(f"prefill_{T}", sig, sample_kwargs=kw))
  conv = conv.signature("decode", Sig(dec, True, L).eval(), sample_kwargs=sample_kwargs(cfg, 1, dec.slots, L))
  conv.convert().export(out_path)
  if emb_path is not None:
    litert_torch.signature("embedder", emb_mod.eval(),
                           sample_kwargs={"token_ids": torch.zeros(1, 1, dtype=torch.int32)}).convert().export(emb_path)


def build(ctx=4096, n_layers=None, with_embedder=False, P=PROMPT_LEN, ring=RING, rope_theta=None):
  """The v1 decoder with the checkpoint's weights (MODEL env = the checkpoint dir). Returns cfg, dec, emb, keys."""
  kw = {"cache": ctx}
  if rope_theta is not None:
    kw["rope_theta"] = float(rope_theta)
  cfg = cfg_from_hf(**kw)
  get, _ = ud.hf_tensor_reader()
  dec = V1Decoder(cfg, n_layers, P, ring).eval()
  used = []
  for i, layer in enumerate(dec.layers):
    used += ud.load_layer(layer, cfg, i, get)
  with torch.no_grad():
    dec.norm.copy_(get("model.norm.weight").float())
    dec.lm_head.weight.copy_(get("lm_head.weight").float())
  used += ["model.norm.weight", "lm_head.weight"]
  emb = None
  if with_embedder:
    emb = ud.Embedder(cfg).eval()
    with torch.no_grad():
      emb.embed_tokens.weight.copy_(get("model.embed_tokens.weight").float())
    used.append("model.embed_tokens.weight")
  return cfg, dec, emb, used


def runtime_mask(cfg, pos0, T, valid=None):
  """The runtime's mask rows for T tokens at pos0..pos0+T-1: 0 on [0, step], -0.7 * FLT_MAX elsewhere; rows past
  `valid` (prefill pad rows) all hidden."""
  m = ud.causal_mask(cfg, pos0, T)
  if valid is not None and valid < T:
    m[:, :, valid:, :] = ud.MASK_NEG
  return m
