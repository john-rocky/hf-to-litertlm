"""Dory (nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16, looped nGPT) for litert-torch export_hf.

Every formula follows vLLM PR #60149 at head 55076950b6c155f384d6766d63563c3bc20a0e3f:
`vllm/model_executor/models/dory.py` (cited as vllm:<line>) and
`vllm/transformers_utils/configs/dory.py` (cited as vllm-cfg:<line>).

Two forwards share the same modules and weights:
  * DoryForCausalLM.forward: the export graph. The recurrent loops are unrolled, each KV slot is
    a LiteRTLMCache layer (DUS update, transposed attention over the whole cache, the runtime's
    causal mask plus the in-graph sliding mask), driven by export_hf's prefill / decode modules.
  * DoryEagerReference: the same block math in eager torch with a plain position-indexed cache
    per slot and explicit causal / sliding masks (the form of vLLM's ReferenceAttention,
    tests/models/test_dory.py:25-50), fed chunk by chunk like the runtime.

KV slot layout (both forwards, see kv_slot_plan):
  per_loop:  input attention blocks, then the recurrent attention blocks of loop 0, loop 1, ...,
             then the output attention blocks; every (loop, block) has its own slot (vllm:108-132).
  last_loop: one slot per attention block. Every loop writes this chunk's K/V at the chunk's
             positions before it attends, so loop i reads loop-i K/V for this chunk and last-loop
             K/V for earlier chunks (vllm:151-154); the cache keeps the last loop's K/V.
"""

import dataclasses
import math
import types

import torch
from torch import nn
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict
import transformers
from transformers.configuration_utils import PreTrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

EPS = 1e-6  # vllm:34


@strict
class DoryConfig(PreTrainedConfig):
  """vllm-cfg:9-137 in the transformers-5 form: annotated fields, __post_init__, @strict on the
  subclass. PreTrainedConfig's __init_subclass__ makes every subclass a dataclass, and its class
  validators are collected on the base (no 'mlp' layer type there); @strict here collects this
  class's validate_* overrides, as vLLM's DoryConfig does."""

  model_type = "dory"
  default_theta = 10_000.0

  vocab_size: int = 131072
  hidden_size: int = 2560
  num_hidden_layers: int = 72
  intermediate_size: int = 7168
  num_attention_heads: int = 16
  num_key_value_heads: int = 4
  head_dim: int = 256
  max_position_embeddings: int = 8192
  tie_word_embeddings: bool = False
  hidden_act: str = "silu"
  initializer_range: float = 0.02
  sliding_window: int | None = 1024
  rope_parameters: dict | None = None
  layer_types: list[str] | None = None
  ngpt_alpha_init: float = 0.05
  attention_output_gate_logit_scale_init: float | None = None
  mlp_bias: bool = False
  attention_bias: bool = False
  n_input_layers: int = 0
  n_recurrent_layers: int | None = None
  n_output_layers: int = 0
  n_recurrent_loops: int = 1
  recurrent_kv_cache_mode: str = "per_loop"

  def __post_init__(self, **kwargs):
    if self.layer_types is None:  # vllm-cfg:41-49
      self.layer_types = [
          "mlp" if i % 2 else "full_attention" if i % 8 == 6 else "sliding_attention"
          for i in range(self.num_hidden_layers)
      ]
    if self.rope_parameters is None:  # vllm-cfg:50-58
      self.rope_parameters = {
          kind: {"rope_type": "default", "partial_rotary_factor": 1.0}
          for kind in ("sliding_attention", "full_attention") if kind in self.layer_types
      }
    if self.n_recurrent_layers is None:  # vllm-cfg:59-62
      self.n_recurrent_layers = self.num_hidden_layers - self.n_input_layers - self.n_output_layers
    super().__post_init__(**kwargs)

  def convert_rope_params_to_dict(self, **kwargs):
    """Keep the per-layer-type dict as written: the base adds a top-level rope_theta to it and then
    standardizes it as one global dict (transformers modeling_rope_utils.py:706-727, :745-748)."""
    kwargs.pop("rope_scaling", None)
    kwargs.pop("rope_theta", None)
    return kwargs

  def validate_layer_type(self):  # vllm-cfg:65-76
    if self.layer_types is None or any(
        kind not in {"sliding_attention", "full_attention", "mlp"} for kind in self.layer_types):
      raise ValueError("layer_types must be a list of sliding_attention, full_attention, or mlp")
    if len(self.layer_types) != self.num_hidden_layers:
      raise ValueError("layer_types must have num_hidden_layers entries")

  def validate_architecture(self):  # vllm-cfg:78-112
    parts = (self.n_input_layers, self.n_recurrent_layers, self.n_output_layers)
    if any(n < 0 for n in parts) or sum(parts) != self.num_hidden_layers:
      raise ValueError("Dory input, recurrent, and output layers must sum to num_hidden_layers")
    if self.n_recurrent_loops < 1:
      raise ValueError("n_recurrent_loops must be a positive integer")
    if self.recurrent_kv_cache_mode not in ("per_loop", "last_loop"):
      raise ValueError("recurrent_kv_cache_mode must be per_loop or last_loop")
    if self.num_attention_heads % self.num_key_value_heads:
      raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
    if self.initializer_range <= 0:
      raise ValueError("initializer_range must be positive")
    if "sliding_attention" in self.layer_types and (self.sliding_window is None or self.sliding_window <= 0):
      raise ValueError("sliding_window must be a positive integer")
    if self.attention_bias or self.mlp_bias:
      raise ValueError("Dory requires bias-free projections")
    if self.hidden_act != "silu" or self.tie_word_embeddings:
      raise ValueError("Dory requires SiLU and untied embeddings")

  def validate_rope(self):  # vllm-cfg:114-137 (this port implements partial_rotary_factor 1.0 only)
    attention_types = set(self.layer_types) - {"mlp"}
    if self.rope_parameters is None or (
        set(self.rope_parameters) - {"sliding_attention", "full_attention"}
        or attention_types - self.rope_parameters.keys()):
      raise ValueError("rope_parameters must define each attention block type")
    for rope in self.rope_parameters.values():
      if not isinstance(rope, dict) or rope.get("rope_type", "default") != "default":
        raise ValueError("Dory requires default RoPE")
      if float(rope.get("partial_rotary_factor", 1.0)) != 1.0:
        raise ValueError("this port implements partial_rotary_factor 1.0 only")

  def theta_for(self, layer_type):
    return float(self.rope_parameters[layer_type].get("rope_theta", self.default_theta))


def kv_slot_plan(config):
  """-> list of (block index, loop index or None) in slot order; see the module docstring."""
  n_in, n_rec = config.n_input_layers, config.n_recurrent_layers
  attn = [i for i, t in enumerate(config.layer_types) if t != "mlp"]
  inp = [i for i in attn if i < n_in]
  rec = [i for i in attn if n_in <= i < n_in + n_rec]
  out = [i for i in attn if i >= n_in + n_rec]
  plan = [(i, None) for i in inp]
  if config.recurrent_kv_cache_mode == "per_loop":
    for loop in range(config.n_recurrent_loops):
      plan += [(i, loop) for i in rec]
  else:
    plan += [(i, None) for i in rec]
  plan += [(i, None) for i in out]
  return plan


def kv_slot_lookup(config):
  """-> dict (block index, loop index) -> slot; input / output blocks are keyed with loop 0."""
  lookup = {}
  for slot, (block, loop) in enumerate(kv_slot_plan(config)):
    if loop is None:
      for l in range(config.n_recurrent_loops):
        lookup[(block, l)] = slot
    else:
      lookup[(block, loop)] = slot
  return lookup


def _acc(x):
  """vLLM's .float() accumulation; an fp64 reference run stays in fp64."""
  return x if x.dtype == torch.float64 else x.float()


def justnorm(x, dim=-1):
  """vllm:37-41: x * rsqrt(sum(x^2) + eps), accumulated in fp32."""
  xf = _acc(x)
  y = xf * torch.rsqrt((xf * xf).sum(dim=dim, keepdim=True) + EPS)
  return y.to(x.dtype)


def slerp_alpha(a, b, alpha, alpha_scale):
  """vllm:44-50: justnorm(ya + |alpha * alpha_scale| * (yb - ya)), ya / yb = justnorm(a / b)."""
  lr = (_acc(alpha) * alpha_scale).abs()
  ya, yb = _acc(justnorm(a)), _acc(justnorm(b))
  return justnorm(ya + lr * (yb - ya)).to(a.dtype)


def rotate_half(x):
  x1, x2 = x.chunk(2, dim=-1)
  return torch.cat((-x2, x1), dim=-1)


def rope_cos_sin(position_ids, head_dim, theta, dtype):
  """Neox-style RoPE over the whole head (vllm:101-106, partial_rotary_factor 1.0).
  inv_freq is built from python scalars, never from a module buffer (a buffer computed in
  __init__ is zeroed by the transformers-5.x meta-device load)."""
  acc = torch.float64 if dtype == torch.float64 else torch.float32
  inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=acc) / head_dim))
  freqs = position_ids.to(acc)[..., None] * inv_freq  # [..., t, hd/2]
  emb = torch.cat((freqs, freqs), dim=-1)
  return emb.cos().to(dtype), emb.sin().to(dtype)


class DoryAttention(nn.Module):
  """vllm:53-161: GQA with nGPT qk-norm after RoPE, sigmoid output gate, justnorm before o_proj."""

  def __init__(self, config):
    super().__init__()
    self.num_heads = config.num_attention_heads
    self.num_kv_heads = config.num_key_value_heads
    self.head_dim = config.head_dim
    self.num_key_value_groups = self.num_heads // self.num_kv_heads  # read by transposed_attention
    q_size, kv_size = self.num_heads * self.head_dim, self.num_kv_heads * self.head_dim
    h = config.hidden_size
    self.q_proj = nn.Linear(h, q_size, bias=False)
    self.k_proj = nn.Linear(h, kv_size, bias=False)
    self.v_proj = nn.Linear(h, kv_size, bias=False)
    self.g_proj = nn.Linear(h, q_size, bias=False)
    self.o_proj = nn.Linear(q_size, h, bias=False)
    self.sqk = nn.Parameter(torch.empty(q_size))
    self.s_gate = nn.Parameter(torch.empty(q_size))
    std = config.initializer_range
    self._sqk_factor = self.head_dim ** 0.5 / std  # vllm:94
    g0 = config.attention_output_gate_logit_scale_init
    if g0 is None:
      g0 = config.hidden_size ** 0.5  # vllm:95-97
    self._s_gate_factor = g0 ** 0.5 / std  # vllm:98

  def qk_scale(self):
    """(sqk * f)^2 per (head, dim): multiplies justnorm(q) (vllm:144-147)."""
    return (_acc(self.sqk) * self._sqk_factor).square()

  def gate_scale(self):
    """(s_gate * f)^2 per channel: multiplies g_proj(x) before the sigmoid (vllm:157)."""
    return (_acc(self.s_gate) * self._s_gate_factor).square()

  def project(self, x, cos, sin):
    """x [b, t, h] -> q [b, H, t, hd] (scaled, normalized), k [b, KV, t, hd] (normalized), v, gate."""
    b, t, _ = x.shape
    hd = self.head_dim
    q = self.q_proj(x).view(b, t, self.num_heads, hd).transpose(1, 2)
    k = self.k_proj(x).view(b, t, self.num_kv_heads, hd).transpose(1, 2)
    v = self.v_proj(x).view(b, t, self.num_kv_heads, hd).transpose(1, 2)
    gate = self.g_proj(x)
    q = q * cos + rotate_half(q) * sin  # RoPE first, then the nGPT norm (vllm:139-149)
    k = k * cos + rotate_half(k) * sin
    q = self.qk_scale().to(q.dtype).view(1, self.num_heads, 1, hd) * justnorm(q)
    k = justnorm(k)
    return q, k, v, gate

  def finish(self, attn_out, gate):
    """attn_out [b, t, H*hd] -> o_proj(justnorm(attn_out * sigmoid(gate * s))) (vllm:156-161)."""
    out = attn_out * torch.sigmoid(gate * self.gate_scale().to(gate.dtype))
    out = justnorm(out)
    return self.o_proj(out)


class DoryMLP(nn.Module):
  """vllm:164-196: SwiGLU with nGPT suv scaling and justnorm before down_proj."""

  def __init__(self, config):
    super().__init__()
    h, f = config.hidden_size, config.intermediate_size
    self.gate_proj = nn.Linear(h, f, bias=False)
    self.up_proj = nn.Linear(h, f, bias=False)
    self.down_proj = nn.Linear(f, h, bias=False)
    self.suv_gate = nn.Parameter(torch.empty(f))
    self.suv_up = nn.Parameter(torch.empty(f))
    self._suv_factor = h ** 0.5  # vllm:186

  def forward(self, x):
    g = self.gate_proj(x) * (_acc(self.suv_gate) * self._suv_factor).to(x.dtype)
    u = self.up_proj(x) * (_acc(self.suv_up) * self._suv_factor).to(x.dtype)
    m = F.silu(g) * u
    return self.down_proj(justnorm(m))


class DoryBlock(nn.Module):
  """vllm:199-228: one physical block = attention OR MLP, then the slerp residual."""

  def __init__(self, config, layer_idx):
    super().__init__()
    self.layer_type = config.layer_types[layer_idx]
    self.is_attention = self.layer_type != "mlp"
    self._alpha_scale = config.ngpt_alpha_init / config.initializer_range  # vllm:207
    if self.is_attention:
      self.mixer = DoryAttention(config)
      self.attn_alpha = nn.Parameter(torch.empty(config.hidden_size))
    else:
      self.mixer = DoryMLP(config)
      self.mlp_alpha = nn.Parameter(torch.empty(config.hidden_size))

  def residual(self, h, out):
    alpha = self.attn_alpha if self.is_attention else self.mlp_alpha
    return slerp_alpha(h, out, alpha, self._alpha_scale)


class DoryBackbone(nn.Module):

  def __init__(self, config):
    super().__init__()
    self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
    self.layers = nn.ModuleList([DoryBlock(config, i) for i in range(config.num_hidden_layers)])


class DoryPreTrainedModel(transformers.PreTrainedModel):
  config_class = DoryConfig
  base_model_prefix = "backbone"
  _no_split_modules = ["DoryBlock"]
  _can_compile_fullgraph = True
  _supports_attention_backend = True

  def _init_weights(self, module):
    # transformers' guarded init: a tensor loaded from the checkpoint (_is_hf_initialized) is kept.
    from transformers import initialization as init  # pylint: disable=g-import-not-at-top
    std = self.config.initializer_range
    if isinstance(module, (nn.Linear, nn.Embedding)):
      init.normal_(module.weight, std=std)
    for name in ("sqk", "s_gate", "suv_gate", "suv_up", "attn_alpha", "mlp_alpha", "logit_scale"):
      p = getattr(module, name, None)
      if isinstance(p, nn.Parameter):
        init.constant_(p, std)


class DoryForCausalLM(DoryPreTrainedModel, transformers.GenerationMixin):
  """Module names = the checkpoint's tensor names (backbone.*, lm_head.weight, lm_head.logit_scale)."""

  def __init__(self, config):
    super().__init__(config)
    self.backbone = DoryBackbone(config)
    self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    self.lm_head.logit_scale = nn.Parameter(torch.empty(config.vocab_size))
    self.post_init()

  def get_input_embeddings(self):
    return self.backbone.embeddings

  def set_input_embeddings(self, value):
    self.backbone.embeddings = value

  def get_output_embeddings(self):
    return self.lm_head

  def logits(self, h):
    """vllm:323-329: lm_head(h) * logit_scale / initializer_range."""
    scale = (_acc(self.lm_head.logit_scale) / self.config.initializer_range).to(h.dtype)
    return self.lm_head(h) * scale

  def run_blocks(self, h, attend):
    """vllm:254-279. `attend(block_idx, loop_idx, mixer, x)` returns the attention sublayer output."""
    c = self.config
    n_in, n_rec = c.n_input_layers, c.n_recurrent_layers
    layers = self.backbone.layers

    def block(i, x, loop):
      layer = layers[i]
      out = attend(i, loop, layer.mixer, x) if layer.is_attention else layer.mixer(x)
      return layer.residual(x, out)

    for i in range(n_in):
      h = block(i, h, 0)
    input_h = h
    state = input_h
    for loop in range(c.n_recurrent_loops):
      h = justnorm(state + input_h)
      for i in range(n_in, n_in + n_rec):
        h = block(i, h, loop)
      state = justnorm(h)
    h = state
    for i in range(n_in + n_rec, c.num_hidden_layers):
      h = block(i, h, 0)
    return h

  def forward(
      self,
      input_ids=None,
      inputs_embeds=None,
      position_ids=None,
      past_key_values=None,
      cache_position=None,
      attention_mask=None,
      use_cache=True,
      k_ts_idx=2,
      v_ts_idx=3,
      **kwargs,
  ):
    """The export forward (export_hf's prefill / decode exportables call it)."""
    del cache_position, use_cache
    from litert_torch.generative.export_hf.core import attention as lrt_attn  # pylint: disable=g-import-not-at-top

    assert past_key_values is not None, "the export forward needs the LiteRT cache"
    assert attention_mask is not None, "the export forward needs the runtime mask"
    c = self.config
    h = self.backbone.embeddings(input_ids) if inputs_embeds is None else inputs_embeds
    masks = attention_mask if isinstance(attention_mask, dict) else {
        "full_attention": attention_mask, "sliding_attention": attention_mask}
    rope = {}
    for kind in set(c.layer_types) - {"mlp"}:
      cos, sin = rope_cos_sin(position_ids, c.head_dim, c.theta_for(kind), h.dtype)
      rope[kind] = (cos[:, None], sin[:, None])  # [1, 1, t, hd]
    lookup = kv_slot_lookup(c)

    def attend(i, loop, mixer, x):
      kind = c.layer_types[i]
      cos, sin = rope[kind]
      b, t, _ = x.shape
      q, k, v, gate = mixer.project(x, cos, sin)
      k_all, v_all = past_key_values.layers[lookup[(i, loop)]].update(k, v)
      out, _ = lrt_attn.transposed_attention(
          mixer, q, k_all, v_all, masks[kind], scaling=None,
          k_ts_idx=k_ts_idx, v_ts_idx=v_ts_idx)  # [b, t, H, hd]
      return mixer.finish(out.reshape(b, t, -1), gate)

    h = self.run_blocks(h, attend)
    return CausalLMOutputWithPast(logits=self.logits(h), past_key_values=past_key_values)


class DoryEagerReference:
  """E: eager torch, a position-indexed cache per KV slot, explicit masks (vLLM test form).

  step(ids, positions) runs one chunk (the runtime's prefill chunk or one decode token) and
  returns the logits of every position in the chunk.
  """

  def __init__(self, model, max_len, dtype=torch.float32):
    self.m = model
    c = model.config
    self.c = c
    self.dtype = dtype
    self.lookup = kv_slot_lookup(c)
    n_slots = len(kv_slot_plan(c))
    shape = (c.num_key_value_heads, max_len, c.head_dim)
    self.k = [torch.zeros(shape, dtype=dtype) for _ in range(n_slots)]
    self.v = [torch.zeros(shape, dtype=dtype) for _ in range(n_slots)]
    self.written = [torch.zeros(max_len, dtype=torch.bool) for _ in range(n_slots)]

  @torch.no_grad()
  def step(self, ids, positions):
    c = self.c
    ids = torch.as_tensor(ids, dtype=torch.long).view(1, -1)
    pos = torch.as_tensor(positions, dtype=torch.long).view(1, -1)
    h = self.m.backbone.embeddings(ids).to(self.dtype)
    t = ids.shape[1]
    g = c.num_attention_heads // c.num_key_value_heads

    def attend(i, loop, mixer, x):
      kind = c.layer_types[i]
      cos, sin = rope_cos_sin(pos, c.head_dim, c.theta_for(kind), self.dtype)
      q, k, v, gate = mixer.project(x, cos[:, None], sin[:, None])  # [1, H|KV, t, hd]
      s = self.lookup[(i, loop)]
      p = pos[0]
      self.k[s][:, p] = k[0]  # write this chunk's K/V first (last_loop: overwrites loop i-1)
      self.v[s][:, p] = v[0]
      self.written[s][p] = True
      kpos = torch.arange(self.k[s].shape[1])
      allowed = (kpos[None, :] <= p[:, None]) & self.written[s][None, :]  # [t, S]
      if kind == "sliding_attention":
        allowed &= kpos[None, :] > (p[:, None] - c.sliding_window)  # tests/models/test_dory.py:50
      kh = self.k[s].repeat_interleave(g, dim=0)  # [H, S, hd]: head j reads KV head j // g
      vh = self.v[s].repeat_interleave(g, dim=0)
      scores = torch.einsum("htd,hsd->hts", q[0], kh) / math.sqrt(c.head_dim)
      scores = scores.masked_fill(~allowed[None], float("-inf"))
      out = torch.einsum("hts,hsd->htd", torch.softmax(scores, dim=-1), vh)  # [H, t, hd]
      out = out.transpose(0, 1).reshape(1, t, -1)
      return mixer.finish(out, gate)

    h = self.m.run_blocks(h, attend)
    return self.m.logits(h)[0]


def install_export_hooks(strip_start_token):
  """Register Dory with transformers' Auto classes and the export_hf hooks it needs.

  * DoryLoopCache: the KV slots of kv_slot_plan (export_hf's default cache builds one slot per
    num_hidden_layers entry and asserts on 'mlp'). It returns a plain LiteRTLMCache: torch.export
    flattens the cache by the exact pytree-registered type (the Nanbeige 4.2 pattern).
  * Executor metadata: export_hf lists state buffers from config.layer_types (physical blocks,
    'mlp' raises ValueError); the hook hands it the slot view instead (one entry per slot).
  * strip_start_token: the vendor tokenizer has add_bos_token false and the template emits no
    <s>, while export_hf sets start_token whenever tokenizer.bos_token is set.
  """
  from litert_torch.generative.export_hf.core import cache as lrt_cache  # pylint: disable=g-import-not-at-top
  from litert_torch.generative.export_hf.core import cache_base as lrt_cache_base  # pylint: disable=g-import-not-at-top
  from litert_torch.generative.export_hf.core import litert_lm_builder as lrt_builder  # pylint: disable=g-import-not-at-top
  from litert_torch.generative.export_hf.model_ext import metadata_builder as lrt_meta  # pylint: disable=g-import-not-at-top

  transformers.AutoConfig.register("dory", DoryConfig, exist_ok=True)
  transformers.AutoModelForCausalLM.register(DoryConfig, DoryForCausalLM, exist_ok=True)

  class DoryLoopCache(lrt_cache.LiteRTLMCache):

    @classmethod
    def create_from_config(cls, model_config, export_config, **kwargs):
      layers = [
          lrt_cache.LiteRTLMCacheLayer.create_from_config(model_config, block, export_config, **kwargs)
          for block, _ in kv_slot_plan(model_config)
      ]
      return lrt_cache.LiteRTLMCache(layers)

  lrt_cache_base.CACHE_REGISTRY["DoryLoopCache"] = DoryLoopCache

  orig_exec = lrt_meta.get_executor_metadata_builder

  def exec_builder(model_config):
    base = orig_exec(model_config)
    if base is None or getattr(model_config, "model_type", None) != "dory":
      return base

    def build(sma, export_config, ema, executor_metadata=None):
      plan = kv_slot_plan(model_config)
      view = types.SimpleNamespace(
          layer_types=[model_config.layer_types[b] for b, _ in plan],
          num_hidden_layers=len(plan),
          sliding_window=model_config.sliding_window,
          num_kv_shared_layers=0,
      )
      return base(dataclasses.replace(sma, model_config=view), export_config, ema, executor_metadata)

    return build

  lrt_meta.get_executor_metadata_builder = exec_builder

  if strip_start_token:
    orig_llm = lrt_builder.build_llm_metadata

    def llm_builder(*args, **kwargs):
      md = orig_llm(*args, **kwargs)
      md.ClearField("start_token")
      return md

    lrt_builder.build_llm_metadata = llm_builder
  return "DoryLoopCache"
