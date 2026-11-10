"""Out-of-tree litert-torch export_hf extension for Qwen3.5 MoE (`qwen3_5_moe`, `qwen3_5_moe_text`).

Load with `python -m litert_torch.generative.export_hf ... --extension_modules=<this file>` (litert-torch main,
`model_ext/extension.py`: a standalone .py path registers the subclasses that carry `model_types`).

What it adds on top of litert-torch main's built-in Qwen3.5 extension (`model_ext/qwen3_5/`, dense only):
  * `Qwen35MoeStaticForCausalLM`, a subclass of `Qwen3_5StaticForCausalLM` whose decoder layers keep the upstream
    Gated DeltaNet / full-attention blocks, norms, rope and state handling unchanged and replace the dense MLP with
    the Qwen3.5 MoE block (router + 256 routed experts top-8 + a sigmoid-gated shared expert). The upstream
    `Qwen3_5StaticModelHFWrapper` takes a `Qwen3_5StaticForCausalLM` instance as is, so the upstream exportables,
    patches (fusions / composites), cache and signatures apply unchanged.
  * The routed experts as one TFLite custom op `moe` per layer (the runtime's MoE kernel: XNNPACK on CPU, ML Drift
    on GPU), with `activation = "silu"` (litert-torch's own `moe` lowering writes `gelu_tanh`).
  * The router as plain ops in the order the runtime kernels were checked with (FC -> SOFTMAX -> TOPK_V2 -> SUM ->
    DIV -> RESHAPE), and the shared expert as a plain SwiGLU MLP times sigmoid(shared_expert_gate(x)).

Environment switches (read at import):
  Q35MOE_EXPERTS = fp32 (default): the experts' weights are fp32 constants of the `moe` op (7 inputs), laid out in
                   the op's order at load time (`resolve_checkpoint_tensor`) from the checkpoint's fused
                   `mlp.experts.gate_up_proj` / `mlp.experts.down_proj`.
                   graft: the `moe` op gets three 4-element placeholder constants per layer (value = 1000 * layer +
                   10 * slot) instead of the expert weights; graft_moe.py later swaps in the quantized experts
                   (int4 / int8, the #11309 form) read straight from the bf16 checkpoint. No expert weight is ever
                   materialized in fp32.
  Q35MOE_FUSION_GUARD = 2 (default): write `shared + ((routed * 0.5) + (routed - routed * 0.5))`
                   so the moe output has two consumers (MUL and SUB) and no elementwise op is fused into the GPU moe
                   kernel (with the residual ADD fused into it, ML Drift writes invalid WGSL for the kernel and the
                   WebGPU run prints `Validation error`; ../qwen3_30b_a3b_work), with no op whose two inputs are the
                   same tensor (a GPU delegate refused those: "SUB: No support of few identical inputs in the same
                   operation"). Exact in fp32: r * 0.5 is exact,
                   r - r / 2 = r / 2 exactly (Sterbenz), r / 2 + r / 2 = r exactly. 1 = an earlier form
                   `shared + (routed - (routed - routed))` (CPU only: the GPU delegate refuses SUB(r, r)).
                   0 = plain `shared + routed`.
  Q35MOE_SHARED_3D = 0 (default; the CPU form): the shared expert and its gate run on the flattened [T, H] input, as
                   HF does. 1 (the GPU form): they run on the rank-3 [1, T, H] input, like the
                   dense Qwen3.5 MLP, and the product is flattened after the gate. In export_hf's GPU form the shared
                   expert is an `odml.swiglu` composite after a fused gate_up FC; with the flattened input that FC
                   writes [T, 1024], ML Drift reads a rank-2 [T, C] as batch T, and building the prefill graphs stops
                   with "Unable to parse bc coord for BATCH axis. Layout is bhwc, args are X, Y, S" (WebGPU with LiteRT
                   bf49eb62, shape {128, 1, 1, 1024}).
"""

import os

from flatbuffers import flexbuffers
from litert_converter.mlir import ir
from litert_torch.backend import lowerings
from litert_torch.backend.lowerings import utils as lowering_utils
from litert_torch.generative.export_hf.model_ext import extension as extension_lib
from litert_torch.generative.export_hf.model_ext.qwen3_5 import exportable_module as q35_exportable
from litert_torch.generative.export_hf.model_ext.qwen3_5 import modeling_qwen3_5_static as q35_static
# Registers the upstream Qwen3.5 patches (they already name qwen3_5_moe / qwen3_5_moe_text).
from litert_torch.generative.export_hf.model_ext.qwen3_5 import patch as _q35_patch  # noqa: F401
import torch
from torch import nn
import torch.nn.functional as F
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention
from transformers.models.qwen3_5.modular_qwen3_5 import Qwen3_5MLP
from transformers.models.qwen3_5.modular_qwen3_5 import Qwen3_5RMSNorm

EXPERT_MODE = os.environ.get("Q35MOE_EXPERTS", "fp32")
FUSION_GUARD = os.environ.get("Q35MOE_FUSION_GUARD", "2")
assert FUSION_GUARD in ("0", "1", "2"), FUSION_GUARD
SHARED_3D = os.environ.get("Q35MOE_SHARED_3D", "0")
assert SHARED_3D in ("0", "1"), SHARED_3D
# None = upstream behaviour (the Qwen3.5 patch sets the GDN short-conv composite on unless extra_kwargs say
# otherwise); "0" / "1" = force it off / on after the upstream patch ran (the patch reads only extra_kwargs, not the
# export flag of the same name).
SHORT_CONV_COMPOSITE = os.environ.get("Q35MOE_SHORT_CONV_COMPOSITE")
if EXPERT_MODE not in ("fp32", "graft"):
  raise ValueError(f"Q35MOE_EXPERTS must be fp32 or graft, got {EXPERT_MODE!r}")

PLACEHOLDER_LEN = 4

# "0" (default): the prefill `gated_delta_update` op gets the 6 inputs that LiteRT's kernels before 38f1db8e90 take
# (the CPU `GatedDeltaUpdateCustomOpKernel::Run` refuses `inputs.size() != 6`); the 7th input `valid_len` came with
# litert-torch f740471 (10-28) and LiteRT 38f1db8e90 (10-28, "Skip ... right-padded tokens"). Without it the right pads
# are zeroed in the graph instead (q, k, v, beta, g times the valid mask: beta = g = 0 leaves the state unchanged, the
# conv state is still gathered at the last real token) = upstream's own `zero_padded_inputs` path for valid_len None.
# "1": upstream's 7-input form (the pads are skipped inside the kernel; needs a runtime with LiteRT 38f1db8e90+).
GDN_VALID_LEN = os.environ.get("Q35MOE_GDN_VALID_LEN", "0") == "1"

from litert_torch.generative.export_hf.model_ext.qwen3_5 import gated_delta_rule as _q35_gdr  # pylint: disable=g-import-not-at-top

_upstream_apply_gated_delta_rule = _q35_gdr.apply_gated_delta_rule


def _apply_gated_delta_rule_6in(query, key, value, beta, g, recurrent_state, valid_mask=None, valid_len=None, **kw):
  if not GDN_VALID_LEN and valid_len is not None and valid_mask is not None and valid_mask.numel() > 0:
    b, s = query.shape[0], query.shape[1]
    vm4 = valid_mask.view(b, s, 1, 1).to(query.dtype)
    vm3 = valid_mask.view(b, s, 1).to(beta.dtype)
    query, key, value = query * vm4, key * vm4, value * vm4
    beta, g = beta * vm3, g * vm3.to(g.dtype)
    valid_len = None
  return _upstream_apply_gated_delta_rule(query, key, value, beta, g, recurrent_state, valid_mask=valid_mask,
                                          valid_len=valid_len, **kw)


# modeling_qwen3_5_static and gated_delta_rule's own block path both call it through the module attribute / global.
_q35_gdr.apply_gated_delta_rule = _apply_gated_delta_rule_6in

_upstream_apply_patches = _q35_patch.apply_qwen3_5_model_patches


def _apply_patches_with_lane_flags(model, export_config):
  replaced = _upstream_apply_patches(model, export_config)
  if SHORT_CONV_COMPOSITE is not None:
    for module in model.modules():
      if hasattr(module, "use_short_conv_composite"):
        module.use_short_conv_composite = SHORT_CONV_COMPOSITE == "1"
  return replaced


# q35_exportable calls `qwen3_5_patch.apply_qwen3_5_model_patches` through the module attribute.
_q35_patch.apply_qwen3_5_model_patches = _apply_patches_with_lane_flags


def moe_custom_options(num_experts, num_active, model_dim, hidden_dim, weight_type):
  """The `moe` op's flexbuffer map (keys as litert-torch writes them, activation silu)."""
  return bytes(
      flexbuffers.Dumps({
          "num_experts": int(num_experts),
          "num_active_experts": int(num_active),
          "model_dim": int(model_dim),
          "hidden_dim": int(hidden_dim),
          "activation": "silu",
          "weight_type": weight_type,
          "renormalized_top_weights": True,
      })
  )


def _reference(src, top_w, top_i, w_gate, w_ff1, w_lin, per_expert_scale, num_experts, num_active, model_dim,
               hidden_dim):
  """Eager reference of the op (OHWI weights [out, E, 1, in]); loops over the experts that are hit."""
  x = src.reshape(-1, model_dim).to(torch.float32)
  w = top_w.reshape(-1, num_active).to(torch.float32)
  idx = top_i.reshape(-1, num_active).to(torch.long)
  out = torch.zeros_like(x)
  if w_gate.numel() != hidden_dim * num_experts * model_dim:
    return out.reshape(src.shape).to(src.dtype)  # graft placeholders: no expert weights here
  g = w_gate.reshape(hidden_dim, num_experts, model_dim)
  u = w_ff1.reshape(hidden_dim, num_experts, model_dim)
  d = w_lin.reshape(model_dim, num_experts, hidden_dim)
  pes = per_expert_scale.reshape(num_experts).to(torch.float32)
  for e in torch.unique(idx).tolist():
    tok, slot = torch.where(idx == e)
    xe = x[tok]
    h = F.silu(xe @ g[:, e, :].T.to(torch.float32)) * (xe @ u[:, e, :].T.to(torch.float32))
    y = h @ d[:, e, :].T.to(torch.float32)
    out.index_add_(0, tok, y * (w[tok, slot] * pes[e]).unsqueeze(-1))
  return out.reshape(src.shape).to(src.dtype)


if not hasattr(torch.ops, "q35moe") or not hasattr(torch.ops.q35moe, "moe_silu"):

  @torch.library.custom_op("q35moe::moe_silu", mutates_args=())
  def moe_silu(src: torch.Tensor, top_w: torch.Tensor, top_i: torch.Tensor, w_gate: torch.Tensor,
               w_ff1: torch.Tensor, w_lin: torch.Tensor, per_expert_scale: torch.Tensor, num_experts: int,
               num_active: int, model_dim: int, hidden_dim: int) -> torch.Tensor:
    return _reference(src, top_w, top_i, w_gate, w_ff1, w_lin, per_expert_scale, num_experts, num_active,
                      model_dim, hidden_dim)

  @moe_silu.register_fake
  def _moe_silu_fake(src, top_w, top_i, w_gate, w_ff1, w_lin, per_expert_scale, num_experts, num_active,
                     model_dim, hidden_dim):
    del top_w, top_i, w_gate, w_ff1, w_lin, per_expert_scale, num_experts, num_active, model_dim, hidden_dim
    return torch.empty_like(src)

  @lowerings.lower(torch.ops.q35moe.moe_silu)
  def _moe_silu_lower(lctx, src, top_w, top_i, w_gate, w_ff1, w_lin, per_expert_scale, num_experts, num_active,
                      model_dim, hidden_dim):
    opts = moe_custom_options(num_experts, num_active, model_dim, hidden_dim, "fp32")
    op = ir.Operation.create(
        "tfl.custom",
        results=lowering_utils.node_meta_to_ir_types(lctx.node),
        operands=[src, top_w, top_i, w_gate, w_ff1, w_lin, per_expert_scale],
        attributes={
            "custom_code": ir.StringAttr.get("moe"),
            "custom_option": ir.Attribute.parse(f'#tfl<const_bytes: "0x{opts.hex()}">'),
        },
    )
    return op.results[0]


def _load_layout_module():
  """q35moe_ext_layout.py by path (this file is itself loaded by path through --extension_modules; no sys.path
  insert)."""
  import importlib.util  # pylint: disable=g-import-not-at-top
  path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "q35moe_ext_layout.py")
  spec = importlib.util.spec_from_file_location("q35moe_ext_layout", path)
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


_layout = _load_layout_module()
OHWI_SUFFIXES = _layout.OHWI_SUFFIXES
ohwi_from_checkpoint = _layout.ohwi_from_checkpoint


class Qwen35MoeExperts(nn.Module):
  """Routed experts. fp32 mode: parameters `w_gate` / `w_ff1` [I, E, 1, D] and `w_lin` [D, E, 1, I], already in the
  op's layout. They are filled by `Qwen35MoeExtension.resolve_checkpoint_tensor` from the checkpoint's
  `mlp.experts.gate_up_proj` [E, 2I, D] / `down_proj` [E, D, I] (slice + permute in torch at load time): any torch op
  on the expert weights inside forward becomes a constant fold in the converter (F0 2026-11-09 19:37: TFL::SliceOp::fold
  on the 268M-element halves of gate_up_proj ran on one core for 7+ min while the footprint grew past 43 GB)."""

  def __init__(self, config, layer_idx):
    super().__init__()
    self.num_experts = config.num_experts
    self.num_active = config.num_experts_per_tok
    self.model_dim = config.hidden_size
    self.hidden_dim = config.moe_intermediate_size
    self.layer_idx = layer_idx
    e, d, i = self.num_experts, self.model_dim, self.hidden_dim
    if EXPERT_MODE == "fp32":
      self.w_gate = nn.Parameter(torch.empty(i, e, 1, d))
      self.w_ff1 = nn.Parameter(torch.empty(i, e, 1, d))
      self.w_lin = nn.Parameter(torch.empty(d, e, 1, i))
    else:
      for slot, name in enumerate(("ph_gate", "ph_ff1", "ph_lin")):
        self.register_buffer(name, torch.full((PLACEHOLDER_LEN,), float(1000 * layer_idx + 10 * slot)),
                             persistent=False)
    self.register_buffer("per_expert_scale", torch.ones(1, 1, 1, self.num_experts), persistent=False)

  def ohwi_weights(self):
    """gate / ff1 [I, E, 1, D] and linear [D, E, 1, I] (row = out x E + expert, the op's layout)."""
    if EXPERT_MODE != "fp32":
      return self.ph_gate, self.ph_ff1, self.ph_lin
    return self.w_gate, self.w_ff1, self.w_lin

  def forward(self, x, top_w, top_i):
    w_gate, w_ff1, w_lin = self.ohwi_weights()
    n = x.shape[0]
    return torch.ops.q35moe.moe_silu(
        x.reshape(1, n, self.model_dim),
        top_w.reshape(1, n, self.num_active),
        top_i.to(torch.int32).reshape(1, n, self.num_active),
        w_gate, w_ff1, w_lin, self.per_expert_scale,
        self.num_experts, self.num_active, self.model_dim, self.hidden_dim,
    ).reshape(n, self.model_dim)


class Qwen35MoeBlock(nn.Module):
  """HF `Qwen3_5MoeSparseMoeBlock` (transformers 5.22.2 modeling_qwen3_5_moe.py 872-911) with the routed experts as
  the `moe` op. Router: softmax in fp32, top-k, divide by the sum (always)."""

  def __init__(self, config, layer_idx):
    super().__init__()
    self.num_experts = config.num_experts
    self.top_k = config.num_experts_per_tok
    self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
    self.experts = Qwen35MoeExperts(config, layer_idx)
    self.shared_expert = Qwen3_5MLP(config, config.shared_expert_intermediate_size)
    self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False)

  def forward(self, hidden_states):
    bsz, seq, dim = hidden_states.shape
    x = hidden_states.reshape(-1, dim)
    if SHARED_3D == "1":
      # rank 3 through the shared expert and its gate (see Q35MOE_SHARED_3D in the docstring)
      shared = self.shared_expert(hidden_states)
      shared = (torch.sigmoid(self.shared_expert_gate(hidden_states)) * shared).reshape(-1, dim)
    else:
      shared = self.shared_expert(x)
      shared = torch.sigmoid(self.shared_expert_gate(x)) * shared
    probs = torch.softmax(self.gate(x).to(torch.float32), dim=-1)
    top_w, top_i = torch.topk(probs, self.top_k, dim=-1)
    top_w = top_w / top_w.sum(dim=-1, keepdim=True)
    routed = self.experts(x, top_w.to(x.dtype), top_i)
    if FUSION_GUARD == "2":
      # Two consumers of the moe output (MUL, SUB), distinct inputs everywhere, exact in fp32 (see the docstring).
      # The converter keeps it (checked on the exported graph).
      half = routed * 0.5
      out = shared + (half + (routed - half))
    elif FUSION_GUARD == "1":
      # Two consumers of the moe output, exact for finite values (r - r = 0, r - 0 = r). The converter folds
      # `r * 0.0` and `r - r` away but keeps `r - (r - r)`.
      out = shared + (routed - (routed - routed))
    else:
      out = shared + routed
    return out.reshape(bsz, seq, dim)


class Qwen35MoeStaticDecoderLayer(q35_static.Qwen3_5StaticDecoderLayer):
  """The upstream static layer with the MoE block in place of the dense MLP (forward inherited unchanged)."""

  def __init__(self, config, layer_idx):  # pylint: disable=super-init-not-called
    nn.Module.__init__(self)
    config = q35_static._unwrap_config(config)  # pylint: disable=protected-access
    self.layer_idx = layer_idx
    self.hidden_size = config.hidden_size
    self.block_type = config.layer_types[layer_idx]
    if self.block_type == "linear_attention":
      self.linear_attn = q35_static.Qwen3_5StaticGatedDeltaNet(
          hidden_size=config.hidden_size,
          num_v_heads=config.linear_num_value_heads,
          num_k_heads=config.linear_num_key_heads,
          head_k_dim=config.linear_key_head_dim,
          head_v_dim=config.linear_value_head_dim,
          conv_kernel_size=config.linear_conv_kernel_dim,
          rms_norm_eps=config.rms_norm_eps,
          layer_idx=layer_idx,
          use_fused_gdn=getattr(config, "use_fused_gdn", True),
          gdn_mode=getattr(config, "gdn_mode", 0),
          use_fp32=getattr(config, "gdn_use_fp32", True),
          use_short_conv_composite=getattr(config, "use_short_conv_composite", True),
          use_fused_gdn_block=getattr(config, "use_fused_gdn_block", True),
      )
    elif self.block_type == "full_attention":
      self.self_attn = Qwen3_5Attention(config, layer_idx)
    else:
      raise ValueError(f"unknown layer type {self.block_type!r}")
    self.mlp = Qwen35MoeBlock(config, layer_idx)
    self.input_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    self.post_attention_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Qwen35MoeStaticModel(q35_static.Qwen3_5StaticModel):

  def __init__(self, config):  # pylint: disable=super-init-not-called
    nn.Module.__init__(self)
    config = q35_static._unwrap_config(config)  # pylint: disable=protected-access
    self.config = config
    self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, getattr(config, "pad_token_id", None))
    self.layers = nn.ModuleList(
        [Qwen35MoeStaticDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
    self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    self.rotary_emb = q35_static.Qwen3_5StaticRotaryEmbedding(config=config)


class Qwen35MoeStaticForCausalLM(q35_static.Qwen3_5StaticForCausalLM):
  """Constructible as cls(config) (export_lib then copies the checkpoint tensors by parameter name)."""

  def __init__(self, config):  # pylint: disable=super-init-not-called
    nn.Module.__init__(self)
    config = q35_static._unwrap_config(config)  # pylint: disable=protected-access
    self.config = config
    self.model = Qwen35MoeStaticModel(config)
    self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

  @classmethod
  def from_hf_model(cls, hf_model):
    raise NotImplementedError("build Qwen35MoeStaticForCausalLM from its config and the checkpoint")


@extension_lib.register_model_extension(override=True)
class Qwen35MoeExtension(extension_lib.ModelExtension):
  """Qwen3.5 MoE text decoder: upstream Qwen3.5 hybrid layers + the `moe` op per layer."""

  model_types = ("qwen3_5_moe", "qwen3_5_moe_text")

  def get_export_defaults(self, config_dict, task=None, target="cpu_gpu"):
    del config_dict, task, target
    return extension_lib.ExportDefaults(
        task=extension_lib.ExportTask.TEXT_GENERATION,
        export_kwargs={
            "use_jinja_template": True,
            "litert_lm_llm_metadata_override": "qwen3_5",
        },
    )

  def get_hf_model_cls(self, task, model_config, export_config):
    del task, model_config, export_config
    return Qwen35MoeStaticForCausalLM

  def resolve_checkpoint_tensor(self, param_name, try_load):
    """fp32 experts: `...mlp.experts.w_gate / w_ff1 / w_lin` from the checkpoint's fused `gate_up_proj` /
    `down_proj` of the same layer, sliced and laid out in torch here (not in the graph)."""
    for suffix in OHWI_SUFFIXES:
      if param_name.endswith(suffix):
        base = param_name[: -len(suffix)] + ".mlp.experts."
        if suffix.endswith("w_lin"):
          down = try_load(base + "down_proj")
          if down is None:
            raise KeyError(f"{base}down_proj not found for {param_name}")
          return ohwi_from_checkpoint(suffix, None, down)
        gate_up = try_load(base + "gate_up_proj")
        if gate_up is None:
          raise KeyError(f"{base}gate_up_proj not found for {param_name}")
        return ohwi_from_checkpoint(suffix, gate_up, None)
    return None

  def get_prefill_decode_exportables(self, model_config, export_config):
    del model_config
    print(f"Using Qwen3.5 MoE exportables (experts: {EXPERT_MODE}, fusion guard: {FUSION_GUARD}).")
    if export_config.split_cache:
      raise NotImplementedError("split_cache is not supported by this Qwen3.5 MoE extension")
    if export_config.externalize_embedder:
      return (q35_exportable.LiteRTExportableModuleForQwen3_5PrefillExternalEmbedder,
              q35_exportable.LiteRTExportableModuleForQwen3_5GenerateExternalEmbedder)
    return (q35_exportable.LiteRTExportableModuleForQwen3_5Prefill,
            q35_exportable.LiteRTExportableModuleForQwen3_5Generate)
