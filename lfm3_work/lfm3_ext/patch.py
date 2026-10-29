"""LiteRT export patches for LFM3 (`lfm3`) decoders.

LFM3 is LFM2 plus a sliding-attention layer type, sandwich norms, a sigmoid
output gate per attention head, q/k norms and zero-tail partial RoPE on the
full-attention layers. The HF `Lfm3Attention` (output gate, q/k norms, fp32
RoPE) traces as written through litert-torch's `lrt_transposed_attention`
interface: the sliding window lives in the mask the exportable builds, and the
partial rotation lives in the HF rotary embedding (inv_freq 0 on the tail =
cos 1, sin 0). Replaced here, per instance:

- `Lfm3ShortConv` -> LFM2's LiteRT short conv (`model_ext/lfm2/short_conv.py`)
  at width `conv_L_cache` (7): state `[1, hidden, L - 1]`, tap-wise prefill,
  optional `odml.short_conv_step` composite.
- `Lfm3DecoderLayer.forward` -> the same layer with the pad guard handed to the
  conv (the HF layer gives the conv no mask at batch 1).
- `Lfm3Model.forward` -> the inference path only: no packed-document checks,
  the pad guard from position monotonicity, and the final-norm output returned
  as `hidden_states[-1]` for the optional `hidden` signature output.
- `Lfm3RMSNorm` -> LFM2's `odml.rms_norm` composite norm (same math), only as a
  fallback: litert-torch's `core/patches.py` swaps every class decorated with
  `@use_kernel_forward_from_hub("RMSNorm")` for its own composite `RMSNorm`
  when `litert_torch` is imported before the remote code loads (the export
  path); a caller that loads the remote model first gets the swap here.
- `Lfm3MLP` -> LFM2's fused gate-up MLP under `fuse_gate_up` /
  `use_swiglu_composite`.

Modules are matched by class name: each LFM3 repo (text, VL, Omni, Audio)
ships its own `modeling_lfm3.py` as remote code, so the classes are distinct
objects with the same names.
"""

import contextlib
import types

from litert_torch.generative.export_hf.model_ext.lfm2 import patch as lfm2_patch
from litert_torch.generative.export_hf.model_ext.lfm2 import short_conv as lfm2_short_conv
import torch
from transformers import modeling_outputs

_PATCHED = "_lfm3_litert_patched"

# Composites whose LFM2 form does not fit LFM3: the RoPE composites rotate the
# full head dimension (LFM3's full-attention layers rotate 64 of 128, zero-tail
# layout, and its sliding layers use another theta), and the fused attention
# has no output gate.
_UNSUPPORTED_FLAGS = (
    "use_rope_composite",
    "use_qkv_norm_rope_composite",
    "fuse_qkv",
)


class Lfm3LiteRTShortConv(torch.nn.Module):
  """LFM3 gated short conv on LFM2's LiteRT short conv (same B, C, x split).

  LFM2's forward reads only the attributes set here; LFM3 differs in the width
  (7 vs 3) and in `seq_idx` (packed documents, training only).
  """

  forward = lfm2_short_conv.Lfm2ShortConv.forward

  def __init__(self, original: torch.nn.Module, use_short_conv_composite: bool):
    super().__init__()
    self.layer_idx = original.layer_idx
    self.bias = original.bias
    self.conv_L_cache_size = original.conv_kernel_size
    self.use_short_conv_composite = use_short_conv_composite
    self.in_proj = original.in_proj
    self.conv = original.conv  # Only `.weight` [hidden, 1, L] and `.bias`.
    self.out_proj = original.out_proj


def valid_mask_from_positions(position_ids: torch.Tensor) -> torch.Tensor:
  """Real slots of one call: the first, then every slot whose position grows.

  LiteRT-LM fills the unused tail of a prefill call with `input_pos` 0 and the
  pad token (or 0) as input (v0.18.0 / v0.18.1
  `llm_litert_compiled_model_executor.cc`, memset of the position buffer).
  Positions work for token and embedding inputs alike; a token-id test does
  not (externalized embedder, audio frames).

  Args:
    position_ids: `[1, T]` or `[T]` int positions of the call.

  Returns:
    `[1, T]` bool.
  """
  pos = position_ids.reshape(-1)
  first = torch.ones_like(pos[:1], dtype=torch.bool)
  return torch.cat([first, pos[1:] > pos[:-1]]).unsqueeze(0)


def _decoder_layer_forward(
    self,
    hidden_states,
    position_embeddings=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    valid_mask=None,
    cache_position=None,
    **kwargs,
):
  """`Lfm3DecoderLayer.forward` with the pad guard passed to the conv."""
  residual = hidden_states
  hidden_states = self.operator_norm(hidden_states)
  if self.is_attention_layer:
    hidden_states, _ = self.self_attn(
        hidden_states=hidden_states,
        position_embeddings=position_embeddings,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        **kwargs,
    )
  else:
    hidden_states = self.conv(
        hidden_states=hidden_states,
        past_key_values=past_key_values,
        cache_position=cache_position,
        valid_mask=valid_mask,
    )
  if self.use_sandwich_norm:
    hidden_states = self.post_operator_norm(hidden_states)
  hidden_states = hidden_states + residual
  ffn_out = self.feed_forward(self.ffn_norm(hidden_states))
  if self.use_sandwich_norm:
    ffn_out = self.post_ffn_norm(ffn_out)
  return hidden_states + ffn_out


def _model_forward(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    use_cache=None,
    valid_mask=None,
    cache_position=None,
    **kwargs,
):
  """`Lfm3Model.forward` for export; the exportable supplies the masks.

  Returns `hidden_states=(final_norm_output,)`: the `hidden` signature output
  reads it (`exportables.py`).
  """
  del use_cache  # The cache is always the exportable's.
  del valid_mask  # Token-id based (generic exportable); replaced below.
  if inputs_embeds is None:
    inputs_embeds = self.embed_tokens(input_ids)
  if inputs_embeds.shape[1] > 1:
    valid_mask = valid_mask_from_positions(position_ids)
  else:
    valid_mask = None  # Decode: LFM2's short conv takes the token as real.
  layer_types = self.config.layer_types
  if not isinstance(attention_mask, dict):
    if "sliding_attention" in layer_types:
      raise ValueError(
          "LFM3 with sliding layers needs the per-layer-type mask dict that"
          " the exportable builds (config.sliding_window set)."
      )
    attention_mask = {"full_attention": attention_mask}
  hidden_states = inputs_embeds
  position_embeddings = {
      layer_type: self.rotary_emb(
          hidden_states, position_ids=position_ids, layer_type=layer_type
      )
      for layer_type in self.rotary_emb.layer_types
  }
  for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
    layer_type = layer_types[i]
    hidden_states = decoder_layer(
        hidden_states,
        attention_mask=attention_mask.get(layer_type),
        position_embeddings=position_embeddings.get(layer_type),
        position_ids=position_ids,
        past_key_values=past_key_values,
        valid_mask=valid_mask,
        cache_position=cache_position,
        **kwargs,
    )
  hidden_states = self.embedding_norm(hidden_states)
  return modeling_outputs.BaseModelOutputWithPast(
      last_hidden_state=hidden_states,
      past_key_values=past_key_values,
      hidden_states=(hidden_states,),
  )


def _swap_rms_norms(module, replaced):
  """Swaps every `Lfm3RMSNorm` under `module` for LFM2's composite norm."""
  for name, child in module.named_children():
    if type(child).__name__ == "Lfm3RMSNorm":
      norm = lfm2_patch.Lfm2RMSNorm(child.weight.shape[0], child.variance_epsilon)
      norm.weight = child.weight
      setattr(module, name, norm)
      replaced.append((module, name, child))
    else:
      _swap_rms_norms(child, replaced)


@contextlib.contextmanager
def patch_lfm3_model(model, export_config):
  """Applies the LFM3 export patches to every LFM3 module under `model`.

  Idempotent: a second entry (a `CompositeModelExtension` holding two LFM3
  extensions) leaves the model as the first one patched it.

  Args:
    model: Any module tree holding `Lfm3Model` / `Lfm3ShortConv` instances
      (`Lfm3ForCausalLM`, or a VL / Omni / Audio model's language model).
    export_config: The active `ExportableModuleConfig`.

  Yields:
    Control while the patches are active.
  """
  if getattr(model, _PATCHED, False):
    yield
    return
  for flag in _UNSUPPORTED_FLAGS:
    if getattr(export_config, flag, False):
      raise NotImplementedError(f"lfm3: `{flag}` has no LFM3 form yet.")
  fuse_mlp = bool(
      getattr(export_config, "fuse_gate_up", False)
      or getattr(export_config, "use_swiglu_composite", False)
  )
  use_swiglu = bool(getattr(export_config, "use_swiglu_composite", False))
  use_short_conv = bool(getattr(export_config, "use_short_conv_composite", False))

  replaced = []  # (parent, attribute, original module)
  forwards = []  # modules given an instance `forward`
  _swap_rms_norms(model, replaced)
  norms_swapped = len(replaced)
  norms_core = sum(
      1 for m in model.modules()
      if type(m).__name__ == "RMSNorm" and "export_hf.core.patches" in type(m).__module__
  )
  for parent in list(model.modules()):
    for name, child in list(parent.named_children()):
      kind = type(child).__name__
      if kind == "Lfm3ShortConv":
        setattr(parent, name, Lfm3LiteRTShortConv(child, use_short_conv))
        replaced.append((parent, name, child))
      elif kind == "Lfm3MLP" and fuse_mlp:
        setattr(
            parent,
            name,
            lfm2_patch.FusedLfm2MLP(child, use_swiglu_composite=use_swiglu),
        )
        replaced.append((parent, name, child))
  for module in model.modules():
    kind = type(module).__name__
    if kind == "Lfm3DecoderLayer":
      module.forward = types.MethodType(_decoder_layer_forward, module)
      forwards.append(module)
    elif kind == "Lfm3Model":
      module.forward = types.MethodType(_model_forward, module)
      forwards.append(module)
  if not any(type(m).__name__ == "Lfm3Model" for m in model.modules()):
    raise ValueError("lfm3: no Lfm3Model found under the exported module.")
  print(
      f"LFM3 model patch applied: {len(replaced) - norms_swapped} modules"
      f" replaced, {len(forwards)} forwards, RMSNorm: {norms_core} already"
      f" litert-torch core + {norms_swapped} swapped here;"
      f" fuse_gate_up/swiglu={fuse_mlp},"
      f" use_short_conv_composite={use_short_conv}"
  )
  setattr(model, _PATCHED, True)
  try:
    yield
  finally:
    delattr(model, _PATCHED)
    for module in forwards:
      del module.forward
    for parent, name, original in reversed(replaced):
      setattr(parent, name, original)
