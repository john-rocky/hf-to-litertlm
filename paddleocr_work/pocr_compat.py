"""transformers-5.12 compat shims for the PaddleOCR-VL-1.6 remote code.

Two breaks:
  1. ROPE_INIT_FUNCTIONS lost the 'default' key; the remote RotaryEmbedding
     still looks it up at __init__.
  2. PreTrainedModel._init_weights now calls
     module.compute_default_rope_parameters(config) on any *RotaryEmbedding*
     module that has an `original_inv_freq` buffer; the remote classes lack it.

load_pocr() installs both shims and returns the loaded model.
"""
import sys

import torch
from transformers import AutoModelForCausalLM
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS


def _default_rope(config=None, device=None, seq_len=None, **kwargs):
  base = float(getattr(config, "rope_theta", 10000.0))
  dim = getattr(config, "head_dim", None) or (
      config.hidden_size // config.num_attention_heads)
  dim = int(dim * getattr(config, "partial_rotary_factor", 1.0))
  inv_freq = 1.0 / (base ** (
      torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
  return inv_freq, 1.0


if "default" not in ROPE_INIT_FUNCTIONS:
  ROPE_INIT_FUNCTIONS["default"] = _default_rope


def _cdrp(self, config=None, device=None, seq_len=None, **kwargs):
  cfg = config if config is not None else self.config
  return _default_rope(cfg, device=device, seq_len=seq_len)


def load_pocr(model_dir, dtype=torch.float32, attn_implementation="sdpa"):
  cls = get_class_from_dynamic_module(
      "modeling_paddleocr_vl.PaddleOCRVLForConditionalGeneration", model_dir)
  mod = sys.modules[cls.__module__]
  for name in dir(mod):
    obj = getattr(mod, name)
    if isinstance(obj, type) and "RotaryEmbedding" in name:
      obj.compute_default_rope_parameters = _cdrp

  # transformers 5.12 create_causal_mask() dropped `cache_position` and takes 2D
  # position_ids; the remote code passes cache_position + 3D mrope positions.
  # batch=1 / no padding here, so the plain causal mask (position_ids=None) is
  # exact — the mrope positions only feed RoPE, not the mask.
  if hasattr(mod, "create_causal_mask") and not getattr(
      mod.create_causal_mask, "_pocr_shim", False):
    orig_ccm = mod.create_causal_mask

    def _ccm(*args, **kwargs):
      kwargs.pop("cache_position", None)
      pi = kwargs.get("position_ids")
      if pi is not None and pi.dim() == 3:
        kwargs["position_ids"] = None
      return orig_ccm(*args, **kwargs)

    _ccm._pocr_shim = True
    mod.create_causal_mask = _ccm
  model = AutoModelForCausalLM.from_pretrained(
      model_dir, dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=True,
      attn_implementation=attn_implementation).eval()

  # transformers 5.12 generate() no longer synthesizes cache_position for
  # remote-code models; the 1.6 code indexes it unconditionally.
  orig_prep = model.prepare_inputs_for_generation

  # NOTE: the var-keyword MUST be named **kwargs — generate()'s
  # _validate_model_kwargs only unions forward()'s signature when it sees a
  # parameter literally called "kwargs".
  def _prep(input_ids, past_key_values=None, **kwargs):
    if kwargs.get("cache_position") is None:
      past_len = 0
      if past_key_values is not None:
        try:
          past_len = int(past_key_values.get_seq_length())
        except Exception:
          past_len = 0
      kwargs["cache_position"] = torch.arange(
          past_len, past_len + input_ids.shape[1], device=input_ids.device)
    return orig_prep(input_ids, past_key_values=past_key_values, **kwargs)

  model.prepare_inputs_for_generation = _prep
  return model
