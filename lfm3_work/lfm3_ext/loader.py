"""Builds an LFM3 decoder (`Lfm3ForCausalLM`) from any LFM3 checkpoint.

The four LFM3 repos ship the same `modeling_lfm3.py` (docstrings aside) and
keep the decoder under different key prefixes (2026-10-28 safetensors headers,
286 BF16 decoder tensors each, no `lm_head` = tied to `embed_tokens`):

  LiquidAI/LFM3-1.7B      `model.`
  LiquidAI/LFM3-VL-2B     `model.language_model.`
  LiquidAI/LFM3-Omni-2B   `model.language_model.`
  LiquidAI/LFM3-Audio-2B  `lfm.model.`

The model class comes from the repo's own remote code (`code_dir`), the
config from `config` (a dict such as a VL `text_config` or the Audio `lfm`
block, an `Lfm3Config`, or `code_dir/config.json`).
"""

import json
import os

import safetensors
import torch
import transformers
from transformers import dynamic_module_utils
from transformers import initialization

_CONFIG_SKIP = ("architectures", "auto_map", "transformers_version", "dtype", "torch_dtype")


def _remote_class(code_dir, reference):
  return dynamic_module_utils.get_class_from_dynamic_module(reference, code_dir)


def load_lfm3_config(code_dir, config=None):
  """Returns the decoder `Lfm3Config` (remote class from `code_dir`)."""
  config_cls = _remote_class(code_dir, "configuration_lfm3.Lfm3Config")
  if config is None:
    with open(os.path.join(code_dir, "config.json")) as f:
      config = json.load(f)
  if isinstance(config, dict):
    if config.get("model_type") != "lfm3":
      raise ValueError(
          f"lfm3 loader: config model_type is {config.get('model_type')!r};"
          " pass the decoder block (VL / Omni `text_config`, Audio `lfm`)."
      )
    config = config_cls(**{k: v for k, v in config.items() if k not in _CONFIG_SKIP})
  return config


def _weight_files(weights):
  if isinstance(weights, (list, tuple)):
    return list(weights)
  if os.path.isdir(weights):
    files = sorted(
        os.path.join(weights, f) for f in os.listdir(weights) if f.endswith(".safetensors")
    )
    if not files:
      raise FileNotFoundError(f"lfm3 loader: no .safetensors in {weights}")
    return files
  return [weights]


def load_lfm3_causal_lm(
    code_dir,
    weights=None,
    *,
    key_prefix="model.",
    config=None,
    dtype=torch.float32,
):
  """Builds `Lfm3ForCausalLM` and loads the decoder weights under `key_prefix`.

  Args:
    code_dir: Directory with `modeling_lfm3.py`, `configuration_lfm3.py` and
      `lfm3_conv.py` (any of the four repos), and `config.json` /
      `generation_config.json` when `config` is not given.
    weights: A `.safetensors` file, a directory of them, or a list; default
      `code_dir`.
    key_prefix: Checkpoint prefix of the decoder (`model.`,
      `model.language_model.`, `lfm.model.`).
    config: Decoder config: dict, `Lfm3Config`, or None for
      `code_dir/config.json`.
    dtype: Parameter dtype (export traces in float32).

  Returns:
    `Lfm3ForCausalLM` in eval mode, `lm_head` tied to `embed_tokens`.
  """
  config = load_lfm3_config(code_dir, config)
  model_cls = _remote_class(code_dir, "modeling_lfm3.Lfm3ForCausalLM")
  with initialization.no_init_weights():
    model = model_cls(config)
  state = {}
  for path in _weight_files(weights or code_dir):
    with safetensors.safe_open(path, framework="pt") as f:
      for key in f.keys():
        if key.startswith(key_prefix):
          state["model." + key[len(key_prefix):]] = f.get_tensor(key).to(dtype)
  if not state:
    raise ValueError(f"lfm3 loader: no tensor under prefix {key_prefix!r}")
  model = model.to(dtype)
  missing, unexpected = model.load_state_dict(state, strict=False)
  missing = [k for k in missing if k != "lm_head.weight"]
  if missing or unexpected:
    raise ValueError(f"lfm3 loader: missing {missing[:8]}, unexpected {unexpected[:8]}")
  if config.tie_word_embeddings:
    model.lm_head.weight = model.model.embed_tokens.weight
  elif "lm_head.weight" not in state:
    raise ValueError("lfm3 loader: untied config and no lm_head weight")
  gen_path = os.path.join(code_dir, "generation_config.json")
  if os.path.exists(gen_path):
    model.generation_config = transformers.GenerationConfig.from_pretrained(code_dir)
  return model.eval()
