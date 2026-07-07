"""Extract SmolVLM2-500M's inner SmolLM2 (Llama) decoder as a standalone
LlamaForCausalLM for the proven decoder export. The text_model is a LlamaModel;
the lm_head is top-level and tied to the text embeddings.

    ~/clipconv/bin/python scripts/prep_smolvlm2_decoder.py [model_dir] [out_dir]
"""
import sys, os

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer, LlamaForCausalLM

MODEL = sys.argv[1] if len(sys.argv) > 1 else "src_models/smolvlm2-500m"
OUT = sys.argv[2] if len(sys.argv) > 2 else "src_models/smolvlm2-500m-llm"
os.makedirs(OUT, exist_ok=True)


def _get(m, name):
  for base in (getattr(m, "model", None), m):
    if base is not None and hasattr(base, name):
      return getattr(base, name)
  raise AttributeError(name)


print("loading SmolVLM2-500M (bf16, cpu)...")
model = AutoModelForImageTextToText.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
text_model = _get(model, "text_model")  # LlamaModel
lm_head = model.lm_head if hasattr(model, "lm_head") else _get(model, "lm_head")
text_cfg = text_model.config

print("decoder:", type(text_model).__name__, "model_type=", text_cfg.model_type,
      "hidden=", text_cfg.hidden_size, "layers=", text_cfg.num_hidden_layers,
      "kv=", getattr(text_cfg, "num_key_value_heads", None), "vocab=", text_cfg.vocab_size,
      "tie=", text_cfg.tie_word_embeddings, "rope_scaling=", getattr(text_cfg, "rope_scaling", None))

if getattr(text_cfg, "rope_scaling", None):
  print("stripping rope_scaling:", text_cfg.rope_scaling)
  text_cfg.rope_scaling = None
text_cfg.architectures = ["LlamaForCausalLM"]
# transformers 5.12 Llama reads config.rope_parameters["rope_type"] — SmolVLM's
# text_config may not populate it; set a default-RoPE block from rope_theta.
if not getattr(text_cfg, "rope_parameters", None):
  text_cfg.rope_parameters = {"rope_type": "default",
                              "rope_theta": float(getattr(text_cfg, "rope_theta", 100000.0))}

lm = LlamaForCausalLM(text_cfg).eval()
lm.model = text_model
lm.lm_head = lm_head
lm.config = text_cfg

print("saving standalone decoder ->", OUT)
lm.save_pretrained(OUT, safe_serialization=True)
tok = AutoTokenizer.from_pretrained(MODEL)
tok.save_pretrained(OUT)
print("DONE; tokenizer vocab:", tok.vocab_size)
