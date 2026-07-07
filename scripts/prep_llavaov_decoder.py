"""Extract LLaVA-OneVision-0.5B's inner Qwen2-0.5B `language_model` as a standalone
Qwen2 model for the proven decoder export. Native transformers model (no remote
code / no monkeypatch). tie_word_embeddings=True, rope_scaling=None (no strip needed).

    ~/clipconv/bin/python scripts/prep_llavaov_decoder.py [model_dir] [out_dir]
"""
import sys, os

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "src_models/llava-ov-0.5b"
OUT = sys.argv[2] if len(sys.argv) > 2 else "src_models/llava-ov-0.5b-llm"
os.makedirs(OUT, exist_ok=True)


def _get(m, name):
  for base in (getattr(m, "model", None), m):
    if base is not None and hasattr(base, name):
      return getattr(base, name)
  raise AttributeError(name)


print("loading LLaVA-OneVision (bf16, cpu)...")
model = AutoModelForImageTextToText.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
lm = _get(model, "language_model")  # Qwen2ForCausalLM
cfg = lm.config
print("decoder:", type(lm).__name__, "model_type=", cfg.model_type, "hidden=", cfg.hidden_size,
      "layers=", cfg.num_hidden_layers, "vocab=", cfg.vocab_size, "tie=", cfg.tie_word_embeddings,
      "rope_scaling=", getattr(cfg, "rope_scaling", None))

if getattr(cfg, "rope_scaling", None):
  print("stripping rope_scaling:", cfg.rope_scaling)
  cfg.rope_scaling = None
cfg.architectures = ["Qwen2ForCausalLM"]
if hasattr(cfg, "auto_map"):
  del cfg.auto_map

print("saving standalone decoder ->", OUT)
lm.save_pretrained(OUT, safe_serialization=True)
tok = AutoTokenizer.from_pretrained(MODEL)
tok.save_pretrained(OUT)
print("DONE; tokenizer vocab:", tok.vocab_size)
