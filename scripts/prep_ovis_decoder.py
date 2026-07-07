"""Extract Ovis2.5-2B's inner decoder (`model.llm`, a Qwen3ForCausalLM, hidden=2048)
as a STANDALONE Qwen3 model so the proven export path (export_internvl_decoder.py,
BOCTAV4 + externalize_embedder + single_token_embedder) can convert it as the
.litertlm PREFILL_DECODE + EMBEDDER sections.

Ovis exposes the decoder as `.llm` (InternVL used `.language_model`). Qwen3's SP
vocab already contains <think>/<|im_start|> so no added-tokens fix is needed.

    ~/clipconv/bin/python scripts/prep_ovis_decoder.py [out_dir]
"""
import sys, os

import torch

# Ovis remote code predates transformers 5.12 — same 3 compat shims as run_verify.py.
import transformers.modeling_utils as _mu
if not hasattr(_mu.PreTrainedModel, "all_tied_weights_keys"):
  _mu.PreTrainedModel.all_tied_weights_keys = {}
if not hasattr(_mu.PreTrainedModel, "is_parallelizable"):
  _mu.PreTrainedModel.is_parallelizable = False
from transformers import AutoModelForCausalLM

MID = "AIDC-AI/Ovis2.5-2B"
OUT = sys.argv[1] if len(sys.argv) > 1 else "src_models/ovis2_5-2b-llm"
os.makedirs(OUT, exist_ok=True)


def load():
  return AutoModelForCausalLM.from_pretrained(
      MID, trust_remote_code=True, torch_dtype=torch.bfloat16,
      low_cpu_mem_usage=True).eval()


print("loading Ovis2.5-2B (bf16, cpu)...")
try:
  model = load()
except TypeError as e:
  if "tie_weights" not in str(e):
    raise
  for name, mod in list(sys.modules.items()):
    if "modeling_ovis2_5" in name and hasattr(mod, "Ovis2_5"):
      cls = mod.Ovis2_5; _o = cls.tie_weights
      cls.tie_weights = lambda self, *a, **k: _o(self)
  model = load()

lm = model.llm  # Qwen3ForCausalLM
cfg = lm.config
print("decoder:", type(lm).__name__, "model_type=", cfg.model_type,
      "hidden=", cfg.hidden_size, "layers=", cfg.num_hidden_layers,
      "vocab=", cfg.vocab_size, "tie=", getattr(cfg, "tie_word_embeddings", None))

if getattr(cfg, "rope_scaling", None):
  print("stripping rope_scaling:", cfg.rope_scaling)
  cfg.rope_scaling = None

cfg.architectures = [type(lm).__name__]  # Qwen3ForCausalLM
if hasattr(cfg, "auto_map"):
  del cfg.auto_map

print("saving standalone decoder ->", OUT)
lm.save_pretrained(OUT, safe_serialization=True)

# Ovis carries its Qwen3 text tokenizer as model.text_tokenizer.
model.text_tokenizer.save_pretrained(OUT)
print("saved tokenizer; vocab size:", model.text_tokenizer.vocab_size)
print("DONE")
