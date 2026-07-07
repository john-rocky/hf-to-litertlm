"""Extract InternVL3-2B's inner Qwen2.5-1.5B `language_model` as a STANDALONE
Qwen2 model so the proven export path (export_simple_template.py, BOCTAV4 +
externalize_embedder) can convert it as the .litertlm PREFILL_DECODE section.

Why standalone: AutoModelForCausalLM on the InternVL repo resolves (via auto_map)
to the whole InternVLChatModel, not the decoder. We pull `.language_model`
(a Qwen2ForCausalLM) and save it with the llm_config.

rope_scaling: InternVL adds {rope_type: dynamic, factor 2.0} to extend context.
We export with cache <= original max (32768) so the dynamic NTK scaling never
triggers -> base RoPE is exact. The converter rejects 'dynamic' rope, so we strip
rope_scaling (set None) before saving. (Same rationale as the Phi static-rope fix.)

    ~/clipconv/bin/python scripts/prep_internvl_decoder.py [model_dir] [out_dir]
"""
import sys, os, shutil

import torch
from transformers import AutoModel, AutoTokenizer

# Remote InternVLChatModel predates transformers 5.12's weight-tying refactor;
# provide a class-level default so it loads (safe: composite has no tied params).
import transformers.modeling_utils as _mu
if not hasattr(_mu.PreTrainedModel, "all_tied_weights_keys"):
  _mu.PreTrainedModel.all_tied_weights_keys = {}

MODEL = sys.argv[1] if len(sys.argv) > 1 else "src_models/internvl3-2b"
OUT = sys.argv[2] if len(sys.argv) > 2 else "src_models/internvl3-2b-llm"
os.makedirs(OUT, exist_ok=True)

print("loading InternVLChatModel (bf16, cpu)...")
model = AutoModel.from_pretrained(
    MODEL, trust_remote_code=True, torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True).eval()

lm = model.language_model  # Qwen2ForCausalLM
cfg = lm.config
print("decoder:", type(lm).__name__, "model_type=", cfg.model_type,
      "hidden=", cfg.hidden_size, "layers=", cfg.num_hidden_layers,
      "vocab=", cfg.vocab_size, "tie=", cfg.tie_word_embeddings)

# Strip dynamic rope_scaling -> base RoPE (valid for cache <= max_position).
if getattr(cfg, "rope_scaling", None):
  print("stripping rope_scaling:", cfg.rope_scaling)
  cfg.rope_scaling = None

# Make sure it saves as a plain Qwen2 (no remote auto_map carried over).
cfg.architectures = [type(lm).__name__]  # generic: Qwen2ForCausalLM or Qwen3ForCausalLM
if hasattr(cfg, "auto_map"):
  del cfg.auto_map

print("saving standalone decoder ->", OUT)
lm.save_pretrained(OUT, safe_serialization=True)

# Tokenizer (Qwen2.5 BPE + InternVL special image tokens).
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
tok.save_pretrained(OUT)
print("saved tokenizer; vocab size:", tok.vocab_size)
print("DONE")
