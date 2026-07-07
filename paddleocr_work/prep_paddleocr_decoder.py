"""Extract PaddleOCR-VL-1.6's ERNIE-4.5-0.3B decoder as a standalone
LlamaForCausalLM for the proven fast_vlm decoder export.

The remote Ernie4_5Model is layout-identical to Llama (embed_tokens,
layers[i].self_attn.{q,k,v,o}_proj, mlp.{gate,up,down}_proj, input/post
layernorms, norm, untied lm_head; SiLU; RMSNorm; GQA kv=2; head_dim 128;
rope theta 500k standard-neox). Its only non-Llama trait is M-RoPE
(mrope_section 16/24/24) — but with the sequential positions the fast_vlm
runtime provides, all 3 mrope streams are equal and mrope == plain 1D RoPE,
so a vanilla Llama export is EXACT for the deployed contract (validated by
the text-only logits parity below).

    .venv/bin/python paddleocr_work/prep_paddleocr_decoder.py [model_dir] [out_dir]
"""
import os
import sys

import torch
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from pocr_compat import load_pocr  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "src_models/paddleocr-vl-1.6")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "src_models/paddleocr-vl-1.6-llm")
os.makedirs(OUT, exist_ok=True)

print("loading PaddleOCR-VL-1.6 (fp32, cpu)...")
model = load_pocr(MODEL, dtype=torch.float32, attn_implementation="eager")
src_cfg = model.config

cfg = LlamaConfig(
    vocab_size=src_cfg.vocab_size,
    hidden_size=src_cfg.hidden_size,
    intermediate_size=src_cfg.intermediate_size,
    num_hidden_layers=src_cfg.num_hidden_layers,
    num_attention_heads=src_cfg.num_attention_heads,
    num_key_value_heads=src_cfg.num_key_value_heads,
    head_dim=src_cfg.head_dim,
    hidden_act="silu",
    max_position_embeddings=src_cfg.max_position_embeddings,
    rms_norm_eps=src_cfg.rms_norm_eps,
    rope_theta=float(src_cfg.rope_theta),
    tie_word_embeddings=False,
    attention_bias=False,
    mlp_bias=False,
    pad_token_id=src_cfg.pad_token_id,
    eos_token_id=2,
    bos_token_id=1,
)
cfg.architectures = ["LlamaForCausalLM"]
# Do NOT hand-set cfg.rope_parameters: overriding it flips transformers 5.12's
# Llama rope init onto a different path and logits drift ~0.6; with plain
# rope_theta the wrap is BIT-EXACT vs the ERNIE decoder (probe verified).

lm = LlamaForCausalLM._from_config(cfg, attn_implementation="eager").eval()
sd = {k: v for k, v in model.state_dict().items()
      if k.startswith("model.") or k.startswith("lm_head.")}
missing, unexpected = lm.load_state_dict(sd, strict=False)
missing = [k for k in missing if "rotary_emb" not in k]
unexpected = [k for k in unexpected if "rotary_emb" not in k]
assert not missing, f"missing keys: {missing[:8]}"
assert not unexpected, f"unexpected keys: {unexpected[:8]}"
print("state_dict mapped 1:1")

# --- text-only logits parity: mrope(t=h=w) must equal plain 1D rope ---
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
ids = tok("User: List three prime numbers.\nAssistant:\n",
          return_tensors="pt").input_ids
with torch.no_grad():
  ref = model(input_ids=ids, use_cache=False).logits.float()
  got = lm(input_ids=ids, use_cache=False).logits.float()
maxdiff = float((ref - got).abs().max())
top_match = bool((ref[0, -1].argmax() == got[0, -1].argmax()).item())
print(f"text-only logits maxdiff={maxdiff:.6f} top1_match={top_match}")
assert maxdiff < 1e-3 and top_match, "wrap parity failed"

print("saving standalone decoder (bf16) ->", OUT)
lm = lm.to(torch.bfloat16)
lm.save_pretrained(OUT, safe_serialization=True)
tok.save_pretrained(OUT)
# keep the SP model for the bundle tokenizer
import shutil
spm = os.path.join(MODEL, "tokenizer.model")
if os.path.exists(spm):
  shutil.copy(spm, os.path.join(OUT, "tokenizer.model"))
print("DONE")
