"""Extract Qwen2-VL-2B's inner text decoder (Qwen2VLTextModel == Qwen2 with
attention bias + M-RoPE) as a standalone Qwen2ForCausalLM for the fast_vlm
decoder export.

With the sequential 1-D positions the fast_vlm runtime feeds, M-RoPE collapses to
plain 1-D RoPE (all 3 mrope streams equal) — proven quality-safe for VQA/OCR by
qwen2vl_work/ab_mrope_vqa_gate.py. So a vanilla Qwen2 export is exact for the
deployed contract (verified bit-exact on text-only logits below).

lm_head is tied to embed_tokens.

    .venv/bin/python qwen2vl_work/prep_qwen2vl_decoder.py [model_dir] [out_dir]
"""
import os
import shutil
import sys

import torch
from transformers import (AutoTokenizer, Qwen2Config, Qwen2ForCausalLM,
                          Qwen2VLForConditionalGeneration)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODEL = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "src_models/qwen2-vl-2b")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "src_models/qwen2-vl-2b-llm")
os.makedirs(OUT, exist_ok=True)

print("loading Qwen2-VL-2B (fp32, cpu)...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL, dtype=torch.float32, low_cpu_mem_usage=True, attn_implementation="eager").eval()
src = model.config
tcfg = src.get_text_config()
_rope_theta = getattr(tcfg, "rope_theta", None) or \
    getattr(tcfg, "rope_parameters", {}).get("rope_theta", 1000000.0)

cfg = Qwen2Config(
    vocab_size=tcfg.vocab_size,
    hidden_size=tcfg.hidden_size,
    intermediate_size=tcfg.intermediate_size,
    num_hidden_layers=tcfg.num_hidden_layers,
    num_attention_heads=tcfg.num_attention_heads,
    num_key_value_heads=tcfg.num_key_value_heads,
    hidden_act="silu",
    max_position_embeddings=tcfg.max_position_embeddings,
    rms_norm_eps=tcfg.rms_norm_eps,
    rope_theta=float(_rope_theta),
    tie_word_embeddings=True,
    attention_dropout=0.0,
    bos_token_id=getattr(tcfg, "bos_token_id", 151643),
    eos_token_id=getattr(tcfg, "eos_token_id", 151645),
)
cfg.architectures = ["Qwen2ForCausalLM"]
# NOTE: do NOT set cfg.torch_dtype=bf16 before _from_config — it would
# instantiate the model in bf16 and bf16-round the fp32 weights on load
# (logits drift ~3.5 on the 152k vocab). Build fp32, verify, cast at save.

lm = Qwen2ForCausalLM._from_config(cfg, attn_implementation="eager").eval()
# remap model.language_model.* -> model.* ; lm_head tied to embed_tokens
sd = {}
for k, v in model.state_dict().items():
  if k.startswith("model.language_model."):
    sd["model." + k[len("model.language_model."):]] = v
lm.load_state_dict(sd, strict=False)
lm.tie_weights()  # lm_head <- embed_tokens

# text-only logits parity (mrope with equal streams must == plain 1-D rope)
tok = AutoTokenizer.from_pretrained(MODEL)
ids = tok("<|im_start|>user\nList three prime numbers.<|im_end|>\n<|im_start|>assistant\n",
          return_tensors="pt", add_special_tokens=False).input_ids
lang = model.model.language_model
with torch.no_grad():
  h_ref = lang(input_ids=ids, use_cache=False).last_hidden_state
  ref = torch.nn.functional.linear(h_ref, model.lm_head.weight).float()
  got = lm(input_ids=ids, use_cache=False).logits.float()
maxdiff = float((ref - got).abs().max())
top_match = bool((ref[0, -1].argmax() == got[0, -1].argmax()).item())
print(f"text-only logits maxdiff={maxdiff:.6f} top1_match={top_match}")
assert maxdiff < 1e-3 and top_match, "wrap parity failed"

print("saving standalone decoder (bf16) ->", OUT)
lm.to(torch.bfloat16).save_pretrained(OUT, safe_serialization=True)
tok.save_pretrained(OUT)
for f in ("vocab.json", "merges.txt", "tokenizer.json"):
  s = os.path.join(MODEL, f)
  if os.path.exists(s):
    shutil.copy(s, os.path.join(OUT, f))
print("DONE")
