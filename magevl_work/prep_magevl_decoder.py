"""Extract Mage-VL's inner text decoder as a standalone Qwen3ForCausalLM.

Mage-VL's language model is instantiated via AutoModel.from_config(text_config)
with model_type "qwen3" — i.e. it IS a stock transformers Qwen3Model (fine-tuned
from Qwen3-4B-Instruct-2507; 36L/2560h, all full_attention, GQA kv8, untied
lm_head). The modeling code feeds simple 1-D position_ids (no M-RoPE), so the
fast_vlm runtime contract is exact — no positional compromise at all.

The remap is purely mechanical (model.language_model.* -> model.*, lm_head kept):
strict=True load against Qwen3ForCausalLM is the 1:1 proof; no VL-side forward
needed. Weights are bf16 in the checkpoint and stay bf16 (identity round-trip).

Runs in the pinned .venv (transformers 5.12.1 has Qwen3; no remote code needed).

    .venv/bin/python magevl_work/prep_magevl_decoder.py [model_dir] [out_dir]
"""
import glob
import json
import os
import shutil
import sys

import torch
from safetensors import safe_open
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODEL = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "src_models/mage-vl")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "src_models/mage-vl-llm")
os.makedirs(OUT, exist_ok=True)

with open(os.path.join(MODEL, "config.json")) as f:
  tcfg = json.load(f)["text_config"]

cfg = Qwen3Config(
    vocab_size=tcfg["vocab_size"],
    hidden_size=tcfg["hidden_size"],
    intermediate_size=tcfg["intermediate_size"],
    num_hidden_layers=tcfg["num_hidden_layers"],
    num_attention_heads=tcfg["num_attention_heads"],
    num_key_value_heads=tcfg["num_key_value_heads"],
    head_dim=tcfg["head_dim"],
    hidden_act=tcfg["hidden_act"],
    max_position_embeddings=tcfg["max_position_embeddings"],
    rms_norm_eps=tcfg["rms_norm_eps"],
    rope_theta=float(tcfg["rope_parameters"]["rope_theta"]),
    attention_bias=tcfg["attention_bias"],
    tie_word_embeddings=False,
    attention_dropout=0.0,
    bos_token_id=tcfg["bos_token_id"],
    eos_token_id=tcfg["eos_token_id"],
    use_sliding_window=False,
)
cfg.architectures = ["Qwen3ForCausalLM"]

print("building Qwen3ForCausalLM (bf16; checkpoint is bf16 -> identity load)...")
lm = Qwen3ForCausalLM._from_config(cfg, attn_implementation="eager").to(torch.bfloat16).eval()

sd = {}
for shard in sorted(glob.glob(os.path.join(MODEL, "model-*.safetensors"))):
  with safe_open(shard, framework="pt") as f:
    for k in f.keys():
      if k.startswith("model.language_model."):
        sd["model." + k[len("model.language_model."):]] = f.get_tensor(k)
      elif k == "lm_head.weight":
        sd[k] = f.get_tensor(k)
missing, unexpected = lm.load_state_dict(sd, strict=False)
missing = [m for m in missing if "rotary_emb" not in m and "inv_freq" not in m]
assert not missing, f"missing: {missing[:8]}"
assert not unexpected, f"unexpected: {unexpected[:8]}"
print(f"loaded {len(sd)} tensors, strict 1:1 OK")

tok = AutoTokenizer.from_pretrained(MODEL)
prompt = "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n"
ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids
with torch.no_grad():
  logits = lm(input_ids=ids, use_cache=False).logits.float()
assert torch.isfinite(logits).all(), "non-finite logits"
top5 = logits[0, -1].topk(5).indices.tolist()
print("sanity next-token top5:", [tok.decode([t]) for t in top5])

print("saving standalone decoder (bf16) ->", OUT)
lm.save_pretrained(OUT, safe_serialization=True)
tok.save_pretrained(OUT)
for f in ("vocab.json", "merges.txt", "tokenizer.json", "generation_config.json",
          "chat_template.jinja", "special_tokens_map.json", "added_tokens.json"):
  s = os.path.join(MODEL, f)
  if os.path.exists(s):
    shutil.copy(s, os.path.join(OUT, f))
print("DONE")
