"""Re-host North-Micro-Vision-Instruct's text decoder (CohereCompassTextModel) as a
standalone Cohere2ForCausalLM for the fast_vlm decoder export.

Why Cohere2 is the exact host (all four Cohere-isms are shared):
  * parallel block  h = x + attn(ln x) + mlp(ln x), one input_layernorm per layer
  * Cohere LayerNorm (mean subtracted, no bias)
  * layer_types SSSF x 7: sliding layers carry RoPE (theta 5e4), full-attention
    layers carry NO positional encoding (Cohere2Attention applies rope only when
    sliding_window is set; CohereCompass gives full layers position_embeddings=None)
  * logit_scale 0.25 on the tied 262144 x 2048 head
Compass's interleaved M-RoPE (sections [24,20,20]) collapses to plain 1-D RoPE when
all three position streams are equal — which is exactly what the fast_vlm runtime
feeds (sequential 1-D positions) — so text-only logits must match bit-exact-ish.

    .venv-vl0930-t515/bin/python northmv_work/prep_northmv_decoder.py [hf_id_or_dir] [out_dir]
"""
import os
import shutil
import sys

import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import (AutoModelForImageTextToText, AutoTokenizer, Cohere2Config,
                          Cohere2ForCausalLM)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODEL = sys.argv[1] if len(sys.argv) > 1 else "CohereLabs/North-Micro-Vision-Instruct"
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "src_models/north-micro-vision-llm")
os.makedirs(OUT, exist_ok=True)

print("loading North-Micro-Vision-Instruct (fp32, cpu)...")
model = AutoModelForImageTextToText.from_pretrained(
    MODEL, dtype=torch.float32, low_cpu_mem_usage=True, attn_implementation="eager").eval()
tcfg = model.config.get_text_config()
sl_rope = tcfg.rope_parameters["sliding_attention"]
assert tcfg.rope_parameters["full_attention"] is None, "full layers are expected NoPE"
assert sl_rope["rope_type"] == "default"

cfg = Cohere2Config(
    vocab_size=tcfg.vocab_size,
    hidden_size=tcfg.hidden_size,
    intermediate_size=tcfg.intermediate_size,
    logit_scale=tcfg.logit_scale,
    num_hidden_layers=tcfg.num_hidden_layers,
    num_attention_heads=tcfg.num_attention_heads,
    num_key_value_heads=tcfg.num_key_value_heads,
    hidden_act=tcfg.hidden_act,
    max_position_embeddings=tcfg.max_position_embeddings,
    layer_norm_eps=tcfg.layer_norm_eps,
    pad_token_id=tcfg.pad_token_id,
    bos_token_id=tcfg.bos_token_id,
    eos_token_id=tcfg.eos_token_id,
    tie_word_embeddings=True,
    rope_parameters={"rope_type": "default", "rope_theta": float(sl_rope["rope_theta"])},
    attention_bias=tcfg.attention_bias,
    attention_dropout=0.0,
    sliding_window=tcfg.sliding_window,
    layer_types=list(tcfg.layer_types),
)
assert cfg.head_dim == tcfg.head_dim, (cfg.head_dim, tcfg.head_dim)
cfg.architectures = ["Cohere2ForCausalLM"]
cfg.northmv_rope_layout = "half_split_llama_style"  # marker: needs northmv_rope_patch

lm = Cohere2ForCausalLM._from_config(cfg, attn_implementation="eager").eval()

# RoPE layout: CohereCompass rotates Llama-style half-split pairs (i, i+D/2) with
# cos/sin = cat(freqs, freqs); stock Cohere2 rotates GPT-J-style interleaved pairs
# (2i, 2i+1) via torch.stack(...).flatten -- a 5-D CONCATENATION the LiteRT GPU delegate
# rejects ("Tensor dimensions must be less than 5"), and its repeat_interleave cos/sin is a
# BROADCAST_TO (also rejected). So the host is Cohere2 with its rope functions patched to
# Compass's Llama-style math (northmv_rope_patch.py) and the weights are copied VERBATIM
# (no head-dim permutation). The same patch must be active wherever this checkpoint is
# loaded (prep parity check here, the export); the saved config carries a marker.
from northmv_rope_patch import patch_cohere2_rope  # noqa: E402
patch_cohere2_rope()

sd = {}
for k, v in model.state_dict().items():
  if k.startswith("model.language_model."):
    sd["model." + k[len("model.language_model."):]] = v
missing, unexpected = lm.load_state_dict(sd, strict=False)
missing = [m for m in missing if m != "lm_head.weight"]
print("missing:", missing, "unexpected:", unexpected)
assert not missing and not unexpected
lm.tie_weights()
assert lm.lm_head.weight.data_ptr() == lm.model.embed_tokens.weight.data_ptr()

# text-only logits parity: Compass (mrope with equal streams, NoPE full layers)
# vs Cohere2 (1-D rope on sliding, NoPE full layers)
tok = AutoTokenizer.from_pretrained(MODEL)
prompt = ("<BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>List three prime numbers and "
          "explain why they are prime.<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>")
ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids
print("prompt ids:", ids.shape, ids[0, :6].tolist())
with torch.no_grad():
  ref = model(input_ids=ids, use_cache=False).logits.float()
  got = lm(input_ids=ids, use_cache=False).logits.float()
maxdiff = float((ref - got).abs().max())
top1 = (ref.argmax(-1) == got.argmax(-1)).float().mean().item()
print(f"text-only logits maxdiff={maxdiff:.3e} top1_agreement(all positions)={top1:.4f} "
      f"|ref|max={ref.abs().max():.2f}")
assert maxdiff < 1e-3 and top1 == 1.0, "rehost parity failed"

# 24-token greedy continuation must agree too (exercises the KV path)
with torch.no_grad():
  g_ref = model.generate(input_ids=ids, max_new_tokens=24, do_sample=False)[0, ids.shape[1]:]
  g_got = lm.generate(input_ids=ids, max_new_tokens=24, do_sample=False)[0, ids.shape[1]:]
print("greedy ref:", repr(tok.decode(g_ref)))
print("greedy got:", repr(tok.decode(g_got)))
print("greedy identical:", bool(torch.equal(g_ref, g_got)))

# size ledger for the wi8 / int4 estimates
n_embed = lm.model.embed_tokens.weight.numel()
n_total = sum(p.numel() for p in lm.parameters())
n_layers = n_total - n_embed  # lm_head is tied (counted once)
print(f"decoder params: layers {n_layers/1e6:.0f}M + embed {n_embed/1e6:.0f}M = {n_total/1e6:.0f}M")
print(f"  wi8  : PREFILL_DECODE ~{n_layers/2**30:.2f} GiB + EMBEDDER(int8) ~{n_embed/2**30:.2f} GiB")
print(f"  int4 : PREFILL_DECODE ~{n_layers/2/2**30:.2f} GiB + EMBEDDER(int8) ~{n_embed/2**30:.2f} GiB")

print("saving standalone decoder (bf16) ->", OUT)
lm.to(torch.bfloat16).save_pretrained(OUT, safe_serialization=True)
tok.save_pretrained(OUT)
print("DONE")
