"""Probe granite-docling-258M (idefics3) against the SmolVLM2 fast_vlm rail.

Verifies, before any conversion:
  1. module layout: model.model.{vision_model,connector,text_model} + lm_head
  2. vision embeddings: patch_embedding/position_embedding attrs (static-pos patch target)
  3. connector: pixel_shuffle scale + modality_projection out dim (expect 64 x 576)
  4. decoder embedding rows vs tokenizer len (<end_of_utterance>=100352 is out of range?)
  5. processor render with do_image_splitting=False (the single-global-image rail form)

    .venv/bin/python docling_work/probe_docling_structure.py
"""
import json, sys

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL = "src_models/granite-docling-258m"

model = AutoModelForImageTextToText.from_pretrained(
    MODEL, torch_dtype=torch.float32, attn_implementation="eager",
    low_cpu_mem_usage=True).eval()
proc = AutoProcessor.from_pretrained(MODEL)

print("== top:", type(model).__name__)
inner = getattr(model, "model", model)
print("== inner:", type(inner).__name__)
for name in ("vision_model", "connector", "text_model"):
    ok = hasattr(inner, name) or hasattr(model, name)
    obj = getattr(inner, name, getattr(model, name, None))
    print(f"   {name}: {'OK' if ok else 'MISSING'} ->", type(obj).__name__ if obj is not None else None)

vm = inner.vision_model
emb = vm.embeddings
print("== vision emb:", type(emb).__name__,
      "| patch_embedding:", hasattr(emb, "patch_embedding"),
      "| position_embedding:", hasattr(emb, "position_embedding"),
      "| num_patches:", getattr(emb, "num_patches", None))

conn = inner.connector
print("== connector:", type(conn).__name__)
print(conn)
sf = getattr(model.config, "scale_factor", None)
print("   scale_factor:", sf)

tm = inner.text_model
cfg = tm.config
print("== text:", type(tm).__name__, "hidden", cfg.hidden_size, "layers", cfg.num_hidden_layers,
      "heads", cfg.num_attention_heads, "kv", cfg.num_key_value_heads,
      "vocab", cfg.vocab_size, "tie", cfg.tie_word_embeddings,
      "rope_theta", getattr(cfg, "rope_theta", None), "scaling", getattr(cfg, "rope_scaling", None))
wte = tm.embed_tokens.weight
print("   embed rows:", wte.shape, "| lm_head:", model.lm_head.weight.shape,
      "| tied:", model.lm_head.weight.data_ptr() == wte.data_ptr())
tok = proc.tokenizer
print("   tokenizer len:", len(tok), "| eou id:", tok.convert_tokens_to_ids("<end_of_utterance>"))

# vision forward smoke: one 512x512 through vision+connector eager
with torch.no_grad():
    px = torch.rand(1, 3, 512, 512)
    vout = vm(pixel_values=px).last_hidden_state
    cout = conn(vout)
print("== vision fwd:", list(vout.shape), "-> connector:", list(cout.shape))

# processor render, single global image (rail form)
from PIL import Image
proc.image_processor.do_image_splitting = False
img = Image.new("RGB", (512, 512), "white")
msgs = [{"role": "user", "content": [{"type": "image"},
                                     {"type": "text", "text": "Convert this page to docling."}]}]
prompt = proc.apply_chat_template(msgs, add_generation_prompt=True)
print("== chat-template prompt:", json.dumps(prompt))
inputs = proc(text=prompt, images=[img], return_tensors="pt")
ids = inputs.input_ids[0].tolist()
print("== input_ids len:", len(ids), "| pixel_values:", list(inputs.pixel_values.shape))
# decode with image runs compressed
out, run = [], 0
img_id = model.config.image_token_id
for t in ids:
    if t == img_id:
        run += 1; continue
    if run:
        out.append(f"<image>*{run}"); run = 0
    out.append(tok.convert_ids_to_tokens(t))
if run:
    out.append(f"<image>*{run}")
print("== token layout:", " ".join(out))
