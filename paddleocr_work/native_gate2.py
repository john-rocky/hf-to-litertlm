"""The definitive pre-ship gates, on the PROVEN README-verbatim native path:

  A: true M-RoPE, native dynamic resolution   (reference — known perfect)
  B: forced 1D sequential positions           (what the exported decoder computes)
  C: 1D + square STATIC resize                (the exact deployed fast_vlm contract)

  .venv/bin/python paddleocr_work/native_gate2.py
"""
import os

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODEL = os.path.join(ROOT, "src_models/paddleocr-vl-1.6")
STATIC = int(os.environ.get("STATIC", "560"))

model = AutoModelForImageTextToText.from_pretrained(
    MODEL, dtype=torch.bfloat16).eval()
processor = AutoProcessor.from_pretrained(MODEL)

holder = model.model if hasattr(model.model, "get_rope_index") else model
orig_rope_index = holder.get_rope_index


def seq_rope_index(input_ids=None, image_grid_thw=None, video_grid_thw=None,
                   second_per_grid_ts=None, attention_mask=None, **kw):
  bsz, seqlen = input_ids.shape
  pos = torch.arange(seqlen, dtype=input_ids.dtype,
                     device=input_ids.device).view(1, 1, seqlen).expand(3, bsz, seqlen)
  deltas = torch.zeros(bsz, 1, dtype=input_ids.dtype, device=input_ids.device)
  return pos.contiguous(), deltas


def run(img, prompt, mode):
  holder.get_rope_index = seq_rope_index if mode != "mrope" else orig_rope_index
  if hasattr(holder, "rope_deltas"):
    holder.rope_deltas = None
  if hasattr(model, "rope_deltas"):
    model.rope_deltas = None
  messages = [{"role": "user", "content": [
      {"type": "image", "image": img},
      {"type": "text", "text": prompt}]}]
  inputs = processor.apply_chat_template(
      messages, add_generation_prompt=True, tokenize=True, return_dict=True,
      return_tensors="pt",
      images_kwargs={"size": {"shortest_edge": 112896,
                              "longest_edge": 1280 * 28 * 28}},
  )
  with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=512)
  return processor.decode(out[0][inputs["input_ids"].shape[-1]:-1])


for name, prompt in [("para", "OCR:"), ("table", "Table Recognition:")]:
  img = Image.open(os.path.join(HERE, f"testdocs/{name}.png")).convert("RGB")
  img_sq = img.resize((STATIC, STATIC), Image.Resampling.BILINEAR)
  a = run(img, prompt, "mrope")
  b = run(img, prompt, "1d")
  c = run(img_sq, prompt, "1d")
  print(f"\n########## {name} ##########")
  print(f"--- A mrope+dyn ---\n{a}")
  print(f"--- B 1d+dyn ---\n{b}")
  print(f"--- C 1d+static{STATIC} ---\n{c}")
  print("--- A==B:", a == b, "A==C:", a == c, flush=True)
