"""CRITICAL de-risk for the Qwen2-VL-2B fast_vlm ride: does replacing the true
3-D M-RoPE with plain 1-D sequential positions (what the fast_vlm runtime feeds
the decoder) preserve quality on GENERAL VQA — not just OCR/doc?

PaddleOCR proved 1-D is safe for structured raster-order OCR. General VQA and
spatial reasoning may lean harder on true 2-D visual positions. This runs both
and prints them side by side on describe / count / spatial / OCR prompts.

  .venv/bin/python qwen2vl_work/ab_mrope_vqa_gate.py
"""
import os

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODEL = os.path.join(ROOT, "src_models/qwen2-vl-2b")

PHOTO = os.path.join(ROOT, "colorization_work/repo/imgs/ansel_adams.jpg")
DOC = os.path.join(ROOT, "paddleocr_work/testdocs/para.png")
TABLE = os.path.join(ROOT, "paddleocr_work/testdocs/table.png")

PROMPTS = [
    (PHOTO, "Describe this image in detail."),
    (PHOTO, "What is in the foreground versus the background?"),
    (TABLE, "How many data rows are in this table, and what is the Total revenue?"),
    (DOC, "Extract all the text from this image."),
]

print("loading Qwen2-VL-2B (fp32, cpu)...", flush=True)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL, dtype=torch.float32, low_cpu_mem_usage=True).eval()
processor = AutoProcessor.from_pretrained(MODEL)

orig_get_rope_index = model.model.get_rope_index if hasattr(model, "model") \
    else model.get_rope_index
holder = model.model if hasattr(model.model, "get_rope_index") else model


def seq_rope_index(input_ids=None, image_grid_thw=None, video_grid_thw=None,
                   second_per_grid_ts=None, attention_mask=None, **kw):
  # 3 identical streams = plain 1-D positions (the fast_vlm contract)
  bsz, seqlen = input_ids.shape
  pos = torch.arange(seqlen, dtype=input_ids.dtype,
                     device=input_ids.device).view(1, 1, seqlen).expand(3, bsz, seqlen)
  deltas = torch.zeros(bsz, 1, dtype=input_ids.dtype, device=input_ids.device)
  return pos.contiguous(), deltas


def run(img_path, prompt, mode):
  holder.get_rope_index = seq_rope_index if mode == "1d" else orig_get_rope_index
  if hasattr(model, "rope_deltas"):
    model.rope_deltas = None
  img = Image.open(img_path).convert("RGB")
  messages = [{"role": "user", "content": [
      {"type": "image"}, {"type": "text", "text": prompt}]}]
  text = processor.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=True)
  inputs = processor(text=[text], images=[img], return_tensors="pt")
  with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
  return processor.batch_decode(out[:, inputs.input_ids.shape[1]:],
                                skip_special_tokens=True)[0].strip()


for img_path, prompt in PROMPTS:
  a = run(img_path, prompt, "mrope")
  b = run(img_path, prompt, "1d")
  print(f"\n===== [{os.path.basename(img_path)}] {prompt} =====")
  print(f"--- A true M-RoPE ---\n{a}")
  print(f"--- B forced 1D ---\n{b}")
  print(f"--- IDENTICAL: {a == b} ---", flush=True)
