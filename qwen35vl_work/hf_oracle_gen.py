"""HF fp32 oracle generations for the Qwen3.5-2B fast_vlm gate.

Runs the full HF path (M-RoPE, dynamic-res processor pinned to 512x512 inputs)
greedily on 3 fixtures x 3 prompts, with the SAME think contract as the bundle
(enable_thinking=False -> empty <think> block scaffold). Optionally the 1-D
position ablation (POS1D=1): all M-RoPE channels sequential — what the fast_vlm
runtime contract can express (northmv's fold1d analog; no deepstack here, and
the raster/merge rewrite is numerically exact, so 1-D positions are the ONLY
contract cost).

    .venv-vl093/bin/python qwen35vl_work/hf_oracle_gen.py [out.json]
"""
import json
import os
import sys

import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODEL = os.environ.get("MODEL", "Qwen/Qwen3.5-2B")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "hf_oracle.json")
POS1D = os.environ.get("POS1D") == "1"
N_NEW = int(os.environ.get("N_NEW", "64"))
FIXTURES = os.path.join(ROOT, "northmv_work", "fixtures")
IMGS = ["cats_512.png", "kitchen1_512.png", "kitchen2_512.png"]
PROMPTS = ["What is in this image?",
           "Describe the main colors in this image.",
           "Where is this scene?"]

from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: E402

model = AutoModelForImageTextToText.from_pretrained(
    MODEL, dtype=torch.float32, low_cpu_mem_usage=True,
    attn_implementation="eager").eval()
processor = AutoProcessor.from_pretrained(MODEL)

if POS1D:
  # All M-RoPE channels = sequential text positions (the fast_vlm contract).
  orig = model.model.get_rope_index

  def seq_rope_index(*a, **kw):
    pos3, delta = orig(*a, **kw)          # pos3 [n_ch, B, S]
    S = pos3.shape[-1]
    seq = torch.arange(S, device=pos3.device).view(1, 1, S).expand_as(pos3)
    return seq.contiguous(), torch.zeros_like(delta)

  model.model.get_rope_index = seq_rope_index

res = {"model": MODEL, "pos1d": POS1D, "n_new": N_NEW, "cases": []}
for img_name in IMGS:
  im = Image.open(os.path.join(FIXTURES, img_name)).convert("RGB")
  assert im.size == (512, 512), im.size
  for prompt in PROMPTS:
    messages = [{"role": "user", "content": [
        {"type": "image", "image": im},
        {"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt", enable_thinking=False)
    with torch.no_grad():
      out = model.generate(**inputs, max_new_tokens=N_NEW, do_sample=False)
    text = processor.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True)
    res["cases"].append({"image": img_name, "prompt": prompt, "text": text})
    print(f"[{img_name} | {prompt}]\n{text}\n", flush=True)

with open(OUT, "w") as f:
  json.dump(res, f, indent=2, ensure_ascii=False)
print("ORACLE_DONE", OUT)
