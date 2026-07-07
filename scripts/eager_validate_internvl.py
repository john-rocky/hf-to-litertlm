"""Sanity: run InternVL3-2B (eager) on a real image with the SAME single-image,
ImageNet-normalized, 448x448 preprocessing the fast_vlm bundle assumes. If the
model describes the image coherently here, our pipeline understanding is correct
and the converted tflites (parity 1.0 vs eager) will preserve it.

    ~/clipconv/bin/python scripts/eager_validate_internvl.py [image] [question]
"""
import sys

import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import transformers.modeling_utils as _mu
if not hasattr(_mu.PreTrainedModel, "all_tied_weights_keys"):
  _mu.PreTrainedModel.all_tied_weights_keys = {}

MODEL = "src_models/internvl3-2b"
IMG = sys.argv[1] if len(sys.argv) > 1 else "/tmp/test_image.jpg"
Q = sys.argv[2] if len(sys.argv) > 2 else "Describe this image in detail."

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
tf = T.Compose([
    T.Lambda(lambda im: im.convert("RGB")),
    T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])

print("loading InternVL3-2B...")
model = AutoModel.from_pretrained(
    MODEL, trust_remote_code=True, torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True).eval()
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, use_fast=False)

pixel_values = tf(Image.open(IMG)).unsqueeze(0).to(torch.bfloat16)  # [1,3,448,448]
print("pixel_values:", tuple(pixel_values.shape))

gen = dict(max_new_tokens=128, do_sample=False)
question = "<image>\n" + Q
resp = model.chat(tok, pixel_values, question, gen)
print("\n=== QUESTION ===\n" + Q)
print("\n=== RESPONSE (eager, single 448 tile) ===\n" + resp)
