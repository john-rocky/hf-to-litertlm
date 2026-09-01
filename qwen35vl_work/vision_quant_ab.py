"""fp16 variants of the Qwen3.5 vision tflites + quant A/B on REAL images.

The converter's parity/int8 numbers are measured on a noise image (worst case
for DRQ). This measures fp32/fp16/int8 encoder+adapter combinations against the
HF fp32 visual reference on the 3 real fixtures + noise, like northmv's
vision_quant_ab.py.

    .venv-vl093/bin/python qwen35vl_work/vision_quant_ab.py out/qwen35vl-vision
"""
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
VIS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "out/qwen35vl-vision")
MODEL = os.environ.get("MODEL", "Qwen/Qwen3.5-2B")
IMG = 512
# Fixture set is a MEASUREMENT choice: photo fixtures for captioning towers,
# document fixtures (docling_work/*_512.png) for OCR towers — OCR is harder on
# int8 vision and the photo numbers must not be quoted for it.
FIXTURES = os.environ.get("FIXTURES", os.path.join(ROOT, "northmv_work", "fixtures"))
FIXTURE_FILES = os.environ.get(
    "FIXTURE_FILES", "cats_512.png,kitchen1_512.png,kitchen2_512.png").split(",")


def tfl_run(p, x):
  from ai_edge_litert.interpreter import Interpreter
  it = Interpreter(model_path=p)
  it.allocate_tensors()
  d = it.get_input_details()[0]
  it.set_tensor(d["index"], np.asarray(x, dtype=d["dtype"]))
  it.invoke()
  return it.get_tensor(it.get_output_details()[0]["index"])


def quant_fp16(src, dst):
  # WF16 float_casting JSON recipe (the export_internvl_decoder.py pattern —
  # the recipe_manager dynamic-config route silently no-ops for float_casting).
  import copy
  from ai_edge_quantizer import quantizer
  import ai_edge_quantizer.recipe as r
  base = copy.deepcopy(r.dynamic_wi8_afp32()[0])
  base["algorithm_key"] = "float_casting"
  base["op_config"]["weight_tensor_config"] = {
      "num_bits": 16, "symmetric": True, "granularity": "TENSORWISE",
      "dtype": "FLOAT"}
  base["op_config"]["compute_precision"] = "FLOAT"
  base["op_config"]["explicit_dequantize"] = False
  recipes = []
  for op in ("FULLY_CONNECTED", "CONV_2D"):  # BATCH_MATMUL: weightless + unsupported by float_casting
    rr = copy.deepcopy(base)
    rr["operation"] = op
    recipes.append(rr)
  q = quantizer.Quantizer(src, recipes)
  q.quantize().export_model(dst)
  return round(os.path.getsize(dst) / 1e6, 1)


def main():
  # HF reference
  from transformers import AutoModelForImageTextToText, AutoProcessor
  model = AutoModelForImageTextToText.from_pretrained(
      MODEL, dtype=torch.float32, low_cpu_mem_usage=True,
      attn_implementation="eager").eval()
  processor = AutoProcessor.from_pretrained(MODEL)
  visual = model.model.visual
  visual.config._attn_implementation = "eager"

  imgs = {}
  for f in FIXTURE_FILES:
    p = os.path.join(FIXTURES, f)
    im = Image.open(p).convert("RGB").resize((IMG, IMG), Image.BICUBIC)
    imgs[f] = im
  torch.manual_seed(0)
  noise_u8 = (torch.rand(IMG, IMG, 3) * 255).round().clamp(0, 255).to(torch.uint8)
  imgs["noise"] = Image.fromarray(noise_u8.numpy(), mode="RGB")

  refs = {}
  with torch.no_grad():
    for name, im in imgs.items():
      pp = processor.image_processor(images=[im], return_tensors="pt")
      refs[name] = visual(pp["pixel_values"], grid_thw=pp["image_grid_thw"]).pooler_output.numpy()

  # fp16 variants
  res = {}
  for part in ("vision_encoder", "vision_adapter"):
    src = os.path.join(VIS, f"{part}.tflite")
    dst = os.path.join(VIS, f"{part}_fp16.tflite")
    if not os.path.exists(dst):
      res[f"{part}_fp16_mb"] = quant_fp16(src, dst)

  combos = {
      "fp32/fp32": ("vision_encoder.tflite", "vision_adapter.tflite"),
      "fp16/fp16": ("vision_encoder_fp16.tflite", "vision_adapter_fp16.tflite"),
      "fp16/int8": ("vision_encoder_fp16.tflite", "vision_adapter_int8.tflite"),
      "int8/int8": ("vision_encoder_int8.tflite", "vision_adapter_int8.tflite"),
  }
  for combo, (ef, af) in combos.items():
    ep, ap_ = os.path.join(VIS, ef), os.path.join(VIS, af)
    if not (os.path.exists(ep) and os.path.exists(ap_)):
      continue
    corrs = {}
    for name, im in imgs.items():
      x = np.asarray(im, dtype=np.float32)[None] / 255.0
      feats = tfl_run(ep, x)
      out = tfl_run(ap_, feats).reshape(-1).astype("float64")
      rf = refs[name].reshape(-1).astype("float64")
      corrs[name] = round(float(np.corrcoef(out, rf)[0, 1]), 6)
    res[combo] = corrs
    print(combo, corrs, flush=True)

  with open(os.path.join(VIS, "quant_ab.json"), "w") as f:
    json.dump(res, f, indent=2)
  print("AB_DONE", json.dumps(res))


if __name__ == "__main__":
  main()
