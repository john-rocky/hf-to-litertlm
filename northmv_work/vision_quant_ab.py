"""A/B the vision quantization for the North-Micro-Vision fast_vlm ride.

The fp32 tflites in <vis_dir> are exact (corr 1.0 vs HF). Dynamic-range int8
(`dynamic_wi8_afp32`: int8 weights AND int8-quantized activations at runtime) landed
at corr 0.98 / maxdiff 1.7 on a noise image, so this measures on REAL images and
tries the weight-only variants:

  int8dyn : dynamic_wi8_afp32 (INTEGER compute)          -- what the converter emitted
  wi8f    : int8 weights, FLOAT compute (explicit dequant) -- no activation quantization
  fp16    : float_casting fp16 weights

for encoder x adapter combinations, against the HF visual path (merger + deepstack sum).

    .venv-vl0930-t515/bin/python northmv_work/vision_quant_ab.py out/northmv-vision-fold
"""
import copy
import io
import json
import os
import sys

import numpy as np
import requests
import torch
from PIL import Image

from transformers import AutoModelForImageTextToText, AutoProcessor

VIS = sys.argv[1] if len(sys.argv) > 1 else "out/northmv-vision-fold"
MODEL = "CohereLabs/North-Micro-Vision-Instruct"
IMG = 512
IMAGES = [
    "http://images.cocodataset.org/val2017/000000039769.jpg",
    "http://images.cocodataset.org/val2017/000000397133.jpg",
    "http://images.cocodataset.org/val2017/000000037777.jpg",
]


def tfl_run(p, x):
  from ai_edge_litert.interpreter import Interpreter
  it = Interpreter(model_path=p, num_threads=8)
  it.allocate_tensors()
  d = it.get_input_details()[0]
  it.set_tensor(d["index"], np.asarray(x, dtype=d["dtype"]))
  it.invoke()
  return it.get_tensor(it.get_output_details()[0]["index"])


def make_recipe(kind):
  import ai_edge_quantizer.recipe as r
  base = r.dynamic_wi8_afp32()[0]
  if kind == "int8dyn":
    return [base]
  q = copy.deepcopy(base)
  if kind == "wi8f":
    q["op_config"]["compute_precision"] = "FLOAT"
    q["op_config"]["explicit_dequantize"] = True
    return [q]
  if kind == "fp16":
    q["algorithm_key"] = "float_casting"
    q["op_config"]["weight_tensor_config"] = {
        "num_bits": 16, "symmetric": True, "granularity": "TENSORWISE", "dtype": "FLOAT"}
    q["op_config"]["compute_precision"] = "FLOAT"
    q["op_config"]["explicit_dequantize"] = False
    return [q]
  raise ValueError(kind)


def quantize(src, dst, kind):
  if os.path.exists(dst):
    return dst
  from ai_edge_quantizer import quantizer
  qt = quantizer.Quantizer(src, make_recipe(kind))
  qt.quantize().export_model(dst)
  return dst


def main():
  model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.float32).eval()
  processor = AutoProcessor.from_pretrained(MODEL)
  visual = model.model.visual

  imgs = []
  for u in IMAGES:
    raw = Image.open(io.BytesIO(requests.get(u, timeout=60).content)).convert("RGB")
    imgs.append(("coco" + u.rsplit("/", 1)[-1][-8:-4], raw.resize((IMG, IMG), Image.BICUBIC)))
  torch.manual_seed(0)
  imgs.append(("noise", Image.fromarray(
      (torch.rand(IMG, IMG, 3) * 255).round().to(torch.uint8).numpy(), "RGB")))

  refs, inputs = [], []
  for name, pil in imgs:
    pp = processor.image_processor(images=[pil], return_tensors="pt")
    with torch.no_grad():
      vo = visual(pp["pixel_values"], grid_thw=pp["image_grid_thw"])
      ref = vo.pooler_output + sum(vo.deepstack_features)
    refs.append(ref.numpy().astype(np.float64).reshape(-1))
    inputs.append((np.asarray(pil, dtype=np.float32) / 255.0)[None])

  variants = {"fp32": (os.path.join(VIS, "vision_encoder.tflite"), os.path.join(VIS, "vision_adapter.tflite"))}
  for kind in ("int8dyn", "wi8f", "fp16"):
    e = quantize(variants["fp32"][0], os.path.join(VIS, f"vision_encoder_{kind}.tflite"), kind)
    a = quantize(variants["fp32"][1], os.path.join(VIS, f"vision_adapter_{kind}.tflite"), kind)
    variants[kind] = (e, a)
    print(f"{kind}: enc {os.path.getsize(e)/1e6:.0f} MB adp {os.path.getsize(a)/1e6:.0f} MB", flush=True)

  combos = [("fp32", "fp32"), ("int8dyn", "int8dyn"), ("int8dyn", "fp32"), ("fp32", "int8dyn"),
            ("wi8f", "wi8f"), ("wi8f", "int8dyn"), ("int8dyn", "wi8f"), ("fp16", "fp16"),
            ("fp16", "int8dyn"), ("wi8f", "fp16")]
  out = {}
  for ek, ak in combos:
    rows = []
    for (name, _), x, ref in zip(imgs, inputs, refs):
      feats = tfl_run(variants[ek][0], x)
      got = tfl_run(variants[ak][1], feats).astype(np.float64).reshape(-1)
      c = float(np.corrcoef(got, ref)[0, 1])
      md = float(np.abs(got - ref).max())
      rel = float(np.linalg.norm(got - ref) / np.linalg.norm(ref))
      rows.append({"img": name, "corr": c, "maxdiff": md, "rel_l2": rel})
    key = f"enc={ek} adp={ak}"
    out[key] = rows
    print(key, " | ".join(f"{r['img']} corr {r['corr']:.5f} rel {r['rel_l2']:.4f}" for r in rows), flush=True)
  json.dump(out, open(os.path.join(VIS, "quant_ab.json"), "w"), indent=1)


if __name__ == "__main__":
  main()
