#!/usr/bin/env python3
"""r2 step 1: the 448x448 fixture set for the vision parity + quant A/B.

  FIXTURE_SRC=<dir with cats_512.png, kitchen1_512.png, kitchen2_512.png> python vision_fixtures.py OUT_DIR

  - the 3 English photo fixtures ($FIXTURE_SRC/{cats,kitchen1,kitchen2}_512.png; not in this repository)
    resized to 448x448 with PIL BICUBIC -> OUT_DIR/<name>_448.png
  - one noise image: torch.manual_seed(0), torch.rand(448,448,3) * 255 rounded to uint8
    -> OUT_DIR/noise_448.png

Every consumer (HF reference, eager re-author, tflite A/B) reads these PNGs, so the
pixels are identical across venvs. Prints one JSON line per file (bytes, sha256 of the PNG
and of the decoded RGB array).
"""
import hashlib
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
FIX = os.environ.get("FIXTURE_SRC", os.path.join(REPO, "northmv_work", "fixtures"))
NAMES = ("cats", "kitchen1", "kitchen2")
IMG = 448


def record(path):
  raw = open(path, "rb").read()
  arr = np.asarray(Image.open(path).convert("RGB"))
  return {"file": os.path.basename(path), "bytes": len(raw),
          "sha256": hashlib.sha256(raw).hexdigest(),
          "rgb_shape": list(arr.shape), "rgb_sha256": hashlib.sha256(arr.tobytes()).hexdigest()}


def main():
  out = sys.argv[1]
  os.makedirs(out, exist_ok=True)
  rows = []
  for n in NAMES:
    src = os.path.join(FIX, f"{n}_512.png")
    im = Image.open(src).convert("RGB").resize((IMG, IMG), Image.BICUBIC)
    dst = os.path.join(out, f"{n}_448.png")
    im.save(dst)
    r = record(dst)
    r["source"] = os.path.relpath(src, REPO)
    r["source_sha256"] = hashlib.sha256(open(src, "rb").read()).hexdigest()
    rows.append(r)
  torch.manual_seed(0)
  noise = (torch.rand(IMG, IMG, 3) * 255).round().clamp(0, 255).to(torch.uint8).numpy()
  dst = os.path.join(out, "noise_448.png")
  Image.fromarray(noise, mode="RGB").save(dst)
  r = record(dst)
  r["source"] = "torch.manual_seed(0); (torch.rand(448,448,3)*255).round().clamp(0,255).uint8, torch " + torch.__version__
  rows.append(r)
  for r in rows:
    print(json.dumps(r))
  with open(os.path.join(out, "fixtures.json"), "w") as f:
    json.dump(rows, f, indent=1)
    f.write("\n")


if __name__ == "__main__":
  main()
