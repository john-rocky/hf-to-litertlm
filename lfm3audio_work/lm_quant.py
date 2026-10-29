#!/usr/bin/env python3
"""lm_quant.py --src FLOAT_DIR --recipe JSON --out DIR - quantizes the float decoder and embedder that
`lfm3_ext.export_lfm3` wrote (quantization_recipe 'none', keep_temporary_files, externalize_embedder) with an
ai-edge-quantizer recipe through litert-torch's own `quantize_model`:
  FLOAT_DIR/model.tflite    -> DIR/decoder.tflite
  FLOAT_DIR/embedder.tflite -> DIR/embedder.tflite   (the same recipe: its EMBEDDING_LOOKUP entry)
Every blockwise 4-bit symmetric scale is floored at 2^-14 before the weights are rounded (XNNPACK refuses a smaller
block scale; `install_int4_floor` of lfm3_work/quant_pack.py). Prints how many scales the floor raised.
Needs PYTHONPATH=<this repository>/lfm3_work.
"""
import argparse
import hashlib
import os
import shutil
import time

from quant_pack import install_int4_floor


def sha256(p):
  h = hashlib.sha256()
  with open(p, "rb") as f:
    for c in iter(lambda: f.read(1 << 24), b""):
      h.update(c)
  return h.hexdigest()


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--src", required=True)
  ap.add_argument("--recipe", required=True)
  ap.add_argument("--out", required=True)
  a = ap.parse_args()
  stats = install_int4_floor()
  from litert_torch.generative.export_hf.core import quantization  # pylint: disable=g-import-not-at-top
  os.makedirs(a.out, exist_ok=True)
  recipe = os.path.abspath(a.recipe)
  for src, dst in (("model.tflite", "decoder.tflite"), ("embedder.tflite", "embedder.tflite")):
    t0 = time.time()
    q = quantization.quantize_model(os.path.abspath(os.path.join(a.src, src)), recipe)
    out = os.path.join(a.out, dst)
    shutil.move(q, out)
    print(f"{out}: {os.path.getsize(out)} B sha256 {sha256(out)} ({time.time() - t0:.1f}s)", flush=True)
  print(f"int4 block scales floored at 2^-14: {stats['scales']} in {stats['tensors']} tensors", flush=True)


if __name__ == "__main__":
  main()
