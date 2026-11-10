#!/usr/bin/env python3
"""quant_tflite.py FLOAT.tflite OUT.tflite RECIPE.json - the nightly litert_quantizer on a float export_hf graph:
Quantizer(<mmap of the float file>, recipe).quantize(serialize_to_path=OUT) (the float file is not read into memory),
with every 4-bit blockwise symmetric scale floored at 2^-14 (the MoE experts reach the same scale function through materialize_moe ->
naive_min_max_quantize -> uniform_quantize_tensor.tensor_zp_scale_from_min_max, read 2026-11-09 in
litert-quantizer-nightly 0.12.0.dev20261109). MoE custom ops get the #11309 form (7 -> 10 inputs, weight_type from
the weight dtype) inside the quantizer (model_modifier._update_quantized_moe_ops).
Writes OUT.quant.json (recipe, counts of blockwise int4 scale calls / scales / floored scales, seconds, bytes).
"""
import json
import mmap
import os
import sys
import time

import numpy as np


def main():
  src, out, rp = sys.argv[1], sys.argv[2], sys.argv[3]
  from litert_quantizer import quantizer as quantizer_lib
  from litert_quantizer.algorithms.uniform_quantize import uniform_quantize_tensor as u
  recipe = json.load(open(rp))
  orig = u.tensor_zp_scale_from_min_max
  floor = np.float32(2.0 ** -14)
  st = {"calls": 0, "floored": 0, "scales": 0}

  def floored(min_value, max_value, num_bits, symmetric, granularity, *a, **k):
    zp, scale = orig(min_value, max_value, num_bits, symmetric, granularity, *a, **k)
    if "BLOCKWISE" in str(granularity) and num_bits == 4:
      if not symmetric:
        raise ValueError("asymmetric blockwise int4 is not covered by the floor")
      st["calls"] += 1
      low = scale < floor
      st["scales"] += int(np.size(scale))
      n = int(np.count_nonzero(low))
      if n:
        st["floored"] += n
        scale = np.where(low, floor, scale).astype(np.asarray(scale).dtype)
    return zp, scale

  u.tensor_zp_scale_from_min_max = floored
  t0 = time.time()
  with open(src, "rb") as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
  try:
    qt = quantizer_lib.Quantizer(mm, recipe)
    qt.quantize(serialize_to_path=out)
  finally:
    u.tensor_zp_scale_from_min_max = orig
  rec = {"float": os.path.abspath(src), "float_bytes": os.path.getsize(src), "out": os.path.abspath(out),
         "out_bytes": os.path.getsize(out), "recipe": os.path.abspath(rp), "recipe_body": recipe,
         "blockwise_int4_calls": st["calls"], "block_scales": st["scales"], "scales_floored": st["floored"],
         "seconds": round(time.time() - t0, 1)}
  json.dump(rec, open(out + ".quant.json", "w"), indent=1)
  print("QUANT_DONE", json.dumps({k: v for k, v in rec.items() if k != "recipe_body"}))
  return 0


if __name__ == "__main__":
  sys.exit(main())
