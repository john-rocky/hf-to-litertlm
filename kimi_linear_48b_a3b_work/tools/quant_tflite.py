#!/usr/bin/env python3
"""quant_tflite.py FLOAT.tflite OUT.tflite RECIPE.json - litert_quantizer with a recipe on the fp32 export:
Quantizer(<mmap of the float file>, recipe).quantize(serialize_to_path=OUT), inside quant_common.floored_blockwise_scales
(it changes 4-bit blockwise scales only, and tools/make_recipe.py's recipe has none: int8 per channel). Writes
OUT.quant.json (the recipe, the floor's counts, seconds, bytes).
"""
import importlib.util
import json
import mmap
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
  if len(sys.argv) != 4:
    print(__doc__.splitlines()[0], file=sys.stderr)
    return 2
  src, out, rp = sys.argv[1], sys.argv[2], sys.argv[3]
  spec = importlib.util.spec_from_file_location("quant_common", os.path.join(HERE, "quant_common.py"))
  qc = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(qc)
  from litert_quantizer import quantizer as quantizer_lib
  recipe = json.load(open(rp))
  stats = {}
  t0 = time.time()
  with open(src, "rb") as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
  with qc.floored_blockwise_scales(stats):
    quantizer_lib.Quantizer(mm, recipe).quantize(serialize_to_path=out)
  rec = {"float": os.path.abspath(src), "float_bytes": os.path.getsize(src), "out": os.path.abspath(out),
         "out_bytes": os.path.getsize(out), "recipe": os.path.abspath(rp), "recipe_body": recipe,
         "blockwise_int4_calls": stats.get("calls", 0), "block_scales": stats.get("scales", 0),
         "scales_floored": stats.get("floored", 0), "seconds": round(time.time() - t0, 1)}
  json.dump(rec, open(out + ".quant.json", "w"), indent=1)
  print("QUANT_DONE", json.dumps({k: v for k, v in rec.items() if k != "recipe_body"}))
  return 0


if __name__ == "__main__":
  sys.exit(main())
