#!/usr/bin/env python3
"""requant_tflite.py FLOAT.tflite RECIPE.json OUT.tflite [--int4-floor] - quantize the float TFLite an export left behind
(export_hf ... --keep_temporary_files=True keeps OUTDIR/model.tflite) with another recipe, the way export_hf_lowmem.py's
quantize step does it: litert_quantizer.Quantizer(<mmap of the float model>, <recipe list>).quantize(serialize_to_path=OUT),
so the float model is never read into memory. With the export's own recipe it writes the export's model_quantized.tflite
byte for byte (checked on the Qwen3.5-0.8B export). --int4-floor raises every blockwise symmetric int4 scale below 2^-14
to 2^-14 (XNNPACK refuses a zero scale) and counts them. One JSON line on stdout: the recipe's sha256, sizes, the counts.
Refuses an existing OUT."""
import argparse
import hashlib
import json
import mmap
import os
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("float_tflite")
    ap.add_argument("recipe")
    ap.add_argument("out")
    ap.add_argument("--int4-floor", action="store_true")
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"refusing to overwrite {a.out}")
    from litert_quantizer import quantizer as quantizer_lib
    from litert_quantizer.algorithms.uniform_quantize import uniform_quantize_tensor as u
    recipe_bytes = open(a.recipe, "rb").read()
    recipe = json.loads(recipe_bytes)
    orig = u.tensor_zp_scale_from_min_max
    floor = np.float32(2.0 ** -14)
    st = {"blockwise_int4_calls": 0, "block_scales": 0, "scales_floored": 0}

    def floored(min_value, max_value, num_bits, symmetric, granularity, *args, **kw):
        zp, scale = orig(min_value, max_value, num_bits, symmetric, granularity, *args, **kw)
        if a.int4_floor and "BLOCKWISE" in str(granularity) and num_bits == 4:
            if not symmetric:
                raise ValueError("requant_tflite: asymmetric blockwise int4 is not covered by the floor")
            st["blockwise_int4_calls"] += 1
            low = scale < floor
            st["block_scales"] += int(np.size(scale))
            n = int(np.count_nonzero(low))
            if n:
                st["scales_floored"] += n
                scale = np.where(low, floor, scale).astype(np.asarray(scale).dtype)
        return zp, scale

    u.tensor_zp_scale_from_min_max = floored
    t0 = time.time()
    with open(a.float_tflite, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    try:
        qt = quantizer_lib.Quantizer(mm, recipe)
        qt.quantize(serialize_to_path=a.out)
    finally:
        u.tensor_zp_scale_from_min_max = orig
    print(json.dumps({"float": a.float_tflite, "float_bytes": os.path.getsize(a.float_tflite), "recipe": a.recipe,
                      "recipe_sha256": hashlib.sha256(recipe_bytes).hexdigest(), "out": a.out,
                      "out_bytes": os.path.getsize(a.out), "int4_floor": a.int4_floor, **st,
                      "s": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
