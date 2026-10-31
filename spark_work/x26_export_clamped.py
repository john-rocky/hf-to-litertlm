#!/usr/bin/env python3
"""x26_export_clamped.py <args of spark_work/convert_spark.py> -- the Spark int4 export with a clamp-before-quantize
floor of 2^-14 on blockwise int4 scales (a block scale that underflows fp16 to zero or a subnormal breaks the GPU path;
the floor is a no-op where the scales are normal, as on both Spark-X2.6 int4 exports: CLAMP_TOTAL 0).

The patch: wrap
ai_edge_quantizer.algorithms.uniform_quantize.uniform_quantize_tensor.tensor_zp_scale_from_min_max (every caller looks it
up as a module attribute at call time) so a blockwise 4-bit symmetric request returns max(scale, 2^-14); channelwise / int8
requests pass through. Each floored tensor is logged and `CLAMP_TOTAL tensors=<n> scales=<k>` is printed at exit. Then
spark_work/convert_spark.py runs in this process via runpy (it sets NO_START_TOKEN / USE_JINJA / CACHE / PREFILL and runs
scripts/export_simple_template.py), so the export uses the patched module.

  EXTERNALIZE_EMBEDDER=1 python spark_work/x26_export_clamped.py \
      src_models/Spark-X2.6-1.7B-export out/spark26-1.7b-int4b32 templates/spark26_think.jinja BOCTAV4
"""
import atexit
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "convert_spark.py")

import transformers  # noqa: E402,F401  (the export script's first import; keep the order)
from litert_torch.generative.export_hf.core import export_lib  # noqa: E402,F401
import numpy as np  # noqa: E402
from ai_edge_quantizer.algorithms.uniform_quantize import uniform_quantize_tensor as _u  # noqa: E402

_FLOOR = np.float32(2.0 ** -14)
_orig = _u.tensor_zp_scale_from_min_max
_stats = {"tensors": 0, "scales": 0}


def _clamped(min_value, max_value, num_bits, symmetric, granularity, clipping_values=None):
  zp, scale = _orig(min_value, max_value, num_bits, symmetric, granularity, clipping_values)
  if _u.is_blockwise(granularity) and num_bits == 4:
    if not symmetric:
      raise ValueError("x26_export_clamped: asymmetric blockwise int4 is not covered by this floor")
    low = scale < _FLOOR
    n = int(np.count_nonzero(low))
    if n:
      scale = np.where(low, _FLOOR, scale).astype(scale.dtype)
      _stats["tensors"] += 1
      _stats["scales"] += n
      print(f"CLAMP floor 2^-14: {n} of {scale.size} block scales", flush=True)
  return zp, scale


_u.tensor_zp_scale_from_min_max = _clamped
atexit.register(lambda: print(f"CLAMP_TOTAL tensors={_stats['tensors']} scales={_stats['scales']}", flush=True))
print("PATCHED uniform_quantize_tensor.tensor_zp_scale_from_min_max: blockwise int4 scales floored at 2^-14", flush=True)

sys.argv = [SCRIPT] + sys.argv[1:]
runpy.run_path(SCRIPT, run_name="__main__")
