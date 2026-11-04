#!/usr/bin/env python3
"""export_cowork_decoder.py <llm_dir> <out_dir> - scripts/export_internvl_decoder.py (run by path, unchanged; env CACHE /
PREFILL / RECIPE as for it) with every blockwise int4 block scale floored at 2^-14 before the weights are quantized
(clamp-before-quantize, the same floor as spark_work/x26_export_clamped.py: a block scale that underflows fp16 to zero or
a subnormal breaks the GPU path).

The quantizer (checked for ai-edge-quantizer 0.9.0):
uniform_quantize_tensor.tensor_zp_scale_from_min_max(min, max, num_bits, symmetric, granularity, clipping_values=None)
rounds a blockwise scale through bf16 -> fp16 -> fp32 and returns it; octav.py (the BOCTAV4_128 recipe's algorithm),
naive_min_max_quantize.py and gptq.py look the function up as a module attribute at call time, so replacing the
attribute reaches every caller. For a blockwise 4-bit symmetric request the wrapper returns max(scale, 2^-14) (fp16's
smallest normal, a power of two: flooring after the function's fp16 round trip equals flooring before it); the caller
then quantizes with the floored scale, so a dead block gets payload round(w / 2^-14) instead of w / 0. Channelwise, int8
(the EMBEDDING_LOOKUP rule is channelwise int8) and asymmetric requests pass through; an asymmetric blockwise int4
request stops the export. Each floored tensor is logged; the totals go to stdout at exit as
`CLAMP_TOTAL calls=<blockwise int4 calls> tensors=<floored tensors> scales=<floored scales>`. For Mage-VL-Cowork's
decoder (CACHE=4096 PREFILL=128,512,1024 RECIPE=BOCTAV4_128) the shipped export printed
`CLAMP_TOTAL calls=253 tensors=0 scales=0 of 31424000`: no scale was floored.

Import order = the export script's own (ai_edge_quantizer.recipe first), then the patch, then the script via runpy as
__main__ with the same argv, so every module it imports is the patched one.

    CACHE=4096 PREFILL=128,512,1024 RECIPE=BOCTAV4_128 PYTHONHASHSEED=0 \\
      python magevlcowork_work/export_cowork_decoder.py <llm_dir> <out_dir>
"""
import atexit
import os
import runpy
import sys

SCRIPT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts",
                                       "export_internvl_decoder.py"))

import ai_edge_quantizer.recipe  # noqa: E402,F401  (the export script's first import; keep the order)
import numpy as np  # noqa: E402
from ai_edge_quantizer.algorithms.uniform_quantize import uniform_quantize_tensor as _u  # noqa: E402

_FLOOR = np.float32(2.0 ** -14)
_orig = _u.tensor_zp_scale_from_min_max
_stats = {"calls": 0, "tensors": 0, "scales": 0, "total_scales": 0}


def _clamped(min_value, max_value, num_bits, symmetric, granularity, *args, **kwargs):
  zp, scale = _orig(min_value, max_value, num_bits, symmetric, granularity, *args, **kwargs)
  if _u.is_blockwise(granularity) and num_bits == 4:
    _stats["calls"] += 1
    _stats["total_scales"] += int(np.size(scale))
    if not symmetric:
      raise ValueError("export_cowork_decoder: asymmetric blockwise int4 is not covered by this floor")
    low = scale < _FLOOR
    n = int(np.count_nonzero(low))
    if n:
      scale = np.where(low, _FLOOR, scale).astype(scale.dtype)
      _stats["tensors"] += 1
      _stats["scales"] += n
      print(f"CLAMP floor 2^-14: {n} of {scale.size} block scales", flush=True)
  return zp, scale


_u.tensor_zp_scale_from_min_max = _clamped
atexit.register(lambda: print(f"CLAMP_TOTAL calls={_stats['calls']} tensors={_stats['tensors']} "
                              f"scales={_stats['scales']} of {_stats['total_scales']}", flush=True))
print("PATCHED uniform_quantize_tensor.tensor_zp_scale_from_min_max: blockwise int4 scales floored at 2^-14", flush=True)

sys.argv = [SCRIPT] + sys.argv[1:]
runpy.run_path(SCRIPT, run_name="__main__")
