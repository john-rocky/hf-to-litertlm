#!/usr/bin/env python3
"""quantize_clamped.py <../minicpm5_work/quantize_minicpm5.py arguments> - run ../minicpm5_work/quantize_minicpm5.py unchanged with
ONE change inside ai-edge-quantizer (0.8.0): clamp-before-quantize for blockwise int4 scales — no blockwise int4 scale is allowed
below 2^-14 (float16's smallest normal).

Why (measured on Falcon-H1R-7B with scan_scales.py): the plain int4 build carried 96 zero + 748 subnormal fp16 block scales in 7
Mamba-mixer projection tensors (dead or near-dead 32-weight blocks: bound / 7 under float16's normal range). XNNPACK prepares a
blockwise int4 FC only when every scale is std::isnormal, so the CPU path refuses such a file; a zero scale also hides the w / 0
saturated payload the quantizer writes for it. With the scale raised to 2^-14 it is normal and the payload is the nearest integer
of w / 2^-14 (= 0 for |w| < 2^-15).

How: ai_edge_quantizer.algorithms.uniform_quantize.uniform_quantize_tensor.tensor_zp_scale_from_min_max is wrapped (every caller in
0.8.0 — naive_min_max_quantize, octav, gptq — calls it through the module attribute); for a BLOCKWISE granularity and num_bits 4 the
returned scale is max(scale, 2^-14). The original computes the block scale, converts it bfloat16 -> float16 and returns it in
float32; taking the max after that conversion equals taking it before (2^-14 is exact in both formats and the conversion is
monotone). Channelwise / int8 scales are untouched (the int8 file is not built through this wrapper). Prints the number of clamped
scales per call class at exit.
"""
import atexit
import os
import runpy
import sys

import numpy as np
from ai_edge_quantizer.algorithms.uniform_quantize import uniform_quantize_tensor as uqt

MIN_SCALE = np.float32(2.0 ** -14)
_orig = uqt.tensor_zp_scale_from_min_max
_count = {"calls": 0, "blockwise_int4_calls": 0, "clamped_scales": 0, "zero_or_subnormal_before": 0}


def _clamped(min_value, max_value, num_bits, symmetric, granularity, clipping_values=None):
    zp, scale = _orig(min_value, max_value, num_bits, symmetric, granularity, clipping_values)
    _count["calls"] += 1
    if uqt.is_blockwise(granularity) and num_bits == 4:
        _count["blockwise_int4_calls"] += 1
        s = np.asarray(scale, dtype=np.float32)
        low = s < MIN_SCALE
        n = int(low.sum())
        if n:
            _count["clamped_scales"] += n
            _count["zero_or_subnormal_before"] += int((s.astype(np.float16) < np.float16(2.0 ** -14)).sum())
            scale = np.maximum(s, MIN_SCALE)
    return zp, scale


uqt.tensor_zp_scale_from_min_max = _clamped
atexit.register(lambda: print("quantize_clamped:", _count, flush=True))

script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "minicpm5_work", "quantize_minicpm5.py")
sys.argv = [script] + sys.argv[1:]
runpy.run_path(script, run_name="__main__")
