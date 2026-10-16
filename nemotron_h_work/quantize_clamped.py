#!/usr/bin/env python3
"""quantize_clamped.py <minicpm5_work/quantize_minicpm5.py arguments> - run minicpm5_work/quantize_minicpm5.py with one change inside
ai-edge-quantizer (0.8.0): every blockwise int4 scale is floored at 2^-14 (float16's smallest normal) before it is used
(clamp-before-quantize).

Why: a blockwise int4 FC whose block is dead or near dead (|w| below float16's normal range / 7) gets a zero or subnormal fp16 scale.
XNNPACK prepares a blockwise int4 FC only when every scale is std::isnormal, so the CPU path refuses such a file, and a zero scale
hides the w / 0 payload the quantizer writes for it. With the minimum in place the scale is normal and the payload is w / 2^-14 rounded
(= 0 for |w| < 2^-15). NVIDIA-Nemotron-Nano-9B-v2's int4 build floors 66,388 scales; the scan of the result shows none zero or
subnormal.

How: ai_edge_quantizer.algorithms.uniform_quantize.uniform_quantize_tensor.tensor_zp_scale_from_min_max is wrapped (every caller in
0.8.0 calls it through the module attribute); for a BLOCKWISE granularity and num_bits 4 the returned scale is max(scale, 2^-14).
The original computes the block scale, rounds it through bfloat16 -> float16 and returns it in float32; flooring after that conversion
is exact (2^-14 is representable in both formats and the conversion is monotone). Channelwise / int8 scales are untouched.
Prints the number of clamped scales per call class at exit.
"""
import atexit
import os
import runpy
import sys

import numpy as np
from ai_edge_quantizer.algorithms.uniform_quantize import uniform_quantize_tensor as uqt

SCALE_MIN = np.float32(2.0 ** -14)
_orig = uqt.tensor_zp_scale_from_min_max
_count = {"calls": 0, "blockwise_int4_calls": 0, "clamped_scales": 0, "zero_or_subnormal_before": 0}


def _clamped(min_value, max_value, num_bits, symmetric, granularity, clipping_values=None):
    zp, scale = _orig(min_value, max_value, num_bits, symmetric, granularity, clipping_values)
    _count["calls"] += 1
    if uqt.is_blockwise(granularity) and num_bits == 4:
        _count["blockwise_int4_calls"] += 1
        s = np.asarray(scale, dtype=np.float32)
        low = s < SCALE_MIN
        n = int(low.sum())
        if n:
            _count["clamped_scales"] += n
            _count["zero_or_subnormal_before"] += int((s.astype(np.float16) < np.float16(2.0 ** -14)).sum())
            scale = np.maximum(s, SCALE_MIN)
    return zp, scale


uqt.tensor_zp_scale_from_min_max = _clamped
atexit.register(lambda: print("quantize_clamped:", _count, flush=True))

script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "minicpm5_work", "quantize_minicpm5.py")
sys.argv = [script] + sys.argv[1:]
runpy.run_path(script, run_name="__main__")
