#!/usr/bin/env python3
"""export_clamped_lw.py <args of granite41_work/export_lightweight.py> - granite41_work/export_lightweight.py (litert-torch's
export with the lightweight conversion) with ai-edge-quantizer's blockwise int4 scales floored at 2^-14 before the weights are
quantized, for the BOCTAV4 (OCTAV blockwise-32 int4) exports of convert_olmo3_7b_think.sh.

Why (measured on this model, 2026-10-16): BOCTAV4 (ai-edge-quantizer 0.9.0) wrote 1,574,919 zero and 2,031,784 subnormal fp16 block
scales of 215,210,752 into the FULLY_CONNECTED weights, most in the q_proj of the late layers. XNNPACK prepares a blockwise
FULLY_CONNECTED only when every scale is std::isnormal, and a zero scale also hides a w/0 payload.

What the patch does: wraps ai_edge_quantizer.algorithms.uniform_quantize.uniform_quantize_tensor.tensor_zp_scale_from_min_max; for
a blockwise 4-bit symmetric request it returns max(scale, 2^-14) (the fp16 min normal), and the caller quantizes the weights with
the floored scale (a dead block gets the payload nearest to w / 2^-14, 0 for |w| < 2^-15). Channelwise / int8 /
asymmetric requests pass through unchanged. Each floored tensor is logged; the total goes to stdout at exit as
`CLAMP_TOTAL tensors=<n> scales=<k>` (this model: tensors=55 scales=3606703, = the zero + subnormal count above). Run the block-scale
scan (nemotron_h_work/scan_scales.py) on the result: every scale normal."""
import atexit
import os
import runpy
import sys

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "granite41_work", "export_lightweight.py")

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
            raise ValueError("export_clamped_lw: asymmetric blockwise int4 is not covered by this lower bound")
        low = scale < _FLOOR
        n = int(np.count_nonzero(low))
        if n:
            scale = np.where(low, _FLOOR, scale).astype(scale.dtype)
            _stats["tensors"] += 1
            _stats["scales"] += n
            print(f"CLAMP lower bound 2^-14: {n} of {scale.size} block scales", flush=True)
    return zp, scale


_u.tensor_zp_scale_from_min_max = _clamped
atexit.register(lambda: print(f"CLAMP_TOTAL tensors={_stats['tensors']} scales={_stats['scales']}", flush=True))
print("PATCHED uniform_quantize_tensor.tensor_zp_scale_from_min_max: blockwise int4 scales floored at 2^-14", flush=True)

sys.argv = [SCRIPT] + sys.argv[1:]
runpy.run_path(SCRIPT, run_name="__main__")
