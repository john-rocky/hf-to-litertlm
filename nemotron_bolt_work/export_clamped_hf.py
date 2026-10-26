#!/usr/bin/env python3
"""export_clamped_hf.py <export_hf flags...> - litert-torch's stock `python -m litert_torch.generative.export_hf`
(export_main.main -> fire.Fire(export)) with every blockwise int4 block scale floored at 2^-14 before the weights are
quantized. XNNPACK refuses zero and subnormal fp16 block scales; on NVIDIA-Nemotron-3.5-Bolt-E4B the plain min-max
blockwise-32 recipe would write 13 of them (in 4 of the 344 int4 weights), and the floor raises exactly those.

The floor touches only blockwise 4-bit symmetric requests (int8, channelwise and the int4 channelwise embedding tables
pass through); an asymmetric blockwise int4 request stops the export. Each floored tensor is logged; the totals go to
stdout at exit as `CLAMP_TOTAL calls=<blockwise int4 calls> tensors=<floored tensors> scales=<floored scales>`.
ai-edge-quantizer-nightly 0.10.0.dev20261013's tensor_zp_scale_from_min_max takes a 7th argument (`signed`, passed by
keyword), so the wrapper passes *args / **kwargs through unchanged.
"""
import atexit
import sys

from litert_torch.generative.export_hf import export_main  # the export's own import order first
import numpy as np
from ai_edge_quantizer.algorithms.uniform_quantize import uniform_quantize_tensor as _u

_FLOOR = np.float32(2.0 ** -14)
_orig = _u.tensor_zp_scale_from_min_max
_stats = {"calls": 0, "tensors": 0, "scales": 0}


def _clamped(min_value, max_value, num_bits, symmetric, granularity, *args, **kwargs):
    zp, scale = _orig(min_value, max_value, num_bits, symmetric, granularity, *args, **kwargs)
    if _u.is_blockwise(granularity) and num_bits == 4:
        _stats["calls"] += 1
        if not symmetric:
            raise ValueError("export_clamped_hf: asymmetric blockwise int4 is not covered by this floor")
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
                              f"scales={_stats['scales']}", flush=True))
print("PATCHED uniform_quantize_tensor.tensor_zp_scale_from_min_max: blockwise int4 scales floored at 2^-14", flush=True)

sys.argv = ["litert_torch.generative.export_hf"] + sys.argv[1:]
export_main.main(None)
