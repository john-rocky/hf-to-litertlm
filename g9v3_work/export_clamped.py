#!/usr/bin/env python3
"""export_clamped.py <args of scripts/export_simple_template.py> - the same export with a clamp-before-quantize minimum on
blockwise int4 scales, used by convert_g9v3_3b.sh for the int4 file.

Why: XNNPACK checks std::isnormal on every blockwise scale and refuses a file with a zero or subnormal fp16 scale (CPU), and a
zero scale also hides a w/0 payload. On G9v3-3B both int4 paths produce such scales without the minimum:
  - min-max / OCTAV blockwise-32 (`tensor_zp_scale_from_min_max`): 1664 zero and 1970 subnormal fp16 block scales (layer 0
    gate / up, layer 51 down_proj, o_proj of several layers);
  - the GPTQ recovery path (recipes/gptqrec_int4_block128.json, `get_zp_scale_from_dequantized_symmetric_weights`), which derives
    a block's scale from the levels present: 365 GPTQ group scales below 2^-14 (layer 51 down_proj / o_proj) give subnormal
    fp16 scales, and 416 all-zero 128-blocks (layer 0 gate / up) get min_scale 1e-9 = 0 in fp16.
What the minimum does: blockwise 4-bit symmetric scales below 2^-14 (the fp16 min normal) are raised to 2^-14 before the caller
quantizes the weights with them = clamp-before-quantize (a dead block's payload is the nearest integer to w / 2^-14, 0 for |w| < 2^-15; in the
recovery path the error stays <= 2^-15 per weight, inside the recovery's 1e-4 absolute check). Channelwise / int8 /
asymmetric requests pass through unchanged. Counts go to stdout: `CLAMP min 2^-14: …` per tensor and `CLAMP_TOTAL` /
`CLAMP_REC_TOTAL` at exit.

Import order = as in the export script (transformers, then litert_torch export_lib, then the quantizer); then the script runs
via runpy as __main__ with the same argv, so every module it imports is the patched one."""
import atexit
import os
import runpy
import sys
import types

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts", "export_simple_template.py")

import transformers  # noqa: E402,F401  (the export script's first import; keep the order)
from litert_torch.generative.export_hf.core import export_lib  # noqa: E402,F401
import numpy as np  # noqa: E402
from ai_edge_quantizer.algorithms.uniform_quantize import uniform_quantize_tensor as _u  # noqa: E402
from ai_edge_quantizer.algorithms.uniform_quantize import dequantized_weight_recovery as _dwr  # noqa: E402

_MIN = np.float32(2.0 ** -14)

# --- min-max / OCTAV path
_orig = _u.tensor_zp_scale_from_min_max
_stats = {"tensors": 0, "scales": 0}


def _clamped(min_value, max_value, num_bits, symmetric, granularity, clipping_values=None):
    zp, scale = _orig(min_value, max_value, num_bits, symmetric, granularity, clipping_values)
    if _u.is_blockwise(granularity) and num_bits == 4:
        if not symmetric:
            raise ValueError("export_clamped: asymmetric blockwise int4 is not covered by this minimum")
        low = scale < _MIN
        n = int(np.count_nonzero(low))
        if n:
            scale = np.where(low, _MIN, scale).astype(scale.dtype)
            _stats["tensors"] += 1
            _stats["scales"] += n
            print(f"CLAMP min 2^-14: {n} of {scale.size} block scales", flush=True)
    return zp, scale


_u.tensor_zp_scale_from_min_max = _clamped
atexit.register(lambda: print(f"CLAMP_TOTAL tensors={_stats['tensors']} scales={_stats['scales']}", flush=True))

# --- GPTQ recovery path. The wrapper is re-applied whenever the module attribute is assigned (a module class with a
# __setattr__ hook), so it stays the outermost layer after the export script's GPTQREC_GCD_FIX wrapper replaces the function
# (the GCD refinement divides a block scale by k and could otherwise bring a raised block back below 2^-14). Counts: the
# innermost layer = the blocks below 2^-14 the recovery produced; an outer layer = blocks the GCD refinement took below again.
_REC = "get_zp_scale_from_dequantized_symmetric_weights"
_REC_ORIG = getattr(_dwr, _REC)
_rstats = {"tensors": 0, "scales": 0, "refloored_after_gcd": 0}


def _min_rec(fn):
    if getattr(fn, "_raised", False):
        return fn
    inner = fn is _REC_ORIG

    def _raised(dequant_vals, quantized_dimension=None, block_size=0, min_scale=1e-9):
        zp, scales = fn(dequant_vals, quantized_dimension, block_size, min_scale)
        if block_size:
            low = scales < _MIN
            n = int(np.count_nonzero(low))
            if n:
                scales = np.where(low, _MIN, scales).astype(scales.dtype)
                if inner:
                    _rstats["tensors"] += 1
                    _rstats["scales"] += n
                    print(f"CLAMP_REC min 2^-14: {n} of {scales.size} recovered block scales", flush=True)
                else:
                    _rstats["refloored_after_gcd"] += n
                    print(f"CLAMP_REC min 2^-14 again after the GCD refinement: {n} block scales", flush=True)
        return zp, scales

    _raised._raised = True
    return _raised


class _FloorModule(types.ModuleType):
    def __setattr__(self, name, value):
        if name == _REC and callable(value):
            value = _min_rec(value)
        super().__setattr__(name, value)


_dwr.__class__ = _FloorModule
setattr(_dwr, _REC, _REC_ORIG)
atexit.register(lambda: print(f"CLAMP_REC_TOTAL tensors={_rstats['tensors']} scales={_rstats['scales']} "
                              f"refloored_after_gcd={_rstats['refloored_after_gcd']}", flush=True))
print("PATCHED blockwise int4 scale minimum 2^-14 (min-max path and GPTQ recovery path)", flush=True)

sys.argv = [SCRIPT] + sys.argv[1:]
runpy.run_path(SCRIPT, run_name="__main__")
