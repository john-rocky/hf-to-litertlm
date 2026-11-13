"""Shared quantization helpers (quant_tflite.py, graft_moe.py).

floored_blockwise_scales(): context manager that floors every 4-bit blockwise symmetric scale at 2^-14 inside
litert_quantizer's own scale function (uniform_quantize_tensor.tensor_zp_scale_from_min_max, which both the FC path and
the MoE path reach: materialize_moe -> naive_min_max_quantize.get_tensor_quant_params), counting calls / scales /
floored scales (a scale of 0 for an all-zero block would make the kernel divide by zero).

expert_quant_params(w, bits, block): the quantizer's own parameters for one MoE expert weight in the op's layout
[out, E, 1, in] = what materialize_moe computes for a `moe` input under the recipe entry
{operation CUSTOM_OP, num_bits, symmetric, BLOCKWISE_<block>, INT, quantized_dimensions [0, 1]}
(naive_min_max_quantize.get_tensor_quant_params with no calibration statistics: min / max per block of `block`
inputs, the min-max scale, uniform_quantize). Returns UniformQuantParams with quantized_data.
"""
import contextlib

import numpy as np

FLOOR = np.float32(2.0 ** -14)


@contextlib.contextmanager
def floored_blockwise_scales(stats):
  from litert_quantizer.algorithms.uniform_quantize import uniform_quantize_tensor as u
  orig = u.tensor_zp_scale_from_min_max

  def floored(min_value, max_value, num_bits, symmetric, granularity, *a, **k):
    zp, scale = orig(min_value, max_value, num_bits, symmetric, granularity, *a, **k)
    if "BLOCKWISE" in str(granularity) and num_bits == 4:
      if not symmetric:
        raise ValueError("asymmetric blockwise int4 is not covered by the floor")
      stats["calls"] = stats.get("calls", 0) + 1
      low = scale < FLOOR
      stats["scales"] = stats.get("scales", 0) + int(np.size(scale))
      n = int(np.count_nonzero(low))
      if n:
        stats["floored"] = stats.get("floored", 0) + n
        scale = np.where(low, FLOOR, scale).astype(np.asarray(scale).dtype)
    return zp, scale

  u.tensor_zp_scale_from_min_max = floored
  try:
    yield stats
  finally:
    u.tensor_zp_scale_from_min_max = orig


def expert_weight_config(bits, block):
  """block > 0: BLOCKWISE_<block>; block == 0: CHANNELWISE over (0, 1) = one scale per (output row, expert), the `moe`
  op's per-output-channel case (scale [out, E, 1, 1])."""
  from litert_quantizer import qtyping
  gran = qtyping.QuantGranularity.CHANNELWISE if block == 0 else qtyping.QuantGranularity(f"BLOCKWISE_{block}")
  return qtyping.TensorQuantizationConfig(
      num_bits=bits, symmetric=True, granularity=gran,
      dtype=qtyping.TensorDataType.INT, quantized_dimensions=(0, 1))


def expert_quant_params(w, bits, block):
  from litert_quantizer import qtyping
  from litert_quantizer.algorithms.uniform_quantize import naive_min_max_quantize as nmm
  cfg = expert_weight_config(bits, block)
  op_info = qtyping.OpInfo(
      op=None, op_name=qtyping.TFLOperationName.CUSTOM_OP, subgraph_op_index=0,
      op_quant_config=qtyping.OpQuantizationConfig(weight_tensor_config=cfg,
                                                   compute_precision=qtyping.ComputePrecision.INTEGER))
  return nmm.get_tensor_quant_params(op_info, cfg, w)
