#!/usr/bin/env python3
"""The fp16 weight-only Qwen3.5 hybrid decoder: the float PREFILL_DECODE export (export_qwen35vl_decoder.py writes it as
model.tflite before its int8 step) with every FULLY_CONNECTED weight cast to fp16 (ai-edge-quantizer FLOAT_CASTING,
16 bits, float compute, an explicit DEQUANTIZE per weight), except the FCs whose output tensor name matches
'Linear_in_proj_(a|b);' (the 36 GatedDeltaNet decay/gate projections, kept fp32 like the int8 file's float set).
Written straight to DST (quantize(serialize_to_path=...)); about four times the input size in RAM (3.07 GB in, 11.9 GiB
peak RSS with ai-edge-quantizer 0.8.0).

  python quant_fp16w.py <decoder dir>/model.tflite <decoder dir>/model_fp16w.tflite
"""
import os
import sys
import time

from ai_edge_quantizer import qtyping, quantizer, recipe_manager
from ai_edge_quantizer.algorithm_manager import AlgorithmName

FLOAT_FC = "Linear_in_proj_(a|b);"


def main():
  src, dst = sys.argv[1], sys.argv[2]
  t0 = time.time()
  rm = recipe_manager.RecipeManager()
  rm.add_weight_only_config(regex=".*", operation_name=qtyping.TFLOperationName.FULLY_CONNECTED, num_bits=16,
                            algorithm_key=AlgorithmName.FLOAT_CASTING)
  rm.add_quantization_config(regex=FLOAT_FC, operation_name=qtyping.TFLOperationName.FULLY_CONNECTED,
                             algorithm_key=AlgorithmName.NO_QUANTIZE, op_config=None)
  recipe = rm.get_quantization_recipe()
  print("recipe", recipe, flush=True)
  q = quantizer.Quantizer(src, recipe)
  q.quantize(serialize_to_path=dst)
  print("FP16W_DONE", dst, os.path.getsize(dst), "bytes", round(time.time() - t0, 1), "s", flush=True)


if __name__ == "__main__":
  main()
