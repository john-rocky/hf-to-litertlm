#!/usr/bin/env python3
"""quant_fp16w.py IN.tflite OUT.tflite - casts every FULLY_CONNECTED weight of a float graph to fp16 with
ai-edge-quantizer (0.10.0; FLOAT_CASTING, weight-only: a DEQUANTIZE in front of each FC, activations stay fp32).
Embedding tables, convolutions and the other constants stay fp32. This is the `.litertlm` file's audio section."""
import sys


def recipe():
  from ai_edge_quantizer import qtyping, recipe_manager  # pylint: disable=g-import-not-at-top
  from ai_edge_quantizer.algorithm_manager import AlgorithmName  # pylint: disable=g-import-not-at-top
  rm = recipe_manager.RecipeManager()
  rm.add_weight_only_config(regex=".*", operation_name=qtyping.TFLOperationName.FULLY_CONNECTED, num_bits=16,
                            granularity=qtyping.QuantGranularity.CHANNELWISE,
                            algorithm_key=AlgorithmName.FLOAT_CASTING)
  return rm.get_quantization_recipe()


def main():
  import pathlib  # pylint: disable=g-import-not-at-top
  from ai_edge_quantizer import quantizer  # pylint: disable=g-import-not-at-top
  src, dst = pathlib.Path(sys.argv[1]).resolve(), pathlib.Path(sys.argv[2]).resolve()
  qt = quantizer.Quantizer(str(src), recipe())
  res = qt.quantize()
  dst.parent.mkdir(parents=True, exist_ok=True)
  res.export_model(str(dst), overwrite=True)
  print(f"{dst}: {dst.stat().st_size} B")


if __name__ == "__main__":
  main()
