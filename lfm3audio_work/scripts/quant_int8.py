#!/usr/bin/env python3
"""quant_int8.py IN.tflite OUT.tflite - ai-edge-quantizer (0.10.0)'s dynamic_wi8_afp32 recipe on a float graph:
int8 per-channel weights for every op the recipe supports (here the 36 FULLY_CONNECTED layers and the
EMBEDDING_LOOKUP table of the depth decoder's pass) and float32 activations, which the recipe leaves to be quantized
to int8 on the fly during inference. This is lfm3_audio_depth_pass_r4b_int8.tflite, from lfm3_audio_depth_pass_r4b.tflite."""
import sys


def main():
  import pathlib  # pylint: disable=g-import-not-at-top
  from ai_edge_quantizer import quantizer, recipe  # pylint: disable=g-import-not-at-top
  src, dst = pathlib.Path(sys.argv[1]).resolve(), pathlib.Path(sys.argv[2]).resolve()
  qt = quantizer.Quantizer(str(src), recipe.dynamic_wi8_afp32())
  res = qt.quantize()
  dst.parent.mkdir(parents=True, exist_ok=True)
  res.export_model(str(dst), overwrite=True)
  print(f"{dst}: {dst.stat().st_size} B")


if __name__ == "__main__":
  main()
