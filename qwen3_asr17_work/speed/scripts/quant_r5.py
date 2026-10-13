#!/usr/bin/env python3
"""Round 5 of the Qwen3-ASR-1.7B speed ladder: quant_r3.py (the lane's quant_sections.py with round 3's recipe names)
with the output root moved to out/r5 (out/r5/sections/<tag>/, out/r5/unpack/<arm>/, out/r5/export/<arm>/model.litertlm),
so the round's arm never shares a path with round 3's files. The quantization code is quant_sections.py's, unchanged.

  cd qwen3_asr17_work && PYTHONPATH=out/pyoverlay_builder017 ~/venvs/ltmain0918/bin/python speed/scripts/quant_r5.py \
      --src out/r5/unpack/f32_pb128_256 --tag F6_30s_pb --arm F6_C --lm dynamic_wi8_afp32 --emb dynamic_wi8_afp32 \
      --enc fp16 --pack
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quant_r3  # noqa: E402  (sets quant_sections.recipe_of = recipe_of_r3 and OUT = out/r3)

quant_r3.quant_sections.OUT = os.path.join(quant_r3.LANE, "out", "r5")

if __name__ == "__main__":
    quant_r3.quant_sections.main()
