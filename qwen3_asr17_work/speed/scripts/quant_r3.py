#!/usr/bin/env python3
"""Round 3 of the Qwen3-ASR-1.7B speed ladder: the lane's quant_sections.py (unchanged code) with its output root moved
to out/r3 (out/r3/sections/<tag>/, out/r3/unpack/<arm>/, out/r3/export/<arm>/model.litertlm), so the round's arms never
share a path with the 10-03 files (out/export/q17_30s_C_lt is the ship file's symlink). recipe_of_r3() adds this round's
own recipe names and hands every other name to quant_sections.recipe_of.

  cd qwen3_asr17_work && PYTHONPATH=out/pyoverlay_builder017 ~/venvs/ltmain0918/bin/python speed/scripts/quant_r3.py \
      --src out/r3/unpack/F0_f32_30s --tag F0_30s --arm F0_C --lm dynamic_wi8_afp32 --emb dynamic_wi8_afp32 --enc fp16 --pack
"""
import os
import sys

LANE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, LANE)
import quant_sections  # noqa: E402

OUT_R3 = os.path.join(LANE, "out", "r3")
_recipe_of_lane = quant_sections.recipe_of

# Encoder: dynamic int8 (dynamic_wi8_afp32's config) on every op except the in-graph log-mel ops, which stay float.
# The mel ops are matched by their output tensor names (aeq's op scope = output tensor names), listed in MEL_SCOPE_RX
# once they are read from the fp32 encoder tflite (see ROUND3.md); a regex that matches nothing leaves the recipe equal
# to dynamic_wi8_afp32, so the caller compares the section bytes against that arm.
MEL_SCOPE_RX = os.environ.get("R3_MEL_SCOPE_RX", "")


def recipe_of_r3(name):
    if name == "dyn8_melf32":
        assert MEL_SCOPE_RX, "R3_MEL_SCOPE_RX (regex on the mel ops' output tensor names) is not set"
        from ai_edge_quantizer import qtyping, recipe_manager
        rm = recipe_manager.RecipeManager()
        rm.add_dynamic_config(regex=r"^(?!.*(?:" + MEL_SCOPE_RX + r")).*$",
                              operation_name=qtyping.TFLOperationName.ALL_SUPPORTED, num_bits=8)
        return rm.get_quantization_recipe()
    return _recipe_of_lane(name)


quant_sections.OUT = OUT_R3
quant_sections.recipe_of = recipe_of_r3

if __name__ == "__main__":
    quant_sections.main()
