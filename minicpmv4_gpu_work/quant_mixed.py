#!/usr/bin/env python3
"""quant_mixed.py IN.tflite OUT.tflite [--float-fc REGEX ...] [--fp16-fc REGEX ...] [--json OUT.json]: quant.py's
wi4b32_wi8 recipe (every FULLY_CONNECTED int4 blockwise-32 min-max, the lm_head int8 per channel, block
scales floored at 2^-14 by its clamp patch) with NO_QUANTIZE overrides on the FCs whose scope matches a --float-fc regex
(minicpm5_work/quantize_minicpm5.py build_recipe float_fc_regexes; the scope is the op's output tensor name + ";", so
anchor with ";"). --fp16-fc gives the matching FCs fp16 weights instead (FLOAT_CASTING weight-only: a DEQUANTIZE feeds
a float FC). quant.py is loaded by path and not changed. Run with PYTHONPATH=../muse_glimmer_work/shim (reproduce.sh)."""
import argparse
import importlib.util
import json
import os
import time

MQ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quant.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--float-fc", action="append", default=[])
    # FCs whose scope matches get fp16 weights (FLOAT_CASTING weight-only: a DEQUANTIZE feeds a float FC) instead of
    # int4, appended after the base recipe so the override wins as the NO_QUANTIZE ones do
    ap.add_argument("--fp16-fc", action="append", default=[])
    ap.add_argument("--json")
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("quant_mixed_base", MQ)
    mq = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mq)
    stats = {"tensors": 0, "scales": 0, "blockwise_int4_calls": 0}
    mq.clamp_patch(stats)
    q5 = mq.load_by_path("quantize_minicpm5_recipe", os.path.join(mq.REPO, "minicpm5_work", "quantize_minicpm5.py"))
    q5.LM_HEAD_REGEX = mq.MUSE_LM_HEAD
    recipe = q5.build_recipe("wi4b32_wi8", "minmax", a.float_fc)
    if a.fp16_fc:
        from ai_edge_quantizer import qtyping, recipe_manager
        from ai_edge_quantizer.algorithm_manager import AlgorithmName
        rm16 = recipe_manager.RecipeManager()
        for rx in a.fp16_fc:
            rm16.add_weight_only_config(regex=rx, operation_name=qtyping.TFLOperationName.FULLY_CONNECTED, num_bits=16,
                                        granularity=qtyping.QuantGranularity.CHANNELWISE,
                                        algorithm_key=AlgorithmName.FLOAT_CASTING)
        recipe = list(recipe) + list(rm16.get_quantization_recipe())
    from ai_edge_quantizer import quantizer
    t0 = time.time()
    qt = quantizer.Quantizer(a.inp, recipe)
    if qt.need_calibration:
        raise SystemExit("recipe unexpectedly needs calibration")
    qt.quantize(serialize_to_path=a.out)
    rec = {"in": a.inp, "in_bytes": os.path.getsize(a.inp), "out": a.out, "out_bytes": os.path.getsize(a.out),
           "kind": "wi4b32_wi8 + NO_QUANTIZE / fp16 overrides", "float_fc": a.float_fc, "fp16_fc": a.fp16_fc, "clamp_floor_2^-14": stats,
           "seconds": round(time.time() - t0, 1), "recipe": recipe}
    print(f"QUANT_DONE {rec['in_bytes']} -> {rec['out_bytes']} B in {rec['seconds']} s; float {a.float_fc}; fp16 {a.fp16_fc}; clamp {stats}",
          flush=True)
    if a.json:
        json.dump(rec, open(a.json, "w"), indent=1, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
