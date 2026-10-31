#!/usr/bin/env python3
"""tfl_int8_weights.py IN.tflite OUT.tflite [--ops FULLY_CONNECTED,CONV_2D] [--json OUT.json]

Dynamic-range int8 with ai-edge-quantizer 0.10.0: the weights of the named op types become int8 per output channel
(symmetric, min-max), and the activations stay float32 (a GPU keeps them in float; the CPU's XNNPACK quantizes them to
int8 at run time). Only the named op types are quantized, by default FULLY_CONNECTED and CONV_2D: the recipe's
ALL_SUPPORTED set would also take DEPTHWISE_CONV_2D and constant tables of BATCH_MATMUL / MUL / ADD, which stay float32
here. The result is written straight to OUT (`quantize(serialize_to_path=...)`).
"""
import argparse
import json
import os
import sys
import time


def build(ops):
    from ai_edge_quantizer import recipe_manager, qtyping
    from ai_edge_quantizer.algorithm_manager import AlgorithmName
    G, OP = qtyping.QuantGranularity, qtyping.TFLOperationName
    rm = recipe_manager.RecipeManager()
    for name in ops:
        rm.add_dynamic_config(regex=".*", operation_name=getattr(OP, name), num_bits=8,
                              granularity=G.CHANNELWISE, algorithm_key=AlgorithmName.MIN_MAX_UNIFORM_QUANT)
    return rm.get_quantization_recipe()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--ops", default="FULLY_CONNECTED,CONV_2D")
    ap.add_argument("--json")
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"refusing to overwrite {a.out}")
    ops = [o for o in a.ops.split(",") if o]
    recipe = build(ops)
    from ai_edge_quantizer import quantizer
    t0 = time.time()
    qt = quantizer.Quantizer(a.inp, recipe)
    if qt.need_calibration:
        raise SystemExit("recipe unexpectedly needs calibration")
    qt.quantize(serialize_to_path=a.out)
    rec = {"in": a.inp, "in_bytes": os.path.getsize(a.inp), "out": a.out, "out_bytes": os.path.getsize(a.out),
           "ops": ops, "seconds": round(time.time() - t0, 1), "recipe": recipe}
    print(f"QUANT_DONE {ops} {rec['in_bytes']} -> {rec['out_bytes']} B in {rec['seconds']} s", flush=True)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rec, f, indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
