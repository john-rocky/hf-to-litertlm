#!/usr/bin/env python3
"""muse_quant.py IN.tflite OUT.tflite KIND [--lm-head-regex RX] [--json OUT.json]

Quantizes one exported section with ai-edge-quantizer:
  wi4b128_wi8 / wi4b32_wi8  every FULLY_CONNECTED int4 blockwise-128 / -32 min-max, the lm_head FC int8 per
                            channel (the Muse wrapper's lm_head output is named "...ForCausalLM_model;"), embedding
                            lookups int8 per channel
  wi8                       every supported op int8 (dynamic, per channel)
  emb8                      EMBEDDING_LOOKUP int8 per channel only (the EMBEDDER section)
Block scales of int4 blockwise tensors are floored at 2^-14 while quantizing (a scale below 2^-14 is zero or subnormal
in fp16: XNNPACK refuses such FC scales, and a zero-scale block holds w / 0), and the output is serialized straight
into the output file (`qt.quantize(serialize_to_path=...)`, a lower memory peak than export_model).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

MUSE_LM_HEAD = r"(decode_logits_output|Linear_lm_head|ForCausalLM_model;)"


def clamp_patch(stats):
    from ai_edge_quantizer.algorithms.uniform_quantize import uniform_quantize_tensor as u
    import inspect
    floor = np.float32(2.0 ** -14)
    orig = u.tensor_zp_scale_from_min_max
    sig = inspect.signature(orig)

    def clamped(*args, **kwargs):
        # forwards every argument (0.10.0 added `signed`); reads the three it needs by name
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        num_bits, symmetric, granularity = (bound.arguments[k] for k in ("num_bits", "symmetric", "granularity"))
        zp, scale = orig(*args, **kwargs)
        if u.is_blockwise(granularity) and num_bits == 4:
            if not symmetric:
                raise ValueError("asymmetric blockwise int4 is not covered by the floor")
            low = scale < floor
            k = int(np.count_nonzero(low))
            stats["blockwise_int4_calls"] += 1
            if k:
                scale = np.where(low, floor, scale).astype(scale.dtype)
                stats["tensors"] += 1
                stats["scales"] += k
        return zp, scale

    u.tensor_zp_scale_from_min_max = clamped


def build(kind, lm_head_regex):
    from ai_edge_quantizer import recipe_manager, qtyping
    from ai_edge_quantizer.algorithm_manager import AlgorithmName
    G, OP = qtyping.QuantGranularity, qtyping.TFLOperationName
    rm = recipe_manager.RecipeManager()
    if kind in ("wi4b128_wi8", "wi4b32_wi8"):
        blk = {"wi4b32_wi8": G.BLOCKWISE_32, "wi4b128_wi8": G.BLOCKWISE_128}[kind]
        rm.add_dynamic_config(regex=".*", operation_name=OP.FULLY_CONNECTED, num_bits=4, granularity=blk,
                              algorithm_key=AlgorithmName.MIN_MAX_UNIFORM_QUANT)
        rm.add_dynamic_config(regex=lm_head_regex, operation_name=OP.FULLY_CONNECTED, num_bits=8,
                              granularity=G.CHANNELWISE)
        rm.add_dynamic_config(regex=".*", operation_name=OP.EMBEDDING_LOOKUP, num_bits=8, granularity=G.CHANNELWISE)
        return rm.get_quantization_recipe()
    if kind == "wi8":
        rm.add_dynamic_config(regex=".*", operation_name=OP.ALL_SUPPORTED, num_bits=8)
    elif kind == "emb8":
        rm.add_dynamic_config(regex=".*", operation_name=OP.EMBEDDING_LOOKUP, num_bits=8,
                              granularity=G.CHANNELWISE, algorithm_key=AlgorithmName.MIN_MAX_UNIFORM_QUANT)
    else:
        raise SystemExit(f"unknown kind {kind}")
    return rm.get_quantization_recipe()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("kind")
    ap.add_argument("--lm-head-regex", default=MUSE_LM_HEAD)
    ap.add_argument("--json")
    a = ap.parse_args()
    stats = {"tensors": 0, "scales": 0, "blockwise_int4_calls": 0}
    clamp_patch(stats)
    recipe = build(a.kind, a.lm_head_regex)
    from ai_edge_quantizer import quantizer
    t0 = time.time()
    qt = quantizer.Quantizer(a.inp, recipe)
    if qt.need_calibration:
        raise SystemExit("recipe unexpectedly needs calibration")
    qt.quantize(serialize_to_path=a.out)
    rec = {"in": a.inp, "in_bytes": os.path.getsize(a.inp), "out": a.out, "out_bytes": os.path.getsize(a.out),
           "kind": a.kind, "lm_head_regex": a.lm_head_regex, "clamp_floor_2^-14": stats,
           "seconds": round(time.time() - t0, 1), "recipe": recipe}
    print(f"QUANT_DONE {a.kind} {rec['in_bytes']} -> {rec['out_bytes']} B in {rec['seconds']} s; clamp {stats}",
          flush=True)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rec, f, indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
