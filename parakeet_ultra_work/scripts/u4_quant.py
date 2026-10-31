#!/usr/bin/env python3
"""The int8f CPU form of the parakeet-ultra encoder (round 4; venv .venv-q: ai-edge-quantizer
0.10.0, ai-edge-litert 2.3.0). No model execution. u1_quant.py stays as it is; this file takes its scan and its form.

  int8f <fp32 src> <dst> [--ops FULLY_CONNECTED,CONV_2D]
        recipe.dynamic_wi8_afp32()'s own op config (int8 weights, symmetric, one scale per output channel, compute
        precision INTEGER without explicit dequantize = activations quantized to int8 at run time, XNNPACK qd8) on
        the ops of --ops only, instead of every op aeq supports: round 1's int8 file also holds the 24 constant
        position-table BATCH_MATMUL operands int8 per channel, this form keeps them FLOAT32 (and DEPTHWISE_CONV_2D
        unless --ops names it).
After writing: u1_quant.scan (per quantized op its weight dtype and granularity) and r2_fb.census, then the checks:
every FULLY_CONNECTED / CONV_2D constant weight of the source is INT8 channelwise in the file (the same count as the
source), every BATCH_MATMUL constant stays FLOAT32, DEPTHWISE_CONV_2D INT8 when --ops names it else FLOAT32, the op
histogram equals the source's. Exit 3 when a check fails.
CLI: u4_quant.py int8f <src> <dst> --json <out> [--ops FULLY_CONNECTED,CONV_2D[,DEPTHWISE_CONV_2D]]
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r2_fb  # noqa: E402
import u1_quant as Q  # noqa: E402

KNOWN = ("FULLY_CONNECTED", "CONV_2D", "DEPTHWISE_CONV_2D", "BATCH_MATMUL")


def int8f_recipe(ops):
    from ai_edge_quantizer import recipe
    base = recipe.dynamic_wi8_afp32()
    if len(base) != 1 or base[0].get("operation") != "*":
        raise SystemExit(f"u4_quant: dynamic_wi8_afp32() is not one '*' entry: {base}")
    return [dict(base[0], operation=op) for op in ops], base


def main(argv):
    kind, src, dst = argv[1], argv[2], argv[3]
    if kind != "int8f":
        raise SystemExit(kind)
    out_json = argv[argv.index("--json") + 1]
    ops = (argv[argv.index("--ops") + 1] if "--ops" in argv else "FULLY_CONNECTED,CONV_2D").split(",")
    if any(o not in KNOWN for o in ops) or "BATCH_MATMUL" in ops:
        raise SystemExit(f"--ops {ops}: only FULLY_CONNECTED / CONV_2D / DEPTHWISE_CONV_2D")
    from ai_edge_quantizer import quantizer
    rcp, base = int8f_recipe(ops)
    t0 = time.time()
    qt = quantizer.Quantizer(src, rcp)
    tmp = dst[:-7] + ".tmp.tflite"
    qt.quantize().export_model(tmp, overwrite=True)
    os.replace(tmp, dst)
    q_s = round(time.time() - t0, 1)
    src_c, dst_c = r2_fb.census(src), r2_fb.census(dst)
    rep = {"kind": kind, "ops": ops, "recipe": rcp, "recipe_base_dynamic_wi8_afp32": base, "src": src,
           "src_bytes": os.path.getsize(src), "dst": dst, "bytes": os.path.getsize(dst), "sha256": Q.sha256(dst),
           "quantize_s": q_s, "scan": Q.scan(dst), "census": dst_c,
           "src_const_weight_dtypes": src_c["const_weight_dtypes"]}
    cw, sw = dst_c["const_weight_dtypes"], src_c["const_weight_dtypes"]
    hist = rep["scan"]["weight_hist"]
    bad = []
    for op in KNOWN:
        n_src = sum(v for k, v in sw.items() if k.split(":")[0] == op)
        want = "INT8" if op in ops else "FLOAT32"
        n_want = cw.get(f"{op}:{want}", 0)
        if n_want != n_src or sum(v for k, v in cw.items() if k.split(":")[0] == op) != n_src:
            bad.append(f"{op}: {n_want} of {n_src} constants {want} ({ {k: v for k, v in cw.items() if k.startswith(op)} })")
        if want == "INT8" and hist.get(f"{op}:INT8:channelwise", 0) != n_src:
            bad.append(f"{op}: {hist.get(f'{op}:INT8:channelwise', 0)} of {n_src} INT8 channelwise")
    if dst_c["ops"] != src_c["ops"]:
        bad.append(f"op histogram changed: {src_c['ops']} -> {dst_c['ops']}")
    rep["bad"] = bad
    json.dump(rep, open(out_json, "w"), indent=1, default=str)
    print(json.dumps({k: rep[k] for k in ("kind", "ops", "bytes", "sha256", "quantize_s", "bad")}),
          json.dumps(hist), json.dumps(cw), flush=True)
    return 3 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
