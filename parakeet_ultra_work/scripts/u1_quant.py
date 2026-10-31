#!/usr/bin/env python3
"""CPU quantization forms of the ultra encoder (venv .venv-q: ai-edge-quantizer 0.10.0, ai-edge-litert 2.3.0). No model
execution. The weights are dense (not ternary), so every form here is lossy and is judged by the WER lines.

  int8  <fp32 src> <dst>               recipe.dynamic_wi8_afp32() = int8 weights, one scale per output channel, every
                                       op aeq supports; activations quantized to int8 at run time (XNNPACK qd8)
  int4  <fp32 src> <dst> --block 32|128
                                       FULLY_CONNECTED: min-max int4 symmetric BLOCKWISE_<block>, with the fp16
                                       block scale floored at 2^-14 BEFORE the weights are quantized (clamp-before-
                                       quantize: patch_floor() below wraps aeq 0.10.0's
                                       tensor_zp_scale_from_min_max);
                                       CONV_2D (the two 1x1 convs of the subsampling: aeq's int4 blockwise skips
                                       CONV_2D, so they get a rule of their own): int8 channelwise dynamic, so
                                       no CONV_2D weight stays FLOAT32. DEPTHWISE_CONV_2D stays float (not taken).
After writing: a scan of the file = per quantized op its weight dtype and granularity; for every blockwise INT4 tensor
the block scales (count, zero, subnormal < 2^-14, min, max); the dtype histogram of the FULLY_CONNECTED / CONV_2D /
DEPTHWISE_CONV_2D / BATCH_MATMUL constant weights (r2_fb.census). Exit 3 when a blockwise scale is 0 or subnormal or a
CONV_2D weight is FLOAT32 (int4), or nothing was quantized.
CLI: u1_quant.py int8 <src> <dst> --json <out>   |   u1_quant.py int4 <src> <dst> --block 32 --json <out>
"""
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import int4_check as C  # noqa: E402
import r2_fb  # noqa: E402
from ai_edge_litert import schema_py_generated as S  # noqa: E402
from ai_edge_litert.tools import mmap_utils  # noqa: E402

FLOOR = np.float32(2.0 ** -14)
STATS = {"tensors_floored": 0, "scales_floored": 0}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def patch_floor():
    from ai_edge_quantizer.algorithms.uniform_quantize import uniform_quantize_tensor as u
    orig = u.tensor_zp_scale_from_min_max

    def clamped(min_value, max_value, num_bits, symmetric, granularity, clipping_values=None, signed=True):
        zp, scale = orig(min_value, max_value, num_bits, symmetric, granularity, clipping_values, signed)
        if u.is_blockwise(granularity) and num_bits == 4:
            if not symmetric:
                raise ValueError("u1_quant: asymmetric blockwise int4 is not covered by this floor")
            low = scale < FLOOR
            n = int(np.count_nonzero(low))
            if n:
                scale = np.where(low, FLOOR, scale).astype(scale.dtype)
                STATS["tensors_floored"] += 1
                STATS["scales_floored"] += n
        return zp, scale

    u.tensor_zp_scale_from_min_max = clamped


def int4_recipe(block):
    gran = {32: "BLOCKWISE_32", 128: "BLOCKWISE_128"}[block]
    return [
        {"regex": ".*", "operation": "FULLY_CONNECTED", "algorithm_key": "min_max_uniform_quantize",
         "op_config": {"weight_tensor_config": {"num_bits": 4, "symmetric": True, "granularity": gran, "dtype": "INT"},
                       "compute_precision": "INTEGER", "explicit_dequantize": False, "skip_checks": False,
                       "min_weight_elements": 0}},
        {"regex": ".*", "operation": "CONV_2D", "algorithm_key": "min_max_uniform_quantize",
         "op_config": {"weight_tensor_config": {"num_bits": 8, "symmetric": True, "granularity": "CHANNELWISE",
                                                "dtype": "INT"},
                       "compute_precision": "INTEGER", "explicit_dequantize": False, "skip_checks": False,
                       "min_weight_elements": 0}},
    ]


def scan(path):
    data = mmap_utils.get_file_contents(path)
    m = S.Model.GetRootAs(data, 0)
    rows, blk = [], {"tensors": 0, "count": 0, "zero": 0, "subnormal": 0, "min": None, "max": None}
    for si in range(m.SubgraphsLength()):
        sg = m.Subgraphs(si)
        for i in range(sg.OperatorsLength()):
            op = sg.Operators(i)
            oc = m.OperatorCodes(op.OpcodeIndex())
            name = C.BO.get(max(oc.BuiltinCode(), oc.DeprecatedBuiltinCode()))
            if name not in ("FULLY_CONNECTED", "CONV_2D", "DEPTHWISE_CONV_2D", "BATCH_MATMUL", "EMBEDDING_LOOKUP"):
                continue
            if op.InputsLength() < 2 or op.Inputs(1) < 0:
                continue
            t = sg.Tensors(op.Inputs(1))
            if m.Buffers(t.Buffer()).DataLength() == 0 and m.Buffers(t.Buffer()).Size() == 0:
                continue  # a runtime operand, not a constant weight
            q = t.Quantization()
            row = {"op": name, "dtype": C.TT.get(t.Type()), "shape": t.ShapeAsNumpy().tolist(), "gran": None}
            if q is not None and q.DetailsType() == S.QuantizationDetails.BlockwiseQuantization:
                bq = S.BlockwiseQuantization()
                tab = q.Details()
                bq.Init(tab.Bytes, tab.Pos)
                st = sg.Tensors(bq.Scales())
                sc = np.frombuffer(C.buf_bytes(m, data, st.Buffer()), np.float16).astype(np.float32)
                row.update({"gran": f"blockwise{int(bq.BlockSize())}", "scales_dtype": C.TT.get(st.Type()),
                            "n_scales": int(sc.size), "zero": int((sc == 0).sum()),
                            "subnormal": int(((sc > 0) & (sc < FLOOR)).sum()), "min_scale": float(sc.min())})
                blk["tensors"] += 1
                blk["count"] += int(sc.size)
                blk["zero"] += row["zero"]
                blk["subnormal"] += row["subnormal"]
                blk["min"] = float(sc.min()) if blk["min"] is None else min(blk["min"], float(sc.min()))
                blk["max"] = float(sc.max()) if blk["max"] is None else max(blk["max"], float(sc.max()))
            elif q is not None and q.ScaleLength():
                row["gran"] = "channelwise" if q.ScaleLength() > 1 else "tensor"
                row["n_scales"] = int(q.ScaleLength())
            rows.append(row)
    hist = {}
    for r in rows:
        k = f"{r['op']}:{r['dtype']}:{r['gran']}"
        hist[k] = hist.get(k, 0) + 1
    return {"weight_hist": hist, "blockwise_scales": blk, "conv2d_float32": sum(
        1 for r in rows if r["op"] == "CONV_2D" and r["dtype"] == "FLOAT32")}


def main(argv):
    kind, src, dst = argv[1], argv[2], argv[3]
    out_json = argv[argv.index("--json") + 1]
    block = int(argv[argv.index("--block") + 1]) if "--block" in argv else None
    from ai_edge_quantizer import quantizer, recipe
    t0 = time.time()
    if kind == "int8":
        qt = quantizer.Quantizer(src)
        qt.load_quantization_recipe(recipe.dynamic_wi8_afp32())
    elif kind == "int4":
        patch_floor()
        qt = quantizer.Quantizer(src, int4_recipe(block))
    else:
        raise SystemExit(kind)
    tmp = dst[:-7] + ".tmp.tflite"
    qt.quantize().export_model(tmp, overwrite=True)
    os.replace(tmp, dst)
    q_s = round(time.time() - t0, 1)
    rep = {"kind": kind, "block": block, "src": src, "src_bytes": os.path.getsize(src), "dst": dst,
           "bytes": os.path.getsize(dst), "sha256": sha256(dst), "quantize_s": q_s, "floor": dict(STATS),
           "scan": scan(dst), "census": r2_fb.census(dst)}
    bad = []
    s = rep["scan"]
    if not any((":INT8:" in k or ":INT4:" in k) for k in s["weight_hist"]):
        bad.append("nothing quantized")
    if kind == "int4":
        if s["blockwise_scales"]["zero"] or s["blockwise_scales"]["subnormal"]:
            bad.append("zero / subnormal blockwise scale")
        if s["conv2d_float32"]:
            bad.append("CONV_2D weight left FLOAT32")
        if not s["blockwise_scales"]["tensors"]:
            bad.append("no blockwise tensor")
    rep["bad"] = bad
    json.dump(rep, open(out_json, "w"), indent=1, default=str)
    print(json.dumps({k: rep[k] for k in ("kind", "block", "bytes", "quantize_s", "floor", "bad")}),
          json.dumps(s["weight_hist"]), json.dumps(s["blockwise_scales"]), flush=True)
    return 3 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
