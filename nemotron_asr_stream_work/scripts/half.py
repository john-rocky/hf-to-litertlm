#!/usr/bin/env python3
"""fp16ws form of the r1 conformer (venv .venv-q, no model execution;
the check written for an F32 checkpoint): ai-edge-quantizer 0.10.0 float_casting (fp16 weight-only, explicit
DEQUANTIZE, FLOAT compute) on every FULLY_CONNECTED, CONV_2D and DEPTHWISE_CONV_2D weight of the fp32 file. The folded
relative-position tables (BATCH_MATMUL constants) are not taken by float_casting and stay fp32.
Check: per weight tensor, the fp16 values against np.float16(the same op's fp32 weight) (matched by the
op's output names) = round-to-nearest-even of the fp32 source; counts per op type of tensors equal to that rounding,
of fp32 values above the fp16 range (|w| > 65,504) and of nonzero fp32 values that land on fp16 zero / subnormals.
CLI: half.py <fp32 src> <dst> --json results/r1/q_fp16ws.json
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
from ai_edge_quantizer import qtyping, quantizer  # noqa: E402

OPS = ("FULLY_CONNECTED", "CONV_2D", "DEPTHWISE_CONV_2D")


def weights_by_scope(path):
    data = mmap_utils.get_file_contents(path)
    m = S.Model.GetRootAs(data, 0)
    sg = m.Subgraphs(0)
    producer = {}
    for i in range(sg.OperatorsLength()):
        op = sg.Operators(i)
        for j in range(op.OutputsLength()):
            producer[op.Outputs(j)] = op
    out = {}
    for i in range(sg.OperatorsLength()):
        op = sg.Operators(i)
        oc = m.OperatorCodes(op.OpcodeIndex())
        name = C.BO.get(max(oc.BuiltinCode(), oc.DeprecatedBuiltinCode()))
        if name not in OPS:
            continue
        scope = ";".join(sg.Tensors(op.Outputs(j)).Name().decode() for j in range(op.OutputsLength()))
        ti = op.Inputs(1)
        t = sg.Tensors(ti)
        raw = C.buf_bytes(m, data, t.Buffer())
        if raw:
            dt = C.TT.get(t.Type())
            out[scope] = (name, dt, np.frombuffer(raw, np.float32 if dt == "FLOAT32" else np.float16))
            continue
        p = producer.get(ti)
        if p is not None:
            src = sg.Tensors(p.Inputs(0))
            raw = C.buf_bytes(m, data, src.Buffer())
            dt = C.TT.get(src.Type())
            out[scope] = (name, "DEQUANTIZE<" + dt, np.frombuffer(raw, np.float16 if dt == "FLOAT16" else np.float32))
    return out


def main(argv):
    src, dst = argv[1], argv[2]
    out_json = argv[argv.index("--json") + 1]
    t0 = time.time()
    qt = quantizer.Quantizer(src)
    for opn in (qtyping.TFLOperationName.FULLY_CONNECTED, qtyping.TFLOperationName.CONV_2D,
                qtyping.TFLOperationName.DEPTHWISE_CONV_2D):
        qt.update_quantization_recipe(
            regex=".*", operation_name=opn,
            op_config=qtyping.OpQuantizationConfig(
                weight_tensor_config=qtyping.TensorQuantizationConfig(num_bits=16, dtype=qtyping.TensorDataType.FLOAT),
                compute_precision=qtyping.ComputePrecision.FLOAT, explicit_dequantize=True),
            algorithm_key=quantizer.AlgorithmName.FLOAT_CASTING)
    res = qt.quantize()
    res.export_model(dst, overwrite=True)
    aeq_s = round(time.time() - t0, 1)
    a, b = weights_by_scope(src), weights_by_scope(dst)
    summ = {}
    for scope, (name, dt, wh) in b.items():
        s = summ.setdefault(name, {"ops": 0, "fp16": 0, "rounding_equal": 0, "max_rel_vs_fp32": 0.0, "unmatched": 0,
                                   "above_fp16_max": 0, "to_zero": 0, "to_subnormal": 0})
        s["ops"] += 1
        if scope not in a:
            s["unmatched"] += 1
            continue
        w32 = a[scope][2].astype(np.float32)
        if "FLOAT16" not in dt:
            continue
        s["fp16"] += 1
        if wh.shape != w32.shape:
            s["unmatched"] += 1
            continue
        r16 = w32.astype(np.float16)
        if np.array_equal(wh.view(np.uint16), r16.view(np.uint16)):
            s["rounding_equal"] += 1
        s["above_fp16_max"] += int((np.abs(w32) > 65504.0).sum())
        nz = w32 != 0
        s["to_zero"] += int((nz & (r16 == 0)).sum())
        s["to_subnormal"] += int((np.abs(r16) < np.float16(2.0 ** -14)).sum() - (r16 == 0).sum())
        w16 = wh.astype(np.float32)
        s["max_rel_vs_fp32"] = max(s["max_rel_vs_fp32"],
                                   float(np.abs(w16 - w32).max() / max(np.abs(w32).max(), 1e-30)))
    h = hashlib.sha256()
    with open(dst, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    rep = {"src": src, "dst": dst, "aeq_s": aeq_s, "per_op": summ, "bytes": os.path.getsize(dst),
           "sha256": h.hexdigest(), "census": r2_fb.census(dst), "total_s": round(time.time() - t0, 1)}
    json.dump(rep, open(out_json, "w"), indent=1)
    print(json.dumps({k: rep[k] for k in ("aeq_s", "per_op", "bytes", "total_s")}), flush=True)
    ok = summ.get("FULLY_CONNECTED", {}).get("fp16", 0) > 0 and all(
        v["fp16"] == v["rounding_equal"] and not v["unmatched"] for v in summ.values())
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main(sys.argv))
