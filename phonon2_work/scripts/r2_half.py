#!/usr/bin/env python3
"""enc30_fp16w (venv .venv-q, no model execution): ai-edge-quantizer 0.10.0 float_casting (fp16 weight-only,
explicit DEQUANTIZE, FLOAT compute) on every FULLY_CONNECTED, CONV_2D and DEPTHWISE_CONV_2D weight of the fp32
encoder file. The folded relative-position tables (BATCH_MATMUL constants) are not taken by float_casting and stay fp32.
Check: every fp16 weight, cast back to fp32, against the same op's weight in the fp32 file (matched by the op's output
names): exact count and max relative difference per op type (the ternary FC weights are code x fp16 scale and the
dense redux weights are F16, so they are expected exact; weights the converter folded a BatchNorm into are not).
CLI: r2_half.py <fp32 src> <dst> --json results/r2_fp16w.json
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
        s = summ.setdefault(name, {"ops": 0, "fp16": 0, "exact": 0, "max_rel": 0.0, "unmatched": 0})
        s["ops"] += 1
        if scope not in a:
            s["unmatched"] += 1
            continue
        w32 = a[scope][2].astype(np.float32)
        if "FLOAT16" not in dt:
            continue
        s["fp16"] += 1
        w16 = wh.astype(np.float32)
        if w16.shape == w32.shape and np.array_equal(w16, w32):
            s["exact"] += 1
        elif w16.shape == w32.shape:
            s["max_rel"] = max(s["max_rel"], float(np.abs(w16 - w32).max() / max(np.abs(w32).max(), 1e-30)))
    h = hashlib.sha256()
    with open(dst, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    rep = {"src": src, "dst": dst, "aeq_s": aeq_s, "per_op": summ, "bytes": os.path.getsize(dst),
           "sha256": h.hexdigest(), "census": r2_fb.census(dst), "total_s": round(time.time() - t0, 1)}
    json.dump(rep, open(out_json, "w"), indent=1)
    print(json.dumps({k: rep[k] for k in ("aeq_s", "per_op", "bytes", "total_s")}), flush=True)
    return 0 if summ.get("FULLY_CONNECTED", {}).get("fp16", 0) > 0 else 3


if __name__ == "__main__":
    sys.exit(main(sys.argv))
