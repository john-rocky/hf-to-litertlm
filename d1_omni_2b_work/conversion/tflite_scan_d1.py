"""Static scan of a .tflite (the schema only; the file is mmapped, so a > 2 GiB file is fine).

scan(path) -> dict: signatures (inputs / outputs: name, shape, dtype), operator count and histogram, custom ops,
tensor dtype histogram, INT64 / rank > 4 tensors, BROADCAST_TO count, every FULLY_CONNECTED's weight dtype
(`fc_weight_dtypes`), every op whose weights are int8 (`int8_weight_ops`), constants by dtype (bytes), and the
largest activation tensors (non-constant, numel x dtype size, top 8) and constants (top 5).

    python3 tflite_scan_d1.py file.tflite      (any venv with ai_edge_litert)
"""
from __future__ import annotations

import collections
import hashlib
import json
import mmap
import sys

SIZE = {"FLOAT32": 4, "FLOAT16": 2, "INT32": 4, "INT64": 8, "INT8": 1, "UINT8": 1, "BOOL": 1, "INT16": 2,
        "BFLOAT16": 2, "FLOAT64": 8}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(1 << 24):
            h.update(b)
    return h.hexdigest()


def scan(path, with_sha=True):
    from ai_edge_litert import schema_py_generated as schema

    f = open(path, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    model = schema.Model.GetRootAs(mm, 0)
    op_names = {v: k for k, v in vars(schema.BuiltinOperator).items() if isinstance(v, int)}
    type_names = {v: k for k, v in vars(schema.TensorType).items() if isinstance(v, int)}
    codes = []
    for i in range(model.OperatorCodesLength()):
        c = model.OperatorCodes(i)
        name = op_names.get(max(c.BuiltinCode(), c.DeprecatedBuiltinCode()), "UNKNOWN")
        if name == "CUSTOM":
            name = "CUSTOM:" + (c.CustomCode().decode() if c.CustomCode() else "")
        codes.append(name)

    def has_data(t):
        b = model.Buffers(t.Buffer())
        return b is not None and (b.DataLength() > 0 or b.Offset() > 1)

    out = {"file": str(path), "subgraphs": model.SubgraphsLength()}
    ops = collections.Counter()
    dtypes = collections.Counter()
    const_bytes = collections.Counter()
    rank_gt4, int64, fc_w, int8_ops = 0, 0, collections.Counter(), collections.Counter()
    acts, consts, deq_out = [], [], set()
    for si in range(model.SubgraphsLength()):
        sg = model.Subgraphs(si)
        for ti in range(sg.TensorsLength()):
            t = sg.Tensors(ti)
            dt = type_names.get(t.Type(), str(t.Type()))
            shape = t.ShapeAsNumpy().tolist() if t.ShapeLength() else []
            dtypes[dt] += 1
            if len(shape) > 4:
                rank_gt4 += 1
            if dt == "INT64":
                int64 += 1
            n = 1
            for d in shape:
                n *= max(int(d), 1)
            nbytes = n * SIZE.get(dt, 4)
            name = t.Name().decode(errors="replace") if t.Name() else ""
            if has_data(t):
                const_bytes[dt] += nbytes
                consts.append((nbytes, shape, dt, name[:80]))
            else:
                acts.append((nbytes, shape, dt, name[:80], (si, ti)))
        for oi in range(sg.OperatorsLength()):
            op = sg.Operators(oi)
            name = codes[op.OpcodeIndex()]
            ops[name] += 1
            if name == "DEQUANTIZE" and op.OutputsLength():
                deq_out.add((si, op.Outputs(0)))
            ins = [op.Inputs(j) for j in range(op.InputsLength())]
            if name == "FULLY_CONNECTED" and len(ins) > 1 and ins[1] >= 0:
                w = sg.Tensors(ins[1])
                wdt = type_names.get(w.Type())
                src = "const" if has_data(w) else "computed"
                if src == "computed":   # a DEQUANTIZE feeding the FC: report the stored dtype
                    for oj in range(sg.OperatorsLength()):
                        o2 = sg.Operators(oj)
                        if o2.OutputsLength() and o2.Outputs(0) == ins[1] and codes[o2.OpcodeIndex()] == "DEQUANTIZE":
                            wdt = type_names.get(sg.Tensors(o2.Inputs(0)).Type()) + "->DEQUANTIZE"
                            break
                fc_w[f"{wdt}"] += 1
            for j in ins:
                if j >= 0:
                    t = sg.Tensors(j)
                    if type_names.get(t.Type()) == "INT8" and has_data(t):
                        int8_ops[name] += 1
                        break
    sigs = []
    for i in range(model.SignatureDefsLength()):
        sd = model.SignatureDefs(i)
        sg = model.Subgraphs(sd.SubgraphIndex())

        def io(n, get):
            r = []
            for j in range(n):
                tm = get(j)
                t = sg.Tensors(tm.TensorIndex())
                r.append({"name": tm.Name().decode(), "shape": t.ShapeAsNumpy().tolist() if t.ShapeLength() else [],
                          "dtype": type_names.get(t.Type())})
            return r
        sigs.append({"key": sd.SignatureKey().decode(), "inputs": io(sd.InputsLength(), sd.Inputs),
                     "outputs": io(sd.OutputsLength(), sd.Outputs)})
    acts.sort(key=lambda x: -x[0])
    consts.sort(key=lambda x: -x[0])
    out.update(signatures=sigs, operator_count=sum(ops.values()), op_histogram=dict(ops.most_common()),
               custom_ops=sorted(n for n in ops if n.startswith("CUSTOM")), tensor_dtype_histogram=dict(dtypes),
               rank_gt4_tensor_count=rank_gt4, int64_tensor_count=int64, broadcast_to_count=ops.get("BROADCAST_TO", 0),
               fc_weight_dtypes=dict(fc_w), int8_weight_ops=dict(int8_ops),
               const_bytes_by_dtype=dict(const_bytes),
               largest_tensors=[{"bytes": b, "shape": s, "dtype": d, "name": n} for b, s, d, n, _ in acts[:8]],
               # the runtime buffers the graph computes, without the DEQUANTIZE outputs (the float16 weights widened
               # to float32: a CPU-side copy; the GPU delegate takes the float16 constants as weights)
               largest_activations=[{"bytes": b, "shape": s, "dtype": d, "name": n} for b, s, d, n, k in acts
                                    if k not in deq_out][:200],
               dequantize_outputs=len(deq_out),
               largest_constants=[{"bytes": b, "shape": s, "dtype": d, "name": n} for b, s, d, n in consts[:5]])
    import os
    out["bytes"] = os.path.getsize(path)
    if with_sha:
        out["sha256"] = sha256_file(path)
    return out


if __name__ == "__main__":
    print(json.dumps(scan(sys.argv[1]), indent=1))
