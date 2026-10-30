#!/usr/bin/env python3
"""tfl_census.py MODEL.tflite [--json OUT]

Op / shape census of one .tflite (header only; works on the external-buffer layout of > 2 GB files): per op type
the distinct (input shapes, constness) -> output shape forms with counts, and every op whose inputs mix a runtime
tensor with a non-scalar constant. Used to list the GPU-placement-sensitive sites of the vision graphs (ML Drift
places a runtime rank-2/3 tensor batch-first and a constant right-aligned).
"""
import argparse
import collections
import json
import mmap
import sys

import numpy as np
from ai_edge_litert import schema_py_generated as s

OPNAMES = {v: k for k, v in s.BuiltinOperator.__dict__.items() if not k.startswith("_")}
SHAPE_OPS = ("FULLY_CONNECTED", "CONV_2D", "RESHAPE", "TRANSPOSE", "STRIDED_SLICE", "SLICE", "MEAN", "SUM", "PAD",
             "GATHER", "CONCATENATION", "DEPTHWISE_CONV_2D")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--json")
    a = ap.parse_args()
    f = open(a.model, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    m = s.Model.GetRootAs(mm, 0)
    sg = m.Subgraphs(0)

    def is_const(t):
        b = m.Buffers(t.Buffer())
        return b.Offset() > 1 or (not b.DataIsNone() and b.DataLength() > 0)

    def const_i32(ti):
        b = m.Buffers(sg.Tensors(ti).Buffer())
        if b.Offset() > 1:
            return np.frombuffer(mm[b.Offset():b.Offset() + b.Size()], dtype=np.int32).tolist()
        return b.DataAsNumpy().view("int32").tolist()

    def desc(ti):
        if ti < 0:
            return "-"
        t = sg.Tensors(ti)
        shp = list(t.ShapeAsNumpy()) if t.ShapeLength() else []
        return ("C" if is_const(t) else "R") + str([int(x) for x in shp])

    forms = collections.Counter()
    mixed = collections.Counter()
    for i in range(sg.OperatorsLength()):
        op = sg.Operators(i)
        oc = m.OperatorCodes(op.OpcodeIndex())
        code = max(oc.BuiltinCode(), oc.DeprecatedBuiltinCode())
        name = OPNAMES.get(code, str(code))
        if name == "CUSTOM":
            name = "CUSTOM:" + oc.CustomCode().decode()
        ins = [desc(op.Inputs(j)) for j in range(op.InputsLength())]
        outs = [desc(op.Outputs(j)) for j in range(op.OutputsLength())]
        extra = ""
        if name == "TRANSPOSE":
            extra = " perm" + str(const_i32(op.Inputs(1)))
        forms[(name, " ".join(ins) + " -> " + " ".join(outs) + extra)] += 1
        rt = [d for d in ins if d.startswith("R")]
        cs = [d for d in ins if d.startswith("C") and d not in ("C[]", "C[1]")]
        if rt and cs and name not in SHAPE_OPS:
            mixed[(name, " ".join(ins))] += 1
    by_op = collections.defaultdict(list)
    for (name, form), n in sorted(forms.items()):
        by_op[name].append((n, form))
    print(f"{a.model}: ops {sg.OperatorsLength()} tensors {sg.TensorsLength()}")
    for name in sorted(by_op, key=lambda k: -sum(n for n, _ in by_op[k])):
        tot = sum(n for n, _ in by_op[name])
        print(f"== {name} x{tot} ({len(by_op[name])} forms)")
        for n, form in sorted(by_op[name], key=lambda x: -x[0])[:12]:
            print(f"   {n:5d}  {form}")
    print("== runtime + non-scalar constant (elementwise-like):")
    for (name, ins), n in sorted(mixed.items(), key=lambda x: -x[1]):
        print(f"   {n:5d}  {name}  {ins}")
    if a.json:
        json.dump({"model": a.model, "forms": [[k[0], k[1], v] for k, v in forms.items()],
                   "mixed": [[k[0], k[1], v] for k, v in mixed.items()]}, open(a.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
