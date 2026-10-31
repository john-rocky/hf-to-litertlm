#!/usr/bin/env python3
"""tfl_same_fn.py A.tflite B.tflite [--seed N] [--out OUT.json]

Checks that B is A with nothing but tensor names changed (tfl_unique_names.py): (1) structure, from the flatbuffers —
the same operators in the same order (opcode, inputs, outputs, builtin options type), the same tensors (shape, type,
buffer size) and the same graph inputs / outputs; names may differ; (2) function — both run once on LiteRT's
CompiledModel on the CPU (XNNPACK, fp32) with the same input (uniform [0, 1) from numpy's default_rng(seed), in the
shape of the first input of the first signature); the outputs must be identical (largest |difference| = 0).
Exit 0 and SAME_FN when both hold, 1 otherwise.
"""
import argparse
import json
import mmap
import sys

import numpy as np
from ai_edge_litert import schema_py_generated as s


def header(path):
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    m = s.Model.GetRootAs(mm, 0)
    sg = m.Subgraphs(0)

    def bsize(t):
        b = m.Buffers(t.Buffer())
        return int(b.Size()) if b.Offset() > 1 else (0 if b.DataIsNone() else int(b.DataLength()))

    ops = []
    for i in range(sg.OperatorsLength()):
        op = sg.Operators(i)
        oc = m.OperatorCodes(op.OpcodeIndex())
        ops.append((max(oc.BuiltinCode(), oc.DeprecatedBuiltinCode()), tuple(op.InputsAsNumpy().tolist()),
                    tuple(op.OutputsAsNumpy().tolist()), op.BuiltinOptionsType()))
    tens = []
    names = []
    for i in range(sg.TensorsLength()):
        t = sg.Tensors(i)
        tens.append((tuple(t.ShapeAsNumpy().tolist()) if t.ShapeLength() else (), t.Type(), bsize(t)))
        names.append(t.Name())
    io = (tuple(sg.InputsAsNumpy().tolist()), tuple(sg.OutputsAsNumpy().tolist()))
    return ops, tens, names, io


def run_cpu(path, seed):
    from ai_edge_litert.compiled_model import CompiledModel
    from ai_edge_litert.hardware_accelerator import HardwareAccelerator
    from ai_edge_litert.options import Options
    model = CompiledModel.from_file(path, options=Options(hardware_accelerators=HardwareAccelerator.CPU))
    sig = list(model.get_signature_list().keys())[0]
    (iname, idet), (oname, odet) = (next(iter(model.get_input_tensor_details(sig).items())),
                                    next(iter(model.get_output_tensor_details(sig).items())))
    x = np.random.default_rng(seed).random(tuple(idet["shape"]), dtype=np.float32)
    ib = model.create_input_buffer_by_name(sig, iname)
    ib.write(np.ascontiguousarray(x))
    ob = model.create_output_buffer_by_name(sig, oname)
    model.run_by_name(sig, {iname: ib}, {oname: ob})
    out = np.array(ob.read(int(np.prod(odet["shape"])), np.float32), dtype=np.float32).reshape(odet["shape"])
    del model
    return out, list(idet["shape"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()
    oa, ta, na, ia = header(a.a)
    ob, tb, nb, ib = header(a.b)
    struct = {"ops_equal": oa == ob, "tensors_equal": ta == tb, "io_equal": ia == ib,
              "n_ops": [len(oa), len(ob)], "n_tensors": [len(ta), len(tb)],
              "names_changed": sum(1 for x, y in zip(na, nb) if x != y)}
    print("structure", struct, flush=True)
    ya, shape_in = run_cpu(a.a, a.seed)
    yb, _ = run_cpu(a.b, a.seed)
    d = float(np.abs(ya - yb).max())
    ok = struct["ops_equal"] and struct["tensors_equal"] and struct["io_equal"] and d == 0.0
    rec = {"a": a.a, "b": a.b, "seed": a.seed, "input_shape": shape_in, "structure": struct,
           "cpu_max_abs_diff": d, "output_shape": list(ya.shape), "same_fn": ok}
    print(f"cpu max |d| {d} input {shape_in} output {list(ya.shape)}", flush=True)
    print("SAME_FN" if ok else "NOT_SAME_FN", flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rec, f, indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
