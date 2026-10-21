#!/usr/bin/env python3
"""r2 flatbuffer tools (venv .venv-q: ai-edge-litert 2.3.0 schema + flatbuffer_utils). No model execution.

census(path)             op histogram, FLEX / CUSTOM / rank > 4 / INT64 tensors, SELECT_V2 count, the integer-dtype
                         op outputs, weight dtypes of FULLY_CONNECTED / CONV / DEPTHWISE / BATCH_MATMUL constants,
                         signatures with input / output names, dtypes and shapes.
lift_bmm(src, dst)       every BATCH_MATMUL whose two inputs are runtime rank-3 tensors gets RESHAPEs to rank 4 in
                         front ([B,M,K] -> [1,B,M,K]) and a RESHAPE back to rank 3 after (the LiteRT #10445 class:
                         the converter folds a batch-1 rank-4 matmul back to rank 3, so the lift is done here).
                         New tensors and ops are appended; the rest of the graph is untouched. The file is written
                         with external buffers (Buffer.offset / size, 16-byte aligned), the layout ai-edge-quantizer
                         writes.
CLI: r2_fb.py census <file> [--json out]   |   r2_fb.py lift <src> <dst> [--json out]
"""
import json
import os
import sys

import numpy as np

from ai_edge_litert import schema_py_generated as S
from ai_edge_litert.tools import flatbuffer_utils as FU
from ai_edge_litert.tools import mmap_utils

TT = {v: k for k, v in S.TensorType.__dict__.items() if not k.startswith("_")}
BO = {v: k for k, v in S.BuiltinOperator.__dict__.items() if not k.startswith("_")}


def _code(oc):
    return max(oc.BuiltinCode(), oc.DeprecatedBuiltinCode())


def _buf_len(m, idx):
    b = m.Buffers(idx)
    return b.DataLength() or b.Size()


def census(path):
    data = mmap_utils.get_file_contents(path)
    m = S.Model.GetRootAs(data, 0)
    ops, custom, flex = {}, {}, 0
    rank_gt4, int64, w_dtypes, int_ops = [], [], {}, {}
    for s in range(m.SubgraphsLength()):
        sg = m.Subgraphs(s)
        for i in range(sg.TensorsLength()):
            t = sg.Tensors(i)
            if t.ShapeLength() > 4:
                rank_gt4.append((t.Name().decode(), t.ShapeAsNumpy().tolist()))
            if TT.get(t.Type()) == "INT64":
                int64.append(t.Name().decode())
        for i in range(sg.OperatorsLength()):
            op = sg.Operators(i)
            oc = m.OperatorCodes(op.OpcodeIndex())
            name = BO.get(_code(oc), str(_code(oc)))
            if name == "CUSTOM":
                cc = oc.CustomCode().decode() if oc.CustomCode() else "?"
                if cc.startswith("Flex"):
                    flex += 1
                custom[cc] = custom.get(cc, 0) + 1
            ops[name] = ops.get(name, 0) + 1
            if op.OutputsLength():
                ot = sg.Tensors(op.Outputs(0))
                if TT.get(ot.Type()) not in ("FLOAT32", "FLOAT16"):
                    k = f"{name}->{TT.get(ot.Type())}"
                    int_ops[k] = int_ops.get(k, 0) + 1
            if name in ("FULLY_CONNECTED", "CONV_2D", "DEPTHWISE_CONV_2D", "BATCH_MATMUL", "EMBEDDING_LOOKUP",
                        "GATHER"):
                for j in range(op.InputsLength()):
                    ti = op.Inputs(j)
                    if ti < 0:
                        continue
                    t = sg.Tensors(ti)
                    if _buf_len(m, t.Buffer()) > 0:
                        k = f"{name}:{TT.get(t.Type())}"
                        w_dtypes[k] = w_dtypes.get(k, 0) + 1
                        break
    sigs = []
    for k in range(m.SignatureDefsLength()):
        sd = m.SignatureDefs(k)
        sg = m.Subgraphs(sd.SubgraphIndex())

        def io(n, get):
            res = []
            for j in range(n):
                tm = get(j)
                t = sg.Tensors(tm.TensorIndex())
                res.append({"name": tm.Name().decode(), "dtype": TT.get(t.Type()), "shape": t.ShapeAsNumpy().tolist()})
            return res
        sigs.append({"key": sd.SignatureKey().decode(), "inputs": io(sd.InputsLength(), sd.Inputs),
                     "outputs": io(sd.OutputsLength(), sd.Outputs)})
    return {"file": path, "bytes": os.path.getsize(path), "n_ops": sum(ops.values()),
            "ops": dict(sorted(ops.items(), key=lambda x: -x[1])), "flex": flex, "custom": custom,
            "rank_gt4_tensors": len(rank_gt4), "rank_gt4_examples": rank_gt4[:5], "int64_tensors": len(int64),
            "select_v2": ops.get("SELECT_V2", 0) + ops.get("SELECT", 0), "non_float_op_outputs": int_ops,
            "const_weight_dtypes": w_dtypes, "signatures": sigs}


def _round16(x):
    return (x + 15) & ~15


def write_external(model, dst):
    """Serialize a ModelT with every non-empty buffer stored after the flatbuffer (offset / size, 16-aligned)."""
    datas = {}
    for i, buf in enumerate(model.buffers):
        d = buf.data
        if d is not None and len(d) > 0:
            datas[i] = d
            buf.data = None
            buf.offset = 1
            buf.size = 1
    head = FU.convert_object_to_bytearray(model)
    off = _round16(len(head))
    packed = S.Model.GetRootAs(head, 0)
    plan = []
    for i, d in datas.items():
        n = len(d) if not isinstance(d, np.ndarray) else d.nbytes
        FU.update_packed_buffer(packed.Buffers(i), offset=off, size=n)
        plan.append((off, d))
        off = _round16(off + n)
    with open(dst, "wb") as f:
        f.write(head)
        pos = len(head)
        for o, d in plan:
            if o > pos:
                f.write(b"\0" * (o - pos))
                pos = o
            mv = memoryview(d).cast("B") if not isinstance(d, np.ndarray) else memoryview(np.ascontiguousarray(d)).cast("B")
            f.write(mv)
            pos += len(mv)
        if off > pos:
            f.write(b"\0" * (off - pos))
    return os.path.getsize(dst)


def _opcode_index(model, code):
    for i, oc in enumerate(model.operatorCodes):
        if max(oc.builtinCode, oc.deprecatedBuiltinCode) == code:
            return i
    oc = S.OperatorCodeT()
    oc.builtinCode = code
    oc.deprecatedBuiltinCode = min(code, 127)
    oc.version = 1
    model.operatorCodes.append(oc)
    return len(model.operatorCodes) - 1


def lift_bmm(src, dst):
    data = mmap_utils.get_file_contents(src)
    model = FU.read_model_from_bytearray(data)
    bmm = S.BuiltinOperator.BATCH_MATMUL
    rs_idx = _opcode_index(model, S.BuiltinOperator.RESHAPE)
    rep = {"src": src, "dst": dst, "lifted": 0, "shapes": {}}

    def is_runtime(sg, ti):
        b = model.buffers[sg.tensors[ti].buffer]
        return b.data is None or len(b.data) == 0

    for sg in model.subgraphs:
        def add_tensor(name, shape, ttype, data=None):
            t = S.TensorT()
            t.name = name.encode() if isinstance(name, str) else name
            t.shape = list(shape)
            t.type = ttype
            if data is None:
                t.buffer = 0
            else:
                b = S.BufferT()
                b.data = np.frombuffer(np.asarray(data).tobytes(), np.uint8).copy()
                model.buffers.append(b)
                t.buffer = len(model.buffers) - 1
            sg.tensors.append(t)
            return len(sg.tensors) - 1

        def reshape_op(x, y, shape, tag):
            st = add_tensor(f"{tag}/shape", [len(shape)], S.TensorType.INT32, np.array(shape, np.int32))
            op = S.OperatorT()
            op.opcodeIndex = rs_idx
            op.inputs = [x, st]
            op.outputs = [y]
            op.builtinOptionsType = S.BuiltinOptions.ReshapeOptions
            opt = S.ReshapeOptionsT()
            opt.newShape = list(shape)
            op.builtinOptions = opt
            return op

        new_ops = []
        for op in sg.operators:
            code = model.operatorCodes[op.opcodeIndex]
            if max(code.builtinCode, code.deprecatedBuiltinCode) != bmm or len(op.inputs) < 2:
                new_ops.append(op)
                continue
            a, b = op.inputs[0], op.inputs[1]
            o = op.outputs[0]
            ta, tb, to = sg.tensors[a], sg.tensors[b], sg.tensors[o]
            if not (is_runtime(sg, a) and is_runtime(sg, b) and len(ta.shape) == 3 and len(tb.shape) == 3
                    and len(to.shape) == 3):
                new_ops.append(op)
                continue
            tag = (to.name.decode(errors="replace") if isinstance(to.name, bytes) else str(to.name)) + "/r4"
            sa, sb, so = [1] + list(ta.shape), [1] + list(tb.shape), [1] + list(to.shape)
            a4 = add_tensor(tag + "/a", sa, ta.type)
            b4 = add_tensor(tag + "/b", sb, tb.type)
            o4 = add_tensor(tag + "/o", so, to.type)
            new_ops.append(reshape_op(a, a4, sa, tag + "/ra"))
            new_ops.append(reshape_op(b, b4, sb, tag + "/rb"))
            op.inputs = [a4, b4] + list(op.inputs[2:])
            op.outputs = [o4]
            new_ops.append(op)
            new_ops.append(reshape_op(o4, o, list(to.shape), tag + "/ro"))
            rep["lifted"] += 1
            k = f"{list(ta.shape)}x{list(tb.shape)}"
            rep["shapes"][k] = rep["shapes"].get(k, 0) + 1
        sg.operators = new_ops
    rep["bytes"] = write_external(model, dst)
    return rep


def main(argv):
    cmd = argv[1]
    out_json = argv[argv.index("--json") + 1] if "--json" in argv else None
    if cmd == "census":
        r = census(argv[2])
    elif cmd == "lift":
        r = lift_bmm(argv[2], argv[3])
    else:
        raise SystemExit(cmd)
    s = json.dumps(r, indent=1)
    if out_json:
        open(out_json, "w").write(s)
    print(json.dumps({k: v for k, v in r.items() if k not in ("signatures",)})[:2000])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
