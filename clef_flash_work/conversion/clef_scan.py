"""The export's acceptance scan: scripts/tflite_scan.py (kev_work's static flatbuffer reader, copied unchanged) plus
the lightweight path's three checks (memory export-fp32-ram-ceiling), all read from the schema without running the
model (the file is mmapped):

  all_constant_input_ops   ops whose every input is a constant tensor (not produced by an op, not a graph input):
                           0 required (a fold the lightweight path left for the runtime and nobody did)
  fully_connected          every FULLY_CONNECTED weight (input 1) is a constant with data in the file; its dtypes
  constant_bytes           bytes of the distinct constant buffers, by dtype (the size the file should be)

    python clef_scan.py <file.tflite> [--json out.json]"""
import collections
import json
import mmap
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tflite_scan import scan  # noqa: E402


def lightweight_checks(path) -> dict:
    from ai_edge_litert import schema_py_generated as schema
    f = open(path, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    model = schema.Model.GetRootAs(mm, 0)
    op_names = {v: k for k, v in vars(schema.BuiltinOperator).items() if isinstance(v, int)}
    type_names = {v: k for k, v in vars(schema.TensorType).items() if isinstance(v, int)}
    codes = []
    for i in range(model.OperatorCodesLength()):
        c = model.OperatorCodes(i)
        codes.append(op_names.get(max(c.BuiltinCode(), c.DeprecatedBuiltinCode()), "UNKNOWN"))

    def buf_size(bi):
        b = model.Buffers(bi)
        if b.DataLength() > 0:
            return b.DataLength()
        if b.Offset() > 1:
            return b.Size()
        return 0

    all_const, fc_rows, const_buffers = [], collections.Counter(), {}
    fc_weight_dtypes, fc_nonconst, fc_empty = collections.Counter(), [], []
    for gi in range(model.SubgraphsLength()):
        g = model.Subgraphs(gi)
        tensors = [g.Tensors(t) for t in range(g.TensorsLength())]
        produced = set()
        ops = [g.Operators(o) for o in range(g.OperatorsLength())]
        for op in ops:
            if op.OutputsLength():
                produced.update(int(x) for x in op.OutputsAsNumpy())
        graph_inputs = {int(x) for x in g.InputsAsNumpy()} if g.InputsLength() else set()
        is_const = lambda t: t >= 0 and t not in produced and t not in graph_inputs
        for ti, t in enumerate(tensors):
            if is_const(ti) and buf_size(t.Buffer()) > 0:
                const_buffers[t.Buffer()] = (type_names.get(t.Type(), str(t.Type())), buf_size(t.Buffer()))
        for oi, op in enumerate(ops):
            name = codes[op.OpcodeIndex()]
            ins = [int(x) for x in op.InputsAsNumpy()] if op.InputsLength() else []
            real = [t for t in ins if t >= 0]
            if real and all(is_const(t) for t in real):
                all_const.append({"subgraph": gi, "op_index": oi, "op": name,
                                  "inputs": [{"shape": tensors[t].ShapeAsNumpy().tolist() if tensors[t].ShapeLength() else [],
                                              "dtype": type_names.get(tensors[t].Type())} for t in real],
                                  "output": tensors[int(op.OutputsAsNumpy()[0])].Name().decode() if op.OutputsLength() else None})
            if name == "FULLY_CONNECTED":
                w = ins[1]
                fc_rows["total"] += 1
                if not is_const(w):
                    fc_nonconst.append({"op_index": oi, "weight": tensors[w].Name().decode()})
                elif buf_size(tensors[w].Buffer()) == 0:
                    fc_empty.append({"op_index": oi, "weight": tensors[w].Name().decode()})
                fc_weight_dtypes[type_names.get(tensors[w].Type())] += 1
    by_dtype = collections.Counter()
    for dt, n in const_buffers.values():
        by_dtype[dt] += n
    mm.close()
    f.close()
    return {"all_constant_input_ops": len(all_const), "all_constant_input_op_sites": all_const[:50],
            "all_constant_input_op_names": dict(collections.Counter(s["op"] for s in all_const)),
            "fully_connected": fc_rows["total"], "fc_weight_dtypes": dict(fc_weight_dtypes),
            "fc_weight_not_constant": len(fc_nonconst), "fc_weight_not_constant_sites": fc_nonconst[:20],
            "fc_weight_constant_without_data": len(fc_empty), "fc_weight_constant_without_data_sites": fc_empty[:20],
            "constant_buffers": len(const_buffers), "constant_bytes": int(sum(by_dtype.values())),
            "constant_bytes_by_dtype": dict(by_dtype)}


def full_scan(path) -> dict:
    s = scan(path)
    s["lightweight"] = lightweight_checks(path)
    return s


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("tflite")
    ap.add_argument("--json")
    a = ap.parse_args()
    d = full_scan(a.tflite)
    if a.json:
        out = Path(a.json)
        assert not out.exists(), f"refusing to overwrite {out}"
        out.write_text(json.dumps(d, indent=1) + "\n")
    print(json.dumps({k: d[k] for k in ("file", "bytes", "operator_count", "forbidden_counts", "int64_tensor_count",
                                        "rank_gt4_tensor_count", "custom_op_count", "fully_connected_count")}, indent=1))
    print(json.dumps({k: v for k, v in d["lightweight"].items() if not k.endswith("_sites")}, indent=1))
