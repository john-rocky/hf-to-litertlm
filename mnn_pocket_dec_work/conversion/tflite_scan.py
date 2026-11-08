"""Static flatbuffer scan of an exported .tflite (reads the schema only, never runs the model).

Lazy reader over ai_edge_litert's generated schema (no object-API unpack, so a > 2 GiB file is fine: the file is
mmapped; buffers stored past the flatbuffer are read by offset/size). Reports: signatures, op histogram and total,
tensor rank / dtype histograms, INT64 tensors, rank > 4 tensors, the GPU-rejection op list (GATHER_ND CAST SELECT_V2
BROADCAST_TO MAXIMUM; every site with its output tensor name and input shapes), every PAD with its rank and padded
axes, every BATCH_MATMUL's input shapes (grouped), FULLY_CONNECTED / EMBEDDING_LOOKUP / GATHER counts, CUSTOM op names,
STABLEHLO op names.

    python tflite_scan.py <file.tflite> [--json out.json]"""
import argparse
import collections
import hashlib
import json
import mmap
from pathlib import Path

import numpy as np

FORBIDDEN_OPS = ("GATHER_ND", "CAST", "SELECT_V2", "BROADCAST_TO", "MAXIMUM")
NP_OF = {"FLOAT32": np.float32, "INT32": np.int32, "INT64": np.int64, "FLOAT16": np.float16, "INT8": np.int8,
         "UINT8": np.uint8, "BOOL": np.bool_, "INT16": np.int16}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 24):
            h.update(block)
    return h.hexdigest()


def scan(path, with_sha=True):
    from ai_edge_litert import schema_py_generated as schema
    path = Path(path)
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

    def buffer_array(tensor):
        b = model.Buffers(tensor.Buffer())
        dtype = NP_OF.get(type_names.get(tensor.Type()))
        if dtype is None:
            return None
        if b.DataLength() > 0:
            raw = b.DataAsNumpy().tobytes()
        elif b.Offset() > 1:
            raw = mm[b.Offset(): b.Offset() + b.Size()]
        else:
            return None
        shape = tensor.ShapeAsNumpy().tolist() if tensor.ShapeLength() else []
        return np.frombuffer(raw, dtype=dtype).reshape(shape)

    hist, ranks, dtypes = collections.Counter(), collections.Counter(), collections.Counter()
    forbidden = collections.defaultdict(list)
    pads, bmm_groups, customs, stablehlo = [], collections.Counter(), collections.Counter(), collections.Counter()
    int64_names, rank_gt4_names = [], []
    bmm_const_left = 0
    for gi in range(model.SubgraphsLength()):
        g = model.Subgraphs(gi)
        tensors = [g.Tensors(t) for t in range(g.TensorsLength())]
        shape = lambda t: tensors[t].ShapeAsNumpy().tolist() if tensors[t].ShapeLength() else []
        tname = lambda t: tensors[t].Name().decode() if t >= 0 else None
        ttype = lambda t: type_names.get(tensors[t].Type(), str(tensors[t].Type()))
        produced, graph_inputs = set(), {int(x) for x in g.InputsAsNumpy()} if g.InputsLength() else set()
        ops = [g.Operators(o) for o in range(g.OperatorsLength())]
        for op in ops:
            if op.OutputsLength():
                produced.update(int(x) for x in op.OutputsAsNumpy())
        is_const = lambda t: t >= 0 and t not in produced and t not in graph_inputs
        for ti, t in enumerate(tensors):
            s = shape(ti)
            ranks[len(s)] += 1
            dtypes[ttype(ti)] += 1
            if t.Type() == schema.TensorType.INT64:
                int64_names.append({"tensor": tname(ti), "shape": s, "constant": is_const(ti)})
            if len(s) > 4:
                rank_gt4_names.append({"tensor": tname(ti), "shape": s})
        for oi, op in enumerate(ops):
            name = codes[op.OpcodeIndex()]
            hist[name] += 1
            ins = [int(x) for x in op.InputsAsNumpy()] if op.InputsLength() else []
            outs = [int(x) for x in op.OutputsAsNumpy()] if op.OutputsLength() else []
            site = lambda: {"subgraph": gi, "op_index": oi, "outputs": [tname(t) for t in outs],
                            "inputs": [{"shape": shape(t), "dtype": ttype(t), "constant": is_const(t)} for t in ins if t >= 0]}
            if name in FORBIDDEN_OPS:
                forbidden[name].append(site())
            if name.startswith("CUSTOM"):
                customs[name] += 1
            if name.startswith("STABLEHLO"):
                stablehlo[name] += 1
            if name in ("PAD", "PADV2", "MIRROR_PAD"):
                x, p = ins[0], ins[1]
                arr = buffer_array(tensors[p]) if is_const(p) else None
                axes = None if arr is None else [int(a) for a in range(arr.shape[0]) if (arr[a] != 0).any()]
                pads.append({"op": name, "op_index": oi, "input_shape": shape(x), "rank": len(shape(x)),
                             "paddings": None if arr is None else arr.tolist(), "padded_axes": axes,
                             "last_axis_only": None if axes is None else axes == [len(shape(x)) - 1],
                             "output": tname(outs[0])})
            if name == "BATCH_MATMUL":
                key = json.dumps([shape(t) for t in ins] + [["const" if is_const(t) else "act" for t in ins]])
                bmm_groups[key] += 1
                if is_const(ins[0]):
                    bmm_const_left += 1
    sigs = []
    for si in range(model.SignatureDefsLength()):
        sd = model.SignatureDefs(si)
        g = model.Subgraphs(sd.SubgraphIndex())
        entry = {"key": sd.SignatureKey().decode(), "subgraph": sd.SubgraphIndex()}
        for side, n, get in (("inputs", sd.InputsLength(), sd.Inputs), ("outputs", sd.OutputsLength(), sd.Outputs)):
            entry[side] = []
            for k in range(n):
                tm = get(k)
                t = g.Tensors(tm.TensorIndex())
                entry[side].append({"name": tm.Name().decode(), "shape": t.ShapeAsNumpy().tolist() if t.ShapeLength() else [],
                                    "dtype": type_names.get(t.Type(), str(t.Type()))})
        sigs.append(entry)
    doc = {
        "file": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path) if with_sha else None,
        "subgraphs": model.SubgraphsLength(), "signatures": sigs, "operator_count": sum(hist.values()),
        "op_histogram": dict(sorted(hist.items())),
        "tensor_rank_histogram": {str(k): v for k, v in sorted(ranks.items())},
        "tensor_dtype_histogram": dict(sorted(dtypes.items())),
        "int64_tensor_count": len(int64_names), "int64_tensors": int64_names[:50],
        "rank_gt4_tensor_count": len(rank_gt4_names), "rank_gt4_tensors": rank_gt4_names[:50],
        "forbidden_ops_checked": list(FORBIDDEN_OPS),
        "forbidden_counts": {k: len(forbidden.get(k, [])) for k in FORBIDDEN_OPS},
        "forbidden_total": sum(len(v) for v in forbidden.values()),
        "forbidden_sites": {k: v[:200] for k, v in forbidden.items()},
        "pad_count": len(pads), "pads": pads,
        "pad_summary": dict(collections.Counter(f"rank{p['rank']} axes{p['padded_axes']}" for p in pads)),
        "batch_matmul_count": hist.get("BATCH_MATMUL", 0),
        "batch_matmul_shape_groups": [{"inputs": json.loads(k)[:-1], "kinds": json.loads(k)[-1], "count": v}
                                      for k, v in sorted(bmm_groups.items(), key=lambda kv: -kv[1])],
        "batch_matmul_all_rank4": all(all(len(s) == 4 for s in json.loads(k)[:-1]) for k in bmm_groups),
        "batch_matmul_constant_left": bmm_const_left,
        "fully_connected_count": hist.get("FULLY_CONNECTED", 0),
        "embedding_lookup_count": hist.get("EMBEDDING_LOOKUP", 0), "gather_count": hist.get("GATHER", 0),
        "custom_ops": dict(customs), "custom_op_count": sum(customs.values()),
        "stablehlo_ops": dict(stablehlo),
    }
    mm.close()
    f.close()
    return doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tflite")
    ap.add_argument("--json")
    a = ap.parse_args()
    doc = scan(a.tflite)
    if a.json:
        out = Path(a.json)
        assert not out.exists(), f"refusing to overwrite {out}"
        out.write_text(json.dumps(doc, indent=1) + "\n")
    brief = {k: doc[k] for k in ("file", "bytes", "sha256", "operator_count", "signatures", "forbidden_counts", "pad_count",
                                 "pad_summary", "int64_tensor_count", "rank_gt4_tensor_count", "custom_ops",
                                 "batch_matmul_count", "batch_matmul_all_rank4", "fully_connected_count",
                                 "embedding_lookup_count", "gather_count", "stablehlo_ops")}
    print(json.dumps(brief, indent=1))


if __name__ == "__main__":
    main()
