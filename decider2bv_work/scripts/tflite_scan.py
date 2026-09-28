"""Flatbuffer-only scan of a .tflite (no Interpreter, no allocation): op histogram per signature subgraph and in
total, custom ops, signature inputs/outputs with shape and dtype, max tensor rank, and the forbidden-op counts the
round gates read (GATHER*, FLEX, CUSTOM, STABLEHLO_*, BROADCAST_TO). The pattern of decider_work/scripts/inspect_export.py.

    <any venv with ai-edge-litert> python -B scripts/tflite_scan.py <file.tflite> [...]
"""
import hashlib
import json
import mmap
import sys
from collections import Counter

from ai_edge_litert import schema_py_generated as schema


def enum_names(cls):
    return {v: k for k, v in vars(cls).items() if isinstance(v, int)}


def sha256_file(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def scan(path, with_sha=True):
    dtype_names = enum_names(schema.TensorType)
    builtin_names = enum_names(schema.BuiltinOperator)
    with open(path, 'rb') as stream:
        buf = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
        model = schema.Model.GetRootAsModel(buf, 0)
        codes = []
        for i in range(model.OperatorCodesLength()):
            op = model.OperatorCodes(i)
            name = builtin_names[max(op.BuiltinCode(), op.DeprecatedBuiltinCode())]
            custom = op.CustomCode().decode() if op.CustomCode() else None
            codes.append((name, custom))
        total, custom_hist, max_rank = Counter(), Counter(), 0
        per_subgraph = []
        for i in range(model.SubgraphsLength()):
            sg = model.Subgraphs(i)
            hist = Counter()
            for j in range(sg.OperatorsLength()):
                name, custom = codes[sg.Operators(j).OpcodeIndex()]
                hist[name] += 1
                if custom:
                    custom_hist[custom] += 1
            for j in range(sg.TensorsLength()):
                max_rank = max(max_rank, sg.Tensors(j).ShapeLength())
            total.update(hist)
            per_subgraph.append(dict(index=i, name=sg.Name().decode() if sg.Name() else '', operator_count=sg.OperatorsLength(),
                                     op_histogram=dict(sorted(hist.items()))))
        sigs = {}
        for i in range(model.SignatureDefsLength()):
            sig = model.SignatureDefs(i)
            key = sig.SignatureKey().decode()
            sg = model.Subgraphs(sig.SubgraphIndex())
            entry = dict(subgraph_index=sig.SubgraphIndex(), inputs={}, outputs={})
            for kind, count, get in (('inputs', sig.InputsLength(), sig.Inputs), ('outputs', sig.OutputsLength(), sig.Outputs)):
                for j in range(count):
                    tm = get(j)
                    tensor = sg.Tensors(tm.TensorIndex())
                    entry[kind][tm.Name().decode()] = dict(shape=[tensor.Shape(k) for k in range(tensor.ShapeLength())],
                                                           dtype=dtype_names[tensor.Type()])
            sigs[key] = entry
        buf.close()
    forbidden = {name: n for name, n in total.items()
                 if 'GATHER' in name or name.startswith('STABLEHLO') or name in ('CUSTOM', 'BROADCAST_TO')}
    forbidden['FLEX'] = sum(n for name, n in custom_hist.items() if name.lower().startswith('flex'))
    out = dict(path=str(path), bytes=None, parser='ai_edge_litert.schema_py_generated only; no Interpreter',
               op_histogram=dict(sorted(total.items())), custom_histogram=dict(custom_hist), operator_count=sum(total.values()),
               max_tensor_rank=max_rank, forbidden=forbidden, forbidden_total=sum(forbidden.values()),
               signatures=sigs, subgraphs=per_subgraph)
    import os
    out['bytes'] = os.path.getsize(path)
    if with_sha:
        out['sha256'] = sha256_file(path)
    return out


if __name__ == '__main__':
    for p in sys.argv[1:]:
        r = scan(p)
        print(json.dumps({k: r[k] for k in ('path', 'bytes', 'operator_count', 'max_tensor_rank', 'forbidden', 'op_histogram')}, indent=1))
        print('signatures:', {k: (list(v['inputs'])[:6], len(v['inputs']), list(v['outputs'])[:4], len(v['outputs'])) for k, v in r['signatures'].items()})
