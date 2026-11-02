"""Helpers of the d1-Omni-2B batch and multi8 exports, copied from export_multi8.py of the d1-2B conversion
(sample_rows = the batch export's sample, seed 0; op_sequences = per signature the ops with their input / output
shapes and a sha256 of that list). No sys.path change here; standard library + numpy + ai_edge_litert's schema only.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import mmap
import sys


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(1 << 24):
            h.update(b)
    return h.hexdigest()


def load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def sample_rows(B, L, hidden, seed=0):
    """export_batch.sample_rows of the d1-2B conversion: B rows of different lengths (one full-length row, the rest
    43..L), state halves, 1-4 questions; embeds = small random values (the batch-1 export sample's scale)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    rows = []
    for j in range(B):
        T = L if j == 0 else int(rng.integers(43, L + 1))
        st = int(rng.integers(5, T // 2))
        nq = int(rng.integers(1, 5))
        tags = [-1] * st + [1 + (i * nq) // (T - st) for i in range(T - st)]
        rows.append((T, st, tags, (rng.standard_normal((T, hidden)) * 0.02).astype(np.float32)))
    return rows


def op_sequences(path):
    """Per signature: the subgraph's ops as (op name, input shapes, output shapes), and a sha256 of that list."""
    from ai_edge_litert import schema_py_generated as schema

    f = open(path, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    m = schema.Model.GetRootAs(mm, 0)
    bo = {v: k for k, v in vars(schema.BuiltinOperator).items() if isinstance(v, int) and not k.startswith("_")}
    out = {}
    for i in range(m.SignatureDefsLength()):
        sd = m.SignatureDefs(i)
        sg = m.Subgraphs(sd.SubgraphIndex())

        def shp(t):
            if t < 0:
                return None
            x = sg.Tensors(t)
            return [int(x.Shape(k)) for k in range(x.ShapeLength())]
        seq = []
        for o in range(sg.OperatorsLength()):
            op = sg.Operators(o)
            oc = m.OperatorCodes(op.OpcodeIndex())
            name = bo.get(max(oc.BuiltinCode(), oc.DeprecatedBuiltinCode()), "?")
            seq.append([name, [shp(int(t)) for t in op.InputsAsNumpy()], [shp(int(t)) for t in op.OutputsAsNumpy()]])
        blob = json.dumps(seq, separators=(",", ":")).encode()
        out[sd.SignatureKey().decode()] = {"ops": len(seq), "seq_sha256": hashlib.sha256(blob).hexdigest()}
    return out
