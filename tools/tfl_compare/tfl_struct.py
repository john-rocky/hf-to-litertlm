#!/usr/bin/env python3
"""tfl_struct.py A.litertlm B.litertlm SECTION OUT.json [N_SUBGRAPHS], after tfl_swapcheck.py: are two TFLiteModel
sections the same computation in a different order? Per subgraph (default the first 7 = the signatures) every tensor gets a
structural signature that ignores tensor / buffer / operator indices and names:
  subgraph input  -> ("in", its position in the subgraph's input list)
  constant        -> ("const", type, shape, quantization signature, sha1 of the buffer bytes)
  operator output -> ("out", the operator's signature, output position, type, shape, quantization signature)
  operator        -> (builtin code, custom code, the builtin options / builtin options 2 / custom options as values, the
                     signatures of its inputs; -1 = "none")
  blockwise / affine quantization -> block size + the scale / zero-point tensors' signatures, or the scale / zero-point values
Equal = the multiset of operator signatures and the ordered output signatures are equal on every compared subgraph (the
decomposition subgraphs the composites call are compared by index: tfl_swapcheck.py shows them identical). Read-only, mmap; needs ai-edge-litert."""
import collections
import hashlib
import json
import mmap
import sys

import numpy as np
from ai_edge_litert import schema_py_generated as schema

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from tfl_diff import section  # noqa: E402


def H(x):
    return hashlib.sha1(repr(x).encode()).hexdigest()[:20]


def val(o):
    """a stable value for an object-API options object"""
    if o is None:
        return None
    if isinstance(o, np.ndarray):
        return ("nd", str(o.dtype), o.tolist())
    if isinstance(o, (list, tuple)):
        return tuple(val(x) for x in o)
    if isinstance(o, (bytes, bytearray)):
        return ("b", bytes(o).hex())
    if hasattr(o, "__dict__"):
        return (type(o).__name__, tuple(sorted((k, val(v)) for k, v in vars(o).items())))
    return o


class Model:
    def __init__(self, path, idx):
        _, b, e = section(path, idx)
        self.f = open(path, "rb")
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        self.view = memoryview(self.mm)[b:e]
        self.fb = schema.Model.GetRootAsModel(self.view, 0)
        self.bh = {}
        self.codes = [self.fb.OperatorCodes(i) for i in range(self.fb.OperatorCodesLength())]

    def buf_hash(self, bi):
        if bi not in self.bh:
            b = self.fb.Buffers(bi)
            if b.DataLength():
                off, size = b._tab.Vector(b._tab.Offset(4)), b.DataLength()
            elif b.Size():
                off, size = b.Offset(), b.Size()
            else:
                off, size = 0, 0
            self.bh[bi] = (size, hashlib.sha1(self.view[off:off + size]).hexdigest()[:20]) if size else None
        return self.bh[bi]

    def subgraph_sig(self, si):
        sg = self.fb.Subgraphs(si)
        inputs = sg.InputsAsNumpy().tolist() if sg.InputsLength() else []
        outputs = sg.OutputsAsNumpy().tolist() if sg.OutputsLength() else []
        sig = {}
        tens = [schema.TensorT.InitFromObj(sg.Tensors(i)) for i in range(sg.TensorsLength())]

        def qsig(t):
            q = t.quantization
            if q is None:
                return None
            det = None
            if q.detailsType == schema.QuantizationDetails.BlockwiseQuantization and q.details is not None:
                d = q.details
                det = ("blockwise", d.blockSize, tsig(d.scales) if d.scales >= 0 else None,
                       tsig(d.zeroPoints) if d.zeroPoints is not None and d.zeroPoints >= 0 else None)
            return (val(q.scale), val(q.zeroPoint), q.quantizedDimension, det)

        def tsig(i):
            if i < 0:
                return "none"
            if i in sig:
                return sig[i]
            t = tens[i]
            if i in inputs:
                s = ("in", inputs.index(i))
            else:
                bh = self.buf_hash(t.buffer) if t.buffer else None
                if bh is not None:
                    s = ("const", t.type, val(t.shape), qsig(t), bh)
                else:
                    s = ("other", t.type, val(t.shape), t.isVariable)
            sig[i] = H(s)
            return sig[i]

        ops = collections.Counter()
        for oi in range(sg.OperatorsLength()):
            o = sg.Operators(oi)
            ot = schema.OperatorT.InitFromObj(o)
            c = self.codes[ot.opcodeIndex]
            code = (max(c.BuiltinCode(), c.DeprecatedBuiltinCode()), c.CustomCode())
            ins = tuple(tsig(i) for i in (ot.inputs.tolist() if ot.inputs is not None else []))
            osig = H((code, val(ot.builtinOptions), val(getattr(ot, "builtinOptions2", None)), val(ot.customOptions), ins))
            ops[osig] += 1
            for j, x in enumerate(ot.outputs.tolist() if ot.outputs is not None else []):
                t = tens[x]
                sig[x] = H(("out", osig, j, t.type, val(t.shape), qsig(t)))
        name = bytes(sg.Name() or b"").decode(errors="replace")
        return name, ops, tuple(tsig(i) for i in outputs), len(inputs)


def main(a, b, idx, out, n=7):
    ma, mb = Model(a, idx), Model(b, idx)
    rows, same = [], True
    for si in range(n):
        na, oa, outa, nia = ma.subgraph_sig(si)
        nb, ob, outb, nib = mb.subgraph_sig(si)
        eq_ops, eq_out = oa == ob, outa == outb
        same = same and eq_ops and eq_out and na == nb
        rows.append({"subgraph": si, "name": [na, nb], "ops": [sum(oa.values()), sum(ob.values())], "same_op_multiset": eq_ops,
                     "same_outputs": eq_out, "inputs": [nia, nib],
                     "ops_only_a": sum((oa - ob).values()), "ops_only_b": sum((ob - oa).values())})
        print(f"{si} {na}: ops {sum(oa.values())} / {sum(ob.values())}, same op multiset {eq_ops}, same outputs {eq_out} "
              f"(only a {sum((oa - ob).values())}, only b {sum((ob - oa).values())})")
    res = {"a": a, "b": b, "section": idx, "subgraphs": rows, "structurally_equal": same}
    json.dump(res, open(out, "w"), indent=1)
    print("STRUCTURALLY EQUAL" if same else "DIFFERENT")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5]) if len(sys.argv) > 5 else 7))
