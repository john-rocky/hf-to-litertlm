"""List the elementwise ops (ADD / MUL / SUB / DIV / SELECT_V2 / MAXIMUM / MINIMUM) of a .tflite whose operands have
different ranks or include a constant of rank >= 2 (ML Drift's placement: a runtime rank-3 tensor is BHWC(a,1,b,c),
a constant is right-aligned HWC, so a runtime rank-3 + constant rank-4 pair is read misaligned). Groups identical
patterns; prints one example op index per pattern.

  python conversion/rank_dump.py <model.tflite> [--all]
"""
import collections
import json
import mmap
import sys

from ai_edge_litert import schema_py_generated as schema

OPS = {"ADD", "MUL", "SUB", "DIV", "SELECT_V2", "MAXIMUM", "MINIMUM", "SQUARED_DIFFERENCE", "POW"}


def main():
    path = sys.argv[1]
    f = open(path, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    m = schema.Model.GetRootAs(mm, 0)
    bo = {v: k for k, v in vars(schema.BuiltinOperator).items() if isinstance(v, int) and not k.startswith("_")}
    sg = m.Subgraphs(0)

    def info(i):
        t = sg.Tensors(i)
        b = m.Buffers(t.Buffer())
        const = b is not None and (b.DataLength() > 0 or b.Offset() > 1)
        return const, [int(t.Shape(k)) for k in range(t.ShapeLength())]
    pats = collections.OrderedDict()
    for o in range(sg.OperatorsLength()):
        op = sg.Operators(o)
        oc = m.OperatorCodes(op.OpcodeIndex())
        name = bo.get(max(oc.BuiltinCode(), oc.DeprecatedBuiltinCode()), "?")
        if name not in OPS:
            continue
        ins = [info(int(i)) for i in op.InputsAsNumpy() if int(i) >= 0]
        ranks = [len(s) for _, s in ins]
        mixed = len({r for r in ranks if r > 0}) > 1
        const_hi = any(c and len(s) >= 2 and any(d > 1 for d in s) for c, s in ins)
        if not (mixed or (const_hi and "--all" in sys.argv) or (mixed and const_hi)):
            continue
        out = info(int(op.OutputsAsNumpy()[0]))[1]
        key = (name, tuple(("C" if c else "R", tuple(s)) for c, s in ins), tuple(out))
        pats.setdefault(key, []).append(o)
    rows = []
    for (name, ins, out), idx in pats.items():
        kinds = [k for k, _ in ins]
        ranks = [len(s) for _, s in ins]
        risk = "RISK runtime-rank<4 + const-rank4" if any(k == "R" and r == 3 for k, r in zip(kinds, ranks)) and any(
            k == "C" and r == 4 for k, r in zip(kinds, ranks)) else ("mixed-rank" if len({r for r in ranks if r}) > 1 else "")
        rows.append({"op": name, "inputs": [f"{k}{list(s)}" for k, s in ins], "out": list(out), "count": len(idx),
                     "first": idx[0], "risk": risk})
        print(f"{name:8s} x{len(idx):3d} first op {idx[0]:5d}  {' + '.join(f'{k}{list(s)}' for k, s in ins)} -> "
              f"{list(out)}  {risk}")
    return rows


if __name__ == "__main__":
    main()
