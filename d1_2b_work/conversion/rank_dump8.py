"""rank_dump.py over EVERY subgraph of a .tflite (rank_dump.py reads subgraph 0 only; the multi-signature decide
file holds one subgraph per signature): per subgraph, the elementwise ops (ADD / MUL / SUB / DIV / SELECT_V2 / MAXIMUM /
MINIMUM / SQUARED_DIFFERENCE / POW) whose operands have different ranks or include a constant of rank >= 2, grouped by
pattern, with the risky pattern flagged (a runtime rank-3 operand with a constant rank-4 operand: ML Drift places a
runtime rank 3 as BHWC(a,1,b,c) and right-aligns a constant, so the pair is read misaligned). The classification is
rank_dump.py's line for line.

  python conversion/rank_dump8.py <model.tflite> [--all] [--json out.json]
"""
from __future__ import annotations

import collections
import json
import mmap
import sys

from ai_edge_litert import schema_py_generated as schema

OPS = {"ADD", "MUL", "SUB", "DIV", "SELECT_V2", "MAXIMUM", "MINIMUM", "SQUARED_DIFFERENCE", "POW"}


def dump(path: str, show_all: bool = False, quiet: bool = False) -> list[dict]:
    f = open(path, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    m = schema.Model.GetRootAs(mm, 0)
    bo = {v: k for k, v in vars(schema.BuiltinOperator).items() if isinstance(v, int) and not k.startswith("_")}
    names = {}
    for i in range(m.SignatureDefsLength()):
        sd = m.SignatureDefs(i)
        names[sd.SubgraphIndex()] = sd.SignatureKey().decode()
    out = []
    for si in range(m.SubgraphsLength()):
        sg = m.Subgraphs(si)

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
            if not (mixed or (const_hi and show_all) or (mixed and const_hi)):
                continue
            o_shape = info(int(op.OutputsAsNumpy()[0]))[1]
            key = (name, tuple(("C" if c else "R", tuple(s)) for c, s in ins), tuple(o_shape))
            pats.setdefault(key, []).append(o)
        rows = []
        for (name, ins, o_shape), idx in pats.items():
            kinds = [k for k, _ in ins]
            ranks = [len(s) for _, s in ins]
            risk = ("RISK runtime-rank<4 + const-rank4" if any(k == "R" and r == 3 for k, r in zip(kinds, ranks)) and any(
                k == "C" and r == 4 for k, r in zip(kinds, ranks)) else
                ("mixed-rank" if len({r for r in ranks if r}) > 1 else ""))
            # the wider class: a constant of rank >= 2 (some dim > 1) beside a runtime operand of another
            # rank (right-aligned constant vs batch-first runtime placement); RISK is one case of it
            shapes = [s for _, s in ins]
            const_mixed = any(k == "C" and len(s) >= 2 and any(d > 1 for d in s) and any(
                k2 == "R" and len(s2) != len(s) for k2, s2 in zip(kinds, shapes)) for k, s in zip(kinds, shapes))
            rows.append({"op": name, "inputs": [f"{k}{list(s)}" for k, s in ins], "out": list(o_shape),
                         "count": len(idx), "first": idx[0], "risk": risk, "const_mixed_rank": const_mixed})
            if not quiet:
                print(f"[{si} {names.get(si, '?')}] {name:8s} x{len(idx):3d} first op {idx[0]:5d}  "
                      f"{' + '.join(f'{k}{list(s)}' for k, s in ins)} -> {list(o_shape)}  {risk}")
        out.append({"subgraph": si, "signature": names.get(si), "patterns": rows,
                    "risk_pairs": sum(r["count"] for r in rows if r["risk"].startswith("RISK")),
                    "const_mixed_rank_ops": sum(r["count"] for r in rows if r["const_mixed_rank"]),
                    "mixed_rank_ops": sum(r["count"] for r in rows if r["risk"])})
    return out


if __name__ == "__main__":
    res = dump(sys.argv[1], "--all" in sys.argv)
    for s in res:
        print(f"subgraph {s['subgraph']} {s['signature']}: risk_pairs {s['risk_pairs']}, const mixed-rank ops "
              f"{s['const_mixed_rank_ops']}, mixed-rank ops {s['mixed_rank_ops']}")
    if "--json" in sys.argv:
        json.dump(res, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)
