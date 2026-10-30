#!/usr/bin/env python3
"""Round 2: a flatbuffer-only census of .tflite files (mmap, no interpreter: the fp32 decoder is 6.9 GB).

Per file: the signatures (input / output names, shapes, dtypes), per signature subgraph the op histogram, the
flagged ops (GATHER, GATHER_ND, CUSTOM, FLEX, STABLEHLO, BROADCAST_TO, RELU_0_TO_1), tensors of rank > 4,
operator inputs of rank 0 (memory rank0-reduction-keepdim-gpu), the dtype of every FULLY_CONNECTED / CONV / EMBEDDING
weight (input 1 of FC / CONV, input 1 of EMBEDDING_LOOKUP = the table), DEQUANTIZE count, and the derived-position op
counts (RELU, COS, SIN). -> one json.

    ~/venvs/lt0100dev/bin/python -I tools/wv_census.py OUT.json FILE.tflite [FILE.tflite ...]
"""
import json
import mmap
import sys
from pathlib import Path

FLAGGED = ("GATHER", "GATHER_ND", "CUSTOM", "FLEX", "STABLEHLO", "BROADCAST_TO", "RELU_0_TO_1")


def census(path):
    from ai_edge_litert import schema_py_generated as fb
    names = {v: k for k, v in vars(fb.BuiltinOperator).items() if not k.startswith("_")}
    tnames = {v: k for k, v in vars(fb.TensorType).items() if not k.startswith("_")}
    with open(path, "rb") as f:
        buf = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    m = fb.Model.GetRootAsModel(buf, 0)
    codes = []
    for i in range(m.OperatorCodesLength()):
        oc = m.OperatorCodes(i)
        code = max(oc.BuiltinCode(), oc.DeprecatedBuiltinCode())
        name = names.get(code, str(code))
        if name == "CUSTOM" and oc.CustomCode():
            name = "CUSTOM:" + oc.CustomCode().decode()
        codes.append(name)
    sig_of, sigs = {}, {}
    for i in range(m.SignatureDefsLength()):
        s = m.SignatureDefs(i)
        key = s.SignatureKey().decode()
        sig_of[s.SubgraphIndex()] = key
        sg = m.Subgraphs(s.SubgraphIndex())

        def tdesc(t):
            ten = sg.Tensors(t)
            return {"shape": [int(ten.Shape(j)) for j in range(ten.ShapeLength())], "dtype": tnames.get(ten.Type())}
        sigs[key] = {"inputs": {s.Inputs(j).Name().decode(): tdesc(s.Inputs(j).TensorIndex())
                                for j in range(s.InputsLength())},
                     "outputs": {s.Outputs(j).Name().decode(): tdesc(s.Outputs(j).TensorIndex())
                                 for j in range(s.OutputsLength())}}
    subs = []
    for si in range(m.SubgraphsLength()):
        sg = m.Subgraphs(si)
        hist, wdt, rank0, r5 = {}, {}, 0, 0
        for ti in range(sg.TensorsLength()):
            if sg.Tensors(ti).ShapeLength() > 4:
                r5 += 1
        for oi in range(sg.OperatorsLength()):
            op = sg.Operators(oi)
            name = codes[op.OpcodeIndex()]
            hist[name] = hist.get(name, 0) + 1
            ins = [op.Inputs(j) for j in range(op.InputsLength())]
            for t in ins:
                if t >= 0 and sg.Tensors(t).ShapeLength() == 0 and sg.Tensors(t).Buffer() == 0:
                    rank0 += 1
            widx = {"FULLY_CONNECTED": 1, "CONV_2D": 1, "DEPTHWISE_CONV_2D": 1, "EMBEDDING_LOOKUP": 1,
                    "BATCH_MATMUL": None}.get(name)
            if widx is not None and len(ins) > widx and ins[widx] >= 0:
                key = f"{name}:{tnames.get(sg.Tensors(ins[widx]).Type())}"
                wdt[key] = wdt.get(key, 0) + 1
        flagged = {n: sum(c for k, c in hist.items() if k == n or (n in ("CUSTOM", "FLEX", "STABLEHLO")
                                                                   and n in k.upper())) for n in FLAGGED}
        subs.append({"subgraph": si, "signature": sig_of.get(si), "ops": sum(hist.values()),
                     "hist": dict(sorted(hist.items())), "flagged": flagged, "weight_dtypes": wdt,
                     "rank0_activation_inputs": rank0, "tensors_rank_gt4": r5,
                     "derived_rope_ops": {k: hist.get(k, 0) for k in ("RELU", "COS", "SIN", "CAST")},
                     "dequantize": hist.get("DEQUANTIZE", 0)})
    return {"file": str(path), "bytes": Path(path).stat().st_size, "signatures": sigs, "subgraphs": subs,
            "flagged_total": {n: sum(s["flagged"][n] for s in subs) for n in FLAGGED},
            "rank_gt4_total": sum(s["tensors_rank_gt4"] for s in subs)}


def main():
    out, files = sys.argv[1], sys.argv[2:]
    res = {}
    for p in files:
        res[Path(p).name] = census(p)
        c = res[Path(p).name]
        print(Path(p).name, c["bytes"], "flagged", {k: v for k, v in c["flagged_total"].items() if v},
              "rank>4", c["rank_gt4_total"], "subgraphs", len(c["subgraphs"]),
              "w", c["subgraphs"][0]["weight_dtypes"], flush=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    sys.exit(main())
