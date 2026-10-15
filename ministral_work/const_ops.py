#!/usr/bin/env python3
"""const_ops.py BUNDLE.litertlm OUT.json - a check for the lightweight export (it defers constant folding):
for the prefill_decode TFLite section of a bundle, per signature subgraph, (1) the op histogram and (2) the ops whose every
input is a constant (a tensor backed by a buffer with inline data or an external Buffer.offset/size) = folding the converter
left in the graph, by op type, with the output shapes of a few. The section is mmapped and only the flatbuffer structure is
walked (no weight bytes are read). Decomposition subgraphs (odml.*) are listed apart. Read only; the export python
(ai_edge_litert schema + litert_lm_builder header)."""
import collections
import json
import mmap
import os
import sys

from ai_edge_litert import schema_py_generated as fb
from litert_lm_builder import litertlm_core
from litert_lm_builder import litertlm_header_schema_py_generated as hs

BN = {v: k for k, v in vars(fb.BuiltinOperator).items() if not k.startswith("_")}


def sections(path):
    with open(path, "rb") as f:
        head = f.read(1 << 16)
    he = int.from_bytes(head[litertlm_core.HEADER_END_LOCATION_BYTE_OFFSET:litertlm_core.HEADER_END_LOCATION_BYTE_OFFSET + 8],
                        "little")
    meta = hs.LiteRTLMMetaData.GetRootAs(bytearray(head[litertlm_core.HEADER_BEGIN_BYTE_OFFSET:he]), 0)
    out = []
    for i in range(meta.SectionMetadata().ObjectsLength()):
        so = meta.SectionMetadata().Objects(i)
        items = {}
        for j in range(so.ItemsLength()):
            it = so.Items(j)
            if it.ValueType() == hs.VData.StringValue:
                sv = hs.StringValue()
                sv.Init(it.Value().Bytes, it.Value().Pos)
                items[it.Key().decode()] = sv.Value().decode() if sv.Value() else ""
        out.append((litertlm_core.any_section_data_type_to_string(so.DataType()), so.BeginOffset(), so.EndOffset(), items))
    return out


def op_name(m, op):
    oc = m.OperatorCodes(op.OpcodeIndex())
    code = max(oc.BuiltinCode(), oc.DeprecatedBuiltinCode())
    name = BN.get(code, str(code))
    if name == "CUSTOM" and oc.CustomCode():
        name = "CUSTOM:" + oc.CustomCode().decode()
    return name


def main(path, out):
    tfl = [s for s in sections(path) if s[0] == "TFLiteModel"]
    pd = [s for s in tfl if s[3].get("model_type", "").lower().replace("tf_lite_", "") == "prefill_decode"] or \
        sorted(tfl, key=lambda s: s[2] - s[1], reverse=True)
    _, b, e, items = pd[0]
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        buf = memoryview(mm)[b:e]
        m = fb.Model.GetRootAsModel(buf, 0)

        def is_const(sg, t):
            if t < 0:
                return True  # optional input
            bi = sg.Tensors(t).Buffer()
            if bi <= 0:
                return False
            bb = m.Buffers(bi)
            return bb.DataLength() > 0 or bb.Offset() > 1

        sig_names = {}
        for k in range(m.SignatureDefsLength()):
            sd = m.SignatureDefs(k)
            sig_names[sd.SubgraphIndex()] = sd.SignatureKey().decode()
        rec = {"bundle": os.path.abspath(path), "section_items": items, "signatures": {}, "decomposition_subgraphs": 0}
        for si in range(m.SubgraphsLength()):
            sg = m.Subgraphs(si)
            name = sig_names.get(si) or (sg.Name().decode() if sg.Name() else f"sg{si}")
            if si not in sig_names:
                rec["decomposition_subgraphs"] += 1
                continue
            hist, const_hist, samples = collections.Counter(), collections.Counter(), collections.defaultdict(list)
            for oi in range(sg.OperatorsLength()):
                op = sg.Operators(oi)
                n = op_name(m, op)
                hist[n] += 1
                ins = [op.Inputs(x) for x in range(op.InputsLength())]
                if ins and all(is_const(sg, t) for t in ins):
                    const_hist[n] += 1
                    if len(samples[n]) < 3 and op.OutputsLength():
                        ot = sg.Tensors(op.Outputs(0))
                        samples[n].append([ot.Shape(x) for x in range(ot.ShapeLength())])
            rec["signatures"][name] = {"n_ops": sum(hist.values()), "ops": dict(sorted(hist.items())),
                                       "const_only_ops": dict(sorted(const_hist.items())),
                                       "const_only_total": sum(const_hist.values()),
                                       "const_only_output_shapes": dict(samples)}
    json.dump(rec, open(out, "w"), indent=1)
    for name, r in rec["signatures"].items():
        print(f"{name}: {r['n_ops']} ops; const-only {r['const_only_total']} {r['const_only_ops']}")
    print(f"decomposition subgraphs: {rec['decomposition_subgraphs']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
