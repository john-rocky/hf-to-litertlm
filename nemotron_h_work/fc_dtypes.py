#!/usr/bin/env python3
"""fc_dtypes.py BUNDLE.litertlm OUT.json [FLOAT_RE] - the weight form of every FULLY_CONNECTED and EMBEDDING_LOOKUP in the
prefill_decode TFLite section, per signature subgraph: the weight input's tensor type (FLOAT32 / INT8 / INT4 / ...), its
quantization form (none / per-tensor / per-axis with the axis / blockwise with the block size), whether the weight is a constant
buffer (inline or external offset), and for the float ones the op's output tensor name (the scope ai-edge-quantizer matches).
FLOAT_RE (default 'Linear_in_proj_(a|b);') = a regex of FCs a recipe keeps float on purpose: counted apart (matched against the
output tensor name + ';', the quantizer's scope string); convert_nemotron_nano_9b.sh passes '(?!x)x' (none). The section is mmapped
and only the flatbuffer structure is walked (no weight bytes are read), so a 37 GB float export costs no RAM. Read only; needs
ai-edge-litert and litert-lm-builder, and const_ops.py next to it. Prints one line per signature and a total; exit 0."""
import collections
import json
import mmap
import os
import re
import sys

from ai_edge_litert import schema_py_generated as fb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from const_ops import op_name, sections  # noqa: E402

TT = {v: k for k, v in vars(fb.TensorType).items() if not k.startswith("_")}


def qform(t):
    q = t.Quantization()
    if q is None:
        return "none"
    if q.DetailsType() == fb.QuantizationDetails.BlockwiseQuantization:
        bq = fb.BlockwiseQuantization()
        tab = q.Details()
        bq.Init(tab.Bytes, tab.Pos)
        return f"blockwise{bq.BlockSize()}"
    n = q.ScaleLength()
    if n == 0:
        return "none"
    if n == 1:
        return "per-tensor"
    return f"per-axis(dim {q.QuantizedDimension()})"


def main(path, out, float_re="Linear_in_proj_(a|b);"):
    rx = re.compile(float_re)
    tfl = [s for s in sections(path) if s[0] == "TFLiteModel"]
    pd = [s for s in tfl if s[3].get("model_type", "").lower().replace("tf_lite_", "") == "prefill_decode"] or \
        sorted(tfl, key=lambda s: s[2] - s[1], reverse=True)
    _, b, e, items = pd[0]
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        m = fb.Model.GetRootAsModel(memoryview(mm)[b:e], 0)
        sig = {}
        for k in range(m.SignatureDefsLength()):
            sd = m.SignatureDefs(k)
            sig[sd.SubgraphIndex()] = sd.SignatureKey().decode()
        rec = {"bundle": os.path.abspath(path), "section_items": items, "float_regex": float_re, "signatures": {}}
        for si, name in sorted(sig.items()):
            sg = m.Subgraphs(si)
            forms = collections.Counter()
            float_fc, float_fc_re, nonconst = [], 0, 0
            for oi in range(sg.OperatorsLength()):
                op = sg.Operators(oi)
                n = op_name(m, op)
                if n not in ("FULLY_CONNECTED", "EMBEDDING_LOOKUP"):
                    continue
                wi = op.Inputs(1)
                w = sg.Tensors(wi)
                bb = m.Buffers(w.Buffer()) if w.Buffer() > 0 else None
                const = bool(bb and (bb.DataLength() > 0 or bb.Offset() > 1))
                dt = TT.get(w.Type(), str(w.Type()))
                forms[f"{n} {dt} {qform(w)}{'' if const else ' NON-CONST'}"] += 1
                nonconst += int(not const)
                if n == "FULLY_CONNECTED" and dt in ("FLOAT32", "FLOAT16"):
                    oname = sg.Tensors(op.Outputs(0)).Name().decode()
                    hit = bool(rx.search(oname + ";"))
                    float_fc_re += int(hit)
                    float_fc.append({"output": oname[-160:], "weight_shape": [w.Shape(x) for x in range(w.ShapeLength())],
                                     "matches_float_regex": hit})
            rec["signatures"][name] = {"forms": dict(sorted(forms.items())), "float_fc": len(float_fc),
                                       "float_fc_matching_regex": float_fc_re,
                                       "float_fc_not_matching_regex": [x for x in float_fc if not x["matches_float_regex"]][:20],
                                       "float_fc_examples": float_fc[:4], "non_constant_weights": nonconst}
    json.dump(rec, open(out, "w"), indent=1)
    for name, r in rec["signatures"].items():
        print(f"{name}: {r['forms']}; float FC {r['float_fc']} (regex {r['float_fc_matching_regex']}, other "
              f"{len(r['float_fc_not_matching_regex'])}); non-constant weights {r['non_constant_weights']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
