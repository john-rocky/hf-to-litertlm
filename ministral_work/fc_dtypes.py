#!/usr/bin/env python3
"""fc_dtypes.py BUNDLE.litertlm OUT.json - a check for the lightweight export (FC weights all int8 / int4, 0 fp32 FC
weight; file size as expected): for every TFLite section of a bundle (mmapped; only the flatbuffer structure and the Buffer size
fields are read, never the weight bytes):
  - per signature subgraph: the FULLY_CONNECTED weight inputs (slot 1) by dtype / quantization form (blockwise<N> /
    channelwise / per-tensor / none) and their element counts; EMBEDDING_LOOKUP / GATHER table inputs the same way
  - per section: the constant buffers (inline data or external Buffer.offset/size), deduplicated by buffer index, by tensor
    dtype: count, bytes, elements
  - the unique FC weight buffers by form (a weight shared by the 6 prefill signatures and decode counts once)
Verdict line: fp32 FC weights (unique buffers) = 0 required; the element totals are printed for the size check against the
text tower's config (decoder 34 x (q 4096x4096 + k/v 1024x4096 + o 4096x4096 + gate/up/down 3 x 14336x4096) + lm_head
131072x4096 = 7,952,400,384 FC weights; embedding 131072x4096 = 536,870,912 in the embedder section). The export python
(ai_edge_litert schema + litert_lm_builder header). Exit 0 = no fp32 FC weight, 1 = some."""
import collections
import json
import mmap
import os
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
        d = fb.BlockwiseQuantization()
        d.Init(q.Details().Bytes, q.Details().Pos)
        return f"blockwise{d.BlockSize()}"
    n = q.ScaleLength()
    return "channelwise" if n > 1 else ("per-tensor" if n == 1 else "none")


def numel(t):
    n = 1
    for i in range(t.ShapeLength()):
        n *= t.Shape(i)
    return n


def main(path, out):
    rec = {"bundle": os.path.abspath(path), "file_bytes": os.path.getsize(path), "sections": []}
    bad = 0
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        for typ, b, e, items in sections(path):
            srec = {"type": typ, "bytes": e - b, "items": items}
            if typ != "TFLiteModel":
                rec["sections"].append(srec)
                continue
            m = fb.Model.GetRootAsModel(memoryview(mm)[b:e], 0)
            sig = {}
            for k in range(m.SignatureDefsLength()):
                sd = m.SignatureDefs(k)
                sig[sd.SubgraphIndex()] = sd.SignatureKey().decode()

            def is_const(bi):
                if bi <= 0:
                    return False
                bb = m.Buffers(bi)
                return bb.DataLength() > 0 or bb.Offset() > 1

            def bsize(bi):
                bb = m.Buffers(bi)
                return bb.DataLength() if bb.DataLength() > 0 else bb.Size()

            const_by_dtype = collections.defaultdict(lambda: {"buffers": 0, "bytes": 0, "elements": 0})
            seen = set()
            fc_unique = collections.defaultdict(lambda: {"buffers": 0, "elements": 0, "bytes": 0})
            fc_seen = set()
            per_sig = {}
            for si in range(m.SubgraphsLength()):
                sg = m.Subgraphs(si)
                for ti in range(sg.TensorsLength()):
                    t = sg.Tensors(ti)
                    bi = t.Buffer()
                    if bi in seen or not is_const(bi):
                        continue
                    seen.add(bi)
                    d = const_by_dtype[f"{TT.get(t.Type(), t.Type())}/{qform(t)}"]
                    d["buffers"] += 1
                    d["bytes"] += bsize(bi)
                    d["elements"] += numel(t)
                if si not in sig:
                    continue
                fcs, tabs = collections.Counter(), collections.Counter()
                fce = collections.Counter()
                for oi in range(sg.OperatorsLength()):
                    op = sg.Operators(oi)
                    n = op_name(m, op)
                    if n == "FULLY_CONNECTED" and op.InputsLength() > 1 and op.Inputs(1) >= 0:
                        w = sg.Tensors(op.Inputs(1))
                        key = f"{TT.get(w.Type(), w.Type())}/{qform(w)}" + ("" if is_const(w.Buffer()) else "/NON-CONST")
                        fcs[key] += 1
                        fce[key] += numel(w)
                        if w.Buffer() not in fc_seen and is_const(w.Buffer()):
                            fc_seen.add(w.Buffer())
                            u = fc_unique[key]
                            u["buffers"] += 1
                            u["elements"] += numel(w)
                            u["bytes"] += bsize(w.Buffer())
                    if n in ("EMBEDDING_LOOKUP", "GATHER"):
                        for x in range(op.InputsLength()):
                            ti = op.Inputs(x)
                            if ti >= 0 and is_const(sg.Tensors(ti).Buffer()):
                                w = sg.Tensors(ti)
                                tabs[f"{n}:{TT.get(w.Type(), w.Type())}/{qform(w)}{list(w.ShapeAsNumpy()) if w.ShapeLength() else []}"] += 1
                per_sig[sig[si]] = {"fc_weights": dict(fcs), "fc_weight_elements": dict(fce), "tables": dict(tabs)}
            srec.update({"signatures": per_sig, "constants_by_dtype": dict(const_by_dtype),
                         "fc_unique_weights": dict(fc_unique)})
            fp32_fc = sum(v["buffers"] for k, v in fc_unique.items() if k.startswith("FLOAT32"))
            nonconst = sum(c for s in per_sig.values() for k, c in s["fc_weights"].items() if k.endswith("NON-CONST"))
            srec["fp32_fc_unique"] = fp32_fc
            srec["fc_non_constant_weight_ops"] = nonconst
            bad += fp32_fc
            rec["sections"].append(srec)
    rec["verdict"] = "PASS" if bad == 0 else "FAIL"
    json.dump(rec, open(out, "w"), indent=1)
    for s in rec["sections"]:
        if s["type"] != "TFLiteModel":
            print(f"{s['type']}: {s['bytes']:,} B")
            continue
        print(f"TFLiteModel {s['items'].get('model_type', '?')}: {s['bytes']:,} B; signatures {list(s['signatures'])}")
        for k, v in sorted(s["fc_unique_weights"].items()):
            print(f"  FC unique weights {k}: {v['buffers']} buffers, {v['elements']:,} elements, {v['bytes']:,} B")
        for k, v in sorted(s["constants_by_dtype"].items(), key=lambda kv: -kv[1]["bytes"]):
            if v["bytes"] >= 1 << 20:
                print(f"  constants {k}: {v['buffers']} buffers, {v['bytes']:,} B, {v['elements']:,} elements")
        first = next(iter(s["signatures"].values()), None)
        if first:
            print(f"  first signature FC weights {first['fc_weights']} tables {first['tables']}")
        print(f"  fp32 FC weights (unique) {s['fp32_fc_unique']}, FC ops with a non-constant weight {s['fc_non_constant_weight_ops']}")
    print(f"fc_dtypes {os.path.basename(path)}: {rec['file_bytes']:,} B -> {rec['verdict']}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
