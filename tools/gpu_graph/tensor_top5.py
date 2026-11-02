#!/usr/bin/env python3
"""tensor_top5.py BUNDLE.litertlm OUT.json [LIMIT_GIB] - check, before a phone run, that no GPU buffer of a bundle exceeds a limit.

For every signature subgraph of the bundle's prefill_decode TFLite section, every tensor's shape x 4 B (= the float32 size a GPU
delegate would allocate for it) is computed, and two top-5 lists are written per signature:
  - non-constant tensors (activations, inputs, outputs, state): the check. PASS = no non-constant tensor of any signature reaches
    LIMIT_GIB (default 1 = the max single allocation the Pixel 11 Pro's PowerVR OpenCL reported: 1,073,741,824 B);
  - constant tensors (weights): their shape x 4 B and their bytes as stored. Listed as a note, not judged: on the GPU a quantized
    weight is not necessarily held as float32, and the phone's failure was an activation (prefill_1024's attention score
    {1, 4, 4096, 1, 32768} float32 = 2 GiB, in Qwen3.5-4B-3D-Coder's 32,768-token file).
The section is mmapped and only the flatbuffer structure is walked (no weight bytes are read). Read only. Needs ai-edge-litert (the
TFLite schema) and litert-lm-builder (the bundle header, through this repo's qwen35_work/const_ops.py, loaded by its path).
Prints one line per signature and a verdict line TENSOR_TOP5 PASS|FAIL; exit 0 on PASS, 1 on FAIL."""
import importlib.util
import json
import math
import mmap
import os
import sys

from ai_edge_litert import schema_py_generated as fb

CONST_OPS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "qwen35_work", "const_ops.py")
_spec = importlib.util.spec_from_file_location("qwen35_const_ops", CONST_OPS)
const_ops = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(const_ops)

TT = {v: k for k, v in vars(fb.TensorType).items() if not k.startswith("_")}
STORED = {"FLOAT32": 4, "FLOAT16": 2, "BFLOAT16": 2, "FLOAT64": 8, "INT64": 8, "INT32": 4, "UINT32": 4, "INT16": 2,
          "UINT16": 2, "INT8": 1, "UINT8": 1, "BOOL": 1, "INT4": 0.5}


def main(path, out, limit_gib="1"):
    limit = int(float(limit_gib) * (1 << 30))
    tfl = [s for s in const_ops.sections(path) if s[0] == "TFLiteModel"]
    pd = [s for s in tfl if s[3].get("model_type", "").lower().replace("tf_lite_", "") == "prefill_decode"] or \
        sorted(tfl, key=lambda s: s[2] - s[1], reverse=True)
    _, b, e, items = pd[0]
    rec = {"bundle": os.path.abspath(path), "bundle_bytes": os.path.getsize(path), "section_items": items,
           "limit_bytes": limit, "rule": "non-constant tensor shape x 4 B < limit in every signature", "signatures": {}}
    worst = []
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        m = fb.Model.GetRootAsModel(memoryview(mm)[b:e], 0)
        sig = {}
        for k in range(m.SignatureDefsLength()):
            sd = m.SignatureDefs(k)
            sig[sd.SubgraphIndex()] = sd.SignatureKey().decode()
        for si, name in sorted(sig.items()):
            sg = m.Subgraphs(si)
            nonc, cons = [], []
            for ti in range(sg.TensorsLength()):
                t = sg.Tensors(ti)
                shape = [t.Shape(x) for x in range(t.ShapeLength())]
                n = math.prod(shape) if shape else 1
                bb = m.Buffers(t.Buffer()) if t.Buffer() > 0 else None
                const = bool(bb and (bb.DataLength() > 0 or bb.Offset() > 1))
                dt = TT.get(t.Type(), str(t.Type()))
                row = {"tensor": ti, "name": t.Name().decode(errors="replace")[-140:], "shape": shape, "dtype": dt,
                       "bytes_x4": n * 4}
                if const:
                    row["bytes_stored"] = (bb.Size() if bb.Offset() > 1 else bb.DataLength()) or int(n * STORED.get(dt, 4))
                    cons.append(row)
                else:
                    nonc.append(row)
            nonc.sort(key=lambda r: -r["bytes_x4"])
            cons.sort(key=lambda r: -r["bytes_x4"])
            over = [r for r in nonc if r["bytes_x4"] >= limit]
            rec["signatures"][name] = {"subgraph": si, "tensors": sg.TensorsLength(), "non_constant_top5": nonc[:5],
                                       "non_constant_at_or_over_limit": len(over),
                                       "constant_top5": cons[:5],
                                       "constant_at_or_over_limit_x4": len([r for r in cons if r["bytes_x4"] >= limit])}
            if nonc:
                worst.append((nonc[0]["bytes_x4"], name, nonc[0]["shape"]))
    ok = all(v["non_constant_at_or_over_limit"] == 0 for v in rec["signatures"].values())
    rec["verdict"] = "PASS" if ok else "FAIL"
    json.dump(rec, open(out, "w"), indent=1)
    for name, v in rec["signatures"].items():
        top = v["non_constant_top5"][0] if v["non_constant_top5"] else None
        ctop = v["constant_top5"][0] if v["constant_top5"] else None
        print(f"{name}: largest non-constant {top['shape'] if top else '-'} {top['dtype'] if top else ''} = "
              f"{(top['bytes_x4'] / (1 << 20)) if top else 0:,.1f} MiB at x4 B; at or over the limit {v['non_constant_at_or_over_limit']}; "
              f"largest constant {ctop['shape'] if ctop else '-'} {ctop['dtype'] if ctop else ''} "
              f"(x4 B {(ctop['bytes_x4'] / (1 << 20)) if ctop else 0:,.1f} MiB, stored {(ctop['bytes_stored'] / (1 << 20)) if ctop else 0:,.1f} MiB)")
    print(f"TENSOR_TOP5 {rec['verdict']} limit {limit:,} B; largest non-constant over all signatures: "
          f"{max(worst)[0] / (1 << 20):,.1f} MiB ({max(worst)[1]}, {max(worst)[2]})" if worst else "TENSOR_TOP5 no tensors")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
