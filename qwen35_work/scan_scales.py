#!/usr/bin/env python3
"""scan_scales.py X.litertlm [OUT.json] - scan every blockwise tensor of every TFLiteModel section: zero fp16 scales,
subnormal scales 0 < s < 2^-14, scales at exactly 2^-14, min scale (XNNPACK's check is std::isnormal); a section with no
blockwise tensor prints `blockwise tensors 0`; sections are found by the bundle header (litertlm_core).
Exit 0 = every section all normal, 1 = not."""
import json
import sys

import numpy as np
from ai_edge_litert import schema_py_generated as schema
from litert_lm_builder import litertlm_core
from litert_lm_builder import litertlm_header_schema_py_generated as hschema

TINY = np.float16(2.0 ** -14)


def sections(path):
    with open(path, "rb") as f:
        prefix = f.read(32)
        end = int.from_bytes(prefix[litertlm_core.HEADER_END_LOCATION_BYTE_OFFSET:
                                    litertlm_core.HEADER_END_LOCATION_BYTE_OFFSET + 8], "little")
        f.seek(0)
        head = f.read(end)
    md = hschema.LiteRTLMMetaData.GetRootAs(head[litertlm_core.HEADER_BEGIN_BYTE_OFFSET:], 0)
    out = []
    for i in range(md.SectionMetadata().ObjectsLength()):
        s = md.SectionMetadata().Objects(i)
        out.append((i, litertlm_core.any_section_data_type_to_string(s.DataType()), s.BeginOffset(), s.EndOffset()))
    return out


def main(path, out=None):
    data = open(path, "rb").read()
    ok_all, rows = True, []
    for idx, typ, b, e in sections(path):
        if typ != "TFLiteModel":
            continue
        view = memoryview(data)[b:e]
        fb = schema.Model.GetRootAsModel(bytes(view), 0)
        seen = set()
        n_t = n_zero = n_sub = n_tiny = n_scales = 0
        gmin = None
        per = []
        for si in range(fb.SubgraphsLength()):
            sg = fb.Subgraphs(si)
            for i in range(sg.TensorsLength()):
                t = sg.Tensors(i)
                q = t.Quantization()
                if q is None or q.DetailsType() != schema.QuantizationDetails.BlockwiseQuantization:
                    continue
                bq = schema.BlockwiseQuantization()
                tab = q.Details()
                bq.Init(tab.Bytes, tab.Pos)
                st = sg.Tensors(bq.Scales())
                bi = st.Buffer()
                if bi in seen:
                    continue
                seen.add(bi)
                buf = fb.Buffers(bi)
                off, size = (buf._tab.Vector(buf._tab.Offset(4)), buf.DataLength()) if buf.DataLength() else (buf.Offset(), buf.Size())
                sc = np.frombuffer(bytes(view[off:off + size]), dtype=np.float16)
                z = int((sc == 0).sum())
                s = int(((sc > 0) & (sc < TINY)).sum())
                ti = int((sc == TINY).sum())
                mn = float(sc.min())
                n_t += 1
                n_zero += z
                n_sub += s
                n_tiny += ti
                n_scales += sc.size
                gmin = mn if gmin is None else min(gmin, mn)
                if z or s:
                    per.append({"tensor": t.Name().decode(errors="replace")[-160:], "zero": z, "subnormal": s, "scales": int(sc.size)})
        ok = n_zero == 0 and n_sub == 0
        ok_all = ok_all and ok
        row = {"section": idx, "blockwise_tensors": n_t, "scales": n_scales, "zero": n_zero, "subnormal": n_sub,
               "at_2^-14": n_tiny, "min_scale": gmin, "all_normal": ok, "offenders": per}
        rows.append(row)
        mins = "n/a" if gmin is None else f"{gmin:.6g}"
        print(f"section {idx}: blockwise tensors {n_t}; scales {n_scales}; zero scales {n_zero}; subnormal scales {n_sub}; "
              f"scales at 2^-14 {n_tiny}; min scale {mins}; all normal: {ok}")
        for r in per[:10]:
            print("   ", r)
    if out:
        json.dump({"bundle": path, "sections": rows, "all_normal": ok_all}, open(out, "w"), indent=1)
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None))
