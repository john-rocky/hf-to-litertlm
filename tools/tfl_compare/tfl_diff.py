#!/usr/bin/env python3
"""tfl_diff.py A.litertlm B.litertlm SECTION_INDEX OUT.json: where two .litertlm files' TFLiteModel sections differ
(a recipe re-run's main graph vs the published one: same size, different sha256). Reads both sections through mmap (no copy),
compares them in 16 MiB chunks (numpy), then maps every differing byte to the flatbuffer: a buffer (its tensors: subgraph, name,
type, shape, and for an fp16 scale buffer the max |a - b| / max relative difference, for an int4 / int8 weight buffer the number of
differing bytes) or the non-buffer part of the flatbuffer (names, operators, metadata). Read-only. Needs the ai-edge-litert and
litert-lm-builder packages (the TFLite schema and the bundle header schema)."""
import json
import mmap
import sys

import numpy as np
from ai_edge_litert import schema_py_generated as schema
from litert_lm_builder import litertlm_core
from litert_lm_builder import litertlm_header_schema_py_generated as hschema

TYPES = {getattr(schema.TensorType, k): k for k in dir(schema.TensorType) if not k.startswith("_")}


def section(path, want):
    with open(path, "rb") as f:
        prefix = f.read(32)
        end = int.from_bytes(prefix[litertlm_core.HEADER_END_LOCATION_BYTE_OFFSET:
                                    litertlm_core.HEADER_END_LOCATION_BYTE_OFFSET + 8], "little")
        f.seek(0)
        head = f.read(end)
    md = hschema.LiteRTLMMetaData.GetRootAs(head[litertlm_core.HEADER_BEGIN_BYTE_OFFSET:], 0)
    s = md.SectionMetadata().Objects(want)
    return litertlm_core.any_section_data_type_to_string(s.DataType()), s.BeginOffset(), s.EndOffset()


def buffers(view):
    fb = schema.Model.GetRootAsModel(view, 0)
    spans = []  # (start, end, buffer index) relative to the section
    for bi in range(fb.BuffersLength()):
        b = fb.Buffers(bi)
        if b.DataLength():
            off, size = b._tab.Vector(b._tab.Offset(4)), b.DataLength()
        elif b.Size():
            off, size = b.Offset(), b.Size()
        else:
            continue
        spans.append((off, off + size, bi))
    users = {}
    for si in range(fb.SubgraphsLength()):
        sg = fb.Subgraphs(si)
        for ti in range(sg.TensorsLength()):
            t = sg.Tensors(ti)
            users.setdefault(t.Buffer(), []).append({"subgraph": si, "tensor": bytes(t.Name() or b"").decode(errors="replace")[-120:],
                                                     "type": TYPES.get(t.Type(), t.Type()), "shape": t.ShapeAsNumpy().tolist()
                                                     if t.ShapeLength() else []})
    return fb, sorted(spans), users


def main(a_path, b_path, idx, out):
    ta, ba, ea = section(a_path, idx)
    tb, bb, eb = section(b_path, idx)
    fa, fb_ = open(a_path, "rb"), open(b_path, "rb")
    ma = mmap.mmap(fa.fileno(), 0, access=mmap.ACCESS_READ)
    mb = mmap.mmap(fb_.fileno(), 0, access=mmap.ACCESS_READ)
    na, nb = ea - ba, eb - bb
    res = {"a": a_path, "b": b_path, "section": idx, "types": [ta, tb], "bytes": [na, nb]}
    n = min(na, nb)
    CH = 16 << 20
    diff_offsets = []  # first byte of each differing run (relative), capped
    n_diff = 0
    runs = []
    for off in range(0, n, CH):
        k = min(CH, n - off)
        x = np.frombuffer(ma, dtype=np.uint8, count=k, offset=ba + off)
        y = np.frombuffer(mb, dtype=np.uint8, count=k, offset=bb + off)
        d = np.nonzero(x != y)[0]
        if d.size == 0:
            continue
        n_diff += int(d.size)
        d = d + off
        # collapse into runs
        starts = np.concatenate(([d[0]], d[1:][np.diff(d) > 1]))
        ends = np.concatenate((d[:-1][np.diff(d) > 1], [d[-1]]))
        for s, e in zip(starts.tolist(), ends.tolist()):
            if runs and s <= runs[-1][1] + 1:
                runs[-1][1] = e
            else:
                runs.append([s, e])
    res["differing_bytes"] = n_diff
    res["differing_runs"] = len(runs)
    res["first_runs"] = runs[:20]
    fba, spans, users = buffers(memoryview(ma)[ba:ea])
    # map runs to buffers
    per_buf = {}
    outside = 0
    j = 0
    for s, e in runs:
        hit = False
        # binary-ish walk: spans are sorted by start
        while j < len(spans) and spans[j][1] <= s:
            j += 1
        k = j
        while k < len(spans) and spans[k][0] <= e:
            bs, be, bi = spans[k]
            lo, hi = max(s, bs), min(e + 1, be)
            if lo < hi:
                per_buf.setdefault(bi, [0, bs, be])[0] += hi - lo
                hit = True
            k += 1
        if not hit:
            outside += e - s + 1
    res["bytes_outside_buffers"] = outside
    rows = []
    for bi, (cnt, bs, be) in sorted(per_buf.items(), key=lambda kv: -kv[1][0]):
        u = users.get(bi, [])
        row = {"buffer": bi, "size": be - bs, "differing_bytes": cnt, "tensors": u[:3], "n_tensors": len(u)}
        typ = u[0]["type"] if u else None
        if typ == "FLOAT16":
            xa = np.frombuffer(ma, dtype=np.float16, count=(be - bs) // 2, offset=ba + bs).astype(np.float32)
            xb = np.frombuffer(mb, dtype=np.float16, count=(be - bs) // 2, offset=bb + bs).astype(np.float32)
            dd = np.abs(xa - xb)
            row["fp16_differing"] = int((xa != xb).sum())
            row["fp16_max_abs"] = float(dd.max())
            row["fp16_max_rel"] = float((dd / np.maximum(np.abs(xa), 1e-30)).max())
        rows.append(row)
    res["differing_buffers"] = len(rows)
    res["buffers"] = rows[:60]
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("types", "bytes", "differing_bytes", "differing_runs", "bytes_outside_buffers",
                                          "differing_buffers")}))
    for r in rows[:25]:
        t = r["tensors"][0] if r["tensors"] else {}
        extra = (f" fp16 differing {r['fp16_differing']} max|d| {r['fp16_max_abs']:.3g} max rel {r['fp16_max_rel']:.3g}"
                 if "fp16_differing" in r else "")
        print(f"buffer {r['buffer']} size {r['size']} differing {r['differing_bytes']} tensors {r['n_tensors']} "
              f"{t.get('type')} {t.get('shape')} sg {t.get('subgraph')} {t.get('tensor', '')[-70:]}{extra}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]))
