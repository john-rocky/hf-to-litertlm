#!/usr/bin/env python3
"""Per TFLiteModel section: what LiteRT's externalize_tflite_flatbuffer moved to the weights blob, what stayed in the
graph, and whether every tensor still carries the original bytes. Reads the .tflite flatbuffers only (ai_edge_litert's
generated schema, lazy API; no interpreter, no runtime).

  tflite_breakdown.py --orig MT=ORIG.tflite [--ext MT=NEW.tflite --blob MT=WEIGHTS] [...] --out breakdown.md
                      [--json breakdown.json] [--tsv externalized_by_dtype.tsv] [--graph-mark-mib 50] [--min-mib 1]

A model type given with --ext and --blob is an externalized section; these are checked (each failure is a reason):
  * ORIG and NEW have the same subgraphs, tensors (count, dtype, shape) and buffer count (the tool keeps all of them);
  * every NEW buffer is empty (cleared) or byte-equal to ORIG's buffer of the same index;
  * every tensor that is not external keeps its buffer index, and that buffer still holds ORIG's bytes;
  * every external tensor has no buffer, and BLOB[offset, offset + length) is byte-equal to the ORIG buffer it used;
  * payload: ORIG buffer bytes = NEW buffer bytes kept + bytes of the buffers the tool cleared;
  * every external buffer belongs to the group `tflite_weights` (the name LiteRT-LM binds the TFLiteWeights section
    to), starts 64-byte aligned and ends inside BLOB; BLOB size = the highest offset + length.
Written, not checked: externalized tensors by dtype (FLOAT_EXTERNAL=yes when a float dtype went to the blob; the
WebGPU streamed loading path takes INT8 / INT4 / INT2 external weights only), the NEW graph size with a mark above
--graph-mark-mib, the largest payloads still in the graph, buffers >= --min-mib. A model type given with --orig only
is a section kept inline: described, not checked. Exit 0 = every check passed, 1 = a check failed, 2 = bad input.
"""
import argparse
import collections
import hashlib
import json
import mmap
import os
import sys

import flatbuffers
from ai_edge_litert import schema_py_generated as S

GROUP = "tflite_weights"
ALIGN = 64
FLOATS = {"FLOAT32", "FLOAT16", "BFLOAT16", "FLOAT64"}
STREAMED = {"INT8", "INT4", "INT2"}


def enum_names(cls):
    return {v: k for k, v in vars(cls).items() if not k.startswith("_") and isinstance(v, int)}


TYPE = enum_names(S.TensorType)
OP = enum_names(S.BuiltinOperator)


def mib(n):
    return f"{n / 1048576:.2f}"


class Model:
    def __init__(self, path):
        self.path = path
        self.buf = open(path, "rb").read()
        self.mv = memoryview(self.buf)
        self.m = S.Model.GetRootAs(self.buf, 0)

    def buffer(self, i):
        """(memoryview of the payload, 'inline' | 'appended') or (None, None) for an empty buffer."""
        b = self.m.Buffers(i)
        n = b.DataLength()
        if n:
            o = flatbuffers.number_types.UOffsetTFlags.py_type(b._tab.Offset(4))
            start = b._tab.Vector(o)
            return self.mv[start:start + n], "inline"
        off, size = b.Offset(), b.Size()
        if off > 1 and size > 0:
            return self.mv[off:off + size], "appended"
        return None, None

    def op_name(self, opcode_index):
        c = self.m.OperatorCodes(opcode_index)
        code = c.BuiltinCode()
        if code == 0 and c.DeprecatedBuiltinCode() != 0:
            code = c.DeprecatedBuiltinCode()
        name = OP.get(code, f"op{code}")
        if name == "CUSTOM" and c.CustomCode():
            name = f"CUSTOM:{c.CustomCode().decode()}"
        return name

    def consumers(self):
        """(subgraph, tensor) -> ['OP@input', ...]; blockwise-quantization scale / zero-point tensors are tagged."""
        cons = collections.defaultdict(list)
        for sg_i in range(self.m.SubgraphsLength()):
            sg = self.m.Subgraphs(sg_i)
            for t_i in range(sg.TensorsLength()):
                q = sg.Tensors(t_i).Quantization()
                if q is not None and q.DetailsType() == S.QuantizationDetails.BlockwiseQuantization:
                    bq = S.BlockwiseQuantization()
                    tab = q.Details()
                    bq.Init(tab.Bytes, tab.Pos)
                    cons[(sg_i, bq.Scales())].append("blockwise-scales-of")
                    if bq.ZeroPoints() > 0:
                        cons[(sg_i, bq.ZeroPoints())].append("blockwise-zero-points-of")
            for o_i in range(sg.OperatorsLength()):
                o = sg.Operators(o_i)
                nm = self.op_name(o.OpcodeIndex())
                for pos in range(o.InputsLength()):
                    ti = o.Inputs(pos)
                    if ti >= 0:
                        cons[(sg_i, ti)].append(f"{nm}@{pos}")
        return cons


def shape_of(t):
    return [int(t.Shape(k)) for k in range(t.ShapeLength())]


def payload_table(M, cons, min_bytes):
    """Buffer payload grouped by (consumer ops, dtype) + the buffers >= min_bytes."""
    users = collections.defaultdict(list)
    for sg_i in range(M.m.SubgraphsLength()):
        sg = M.m.Subgraphs(sg_i)
        for t_i in range(sg.TensorsLength()):
            t = sg.Tensors(t_i)
            if t.Buffer() > 0:
                users[t.Buffer()].append((sg_i, t_i, (t.Name() or b"").decode(errors="replace"),
                                          TYPE.get(t.Type(), str(t.Type())), shape_of(t)))
    meta = {M.m.Metadata(k).Buffer(): (M.m.Metadata(k).Name() or b"").decode(errors="replace")
            for k in range(M.m.MetadataLength())}
    agg = collections.defaultdict(lambda: [0, 0])
    big = []
    total = 0
    for bi in range(M.m.BuffersLength()):
        v, where = M.buffer(bi)
        if v is None:
            continue
        n = len(v)
        total += n
        us = users.get(bi, [])
        ops = sorted({c for (sg_i, t_i, *_r) in us for c in cons.get((sg_i, t_i), [])})
        dts = sorted({u[3] for u in us}) or ([f"(metadata `{meta[bi]}`)"] if bi in meta else ["(no tensor)"])
        if not ops and us:
            ops = ["(no op input)"]
        key = (",".join(ops) or "(no consumer)", ",".join(dts))
        agg[key][0] += 1
        agg[key][1] += n
        if n >= min_bytes:
            big.append((n, bi, where, us, ops))
    return total, agg, big


def check_externalized(mt, orig, new, blob, min_bytes, mark_bytes):
    I, X = Model(orig), Model(new)
    why = []
    cons = X.consumers()
    nsg = I.m.SubgraphsLength()
    if nsg != X.m.SubgraphsLength():
        why.append(f"subgraphs {nsg} vs {X.m.SubgraphsLength()}")
    nb = I.m.BuffersLength()
    if nb != X.m.BuffersLength():
        why.append(f"buffers {nb} vs {X.m.BuffersLength()}")
    # buffers
    isha, cleared, changed = {}, set(), []
    p_orig = p_kept = p_cleared = 0
    for i in range(min(nb, X.m.BuffersLength())):
        vi, _ = I.buffer(i)
        vx, _ = X.buffer(i)
        if vi is not None:
            isha[i] = (len(vi), hashlib.sha256(vi).digest())
            p_orig += len(vi)
        if vx is not None:
            p_kept += len(vx)
            if vi is None or len(vx) != len(vi) or hashlib.sha256(vx).digest() != isha[i][1]:
                changed.append(i)
        elif vi is not None:
            cleared.add(i)
            p_cleared += len(vi)
    if changed:
        why.append(f"{len(changed)} NEW buffers differ from ORIG's buffer of the same index (first {changed[:5]})")
    if p_orig != p_kept + p_cleared:
        why.append(f"payload: ORIG {p_orig:,} != kept {p_kept:,} + cleared {p_cleared:,}")
    # external buffers and the blob
    ext = {}
    for k in range(X.m.ExternalBuffersLength()):
        e = X.m.ExternalBuffers(k)
        ext[e.Id()] = (e.Group(), e.Offset(), e.Length(), (e.Packing() or b"").decode())
    groups = [(X.m.ExternalBufferGroups(k).Name() or b"").decode() for k in range(X.m.ExternalBufferGroupsLength())]
    bsize = os.path.getsize(blob)
    fb = open(blob, "rb")
    bm = mmap.mmap(fb.fileno(), 0, access=mmap.ACCESS_READ) if bsize else b""
    bad_group = sum(1 for g, *_ in ext.values() if not (0 <= g < len(groups) and groups[g] == GROUP))
    misaligned = sum(1 for _, off, _, _ in ext.values() if off % ALIGN)
    outside = sum(1 for _, off, ln, _ in ext.values() if off + ln > bsize)
    end = max((off + ln for _, off, ln, _ in ext.values()), default=0)
    uniq = {(off, ln) for _, off, ln, _ in ext.values()}
    ubytes = sum(ln for _, ln in uniq)
    if bad_group:
        why.append(f"{bad_group} external buffers not in group `{GROUP}` (groups {groups})")
    if misaligned:
        why.append(f"{misaligned} external offsets not {ALIGN}-byte aligned")
    if outside:
        why.append(f"{outside} external buffers end past the blob ({bsize:,} B)")
    if end != bsize:
        why.append(f"blob size {bsize:,} != highest offset + length {end:,}")
    # tensors
    slice_sha = {}
    bad_t = collections.Counter()
    first_bad = []
    by_dtype = collections.defaultdict(lambda: [0, 0, set(), collections.Counter()])
    float_ext = []

    def bad(kind, where):
        bad_t[kind] += 1
        if len(first_bad) < 5:
            first_bad.append(f"{kind} {where}")
    for sg_i in range(min(nsg, X.m.SubgraphsLength())):
        si, sx = I.m.Subgraphs(sg_i), X.m.Subgraphs(sg_i)
        if si.TensorsLength() != sx.TensorsLength():
            bad("tensor count", f"sg{sg_i}")
            continue
        for t_i in range(si.TensorsLength()):
            ti, tx = si.Tensors(t_i), sx.Tensors(t_i)
            if ti.Type() != tx.Type() or shape_of(ti) != shape_of(tx):
                bad("dtype/shape changed", f"sg{sg_i}:t{t_i}")
            if ti.ExternalBuffer():
                bad("ORIG tensor already external", f"sg{sg_i}:t{t_i}")
            bi, bx, ex = ti.Buffer(), tx.Buffer(), tx.ExternalBuffer()
            if ex:
                e = ext.get(ex)
                if bx != 0:
                    bad("external tensor keeps a buffer", f"sg{sg_i}:t{t_i}")
                if e is None:
                    bad("unknown external id", f"sg{sg_i}:t{t_i}")
                    continue
                src = isha.get(bi)
                _, off, ln, _ = e
                if src is None:
                    bad("external tensor without ORIG data", f"sg{sg_i}:t{t_i}")
                elif ln != src[0] or off + ln > bsize:
                    bad("external length differs", f"sg{sg_i}:t{t_i}")
                else:
                    if (off, ln) not in slice_sha:
                        slice_sha[(off, ln)] = hashlib.sha256(bm[off:off + ln]).digest()
                    if slice_sha[(off, ln)] != src[1]:
                        bad("blob bytes differ from ORIG", f"sg{sg_i}:t{t_i}")
                dtype = TYPE.get(tx.Type(), str(tx.Type()))
                r = by_dtype[dtype]
                r[0] += 1
                r[1] += ln
                r[2].add((off, ln))
                tc = cons.get((sg_i, t_i), [])
                for c in tc:
                    r[3][c] += 1
                if dtype in FLOATS:
                    float_ext.append(dict(name=(tx.Name() or b"").decode(errors="replace"), dtype=dtype,
                                          shape=shape_of(tx), consumers=sorted(set(tc)), bytes=ln, offset=off,
                                          subgraph=sg_i))
            else:
                if bx != bi:
                    bad("buffer index changed", f"sg{sg_i}:t{t_i}")
                elif bi in cleared:
                    bad("tensor lost its data (buffer cleared)", f"sg{sg_i}:t{t_i}")
    n_ext = sum(v[0] for v in by_dtype.values())
    if bad_t:
        why.append("tensors: " + ", ".join(f"{k} {n}" for k, n in bad_t.items()) + f" (first: {'; '.join(first_bad)})")
    if bm:
        bm.close()
    fb.close()
    total, agg, big = payload_table(X, cons, min_bytes)
    gbytes = os.path.getsize(new)
    return dict(model_type=mt, form="externalized", orig=orig, new=new, blob=blob, orig_bytes=os.path.getsize(orig),
                new_graph_bytes=gbytes, blob_bytes=bsize, subgraphs=nsg, buffers=nb, ext_tensors=n_ext,
                ext_buffers=len(ext), unique_bytes=ubytes, orig_payload=p_orig, kept_payload=p_kept,
                cleared_payload=p_cleared, cleared_buffers=len(cleared), groups=groups,
                packing=dict(collections.Counter(v[3] for v in ext.values())),
                by_dtype={d: dict(tensors=v[0], tensor_bytes=v[1], unique_bytes=sum(ln for _, ln in v[2]),
                                  consumers=dict(v[3].most_common())) for d, v in by_dtype.items()},
                float_ext=float_ext, graph_over_mark=gbytes > mark_bytes, kept_by=agg, big=big, reasons=why)


def describe_inline(mt, orig, min_bytes, mark_bytes):
    I = Model(orig)
    total, agg, big = payload_table(I, I.consumers(), min_bytes)
    gbytes = os.path.getsize(orig)
    return dict(model_type=mt, form="kept_inline", orig=orig, orig_bytes=gbytes, new_graph_bytes=gbytes, blob_bytes=0,
                ext_tensors=0, ext_buffers=0, unique_bytes=0, orig_payload=total, kept_payload=total, cleared_payload=0,
                by_dtype={}, float_ext=[], graph_over_mark=gbytes > mark_bytes, kept_by=agg, big=big, reasons=[])


def kv(specs):
    d = collections.OrderedDict()
    for s in specs:
        k, v = s.split("=", 1)
        d[k] = v
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", action="append", required=True, help="MT=original section .tflite")
    ap.add_argument("--ext", action="append", default=[], help="MT=the tool's model.tflite")
    ap.add_argument("--blob", action="append", default=[], help="MT=the tool's weights blob")
    ap.add_argument("--out", required=True)
    ap.add_argument("--json", default=None)
    ap.add_argument("--tsv", default=None)
    ap.add_argument("--graph-mark-mib", type=float, default=50.0)
    ap.add_argument("--min-mib", type=float, default=1.0)
    a = ap.parse_args()
    origs, exts, blobs = kv(a.orig), kv(a.ext), kv(a.blob)
    if set(exts) != set(blobs) or not set(exts) <= set(origs):
        print("ERROR: every --ext needs a --blob and an --orig of the same model type")
        return 2
    min_bytes, mark = int(a.min_mib * 1048576), int(a.graph_mark_mib * 1048576)
    res = []
    for mt, p in origs.items():
        try:
            res.append(check_externalized(mt, p, exts[mt], blobs[mt], min_bytes, mark) if mt in exts
                       else describe_inline(mt, p, min_bytes, mark))
        except Exception as e:  # unreadable flatbuffer or blob
            print(f"ERROR: {mt}: {type(e).__name__}: {e}")
            return 2
    reasons = [f"{r['model_type']}: {w}" for r in res for w in r["reasons"]]
    float_any = any(r["float_ext"] for r in res)
    over = [r["model_type"] for r in res if r["graph_over_mark"]]
    L = ["# TFLite sections: what moved to the weights blob (`tflite_breakdown.py`; flatbuffer accessors only)", "",
         "## Checks per section", "",
         "| model_type | form | ORIG section bytes | NEW graph bytes (MiB) | weights blob bytes (MiB) | externalized tensors "
         "| external buffers (unique bytes) | buffers cleared | payload: ORIG = kept + cleared | result |",
         "|---|---|---:|---:|---:|---:|---:|---:|---|---|"]
    for r in res:
        pay = (f"{r['orig_payload']:,} = {r['kept_payload']:,} + {r['cleared_payload']:,}"
               if r["form"] == "externalized" else "— (not processed)")
        res_s = ("PASS" if not r["reasons"] else "**FAIL**") if r["form"] == "externalized" else "kept inline"
        L.append(f"| {r['model_type']} | {r['form']} | {r['orig_bytes']:,} | {r['new_graph_bytes']:,} ({mib(r['new_graph_bytes'])}) "
                 f"| {r['blob_bytes']:,} ({mib(r['blob_bytes'])}) | {r['ext_tensors']:,} | {r['ext_buffers']:,} "
                 f"({r['unique_bytes']:,}) | {r.get('cleared_buffers', 0):,} | {pay} | {res_s} |")
    L += [""] + [f"- FAIL {w}" for w in reasons]
    L += ["- every externalized tensor's blob bytes were compared with the ORIG buffer it used, and every other tensor's "
          "buffer with ORIG's (sha256 per buffer); external buffers: group `tflite_weights`, 64-byte aligned, blob size = "
          "highest offset + length"]
    for r in res:
        if r["form"] == "externalized":
            L.append(f"- {r['model_type']}: groups {r['groups']}, packing {r['packing']}, unique blob bytes "
                     f"{r['unique_bytes']:,} vs cleared payload {r['cleared_payload']:,}"
                     + ("" if r["unique_bytes"] == r["cleared_payload"] else " (differ: identical buffers merged, or a "
                        "buffer is both external and still referenced inline)"))
    L += ["", "## Externalized tensors, section × dtype", "",
          "| model_type | dtype | tensors | unique bytes | MiB | consumer op @ input (count) | WebGPU streamed path |",
          "|---|---|---:|---:|---:|---|---|"]
    rows = []
    for r in res:
        for d, v in sorted(r["by_dtype"].items(), key=lambda kv_: -kv_[1]["unique_bytes"]):
            ok = ("in the dtype list (INT8 / INT4 / INT2)" if d in STREAMED else
                  "**float external weight: not on that path**" if d in FLOATS else f"not in the list ({d})")
            c = ", ".join(f"{k} ({n})" for k, n in v["consumers"].items())
            L.append(f"| {r['model_type']} | {d} | {v['tensors']:,} | {v['unique_bytes']:,} | {mib(v['unique_bytes'])} | {c} | {ok} |")
            rows.append((r["model_type"], d, v["tensors"], v["unique_bytes"], c))
    L += ["", f"FLOAT_EXTERNAL={'yes' if float_any else 'no'}"]
    if float_any:
        L += ["", "### Float tensors in the blob", "",
              "| model_type | tensor | dtype | shape | consumer op @ input | bytes | offset | subgraph |",
              "|---|---|---|---|---|---:|---:|---:|"]
        for r in res:
            for f in sorted(r["float_ext"], key=lambda x: (-x["bytes"], x["name"])):
                L.append(f"| {r['model_type']} | `{f['name']}` | {f['dtype']} | {f['shape']} | {', '.join(f['consumers']) or '(none)'} "
                         f"| {f['bytes']:,} | {f['offset']:,} | {f['subgraph']} |")
    L += ["", f"## Graph after externalization (mark above {a.graph_mark_mib:g} MiB = {mark:,} B)", "",
          "| model_type | NEW graph bytes | MiB | over the mark | payload still in the graph | largest payloads still in the graph |",
          "|---|---:|---:|---|---:|---|"]
    for r in res:
        top = sorted(r["kept_by"].items(), key=lambda kv_: -kv_[1][1])[:4]
        tops = "; ".join(f"{ops} {dts} ×{cnt} {nbytes:,} B" for (ops, dts), (cnt, nbytes) in top)
        L.append(f"| {r['model_type']} | {r['new_graph_bytes']:,} | {mib(r['new_graph_bytes'])} | "
                 f"{'**GRAPH_OVER_MARK**' if r['graph_over_mark'] else 'no'} | {r['kept_payload']:,} | {tops} |")
    L += ["", f"## Buffers of {a.min_mib:g} MiB or more still in each graph", ""]
    for r in res:
        L.append(f"- {r['model_type']}: {len(r['big'])}")
        for n, bi, where, us, ops in sorted(r["big"], key=lambda x: -x[0])[:12]:
            ts = "; ".join(f"`{u[2]}` ({u[3]}, {u[4]})" for u in us[:2]) + (f"; +{len(us) - 2} more" if len(us) > 2 else "")
            L.append(f"  - buffer {bi}: {n:,} B {where}; {ts}; {', '.join(ops) or '(none)'}")
    verdict = "VERDICT: PASS" if not reasons else "VERDICT: FAIL (" + "; ".join(reasons) + ")"
    L += ["", verdict]
    with open(a.out, "w", encoding="utf-8") as fo:
        fo.write("\n".join(L) + "\n")
    if a.tsv:
        with open(a.tsv, "w", encoding="utf-8") as fo:
            fo.write("model_type\tdtype\ttensors\tunique_bytes\tconsumers\n")
            for row in rows:
                fo.write("\t".join(str(x) for x in row) + "\n")
    if a.json:
        out = []
        for r in res:
            r2 = {k: v for k, v in r.items() if k not in ("kept_by", "big")}
            r2["kept_top"] = [dict(consumers=ops, dtype=dts, buffers=cnt, bytes=nbytes)
                              for (ops, dts), (cnt, nbytes) in sorted(r["kept_by"].items(), key=lambda kv_: -kv_[1][1])[:8]]
            r2["big_buffers"] = len(r["big"])
            out.append(r2)
        with open(a.json, "w", encoding="utf-8") as fo:
            json.dump(dict(sections=out, float_external=float_any, graph_mark_bytes=mark, graph_over_mark=over,
                           verdict="PASS" if not reasons else "FAIL", reasons=reasons), fo, indent=1)
    print(verdict)
    print(f"FLOAT_EXTERNAL={'yes' if float_any else 'no'}")
    print(f"GRAPH_OVER_MARK={','.join(over) or 'none'}")
    return 0 if not reasons else 1


if __name__ == "__main__":
    sys.exit(main())
