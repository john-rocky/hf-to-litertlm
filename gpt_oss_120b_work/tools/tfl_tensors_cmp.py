#!/usr/bin/env python3
"""tfl_tensors_cmp.py A.litertlm B.litertlm OUT.json [--subgraph 0|all]

Checks a partial re-run against a published file: every constant tensor of A's TFLiteModel section (for example the
file that `LAYERS=2 convert_gpt_oss_120b.sh` makes) against the tensor of the same name in B's (the published 36-layer
file): the buffer bytes (sha256 of each, read with large preads, which on a USB hard disk are far faster than mmap page
faults) and the quantization parameters (scales, zero points, quantized dimension). Names in A with no tensor of that
name in B, and tensors whose type or shape differ, are listed. The flatbuffer itself is read through mmap (only its
tables are touched). Read-only. Exit 0 when every tensor of A that B has is equal. Needs the ai-edge-litert package (the
TFLite schema) and litert-lm-builder (the container header), which tools/requirements.txt does not install; checked
with ai-edge-litert 2.3.0 and litert-lm-builder 0.19.1.

--subgraph N: the named tensors of subgraph N only. Unnamed constants all share the key '' there, so only the last one
of them is compared, and a subgraph does not hold every tensor (the writer's prefill subgraphs leave out the last
layer's q / o projections, its experts, the final norm and lm_head).
--subgraph all: every subgraph. Named tensors by name over the whole model (a name used for two buffers in one file is
listed as a clash); unnamed constants (shape operands, the attention mask, other constants the writer makes) per
signature as a multiset of (type, shape, bytes, sha256, quantization): each unnamed constant of A must have an equal
one in B's signature of the same name (B has more layers, so more of them). Exit 0 only when every named tensor is in B
and equal, every unnamed constant is matched, and neither file has a clash. Run a must-differ control beside it: a copy
of A with one byte of one tensor changed must exit 1.
"""
import argparse
import hashlib
import json
import mmap
import os
import sys
import time

from ai_edge_litert import schema_py_generated as schema
from litert_lm_builder import litertlm_core
from litert_lm_builder import litertlm_header_schema_py_generated as hschema

TYPES = {getattr(schema.TensorType, k): k for k in dir(schema.TensorType) if not k.startswith("_")}
CHUNK = 64 << 20


def tflite_section(path):
    with open(path, "rb") as f:
        prefix = f.read(32)
        end = int.from_bytes(prefix[litertlm_core.HEADER_END_LOCATION_BYTE_OFFSET:
                                    litertlm_core.HEADER_END_LOCATION_BYTE_OFFSET + 8], "little")
        f.seek(0)
        head = f.read(end)
    md = hschema.LiteRTLMMetaData.GetRootAs(head[litertlm_core.HEADER_BEGIN_BYTE_OFFSET:], 0)
    sm = md.SectionMetadata()
    for i in range(sm.ObjectsLength()):
        s = sm.Objects(i)
        if litertlm_core.any_section_data_type_to_string(s.DataType()) == "TFLiteModel":
            return s.BeginOffset(), s.EndOffset()
    raise SystemExit(f"{path}: no TFLiteModel section")


def tensors(view, sg_index):
    """The constant tensors of one subgraph by name (unnamed ones share the key '': the last one stays)."""
    return dict(tensor_rows(view, sg_index))


def signatures(view):
    fb = schema.Model.GetRootAsModel(view, 0)
    keys = {fb.SignatureDefs(i).SubgraphIndex(): fb.SignatureDefs(i).SignatureKey().decode()
            for i in range(fb.SignatureDefsLength())}
    return [keys.get(i, f"subgraph{i}") for i in range(fb.SubgraphsLength())]


def all_tensors(view):
    """Every subgraph: named constants by name, unnamed ones per signature as a list; names used for two buffers."""
    named, unnamed, clash = {}, {}, []
    for i, key in enumerate(signatures(view)):
        for name, r in tensor_rows(view, i):
            if not name:
                unnamed.setdefault(key, []).append(r)
            elif name in named and named[name]["off"] != r["off"]:
                clash.append(name)
            else:
                named.setdefault(name, r)
    return named, unnamed, clash


def tensor_rows(view, sg_index):
    fb = schema.Model.GetRootAsModel(view, 0)
    sg = fb.Subgraphs(sg_index)
    for ti in range(sg.TensorsLength()):
        t = sg.Tensors(ti)
        b = fb.Buffers(t.Buffer())
        if b.DataLength():
            off, size = b._tab.Vector(b._tab.Offset(4)), b.DataLength()
        elif b.Size():
            off, size = b.Offset(), b.Size()
        else:
            continue
        q = t.Quantization()
        qp = None
        if q is not None and (q.ScaleLength() or q.ZeroPointLength()):
            h = hashlib.sha256()
            if q.ScaleLength():
                h.update(q.ScaleAsNumpy().tobytes())
            if q.ZeroPointLength():
                h.update(q.ZeroPointAsNumpy().tobytes())
            qp = f"{h.hexdigest()}:{q.ScaleLength()}:{q.QuantizedDimension()}"
        name = (t.Name() or b"").decode(errors="replace")
        yield name, {"off": off, "size": size, "type": TYPES.get(t.Type(), t.Type()),
                     "shape": t.ShapeAsNumpy().tolist() if t.ShapeLength() else [], "quant": qp}


def sha(fd, base, off, size):
    h = hashlib.sha256()
    done = 0
    while done < size:
        n = min(CHUNK, size - done)
        h.update(os.pread(fd, n, base + off + done))
        done += n
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("out")
    ap.add_argument("--subgraph", default="0", help="a subgraph index, or all")
    a = ap.parse_args()
    if a.subgraph != "all":
        a.subgraph = int(a.subgraph)
    t0 = time.time()
    side = {}
    for key, path in (("a", a.a), ("b", a.b)):
        begin, end = tflite_section(path)
        fd = os.open(path, os.O_RDONLY)
        mm = mmap.mmap(fd, end, prot=mmap.PROT_READ)
        view = memoryview(mm)[begin:end]
        if a.subgraph == "all":
            named, unnamed, clash = all_tensors(view)
            side[key] = {"path": path, "fd": fd, "begin": begin, "mm": mm, "view": view, "t": named,
                         "unnamed": unnamed, "clash": clash}
            print(f"{key}: {path}: TFLiteModel [{begin}, {end}), every subgraph: {len(named)} named constant tensors, "
                  f"unnamed {', '.join(f'{k} {len(v)}' for k, v in unnamed.items())}, clashes {len(clash)}", flush=True)
        else:
            side[key] = {"path": path, "fd": fd, "begin": begin, "mm": mm, "view": view,
                         "t": tensors(view, a.subgraph)}
            print(f"{key}: {path}: TFLiteModel [{begin}, {end}), subgraph {a.subgraph}: {len(side[key]['t'])} constant "
                  f"tensors", flush=True)
    A, B = side["a"]["t"], side["b"]["t"]
    rows, equal, differ, missing, meta_diff, nbytes = [], 0, [], [], [], 0
    for name, ta in sorted(A.items(), key=lambda kv: kv[1]["off"]):
        tb = B.get(name)
        if tb is None:
            missing.append(name)
            continue
        if (ta["type"], ta["shape"], ta["size"]) != (tb["type"], tb["shape"], tb["size"]):
            meta_diff.append({"name": name, "a": [ta["type"], ta["shape"], ta["size"]],
                              "b": [tb["type"], tb["shape"], tb["size"]]})
            continue
        ha = sha(side["a"]["fd"], side["a"]["begin"], ta["off"], ta["size"])
        hb = sha(side["b"]["fd"], side["b"]["begin"], tb["off"], tb["size"])
        nbytes += ta["size"]
        ok = ha == hb and ta["quant"] == tb["quant"]
        rows.append({"name": name, "type": ta["type"], "shape": ta["shape"], "bytes": ta["size"], "sha256": ha,
                     "equal": ok, "quant_equal": ta["quant"] == tb["quant"]})
        if ok:
            equal += 1
        else:
            differ.append(name)
    res = {"a": a.a, "b": a.b, "subgraph": a.subgraph, "compared": len(rows), "equal": equal, "bytes_compared": nbytes,
           "differ": differ, "only_in_a": missing, "type_or_shape_differ": meta_diff, "seconds": round(time.time() - t0, 1),
           "rows": rows}
    if a.subgraph == "all":
        return finish_all(a, side, res, t0)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"compared {len(rows)} tensors ({nbytes:,} B): equal {equal}, differ {len(differ)}, type/shape differ "
          f"{len(meta_diff)}, only in A {len(missing)}; {res['seconds']} s")
    for n in differ[:10] + [m["name"] for m in meta_diff[:10]] + missing[:10]:
        print("  ", n)
    return 0 if not differ and not meta_diff else 1


def finish_all(a, side, res, t0):
    """--subgraph all: the unnamed constants per signature (multiset match), then the result and the exit code."""
    cache = {}

    def h(key, r):
        k = (key, r["off"])
        if k not in cache:
            cache[k] = sha(side[key]["fd"], side[key]["begin"], r["off"], r["size"])
        return cache[k]

    count, nbytes, unmatched, per_sig = 0, 0, [], {}
    for sig, la in side["a"]["unnamed"].items():
        want = {(r["type"], tuple(r["shape"]), r["size"]) for r in la}
        pool = {}
        for r in side["b"]["unnamed"].get(sig, []):
            if (r["type"], tuple(r["shape"]), r["size"]) in want:
                k = (r["type"], tuple(r["shape"]), r["size"], h("b", r), r["quant"])
                pool[k] = pool.get(k, 0) + 1
        per_sig[sig] = {"a": len(la), "b": len(side["b"]["unnamed"].get(sig, [])), "unmatched": 0}
        for r in la:
            k = (r["type"], tuple(r["shape"]), r["size"], h("a", r), r["quant"])
            count += 1
            nbytes += r["size"]
            if pool.get(k, 0) > 0:
                pool[k] -= 1
            else:
                per_sig[sig]["unmatched"] += 1
                unmatched.append({"signature": sig, "type": r["type"], "shape": r["shape"], "bytes": r["size"],
                                  "sha256": k[3], "quant": r["quant"]})
    res["unnamed"] = {"count": count, "bytes": nbytes, "matched": count - len(unmatched), "unmatched": unmatched,
                      "per_signature": per_sig}
    res["clash"] = {"a": side["a"]["clash"], "b": side["b"]["clash"]}
    res["seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"compared {res['compared']} named tensors ({res['bytes_compared']:,} B) over every subgraph: equal "
          f"{res['equal']}, differ {len(res['differ'])}, type/shape differ {len(res['type_or_shape_differ'])}, only in A "
          f"{len(res['only_in_a'])}")
    sigs = ", ".join(f"{k} {v['a']} of {v['b']} in B" for k, v in per_sig.items())
    print(f"unnamed constants: {count} in A ({nbytes:,} B; {sigs}): matched {count - len(unmatched)}, unmatched "
          f"{len(unmatched)}; clashes A {len(res['clash']['a'])} B {len(res['clash']['b'])}; {res['seconds']} s")
    for n in (res["differ"][:10] + [m["name"] for m in res["type_or_shape_differ"][:10]] + res["only_in_a"][:10]
              + [f"unnamed {u['signature']} {u['type']} {u['shape']} {u['sha256'][:12]}" for u in unmatched[:10]]
              + res["clash"]["a"][:5] + res["clash"]["b"][:5]):
        print("  ", n)
    ok = not (res["differ"] or res["type_or_shape_differ"] or res["only_in_a"] or unmatched or res["clash"]["a"]
              or res["clash"]["b"])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
