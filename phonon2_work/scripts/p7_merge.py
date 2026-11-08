#!/usr/bin/env python3
"""The one-file encoders (venv .venv-q, no model execution, cwd = phonon2_work): the encoder files of the buckets -> one .tflite with a subgraph and a signature per
bucket (the 7 fp16ws files -> out/r7/enc_all7_fp16ws.tflite, signatures enc30 enc25 enc20 enc15 enc10 enc7.5 enc5; the 7 int8f files -> out/r7/enc_all7_int8f.tflite),
every constant buffer with the same bytes kept once (the count: merge_count.json). The operator table is the union, the tensors point at the merged buffers, nothing
else in a subgraph changes (r2_fb.write_external writes it); read back after writing: each signature's tensors and ops against its source file (a difference = exit 3).
CLI: p7_merge.py count|merge --out out/r7/enc_all7_fp16ws.tflite --buckets 30,25,20,15,10,7.5,5 --name enc{b}_fp16ws --record-dir results/r7/merge_fp16ws7
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p7_common  # noqa: E402
import r2_fb  # noqa: E402
from ai_edge_litert import schema_py_generated as S  # noqa: E402
from ai_edge_litert.tools import flatbuffer_utils as FU  # noqa: E402
from ai_edge_litert.tools import mmap_utils  # noqa: E402

NAME = "enc{b}_fp16ws"
BO = {v: k for k, v in S.BuiltinOperator.__dict__.items() if not k.startswith("_")}


class Src:
    def __init__(self, bucket):
        self.bucket = bucket
        self.path = p7_common.graph_path(NAME.format(b=bucket))
        self.data = mmap_utils.get_file_contents(self.path)
        self.model = FU.convert_bytearray_to_object(self.data)
        if len(self.model.subgraphs) != 1 or len(self.model.signatureDefs) != 1:
            raise SystemExit(f"{self.path}: {len(self.model.subgraphs)} subgraphs, {len(self.model.signatureDefs)} signatures")

    def view(self, idx):
        """a read-only array over buffer idx's bytes (inline, or Buffer.offset / size in the file)"""
        b = self.model.buffers[idx]
        if b.data is not None and len(b.data) > 0:
            return np.asarray(b.data, dtype=np.uint8)
        if b.size:
            return np.frombuffer(self.data, dtype=np.uint8, count=int(b.size), offset=int(b.offset))
        return None


def sha(a):
    return hashlib.sha256(memoryview(a).cast("B")).hexdigest()


def buffers_used(m):
    """buffer indices the subgraph's tensors name, with the op types that read them"""
    sg = m.subgraphs[0]
    used = {}
    for t in sg.tensors:
        if t.buffer:
            used.setdefault(t.buffer, 0)
            used[t.buffer] += 1
    return used


def count(srcs):
    hashes = {}
    per = {}
    for s in srcs:
        n, b, nz = 0, 0, {}
        for i in range(len(s.model.buffers)):
            v = s.view(i)
            if v is None:
                continue
            h = sha(v)
            hashes.setdefault(h, {"bytes": int(v.size), "in": set()})["in"].add(s.bucket)
            nz[i] = h
            n += 1
            b += int(v.size)
        per[s.bucket] = {"path": s.path, "bytes": os.path.getsize(s.path), "buffers_nonempty": n, "buffer_bytes": b,
                         "unique_in_file": len(set(nz.values())), "tensors_with_buffer": len(buffers_used(s.model)), "index_to_sha": nz}
    buckets = [s.bucket for s in srcs]
    shared_all = [h for h, v in hashes.items() if len(v["in"]) == len(buckets)]
    rep = {"buckets": buckets, "per_file": {b: {k: v for k, v in per[b].items() if k != "index_to_sha"} for b in buckets},
           "distinct_buffers": len(hashes), "distinct_bytes": sum(v["bytes"] for v in hashes.values()),
           "sum_of_files_buffer_bytes": sum(per[b]["buffer_bytes"] for b in buckets),
           "in_all_buckets": len(shared_all), "in_all_buckets_bytes": sum(hashes[h]["bytes"] for h in shared_all),
           "not_in_all": [{"sha": h[:16], "bytes": v["bytes"], "in": sorted(v["in"])} for h, v in sorted(hashes.items(), key=lambda kv: -kv[1]["bytes"]) if len(v["in"]) < len(buckets)][:200]}
    rep["not_in_all_count"] = sum(1 for v in hashes.values() if len(v["in"]) < len(buckets))
    rep["not_in_all_bytes"] = sum(v["bytes"] for v in hashes.values() if len(v["in"]) < len(buckets))
    return rep, hashes


def op_key(oc):
    return (max(oc.builtinCode, oc.deprecatedBuiltinCode), oc.customCode, oc.version)


def merge(srcs, out):
    first = srcs[0].model
    merged = S.ModelT()
    merged.version = first.version
    merged.description = first.description
    merged.operatorCodes = []
    merged.subgraphs = []
    merged.signatureDefs = []
    merged.buffers = [S.BufferT()]
    merged.metadata = []
    code_index = {}
    by_sha = {}

    def merged_buffer(s, idx):
        v = s.view(idx)
        if v is None:
            return 0
        h = sha(v)
        if h not in by_sha:
            b = S.BufferT()
            b.data = v
            merged.buffers.append(b)
            by_sha[h] = len(merged.buffers) - 1
        return by_sha[h]

    for s in srcs:
        m = s.model
        remap_code = []
        for oc in m.operatorCodes:
            k = op_key(oc)
            if k not in code_index:
                merged.operatorCodes.append(oc)
                code_index[k] = len(merged.operatorCodes) - 1
            remap_code.append(code_index[k])
        sg = m.subgraphs[0]
        for t in sg.tensors:
            t.buffer = merged_buffer(s, t.buffer) if t.buffer else 0
        for op in sg.operators:
            op.opcodeIndex = remap_code[op.opcodeIndex]
        sg.name = f"enc{s.bucket}".encode()
        merged.subgraphs.append(sg)
        sd = m.signatureDefs[0]
        sd.signatureKey = f"enc{s.bucket}".encode()
        sd.subgraphIndex = len(merged.subgraphs) - 1
        merged.signatureDefs.append(sd)
    meta_seen = {}
    for md in first.metadata or []:
        # metadata of the first file; its buffers are copied by bytes like the others
        v = srcs[0].view(md.buffer)
        nb = S.MetadataT()
        nb.name = md.name
        nb.buffer = merged_buffer(srcs[0], md.buffer) if v is not None else 0
        merged.metadata.append(nb)
        meta_seen[md.name] = v.size if v is not None else 0
    merged.metadataBuffer = [m.buffer for m in merged.metadata]
    tmp = out + ".tmp"
    n = r2_fb.write_external(merged, tmp)
    os.replace(tmp, out)
    return {"bytes": n, "buffers": len(merged.buffers), "operator_codes": len(merged.operatorCodes), "metadata": {str(k): int(v) for k, v in meta_seen.items()}}


def tensor_table(model, data, sg):
    rows = []
    for t in sg.tensors:
        b = model.buffers[t.buffer] if t.buffer < len(model.buffers) else None
        d = None
        if b is not None:
            if b.data is not None and len(b.data) > 0:
                d = sha(np.asarray(b.data, dtype=np.uint8))
            elif b.size:
                d = sha(np.frombuffer(data, dtype=np.uint8, count=int(b.size), offset=int(b.offset)))
        rows.append((t.name, None if t.shape is None else tuple(int(x) for x in t.shape), t.type, d))
    return rows


def read_back(out, buckets):
    data = mmap_utils.get_file_contents(out)
    model = FU.convert_bytearray_to_object(data)
    res = {"subgraphs": len(model.subgraphs), "signatures": [s.signatureKey.decode() for s in model.signatureDefs], "per": {}}
    bad = []
    for i, b in enumerate(buckets):
        s = Src(b)
        sg_a, sg_b = s.model.subgraphs[0], model.subgraphs[i]
        ta, tb = tensor_table(s.model, s.data, sg_a), tensor_table(model, data, sg_b)
        ops_a = [(max(s.model.operatorCodes[o.opcodeIndex].builtinCode, s.model.operatorCodes[o.opcodeIndex].deprecatedBuiltinCode), tuple(int(x) for x in o.inputs), tuple(int(x) for x in o.outputs)) for o in sg_a.operators]
        ops_b = [(max(model.operatorCodes[o.opcodeIndex].builtinCode, model.operatorCodes[o.opcodeIndex].deprecatedBuiltinCode), tuple(int(x) for x in o.inputs), tuple(int(x) for x in o.outputs)) for o in sg_b.operators]
        sd_a, sd_b = s.model.signatureDefs[0], model.signatureDefs[i]
        io_a = ([(x.name, x.tensorIndex) for x in sd_a.inputs], [(x.name, x.tensorIndex) for x in sd_a.outputs])
        io_b = ([(x.name, x.tensorIndex) for x in sd_b.inputs], [(x.name, x.tensorIndex) for x in sd_b.outputs])
        ok = ta == tb and ops_a == ops_b and io_a == io_b and sd_b.subgraphIndex == i and list(sg_a.inputs) == list(sg_b.inputs) and list(sg_a.outputs) == list(sg_b.outputs)
        res["per"][str(b)] = {"tensors": len(tb), "ops": len(ops_b), "same_as_source": bool(ok), "signature_io": [[(n.decode() if isinstance(n, bytes) else n, t) for n, t in io_b[0]], [(n.decode() if isinstance(n, bytes) else n, t) for n, t in io_b[1]]]}
        if not ok:
            bad.append(b)
    res["bad"] = bad
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["count", "merge"])
    ap.add_argument("--out", default=f"{p7_common.OUT7}/enc_all7_fp16ws.tflite")
    ap.add_argument("--buckets", default="30,25,20,15,10,7.5,5")
    ap.add_argument("--name", default=NAME, help="the source graphs' name pattern (a small test: mel{b})")
    ap.add_argument("--record-dir", default="results/r7/merge_fp16ws7", help="where merge_count.json and merge.json go")
    a = ap.parse_args()
    globals()["NAME"] = a.name
    buckets = [p7_common.num(x) for x in a.buckets.split(",")]
    os.makedirs(a.record_dir, exist_ok=True)
    t0 = time.time()
    srcs = [Src(b) for b in buckets]
    rep, _ = count(srcs)
    rep["time"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    rep["count_s"] = round(time.time() - t0, 1)
    json.dump(rep, open(os.path.join(a.record_dir, "merge_count.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "not_in_all"}), flush=True)
    if a.cmd == "count":
        return 0
    t1 = time.time()
    m = merge(srcs, a.out)
    m["merge_s"] = round(time.time() - t1, 1)
    m["sha256"] = hashlib.sha256(open(a.out, "rb").read()).hexdigest()
    m["census"] = r2_fb.census(a.out)
    del srcs
    m["read_back"] = read_back(a.out, buckets)
    m["path"] = a.out
    m["total_s"] = round(time.time() - t0, 1)
    json.dump(m, open(os.path.join(a.record_dir, "merge.json"), "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in m.items() if k not in ("census",)}, default=str), flush=True)
    return 3 if m["read_back"]["bad"] or m["read_back"]["subgraphs"] != len(buckets) else 0


if __name__ == "__main__":
    sys.exit(main())
