#!/usr/bin/env python3
"""upcast_fp32.py SRC_DIR DST_DIR MANIFEST.json - an F32 copy of a safetensors checkpoint, so that transformers' load
(`_materialize_copy` -> `.to(dtype=float32)`) returns the mmapped tensor itself instead of a new anonymous fp32 copy (for a
30B model that copy is about 110 GB); the converter stays unchanged.
  * every *.safetensors of SRC_DIR (file order, then tensor data order): same tensor names, same shapes, dtype F32; BF16 -> F32 by
    shifting the u16 left 16 bits (exact, reversible), F32 copied as is, any other dtype = stop (exit 1); __metadata__ copied; the
    header is compact JSON padded with spaces to 8 bytes, offsets recomputed in the source's data order;
  * checks while writing: per tensor, sha256 of the source bytes == sha256 of the upper 16 bits of the F32 bytes handed to write()
    (F32 tensors: the bytes themselves); per shard, the sha256 of everything written == the sha256 of the file read back from disk;
  * every other file of SRC_DIR is copied byte for byte (a regular file, the snapshot's symlinks are followed) with its sha256;
    model.safetensors.index.json keeps its weight_map and gets metadata.total_size = the new tensor bytes;
  * I/O: chunks of 64 MiB of source (resident well under 2 GB), F_NOCACHE on every read and write (no page cache), one process;
    run it at background priority (`taskpolicy -b`) on a shared machine. DST_DIR must not exist.
MANIFEST.json: per file (source, sha256 of source, output, sha256 of output, bytes), per tensor (name, dtype in, shape, check), totals,
verdict PASS / FAIL. numpy + stdlib."""
import fcntl
import hashlib
import json
import os
import shutil
import struct
import sys
import time

import numpy as np

CHUNK = 64 << 20
F_NOCACHE = getattr(fcntl, "F_NOCACHE", 48)
ITEM = {"BF16": 2, "F32": 4}


def nocache(fd):
    fcntl.fcntl(fd, F_NOCACHE, 1)


def sha_file(path):
    h = hashlib.sha256()
    fd = os.open(path, os.O_RDONLY)
    nocache(fd)
    try:
        while True:
            b = os.read(fd, CHUNK)
            if not b:
                break
            h.update(b)
    finally:
        os.close(fd)
    return h.hexdigest()


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    return n, hdr


def upcast_shard(src, dst, rec):
    n, hdr = read_header(src)
    meta = hdr.pop("__metadata__", None)
    order = sorted(hdr, key=lambda k: hdr[k]["data_offsets"][0])
    new, off = {}, 0
    sizes = {}
    for k in order:
        t = hdr[k]
        if t["dtype"] not in ITEM:
            raise SystemExit(f"{src}: tensor {k} has dtype {t['dtype']} (only BF16 / F32 handled)")
        nel = 1
        for d in t["shape"]:
            nel *= d
        b0, b1 = t["data_offsets"]
        if b1 - b0 != nel * ITEM[t["dtype"]]:
            raise SystemExit(f"{src}: tensor {k} byte range {b1 - b0} != {nel} x {ITEM[t['dtype']]}")
        sizes[k] = nel * 4
    for k in order:
        new[k] = {"dtype": "F32", "shape": hdr[k]["shape"], "data_offsets": [off, off + sizes[k]]}
        off += sizes[k]
    out_hdr = {}
    if meta is not None:
        out_hdr["__metadata__"] = meta
    for k in hdr:  # keep the source's key order
        out_hdr[k] = new[k]
    hj = json.dumps(out_hdr, separators=(",", ":")).encode()
    hj += b" " * ((-len(hj)) % 8)
    whash = hashlib.sha256()
    sfd = os.open(src, os.O_RDONLY)
    nocache(sfd)
    dfd = os.open(dst, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    nocache(dfd)
    tensors = []
    try:
        head = struct.pack("<Q", len(hj)) + hj
        os.write(dfd, head)
        whash.update(head)
        base = 8 + n
        for k in order:
            t = hdr[k]
            b0, b1 = t["data_offsets"]
            hs, hu = hashlib.sha256(), hashlib.sha256()
            pos = base + b0
            end = base + b1
            while pos < end:
                buf = os.pread(sfd, min(CHUNK, end - pos), pos)
                if not buf:
                    raise SystemExit(f"{src}: short read in {k}")
                pos += len(buf)
                hs.update(buf)
                if t["dtype"] == "BF16":
                    u16 = np.frombuffer(buf, dtype="<u2")
                    out = (u16.astype("<u4") << 16).tobytes()
                    hu.update((np.frombuffer(out, dtype="<u4") >> 16).astype("<u2").tobytes())
                else:
                    out = buf
                    hu.update(out)
                mv = memoryview(out)
                while mv:
                    w = os.write(dfd, mv)
                    mv = mv[w:]
                whash.update(out)
            ok = hs.hexdigest() == hu.hexdigest()
            tensors.append({"name": k, "dtype_in": t["dtype"], "shape": t["shape"], "src_sha256": hs.hexdigest(), "upper16_ok": ok})
            if not ok:
                raise SystemExit(f"{src}: tensor {k}: the upper 16 bits differ from the source bytes")
    finally:
        os.close(sfd)
        os.close(dfd)
    back = sha_file(dst)
    rec["files"].append({"source": src, "output": dst, "kind": "safetensors", "tensors": len(tensors), "bytes": os.path.getsize(dst),
                         "written_sha256": whash.hexdigest(), "readback_sha256": back, "readback_ok": back == whash.hexdigest(),
                         "tensor_bytes": off})
    rec["tensors"].extend(tensors)
    if back != whash.hexdigest():
        raise SystemExit(f"{dst}: read-back sha256 differs from what was written")
    return off


def copy_file(src, dst, rec):
    shutil.copyfile(src, dst)  # follows the snapshot's symlink; a regular file
    s, d = sha_file(src), sha_file(dst)
    rec["files"].append({"source": src, "output": dst, "kind": "copy", "bytes": os.path.getsize(dst), "src_sha256": s,
                         "out_sha256": d, "copy_ok": s == d})
    if s != d:
        raise SystemExit(f"{dst}: copy differs")


def main():
    if len(sys.argv) != 4:
        print(__doc__, file=sys.stderr)
        return 2
    src, dst, man = sys.argv[1:4]
    os.makedirs(dst, exist_ok=False)
    rec = {"src_dir": os.path.abspath(src), "dst_dir": os.path.abspath(dst), "started": time.strftime("%F %T"), "files": [],
           "tensors": []}
    names = sorted(os.listdir(src))
    total = 0
    t0 = time.time()
    for fn in names:
        p = os.path.join(src, fn)
        if os.path.isdir(p) or fn == "model.safetensors.index.json":
            continue
        if fn.endswith(".safetensors"):
            ts = time.time()
            total += upcast_shard(p, os.path.join(dst, fn), rec)
            print(f"{time.strftime('%T')} {fn}: {rec['files'][-1]['tensors']} tensors -> {rec['files'][-1]['bytes']} B in "
                  f"{time.time() - ts:.0f} s (read-back ok {rec['files'][-1]['readback_ok']})", flush=True)
        else:
            copy_file(p, os.path.join(dst, fn), rec)
    ip = os.path.join(src, "model.safetensors.index.json")
    if os.path.exists(ip):
        idx = json.load(open(ip))
        idx.setdefault("metadata", {})["total_size"] = total
        json.dump(idx, open(os.path.join(dst, "model.safetensors.index.json"), "w"), indent=2)
        rec["index"] = {"weight_map_entries": len(idx["weight_map"]), "total_size": total}
    rec["dtypes_in"] = {d: sum(1 for t in rec["tensors"] if t["dtype_in"] == d) for d in ("BF16", "F32")}
    rec["tensor_bytes_out"] = total
    rec["seconds"] = round(time.time() - t0, 1)
    rec["verdict"] = "PASS" if all(t["upper16_ok"] for t in rec["tensors"]) and all(
        f.get("readback_ok", f.get("copy_ok")) for f in rec["files"]) else "FAIL"
    rec["finished"] = time.strftime("%F %T")
    json.dump(rec, open(man, "w"), indent=1)
    print(f"{rec['verdict']}: {len(rec['tensors'])} tensors ({rec['dtypes_in']}), {total} tensor bytes, {rec['seconds']} s -> {dst}")
    return 0 if rec["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
