#!/usr/bin/env python3
"""tables/tables.safetensors: the three bf16 tensors the host reads and the graphs do not hold, their bytes as they are in the source's model.safetensors
(read at the header's offsets). Format = safetensors: an 8-byte little-endian header length, a JSON header padded with spaces to 8-byte alignment, the tensors end to end.
Each tensor's SHA-256 is taken on the source slice and again on the written file. Standard library only, chunks of 16 MiB. Never overwrites.

    python3 conversion/make_tables.py        # from the work directory: reads snapshot/model.safetensors, writes tables/tables.safetensors, results/tables.json"""
import hashlib
import json
import os
import struct
import sys

WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE, REVISION = "openbmb/MiniCPM-OCR-Jev", "a0ed278eb6aa1f333a711665ab610055a4a816ab"
KEYS = {"model.language_model.embed_tokens.weight": [248320, 2048], "model.visual.pos_embed.weight": [2304, 1024], "score.weight": [2, 2048]}
CHUNK = 1 << 24


def slice_sha(path, start, size, out=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        f.seek(start)
        left = size
        while left:
            b = f.read(min(CHUNK, left))
            if not b:
                raise IOError(f"short read at {f.tell()}")
            h.update(b)
            if out is not None:
                out.write(b)
            left -= len(b)
    return h.hexdigest()


def main():
    src, dst = os.path.join(WORK, "snapshot", "model.safetensors"), os.path.join(WORK, "tables", "tables.safetensors")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    assert not os.path.exists(dst), f"never overwrite {dst}"
    with open(src, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        head = json.loads(f.read(n))
    rows, offset, header = [], 0, {"__metadata__": {"source": SOURCE, "revision": REVISION}}
    for key, shape in KEYS.items():
        e = head[key]
        a, b = e["data_offsets"]
        if e["dtype"] != "BF16" or e["shape"] != shape or b - a != shape[0] * shape[1] * 2:
            sys.exit(f"unexpected header entry {key}: {e}")
        header[key] = {"dtype": "BF16", "shape": shape, "data_offsets": [offset, offset + b - a]}
        rows.append({"key": key, "shape": shape, "bytes": b - a, "source_start": 8 + n + a, "offset": offset})
        offset += b - a
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-(8 + len(raw)) % 8)
    tmp = dst + ".tmp"
    with open(tmp, "wb") as out:
        out.write(struct.pack("<Q", len(raw)) + raw)
        for r in rows:
            r["sha256"] = slice_sha(src, r["source_start"], r["bytes"], out)
    for r in rows:
        if slice_sha(tmp, 8 + len(raw) + r["offset"], r["bytes"]) != r["sha256"]:
            sys.exit(f"{r['key']}: the written bytes differ from the source's")
    os.rename(tmp, dst)
    doc = {"source": SOURCE, "revision": REVISION, "file": "tables/tables.safetensors", "bytes": os.path.getsize(dst), "sha256": slice_sha(dst, 0, os.path.getsize(dst)), "tensors": rows}
    os.makedirs(os.path.join(WORK, "results"), exist_ok=True)
    json.dump(doc, open(os.path.join(WORK, "results", "tables.json"), "w"), indent=1)
    print(f"tables/tables.safetensors: {doc['bytes']:,d} bytes, sha256 {doc['sha256']}")
    for r in rows:
        print(f"  {r['key']}: {r['shape']}, {r['bytes']:,d} bytes, sha256 {r['sha256']}")


if __name__ == "__main__":
    main()
