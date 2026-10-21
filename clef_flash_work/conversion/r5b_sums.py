#!/usr/bin/env python3
"""Round 5b: sha256 of files read again from their real path with F_NOCACHE (macOS fcntl 48: the reads bypass the
unified buffer cache, so an HDD copy is read from the disk, not from pages a compile left in memory).

    python3 scripts/r5b_sums.py --file out/r2/clef_rowprefill_L1024_v2_fp16fc.tflite [--file ...] --out results/r5b_sums_L1024.json
        -> {file: {realpath, bytes, sha256, seconds, f_nocache}}; refuses to overwrite --out"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import time
from pathlib import Path

K = Path(__file__).resolve().parents[1]
F_NOCACHE = getattr(fcntl, "F_NOCACHE", 48)
BLOCK = 8 << 20


def sha256_nocache(path: Path) -> dict:
    real = Path(path).resolve()
    t = time.perf_counter()
    h = hashlib.sha256()
    n = 0
    fd = os.open(real, os.O_RDONLY)
    try:
        nocache = fcntl.fcntl(fd, F_NOCACHE, 1) == 0
        while True:
            chunk = os.read(fd, BLOCK)
            if not chunk:
                break
            h.update(chunk)
            n += len(chunk)
    finally:
        os.close(fd)
    return {"realpath": str(real), "bytes": n, "sha256": h.hexdigest(), "seconds": round(time.perf_counter() - t, 2),
            "f_nocache": nocache}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", action="append", required=True, help="K-relative (a symlink is followed)")
    ap.add_argument("--out", required=True, help="K-relative")
    a = ap.parse_args()
    out = K / a.out
    if out.exists():
        raise SystemExit(f"refusing to overwrite {a.out}")
    doc = {"what": "sha256 re-read with F_NOCACHE", "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "files": {}}
    for f in a.file:
        doc["files"][f] = sha256_nocache(K / f)
        print(json.dumps({f: doc["files"][f]}), flush=True)
    doc["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    tmp = out.with_name(out.name + ".part")
    tmp.write_text(json.dumps(doc, indent=1) + "\n")
    tmp.rename(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
