#!/usr/bin/env python3
"""st_pread.py — a safetensors reader that reads each tensor with large sequential pread() calls (no mmap).

On a USB hard disk shared with other jobs, the page faults of an mmap reader (safetensors' safe_open / get_slice) got
~3 MB/s while large sequential reads got 16-25 MB/s. Same idea, for the dtypes of
openai/gpt-oss-120b: U8 (MXFP4 blocks and scales, returned as uint8) and BF16 (returned as float32 by a 16-bit shift,
exact). With wait_ok > 0, a shard is opened only once a download checker has left its <shard>.ok marker.

  r = StPread(src_dir); a = r.get(name)            # numpy array, the checkpoint's shape
  python3 tools/st_pread.py SRC NAME [NAME ...]     # self-check: bytes equal to safe_open(...).get_tensor(...)
"""
import json
import os
import struct
import sys
import time

import numpy as np

CHUNK = 64 << 20


class StPread:
    def __init__(self, src, wait_ok=0):
        self.src = src
        idx = os.path.join(src, "model.safetensors.index.json")
        self.wm = json.load(open(idx))["weight_map"] if os.path.exists(idx) else None
        self.meta = {}
        self.fds = {}
        self.wait_ok = wait_ok
        self.bytes_read = 0
        self.seconds = 0.0

    def _fn(self, name):
        return self.wm[name] if self.wm else "model.safetensors"

    def _open(self, fn):
        if fn in self.fds:
            return self.fds[fn]
        okp, w0 = os.path.join(self.src, fn + ".ok"), time.time()
        while self.wait_ok and self.wm and not os.path.exists(okp):
            if time.time() - w0 > self.wait_ok:
                raise SystemExit(f"FAIL: {fn} has no .ok marker after {self.wait_ok} s")
            time.sleep(30)
        fd = os.open(os.path.join(self.src, fn), os.O_RDONLY)
        n = struct.unpack("<Q", os.pread(fd, 8, 0))[0]
        h = json.loads(os.pread(fd, n, 8))
        h.pop("__metadata__", None)
        self.meta[fn] = (8 + n, h)
        self.fds[fn] = fd
        return fd

    def info(self, name):
        fn = self._fn(name)
        self._open(fn)
        base, h = self.meta[fn]
        e = h[name]
        return e["dtype"], e["shape"], base + e["data_offsets"][0], base + e["data_offsets"][1]

    def raw(self, name):
        fd = self._open(self._fn(name))
        dt, shape, a, b = self.info(name)
        t0 = time.time()
        buf = bytearray(b - a)
        mv = memoryview(buf)
        off = 0
        while off < len(buf):
            n = os.preadv(fd, [mv[off:off + CHUNK]], a + off)
            if n <= 0:
                raise IOError(f"short read {name} at {off}")
            off += n
        self.bytes_read += len(buf)
        self.seconds += time.time() - t0
        return dt, shape, buf

    def get(self, name):
        dt, shape, buf = self.raw(name)
        if dt == "U8":
            return np.frombuffer(buf, np.uint8).reshape(shape)
        if dt == "BF16":
            u = np.frombuffer(buf, np.uint16).astype(np.uint32) << 16
            return u.view(np.float32).reshape(shape)
        if dt == "F32":
            return np.frombuffer(buf, np.float32).reshape(shape)
        raise ValueError(f"{name}: dtype {dt}")


def main():
    src, names = sys.argv[1], sys.argv[2:]
    import torch
    from safetensors import safe_open
    r = StPread(src)
    for nm in names:
        a = r.get(nm)
        with safe_open(os.path.join(src, r._fn(nm)), framework="pt") as f:
            t = f.get_tensor(nm)
        b = (t.view(torch.uint8) if t.dtype == torch.uint8 else t.float()).numpy()
        ok = a.shape == b.shape and a.tobytes() == b.tobytes()
        print(f"{'EQUAL' if ok else 'DIFFER'} {nm} {a.dtype} {list(a.shape)}")
    print(f"read {r.bytes_read / 2**20:.1f} MiB in {r.seconds:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
