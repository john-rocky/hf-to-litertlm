"""st_read.py — a safetensors reader with one large read() per tensor (no mmap).

On a USB disk the page faults of an mmap reader (safetensors' get_tensor) got ~3 MB/s while a sequential 16 MB
read got ~16 MB/s (2026-11-09 20:45). This reader parses the header itself and reads each tensor's bytes with one
unbuffered read (the kernel issues large requests), then converts to float32: BF16 by a 16-bit shift (exact), F16 and
F32 by numpy (exact). Same values as safe_open(...).get_tensor(name).float() — checked by st_read_check.py.
Env NH_NOCACHE=1: the reads bypass the page cache (F_NOCACHE).
"""
import json
import os
import struct

import numpy as np

_DT = {"BF16": (np.uint16, 2), "F16": (np.float16, 2), "F32": (np.float32, 4), "I64": (np.int64, 8),
       "I32": (np.int32, 4)}


class StReader:
    def __init__(self, src):
        idx = os.path.join(src, "model.safetensors.index.json")
        if os.path.exists(idx):
            self.wm = json.load(open(idx))["weight_map"]
        else:
            self.wm = None
        self.src = src
        self.meta = {}   # file -> {name: (dtype, shape, start, end)}
        self.fh = {}
        self.read_files = set()

    def _file(self, name):
        return self.wm[name] if self.wm else "model.safetensors"

    def _header(self, fn):
        if fn not in self.meta:
            p = os.path.join(self.src, fn)
            with open(p, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                h = json.loads(f.read(n))
            h.pop("__metadata__", None)
            base = 8 + n
            self.meta[fn] = {k: (v["dtype"], tuple(v["shape"]), base + v["data_offsets"][0],
                                 base + v["data_offsets"][1]) for k, v in h.items()}
        return self.meta[fn]

    def keys(self):
        if self.wm:
            return list(self.wm)
        return list(self._header("model.safetensors"))

    def has(self, name):
        if self.wm:
            return name in self.wm
        return name in self._header("model.safetensors")

    def raw(self, name):
        fn = self._file(name)
        dt, shape, s, e = self._header(fn)[name]
        if fn not in self.fh:
            self.fh[fn] = open(os.path.join(self.src, fn), "rb", buffering=0)
            if os.environ.get("NH_NOCACHE") == "1":
                # macOS F_NOCACHE (48): the reads bypass the page cache (a pass that streams the 63 GB checkpoint
                # leaves the cache to other processes; the values are the same bytes)
                import fcntl
                fcntl.fcntl(self.fh[fn].fileno(), 48, 1)
        f = self.fh[fn]
        buf = bytearray(e - s)
        mv = memoryview(buf)
        f.seek(s)
        got = 0
        while got < len(buf):
            n = f.readinto(mv[got:])
            if not n:
                raise IOError(f"short read {name}: {got} of {len(buf)}")
            got += n
        self.read_files.add(fn)
        return dt, shape, buf

    def get_np(self, name):
        """float32 numpy array."""
        dt, shape, buf = self.raw(name)
        npdt, _ = _DT[dt]
        a = np.frombuffer(buf, dtype=npdt)
        if dt == "BF16":
            a = (a.astype(np.uint32) << 16).view(np.float32)
        elif dt != "F32":
            a = a.astype(np.float32)
        return a.reshape(shape)

    def get_torch(self, name):
        import torch
        return torch.from_numpy(np.ascontiguousarray(self.get_np(name)))
