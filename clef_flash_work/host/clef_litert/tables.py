"""bfloat16 tensors from safetensors files, read without a tensor framework.

A safetensors file is an 8-byte little-endian header length, a JSON header ({name: {dtype, shape, data_offsets}},
optional "__metadata__") and the data. bfloat16 is the top 16 bits of the float32 with the same value, so a row widens
to float32 exactly by a 16-bit shift (NumPy has no bfloat16: the safetensors 0.8.0 NumPy reader stops on a BF16 tensor
with "TypeError: data type 'bfloat16' not understood").

    Bf16Table(path)              one [rows, dim] BF16 tensor, memory-mapped; `rows(ids)` -> float32 [n, dim]
    load_bf16_tensors(path)      every tensor of a BF16 file -> {name: float32 array} (the joint head, 243 MB)
    copy_bf16_tensor(...)        one tensor's bytes copied into a new one-key file without __metadata__ (the tables)"""
from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import numpy as np


def read_header(path: str | Path) -> tuple[dict, int]:
    """-> (header dict, file offset of the data region)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return header, 8 + n


def widen_bf16(u16: np.ndarray) -> np.ndarray:
    """uint16 bfloat16 bits -> float32 (exact)."""
    return (np.asarray(u16, dtype=np.uint32) << 16).view(np.float32)


class Bf16Table:
    """One BF16 [rows, dim] tensor of a safetensors file, memory-mapped (nothing is read until a row is asked for).
    `key` = the tensor's name (default: the file's only tensor)."""

    def __init__(self, path: str | Path, key: str | None = None):
        self.path = Path(path)
        header, data_start = read_header(self.path)
        names = [k for k in header if k != "__metadata__"]
        if key is None:
            if len(names) != 1:
                raise ValueError(f"{self.path.name} holds {len(names)} tensors; name the key")
            key = names[0]
        e = header[key]
        if e["dtype"] != "BF16" or len(e["shape"]) != 2:
            raise ValueError(f"{key}: expected a 2-D BF16 tensor, got {e['dtype']} {e['shape']}")
        start, end = e["data_offsets"]
        rows, dim = e["shape"]
        if end - start != rows * dim * 2:
            raise ValueError(f"{key}: {end - start} bytes for {rows} x {dim} BF16")
        self.key, self.shape = key, (rows, dim)
        self.u16 = np.memmap(self.path, dtype=np.uint16, mode="r", offset=data_start + start, shape=(rows, dim))
        self.info = {"file": self.path.name, "key": key, "dtype": "BF16", "shape": [rows, dim],
                     "data_offset": data_start + start, "bytes": end - start}

    def rows(self, ids) -> np.ndarray:
        """float32 [len(ids), dim]: the rows of `ids`, widened exactly."""
        return widen_bf16(self.u16[np.asarray(ids, dtype=np.int64)])


def load_bf16_tensors(path: str | Path) -> dict[str, np.ndarray]:
    """Every tensor of a safetensors file whose tensors are all BF16 -> {name: float32 array of its shape}."""
    header, data_start = read_header(path)
    raw = np.memmap(path, dtype=np.uint8, mode="r")
    out = {}
    for name, e in header.items():
        if name == "__metadata__":
            continue
        if e["dtype"] != "BF16":
            raise ValueError(f"{name}: dtype {e['dtype']}, expected BF16")
        start, end = e["data_offsets"]
        n = int(np.prod(e["shape"], dtype=np.int64))
        if end - start != 2 * n:
            raise ValueError(f"{name}: {end - start} bytes for {n} BF16 values")
        u16 = np.frombuffer(raw[data_start + start:data_start + end].tobytes(), dtype="<u2")
        out[name] = widen_bf16(u16).reshape(e["shape"])
    return out


def sha256_file(path: str | Path, block: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_bf16_tensor(src: str | Path, src_key: str, dst: str | Path, dst_key: str, block: int = 1 << 24) -> dict:
    """Write `dst` = a safetensors file with the one tensor `dst_key` whose bytes are `src_key`'s bytes in `src`,
    copied by seek + block reads (the source is never loaded whole), header without __metadata__ (padded with spaces
    to 8 bytes, as the safetensors writer pads). Refuses to overwrite. -> record of what was copied."""
    dst = Path(dst)
    if dst.exists():
        raise FileExistsError(f"refusing to overwrite {dst}")
    header, data_start = read_header(src)
    e = header[src_key]
    if e["dtype"] != "BF16":
        raise ValueError(f"{src_key}: dtype {e['dtype']}")
    start, end = e["data_offsets"]
    n = end - start
    if n != 2 * int(np.prod(e["shape"], dtype=np.int64)):
        raise ValueError(f"{src_key}: {n} bytes for shape {e['shape']}")
    hdr = json.dumps({dst_key: {"dtype": "BF16", "shape": list(e["shape"]), "data_offsets": [0, n]}},
                     separators=(",", ":")).encode()
    hdr += b" " * (-len(hdr) % 8)
    tmp = dst.with_name(dst.name + ".part")
    h_src = hashlib.sha256()
    copied = 0
    with open(src, "rb") as f, open(tmp, "wb") as g:
        g.write(struct.pack("<Q", len(hdr)) + hdr)
        f.seek(data_start + start)
        while copied < n:
            chunk = f.read(min(block, n - copied))
            if not chunk:
                raise IOError(f"{src}: short read at {copied} of {n} bytes")
            h_src.update(chunk)
            g.write(chunk)
            copied += len(chunk)
    tmp.rename(dst)
    return {"source": {"file": Path(src).name, "key": src_key, "dtype": e["dtype"], "shape": list(e["shape"]),
                       "byte_range": [data_start + start, data_start + end], "data_sha256": h_src.hexdigest()},
            "file": dst.name, "key": dst_key, "header_bytes": len(hdr), "data_offset": 8 + len(hdr),
            "bytes": dst.stat().st_size, "bytes_copied": copied}
