#!/usr/bin/env python3
"""prefuse_qwen3.py SRC_BF16_DIR OUT.safetensors MANIFEST.json --qkv LAYERS --gate-up LAYERS|boundary

The fused projections of litert-torch 0.10.0's Qwen3 patch, written once as F32 so that export_qwen3.py can map them
instead of making them. With fuse_qkv / fuse_gate_up, litert-torch's model_ext/qwen3/patch.py builds per layer
  qkv_proj.weight     = torch.cat([q_proj.weight, k_proj.weight, v_proj.weight], dim=0)   (FusedQwen3Attention)
  gate_up_proj.weight = torch.cat([gate_proj.weight, up_proj.weight], dim=0)              (FusedQwen3MLP)
as new float32 tensors in anonymous memory: for Qwen3-32B 1.26 GB per layer, about 80 GB over 64 layers (an export of
the 64 layers was stopped at a 36 GB memory footprint, still rising). export_qwen3.py serves gate||up as a view of the
F32 copy where the two tensors are adjacent in the file; q / k / v are never adjacent, and in Qwen/Qwen3-32B's sharding
every shard boundary falls between a layer's gate_proj and up_proj (layers 2, 6, ..., 62). This file holds exactly those
tensors, under the names the patch gives them:
  model.layers.L.self_attn.qkv_proj.weight   for --qkv layers (`all`, `none` or a list such as 0-3,7)
  model.layers.L.mlp.gate_up_proj.weight     for --gate-up layers (`boundary` = the layers whose gate_proj and up_proj
                                             sit in different files of SRC's index; `all`, `none` or a list)
Bytes: each part read from the bf16 checkpoint (pread + F_NOCACHE), BF16 -> F32 by shifting the u16 left 16 bits (the
conversion of qwen35_work/upcast_fp32.py, exact), the parts concatenated along dim 0 (row-major: the bytes of q, then k,
then v). Checks while writing: per tensor, the sha256 of the upper 16 bits of the F32 bytes written == the sha256 of the
concatenated source bytes; the file read back (F_NOCACHE) has the sha256 of everything written. Header = compact JSON
padded with spaces to 8 bytes, dtype F32, offsets contiguous in the order written (layer, then qkv before gate_up). OUT
must not exist. MANIFEST.json: per tensor (name, parts, shape, offsets, sha256 of the source parts and of the F32
bytes), the file's sha256 and bytes, seconds, verdict PASS / FAIL. numpy + stdlib."""
import argparse
import fcntl
import hashlib
import json
import os
import struct
import time

import numpy as np

CHUNK = 64 << 20
F_NOCACHE = getattr(fcntl, "F_NOCACHE", 48)


def nocache(fd):
    fcntl.fcntl(fd, F_NOCACHE, 1)


class Src:
    def __init__(self, d):
        self.d = d
        self.h = {}
        idx = os.path.join(d, "model.safetensors.index.json")
        if os.path.exists(idx):
            self.wm = json.load(open(idx))["weight_map"]
        else:  # one-file checkpoint
            self.wm = {k: "model.safetensors" for k in self.hdr("model.safetensors")[2] if k != "__metadata__"}

    def hdr(self, fn):
        if fn not in self.h:
            p = os.path.join(self.d, fn)
            fd = os.open(p, os.O_RDONLY)
            nocache(fd)
            n = struct.unpack("<Q", os.pread(fd, 8, 0))[0]
            self.h[fn] = (fd, n, json.loads(os.pread(fd, n, 8)))
        return self.h[fn]

    def info(self, key):
        fd, n, h = self.hdr(self.wm[key])
        return h[key]

    def read(self, key):
        fd, n, h = self.hdr(self.wm[key])
        i = h[key]
        if i["dtype"] != "BF16":
            raise SystemExit(f"{key}: dtype {i['dtype']}, expected BF16")
        s, e = i["data_offsets"]
        buf = bytearray(e - s)
        mv, off = memoryview(buf), 0
        while off < len(buf):
            got = os.pread(fd, min(CHUNK, len(buf) - off), 8 + n + s + off)
            if not got:
                raise IOError(f"short read of {key}")
            mv[off:off + len(got)] = got
            off += len(got)
        return bytes(buf), i["shape"]


def layers_arg(s, n_layers, src):
    if s == "none":
        return []
    if s == "all":
        return list(range(n_layers))
    if s == "boundary":
        return [L for L in range(n_layers)
                if src.wm[f"model.layers.{L}.mlp.gate_proj.weight"] != src.wm[f"model.layers.{L}.mlp.up_proj.weight"]]
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("manifest")
    ap.add_argument("--qkv", default="all")
    ap.add_argument("--gate-up", default="boundary")
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"exists: {a.out}")
    src = Src(a.src)
    cfg = json.load(open(os.path.join(a.src, "config.json")))
    n_layers = cfg["num_hidden_layers"]
    qkv = layers_arg(a.qkv, n_layers, src)
    gu = layers_arg(a.gate_up, n_layers, src)
    plan = []
    for L in range(n_layers):
        p = f"model.layers.{L}."
        if L in qkv:
            plan.append((p + "self_attn.qkv_proj.weight", [p + f"self_attn.{x}_proj.weight" for x in ("q", "k", "v")]))
        if L in gu:
            plan.append((p + "mlp.gate_up_proj.weight", [p + f"mlp.{x}_proj.weight" for x in ("gate", "up")]))
    # header
    hdr, off = {}, 0
    for name, parts in plan:
        shapes = [src.info(k)["shape"] for k in parts]
        cols = {s[1] for s in shapes}
        if len(cols) != 1 or any(len(s) != 2 for s in shapes):
            raise SystemExit(f"{name}: parts {parts} shapes {shapes}")
        rows = sum(s[0] for s in shapes)
        nb = rows * shapes[0][1] * 4
        hdr[name] = {"dtype": "F32", "shape": [rows, shapes[0][1]], "data_offsets": [off, off + nb]}
        off += nb
    hdr["__metadata__"] = {"format": "pt", "made_by": "qwen3_32b_work/prefuse_qwen3.py"}
    hj = json.dumps(hdr, separators=(",", ":")).encode()
    hj += b" " * ((8 - len(hj) % 8) % 8)
    rec = {"src": a.src, "out": a.out, "qkv_layers": qkv, "gate_up_layers": gu, "tensors": [], "verdict": None,
           "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    t0 = time.time()
    fd = os.open(a.out + ".partial", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    nocache(fd)
    whole = hashlib.sha256()
    try:
        head = struct.pack("<Q", len(hj)) + hj
        os.write(fd, head)
        whole.update(head)
        ok = True
        for name, parts in plan:
            hs, hf = hashlib.sha256(), hashlib.sha256()
            prec = []
            for k in parts:
                b, shape = src.read(k)
                hs.update(b)
                u = np.frombuffer(b, dtype=np.uint16).astype(np.uint32) << 16
                fb = u.tobytes()
                hf.update(fb)
                hi = (np.frombuffer(fb, dtype=np.uint32) >> 16).astype(np.uint16).tobytes()
                if hi != b:
                    ok = False
                mv, w = memoryview(fb), 0
                while w < len(fb):
                    w += os.write(fd, mv[w:w + CHUNK])
                whole.update(fb)
                prec.append({"key": k, "shape": shape, "bf16_sha256": hashlib.sha256(b).hexdigest()})
                del b, u, fb, hi, mv
            rec["tensors"].append({"name": name, "shape": hdr[name]["shape"], "data_offsets": hdr[name]["data_offsets"],
                                   "parts": prec, "src_concat_sha256": hs.hexdigest(), "f32_sha256": hf.hexdigest()})
            print(f"{name} {hdr[name]['shape']} {time.time() - t0:.0f} s", flush=True)
        os.fsync(fd)
    finally:
        os.close(fd)
    rec["write_s"] = round(time.time() - t0, 1)
    # read back
    h2 = hashlib.sha256()
    fd = os.open(a.out + ".partial", os.O_RDONLY)
    nocache(fd)
    try:
        while True:
            b = os.read(fd, CHUNK)
            if not b:
                break
            h2.update(b)
    finally:
        os.close(fd)
    rec["file_sha256_written"] = whole.hexdigest()
    rec["file_sha256_readback"] = h2.hexdigest()
    rec["bytes"] = os.path.getsize(a.out + ".partial")
    rec["verdict"] = "PASS" if ok and whole.hexdigest() == h2.hexdigest() else "FAIL"
    rec["seconds"] = round(time.time() - t0, 1)
    rec["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    if rec["verdict"] == "PASS":
        os.replace(a.out + ".partial", a.out)
    json.dump(rec, open(a.manifest, "w"), indent=1)
    print("PREFUSE", rec["verdict"], a.out, rec["bytes"], "B", rec["seconds"], "s", flush=True)
    return 0 if rec["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
