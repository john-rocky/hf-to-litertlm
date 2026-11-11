#!/usr/bin/env python3
"""int4_attention.py: a published Qwen3-30B-A3B file's attention and lm_head weights as int4, written in place.

  python int4_attention.py quant QDIR
  python int4_attention.py write SRC.litertlm DST.litertlm QDIR [--src-sha256 HEX] [--report R.json]

`quant` range-reads 193 bf16 tensors from the Hub (Qwen/Qwen3-30B-A3B at revision ad44e777:
model.layers.{0..47}.self_attn.{q,k,v,o}_proj.weight and lm_head.weight, 2.4 GB, one request at a time, each checked
against the size in the shard's header) and quantizes each to symmetric int4 in blocks of 32 along the input, one fp16
scale per block: per block the clip ratio 1.0, 0.95, ..., 0.5 of max|w| / 7 with the least squared error (codes -8..7),
then 3 rounds of (codes fixed -> least-squares scale -> re-round), the final scale rounded to fp16 (floor 2^-14, fp16's
smallest normal) and the codes computed from that fp16 scale, so the file's (codes, scales) dequantize to exactly the
values this step measured. QDIR/<name>.npz = packed codes (uint8 [R*C/2], low nibble first) + scales (float16 [R, C/32]);
QDIR/quant.json = the sha256 of each tensor's codes and scales; a tensor whose file exists is skipped, so the step can be
run again.

`write` replaces the int8 per-channel attention and lm_head FCs of SRC (found by name in every subgraph; a weight buffer
is shared by the signatures) in a copy of it (`cp -c`, an APFS clone, on macOS; `cp --reflink=auto` elsewhere):
  * the weight buffer keeps its offset, its size becomes R*C/2 (the packed codes are written there);
  * one new external buffer per FC for the fp16 scales, inside the freed rest of the old int8 region (offset = weight
    offset + R*C/2 rounded up to 64 B, R*C/16 bytes);
  * in each subgraph with the tensor: type INT4, quantization = BlockwiseQuantization{scales = a new FLOAT16 tensor
    [R*C/32] named <name>_scales, zero_points -1, block_size 32}, quantized_dimension 0 (no index of an existing tensor
    changes, so every other blockwise reference stays valid);
  * the new structure with zero padding over [section begin, section begin + data_start) (it must fit: the per-channel
    vectors it drops are larger than what it adds).
A converted FC's old int8 bytes past its new codes and scales stay in the file, unread.
Asserts, read back from disk (DST is renamed into place only when all hold): the same size, header bytes and section
table; every source buffer that is not a converted FC at the same index with the same inline bytes or offset and size;
the weight region of both files equal outside the converted FC regions; the converted FCs read back through DST's own
flatbuffer equal QDIR's codes and scales byte for byte; every blockwise tensor of every signature names a scales tensor
called <its name>_scales.

Needs Python 3.12 with numpy, flatbuffers, ai-edge-litert (the TFLite schema) and litert-lm-builder (the container
header); run with 2.5.4, 25.12.19, 2.3.0 and 0.19.0. `quant` needs numpy only.
"""
import argparse
import hashlib
import importlib.util
import json
import mmap
import os
import sys
import time
import urllib.error
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HUB = "https://huggingface.co/Qwen/Qwen3-30B-A3B/resolve/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39/"
FLOOR = np.float32(2.0 ** -14)
BLOCK = 32
LAYERS = 48
CHUNK = 8192  # rows per chunk in the quant step (every step is per row and block)
ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")


def tensor_names():
  return [f"model.layers.{L}.self_attn.{p}.weight" for L in range(LAYERS) for p in ATTN] + ["lm_head.weight"]


def file_sha256(path, start=0, end=None):
  h = hashlib.sha256()
  end = os.path.getsize(path) if end is None else end
  with open(path, "rb") as f:
    f.seek(start)
    left = end - start
    while left > 0:
      b = f.read(min(1 << 24, left))
      if not b:
        raise IOError(f"short read {path} at {end - left}")
      h.update(b)
      left -= len(b)
  return h.hexdigest()


# ---------------------------------------------------------------------------------------------------------------- quant


def http_range(url, begin, end):
  """The bytes [begin, end] of url; a response of another length is read again (a transfer can end early without an error)."""
  want = end - begin + 1
  got = None
  for _ in range(5):
    try:
      req = urllib.request.Request(url, headers={"Range": f"bytes={begin}-{end}"})
      with urllib.request.urlopen(req, timeout=1800) as r:
        data = r.read()
      if len(data) == want:
        return data
      got = len(data)
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
      got = e
    print(f"again: {url} {begin}-{end}: {got}", flush=True)
    time.sleep(5)
  raise SystemExit(f"could not read {url} {begin}-{end}: {got}")


class Checkpoint:
  """The safetensors shards of the checkpoint, read by HTTP range: the index once, each shard's header once."""

  def __init__(self):
    with urllib.request.urlopen(HUB + "model.safetensors.index.json", timeout=600) as r:
      self.weight_map = json.load(r)["weight_map"]
    self.headers = {}

  def header(self, shard):
    if shard not in self.headers:
      n = int.from_bytes(http_range(HUB + shard, 0, 7), "little")
      self.headers[shard] = (8 + n, json.loads(http_range(HUB + shard, 8, 8 + n - 1)))
    return self.headers[shard]

  def tensor(self, name):
    """-> a little-endian uint16 array [R, C] of the bf16 tensor's own bytes."""
    shard = self.weight_map[name]
    base, head = self.header(shard)
    e = head[name]
    if e["dtype"] != "BF16":
      raise SystemExit(f"{name}: {e['dtype']}")
    a, b = e["data_offsets"]
    R, C = e["shape"]
    if b - a != R * C * 2:
      raise SystemExit(f"{name}: {b - a} B for shape {e['shape']}")
    return np.frombuffer(http_range(HUB + shard, base + a, base + b - 1), dtype="<u2").reshape(R, C)


def to_fp16_scale(s):
  s16 = np.maximum(s.astype(np.float32), FLOOR).astype(np.float16)
  assert np.all(np.isfinite(s16)) and np.all(s16 > 0)
  return s16


def quant_rows(W):
  """float32 [R, C] -> (codes int8 [R, C], scales fp16 [R, C / 32])."""
  R, C = W.shape
  wb = W.reshape(R, C // BLOCK, BLOCK).astype(np.float32)
  bound = np.maximum(np.abs(wb).max(axis=2, keepdims=True), np.float32(1e-9))
  best_err = np.full(bound.shape, np.inf, dtype=np.float32)
  best_scale = np.zeros_like(bound)
  for c in [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]:
    scale = np.maximum(bound * np.float32(c) / np.float32(7.0), FLOOR)
    q = np.clip(np.rint(wb / scale), -8, 7)
    err = ((wb - q * scale) ** 2).sum(axis=2, keepdims=True)
    take = err < best_err
    best_err = np.where(take, err, best_err)
    best_scale = np.where(take, scale, best_scale)
  scale = best_scale
  q = np.clip(np.rint(wb / scale), -8, 7)
  for _ in range(3):
    num = (wb * q).sum(axis=2, keepdims=True)
    den = (q * q).sum(axis=2, keepdims=True)
    scale = np.where(den > 0, num / np.where(den > 0, den, 1), scale)
    scale = np.maximum(scale, FLOOR).astype(np.float32)
    q = np.clip(np.rint(wb / scale), -8, 7)
  s16 = to_fp16_scale(scale[..., 0])
  q = np.clip(np.rint(wb / s16.astype(np.float32)[..., None]), -8, 7).astype(np.int8)
  return q.reshape(R, C), s16


def pack4(q):
  """Two codes per byte, low nibble first."""
  u = q.reshape(-1).view(np.uint8) & np.uint8(0x0F)
  assert u.size % 2 == 0
  return (u[0::2] | (u[1::2] << np.uint8(4))).astype(np.uint8)


def cmd_quant(a):
  os.makedirs(a.qdir, exist_ok=True)
  ck = Checkpoint()
  t0 = time.time()
  qj = os.path.join(a.qdir, "quant.json")
  sums = json.load(open(qj)) if os.path.exists(qj) else {}
  done = 0
  for name in tensor_names():
    out = os.path.join(a.qdir, name + ".npz")
    if os.path.exists(out):
      continue
    u = ck.tensor(name)
    R, C = u.shape
    q = np.empty((R, C), np.int8)
    s16 = np.empty((R, C // BLOCK), np.float16)
    for r0 in range(0, R, CHUNK):
      x = u[r0:r0 + CHUNK].astype(np.uint32)
      x <<= 16
      q[r0:r0 + x.shape[0]], s16[r0:r0 + x.shape[0]] = quant_rows(x.view(np.float32))
    packed = pack4(q)
    np.savez(out + ".tmp.npz", codes=packed, scales=s16, shape=np.array([R, C], np.int64))
    os.replace(out + ".tmp.npz", out)
    sums[name] = {"codes_sha256": hashlib.sha256(packed.tobytes()).hexdigest(),
                  "scales_sha256": hashlib.sha256(s16.tobytes()).hexdigest()}
    with open(qj, "w") as f:
      json.dump(sums, f, indent=1, sort_keys=True)
    done += 1
    print(f"{name}: {R} x {C}", flush=True)
  print(f"quantized {done} tensors in {time.time() - t0:.0f} s; {len(tensor_names()) - done} were there -> {a.qdir}")
  return 0


# ---------------------------------------------------------------------------------------------------------------- write


def load_rewrite():
  spec = importlib.util.spec_from_file_location("gpu_rewrite_public", os.path.join(HERE, "gpu_rewrite.py"))
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


def nm(x):
  return x.decode() if isinstance(x, (bytes, bytearray)) else (x or "")


def complement_sha(path, begin, start, end, holes):
  """sha256 of [start, end) of the section (offsets relative to begin) minus the holes [(a, b), ...] (relative)."""
  h = hashlib.sha256()
  pos = start
  with open(path, "rb") as f:
    for a, b in sorted(holes) + [(end, end)]:
      if a > pos:
        f.seek(begin + pos)
        left = a - pos
        while left > 0:
          chunk = f.read(min(1 << 24, left))
          if not chunk:
            raise IOError(f"short read {path}")
          h.update(chunk)
          left -= len(chunk)
      pos = max(pos, b)
  return h.hexdigest()


def cmd_write(a):
  W = load_rewrite()
  G = W.G
  fb, TT = G.fb, G.TT
  t0 = time.time()
  rep = {"src": a.src, "dst": a.dst, "quant": a.qdir}
  if os.path.exists(a.dst):
    raise SystemExit(f"{a.dst} exists")
  if a.src_sha256:
    got = file_sha256(a.src)
    rep["src_sha256"] = got
    if got != a.src_sha256:
      raise SystemExit(f"source sha256 {got} != {a.src_sha256}")
  src = W.Bundle(a.src)
  model = src.model
  plan = {n: {} for n in tensor_names()}
  for si, sg in enumerate(model.subgraphs):
    for ti, t in enumerate(sg.tensors):
      n = nm(t.name)
      if n in plan:
        plan[n].setdefault("where", []).append((si, ti))
  for n, e in plan.items():
    if not e.get("where"):
      raise SystemExit(f"{n}: not a tensor of {a.src}")
    z = np.load(os.path.join(a.qdir, n + ".npz"))
    shape = [int(x) for x in z["shape"]]
    R, C = shape
    e["codes"], e["scales"] = z["codes"], z["scales"]
    assert e["codes"].dtype == np.uint8 and e["codes"].size == R * C // 2, n
    assert e["scales"].dtype == np.float16 and e["scales"].shape == (R, C // BLOCK), n
    s32 = e["scales"].astype(np.float32)
    assert np.all(np.isfinite(s32)) and np.all(s32 > 0), f"{n}: zero / non-finite scale"
    bufs = set()
    for si, ti in e["where"]:
      t = model.subgraphs[si].tensors[ti]
      if t.type != TT.INT8 or list(t.shape) != shape:
        raise SystemExit(f"{n} in subgraph {si}: type {t.type} shape {list(t.shape)}, expected INT8 {shape}")
      q = t.quantization
      if q is None or q.scale is None or len(q.scale) != R or (q.detailsType or 0) != 0:
        raise SystemExit(f"{n} in subgraph {si}: not per-channel int8")
      bufs.add(int(t.buffer))
    if len(bufs) != 1:
      raise SystemExit(f"{n}: {len(bufs)} buffers across subgraphs")
    bi = bufs.pop()
    b = model.buffers[bi]
    if b.offset is None or b.offset <= 1 or b.size != R * C:
      raise SystemExit(f"{n}: buffer {bi} is not an external {R * C}-byte buffer")
    off_s = b.offset + (-(-(R * C // 2) // 64)) * 64
    if off_s + R * C // BLOCK * 2 > b.offset + b.size:
      raise SystemExit(f"{n}: codes + scales do not fit in the old region")
    e.update({"buffer": bi, "off": int(b.offset), "old_size": int(b.size), "off_s": int(off_s), "R": R, "C": C})
    b.size = R * C // 2
    sb = fb.BufferT()
    sb.offset, sb.size = off_s, R * C // BLOCK * 2
    model.buffers.append(sb)
    e["scales_buffer"] = len(model.buffers) - 1
    for si, ti in e["where"]:
      sg = model.subgraphs[si]
      st = fb.TensorT()
      st.name = n + "_scales"
      st.type = TT.FLOAT16
      st.shape = [R * C // BLOCK]
      st.buffer = e["scales_buffer"]
      sg.tensors.append(st)
      t = sg.tensors[ti]
      t.type = TT.INT4
      qp = fb.QuantizationParametersT()
      qp.quantizedDimension = 0
      d = fb.BlockwiseQuantizationT()
      d.scales = len(sg.tensors) - 1
      d.zeroPoints = -1
      d.blockSize = BLOCK
      qp.detailsType = fb.QuantizationDetails.BlockwiseQuantization
      qp.details = d
      t.quantization = qp
  packed = G.pack_model(model)
  rep["structure_bytes"], rep["structure_sha256"] = len(packed), hashlib.sha256(packed).hexdigest()
  if len(packed) > src.data_start:
    raise SystemExit(f"the new structure ({len(packed)} B) does not fit before the weights ({src.data_start} B)")
  tmp = a.dst + ".tmp"
  if os.path.exists(tmp):
    os.remove(tmp)
  W.clone(a.src, tmp)
  try:
    with open(tmp, "r+b") as f:
      f.seek(src.begin)
      f.write(packed)
      f.write(b"\0" * (src.data_start - len(packed)))
      for n, e in plan.items():
        f.seek(src.begin + e["off"])
        f.write(e["codes"].tobytes())
        f.seek(src.begin + e["off_s"])
        f.write(e["scales"].tobytes())
      f.flush()
      os.fsync(f.fileno())
    chk = {"size_same": os.path.getsize(tmp) == src.size}
    chk["head_same"] = file_sha256(a.src, 0, src.begin) == file_sha256(tmp, 0, src.begin)
    chk["sections_same"] = W.header_sections(tmp) == src.secs
    holes = [(e["off"], e["off"] + e["old_size"]) for e in plan.values()]
    chk["complement_same"] = (complement_sha(a.src, src.begin, src.data_start, src.end - src.begin, holes) ==
                              complement_sha(tmp, src.begin, src.data_start, src.end - src.begin, holes))
    conv_bufs = {e["buffer"] for e in plan.values()}
    with open(tmp, "rb") as f:
      mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
      M = fb.Model.GetRootAsModel(mm, src.begin)
      ds, _ = G.external_region(M)
      chk["data_start_same"] = ds == src.data_start
      ok_b, bad_b = 0, []
      with open(a.src, "rb") as fs:
        mms = mmap.mmap(fs.fileno(), 0, access=mmap.ACCESS_READ)
        S = fb.Model.GetRootAsModel(mms, src.begin)
        for i in range(S.BuffersLength()):
          if i in conv_bufs:
            continue
          x, y = S.Buffers(i), M.Buffers(i)
          if x.Offset() > 1:
            good = y.Offset() == x.Offset() and y.Size() == x.Size()
          else:
            good = x.DataLength() == y.DataLength() and (x.DataLength() == 0 or
                                                         bytes(x.DataAsNumpy()) == bytes(y.DataAsNumpy()))
          ok_b += good
          if not good:
            bad_b.append(i)
        mms.close()
      chk["buffers_same"], chk["buffers_differ"] = ok_b, bad_b[:20]
      rb_ok, rb_bad, bw_ok, bw_bad = 0, [], 0, []
      for si in range(M.SubgraphsLength()):
        sg = M.Subgraphs(si)
        nt = sg.TensorsLength()
        for ti in range(nt):
          t = sg.Tensors(ti)
          q = t.Quantization()
          if q is None or q.DetailsType() != fb.QuantizationDetails.BlockwiseQuantization:
            continue
          d = fb.BlockwiseQuantization()
          d.Init(q.Details().Bytes, q.Details().Pos)
          name = nm(t.Name())
          sc = d.Scales()
          good = 0 <= sc < nt and nm(sg.Tensors(sc).Name()) == name + "_scales" and d.ZeroPoints() == -1
          bw_ok += good
          if not good:
            bw_bad.append((si, name, sc))
          if name in plan:
            e = plan[name]
            st = sg.Tensors(sc)
            wb, sbuf = M.Buffers(t.Buffer()), M.Buffers(st.Buffer())
            fine = (t.Type() == TT.INT4 and list(t.ShapeAsNumpy()) == [e["R"], e["C"]] and st.Type() == TT.FLOAT16
                    and list(st.ShapeAsNumpy()) == [e["R"] * e["C"] // BLOCK] and d.BlockSize() == BLOCK
                    and q.QuantizedDimension() == 0 and q.ScaleLength() == 0 and q.ZeroPointLength() == 0
                    and wb.Offset() == e["off"] and wb.Size() == e["R"] * e["C"] // 2
                    and sbuf.Offset() == e["off_s"] and sbuf.Size() == e["R"] * e["C"] // BLOCK * 2)
            if fine:
              cb = mm[src.begin + wb.Offset(): src.begin + wb.Offset() + wb.Size()]
              sbb = mm[src.begin + sbuf.Offset(): src.begin + sbuf.Offset() + sbuf.Size()]
              fine = cb == e["codes"].tobytes() and sbb == e["scales"].tobytes()
            rb_ok += fine
            if not fine:
              rb_bad.append((si, name))
      mm.close()
    want_rb = sum(len(e["where"]) for e in plan.values())
    chk.update({"readback_ok": rb_ok, "readback_expected": want_rb, "readback_bad": rb_bad[:20],
                "blockwise_named_ok": bw_ok, "blockwise_bad": bw_bad[:20]})
    rep["asserts"] = chk
    bad = [k for k in ("size_same", "head_same", "sections_same", "complement_same", "data_start_same") if not chk[k]]
    if bad_b:
      bad.append("buffers")
    if rb_ok != want_rb or rb_bad:
      bad.append("readback")
    if bw_bad:
      bad.append("blockwise_names")
    if bad:
      raise RuntimeError(f"asserts failed: {bad}")
    rep["dst_sha256"] = file_sha256(tmp)
    os.replace(tmp, a.dst)
  except Exception:
    if os.path.exists(tmp):
      os.remove(tmp)
    raise
  rep["seconds"] = round(time.time() - t0, 1)
  if a.report:
    with open(a.report, "w") as f:
      json.dump(rep, f, indent=1, default=str)
  print(f"wrote {a.dst} sha256 {rep['dst_sha256']}, structure {len(packed)} B of {src.data_start}; {len(plan)} FCs in "
        f"{want_rb} tensor slots; {rep['seconds']} s")
  return 0


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  sub = ap.add_subparsers(dest="cmd", required=True)
  q = sub.add_parser("quant")
  q.add_argument("qdir")
  w = sub.add_parser("write")
  w.add_argument("src")
  w.add_argument("dst")
  w.add_argument("qdir")
  w.add_argument("--src-sha256")
  w.add_argument("--report")
  a = ap.parse_args()
  return cmd_quant(a) if a.cmd == "quant" else cmd_write(a)


if __name__ == "__main__":
  sys.exit(main())
