#!/usr/bin/env python3
"""Replace zero blockwise-int4 scales INSIDE a .litertlm without repacking.

Same defect and same fix as bonsai_work/fix_zero_block_scales.py (all-zero 32-weight
blocks -> ai-edge-quantizer emits scale 0 -> XNNPACK refuses to prepare:
"unsupported scale value (0.000000) ... for INT4 tensor"). That tool repacks through
litert-lm-builder with ONE tflite section and would drop an externalized embedder
section; this one patches the fp16 scale bytes at their absolute offsets inside the
bundle, so every other byte of the file is untouched (size identical, sections
identical, only scale fp16 words change).

--zero-payload also zeroes the int4 payload of every patched block. The quantizer computes
the fp16 scale BEFORE it quantizes, so a block whose scale underflowed to 0 was quantized as
w / 0: weights that are exactly 0 give int 0, but weights that are tiny and not 0 give +-inf,
clipped to saturated +7 / -8 nibbles (bytes 0x77 / 0x78 / 0x87 / 0x88). The zero scale hid
those nibbles at inference; a patched nonzero scale makes them weights again. With
--zero-payload the patched block dequantizes to 0. Measured on LiquidAI/LFM2-2.6B (OCTAV int4
block-32): 746,432 zero scales in 26 scale tensors, 13,397,437 bytes changed with
--zero-payload. Without the flag the tool behaves as it always has (only the scale words
change; the MiniCPM5-2B recipe in convert_minicpm5_2b.sh relies on that).

--require-normal: after patching, exit 1 unless every blockwise scale is a normal fp16.
XNNPACK checks std::isnormal, so a subnormal scale (0 < s < 2^-14) is refused like a zero
one, and this tool patches exact zeros only.

Usage: fix_zero_scales_inplace.py <in.litertlm> <out.litertlm> [--zero-payload] [--require-normal]
Prints per-tensor counts and verifies afterwards that the only differing bytes are the
patched scale words (and, with --zero-payload, the payload of the patched blocks).
"""
import os
import re
import sys
import tempfile

import numpy as np
from ai_edge_litert import schema_py_generated as schema
from litert_lm_builder import peek_litertlm_file

TINY = np.float16(2.0 ** -14)


def tflite_sections(path):
  d = tempfile.mkdtemp(prefix="zsi_")
  with open(os.path.join(d, "peek.txt"), "w") as out:
    peek_litertlm_file(path, d, out)
  peek = open(os.path.join(d, "peek.txt")).read()
  secs = []
  for m in re.finditer(r"Section (\d+):\n  Items:\n(.*?)\n  Begin Offset: (\d+)\n  End Offset:\s+(\d+)\n"
                       r"  Data Type:\s+(\S+)", peek, re.S):
    if m.group(5) == "TFLiteModel":
      secs.append((int(m.group(1)), int(m.group(3)), int(m.group(4)), m.group(2).strip()))
  return secs


def buf_range(model, idx):
  """(offset, size) of buffer idx inside the tflite bytes; size 0 when the buffer is empty."""
  buf = model.Buffers(idx)
  if buf.DataLength():
    # inline vector: absolute position of element 0 inside the tflite bytes
    return buf._tab.Vector(buf._tab.Offset(4)), buf.DataLength()
  return buf.Offset(), buf.Size()


def blockwise_scales(model):
  """Yield (weight tensor, scale buffer index) once per scale buffer."""
  seen = set()
  for s in range(model.SubgraphsLength()):
    sg = model.Subgraphs(s)
    for i in range(sg.TensorsLength()):
      t = sg.Tensors(i)
      q = t.Quantization()
      if q is None or q.DetailsType() != schema.QuantizationDetails.BlockwiseQuantization:
        continue
      bq = schema.BlockwiseQuantization()
      tab = q.Details()
      bq.Init(tab.Bytes, tab.Pos)
      bidx = sg.Tensors(bq.Scales()).Buffer()
      if bidx in seen:
        continue
      seen.add(bidx)
      yield t, bidx


def patch_bytes(data, base, end, zero_payload):
  """Patch zero blockwise scales of the tflite at data[base:end] in place.
  Returns (n_fixed, n_tensors, touched, payload_bytes_zeroed)."""
  model = schema.Model.GetRootAsModel(bytes(memoryview(data)[base:end]), 0)  # parse a copy for offsets
  n_fixed = n_tensors = payload_bytes = 0
  touched = []
  for t, bidx in blockwise_scales(model):
    off, size = buf_range(model, bidx)
    if not size:
      continue
    f16 = np.ndarray(size // 2, dtype=np.float16, buffer=data, offset=base + off)
    zeros = f16 == 0
    if zeros.any():
      nzmin = f16[~zeros].min() if (~zeros).any() else np.float16(1e-4)
      f16[zeros] = nzmin
      if zero_payload:
        # the block's nibbles are contiguous per block (block_size / 2 bytes each)
        woff, wsize = buf_range(model, t.Buffer())
        nb = f16.size
        if wsize and wsize % nb == 0:
          payload = np.ndarray((nb, wsize // nb), dtype=np.uint8, buffer=data, offset=base + woff)
          payload[zeros] = 0
          payload_bytes += int(zeros.sum()) * (wsize // nb)
      n_fixed += int(zeros.sum())
      n_tensors += 1
      touched.append((t.Name().decode(errors="replace"), int(zeros.sum()), int(f16.size), float(nzmin)))
  return n_fixed, n_tensors, touched, payload_bytes


def scale_stats(data, base, end):
  """(blockwise tensors, zero scales, subnormal scales, min scale) over the tflite at data[base:end]."""
  view = memoryview(data)[base:end]
  model = schema.Model.GetRootAsModel(bytes(view), 0)
  n_t = n_zero = n_sub = 0
  gmin = None
  for _, bidx in blockwise_scales(model):
    off, size = buf_range(model, bidx)
    if not size:
      continue
    sc = np.frombuffer(bytes(view[off:off + size]), dtype=np.float16)
    n_t += 1
    n_zero += int((sc == 0).sum())
    n_sub += int(((sc > 0) & (sc < TINY)).sum())
    gmin = float(sc.min()) if gmin is None else min(gmin, float(sc.min()))
  return n_t, n_zero, n_sub, gmin


def main():
  flags = {a for a in sys.argv[1:] if a.startswith("--")}
  args = [a for a in sys.argv[1:] if not a.startswith("--")]
  if len(args) != 2 or flags - {"--zero-payload", "--require-normal"}:
    sys.exit(__doc__)
  src, dst = args
  zero_payload, require_normal = "--zero-payload" in flags, "--require-normal" in flags
  data = bytearray(open(src, "rb").read())
  orig = bytes(data)
  total = payload_total = 0
  secs = tflite_sections(src)
  for sid, b, e, items in secs:
    n_fixed, n_tensors, touched, payload_bytes = patch_bytes(data, b, e, zero_payload)
    print(f"section {sid} [{items}]: patched {n_fixed} zero scales across {n_tensors} scale tensors"
          + (f"; zeroed the payload of the patched blocks ({payload_bytes} bytes)" if zero_payload else ""))
    for name, nz, n, eps in touched[:12]:
      print(f"    {name}: {nz}/{n} zero blocks -> eps {eps:.3g}")
    if len(touched) > 12:
      print(f"    ... {len(touched) - 12} more tensors")
    total += n_fixed
    payload_total += payload_bytes
  diff = np.frombuffer(orig, dtype=np.uint8) != np.frombuffer(bytes(data), dtype=np.uint8)
  bound = 2 * total + payload_total
  print(f"bytes changed: {int(diff.sum())} of {len(data)} (<= 2 per patched scale"
        + (" + the payload of the patched blocks" if zero_payload else "") + f": {bound})")
  assert int(diff.sum()) <= bound, "patched more bytes than scale words" + (" and block payloads" if zero_payload else "")
  normal = True
  for sid, b, e, _ in secs:
    n_t, n_zero, n_sub, gmin = scale_stats(data, b, e)
    ok = n_zero == 0 and n_sub == 0
    normal = normal and ok
    print(f"section {sid} after: blockwise tensors {n_t}; zero scales {n_zero}; subnormal scales {n_sub}; "
          f"min scale {gmin if gmin is None else f'{gmin:.6g}'}; all normal: {ok}")
  if require_normal and not normal:
    sys.exit("a zero or subnormal blockwise scale is left (--require-normal); nothing written")
  open(dst, "wb").write(data)
  print("wrote", dst, os.path.getsize(dst), "B")


if __name__ == "__main__":
  main()
