"""write_codes.py INT4_DIR GPTQ_DIR MAP_JSON - the GPTQ codes of gptq_dory.py written into the int4 TFLite, nothing
else.

INT4_DIR/model_quantized.tflite (quantize_dory.py, data-free int4) is renamed to model_quantized_df.tflite and the new
model_quantized.tflite is a copy of it (an APFS clone where the file system has them) in which, for each of the 289
weights of MAP_JSON (fc_map.py, made from this file), the bytes of the weight's buffer (its int4 codes: rows * cols / 2
bytes, low nibble first) are replaced by GPTQ_DIR/int4/<name>.codes.npy. Sizes must match exactly; nothing is resized
or moved. Checks: GPTQ_DIR/int4/<name>.scales.npy equals the block scales in the file (only the codes change); the
written file read back equals the payloads; a full byte comparison with the data-free file finds differences only
inside the written ranges. Writes INT4_DIR/write.json.

Usage: python dory_work/write_codes.py <int4 dir> <gptq dir> <map json>
"""
import hashlib
import json
import mmap
import os
import shutil
import subprocess
import sys
import time

import numpy as np
from ai_edge_litert import schema_py_generated as ts


def sha(path):
  h = hashlib.sha256()
  with open(path, "rb") as f:
    for b in iter(lambda: f.read(64 << 20), b""):
      h.update(b)
  return h.hexdigest()


def buffer_range(m, buf_idx):
  """The byte range of a buffer's data in the file: inline (a flatbuffer vector) or the offset form (Buffer.offset /
  size from the file start, after the flatbuffer)."""
  buf = m.Buffers(buf_idx)
  if buf.DataLength() > 0:
    off = buf._tab.Vector(buf._tab.Offset(4))  # pylint: disable=protected-access
    return off, off + buf.DataLength()
  if buf.Size() > 0:
    return buf.Offset(), buf.Offset() + buf.Size()
  raise SystemExit(f"buffer {buf_idx} holds no data")


def scale_ranges(m):
  """weight buffer -> (byte range of its block-scale tensor, that tensor's shape), from the first blockwise tensor that
  uses the buffer."""
  out = {}
  for si in range(m.SubgraphsLength()):
    sg = m.Subgraphs(si)
    for ti in range(sg.TensorsLength()):
      t = sg.Tensors(ti)
      if t.Buffer() in out:
        continue
      q = t.Quantization()
      if q is None or q.DetailsType() != ts.QuantizationDetails.BlockwiseQuantization:
        continue
      bq = ts.BlockwiseQuantization()
      tab = q.Details()
      bq.Init(tab.Bytes, tab.Pos)
      st = sg.Tensors(bq.Scales())
      out[t.Buffer()] = (buffer_range(m, st.Buffer()), [int(x) for x in st.ShapeAsNumpy()])
  return out


def main(int4_dir, gptq_dir, map_json):
  t0 = time.time()
  dst = os.path.join(int4_dir, "model_quantized.tflite")
  df = os.path.join(int4_dir, "model_quantized_df.tflite")
  if not os.path.exists(df):
    os.rename(dst, df)
  if os.path.exists(dst):
    os.remove(dst)
  if subprocess.run(["cp", "-c", df, dst]).returncode != 0:  # APFS clone; a plain copy elsewhere
    shutil.copyfile(df, dst)
  tensors = json.load(open(map_json))["tensors"]
  ranges = []
  with open(dst, "r+b") as fh:
    mm = mmap.mmap(fh.fileno(), 0)
    m = ts.Model.GetRootAs(memoryview(mm), 0)
    scales = scale_ranges(m)
    for t in tensors:
      o, i = t["shape"]
      a, b = buffer_range(m, t["int4_buffer"])
      codes = np.load(os.path.join(gptq_dir, "int4", t["name"] + ".codes.npy"))
      assert codes.dtype == np.uint8 and codes.size == b - a == o * i // 2, (t["name"], codes.size, b - a)
      (sa, sb), shp = scales[t["int4_buffer"]]
      sc = np.load(os.path.join(gptq_dir, "int4", t["name"] + ".scales.npy")).astype(np.float16)
      assert shp == [o, i // 32] and sc.nbytes == sb - sa, (t["name"], shp, sc.shape)
      assert bytes(mm[sa:sb]) == sc.tobytes(), f"{t['name']}: GPTQ scales differ from the file's"
      mm[a:b] = codes.tobytes()
      ranges.append((a, b, t["name"]))
    mm.flush()
    for a, b, n in ranges:
      assert mm[a:b] == np.load(os.path.join(gptq_dir, "int4", n + ".codes.npy")).tobytes(), n
    # mm stays open until exit: the flatbuffer objects above still hold views of it (close would raise BufferError)
  a_mm = np.memmap(df, dtype=np.uint8, mode="r")
  b_mm = np.memmap(dst, dtype=np.uint8, mode="r")
  assert a_mm.size == b_mm.size
  ranges.sort()
  starts = np.array([r[0] for r in ranges], dtype=np.int64)
  ends = np.array([r[1] for r in ranges], dtype=np.int64)
  outside, diff_total, step = 0, 0, 1 << 26
  touched = np.zeros(len(ranges), dtype=bool)
  for k in range(0, a_mm.size, step):
    d = np.nonzero(a_mm[k:k + step] != b_mm[k:k + step])[0]
    if d.size == 0:
      continue
    d = d.astype(np.int64) + k
    diff_total += d.size
    idx = np.searchsorted(starts, d, side="right") - 1
    ok = (idx >= 0) & (d < ends[np.clip(idx, 0, None)])
    outside += int((~ok).sum())
    touched[np.unique(idx[ok])] = True
  rec = dict(data_free=df, written=len(ranges), bytes_written=int(sum(r[1] - r[0] for r in ranges)),
             bytes_differing=int(diff_total), bytes_differing_outside_written_ranges=outside,
             ranges_with_a_difference=int(touched.sum()), size=int(b_mm.size), sha256=sha(dst),
             seconds=round(time.time() - t0, 1), ok=outside == 0)
  json.dump(rec, open(os.path.join(int4_dir, "write.json"), "w"), indent=1)
  print(f"WRITE {'OK' if rec['ok'] else 'FAIL'}: {len(ranges)} weights ({rec['bytes_written']} B), differing bytes "
        f"{diff_total}, outside the written ranges {outside}; {dst} sha256 {rec['sha256']}", flush=True)
  return 0 if rec["ok"] else 1


if __name__ == "__main__":
  if len(sys.argv) != 4:
    raise SystemExit(__doc__)
  sys.exit(main(*sys.argv[1:4]))
