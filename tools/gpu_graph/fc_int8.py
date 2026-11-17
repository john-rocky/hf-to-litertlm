#!/usr/bin/env python3
"""fc_int8.py IN.tflite OUT.tflite --form drq|deq [--json OUT.json] [--dry-run]
   fc_int8.py --describe IN.tflite

Rewrites the FULLY_CONNECTED (FC) weights of a LiteRT-LM prefill/decode graph from float16 to int8. Every FC weight
of IN must be a FLOAT16 constant read through DEQUANTIZE. Every other op, tensor and buffer is left as it is.

Quantization: symmetric, per output channel (axis 0 of the [O, I] weight), the rule ai-edge-quantizer applies to
8-bit weights (0.9.0: tensor_zp_scale_from_min_max, and the narrow range of uniform_quantize). w = the float16 values
as float32, bound[o] = max(max |w[o, :]|, 1e-9), scale[o] = bound[o] / 127, q = clip(rint(w / scale), -127, 127) as
INT8, zero point 0, quantized_dimension 0.

  drq  each FC reads the INT8 tensor directly and the DEQUANTIZE that fed it is removed (int8 weights, float input and
       output: on the CPU, the FC quantizes its input at run time)
  deq  each DEQUANTIZE reads the INT8 tensor instead of the FLOAT16 one; the FC still reads float32 (int8 storage only)

Weights stay shared: one INT8 buffer is written per FLOAT16 buffer, so signatures that read one weight (prefill and
decode) still read one buffer. The FLOAT16 tensors stay in the graph without a buffer (buffer 0), and a FLOAT16 buffer
is emptied once no tensor names it. Operator versions are not changed (ai-edge-quantizer does not change them either).

Layout: a source without external buffers is written with every buffer inline, as gpu_graph_retrofit.py writes one. A
source with external buffers (Buffer.offset/size) gets a new external region: every external buffer the source keeps
(named by a tensor or not), and each INT8 buffer that replaces an external FLOAT16 one, in buffer order, each at a
multiple of the source's own external alignment (the largest power of two up to 16384 that divides every source
offset), the region starting at a multiple of 16384 after the structure.

Output: one JSON line on stdout (sites, unique weights, bytes, the quantization error, the layout, both files' sha256),
also written to --json. --dry-run quantizes and prints the line without writing OUT. --describe IN prints one JSON line
about IN instead: its FC weights by kind, the FC and DEQUANTIZE operator versions, the FC options, the external
buffers and their alignment, the signatures and the op counts.

Refusals (exit 1 with a message, nothing written): OUT exists; an FC weight that is not DEQUANTIZE(FLOAT16 constant) of
rank 2; no FC weight read through DEQUANTIZE(FLOAT16); a DEQUANTIZE output that something other than FCs reads (drq).

Example run: PaddleOCR-VL-1.6's prefill/decode graph (725,540,096 bytes) reads all 127 of its FC weights this way, at
490 FC sites; --form drq wrote it as 375,345,152 bytes.
"""
import argparse
import collections
import hashlib
import importlib.util
import json
import os
import sys
import time

import numpy as np

# the TFLite reader and writer of gpu_graph_retrofit.py, loaded by path from this file's directory (sys.path is not
# changed; the file is only read)
_spec = importlib.util.spec_from_file_location(
    "gpu_graph_retrofit", os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpu_graph_retrofit.py"))
ggr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ggr)
fb, OP, TT = ggr.fb, ggr.OP, ggr.TT
TTN = {v: k for k, v in vars(TT).items() if not k.startswith("_")}
REGION_ALIGN = 16384


def builtin(model, op):
  c = model.operatorCodes[op.opcodeIndex]
  return max(c.builtinCode, c.deprecatedBuiltinCode)


def sha256_file(path):
  h = hashlib.sha256()
  with open(path, "rb") as f:
    while True:
      b = f.read(1 << 24)
      if not b:
        return h.hexdigest()
      h.update(b)


def scan(model):
  """-> (sites, other_fc): sites = [(si, fc_oi, dq_oi, src_tensor)], FC weights read as DEQUANTIZE(FLOAT16 const);
  other_fc = Counter of the other FC weight kinds (refused when not empty)."""
  sites, other = [], collections.Counter()
  for si, sg in enumerate(model.subgraphs):
    prod = {}
    for oi, op in enumerate(sg.operators or []):
      for t in op.outputs:
        prod[t] = oi
    for oi, op in enumerate(sg.operators or []):
      if builtin(model, op) != OP.FULLY_CONNECTED:
        continue
      w = op.inputs[1]
      if w in prod and builtin(model, sg.operators[prod[w]]) == OP.DEQUANTIZE:
        dq = prod[w]
        src = sg.operators[dq].inputs[0]
        st = sg.tensors[src]
        if st.type == TT.FLOAT16 and ggr.is_const(model, sg, src) and len(st.shape) == 2:
          sites.append((si, oi, dq, src))
          continue
        other[f"dequant:{TTN.get(st.type, st.type)} rank {len(st.shape)} const {ggr.is_const(model, sg, src)}"] += 1
      else:
        t = sg.tensors[w]
        other[f"{TTN.get(t.type, t.type)} const {ggr.is_const(model, sg, w)}"] += 1
  return sites, other


def quant_rows(w32):
  bound = np.maximum(np.max(np.abs(w32), axis=1), np.float32(1e-9)).astype(np.float32)
  scale = (bound / np.float32(127.0)).astype(np.float32)
  q = np.clip(np.rint(w32 / scale[:, None]), -127, 127).astype(np.int8)
  return q, scale


def tensor_refs(model):
  """-> Counter of buffer index -> number of tensors (all subgraphs) that name it."""
  c = collections.Counter()
  for sg in model.subgraphs:
    for t in sg.tensors:
      c[t.buffer] += 1
  return c


def describe(path):
  model, data_start, size = ggr.load_tflite(path)
  sites, other = scan(model)
  ops = collections.Counter()
  for sg in model.subgraphs:
    for op in sg.operators or []:
      ops[ggr.BN.get(builtin(model, op), builtin(model, op))] += 1
  ext = [b for b in model.buffers if b.offset is not None and b.offset > 1]
  align = ext_alignment(ext)
  versions = {}
  for c in model.operatorCodes:
    b = max(c.builtinCode, c.deprecatedBuiltinCode)
    if b in (OP.FULLY_CONNECTED, OP.DEQUANTIZE):
      versions.setdefault(ggr.BN[b], []).append(c.version)
  uniq = {model.subgraphs[si].tensors[src].buffer for si, _, _, src in sites}
  fcq = collections.Counter()
  for sg in model.subgraphs:
    for op in sg.operators or []:
      if builtin(model, op) == OP.FULLY_CONNECTED:
        o = op.builtinOptions
        fcq[f"asym {getattr(o, 'asymmetricQuantizeInputs', None)} keep {getattr(o, 'keepNumDims', None)} "
            f"fmt {getattr(o, 'weightsFormat', None)}"] += 1
  return {"path": path, "bytes": size, "subgraphs": len(model.subgraphs),
          "signatures": [s.signatureKey.decode() if isinstance(s.signatureKey, bytes) else s.signatureKey
                         for s in (model.signatureDefs or [])],
          "fc_sites_dequant_fp16": len(sites), "fc_unique_fp16_buffers": len(uniq),
          "fc_other": dict(other), "op_versions": versions, "fc_options": dict(fcq),
          "external_buffers": len(ext), "data_start": data_start, "external_alignment": align,
          "ops": dict(ops.most_common(40))}


def ext_alignment(ext):
  if not ext:
    return None
  a = REGION_ALIGN
  while a > 1 and any(b.offset % a for b in ext):
    a //= 2
  return a


def rewrite(model, src_path, form):
  sites, other = scan(model)
  if other:
    raise SystemExit(f"fc_int8: FC weights of another kind: {dict(other)}")
  if not sites:
    raise SystemExit("fc_int8: no FC weight read through DEQUANTIZE(FLOAT16)")
  new = {}  # FLOAT16 buffer -> (INT8 buffer, scale)
  err = {"max_abs_err_over_scale": 0.0, "sum_sq_err": 0.0, "sum_sq_w": 0.0, "elements": 0, "clipped": 0}
  ext_new = set()
  t0 = time.time()
  for si, oi, dq, src in sites:
    st = model.subgraphs[si].tensors[src]
    b16 = st.buffer
    if b16 in new:
      continue
    raw = ggr.buffer_bytes(model, b16, src_path)
    w32 = np.frombuffer(raw, dtype=np.float16).reshape(list(st.shape)).astype(np.float32)
    q, scale = quant_rows(w32)
    d = w32 - q.astype(np.float32) * scale[:, None]
    err["max_abs_err_over_scale"] = max(err["max_abs_err_over_scale"], float(np.max(np.abs(d) / scale[:, None])))
    err["sum_sq_err"] += float(np.sum(d.astype(np.float64) ** 2))
    err["sum_sq_w"] += float(np.sum(w32.astype(np.float64) ** 2))
    err["elements"] += int(w32.size)
    err["clipped"] += int(np.count_nonzero(np.abs(np.rint(w32 / scale[:, None])) > 127))
    nb = fb.BufferT()
    nb.data = q.reshape(-1).view(np.uint8).copy()
    model.buffers.append(nb)
    bi = len(model.buffers) - 1
    old = model.buffers[b16]
    if old.offset is not None and old.offset > 1:
      ext_new.add(bi)
    new[b16] = (bi, scale)
  quant_s = time.time() - t0
  remove = collections.defaultdict(set)
  for si, oi, dq, src in sites:
    sg = model.subgraphs[si]
    st = sg.tensors[src]
    bi, scale = new[st.buffer]
    t = fb.TensorT()
    t.name = (st.name.decode() if isinstance(st.name, bytes) else st.name) + "_int8"
    t.type = TT.INT8
    t.shape = [int(x) for x in st.shape]
    t.shapeSignature = [int(x) for x in st.shapeSignature] if st.shapeSignature is not None else None
    t.buffer = bi
    t.hasRank = st.hasRank
    qp = fb.QuantizationParametersT()
    qp.scale = np.asarray(scale, dtype=np.float32)
    qp.zeroPoint = np.zeros(len(scale), dtype=np.int64)
    qp.quantizedDimension = 0
    t.quantization = qp
    sg.tensors.append(t)
    ti = len(sg.tensors) - 1
    if form == "drq":
      sg.operators[oi].inputs[1] = ti
      remove[si].add(dq)
    else:
      sg.operators[dq].inputs[0] = ti
  if form == "drq":
    for si, dqs in remove.items():
      sg = model.subgraphs[si]
      for dq in dqs:
        out = sg.operators[dq].outputs[0]
        readers = [k for k, op in enumerate(sg.operators) if out in op.inputs]
        if any(builtin(model, sg.operators[k]) != OP.FULLY_CONNECTED for k in readers) or out in sg.outputs:
          raise SystemExit(f"fc_int8: subgraph {si}: DEQUANTIZE {dq}'s output is read by more than FCs")
      sg.operators = [op for k, op in enumerate(sg.operators) if k not in dqs]
  # the FLOAT16 tensors no op reads any more give up their buffer; a buffer no tensor names any more is emptied
  emptied = 0
  for si, sg in enumerate(model.subgraphs):
    read = set()
    for op in sg.operators or []:
      read.update(op.inputs)
    for k, t in enumerate(sg.tensors):
      if t.buffer in new and k not in read:
        t.buffer = 0
  refs = tensor_refs(model)
  for b16 in new:
    if refs[b16] == 0:
      b = model.buffers[b16]
      b.data = None
      b.offset = 0
      b.size = 0
      emptied += 1
  stats = {"sites": len(sites), "unique_weights": len(new), "fp16_buffers_emptied": emptied,
           "dequantize_removed": sum(len(v) for v in remove.values()), "quant_s": round(quant_s, 1),
           "int8_bytes": int(sum(model.buffers[bi].data.size for bi, _ in new.values())),
           "elements": err["elements"], "clipped": err["clipped"],
           "max_abs_err_over_scale": round(err["max_abs_err_over_scale"], 6),
           "rel_rms_err": float(np.sqrt(err["sum_sq_err"] / max(err["sum_sq_w"], 1e-30)))}
  return stats, ext_new


def write_relayout(model, src_path, ext_new, dst_path):
  """New external region for every external buffer the source still has (named by a tensor or not: an offset left
  out would point into the old layout) and the new INT8 buffers in ext_new, in buffer order."""
  src_ext = {i: (b.offset, b.size) for i, b in enumerate(model.buffers)
             if b.offset is not None and b.offset > 1}
  align = ext_alignment([model.buffers[i] for i in src_ext]) or 64
  order = sorted(set(src_ext) | set(ext_new))
  data = {}
  for i in order:
    b = model.buffers[i]
    if i in ext_new:
      data[i] = ("mem", b.data)
      b.data = None
      b.size = int(data[i][1].size)
    else:
      data[i] = ("src",) + src_ext[i]
    b.offset = 2  # placeholder > 1: the structure's size does not depend on the value
  first = ggr.pack_model(model)
  start = -(-len(first) // REGION_ALIGN) * REGION_ALIGN
  cur = start
  for i in order:
    cur = -(-cur // align) * align
    model.buffers[i].offset = cur
    cur += model.buffers[i].size
  struct = ggr.pack_model(model)
  if len(struct) != len(first):
    raise RuntimeError(f"structure size moved with the offsets: {len(first)} -> {len(struct)}")
  with open(dst_path, "wb") as out, open(src_path, "rb") as src:
    out.write(struct)
    pos = len(struct)
    for i in order:
      b = model.buffers[i]
      out.write(b"\0" * (b.offset - pos))
      if data[i][0] == "mem":
        out.write(data[i][1].tobytes())
      else:
        _, off, n = data[i]
        src.seek(off)
        left = n
        while left > 0:
          chunk = src.read(min(1 << 26, left))
          out.write(chunk)
          left -= len(chunk)
      pos = b.offset + b.size
  return {"external": len(order), "alignment": align, "structure_bytes": len(struct), "region_start": start,
          "file_bytes": os.path.getsize(dst_path)}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("src")
  ap.add_argument("dst", nargs="?")
  ap.add_argument("--form", choices=["drq", "deq"])
  ap.add_argument("--json")
  ap.add_argument("--dry-run", action="store_true")
  ap.add_argument("--describe", action="store_true")
  a = ap.parse_args()
  if a.describe:
    print(json.dumps(describe(a.src)))
    return 0
  if not a.dst or not a.form:
    ap.error("IN OUT --form are required")
  if os.path.exists(a.dst):
    raise SystemExit(f"refusing to overwrite {a.dst}")
  t0 = time.time()
  model, data_start, size = ggr.load_tflite(a.src)
  stats, ext_new = rewrite(model, a.src, a.form)
  rec = {"src": a.src, "src_bytes": size, "src_sha256": sha256_file(a.src), "form": a.form, **stats,
         "source_external": data_start is not None}
  if not a.dry_run:
    if data_start is None:
      with open(a.dst, "wb") as f:
        f.write(ggr.pack_model(model))
      rec["layout"] = {"external": 0}
    else:
      rec["layout"] = write_relayout(model, a.src, ext_new, a.dst)
    rec["dst"] = a.dst
    rec["dst_bytes"] = os.path.getsize(a.dst)
    rec["dst_sha256"] = sha256_file(a.dst)
  rec["total_s"] = round(time.time() - t0, 1)
  line = json.dumps(rec)
  print(line)
  if a.json:
    with open(a.json, "w") as f:
      f.write(line + "\n")
  return 0


if __name__ == "__main__":
  sys.exit(main())
