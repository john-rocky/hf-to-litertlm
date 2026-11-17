#!/usr/bin/env python3
"""test_fc_int8.py OUT_DIR - checks of fc_int8.py on a tiny graph written here (no download); the files go to OUT_DIR.

The graph: two signatures (a, b), each its own subgraph x [1, 3, 256] -> FULLY_CONNECTED(keep_num_dims) [64 out] with the weight
read through DEQUANTIZE from a FLOAT16 tensor; both FLOAT16 tensors name ONE buffer (as the prefill and decode
signatures of a LiteRT-LM graph share the decoder's weights). Weight rows: random, one row of zeros (the 1e-9 scale
floor), one row with an outlier. Written twice: every buffer inline, and the same model with its constant buffers
external (Buffer.offset / size after the structure, by this file's own writer, not fc_int8's).

Checks (each prints PASS / FAIL; the last line is TEST-END PASS, or TEST-END FAIL and the failed checks; exit 1 on any
FAIL):
  1  fc_int8 --form drq / deq: sites 2, unique weights 1, FLOAT16 buffers emptied 1 (drq: DEQUANTIZE removed 2)
  2  --describe of the outputs: drq = 2 FC weights INT8 const, none read through DEQUANTIZE(FLOAT16); deq = 2 FC
     weights dequant:INT8
  3  deq on the interpreter = x @ (q * scale).T, q and scale computed here by the rule (per row, max |w| / 127,
     rint, clip +-127), rtol 1e-5, both signatures; control: the same comparison with scale x 2 fails
  4  drq on the interpreter: within 5 % (relative L2) of the FLOAT16 source's output, and not equal to deq's output
     (the hybrid path ran, not a float one)
  5  the external variant: its outputs = the inline variant's, for the source and both forms; the outputs keep the
     external layout (offsets > 1 for the int8 buffer)
  6  the int8 outputs are smaller than the source, and the int8 weight is O x I bytes
Needs ai-edge-litert, flatbuffers and numpy.

    python tools/gpu_graph/test_fc_int8.py /tmp/fc_int8_test
"""
import json
import os
import subprocess
import sys

import flatbuffers
import numpy as np
from ai_edge_litert import schema_py_generated as s
from ai_edge_litert.interpreter import Interpreter

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, "fc_int8.py")
# 64 x 256: the weight (32 KiB in float16) outweighs the per-row scales and zero points the int8 tensors add, so check 6
# measures the weight's bytes (at 6 x 16, the 96 bytes the int8 weight saves are fewer than the quantization tables
# add, and the inline file grows)
O, I, N = 64, 256, 3
TT, OP = s.TensorType, s.BuiltinOperator
fails = []


def check(name, ok, detail=""):
  print(("PASS " if ok else "FAIL ") + name + (f" ({detail})" if detail else ""))
  if not ok:
    fails.append(name)


def weights():
  rng = np.random.default_rng(7)
  w = (rng.standard_normal((O, I)) * 0.2).astype(np.float16)
  w[2, :] = 0
  w[4, 5] = np.float16(3.0)
  return w


def build(w):
  m = s.ModelT()
  m.version = 3
  m.buffers = [s.BufferT()]
  b = s.BufferT()
  b.data = np.frombuffer(w.tobytes(), dtype=np.uint8).copy()
  m.buffers.append(b)
  wb = 1
  m.operatorCodes = []
  for op, ver in ((OP.DEQUANTIZE, 3), (OP.FULLY_CONNECTED, 1)):
    c = s.OperatorCodeT()
    c.builtinCode = op
    c.deprecatedBuiltinCode = min(op, 127)
    c.version = ver
    m.operatorCodes.append(c)
  m.subgraphs, m.signatureDefs = [], []
  for k, key in enumerate(("a", "b")):
    sg = s.SubGraphT()
    sg.tensors, sg.operators = [], []

    def tensor(name, shape, ttype, buf=0):
      t = s.TensorT()
      t.name = f"{key}_{name}"
      t.shape = list(shape)
      t.type = ttype
      t.buffer = buf
      sg.tensors.append(t)
      return len(sg.tensors) - 1

    x = tensor("x", [1, N, I], TT.FLOAT32)
    w16 = tensor("w16", [O, I], TT.FLOAT16, wb)
    w32 = tensor("w32", [O, I], TT.FLOAT32)
    y = tensor("y", [1, N, O], TT.FLOAT32)
    dq = s.OperatorT()
    dq.opcodeIndex, dq.inputs, dq.outputs = 0, [w16], [w32]
    fc = s.OperatorT()
    fc.opcodeIndex, fc.inputs, fc.outputs = 1, [x, w32, -1], [y]
    opt = s.FullyConnectedOptionsT()
    opt.keepNumDims = True
    fc.builtinOptionsType, fc.builtinOptions = s.BuiltinOptions.FullyConnectedOptions, opt
    sg.operators = [dq, fc]
    sg.inputs, sg.outputs, sg.name = [x], [y], key
    m.subgraphs.append(sg)
    sd = s.SignatureDefT()
    sd.signatureKey, sd.subgraphIndex = key, k
    ti, to = s.TensorMapT(), s.TensorMapT()
    ti.name, ti.tensorIndex, to.name, to.tensorIndex = "x", x, "y", y
    sd.inputs, sd.outputs = [ti], [to]
    m.signatureDefs.append(sd)
  return m


def pack(m):
  bld = flatbuffers.Builder(1 << 16)
  bld.Finish(m.Pack(bld), file_identifier=b"TFL3")
  return bytes(bld.Output())


def write_inline(m, path):
  with open(path, "wb") as f:
    f.write(pack(m))


def write_external(m, path, align=64, region=16384):
  """Every buffer with data goes after the structure (an independent writer: placeholder offsets, pack, place)."""
  datas = {}
  for i, b in enumerate(m.buffers):
    if b.data is not None and len(b.data):
      datas[i] = bytes(b.data)
      b.data = None
      b.offset, b.size = 2, len(datas[i])
  first = pack(m)
  cur = -(-len(first) // region) * region
  for i in sorted(datas):
    cur = -(-cur // align) * align
    m.buffers[i].offset = cur
    cur += len(datas[i])
  st = pack(m)
  assert len(st) == len(first)
  with open(path, "wb") as f:
    f.write(st)
    pos = len(st)
    for i in sorted(datas):
      f.write(b"\0" * (m.buffers[i].offset - pos))
      f.write(datas[i])
      pos = m.buffers[i].offset + len(datas[i])


def run(path, x):
  it = Interpreter(model_path=path)
  return {k: it.get_signature_runner(k)(x=x)["y"] for k in ("a", "b")}


def tool(*args):
  r = subprocess.run([sys.executable, "-I", "-B", TOOL, *args], capture_output=True, text=True)
  if r.returncode != 0:
    print(r.stdout[-2000:], r.stderr[-2000:])
    raise SystemExit(f"fc_int8 {' '.join(args)}: exit {r.returncode}")
  return json.loads(r.stdout.strip().splitlines()[-1])


def main():
  out = sys.argv[1]
  os.makedirs(out, exist_ok=True)
  w = weights()
  w32 = w.astype(np.float32)
  bound = np.maximum(np.max(np.abs(w32), axis=1), np.float32(1e-9)).astype(np.float32)
  scale = (bound / np.float32(127.0)).astype(np.float32)
  q = np.clip(np.rint(w32 / scale[:, None]), -127, 127).astype(np.int8)
  x = np.random.default_rng(11).standard_normal((1, N, I)).astype(np.float32)
  res = {}
  for lay in ("inline", "external"):
    src = os.path.join(out, f"src_{lay}.tflite")
    for p in [src] + [os.path.join(out, f"{f}_{lay}.tflite") for f in ("drq", "deq")]:
      if os.path.exists(p):
        os.remove(p)
    (write_inline if lay == "inline" else write_external)(build(w), src)
    res[(lay, "src")] = run(src, x)
    for form in ("drq", "deq"):
      dst = os.path.join(out, f"{form}_{lay}.tflite")
      st = tool(src, dst, "--form", form, "--json", os.path.join(out, f"{form}_{lay}.json"))
      ok = st["sites"] == 2 and st["unique_weights"] == 1 and st["fp16_buffers_emptied"] == 1
      if form == "drq":
        ok = ok and st["dequantize_removed"] == 2
      check(f"1 {lay} {form} counts", ok, f"sites {st['sites']} unique {st['unique_weights']} emptied "
            f"{st['fp16_buffers_emptied']} dq_removed {st['dequantize_removed']}")
      d = tool("--describe", dst)
      if form == "drq":
        ok = d["fc_sites_dequant_fp16"] == 0 and d["fc_other"] == {"INT8 const True": 2}
      else:
        ok = d["fc_sites_dequant_fp16"] == 0 and d["fc_other"] == {"dequant:INT8 rank 2 const True": 2}
      check(f"2 {lay} {form} describe", ok, json.dumps(d["fc_other"]))
      if lay == "external":
        check(f"5 {lay} {form} keeps the external layout", d["external_buffers"] >= 1,
              f"external buffers {d['external_buffers']}")
      check(f"6 {lay} {form} smaller", os.path.getsize(dst) < os.path.getsize(src),
            f"{os.path.getsize(src)} -> {os.path.getsize(dst)}")
      check(f"6 {lay} {form} int8 bytes = O x I", st["int8_bytes"] == O * I, f"{st['int8_bytes']}")
      res[(lay, form)] = run(dst, x)
    exp = np.einsum("bni,oi->bno", x, q.astype(np.float32) * scale[:, None])
    bad = np.einsum("bni,oi->bno", x, q.astype(np.float32) * (2 * scale)[:, None])
    for k in ("a", "b"):
      got = res[(lay, "deq")][k]
      check(f"3 {lay} deq {k} = x @ (q*scale).T", np.allclose(got, exp, rtol=1e-5, atol=1e-6),
            f"max abs diff {float(np.max(np.abs(got - exp))):.3g}")
      check(f"3 {lay} deq {k} control (scale x2 must differ)", not np.allclose(got, bad, rtol=1e-5, atol=1e-6))
      ref = res[(lay, "src")][k]
      drq = res[(lay, "drq")][k]
      rel = float(np.linalg.norm(drq - ref) / np.linalg.norm(ref))
      check(f"4 {lay} drq {k} within 5 % of the source", rel < 0.05, f"rel L2 {rel:.4f}")
      check(f"4 {lay} drq {k} differs from deq (hybrid path ran)", not np.array_equal(drq, got),
            f"max abs diff {float(np.max(np.abs(drq - got))):.3g}")
  for name in ("src", "drq", "deq"):
    for k in ("a", "b"):
      check(f"5 external = inline: {name} {k}",
            np.array_equal(res[("external", name)][k], res[("inline", name)][k]))
  print("TEST-END " + ("FAIL " + ", ".join(fails) if fails else "PASS"))
  return 1 if fails else 0


if __name__ == "__main__":
  sys.exit(main())
