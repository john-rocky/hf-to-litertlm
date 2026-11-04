#!/usr/bin/env python3
"""The vision encoder for XNNPACK's fp16 path with fp16 weights in the file: a graph edit of the fp32 file.

  python fp16_fc_weights.py ENCODER.tflite OUT.tflite --report R.json [--no-meta]

Every FULLY_CONNECTED's weight and bias constant becomes FLOAT16 (fp32 -> fp16 round to nearest even, numpy astype), read
through its own DEQUANTIZE (fp16 -> fp32, op version 3) inserted right before the FC: the TFLite converter's float16 form.
CONV_2D (the patch embed) and every other constant stay fp32. The model metadata `reduced_precision_support` =
`fp16accfp16` goes in with the same write: on the CPU path of LiteRT-LM 0.19.0 it is what turns fp16 on (XNNPACK reads it;
the vision executor sets no FORCE_FP16 and does not read the bundle's activation declaration there).

Why fp16 in the file: with fp32 weights XNNPACK's fp16 path converts them into buffers its weight cache has no identifier
for, and 0.19.0 aborts while creating the engine (`Unknown constant buffer passed to BuildPackIdentifier.`, weight_cache.cc
L738) unless the cache is off (cache_dir=":nocache"). A DEQUANTIZE fp16 -> fp32 that feeds an FC only as weight or bias is
skipped by the delegate, which wires the fp16 constant itself into the op, so the packed weights are keyed by a buffer of
the model and the cache runs. With fp16 weights the fp16 path computes with the values XNNPACK's own conversion would make
(both round to nearest even).

A census read back from the written file ends the run: every FC's weight and bias come from a DEQUANTIZE of a FLOAT16
constant, nothing else reads FLOAT16, and the op count is the source's plus one DEQUANTIZE per converted constant; exit 1
otherwise. The new tensors keep the name suffix they have in the published file (.../r7_fp16_dequantize): renaming them
changes the file's bytes, not its computation.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

NAME_SUFFIX = "r7_fp16_dequantize"


def sha256(path):
  h = hashlib.sha256()
  with open(path, "rb") as f:
    for b in iter(lambda: f.read(1 << 24), b""):
      h.update(b)
  return h.hexdigest()


def nm(x):
  return x.decode() if isinstance(x, bytes) else x


def census(path):
  """Read back: per FC, the types of the tensors feeding its weight / bias and whether a DEQUANTIZE makes them."""
  from ai_edge_litert import schema_py_generated as S
  from ai_edge_litert.tools import mmap_utils
  bo = {v: k for k, v in S.BuiltinOperator.__dict__.items() if not k.startswith("_")}
  tt = {v: k for k, v in S.TensorType.__dict__.items() if not k.startswith("_")}
  m = S.Model.GetRootAs(mmap_utils.get_file_contents(path), 0)
  sg = m.Subgraphs(0)
  prod = {}
  ops = []
  for i in range(sg.OperatorsLength()):
    op = sg.Operators(i)
    oc = m.OperatorCodes(op.OpcodeIndex())
    c = bo.get(max(oc.BuiltinCode(), oc.DeprecatedBuiltinCode()))
    ins = [op.Inputs(j) for j in range(op.InputsLength())]
    ops.append((c, ins, oc.Version()))
    for j in range(op.OutputsLength()):
      prod[op.Outputs(j)] = i
  fc = fp16_fed = 0
  other_fp16 = []
  for i, (c, ins, _) in enumerate(ops):
    if c != "FULLY_CONNECTED":
      continue
    fc += 1
    ok = True
    for t in ins[1:3]:
      if t < 0:
        continue
      p = prod.get(t)
      if p is None or ops[p][0] != "DEQUANTIZE" or tt[sg.Tensors(ops[p][1][0]).Type()] != "FLOAT16":
        ok = False
    fp16_fed += ok
  for i, (c, ins, _) in enumerate(ops):
    if c == "DEQUANTIZE":
      continue
    for t in ins:
      if t >= 0 and tt[sg.Tensors(t).Type()] == "FLOAT16":
        other_fp16.append(i)
  meta = [m.Metadata(i).Name().decode() for i in range(m.MetadataLength())]
  return {"ops": len(ops), "fc": fc, "fc_weight_and_bias_fp16_via_dequantize": fp16_fed,
          "non_dequantize_ops_reading_fp16": other_fp16, "metadata": meta}


def main():
  ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  ap.add_argument("src")
  ap.add_argument("out")
  ap.add_argument("--report", required=True)
  ap.add_argument("--no-meta", action="store_true", help="leave out the fp16 metadata (fp16 weights, fp32 compute)")
  a = ap.parse_args()
  from ai_edge_litert import schema_py_generated as S
  from ai_edge_litert.tools import flatbuffer_utils as fu
  m = fu.read_model(a.src)
  sg = m.subgraphs[0]
  bo = {v: k for k, v in S.BuiltinOperator.__dict__.items() if not k.startswith("_")}

  def code(op):
    oc = m.operatorCodes[op.opcodeIndex]
    return bo[max(oc.builtinCode, oc.deprecatedBuiltinCode)]

  uses = {}
  for op in sg.operators:
    for t in op.inputs:
      if t >= 0:
        uses[int(t)] = uses.get(int(t), 0) + 1
  dq = None
  for i, oc in enumerate(m.operatorCodes):
    if max(oc.builtinCode, oc.deprecatedBuiltinCode) == S.BuiltinOperator.DEQUANTIZE and oc.version == 3:
      dq = i
  if dq is None:
    oc = S.OperatorCodeT()
    oc.builtinCode = S.BuiltinOperator.DEQUANTIZE
    oc.deprecatedBuiltinCode = S.BuiltinOperator.DEQUANTIZE
    oc.version = 3
    m.operatorCodes.append(oc)
    dq = len(m.operatorCodes) - 1
  new_ops = []
  rows = []
  n_src_ops = len(sg.operators)
  for op in sg.operators:
    if code(op) == "FULLY_CONNECTED":
      ins = [int(t) for t in op.inputs]
      for j, kind in ((1, "weight"), (2, "bias")):
        if j >= len(ins) or ins[j] < 0:
          continue
        t = ins[j]
        T = sg.tensors[t]
        assert T.type == S.TensorType.FLOAT32 and uses[t] == 1, (nm(T.name), T.type, uses[t])
        buf = m.buffers[T.buffer]
        w = np.frombuffer(bytes(buf.data), dtype=np.float32)
        h = w.astype(np.float16)
        assert np.isfinite(h).all(), f"{nm(T.name)} overflows fp16"
        back = h.astype(np.float32)
        nz = w != 0
        rows.append({"tensor": nm(T.name), "kind": kind, "n": int(w.size), "absmax": float(np.abs(w).max()),
                     "max_rel_err_nonzero": float((np.abs(back[nz] - w[nz]) / np.abs(w[nz])).max()) if nz.any() else 0.0,
                     "to_zero": int(((h == 0) & nz).sum()), "subnormal": int(((np.abs(back) < 2.0 ** -14) & (h != 0)).sum())})
        buf.data = np.frombuffer(h.tobytes(), dtype=np.uint8).copy()
        T.type = S.TensorType.FLOAT16
        m.buffers.append(S.BufferT())
        D = S.TensorT()
        D.shape = np.array(T.shape, dtype=np.int32)
        D.type = S.TensorType.FLOAT32
        D.buffer = len(m.buffers) - 1
        D.name = nm(T.name) + "/" + NAME_SUFFIX
        sg.tensors.append(D)
        dop = S.OperatorT()
        dop.opcodeIndex = dq
        dop.inputs = np.array([t], dtype=np.int32)
        dop.outputs = np.array([len(sg.tensors) - 1], dtype=np.int32)
        new_ops.append(dop)
        ins[j] = len(sg.tensors) - 1
      op.inputs = np.array(ins, dtype=np.int32)
    new_ops.append(op)
  sg.operators = new_ops
  if not a.no_meta:
    names = [nm(md.name) for md in (m.metadata or [])]
    assert "reduced_precision_support" not in names, names
    b = S.BufferT()
    b.data = np.frombuffer(b"fp16accfp16", dtype=np.uint8).copy()
    m.buffers.append(b)
    md = S.MetadataT()
    md.name = "reduced_precision_support"
    md.buffer = len(m.buffers) - 1
    m.metadata = (m.metadata or []) + [md]
  fu.write_model(m, a.out + ".part")
  os.replace(a.out + ".part", a.out)
  cen = census(a.out)
  ok = (cen["fc"] == cen["fc_weight_and_bias_fp16_via_dequantize"] and not cen["non_dequantize_ops_reading_fp16"]
        and cen["ops"] == n_src_ops + len(rows))
  rep = {"src_sha256": sha256(a.src), "bytes": os.path.getsize(a.out), "sha256": sha256(a.out), "src_ops": n_src_ops,
         "dequantize_added": len(rows), "census": cen, "census_ok": ok,
         "max_rel_err_nonzero": max(r["max_rel_err_nonzero"] for r in rows), "to_zero_total": sum(r["to_zero"] for r in rows),
         "subnormal_total": sum(r["subnormal"] for r in rows), "constants": rows}
  json.dump(rep, open(a.report, "w"), indent=1)
  print("FC_WEIGHTS", json.dumps({k: v for k, v in rep.items() if k != "constants"}), flush=True)
  return 0 if ok else 1


if __name__ == "__main__":
  sys.exit(main())
