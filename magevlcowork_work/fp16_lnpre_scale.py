#!/usr/bin/env python3
"""Keep the vision encoder's first LayerNorm (ln_pre) finite under XNNPACK's fp16 path: a graph edit of the fp32 file.

  python fp16_lnpre_scale.py ENCODER.tflite OUT.tflite --report R.json [--k 3]

ln_pre's input is multiplied by 2^k and its eps by 4^k (default k = 3: x 8, eps 1e-6 x 64 = 6.4e-5, a normal fp16 number).
Why: on a flat screen some rows of ln_pre's variance are fp16 subnormals. The fp16 ADD (variance + eps; eps 1e-6 is a
subnormal too) returns 0 for them, RSQRT gives inf, x * r + (0 - r * mean) gives NaN, and the attention spreads the NaN to
every token of the output. LayerNorm does not change under x -> 2^k x with eps -> 4^k eps, and with a power of two every
fp32 step is exact (the mean, the squared differences, their sum, sqrt(4^k y) = 2^k sqrt(y)), so the fp32 output stays
bit-equal to the input file's.

The edit, on the flatbuffer (ai-edge-litert's flatbuffer_utils), nothing else touched: a MUL by the constant 2^k right after
the producer of ln_pre's input; ln_pre's three readers of that input (MEAN, SQUARED_DIFFERENCE, MUL) read the scaled tensor;
ln_pre's ADD reads a new eps constant (the old one is shared with the other LayerNorms). The three new tensors keep the name
prefix they have in the published file (r7_fix2/...): renaming them changes the file's bytes, not its computation.
"""
import argparse
import hashlib
import json
import os

import numpy as np

NAME_PREFIX = "r7_fix2"


def sha256(path):
  h = hashlib.sha256()
  with open(path, "rb") as f:
    for b in iter(lambda: f.read(1 << 24), b""):
      h.update(b)
  return h.hexdigest()


def main():
  ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  ap.add_argument("src")
  ap.add_argument("out")
  ap.add_argument("--k", type=int, default=3)
  ap.add_argument("--report", required=True)
  a = ap.parse_args()
  from ai_edge_litert import schema_py_generated as S
  from ai_edge_litert.tools import flatbuffer_utils as fu
  m = fu.read_model(a.src)
  sg = m.subgraphs[0]
  bo = {v: k for k, v in S.BuiltinOperator.__dict__.items() if not k.startswith("_")}

  def code(op):
    oc = m.operatorCodes[op.opcodeIndex]
    return bo[max(oc.builtinCode, oc.deprecatedBuiltinCode)]

  # ln_pre = the first MEAN; its input x, the readers of x, the ADD of eps that follows the second MEAN
  i_mean = next(i for i, op in enumerate(sg.operators) if code(op) == "MEAN")
  x = sg.operators[i_mean].inputs[0]
  readers = [(i, j) for i, op in enumerate(sg.operators) for j, t in enumerate(op.inputs) if t == x]
  assert [code(sg.operators[i]) for i, _ in readers] == ["MEAN", "SQUARED_DIFFERENCE", "MUL"], readers
  i_add = next(i for i in range(i_mean + 1, i_mean + 8) if code(sg.operators[i]) == "ADD")
  assert code(sg.operators[i_add + 1]) == "RSQRT", "ln_pre's ADD is not followed by RSQRT"
  eps_t = sg.operators[i_add].inputs[1]
  eps_old = np.frombuffer(bytes(m.buffers[sg.tensors[eps_t].buffer].data), dtype=np.float32).copy()
  assert eps_old.size == 1 and abs(float(eps_old[0]) - 1e-6) < 1e-9, eps_old
  producer = next(i for i, op in enumerate(sg.operators) if x in list(op.outputs))
  scale = np.float32(2.0 ** a.k)
  eps_new = (eps_old * np.float32(4.0 ** a.k)).astype(np.float32)

  eps_shape = np.array(sg.tensors[eps_t].shape if sg.tensors[eps_t].shape is not None else [], dtype=np.int32)

  def const(value, name):   # a float32 constant of the eps tensor's own shape (the graph's scalar form)
    b = S.BufferT()
    b.data = np.frombuffer(np.full(eps_shape, value, dtype=np.float32).tobytes(), dtype=np.uint8).copy()
    m.buffers.append(b)
    t = S.TensorT()
    t.shape = eps_shape.copy()
    t.type = S.TensorType.FLOAT32
    t.buffer = len(m.buffers) - 1
    t.name = name
    sg.tensors.append(t)
    return len(sg.tensors) - 1

  t_scale = const(scale, f"{NAME_PREFIX}/ln_pre_prescale_2^{a.k}")
  t_eps = const(eps_new[0], f"{NAME_PREFIX}/ln_pre_eps_x4^{a.k}")
  tx = sg.tensors[x]
  xb = m.buffers[tx.buffer].data
  assert xb is None or len(xb) == 0, "ln_pre's input is a constant"
  m.buffers.append(S.BufferT())             # an empty buffer = an activation the arena allocates
  xs = S.TensorT()
  xs.shape = np.array(tx.shape, dtype=np.int32)
  if tx.shapeSignature is not None:
    xs.shapeSignature = np.array(tx.shapeSignature, dtype=np.int32)
  xs.type = tx.type
  xs.buffer = len(m.buffers) - 1
  xs.name = f"{NAME_PREFIX}/ln_pre_input_x2^{a.k}"
  sg.tensors.append(xs)
  t_xs = len(sg.tensors) - 1
  mul_code = next(op.opcodeIndex for op in sg.operators if code(op) == "MUL")
  op = S.OperatorT()
  op.opcodeIndex = mul_code
  op.inputs = np.array([x, t_scale], dtype=np.int32)
  op.outputs = np.array([t_xs], dtype=np.int32)
  op.builtinOptionsType = S.BuiltinOptions.MulOptions
  op.builtinOptions = S.MulOptionsT()
  for i, j in readers:
    ins = list(sg.operators[i].inputs)
    ins[j] = t_xs
    sg.operators[i].inputs = np.array(ins, dtype=np.int32)
  ins = list(sg.operators[i_add].inputs)
  ins[1] = t_eps
  sg.operators[i_add].inputs = np.array(ins, dtype=np.int32)
  sg.operators.insert(producer + 1, op)
  fu.write_model(m, a.out + ".part")
  os.replace(a.out + ".part", a.out)
  rep = {"src_sha256": sha256(a.src), "k": a.k, "bytes": os.path.getsize(a.out), "sha256": sha256(a.out),
         "inserted_mul_at": producer + 1, "scaled_input_tensor": int(x), "readers_rewired_src_ops": [i for i, _ in readers],
         "eps_add_src_op": i_add, "eps_old": float(eps_old[0]), "eps_new": float(eps_new[0]),
         "eps_new_is_fp16_normal": bool(float(eps_new[0]) >= 2.0 ** -14)}
  json.dump(rep, open(a.report, "w"), indent=1)
  print("LNPRE_SCALE", json.dumps(rep), flush=True)


if __name__ == "__main__":
  main()
