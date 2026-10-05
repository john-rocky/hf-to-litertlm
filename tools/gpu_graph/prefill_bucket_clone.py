#!/usr/bin/env python3
"""Add prefill buckets to a shipped LiteRT-LM bundle without re-exporting — the weights stay.

An existing prefill signature (e.g. `prefill_128`) is copied at new sequence lengths T'. The copy shares every weight
buffer (same buffer index), so the file grows by graph structure only. What depends on the sequence length T is found
by propagating a symbolic T from the signature inputs through the ops — never by replacing every number equal to T
(SmolLM3-3B has head dim 128 = T, granite-4.2 has head dim 64):

  seeds      `embeddings` [1, T, D] / `tokens` [1, T] (dim 1), `input_pos` [T] or [1, T] (last dim), `mask` [1, 1, T, C]
             (dim 2). Every other signature input (KV caches, param_tensor) is T-free.
  a tensor   holds T in at most one dim, as  dim = o * T * i  (o = outer and i = inner multiplicity inside that dim,
             e.g. the GQA logits [1, bk, g*T, C] have o = g, i = 1), so a RESHAPE can be followed through the row-major
             layout: T keeps its stride S = i * prod(dims after it) and its outer count M = prod(dims before it) * o.
  ops        elementwise (broadcast), RESHAPE (the output dim k with prod(out[:k]) * o' = M and i' * prod(out[k+1:]) = S;
             exactly one k must fit, else the tool stops), TRANSPOSE, CONCATENATION (not along T), SLICE (a full-length
             slice of the T dim keeps T; a fixed slice of it, such as input_pos[0:1], does not), PACK, FULLY_CONNECTED,
             MEAN, BATCH_MATMUL, DYNAMIC_UPDATE_SLICE, STABLEHLO_COMPOSITE (T is propagated into the decomposition
             subgraph, which is copied per bucket). Any other op stops the tool. At every op the rule must reproduce the
             stored output shape at the source length T0 (a self-consistency check of the inference).
  rewrites   tensor shapes (o * T' * i), the RESHAPE shape constants and the SLICE size constants that carry T (new
             small inline buffers), tensor names `prefill_<T0>_*` -> `prefill_<T'>_*`, the subgraph name, one SignatureDef
             per bucket (same input / output names and tensor indices as the source, param_tensor included). The new
             subgraphs (bucket first, then its decompositions) are appended after the existing ones, so every existing subgraph and
             SignatureDef keeps its index and its bytes; the new SignatureDefs are inserted among the prefill ones in
             descending length (the exporter's order). Weight buffers: shared. Composite attributes: copied (T-free).

The embedder section needs no per-length signature: LiteRT-LM 0.17.1 looks embeddings up one token at a time with the
embedder's first signature (it must take 4 bytes = 1 token) and pads the rest of the prefill buffer with token 0's
embedding (runtime/components/embedding_lookup/embedding_lookup_text.cc: Initialize, LookupPrefill); granite-4.2's
shipped bundle runs prefill_256/64/16/4/1 with only `prefill_embedder_1024` + `decode_embedder`.

  prefill_bucket_clone.py build SRC.litertlm DST.litertlm --lengths 16,64 [--source prefill_128] --report R.json
      unpack -> clone inside the TF_LITE_PREFILL_DECODE section -> pack (gpu_graph_retrofit.retrofit_bundle with its
      tflite step swapped; the other sections are asserted byte-identical, the external weight region is copied
      verbatim and gpu_graph_retrofit.buffer_identity must hold; every existing subgraph must pack to the same bytes).
  prefill_bucket_clone.py selftest BUNDLE.litertlm --source prefill_64 --against prefill_16,prefill_256 --report R.json
      clone the source signature (in memory) at the lengths of real signatures of the same file and compare op by op:
      op codes, options, composite name / attrs / version, every tensor (shape, shape signature, dtype, quantization,
      constant bytes or the shared buffer index), subgraph and SignatureDef IO, and every decomposition subgraph.
  prefill_bucket_clone.py tflite SRC.tflite DST.tflite --lengths ... [--source ...] --report R.json
"""
import argparse
import collections
import copy
import hashlib
import json
import os
import re
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpu_graph_retrofit as R  # noqa: E402  load_tflite / write_tflite / buffer_identity / bundle_sections / retrofit_bundle
import shape_variant as SV  # noqa: E402  sg_bytes / composite_ref / inspect / sha256_file
from ai_edge_litert import schema_py_generated as fb  # noqa: E402

OP = fb.BuiltinOperator
TT = fb.TensorType
BN = R.BN
name_of = R.name_of

UNARY = {OP.CAST, OP.COS, OP.SIN, OP.LOGISTIC, OP.RSQRT, OP.SOFTMAX, OP.NEG, OP.EXP, OP.TANH, OP.SQUARE, OP.ABS,
         OP.GELU, OP.RELU, OP.SQRT, OP.LOG, OP.HARD_SWISH, OP.LOGICAL_NOT, OP.DEQUANTIZE, OP.FLOOR, OP.CEIL, OP.ROUND}
BINARY = {OP.ADD, OP.MUL, OP.SUB, OP.DIV, OP.MAXIMUM, OP.MINIMUM, OP.POW, OP.SQUARED_DIFFERENCE, OP.EQUAL, OP.NOT_EQUAL,
          OP.LESS, OP.LESS_EQUAL, OP.GREATER, OP.GREATER_EQUAL, OP.SELECT_V2, OP.LOGICAL_AND, OP.LOGICAL_OR}
NP_OF = {TT.INT32: np.int32, TT.INT64: np.int64}


class CloneError(RuntimeError):
  pass


def shp(sg, t):
  return [int(x) for x in sg.tensors[t].shape]


def prod(xs):
  p = 1
  for x in xs:
    p *= int(x)
  return p


def at_len(shape, st, T):
  """shape with the T dim of structure st = (dim, o, i) evaluated at length T."""
  if st is None:
    return list(shape)
  j, o, i = st
  s = list(shape)
  s[j] = o * T * i
  return s


def referenced(sg):
  """tensors an op, the subgraph IO or a weight's blockwise quantization (scales / zero points) refers to."""
  used = set(sg.inputs) | set(sg.outputs)
  for op in sg.operators:
    used.update(t for t in op.inputs if t >= 0)
    used.update(op.outputs)
  for x in sg.tensors:
    q = x.quantization
    if q is not None and q.details is not None and isinstance(q.details, fb.BlockwiseQuantizationT):
      used.update(t for t in (q.details.scales, q.details.zeroPoints) if t is not None and t >= 0)
  return used


class Model:
  """One flatbuffer model (ModelT) with the per-op helpers the propagation needs."""

  def __init__(self, model, src_path=None):
    self.m = model
    self.src_path = src_path
    self.code = [max(c.builtinCode, c.deprecatedBuiltinCode) for c in model.operatorCodes]

  def opcode(self, op):
    return self.code[op.opcodeIndex]

  def opname(self, op):
    b = self.opcode(op)
    if b == OP.STABLEHLO_COMPOSITE:
      return "COMPOSITE:" + name_of(op.builtinOptions2.name)
    return BN.get(b, str(b))

  def const(self, sg, t):
    """numpy value of an INT32 / INT64 constant tensor, else None."""
    x = sg.tensors[t]
    if x.type not in NP_OF:
      return None
    if x.shape is not None and len(x.shape) and prod(x.shape) == 0:
      return np.zeros(0, dtype=NP_OF[x.type])  # zero-element constant (an empty shape: RESHAPE to a scalar)
    b = self.m.buffers[x.buffer]
    if b.data is not None and len(b.data):
      raw = bytes(b.data)
    elif b.offset is not None and b.offset > 1 and self.src_path:
      with open(self.src_path, "rb") as f:
        f.seek(b.offset)
        raw = f.read(b.size)
    else:
      return None
    return np.frombuffer(raw, dtype=NP_OF[x.type]).copy()


# ---------------------------------------------------------------------------------------------------------------------
# symbolic T propagation
# ---------------------------------------------------------------------------------------------------------------------

class Propagation:
  """T structure of every tensor of one subgraph (and, recursively, of the decompositions its composites call).

  st[t] = None (T-free) or (dim, o, i). rewrites[(op index, input slot)] = (kind, function T -> new constant values).
  """

  def __init__(self, M, si, seeds, T0, label, log):
    self.M, self.si, self.T0, self.label, self.log = M, si, T0, label, log
    self.sg = M.m.subgraphs[si]
    self.st = dict(seeds)
    self.rewrites = {}  # (op index, input slot) -> {"kind", "old", "new_of"(T -> list)}
    self.children = {}  # op index -> Propagation of the decomposition subgraph
    self.notes = collections.Counter()
    self.ambiguous = []  # RESHAPE whose constant holds T0 at a position that is not the T dim (naive replace would err)
    self.fixed_windows = []  # SLICE of a fixed window [0, size) of the T dim
    self.run()
    used = referenced(self.sg)
    self.dead = [t for t in range(len(self.sg.tensors)) if t not in used]
    # dead tensors (no op / IO / quantization refers to them) are copied unchanged; listed when a dim is a multiple of T0
    self.dead_with_T0 = [{"tensor": t, "name": name_of(self.sg.tensors[t].name)[-90:], "shape": shp(self.sg, t)}
                         for t in self.dead if any(v and v % T0 == 0 for v in shp(self.sg, t))]
    for t in used:
      if t not in self.st:
        self.st[t] = None  # weight scales reached only through quantization details: T-free

  def fail(self, oi, msg):
    op = self.sg.operators[oi]
    raise CloneError(f"{self.label} op {oi} {self.M.opname(op)}: {msg} "
                     f"(inputs {[(t, shp(self.sg, t) if t >= 0 else None, self.st.get(t)) for t in op.inputs]}, "
                     f"outputs {[(t, shp(self.sg, t)) for t in op.outputs]})")

  def check(self, oi, t, st, expect_shape=None):
    """st must reproduce the stored shape of t at T0 (and the op rule's shape when given)."""
    stored = shp(self.sg, t)
    if st is not None:
      j, o, i = st
      if j >= len(stored) or o * self.T0 * i != stored[j]:
        self.fail(oi, f"T structure {st} does not give the stored shape {stored} at T0={self.T0}")
    if expect_shape is not None and list(expect_shape) != stored:
      self.fail(oi, f"rule gives shape {expect_shape}, stored {stored}")
    if t in self.st and self.st[t] != st:
      self.fail(oi, f"tensor {t} already has T structure {self.st[t]}, now {st}")
    self.st[t] = st

  def run(self):
    sg = self.sg
    for oi, op in enumerate(sg.operators):
      code = self.M.opcode(op)
      ins = [t for t in op.inputs]
      sts = [self.st.get(t) if t >= 0 else None for t in ins]
      for t in ins:
        if t >= 0 and t not in self.st:
          self.st[t] = None  # constants and T-free inputs
      if code in UNARY:
        self.check(oi, op.outputs[0], sts[0], shp(sg, ins[0]))
        self.notes[BN[code]] += 1
      elif code in BINARY:
        self.binary(oi, op, ins, sts)
      elif code == OP.RESHAPE:
        self.reshape(oi, op, ins, sts)
      elif code == OP.TRANSPOSE:
        perm = self.M.const(sg, ins[1])
        if perm is None or sts[1] is not None:
          self.fail(oi, "TRANSPOSE perm is not a constant")
        perm = [int(x) for x in perm]
        out = [shp(sg, ins[0])[p] for p in perm]
        st = None
        if sts[0] is not None:
          j, o, i = sts[0]
          st = (perm.index(j), o, i)
        self.check(oi, op.outputs[0], st, out)
        self.notes["TRANSPOSE"] += 1
      elif code == OP.CONCATENATION:
        self.concat(oi, op, ins, sts)
      elif code == OP.SLICE:
        self.slice(oi, op, ins, sts)
      elif code == OP.PACK:
        if any(s is not None for s in sts):
          a = op.builtinOptions.axis
          if any(s != sts[0] for s in sts):
            self.fail(oi, "PACK of inputs with different T structures")
          j, o, i = sts[0]
          rank = len(shp(sg, ins[0]))
          a = a if a >= 0 else a + rank + 1
          st = (j if j < a else j + 1, o, i)
        else:
          st = None
        self.check(oi, op.outputs[0], st)
        self.notes["PACK"] += 1
      elif code == OP.FULLY_CONNECTED:
        self.fully_connected(oi, op, ins, sts)
      elif code in (OP.MEAN, OP.SUM, OP.REDUCE_MAX, OP.REDUCE_MIN, OP.REDUCE_PROD):
        self.reduce(oi, op, ins, sts)
      elif code == OP.BATCH_MATMUL:
        self.batch_matmul(oi, op, ins, sts)
      elif code == OP.EMBEDDING_LOOKUP:
        # ids [T] (or [.., T]) x table [V, D] -> [.., T, D]: the T structure of the ids carries over unchanged
        # (the table is T-free); the output gains the trailing embedding dim.
        if sts[1] is not None:
          self.fail(oi, "EMBEDDING_LOOKUP table depends on T")
        ids_shape = shp(sg, ins[0])
        table = shp(sg, ins[1])
        self.check(oi, op.outputs[0], sts[0], list(ids_shape) + list(table[1:]))
        self.notes["EMBEDDING_LOOKUP"] += 1
      elif code == OP.DYNAMIC_UPDATE_SLICE:
        if sts[2] is not None:
          self.fail(oi, "DYNAMIC_UPDATE_SLICE indices depend on T")
        self.check(oi, op.outputs[0], sts[0], shp(sg, ins[0]))
        self.notes["DYNAMIC_UPDATE_SLICE"] += 1
      elif code == OP.STABLEHLO_COMPOSITE:
        self.composite(oi, op, ins, sts)
      else:
        self.fail(oi, "op type not handled by the T propagation")

  def binary(self, oi, op, ins, sts):
    sg = self.sg
    shapes = [shp(sg, t) for t in ins]
    rank = max(len(s) for s in shapes)
    out = []
    st = None
    for d in range(rank):  # right-aligned broadcast
      entries = []  # (size, (o, i) when this operand carries T in this dim)
      for s, x in zip(shapes, sts):
        k = d - (rank - len(s))
        if k < 0:
          continue
        entries.append((s[k], (x[1], x[2]) if (x is not None and x[0] == k) else None))
      v = max(sz for sz, _ in entries)
      if any(sz not in (1, v) for sz, _ in entries):
        self.fail(oi, f"broadcast of {shapes} at dim {d}")
      tst = [x for _, x in entries if x is not None]
      if tst:
        if any(x != tst[0] for x in tst):
          self.fail(oi, "inputs carry T with different structures in one dim")
        if any(x is None and sz != 1 for sz, x in entries):
          self.fail(oi, f"a T-free operand has size != 1 in dim {d} where another carries T (a T-shaped constant?)")
        o, i = tst[0]
        if st is not None:
          self.fail(oi, "T in two dims")
        st = (d, o, i)
      out.append(v)
    self.check(oi, op.outputs[0], st, out)
    self.notes[BN[self.M.opcode(op)]] += 1

  def reshape(self, oi, op, ins, sts):
    sg = self.sg
    out_t = op.outputs[0]
    out = shp(sg, out_t)
    if len(ins) < 2 or ins[1] < 0:
      self.fail(oi, "RESHAPE without a shape tensor (builtin new_shape only) is not handled")
    cval = self.M.const(sg, ins[1])
    if cval is None or sts[1] is not None:
      self.fail(oi, "RESHAPE shape is not a constant")
    if op.builtinOptions is not None and getattr(op.builtinOptions, "newShape", None) is not None and len(op.builtinOptions.newShape):
      self.fail(oi, "RESHAPE carries builtin new_shape (not handled)")
    cl = [int(x) for x in cval]
    if len(cl) != len(out) or any(c not in (-1, v) for c, v in zip(cl, out)):
      self.fail(oi, f"RESHAPE constant {cl} does not match the stored output shape {out}")
    if sts[0] is None:
      if prod(shp(sg, ins[0])) != prod(out):
        self.fail(oi, "element count")
      self.check(oi, out_t, None)
      self.notes["RESHAPE_T_free"] += 1
      return
    j, o, i = sts[0]
    ins0 = shp(sg, ins[0])
    Min = prod(ins0[:j]) * o
    Sin = i * prod(ins0[j + 1:])
    cands = []
    for k in range(len(out)):
      pre, suf = prod(out[:k]), prod(out[k + 1:])
      if Min % pre or Sin % suf:
        continue
      o2, i2 = Min // pre, Sin // suf
      if o2 >= 1 and i2 >= 1 and o2 * self.T0 * i2 == out[k]:
        cands.append((k, o2, i2))
    if len(cands) != 1:
      self.fail(oi, f"RESHAPE: {len(cands)} row-major placements of T ({cands}) for M={Min}, S={Sin}")
    k, o2, i2 = cands[0]
    self.check(oi, out_t, (k, o2, i2))
    others = [p for p, v in enumerate(cl) if v == self.T0 and p != k]
    if others or (self.T0 in ins0 and ins0.index(self.T0) != j):
      self.ambiguous.append({"op": oi, "in_shape": ins0, "const": cl, "T_dim": k, "o": o2, "i": i2,
                             "T0_at_other_positions": others})
    if cl[k] == -1:
      self.notes["RESHAPE_T_inferred_dim"] += 1
      return

    def new_of(T, cl=cl, k=k, o2=o2, i2=i2):
      v = list(cl)
      v[k] = o2 * T * i2
      return v
    self.rewrites[(oi, 1)] = {"kind": "RESHAPE.shape", "old": cl, "new_of": new_of}
    self.notes["RESHAPE_T"] += 1

  def concat(self, oi, op, ins, sts):
    sg = self.sg
    rank = len(shp(sg, ins[0]))
    a = op.builtinOptions.axis
    a = a if a >= 0 else a + rank
    carriers = [s for s in sts if s is not None]
    st = None
    if carriers:
      if any(s[0] == a for s in carriers):
        self.fail(oi, "CONCATENATION along the T dim is not handled")
      if len(carriers) != len(sts) or any(s != carriers[0] for s in carriers):
        self.fail(oi, "CONCATENATION inputs disagree on T")
      st = carriers[0]
    out = list(shp(sg, ins[0]))
    out[a] = sum(shp(sg, t)[a] for t in ins)
    self.check(oi, op.outputs[0], st, out)
    self.notes["CONCATENATION"] += 1

  def slice(self, oi, op, ins, sts):
    sg = self.sg
    begin, size = self.M.const(sg, ins[1]), self.M.const(sg, ins[2])
    if begin is None or size is None or sts[1] is not None or sts[2] is not None:
      self.fail(oi, "SLICE begin / size are not constants")
    begin, size = [int(x) for x in begin], [int(x) for x in size]
    in0 = shp(sg, ins[0])
    out = [(in0[d] - begin[d]) if size[d] == -1 else size[d] for d in range(len(in0))]
    st = None
    if sts[0] is not None:
      j, o, i = sts[0]
      if begin[j] == 0 and (size[j] == -1 or size[j] == in0[j]):
        st = sts[0]
        if size[j] != -1:
          def new_of(T, size=size, j=j, o=o, i=i):
            v = list(size)
            v[j] = o * T * i
            return v
          self.rewrites[(oi, 2)] = {"kind": "SLICE.size", "old": size, "new_of": new_of}
        self.notes["SLICE_T_full"] += 1
      elif size[j] != -1 and begin[j] + size[j] <= in0[j]:
        # a fixed window of the T dim (e.g. input_pos[0:1]): T-free output; only a window at the start is accepted,
        # and every bucket must be at least as long as the window (checked when cloning)
        if begin[j] != 0:
          self.fail(oi, f"SLICE of the T dim at begin {begin[j]} (a window that may move with T)")
        self.fixed_windows.append({"op": oi, "dim": j, "o": o, "i": i, "size": size[j]})
        self.notes["SLICE_T_fixed_window"] += 1
      else:
        self.fail(oi, f"SLICE of the T dim with begin {begin} size {size}")
    else:
      self.notes["SLICE_T_free"] += 1
    self.check(oi, op.outputs[0], st, out)

  def fully_connected(self, oi, op, ins, sts):
    sg = self.sg
    in0 = shp(sg, ins[0])
    w = shp(sg, ins[1])
    if sts[1] is not None or (len(ins) > 2 and ins[2] >= 0 and sts[2] is not None):
      self.fail(oi, "FULLY_CONNECTED weights / bias depend on T")
    keep = bool(op.builtinOptions.keepNumDims)
    st = None
    if keep:
      out = in0[:-1] + [w[0]]
      if sts[0] is not None:
        if sts[0][0] == len(in0) - 1:
          self.fail(oi, "FULLY_CONNECTED contracts over the T dim")
        st = sts[0]
    else:
      out = [prod(in0[:-1]), w[0]]
      if sts[0] is not None:
        j, o, i = sts[0]
        if j == len(in0) - 1:
          self.fail(oi, "FULLY_CONNECTED contracts over the T dim")
        st = (0, prod(in0[:j]) * o, i * prod(in0[j + 1:-1]))
    self.check(oi, op.outputs[0], st, out)
    self.notes["FULLY_CONNECTED"] += 1

  def reduce(self, oi, op, ins, sts):
    sg = self.sg
    axes = self.M.const(sg, ins[1])
    if axes is None:
      self.fail(oi, "reduction axes are not a constant")
    in0 = shp(sg, ins[0])
    axes = sorted({int(a) % len(in0) for a in axes})
    keep = bool(op.builtinOptions.keepDims)
    st = None
    if sts[0] is not None:
      j, o, i = sts[0]
      if j in axes:
        self.fail(oi, "reduction over the T dim")
      st = (j if keep else j - sum(1 for a in axes if a < j), o, i)
    out = [(1 if d in axes else v) for d, v in enumerate(in0)] if keep else [v for d, v in enumerate(in0) if d not in axes]
    self.check(oi, op.outputs[0], st, out)
    self.notes[BN[self.M.opcode(op)]] += 1

  def batch_matmul(self, oi, op, ins, sts):
    sg = self.sg
    a, b = shp(sg, ins[0]), shp(sg, ins[1])
    o_ = op.builtinOptions
    adjx, adjy = bool(o_.adjX), bool(o_.adjY)
    ra, ca = (a[-1], a[-2]) if adjx else (a[-2], a[-1])
    rb, cb = (b[-1], b[-2]) if adjy else (b[-2], b[-1])
    if ca != rb:
      self.fail(oi, "BATCH_MATMUL contraction sizes")
    rank = max(len(a), len(b))
    ba, bb = [1] * (rank - len(a)) + a[:-2], [1] * (rank - len(b)) + b[:-2]
    batch = [max(x, y) for x, y in zip(ba, bb)]
    out = batch + [ra, cb]
    st = None
    for side, s in ((0, sts[0]), (1, sts[1])):
      if s is not None and s[0] < len((a, b)[side]) - 2:
        pos = s[0] + (rank - len((a, b)[side]))
        other = (bb, ba)[side][pos]
        if (sts[1 - side] is None or sts[1 - side][0] + (rank - len((a, b)[1 - side])) != pos) and other != 1:
          self.fail(oi, "BATCH_MATMUL: T in a batch dim against a T-free size != 1")
    for side, s, shape_, adj in ((0, sts[0], a, adjx), (1, sts[1], b, adjy)):
      if s is None:
        continue
      j, o, i = s
      n = len(shape_)
      if j < n - 2:  # batch dim
        pos = j + (rank - n)
      elif side == 0:
        if (j == n - 2) != (not adjx):
          self.fail(oi, "BATCH_MATMUL contracts over T (lhs)")
        pos = rank - 2
      else:
        if (j == n - 1) != (not adjy):
          self.fail(oi, "BATCH_MATMUL contracts over T (rhs)")
        pos = rank - 1
      if st is not None and st != (pos, o, i):
        self.fail(oi, "BATCH_MATMUL: T in both operands at different places")
      st = (pos, o, i)
    self.check(oi, op.outputs[0], st, out)
    self.notes["BATCH_MATMUL"] += 1

  def composite(self, oi, op, ins, sts):
    di = op.builtinOptions2.decompositionSubgraphIndex
    dsg = self.M.m.subgraphs[di]
    if len(dsg.inputs) != len(ins) or len(dsg.outputs) != len(op.outputs):
      self.fail(oi, "composite arity differs from its decomposition subgraph")
    seeds = {}
    for t_main, t_dec, s in zip(ins, dsg.inputs, sts):
      if t_main >= 0 and shp(self.sg, t_main) != shp(dsg, t_dec):
        self.fail(oi, f"composite input shape {shp(self.sg, t_main)} vs decomposition {shp(dsg, t_dec)}")
      seeds[t_dec] = s
    child = Propagation(self.M, di, seeds, self.T0, f"{self.label}/{name_of(dsg.name)}", self.log)
    self.children[oi] = child
    for t_main, t_dec in zip(op.outputs, dsg.outputs):
      self.check(oi, t_main, child.st.get(t_dec), shp(dsg, t_dec))
    self.notes["COMPOSITE:" + name_of(op.builtinOptions2.name)] += 1

  def summary(self):
    tot = collections.Counter(self.notes)
    rw = collections.Counter(v["kind"] for v in self.rewrites.values())
    amb = list(self.ambiguous)
    fw = list(self.fixed_windows)
    for ch in self.children.values():
      s = ch.summary()
      tot.update(s["ops"])
      rw.update(s["rewrites"])
      amb.extend(s["ambiguous"])
      fw.extend(s["fixed_windows"])
    dead = [dict(d, subgraph=self.label) for d in self.dead_with_T0]
    for ch in self.children.values():
      dead.extend(ch.summary()["dead_with_T0"])
    t_tensors = sum(1 for v in self.st.values() if v is not None)
    return {"ops": dict(tot), "rewrites": dict(rw), "ambiguous": amb, "fixed_windows": fw, "t_tensors_main": t_tensors,
            "dead_tensors_main": len(self.dead), "dead_with_T0": dead}


def seeds_for(sg, sd, T0):
  """T structure of the signature inputs (the runtime's prefill contract)."""
  seeds, found = {}, {}
  for tm in sd.inputs:
    n, t = name_of(tm.name), tm.tensorIndex
    s = shp(sg, t)
    if n in ("embeddings", "tokens", "per_layer_embeddings"):
      if len(s) < 2 or s[1] != T0:
        raise CloneError(f"input {n} {s}: dim 1 is not T0={T0}")
      seeds[t] = (1, 1, 1)
    elif n in ("input_pos", "positions", "input_positions"):
      if s[-1] != T0:
        raise CloneError(f"input {n} {s}: last dim is not T0={T0}")
      seeds[t] = (len(s) - 1, 1, 1)
    elif n in ("mask", "mask_local", "attn_mask"):
      if len(s) != 4 or s[2] != T0:
        raise CloneError(f"input {n} {s}: dim 2 is not T0={T0}")
      seeds[t] = (2, 1, 1)
    else:
      seeds[t] = None
    found[n] = s
  return seeds, found


def prefill_len(sg, sd):
  for tm in sd.inputs:
    if name_of(tm.name) in ("input_pos", "positions", "input_positions"):
      return shp(sg, tm.tensorIndex)[-1]
  raise CloneError("no input_pos")


# ---------------------------------------------------------------------------------------------------------------------
# cloning
# ---------------------------------------------------------------------------------------------------------------------

class Cloner:
  def __init__(self, M):
    self.M = M
    self.new_buffers = {}  # (dtype, bytes) -> buffer index created by this tool
    self.counters = collections.Counter()
    # next free decomposition number per composite name (names look like `odml.rms_norm.impl_<n>`)
    self.next_impl = collections.Counter()
    for sg in M.m.subgraphs:
      mt = re.match(r"^(.*\.impl)(?:_(\d+))?$", name_of(sg.name))
      if mt:
        n = int(mt.group(2)) + 1 if mt.group(2) is not None else 0
        self.next_impl[mt.group(1)] = max(self.next_impl[mt.group(1)], n)

  def buffer_for(self, arr):
    key = (arr.dtype.str, arr.tobytes())
    if key not in self.new_buffers:
      b = fb.BufferT()
      b.data = np.frombuffer(arr.tobytes(), dtype=np.uint8).copy()
      self.M.m.buffers.append(b)
      self.new_buffers[key] = len(self.M.m.buffers) - 1
    return self.new_buffers[key]

  def clone_sg(self, prop, T, rename):
    """Copy of prop's subgraph at length T (appended to the model); returns (new index, report)."""
    M = self.M
    src = prop.sg
    sg = copy.deepcopy(src)
    rep = collections.Counter()
    for t, x in enumerate(sg.tensors):
      st = prop.st.get(t)
      if st is not None:
        x.shape = at_len([int(v) for v in x.shape], st, T)
        if x.shapeSignature is not None and len(x.shapeSignature):
          sig = [int(v) for v in x.shapeSignature]
          if -1 in sig:
            raise CloneError(f"{prop.label}: tensor {t} has a dynamic shape signature {sig}")
          x.shapeSignature = at_len(sig, st, T)
        rep["tensors_T"] += 1
      x.name = rename(name_of(x.name))
    # constant rewrites: one new tensor per (op, slot) unless every user of the constant agrees on the new value
    users = collections.defaultdict(list)
    for (oi, slot), rw in prop.rewrites.items():
      users[src.operators[oi].inputs[slot]].append((oi, slot, rw))
    for t, lst in users.items():
      allusers = [(oi, k) for oi, op in enumerate(src.operators) for k, x in enumerate(op.inputs) if x == t]
      vals = {tuple(rw["new_of"](T)) for _, _, rw in lst}
      dtype = NP_OF[src.tensors[t].type]
      if len(vals) == 1 and len(allusers) == len(lst):
        v = np.array(list(vals)[0], dtype=dtype)
        sg.tensors[t].buffer = self.buffer_for(v)
        rep["const_rewritten_in_place"] += 1
      else:
        for oi, slot, rw in lst:
          v = np.array(rw["new_of"](T), dtype=dtype)
          nt = copy.deepcopy(sg.tensors[t])
          nt.buffer = self.buffer_for(v)
          nt.name = f"{name_of(nt.name)}_bucket{T}_op{oi}"
          sg.tensors.append(nt)
          sg.operators[oi].inputs[slot] = len(sg.tensors) - 1
          rep["const_split_new_tensor"] += 1
      for _, _, rw in lst:
        rep[rw["kind"]] += 1
    # The copy takes its subgraph index BEFORE its decompositions are appended: TFLite applies a delegate to the
    # subgraphs in index order and skips a decomposition subgraph only if the delegate that took its composite op has
    # already marked it skippable (tflite/core/interpreter.cc ModifyGraphWithDelegate loop + IsDelegationSkippable).
    # With the parent after its decompositions, every decomposition is delegated on its own first (SmolLM3-3B, one
    # bucket: 177 extra delegate applications per engine; Mac WebGPU init +0.1 s; on the S26 the init did not move).
    M.m.subgraphs.append(sg)
    my_index = len(M.m.subgraphs) - 1
    # decompositions
    for oi, child in prop.children.items():
      op = sg.operators[oi]
      base = re.match(r"^(.*\.impl)(?:_\d+)?$", name_of(child.sg.name))
      if not base:
        raise CloneError(f"decomposition name {name_of(child.sg.name)!r} is not `<name>.impl[_n]`")
      n = self.next_impl[base.group(1)]
      self.next_impl[base.group(1)] += 1
      new_name = f"{base.group(1)}_{n}"
      old_name = name_of(child.sg.name)

      def drename(s, old=old_name, new=new_name):
        return new + s[len(old):] if s.startswith(old + "_") or s == old else s
      di, crep = self.clone_sg(child, T, drename)
      M.m.subgraphs[di].name = new_name
      op.builtinOptions2.decompositionSubgraphIndex = di
      rep["decompositions"] += 1
      for k, v in crep.items():
        rep[k] += v
    return my_index, rep


def clone_signatures(M, source, lengths, log=print):
  """Add prefill_<T> for every T in lengths, copied from the signature `source`. Returns the report."""
  m = M.m
  keys = [name_of(sd.signatureKey) for sd in m.signatureDefs]
  if source not in keys:
    raise CloneError(f"no signature {source} (has {keys})")
  sd = m.signatureDefs[keys.index(source)]
  si = sd.subgraphIndex
  sg = m.subgraphs[si]
  T0 = prefill_len(sg, sd)
  seeds, found = seeds_for(sg, sd, T0)
  prop = Propagation(M, si, seeds, T0, source, log)
  summ = prop.summary()
  # every signature output must be T-free (KV caches); a prefill output that carries T is not handled
  for tm in sd.outputs:
    if prop.st.get(tm.tensorIndex) is not None:
      raise CloneError(f"signature output {name_of(tm.name)} depends on T")
  report = {"source": source, "T0": T0, "seed_inputs": {k: v for k, v in found.items() if not k.startswith("kv_cache")},
            "kv_inputs": sum(1 for k in found if k.startswith("kv_cache")), "propagation": summ, "added": {}}
  cl = Cloner(M)
  existing_lens = {}
  for s in m.signatureDefs:
    k = name_of(s.signatureKey)
    if k.startswith("prefill"):
      existing_lens[prefill_len(m.subgraphs[s.subgraphIndex], s)] = k
  for T in lengths:
    if T in existing_lens:
      raise CloneError(f"the bundle already has a prefill signature of length {T}: {existing_lens[T]}")
    if T < 1:
      raise CloneError(f"length {T}")
    for w in summ["fixed_windows"]:
      if w["size"] > w["o"] * T * w["i"]:
        raise CloneError(f"prefill_{T}: a fixed SLICE window of {w['size']} rows does not fit a T dim of {w['o'] * T * w['i']}")
    old_prefix, new_prefix = f"{source}_", f"prefill_{T}_"

    def rename(s, a=old_prefix, b=new_prefix):
      return b + s[len(a):] if s.startswith(a) else s
    n_sub0 = len(m.subgraphs)
    ni, rep = cl.clone_sg(prop, T, rename)
    key = f"prefill_{T}"
    m.subgraphs[ni].name = key
    nsd = copy.deepcopy(sd)
    nsd.signatureKey = key
    nsd.subgraphIndex = ni
    # insert among the prefill signatures, longest first (exporter order), before the others
    pos = 0
    for idx, s in enumerate(m.signatureDefs):
      k = name_of(s.signatureKey)
      if k.startswith("prefill") and prefill_len(m.subgraphs[s.subgraphIndex], s) > T:
        pos = idx + 1
    m.signatureDefs.insert(pos, nsd)
    existing_lens[T] = key
    report["added"][key] = {"subgraph": ni, "subgraphs_added": len(m.subgraphs) - n_sub0, **dict(rep)}
    log(f"added {key}: subgraph {ni}, {len(m.subgraphs) - n_sub0} subgraphs ({rep['decompositions']} decompositions), "
        f"{rep['RESHAPE.shape']} RESHAPE shapes + {rep['SLICE.size']} SLICE sizes rewritten, "
        f"{rep['tensors_T']} tensors carry T")
  report["new_buffers"] = len(cl.new_buffers)
  report["signature_order"] = [name_of(s.signatureKey) for s in m.signatureDefs]
  return report


# ---------------------------------------------------------------------------------------------------------------------
# comparison (self-test)
# ---------------------------------------------------------------------------------------------------------------------

def opt_dict(o):
  if o is None:
    return None
  d = {}
  for k, v in vars(o).items():
    if isinstance(v, np.ndarray):
      v = v.tolist()
    elif isinstance(v, (list, tuple)):
      v = [x.tolist() if isinstance(x, np.ndarray) else x for x in v]
    d[k] = v
  return d


def quant_dict(q):
  if q is None:
    return None
  d = opt_dict(q)
  if q.details is not None:
    d["details"] = opt_dict(q.details)
  return d


def as_bytes(x):
  if x is None:
    return b""
  if isinstance(x, np.ndarray):
    return x.astype(np.uint8).tobytes()
  return bytes(bytearray(x))


def tensor_cmp(M, sa, ta, sb, tb):
  """differences between tensor ta of subgraph sa and tensor tb of sb (shape, signature, dtype, quantization,
  constant content: inline bytes compared by value, external buffers by index)."""
  A, B = sa.tensors[ta], sb.tensors[tb]
  diffs = []
  if [int(x) for x in A.shape] != [int(x) for x in B.shape]:
    diffs.append(("shape", [int(x) for x in A.shape], [int(x) for x in B.shape]))
  sa_ = [int(x) for x in A.shapeSignature] if A.shapeSignature is not None else []
  sb_ = [int(x) for x in B.shapeSignature] if B.shapeSignature is not None else []
  if sa_ != sb_:
    diffs.append(("shape_signature", sa_, sb_))
  if A.type != B.type:
    diffs.append(("dtype", A.type, B.type))
  if bool(A.isVariable) != bool(B.isVariable) or bool(A.hasRank) != bool(B.hasRank):
    diffs.append(("flags", (A.isVariable, A.hasRank), (B.isVariable, B.hasRank)))
  if quant_dict(A.quantization) != quant_dict(B.quantization):
    diffs.append(("quantization", None, None))
  ba, bb = M.m.buffers[A.buffer], M.m.buffers[B.buffer]
  ia = ba.data is not None and len(ba.data) > 0
  ib = bb.data is not None and len(bb.data) > 0
  ea = ba.offset is not None and ba.offset > 1
  eb = bb.offset is not None and bb.offset > 1
  if (ia, ea) != (ib, eb):
    diffs.append(("buffer_kind", (ia, ea), (ib, eb)))
  elif ia and bytes(ba.data) != bytes(bb.data):
    va = np.frombuffer(bytes(ba.data), NP_OF.get(A.type, np.uint8)).tolist() if A.type in NP_OF else "bytes"
    vb = np.frombuffer(bytes(bb.data), NP_OF.get(B.type, np.uint8)).tolist() if B.type in NP_OF else "bytes"
    diffs.append(("const_value", va, vb))
  elif ea and A.buffer != B.buffer:
    diffs.append(("external_buffer_index", A.buffer, B.buffer))
  return diffs


def compare_subgraphs(M, ia, ib, label, out, name_map):
  sa, sb = M.m.subgraphs[ia], M.m.subgraphs[ib]
  c = out.setdefault("counts", collections.Counter())
  ex = out.setdefault("examples", collections.defaultdict(list))

  def note(kind, detail):
    c[kind] += 1
    if len(ex[kind]) < 4:
      ex[kind].append(detail)
  if len(sa.operators) != len(sb.operators):
    note("op_count", (label, len(sa.operators), len(sb.operators)))
    return
  if len(sa.tensors) != len(sb.tensors):
    note("tensor_count", (label, len(sa.tensors), len(sb.tensors)))
  if list(sa.inputs) != list(sb.inputs) or list(sa.outputs) != list(sb.outputs):
    note("subgraph_io", (label, list(sa.inputs)[:6], list(sb.inputs)[:6]))
  used_a, used_b = referenced(sa), referenced(sb)
  norm = lambda x: re.sub(r";\d+(_scales)?$", r"\1", x)
  for t in range(min(len(sa.tensors), len(sb.tensors))):
    dead = t not in used_a and t not in used_b
    for d in tensor_cmp(M, sa, t, sb, t):
      note(("dead_tensor_" if dead else "tensor_") + d[0], (label, t, name_of(sa.tensors[t].name)[-60:], d[1], d[2]))
    if (t in used_a) != (t in used_b):
      note("tensor_liveness", (label, t))
    na, nb = name_map(name_of(sa.tensors[t].name)), name_of(sb.tensors[t].name)
    if na != nb:
      c["tensor_name"] += 1
      if norm(na) != norm(nb):
        c["tensor_name_after_counter_strip"] += 1
      if len(ex["tensor_name"]) < 4:
        ex["tensor_name"].append((label, t, na[-80:], nb[-80:]))
  c["tensors_compared"] += min(len(sa.tensors), len(sb.tensors))
  for oi, (oa, ob) in enumerate(zip(sa.operators, sb.operators)):
    c["ops_compared"] += 1
    if M.opname(oa) != M.opname(ob):
      note("opcode", (label, oi, M.opname(oa), M.opname(ob)))
      continue
    if list(oa.inputs) != list(ob.inputs) or list(oa.outputs) != list(ob.outputs):
      note("op_io_index", (label, oi, M.opname(oa), list(oa.inputs), list(ob.inputs)))
    if oa.builtinOptionsType != ob.builtinOptionsType or opt_dict(oa.builtinOptions) != opt_dict(ob.builtinOptions):
      note("options", (label, oi, M.opname(oa), opt_dict(oa.builtinOptions), opt_dict(ob.builtinOptions)))
    if oa.builtinOptions2Type != ob.builtinOptions2Type:
      note("options2_type", (label, oi))
    elif oa.builtinOptions2Type == fb.BuiltinOptions2.StableHLOCompositeOptions:
      a2, b2 = oa.builtinOptions2, ob.builtinOptions2
      if name_of(a2.name) != name_of(b2.name) or a2.version != b2.version:
        note("composite_name_version", (label, oi, name_of(a2.name), name_of(b2.name)))
      if as_bytes(a2.compositeAttributes) != as_bytes(b2.compositeAttributes) or \
         a2.compositeAttributesFormat != b2.compositeAttributesFormat:
        note("composite_attrs", (label, oi, name_of(a2.name)))
      c["composites_compared"] += 1
      compare_subgraphs(M, a2.decompositionSubgraphIndex, b2.decompositionSubgraphIndex,
                        f"{label}/{name_of(M.m.subgraphs[b2.decompositionSubgraphIndex].name)}", out,
                        lambda s, A=name_of(M.m.subgraphs[a2.decompositionSubgraphIndex].name),
                        B=name_of(M.m.subgraphs[b2.decompositionSubgraphIndex].name): (B + s[len(A):]) if s.startswith(A) else s)
      c["decompositions_compared"] += 1
    elif opt_dict(oa.builtinOptions2) != opt_dict(ob.builtinOptions2):
      note("options2", (label, oi))
    if as_bytes(oa.customOptions) != as_bytes(ob.customOptions):
      note("custom_options", (label, oi))


def compare_signatures(M, key_a, key_b):
  m = M.m
  sda = next(s for s in m.signatureDefs if name_of(s.signatureKey) == key_a)
  sdb = next(s for s in m.signatureDefs if name_of(s.signatureKey) == key_b)
  out = {}
  io_a = [(name_of(t.name), int(t.tensorIndex)) for t in sda.inputs] + [(name_of(t.name), int(t.tensorIndex)) for t in sda.outputs]
  io_b = [(name_of(t.name), int(t.tensorIndex)) for t in sdb.inputs] + [(name_of(t.name), int(t.tensorIndex)) for t in sdb.outputs]
  out["signature_io_identical"] = io_a == io_b
  sa, sb = m.subgraphs[sda.subgraphIndex], m.subgraphs[sdb.subgraphIndex]
  out["subgraph_names"] = [name_of(sa.name), name_of(sb.name)]
  compare_subgraphs(M, sda.subgraphIndex, sdb.subgraphIndex, key_b, out, lambda s: s)
  out["counts"] = dict(out["counts"])
  out["examples"] = {k: v for k, v in out["examples"].items()}
  return out


# ---------------------------------------------------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------------------------------------------------

def load_section_model(path, model_type="tf_lite_prefill_decode"):
  """ModelT of one TFLite section of a bundle, read in place (structure only; external buffers keep their offsets)."""
  secs = R.bundle_sections(path)
  s = [x for x in secs if x["type"] == "TFLiteModel" and x["items"].get("model_type", "").lower().endswith(model_type.replace("tf_lite_", ""))]
  if len(s) != 1:
    raise CloneError(f"{len(s)} sections of type {model_type}")
  s = s[0]
  with open(path, "rb") as f:
    f.seek(s["begin"])
    head = f.read(min(s["bytes"], 256 << 20))
  pm = fb.Model.GetRootAsModel(head, 0)
  ds, _ = R.external_region(pm)
  if ds is not None and ds > len(head):
    raise CloneError("structure larger than the read window")
  model = fb.ModelT.InitFromPackedBuf(bytearray(head[: ds if ds else s["bytes"]]), 0)
  for b in model.buffers:
    if b.data is not None:
      b.data = np.array(b.data, dtype=np.uint8)
  for sg in model.subgraphs:
    sg.inputs = [int(x) for x in sg.inputs] if sg.inputs is not None else []
    sg.outputs = [int(x) for x in sg.outputs] if sg.outputs is not None else []
    for op in sg.operators or []:
      op.inputs = [int(x) for x in op.inputs] if op.inputs is not None else []
      op.outputs = [int(x) for x in op.outputs] if op.outputs is not None else []
  return model, s


# not differences of the graph: tensor names (the exporter numbers them per bucket), counters, and the shape of a
# tensor no op / IO / quantization refers to (a dead tensor, e.g. the mask CONCATENATION output the retrofit removed)
SOFT = ("tensor_name", "tensor_name_after_counter_strip", "tensors_compared", "ops_compared", "composites_compared",
        "decompositions_compared", "dead_tensor_shape", "dead_tensor_shape_signature")


def selftest(args):
  model, sec = load_section_model(args.bundle)
  M = Model(model)
  against = [x for x in args.against.split(",") if x]
  lens = {}
  for k in against:
    sd = next(s for s in model.signatureDefs if name_of(s.signatureKey) == k)
    lens[k] = prefill_len(model.subgraphs[sd.subgraphIndex], sd)
  # clone into a copy of the model whose real target signatures are renamed out of the way
  for sd in model.signatureDefs:
    if name_of(sd.signatureKey) in against:
      sd.signatureKey = "real_" + name_of(sd.signatureKey)
  n_sub0 = len(model.subgraphs)
  rep = clone_signatures(M, args.source, [lens[k] for k in against])
  res = {"bundle": args.bundle, "source": args.source, "clone_report": rep, "compare": {}}
  for k in against:
    cmp = compare_signatures(M, k, "real_" + k)
    res["compare"][k] = cmp
    hard = {kk: v for kk, v in cmp["counts"].items() if kk not in SOFT}
    print(f"selftest {args.source} -> {k}: ops {cmp['counts'].get('ops_compared')}, tensors {cmp['counts'].get('tensors_compared')}, "
          f"composites {cmp['counts'].get('composites_compared')}, decompositions {cmp['counts'].get('decompositions_compared')}; "
          f"differences {hard or 0}; dead-tensor shape differences {cmp['counts'].get('dead_tensor_shape', 0)}; "
          f"tensor names differing {cmp['counts'].get('tensor_name', 0)} "
          f"({cmp['counts'].get('tensor_name_after_counter_strip', 0)} after stripping the exporter's ';N' counter); "
          f"signature IO identical {cmp['signature_io_identical']}")
    for kk, v in cmp["examples"].items():
      print("   ", kk, v[:2])
  res["subgraphs_added"] = len(model.subgraphs) - n_sub0
  with open(args.report, "w") as f:
    json.dump(res, f, indent=1, default=str)
  ok = all(not {kk for kk in c["counts"] if kk not in SOFT} and c["signature_io_identical"] for c in res["compare"].values())
  print("SELFTEST", "PASS" if ok else "DIFFERENCES (see report)")
  return 0 if ok else 1


def clone_tflite(src, dst, lengths=(), source="prefill_128", dry_run=False):
  """Drop-in for gpu_graph_retrofit.retrofit_tflite inside retrofit_bundle (same call shape)."""
  model, data_start, size = R.load_tflite(src)
  src_model, _, _ = R.load_tflite(src)
  M = Model(model, src)
  n_sub0, n_sig0, n_buf0 = len(model.subgraphs), len(model.signatureDefs), len(model.buffers)
  rep = clone_signatures(M, source, list(lengths))
  rep["src"] = {"path": src, "bytes": size, "subgraphs": n_sub0, "signatures": n_sig0, "buffers": n_buf0}
  rep["dst_counts"] = {"subgraphs": len(model.subgraphs), "signatures": len(model.signatureDefs), "buffers": len(model.buffers)}
  if dry_run:
    return rep
  rep["layout"] = R.write_tflite(model, src, data_start, size, dst)
  rep["dst"] = {"path": dst, "bytes": os.path.getsize(dst)}
  rep["buffer_identity"] = R.buffer_identity(src, dst)
  if not rep["buffer_identity"]["ok"]:
    raise CloneError(f"buffer identity failed: {rep['buffer_identity']}")
  out_model, _, _ = R.load_tflite(dst)
  same = 0
  for i in range(n_sub0):
    if SV.sg_bytes(out_model.subgraphs[i]) != SV.sg_bytes(src_model.subgraphs[i]):
      raise CloneError(f"existing subgraph {i} ({name_of(src_model.subgraphs[i].name)}) changed")
    same += 1
  rep["existing_subgraphs_identical"] = f"{same}/{n_sub0}"
  ka = {name_of(s.signatureKey): s for s in src_model.signatureDefs}
  kb = {name_of(s.signatureKey): s for s in out_model.signatureDefs}
  for k, a in ka.items():
    b = kb[k]
    ta = [(name_of(t.name), int(t.tensorIndex)) for t in a.inputs] + [(name_of(t.name), int(t.tensorIndex)) for t in a.outputs]
    tb = [(name_of(t.name), int(t.tensorIndex)) for t in b.inputs] + [(name_of(t.name), int(t.tensorIndex)) for t in b.outputs]
    if ta != tb or a.subgraphIndex != b.subgraphIndex:
      raise CloneError(f"existing signature {k} changed")
  rep["existing_signatures_identical"] = sorted(ka)
  rep["operator_codes_identical"] = ([(name_of(c.customCode), max(c.builtinCode, c.deprecatedBuiltinCode), c.version) for c in src_model.operatorCodes]
                                     == [(name_of(c.customCode), max(c.builtinCode, c.deprecatedBuiltinCode), c.version) for c in out_model.operatorCodes])
  rep["metadata_identical"] = ([(name_of(x.name), x.buffer) for x in src_model.metadata or []]
                               == [(name_of(x.name), x.buffer) for x in out_model.metadata or []])
  # the new signatures, read back: name, subgraph, length, IO names = the source's
  src_sd = ka[source]
  src_io = [name_of(t.name) for t in src_sd.inputs] + [name_of(t.name) for t in src_sd.outputs]
  rows = []
  for T in lengths:
    b = kb[f"prefill_{T}"]
    sg = out_model.subgraphs[b.subgraphIndex]
    io = [name_of(t.name) for t in b.inputs] + [name_of(t.name) for t in b.outputs]
    if io != src_io:
      raise CloneError(f"prefill_{T}: signature IO names differ from {source}")
    rows.append({"key": f"prefill_{T}", "subgraph": int(b.subgraphIndex), "prefill_len": prefill_len(sg, b),
                 "ops": len(sg.operators), "tensors": len(sg.tensors)})
  rep["new_signatures_readback"] = rows
  return rep


def short_report(rep):
  amb = rep["propagation"]["ambiguous"]
  pat = collections.Counter((tuple(a["in_shape"]), tuple(a["const"]), a["T_dim"]) for a in amb)
  return {"source": rep["source"], "T0": rep["T0"], "added": rep["added"], "rewrites_per_bucket": rep["propagation"]["rewrites"],
          "dead_tensors_left_unchanged": rep["propagation"]["dead_with_T0"],
          "ops_per_bucket": rep["propagation"]["ops"],
          "ambiguous_reshapes_per_bucket": len(amb),
          "ambiguous_patterns": [{"in_shape": list(k[0]), "const": list(k[1]), "T_dim_chosen": k[2], "count": v} for k, v in pat.most_common()],
          "signature_order": rep["signature_order"]}


def build(args):
  lengths = sorted({int(x) for x in args.lengths.split(",") if x}, reverse=True)
  work = args.work or (os.path.splitext(args.dst)[0] + "_work")
  os.makedirs(work, exist_ok=True)
  R.retrofit_tflite = clone_tflite  # retrofit_bundle looks the name up at call time
  rep = R.retrofit_bundle(args.src, args.dst, work, lengths=lengths, source=args.source)
  shutil.rmtree(work, ignore_errors=True)
  rep["bundle"] = {"src": args.src, "dst": args.dst, "src_bytes": os.path.getsize(args.src), "dst_bytes": os.path.getsize(args.dst),
                   "src_sha256": SV.sha256_file(args.src), "dst_sha256": SV.sha256_file(args.dst)}
  rep["inspect_src"] = SV.inspect(args.src)
  rep["inspect_dst"] = SV.inspect(args.dst)
  rep["weight_region_identical"] = (rep["inspect_src"]["prefill_decode_section"]["external_region_sha256"]
                                    == rep["inspect_dst"]["prefill_decode_section"]["external_region_sha256"])
  rep["short"] = short_report(rep)
  with open(args.report, "w") as f:
    json.dump(rep, f, indent=1, default=str)
  short = dict(rep["short"])
  short.update({k: rep[k] for k in ("existing_subgraphs_identical", "operator_codes_identical", "metadata_identical",
                                    "weight_region_identical", "new_signatures_readback", "bundle")})
  short["buffer_identity_ok"] = rep["buffer_identity"]["ok"]
  short["sections"] = [(s["type"], s["items_src"].get("model_type"), s["sha256_src"][:12], s["sha256_dst"][:12], s["replaced"])
                       for s in rep["bundle_sections"]]
  print(json.dumps(short, indent=1, default=str))
  return 0


def tflite_cmd(args):
  lengths = sorted({int(x) for x in args.lengths.split(",") if x}, reverse=True)
  rep = clone_tflite(args.src, args.dst, lengths=lengths, source=args.source)
  rep["short"] = short_report(rep)
  with open(args.report, "w") as f:
    json.dump(rep, f, indent=1, default=str)
  print(json.dumps(rep["short"], indent=1, default=str))
  return 0


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  sub = ap.add_subparsers(dest="cmd", required=True)
  b = sub.add_parser("build")
  b.add_argument("src")
  b.add_argument("dst")
  b.add_argument("--lengths", required=True)
  b.add_argument("--source", default="prefill_128")
  b.add_argument("--report", required=True)
  b.add_argument("--work")
  t = sub.add_parser("tflite")
  t.add_argument("src")
  t.add_argument("dst")
  t.add_argument("--lengths", required=True)
  t.add_argument("--source", default="prefill_128")
  t.add_argument("--report", required=True)
  s = sub.add_parser("selftest")
  s.add_argument("bundle")
  s.add_argument("--source", required=True)
  s.add_argument("--against", required=True)
  s.add_argument("--report", required=True)
  args = ap.parse_args()
  return {"build": build, "tflite": tflite_cmd, "selftest": selftest}[args.cmd](args)


if __name__ == "__main__":
  sys.exit(main())
