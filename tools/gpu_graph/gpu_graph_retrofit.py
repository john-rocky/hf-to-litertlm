#!/usr/bin/env python3
"""Retrofit the GPU KV composites into an already-exported LiteRT-LM prefill/decode graph.

A graph exported without `--apply_gpu_composites` updates the KV cache with two
DYNAMIC_UPDATE_SLICE ops per layer and runs attention with BATCH_MATMUL(adjY). A graph
exported with the flag has the same KV layout and the same weights; the differences are
(see README.md in this directory):

  * DUS(K) + DUS(V)            -> one STABLEHLO_COMPOSITE `odml.cache_update`
                                  (inputs: k_slice [1,bk,T,H], v_slice [1,bk,T,H] (pre-transpose),
                                  param_tensor, cache_k [1,bk,C,H], cache_v [1,bk,H,C],
                                  indices_k, indices_v; outputs: new K, new V)
  * BATCH_MATMUL(a, cache, adjY) -> STABLEHLO_COMPOSITE `odml.runtime_bmm` (inputs: a, cache, param_tensor)
  * a signature input `param_tensor` INT32[1,1,1,7] (the runtime fills {start, end, end, 0, 0, 0, 0}).

This tool rewrites only the graph structure. Every original buffer keeps its bytes: external
buffers (Buffer.offset/size) are copied verbatim as one region and their offsets shifted by a
multiple of 16384, inline buffers are re-serialized unchanged.

Attention mask (`--mask`, default select_float): the exporter applies the mask with SELECT_V2
whenever param_tensor is present (sdpa.py: `mask == 0` -> where(mask, logits, -1e30)). Keeping the
flag-off form (CONCATENATION of the mask over the GQA groups + ADD) on the runtime_bmm output
breaks generation on the Mac WebGPU delegate (LiteRT-LM 0.17.1: full delegation, no Validation
error, garbage text).
  select_float  FLOAT32 mask input kept; EQUAL(mask, 0) -> CAST -> CONCAT(axis 1) -> NOT_EQUAL ->
                RESHAPE -> SELECT_V2, once per signature. Relies on the runtime's float mask
                contract (visible = 0.0 exactly; InitializeAttentionMask).
  select_bool   mask input becomes BOOL (visible = true), as `--use_bool_mask`.
  add_bcast     ADD kept, no CONCATENATION: logits viewed as [bk, g, T, C] + broadcast mask (the
                shape of a reference export made with the flag; also correct on the Mac GPU).
  add           the flag-off form unchanged (record only; broken on the Mac GPU).

Decomposition subgraphs (what the CPU interpreter runs; GPU delegates replace the composite):
  * exporter (default): the original ops, without the exporter's `ADD(x, 0)` tracing artifact —
      cache_update = DUS(cache_k, k, idx_k); RESHAPE (T == 1) or TRANSPOSE [0,1,3,2] of v; DUS(cache_v, ., idx_v)
      runtime_bmm  = BATCH_MATMUL(a, b, adjY=True)
  * v2: runtime_bmm reads the valid length L = clamp(param[1], 0, C), SLICEs the cache (and, for
      probs.V, the probabilities) to L, multiplies, and (q.K) PADs the logits back to C — the shape
      of a reference export's decomposition.

Usage:
  gpu_graph_retrofit.py IN.tflite OUT.tflite [--mask select_float|select_bool|add_bcast|add]
                        [--decomp exporter|v2] [--indices keep|concat] [--no-rhs-cache-update]
                        [--report out.json] [--dry-run]
  gpu_graph_retrofit.py IN.litertlm OUT.litertlm [...]     (unpack -> swap prefill_decode -> pack,
                        every other section asserted byte-identical; needs LITERT_LM_CLI)
"""

import argparse
import collections
import copy
import hashlib
import json
import mmap
import os
import re
import shutil
import subprocess
import sys
import tempfile

import flatbuffers
import numpy as np
from ai_edge_litert import schema_py_generated as fb
from flatbuffers import flexbuffers

OP = fb.BuiltinOperator
TT = fb.TensorType
BN = {v: k for k, v in vars(OP).items() if not k.startswith("_")}
ALIGN = 16384  # external-region shift granularity: keeps every buffer's alignment up to 16 KiB
LITERT_LM_CLI = os.environ.get("LITERT_LM_CLI", "litert-lm")  # the litert-lm CLI (pip install litert-lm), used to unpack / pack bundles


# ----------------------------------------------------------------------------------------------
# reading / writing
# ----------------------------------------------------------------------------------------------

def external_region(m):
  """(data_start, n_external): the first external-buffer offset of a packed Model."""
  starts = []
  for i in range(m.BuffersLength()):
    b = m.Buffers(i)
    if b.Offset() > 1:
      starts.append(b.Offset())
  return (min(starts) if starts else None), len(starts)


def load_tflite(path):
  """-> (ModelT, data_start or None, file_size). Parses only the flatbuffer region."""
  size = os.path.getsize(path)
  with open(path, "rb") as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    m = fb.Model.GetRootAsModel(mm, 0)
    data_start, _ = external_region(m)
    region = bytes(mm[: data_start if data_start else size])
    mm.close()
  model = fb.ModelT.InitFromPackedBuf(bytearray(region), 0)
  # Unpacked inline buffers are numpy views of `region`; materialize so nothing aliases it.
  for b in model.buffers:
    if b.data is not None:
      b.data = np.array(b.data, dtype=np.uint8)
  # Index vectors come back as numpy arrays; plain lists keep list operations simple.
  for sg in model.subgraphs:
    sg.inputs = [int(x) for x in sg.inputs] if sg.inputs is not None else []
    sg.outputs = [int(x) for x in sg.outputs] if sg.outputs is not None else []
    for op in sg.operators or []:
      op.inputs = [int(x) for x in op.inputs] if op.inputs is not None else []
      op.outputs = [int(x) for x in op.outputs] if op.outputs is not None else []
  return model, data_start, size


def pack_model(model):
  builder = flatbuffers.Builder(1 << 22)
  builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
  return bytes(builder.Output())


def write_tflite(model, src_path, src_data_start, src_size, dst_path):
  """Serialize `model`, shift external offsets so the region lands after the new structure,
  copy [src_data_start, EOF) verbatim. Returns a dict describing the layout."""
  if src_data_start is None:
    with open(dst_path, "wb") as f:
      f.write(pack_model(model))
    return {"external": 0}
  ext = [b for b in model.buffers if b.offset is not None and b.offset > 1]
  first = pack_model(model)
  new_start = src_data_start
  while new_start < len(first):
    new_start += ALIGN
  delta = new_start - src_data_start
  for b in ext:
    b.offset += delta
  fbytes = pack_model(model)
  if len(fbytes) != len(first):
    raise RuntimeError(f"flatbuffer size changed with the offsets: {len(first)} -> {len(fbytes)}")
  if len(fbytes) > new_start:
    raise RuntimeError("structure overlaps the external region")
  with open(dst_path, "wb") as out, open(src_path, "rb") as src:
    out.write(fbytes)
    out.write(b"\0" * (new_start - len(fbytes)))
    src.seek(src_data_start)
    left = src_size - src_data_start
    while left > 0:
      chunk = src.read(min(1 << 26, left))
      out.write(chunk)
      left -= len(chunk)
  for b in ext:  # leave the in-memory model as it was read
    b.offset -= delta
  return {"external": len(ext), "src_data_start": src_data_start, "dst_data_start": new_start,
          "delta": delta, "structure_bytes": len(fbytes)}


def sha256_range(path, start, end, h=None):
  h = h or hashlib.sha256()
  with open(path, "rb") as f:
    f.seek(start)
    left = end - start
    while left > 0:
      b = f.read(min(1 << 24, left))
      h.update(b)
      left -= len(b)
  return h.hexdigest()


def buffer_identity(src_path, dst_path):
  """Every buffer of src exists at the same index in dst with the same bytes (inline: same
  bytes; external: same size, offset shifted by one constant, region bytes identical)."""
  def parse(p):
    f = open(p, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    return f, mm, fb.Model.GetRootAsModel(mm, 0)
  fa, ma, A = parse(src_path)
  fbb, mb, B = parse(dst_path)
  res = {"src_buffers": A.BuffersLength(), "dst_buffers": B.BuffersLength(), "inline_same": 0,
         "external_same_size": 0, "empty": 0, "mismatch": []}
  deltas = set()
  for i in range(A.BuffersLength()):
    a, b = A.Buffers(i), B.Buffers(i)
    if a.Offset() > 1:
      if b.Offset() > 1 and a.Size() == b.Size():
        res["external_same_size"] += 1
        deltas.add(b.Offset() - a.Offset())
      else:
        res["mismatch"].append(i)
    elif a.DataLength() > 0:
      if a.DataLength() == b.DataLength() and bytes(a.DataAsNumpy()) == bytes(b.DataAsNumpy()):
        res["inline_same"] += 1
      else:
        res["mismatch"].append(i)
    else:
      res["empty"] += 1
      if b.DataLength() or b.Offset() > 1:
        res["mismatch"].append(i)
  res["offset_deltas"] = sorted(deltas)
  sa, _ = external_region(A)
  sb, _ = external_region(B)
  if sa is not None:
    res["src_region"] = [sa, os.path.getsize(src_path)]
    res["dst_region"] = [sb, os.path.getsize(dst_path)]
    res["region_sha256_src"] = sha256_range(src_path, sa, os.path.getsize(src_path))
    res["region_sha256_dst"] = sha256_range(dst_path, sb, os.path.getsize(dst_path))
    res["region_identical"] = res["region_sha256_src"] == res["region_sha256_dst"]
  res["ok"] = (not res["mismatch"]) and len(deltas) <= 1 and res.get("region_identical", True)
  for f, mm in ((fa, ma), (fbb, mb)):
    mm.close()
    f.close()
  return res


# ----------------------------------------------------------------------------------------------
# graph helpers
# ----------------------------------------------------------------------------------------------

def name_of(x):
  return x.decode() if isinstance(x, (bytes, bytearray)) else (x or "")


class Graph:
  """Index helpers over one SubGraphT."""

  def __init__(self, model, sg):
    self.model, self.sg = model, sg
    self.code_of = {}
    for i, c in enumerate(model.operatorCodes):
      self.code_of[i] = max(c.builtinCode, c.deprecatedBuiltinCode)
    self.refresh()

  def refresh(self):
    self.producer = {}
    self.consumers = collections.defaultdict(list)
    for oi, op in enumerate(self.sg.operators):
      for t in op.outputs:
        self.producer[t] = oi
      for t in op.inputs:
        if t >= 0:
          self.consumers[t].append(oi)

  def opcode(self, op):
    return self.code_of[op.opcodeIndex]

  def tensor(self, i):
    return self.sg.tensors[i]

  def shape(self, i):
    return [int(x) for x in self.sg.tensors[i].shape]


def opcode_index(model, builtin, version=1):
  for i, c in enumerate(model.operatorCodes):
    if max(c.builtinCode, c.deprecatedBuiltinCode) == builtin:
      return i
  oc = fb.OperatorCodeT()
  oc.builtinCode = builtin
  oc.deprecatedBuiltinCode = min(builtin, OP.PLACEHOLDER_FOR_GREATER_OP_CODES)
  oc.version = version
  model.operatorCodes.append(oc)
  return len(model.operatorCodes) - 1


def new_buffer(model, arr):
  b = fb.BufferT()
  b.data = np.frombuffer(np.ascontiguousarray(arr).tobytes(), dtype=np.uint8).copy()
  model.buffers.append(b)
  return len(model.buffers) - 1


def add_tensor(model, sg, name, ttype, shape, const=None, signature=None):
  t = fb.TensorT()
  t.name = name
  t.type = ttype
  t.shape = list(shape)
  t.shapeSignature = list(signature) if signature is not None else list(shape)
  t.buffer = new_buffer(model, const) if const is not None else 0
  t.hasRank = True
  t.quantization = fb.QuantizationParametersT()
  sg.tensors.append(t)
  return len(sg.tensors) - 1


def make_op(model, builtin, inputs, outputs, options_type=0, options=None):
  op = fb.OperatorT()
  op.opcodeIndex = opcode_index(model, builtin)
  op.inputs = list(inputs)
  op.outputs = list(outputs)
  op.builtinOptionsType = options_type
  op.builtinOptions = options
  return op


def composite_op(model, template, name, decomp_index, attrs, inputs, outputs):
  op = copy.deepcopy(template)
  op.inputs = list(inputs)
  op.outputs = list(outputs)
  o = fb.StableHLOCompositeOptionsT()
  o.name = name
  o.decompositionSubgraphIndex = decomp_index
  o.compositeAttributes = list(flexbuffers.Dumps(attrs))
  o.compositeAttributesFormat = 0  # FLEXBUFFERS
  o.version = 0
  op.builtinOptions2Type = fb.BuiltinOptions2.StableHLOCompositeOptions
  op.builtinOptions2 = o
  op.builtinOptionsType = 0
  op.builtinOptions = None
  return op


def is_const(model, sg, t):
  b = model.buffers[sg.tensors[t].buffer]
  return (b.data is not None and len(b.data) > 0) or (b.offset is not None and b.offset > 1)


def stable_topo_sort(model, sg):
  """Kahn's algorithm with the original position as priority."""
  import heapq
  n = len(sg.operators)
  producer = {}
  for oi, op in enumerate(sg.operators):
    for t in op.outputs:
      producer[t] = oi
  deps = [set() for _ in range(n)]
  users = [[] for _ in range(n)]
  for oi, op in enumerate(sg.operators):
    for t in op.inputs:
      if t >= 0 and t in producer and producer[t] != oi:
        deps[oi].add(producer[t])
  for oi in range(n):
    for d in deps[oi]:
      users[d].append(oi)
  indeg = [len(d) for d in deps]
  heap = [oi for oi in range(n) if indeg[oi] == 0]
  heapq.heapify(heap)
  order = []
  while heap:
    oi = heapq.heappop(heap)
    order.append(oi)
    for u in users[oi]:
      indeg[u] -= 1
      if indeg[u] == 0:
        heapq.heappush(heap, u)
  if len(order) != n:
    raise RuntimeError("cycle in operator graph")
  sg.operators = [sg.operators[i] for i in order]
  return order != list(range(n))


def check_topo(model, sg):
  """Every op input that some op produces must be produced earlier (constants, zero-element
  constants with empty buffers and subgraph inputs are never produced)."""
  where = {}
  for oi, op in enumerate(sg.operators):
    for t in op.outputs:
      where[t] = oi
  for oi, op in enumerate(sg.operators):
    for t in op.inputs:
      if t >= 0 and t in where and where[t] >= oi:
        return False, (oi, t)
  return True, None


# ----------------------------------------------------------------------------------------------
# the rewrite
# ----------------------------------------------------------------------------------------------

KV_RX = re.compile(r"^kv_cache_([kv])_(\d+)$")


def find_layers(model, si, sd):
  """Attention layers of one signature subgraph -> list of dicts (sorted by layer)."""
  sg = model.subgraphs[si]
  g = Graph(model, sg)
  kv_in = {}
  for tm in sd.inputs:
    mt = KV_RX.match(name_of(tm.name))
    if mt:
      kv_in[tm.tensorIndex] = (mt.group(1), int(mt.group(2)))
  layers = collections.defaultdict(dict)
  other_dus = 0
  for oi, op in enumerate(sg.operators):
    if g.opcode(op) != OP.DYNAMIC_UPDATE_SLICE:
      continue
    key = kv_in.get(op.inputs[0])
    if key is None:
      other_dus += 1
      continue
    kind, li = key
    if kind in layers[li]:
      raise RuntimeError(f"layer {li}: two DUS on kv_cache_{kind}_{li}")
    layers[li][kind] = oi
  out = []
  for li in sorted(layers):
    d = layers[li]
    if set(d) != {"k", "v"}:
      raise RuntimeError(f"layer {li}: DUS only for {sorted(d)}")
    dk, dv = sg.operators[d["k"]], sg.operators[d["v"]]
    ck, k_slice, idx_k = dk.inputs
    cv, v_slice, idx_v = dv.inputs
    ok_k, ok_v = dk.outputs[0], dv.outputs[0]
    sk, sv = g.shape(ck), g.shape(cv)
    if len(sk) != 4 or len(sv) != 4 or sk[0] != 1 or sk[1] != sv[1] or sk[2] != sv[3] or sk[3] != sv[2]:
      raise RuntimeError(f"layer {li}: unexpected cache shapes K{sk} V{sv}")
    bk, C, H = sk[1], sk[2], sk[3]
    ks = g.shape(k_slice)
    vs = g.shape(v_slice)
    if ks[:2] != [1, bk] or ks[3] != H or vs != [1, bk, H, ks[2]]:
      raise RuntimeError(f"layer {li}: unexpected slice shapes k{ks} v{vs}")
    T = ks[2]
    bmm = {}
    for kind, outt in (("qk", ok_k), ("pv", ok_v)):
      cons = [c for c in g.consumers.get(outt, [])
              if g.opcode(sg.operators[c]) == OP.BATCH_MATMUL and sg.operators[c].inputs[1] == outt]
      if len(cons) > 1:
        raise RuntimeError(f"layer {li}: {len(cons)} BATCH_MATMUL read the {kind} cache")
      for c in cons:
        o = sg.operators[c].builtinOptions
        if o is None or not o.adjY or o.adjX:
          raise RuntimeError(f"layer {li}: BATCH_MATMUL {c} is not (adjX=False, adjY=True)")
        bmm[kind] = c
    out.append({"layer": li, "dus_k": d["k"], "dus_v": d["v"], "cache_k": ck, "cache_v": cv,
                "k_slice": k_slice, "v_slice": v_slice, "idx_k": idx_k, "idx_v": idx_v,
                "out_k": ok_k, "out_v": ok_v, "bk": bk, "C": C, "H": H, "T": T, "bmm": bmm})
  return out, other_dus, g


def decomp_cache_update(model, base_name, L, src_ops, opts):
  """Decomposition subgraph for one odml.cache_update."""
  sg = fb.SubGraphT()
  sg.tensors, sg.operators = [], []
  sg.name = base_name
  bk, C, H, T = L["bk"], L["C"], L["H"], L["T"]
  F, I = TT.FLOAT32, TT.INT32
  a = [add_tensor(model, sg, f"{base_name}_arg{i}", tt, shp) for i, (tt, shp) in enumerate([
      (F, [1, bk, T, H]), (F, [1, bk, T, H]), (I, [1, 1, 1, 7]), (F, [1, bk, C, H]), (F, [1, bk, H, C]),
      (I, [4]), (I, [4])])]
  sg.inputs = a
  out_k = add_tensor(model, sg, f"{base_name}_ret0", F, [1, bk, C, H])
  v_t = add_tensor(model, sg, f"{base_name}_v_transposed", F, [1, bk, H, T])
  out_v = add_tensor(model, sg, f"{base_name}_ret1", F, [1, bk, H, C])
  dus_k = copy.deepcopy(src_ops["dus_k"])
  dus_k.inputs, dus_k.outputs = [a[3], a[0], a[5]], [out_k]
  if T == 1:
    shp = add_tensor(model, sg, f"{base_name}_v_shape", I, [4], np.array([1, bk, H, 1], np.int32))
    vop = make_op(model, OP.RESHAPE, [a[1], shp], [v_t])
  else:
    perm = add_tensor(model, sg, f"{base_name}_perm_0132", I, [4], np.array([0, 1, 3, 2], np.int32))
    vop = make_op(model, OP.TRANSPOSE, [a[1], perm], [v_t], fb.BuiltinOptions.TransposeOptions,
                  fb.TransposeOptionsT())
  dus_v = copy.deepcopy(src_ops["dus_v"])
  dus_v.inputs, dus_v.outputs = [a[4], v_t, a[6]], [out_v]
  sg.operators = [dus_k, vop, dus_v]
  sg.outputs = [out_k, out_v]
  return sg


def decomp_runtime_bmm(model, base_name, a_shape, b_shape, out_shape, bmm_op, is_src, C, variant):
  sg = fb.SubGraphT()
  sg.tensors, sg.operators = [], []
  sg.name = base_name
  F, I = TT.FLOAT32, TT.INT32
  a = add_tensor(model, sg, f"{base_name}_arg0", F, a_shape)
  b = add_tensor(model, sg, f"{base_name}_arg1", F, b_shape)
  p = add_tensor(model, sg, f"{base_name}_arg2", I, [1, 1, 1, 7])
  sg.inputs = [a, b, p]
  if variant == "exporter":
    out = add_tensor(model, sg, f"{base_name}_ret0", F, out_shape)
    op = copy.deepcopy(bmm_op)
    op.inputs, op.outputs = [a, b], [out]
    sg.operators = [op]
    sg.outputs = [out]
    return sg
  # v2: valid length L = clamp(param[1], 0, C)
  ops = []
  flat_shape = add_tensor(model, sg, f"{base_name}_c_flat_shape", I, [1], np.array([7], np.int32))
  pflat = add_tensor(model, sg, f"{base_name}_param_flat", I, [7])
  ops.append(make_op(model, OP.RESHAPE, [p, flat_shape], [pflat]))
  idx1 = add_tensor(model, sg, f"{base_name}_c_idx1", I, [1], np.array([1], np.int32))
  va = add_tensor(model, sg, f"{base_name}_valid_a", I, [1])
  ops.append(make_op(model, OP.GATHER, [pflat, idx1], [va], fb.BuiltinOptions.GatherOptions,
                     fb.GatherOptionsT()))
  zero = add_tensor(model, sg, f"{base_name}_c_zero", I, [1], np.array([0], np.int32))
  vn = add_tensor(model, sg, f"{base_name}_valid_nonneg", I, [1])
  ops.append(make_op(model, OP.MAXIMUM, [va, zero], [vn]))
  cc = add_tensor(model, sg, f"{base_name}_c_cache", I, [1], np.array([C], np.int32))
  vl = add_tensor(model, sg, f"{base_name}_valid_len", I, [1])
  ops.append(make_op(model, OP.MINIMUM, [vn, cc], [vl]))
  begin = add_tensor(model, sg, f"{base_name}_c_begin4", I, [4], np.array([0, 0, 0, 0], np.int32))

  def concat(name, parts):
    t = add_tensor(model, sg, name, I, [4])
    o = fb.ConcatenationOptionsT()
    o.axis = 0
    ops.append(make_op(model, OP.CONCATENATION, parts, [t], fb.BuiltinOptions.ConcatenationOptions, o))
    return t

  if not is_src:
    # q.K: b = K cache [1,bk,C,H] -> [1,bk,L,H]; logits [1,bk,gT,L] -> PAD to C
    pre = add_tensor(model, sg, f"{base_name}_c_rhs_prefix", I, [2], np.array(b_shape[:2], np.int32))
    suf = add_tensor(model, sg, f"{base_name}_c_rhs_suffix", I, [1], np.array([b_shape[3]], np.int32))
    size = concat(f"{base_name}_rhs_size", [pre, vl, suf])
    bs = add_tensor(model, sg, f"{base_name}_rhs_slice", F, [b_shape[0], b_shape[1], 1, b_shape[3]],
                    signature=[b_shape[0], b_shape[1], -1, b_shape[3]])
    ops.append(make_op(model, OP.SLICE, [b, begin, size], [bs]))
    mm = add_tensor(model, sg, f"{base_name}_bmm", F, out_shape[:3] + [1], signature=out_shape[:3] + [-1])
    op = copy.deepcopy(bmm_op)
    op.inputs, op.outputs = [a, bs], [mm]
    ops.append(op)
    padr = add_tensor(model, sg, f"{base_name}_pad_right", I, [1])
    ops.append(make_op(model, OP.SUB, [cc, vl], [padr], fb.BuiltinOptions.SubOptions, fb.SubOptionsT()))
    ppre = add_tensor(model, sg, f"{base_name}_c_pad_prefix", I, [7], np.zeros(7, np.int32))
    pflat8 = add_tensor(model, sg, f"{base_name}_pad_flat", I, [8])
    o = fb.ConcatenationOptionsT()
    o.axis = 0
    ops.append(make_op(model, OP.CONCATENATION, [ppre, padr], [pflat8], fb.BuiltinOptions.ConcatenationOptions, o))
    pshape = add_tensor(model, sg, f"{base_name}_c_pad_shape", I, [2], np.array([4, 2], np.int32))
    pads = add_tensor(model, sg, f"{base_name}_pads", I, [4, 2])
    ops.append(make_op(model, OP.RESHAPE, [pflat8, pshape], [pads]))
    out = add_tensor(model, sg, f"{base_name}_ret0", F, out_shape)
    ops.append(make_op(model, OP.PAD, [mm, pads], [out], fb.BuiltinOptions.PadOptions, fb.PadOptionsT()))
  else:
    # probs.V: a = probs [1,bk,gT,C] -> [1,bk,gT,L]; b = V cache [1,bk,H,C] -> [1,bk,H,L]
    preb = add_tensor(model, sg, f"{base_name}_c_rhs_prefix", I, [3], np.array(b_shape[:3], np.int32))
    sizeb = concat(f"{base_name}_rhs_size", [preb, vl])
    bs = add_tensor(model, sg, f"{base_name}_rhs_slice", F, b_shape[:3] + [1], signature=b_shape[:3] + [-1])
    ops.append(make_op(model, OP.SLICE, [b, begin, sizeb], [bs]))
    prea = add_tensor(model, sg, f"{base_name}_c_lhs_prefix", I, [3], np.array(a_shape[:3], np.int32))
    sizea = concat(f"{base_name}_lhs_size", [prea, vl])
    as_ = add_tensor(model, sg, f"{base_name}_lhs_slice", F, a_shape[:3] + [1], signature=a_shape[:3] + [-1])
    ops.append(make_op(model, OP.SLICE, [a, begin, sizea], [as_]))
    out = add_tensor(model, sg, f"{base_name}_ret0", F, out_shape)
    op = copy.deepcopy(bmm_op)
    op.inputs, op.outputs = [as_, bs], [out]
    ops.append(op)
  sg.operators = ops
  sg.outputs = [out]
  return sg


def buffer_bytes(model, bi, src_path):
  b = model.buffers[bi]
  if b.data is not None and len(b.data):
    return bytes(b.data)
  if b.offset is not None and b.offset > 1:
    with open(src_path, "rb") as f:
      f.seek(b.offset)
      return f.read(b.size)
  return b""


def const_shape(model, sg, key, values, _cache={}):
  k = (id(sg), tuple(values))
  if k not in _cache:
    _cache[k] = add_tensor(model, sg, f"{key}_retrofit_shape_{'x'.join(map(str, values))}", TT.INT32, [len(values)],
                           np.array(values, np.int32))
  return _cache[k]


def rewrite_mask(model, sg, g, key, sd, mode, new_ops_before, remove, replace, rep):
  """Variant (3): attention mask applied with SELECT_V2 like the exporter's GPU-composite graph
  (sdpa.py: mask == 0 -> float -> concat over the GQA groups on axis 1 -> != 0 -> reshape ->
  where(mask, logits, -1e30)). mode 'select_float' keeps the FLOAT32 mask input and starts with
  EQUAL(mask, 0); 'select_bool' turns the input into BOOL (visible = true) and starts with CAST;
  'add_bcast' keeps ADD on a [bk, g, T, C] view of the logits (shape of a reference export made with the flag).
  The flag-off graph repeats the mask over the GQA groups with CONCATENATION(axis 2): one shared op
  in some export lineages, one per layer in others — both are handled; the new mask chain is built
  once per signature."""
  mask_t = next(tm.tensorIndex for tm in sd.inputs if name_of(tm.name) == "mask")
  cats = sorted(set(g.consumers.get(mask_t, [])))
  if not cats:
    raise RuntimeError(f"{key}: mask has no consumer")
  # Older export lineage (e.g. litert-community/SmolLM3-3B `SmolLM3-3B.litertlm`): no CONCATENATION;
  # the logits are viewed as [bk, g, T, C] and the [1, 1, T, C] mask is broadcast-added = already
  # the reference export's shape (mode add_bcast), which runs correctly next to runtime_bmm on the Mac
  # GPU. Kept as is for select_float / add_bcast.
  if all(g.opcode(sg.operators[c]) == OP.ADD for c in cats):
    views = []
    for c in cats:
      aop = sg.operators[c]
      other = aop.inputs[0] if aop.inputs[1] == mask_t else aop.inputs[1]
      shp = g.shape(other)
      if len(shp) != 4 or shp[2:] != g.shape(mask_t)[2:]:
        raise RuntimeError(f"{key}: mask ADD {c} adds to {shp}, not a [bk, g, T, C] view")
      views.append(shp)
    if mode == "select_bool":
      raise RuntimeError(f"{key}: select_bool is not implemented for the broadcast-ADD lineage")
    rep["mask"] = {"mode": "kept_broadcast_add", "requested": mode, "mask_adds": len(cats), "view": views[0],
                   "input_dtype": "FLOAT32"}
    return
  ngroups = None
  for ci in cats:
    cop = sg.operators[ci]
    if g.opcode(cop) != OP.CONCATENATION or any(t != mask_t for t in cop.inputs) or cop.builtinOptions.axis != 2:
      raise RuntimeError(f"{key}: mask consumer op {ci} ({BN.get(g.opcode(cop))}) is not CONCATENATION(mask x g, axis 2)")
    if ngroups not in (None, len(cop.inputs)):
      raise RuntimeError(f"{key}: mask CONCATENATIONs disagree on the group count")
    ngroups = len(cop.inputs)
  _, _, T, C = g.shape(mask_t)
  adds = []  # (concat op index, add op index)
  for ci in cats:
    for a in sorted(set(g.consumers.get(sg.operators[ci].outputs[0], []))):
      if g.opcode(sg.operators[a]) != OP.ADD or sg.operators[a].inputs[1] != sg.operators[ci].outputs[0]:
        raise RuntimeError(f"{key}: consumer {a} of mask CONCATENATION {ci} is not ADD(logits, mask)")
      adds.append((ci, a))
  F, I, B = TT.FLOAT32, TT.INT32, TT.BOOL
  first = cats[0]
  if mode == "add_bcast":
    # Cause isolation for the ADD form (reference export shape): no mask CONCATENATION; the logits
    # [1, bk, g*T, C] are viewed as [bk, g, T, C], the FLOAT32 mask [1, 1, T, C] is broadcast-added,
    # and the sum is viewed back. Values are the same as CONCATENATION + ADD.
    remove.update(cats)
    for _, a in adds:
      aop = sg.operators[a]
      lg = aop.inputs[0]
      _, bk, gT, C2 = g.shape(lg)
      v4 = add_tensor(model, sg, f"{key}_retrofit_logits_bgtc_{a}", F, [bk, ngroups, T, C2])
      s4 = const_shape(model, sg, key, [bk, ngroups, T, C2])
      o1 = make_op(model, OP.RESHAPE, [lg, s4], [v4])
      added = add_tensor(model, sg, f"{key}_retrofit_masked_bgtc_{a}", F, [bk, ngroups, T, C2])
      o2 = copy.deepcopy(aop)
      o2.inputs, o2.outputs = [v4, mask_t], [added]
      s_back = const_shape(model, sg, key, g.shape(aop.outputs[0]))
      o3 = make_op(model, OP.RESHAPE, [added, s_back], list(aop.outputs))
      new_ops_before[a].extend([o1, o2])
      replace[a] = o3
    rep["mask"] = {"mode": mode, "groups": ngroups, "mask_concats": len(cats), "add": len(adds), "input_dtype": "FLOAT32"}
    return
  zero = add_tensor(model, sg, f"{key}_retrofit_mask_zero", F, [], np.array(0.0, np.float32))
  neg = add_tensor(model, sg, f"{key}_retrofit_mask_fill", F, [], np.array(-1e30, np.float32))
  ops = []
  if mode == "select_float":
    eq = add_tensor(model, sg, f"{key}_retrofit_mask_visible", B, [1, 1, T, C])
    ops.append(make_op(model, OP.EQUAL, [mask_t, zero], [eq], fb.BuiltinOptions.EqualOptions, fb.EqualOptionsT()))
    src = eq
  else:
    sg.tensors[mask_t].type = B
    src = mask_t
  mf = add_tensor(model, sg, f"{key}_retrofit_mask_f32", F, [1, 1, T, C])
  co = fb.CastOptionsT()
  co.inDataType, co.outDataType = B, F
  ops.append(make_op(model, OP.CAST, [src], [mf], fb.BuiltinOptions.CastOptions, co))
  cat = add_tensor(model, sg, f"{key}_retrofit_mask_groups", F, [1, ngroups, T, C])
  o = fb.ConcatenationOptionsT()
  o.axis = 1
  ops.append(make_op(model, OP.CONCATENATION, [mf] * ngroups, [cat], fb.BuiltinOptions.ConcatenationOptions, o))
  nb = add_tensor(model, sg, f"{key}_retrofit_mask_groups_bool", B, [1, ngroups, T, C])
  ops.append(make_op(model, OP.NOT_EQUAL, [cat, zero], [nb], fb.BuiltinOptions.NotEqualOptions, fb.NotEqualOptionsT()))
  shp = add_tensor(model, sg, f"{key}_retrofit_mask_shape", I, [4], np.array([1, 1, ngroups * T, C], np.int32))
  cond = add_tensor(model, sg, f"{key}_retrofit_mask_cond", B, [1, 1, ngroups * T, C])
  ops.append(make_op(model, OP.RESHAPE, [nb, shp], [cond]))
  new_ops_before[first].extend(ops)
  remove.update(cats)
  for _, a in adds:
    aop = sg.operators[a]
    replace[a] = make_op(model, OP.SELECT_V2, [cond, aop.inputs[0], neg], list(aop.outputs),
                         fb.BuiltinOptions.SelectV2Options, fb.SelectV2OptionsT())
  rep["mask"] = {"mode": mode, "groups": ngroups, "mask_concats": len(cats), "select_v2": len(adds),
                 "input_dtype": "BOOL" if mode == "select_bool" else "FLOAT32"}


def retrofit(model, src_path, decomp="exporter", indices="keep", rhs_cache_update=True, mask="select_float"):
  """In-place rewrite of every prefill*/decode signature subgraph. Returns a report dict."""
  report = {"decomp": decomp, "indices": indices, "rhs_cache_update": rhs_cache_update, "mask": mask, "signatures": {}}
  comp_code = opcode_index(model, OP.STABLEHLO_COMPOSITE)
  template = None
  for sg in model.subgraphs:
    for op in sg.operators:
      if op.opcodeIndex == comp_code:
        template = op
        break
    if template:
      break
  if template is None:
    template = fb.OperatorT()
    template.opcodeIndex = comp_code
  counters = collections.Counter()
  for sd in model.signatureDefs:
    key = name_of(sd.signatureKey)
    if not (key.startswith("prefill") or key == "decode"):
      continue
    si = sd.subgraphIndex
    sg = model.subgraphs[si]
    if any(name_of(tm.name) == "param_tensor" for tm in sd.inputs):
      raise RuntimeError(f"{key}: already has param_tensor")
    layers, other_dus, g = find_layers(model, si, sd)
    if not layers:
      raise RuntimeError(f"{key}: no attention layer found")
    rep = {"layers": len(layers), "other_dus_untouched": other_dus, "v_producer": collections.Counter()}
    # param_tensor: new tensor at the end, subgraph input right after the mask, SignatureDef sorted
    p = add_tensor(model, sg, f"{key}_param_tensor", TT.INT32, [1, 1, 1, 7])
    mask_t = next((tm.tensorIndex for tm in sd.inputs if name_of(tm.name) == "mask"), None)
    pos = sg.inputs.index(mask_t) + 1 if mask_t in sg.inputs else len(sg.inputs)
    sg.inputs = list(sg.inputs[:pos]) + [p] + list(sg.inputs[pos:])
    tm = fb.TensorMapT()
    tm.name = "param_tensor"
    tm.tensorIndex = p
    sd.inputs = sorted(list(sd.inputs) + [tm], key=lambda x: name_of(x.name))
    # per-subgraph shared constants for the main-graph V rewrite
    shared = {}

    def const_i32(name, values):
      k = (name, tuple(values))
      if k not in shared:
        shared[k] = add_tensor(model, sg, f"{key}_{name}", TT.INT32, [len(values)], np.array(values, np.int32))
      return shared[k]

    remove = set()
    replace = {}  # op index -> new op
    new_ops_before = collections.defaultdict(list)  # op index -> ops inserted before it
    for L in layers:
      li, bk, C, H, T = L["layer"], L["bk"], L["C"], L["H"], L["T"]
      # --- V slice in the pre-transpose layout [1,bk,T,H]
      v = L["v_slice"]
      prod = g.producer.get(v)
      users = g.consumers.get(v, [])
      v_th = None
      if prod is not None and users == [L["dus_v"]] and v not in sg.outputs:
        pop = sg.operators[prod]
        pc = g.opcode(pop)
        if pc == OP.RESHAPE and T == 1:
          pop.inputs = [pop.inputs[0], const_i32("retrofit_v_shape", [1, bk, 1, H])]
          if pop.builtinOptions is not None and getattr(pop.builtinOptions, "newShape", None):
            pop.builtinOptions.newShape = [1, bk, 1, H]
          v_th = v
          rep["v_producer"]["reshape_retargeted"] += 1
        elif pc == OP.TRANSPOSE:
          pt = sg.tensors[pop.inputs[1]]
          raw = buffer_bytes(model, pt.buffer, src_path)
          if pt.type == TT.INT32 and len(raw) == 16:
            old = np.frombuffer(raw, dtype=np.int32).tolist()
            new = [old[0], old[1], old[3], old[2]]
            pop.inputs = [pop.inputs[0], const_i32("retrofit_v_perm", new)]
            v_th = v
            rep["v_producer"]["transpose_perm_composed"] += 1
      if v_th is not None:
        t = sg.tensors[v]
        t.shape = [1, bk, T, H]
        if t.shapeSignature is not None and len(t.shapeSignature):
          t.shapeSignature = [1, bk, T, H]
      else:
        v_th = add_tensor(model, sg, f"{key}_retrofit_l{li}_v_th", TT.FLOAT32, [1, bk, T, H])
        if T == 1:
          op = make_op(model, OP.RESHAPE, [v, const_i32("retrofit_v_shape", [1, bk, 1, H])], [v_th])
        else:
          op = make_op(model, OP.TRANSPOSE, [v, const_i32("retrofit_perm_0132", [0, 1, 3, 2])], [v_th],
                       fb.BuiltinOptions.TransposeOptions, fb.TransposeOptionsT())
        new_ops_before[max(L["dus_k"], L["dus_v"])].append(op)
        rep["v_producer"]["inserted"] += 1
      idx_k, idx_v = L["idx_k"], L["idx_v"]
      if indices == "concat":
        idx_k, idx_v = concat_indices(model, sg, g, key, L, const_i32, new_ops_before, shared, src_path)
      # --- odml.cache_update
      k_ops = {"dus_k": sg.operators[L["dus_k"]], "dus_v": sg.operators[L["dus_v"]]}
      n = counters["odml.cache_update"]
      counters["odml.cache_update"] += 1
      dsg = decomp_cache_update(model, f"odml.cache_update.impl_{n}", L, k_ops, {})
      model.subgraphs.append(dsg)
      cu = composite_op(model, template, "odml.cache_update", len(model.subgraphs) - 1,
                        {"cache_size": C, "head_size": H, "kv_cache_batch_size": bk},
                        [L["k_slice"], v_th, p, L["cache_k"], L["cache_v"], idx_k, idx_v],
                        [L["out_k"], L["out_v"]])
      later, earlier = max(L["dus_k"], L["dus_v"]), min(L["dus_k"], L["dus_v"])
      replace[later] = cu
      remove.add(earlier)
      # --- odml.runtime_bmm
      for kind in ("qk", "pv"):
        if kind not in L["bmm"]:
          continue
        bi = L["bmm"][kind]
        bop = sg.operators[bi]
        a_t, b_t = bop.inputs[0], bop.inputs[1]
        n = counters["odml.runtime_bmm"]
        counters["odml.runtime_bmm"] += 1
        dsg = decomp_runtime_bmm(model, f"odml.runtime_bmm.impl_{n}", g.shape(a_t), g.shape(b_t),
                                 g.shape(bop.outputs[0]), bop, kind == "pv", C, decomp)
        model.subgraphs.append(dsg)
        attrs = {"is_global": True, "is_src": kind == "pv"}
        if rhs_cache_update:
          attrs["rhs_cache_update"] = True
        replace[bi] = composite_op(model, template, "odml.runtime_bmm", len(model.subgraphs) - 1, attrs,
                                   [a_t, b_t, p], list(bop.outputs))
    if mask != "add":
      rewrite_mask(model, sg, g, key, sd, mask, new_ops_before, remove, replace, rep)
    ops = []
    for oi, op in enumerate(sg.operators):
      ops.extend(new_ops_before.get(oi, []))
      if oi in remove:
        continue
      ops.append(replace.get(oi, op))
    sg.operators = ops
    ok, bad = check_topo(model, sg)
    rep["reordered"] = False
    if not ok:
      rep["reordered"] = stable_topo_sort(model, sg)
      ok, bad = check_topo(model, sg)
      if not ok:
        raise RuntimeError(f"{key}: not topologically sortable at {bad}")
    rep["v_producer"] = dict(rep["v_producer"])
    rep["cache_shape"] = {"bk": layers[0]["bk"], "C": layers[0]["C"], "H": layers[0]["H"], "T": layers[0]["T"]}
    rep["bmm_per_layer"] = collections.Counter(len(L["bmm"]) for L in layers)
    rep["bmm_per_layer"] = {str(k): v for k, v in rep["bmm_per_layer"].items()}
    report["signatures"][key] = rep
  report["composites_added"] = dict(counters)
  return report


def concat_indices(model, sg, g, key, L, const_i32, new_ops_before, shared, src_path):
  """Variant (1): indices built like the exporter — CONCATENATION of four INT32[1] tensors: the
  constant parts as INT32[1] constants, the position as the INT32[1] tensor that the original
  RESHAPE turned into a scalar (input_pos in decode, SLICE(input_pos) in prefill)."""
  out = []
  for which, idx_t in (("k", L["idx_k"]), ("v", L["idx_v"])):
    pk = g.producer.get(idx_t)
    pop = sg.operators[pk] if pk is not None else None
    if pop is None or g.opcode(pop) != OP.PACK or len(pop.inputs) != 4:
      raise RuntimeError(f"layer {L['layer']}: indices_{which} not produced by a 4-input PACK")
    ck = ("concat_idx", which, tuple(pop.inputs))
    if ck in shared:
      out.append(shared[ck])
      continue
    parts = []
    for t in pop.inputs:
      vk = ("as1", t)
      if vk not in shared:
        raw = buffer_bytes(model, sg.tensors[t].buffer, src_path)
        rp = g.producer.get(t)
        if raw:
          val = int(np.frombuffer(raw, dtype=np.int32)[0])
          shared[vk] = const_i32(f"retrofit_idx_const_{val}", [val])
        elif rp is not None and g.opcode(sg.operators[rp]) == OP.RESHAPE and g.shape(sg.operators[rp].inputs[0]) == [1]:
          shared[vk] = sg.operators[rp].inputs[0]
        else:
          t1 = add_tensor(model, sg, f"{key}_retrofit_idx_part_{t}", TT.INT32, [1])
          new_ops_before[pk].append(make_op(model, OP.RESHAPE, [t, const_i32("retrofit_shape_1", [1])], [t1]))
          shared[vk] = t1
      parts.append(shared[vk])
    ti = add_tensor(model, sg, f"{key}_retrofit_indices_{which}", TT.INT32, [4])
    o = fb.ConcatenationOptionsT()
    o.axis = 0
    new_ops_before[pk].append(make_op(model, OP.CONCATENATION, parts, [ti], fb.BuiltinOptions.ConcatenationOptions, o))
    shared[ck] = ti
    out.append(ti)
  return out[0], out[1]


# ----------------------------------------------------------------------------------------------
# .litertlm
# ----------------------------------------------------------------------------------------------

def bundle_sections(path):
  from litert_lm_builder import litertlm_core
  from litert_lm_builder import litertlm_header_schema_py_generated as hs
  with open(path, "rb") as f:
    head = f.read(1 << 16)
    he = int.from_bytes(head[litertlm_core.HEADER_END_LOCATION_BYTE_OFFSET:litertlm_core.HEADER_END_LOCATION_BYTE_OFFSET + 8], "little")
    meta = hs.LiteRTLMMetaData.GetRootAs(bytearray(head[litertlm_core.HEADER_BEGIN_BYTE_OFFSET:he]), 0)
    out = []
    for i in range(meta.SectionMetadata().ObjectsLength()):
      so = meta.SectionMetadata().Objects(i)
      items = {}
      for j in range(so.ItemsLength()):
        it = so.Items(j)
        k = it.Key().decode() if it.Key() else None
        if it.ValueType() == hs.VData.StringValue:
          sv = hs.StringValue()
          sv.Init(it.Value().Bytes, it.Value().Pos)
          items[k] = sv.Value().decode() if sv.Value() else ""
        else:
          items[k] = f"<vtype {it.ValueType()}>"
      out.append({"type": litertlm_core.any_section_data_type_to_string(so.DataType()), "begin": so.BeginOffset(),
                  "end": so.EndOffset(), "bytes": so.EndOffset() - so.BeginOffset(), "items": items,
                  "sha256": sha256_range(path, so.BeginOffset(), so.EndOffset())})
  return out


def run_cli(args):
  proc = subprocess.run([LITERT_LM_CLI] + args, stdin=subprocess.DEVNULL, capture_output=True, text=True)
  text = proc.stdout + proc.stderr
  if proc.returncode != 0 or "Error" in text:
    raise RuntimeError(f"litert-lm {' '.join(args[:1])} failed: {text[-2000:]}")
  return text


def retrofit_bundle(src, dst, work, **kw):
  unpack = os.path.join(work, "unpack")
  if os.path.exists(unpack):
    shutil.rmtree(unpack)
  run_cli(["unpack", src, "--output-dir", unpack])
  toml_path = os.path.join(unpack, "model.toml")
  toml = open(toml_path).read()
  m = re.search(r'model_type = "(?:TF_LITE_)?(?i:prefill_decode)"\s*\nsection_type = "TFLiteModel"\s*\ndata_path = "([^"]+)"', toml)
  if not m:
    raise RuntimeError("no prefill_decode TFLiteModel section in model.toml")
  tfl = os.path.join(unpack, m.group(1))
  # 0.17.1 unpack can emit the legacy TF_LITE_* spelling its own pack refuses (scripts/set_activation_type.py)
  toml2 = re.sub(r'model_type = "TF_LITE_([A-Z_]+)"', lambda x: f'model_type = "{x.group(1).lower()}"', toml)
  if toml2 != toml:
    open(toml_path, "w").write(toml2)
  new_tfl = tfl + ".retrofit.tflite"
  report = retrofit_tflite(tfl, new_tfl, **kw)
  os.replace(new_tfl, tfl)
  if os.path.exists(dst):
    os.remove(dst)
  run_cli(["pack", toml_path, "--output", dst])
  if not os.path.exists(dst) or os.path.getsize(dst) == 0:
    raise RuntimeError(f"pack wrote nothing: {dst}")
  A, B = bundle_sections(src), bundle_sections(dst)
  if len(A) != len(B):
    raise RuntimeError("section count changed")
  secs = []
  for a, b in zip(A, B):
    is_pd = a["type"] == "TFLiteModel" and a["items"].get("model_type", "").lower().endswith("prefill_decode")
    row = {"type": a["type"], "items_src": a["items"], "items_dst": b["items"], "bytes_src": a["bytes"],
           "bytes_dst": b["bytes"], "sha256_src": a["sha256"], "sha256_dst": b["sha256"],
           "begin_src": a["begin"], "begin_dst": b["begin"], "replaced": is_pd}
    norm = lambda d: {k: (v.lower().replace("tf_lite_", "") if k == "model_type" else v) for k, v in d.items()}
    if norm(a["items"]) != norm(b["items"]):
      raise RuntimeError(f"section items changed: {a['items']} -> {b['items']}")
    if not is_pd and a["sha256"] != b["sha256"]:
      raise RuntimeError(f"section {a['type']} {a['items']} changed")
    secs.append(row)
  report["bundle_sections"] = secs
  return report


def retrofit_tflite(src, dst, decomp="exporter", indices="keep", rhs_cache_update=True, mask="select_float", dry_run=False):
  model, data_start, size = load_tflite(src)
  report = retrofit(model, src, decomp=decomp, indices=indices, rhs_cache_update=rhs_cache_update, mask=mask)
  report["src"] = {"path": src, "bytes": size}
  if dry_run:
    return report
  report["layout"] = write_tflite(model, src, data_start, size, dst)
  report["dst"] = {"path": dst, "bytes": os.path.getsize(dst)}
  report["buffer_identity"] = buffer_identity(src, dst)
  if not report["buffer_identity"]["ok"]:
    raise RuntimeError(f"buffer identity failed: {report['buffer_identity']}")
  return report


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("src")
  ap.add_argument("dst")
  ap.add_argument("--decomp", choices=["exporter", "v2"], default="exporter")
  ap.add_argument("--indices", choices=["keep", "concat"], default="keep")
  ap.add_argument("--no-rhs-cache-update", action="store_true")
  ap.add_argument("--mask", choices=["select_float", "select_bool", "add_bcast", "add"], default="select_float",
                  help="how the attention mask is applied after runtime_bmm (see the module docstring)")
  ap.add_argument("--report")
  ap.add_argument("--dry-run", action="store_true")
  ap.add_argument("--work", help="work dir for .litertlm unpack (default: next to dst)")
  args = ap.parse_args()
  kw = dict(decomp=args.decomp, indices=args.indices, rhs_cache_update=not args.no_rhs_cache_update, mask=args.mask)
  if args.src.endswith(".litertlm"):
    if args.dry_run:
      raise SystemExit("--dry-run is for .tflite input")
    work = args.work or (os.path.splitext(args.dst)[0] + "_work")
    os.makedirs(work, exist_ok=True)
    report = retrofit_bundle(args.src, args.dst, work, **kw)
    report["bundle"] = {"src": args.src, "dst": args.dst, "src_bytes": os.path.getsize(args.src),
                        "dst_bytes": os.path.getsize(args.dst)}
    shutil.rmtree(work, ignore_errors=True)
  else:
    report = retrofit_tflite(args.src, args.dst, dry_run=args.dry_run, **kw)
  text = json.dumps(report, indent=1, default=str)
  if args.report:
    with open(args.report, "w") as f:
      f.write(text + "\n")
  print(text[:4000])


if __name__ == "__main__":
  main()
