#!/usr/bin/env python3
"""gpu_rewrite.py: GPU composites and a shorter router for a published Qwen3-30B-A3B-family file, written in place.

  python gpu_rewrite.py SRC.litertlm DST.litertlm --rope-theta THETA [--src-sha256 HEX] [--report R.json]
  python gpu_rewrite.py SRC.litertlm --dry --rope-theta THETA [--report R.json]

SRC is a file made by the steps in this directory (either cache length): `qwen3_30b_a3b_mixed_int4_ekv4096.litertlm` or
`qwen3_30b_a3b_mixed_int4_ekv2048.litertlm` of litert-community/Qwen3-30B-A3B, or the two of
litert-community/Qwen3-Coder-30B-A3B-Instruct as first published. THETA is the source checkpoint's `rope_theta`
(Qwen/Qwen3-30B-A3B 1000000, Qwen/Qwen3-Coder-30B-A3B-Instruct 10000000); the script checks it against the graph's
inv_freq constant and stops when they differ.

Changes, in every prefill and decode signature:
  1. KV cache: `../tools/gpu_graph/gpu_graph_retrofit.py` (loaded by path, unchanged) with `--decomp exporter
     --indices keep --rhs-cache-update --mask add_bcast`: the two DYNAMIC_UPDATE_SLICE writes of a layer become one
     `odml.cache_update`, each attention BATCH_MATMUL becomes `odml.runtime_bmm`, and a `param_tensor` input is added.
     The writer reshapes each cache to rank 3 before its BATCH_MATMUL, and gpu_graph_retrofit.py only takes a
     BATCH_MATMUL that reads the cache write's output; on this graph alone it writes the cache composites and no
     `odml.runtime_bmm`, without an error. So a pre-pass (rank4_prepass) first makes each BATCH_MATMUL read the rank-4
     cache itself, and kv_pass stops when a layer whose cache has a reader is left without its runtime_bmm. The last
     layer of a prefill signature has no cache reader (prefill returns no logits) and keeps only its cache_update.
  2. `odml.rms_norm` for each hidden RMSNorm (MUL(x, x) -> MEAN -> ADD(eps) -> RSQRT -> MUL(x, .) -> MUL(., w)).
  3. `odml.qkv_norm_rope` in its 3-input form (x, position [1, T], weight) for each q and k head path: FC output ->
     RESHAPE [1, T, n, hd] -> RMSNorm over hd -> TRANSPOSE [0, 2, 1, 3] -> rotate-half RoPE (SLICE, SLICE, MUL -1,
     CONCATENATION, MUL cos, MUL sin, ADD); max_timescale = THETA, min_timescale 1. The cos / sin table ops that lose
     their readers are removed.
  4. The router: FC logits -> SOFTMAX -> TOPK_V2 -> SUM / DIV (norm_topk_prob) becomes TOPK_V2 on the logits ->
     SOFTMAX over the k values (the same k indices, since softmax keeps the order; the weights e^z_i / sum_topk e^z_j).
The decomposition of each composite of 2 and 3 is its original ops, shared by the composites of the same kind and
shape in a signature after a structural check (ops, options, constant bytes, shapes); the CPU runs the decompositions.
The router change (4) is in the graph itself, for every backend. The tensors and subgraphs this script adds carry the
tags `r3` (the pre-pass) and `r4` (changes 2 and 3) in their names.

Writing: nothing is unpacked. The script reads the TFLite section's structure (the flatbuffer before the external
weight region, about 16 MB), rewrites it in memory, and needs the new structure to fit before the weight region (the
Qwen3-30B-A3B files: 15,505,120 B of 15,579,264). DST = a copy of SRC (`cp -c`, an APFS clone, on macOS; `cp
--reflink=auto` elsewhere) with the new structure and zero padding written over [section begin, section begin +
data_start). Then the asserts, read back from disk: the same size, the same bytes before the TFLite section (header,
metadata, tokenizer), the same weight region (sha256 of both files, one read each), the same section table, and every
buffer of SRC at the same index of DST with the same inline bytes or the same offset and size. The int4 experts'
blockwise scale references are checked after the rewrite (blockwise_check). One source per process:
gpu_graph_retrofit.py caches shape constants by id(subgraph).

Needs Python 3.12 with numpy, flatbuffers, ai-edge-litert (the TFLite schema) and litert-lm-builder (the container
header); run with 2.5.4, 25.12.19, 2.3.0 and 0.19.0.
"""
import argparse
import collections
import copy
import hashlib
import importlib.util
import json
import mmap
import os
import subprocess
import sys
import time

import numpy as np

PUBLIC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "gpu_graph",
                      "gpu_graph_retrofit.py")


def load_public(path):
  spec = importlib.util.spec_from_file_location("gpu_graph_retrofit_public", path)
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


G = load_public(PUBLIC)
fb, OP, TT = G.fb, G.OP, G.TT
BLOCKWISE = fb.QuantizationDetails.BlockwiseQuantization
BN = G.BN
PURE = {OP.RESHAPE, OP.MUL, OP.ADD, OP.CONCATENATION, OP.COS, OP.SIN, OP.CAST, OP.SLICE, OP.STRIDED_SLICE,
        OP.TRANSPOSE, OP.MEAN, OP.RSQRT}
NP = {TT.FLOAT32: np.float32, TT.INT32: np.int32, TT.INT64: np.int64, TT.FLOAT16: np.float16, TT.BOOL: np.bool_,
      TT.INT8: np.int8, TT.UINT8: np.uint8, TT.INT16: np.int16}
KV_ARGS = dict(decomp="exporter", indices="keep", rhs_cache_update=True, mask="add_bcast")


def file_sha256(path):
  h = hashlib.sha256()
  with open(path, "rb") as f:
    for c in iter(lambda: f.read(1 << 24), b""):
      h.update(c)
  return h.hexdigest()


# ----------------------------------------------------------------------------------------------------------------------
# reading the bundle


def header_sections(path):
  from litert_lm_builder import litertlm_core
  from litert_lm_builder import litertlm_header_schema_py_generated as hs
  with open(path, "rb") as f:
    head = f.read(1 << 16)
  he = int.from_bytes(head[litertlm_core.HEADER_END_LOCATION_BYTE_OFFSET:litertlm_core.HEADER_END_LOCATION_BYTE_OFFSET + 8],
                      "little")
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
                "end": so.EndOffset(), "items": items})
  return out


class Bundle:
  """The prefill_decode TFLite section of a .litertlm, its flatbuffer as a ModelT (external buffers not read)."""

  def __init__(self, path):
    self.path = path
    self.size = os.path.getsize(path)
    self.secs = header_sections(path)
    pd = [s for s in self.secs if s["type"] == "TFLiteModel" and
          s["items"].get("model_type", "").lower().endswith("prefill_decode")]
    if len(pd) != 1:
      raise RuntimeError(f"{len(pd)} prefill_decode TFLite sections")
    self.begin, self.end = pd[0]["begin"], pd[0]["end"]
    if self.end != max(s["end"] for s in self.secs):
      raise RuntimeError("the TFLite section is not the last section")
    with open(path, "rb") as f:
      mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
      m = fb.Model.GetRootAsModel(mm, self.begin)
      self.data_start, self.n_ext = G.external_region(m)
      region = bytes(mm[self.begin: self.begin + self.data_start])
      mm.close()
    self.model = fb.ModelT.InitFromPackedBuf(bytearray(region), 0)
    for b in self.model.buffers:
      if b.data is not None:
        b.data = np.array(b.data, dtype=np.uint8)
    for sg in self.model.subgraphs:
      sg.inputs = [int(x) for x in sg.inputs] if sg.inputs is not None else []
      sg.outputs = [int(x) for x in sg.outputs] if sg.outputs is not None else []
      for op in sg.operators or []:
        op.inputs = [int(x) for x in op.inputs] if op.inputs is not None else []
        op.outputs = [int(x) for x in op.outputs] if op.outputs is not None else []
        if op.intermediates is not None and len(op.intermediates):
          raise RuntimeError("an op with intermediates")

  def buffer_bytes(self, bi):
    b = self.model.buffers[bi]
    if b.data is not None and len(b.data):
      return bytes(b.data)
    if b.offset is not None and b.offset > 1:
      with open(self.path, "rb") as f:
        f.seek(self.begin + b.offset)
        return f.read(b.size)
    return b""


def normalize(model):
  """As Bundle.__init__ leaves a freshly read model."""
  for b in model.buffers:
    if b.data is not None:
      b.data = np.array(b.data, dtype=np.uint8)
  for sg in model.subgraphs:
    sg.inputs = [int(x) for x in sg.inputs] if sg.inputs is not None else []
    sg.outputs = [int(x) for x in sg.outputs] if sg.outputs is not None else []
    for op in sg.operators or []:
      op.inputs = [int(x) for x in op.inputs] if op.inputs is not None else []
      op.outputs = [int(x) for x in op.outputs] if op.outputs is not None else []
      if op.intermediates is not None and len(op.intermediates):
        raise RuntimeError("an op with intermediates")


def ext_buffers(model):
  return [b for b in model.buffers if b.offset is not None and b.offset > 1]


# ----------------------------------------------------------------------------------------------------------------------
# 1. the KV cache: the rank-4 pre-pass, then gpu_graph_retrofit.py


def set_shape(t, shape):
  t.shape = list(shape)
  if t.shapeSignature is not None and len(t.shapeSignature):
    t.shapeSignature = list(shape)


def rank4_prepass(model):
  """In place: the BATCH_MATMULs that read a cache through a rank-3 RESHAPE read the rank-4 cache itself."""
  rep = {"signatures": {}}
  for sd in model.signatureDefs:
    key = G.name_of(sd.signatureKey)
    if not (key.startswith("prefill") or key == "decode"):
      continue
    sg = model.subgraphs[sd.subgraphIndex]
    g = G.Graph(model, sg)
    kv_in = {}
    for tm in sd.inputs:
      mt = G.KV_RX.match(G.name_of(tm.name))
      if mt:
        kv_in[tm.tensorIndex] = (mt.group(1), int(mt.group(2)))
    outs = set(sg.outputs)
    shapes = {}

    def shape_const(vals):
      k = tuple(vals)
      if k not in shapes:
        shapes[k] = G.add_tensor(model, sg, f"{key}_r3_rank4_shape_{'x'.join(map(str, vals))}", TT.INT32,
                                 [len(vals)], np.array(vals, np.int32))
      return shapes[k]

    remove = set()
    new_before = collections.defaultdict(list)
    stat = collections.Counter()
    layers = collections.defaultdict(set)
    shapes_seen = collections.Counter()
    for oi, op in enumerate(sg.operators):
      if g.opcode(op) != OP.DYNAMIC_UPDATE_SLICE:
        continue
      kk = kv_in.get(op.inputs[0])
      if kk is None:
        stat["dus_not_on_a_cache"] += 1
        continue
      kind, li = kk
      where = f"{key} layer {li} {kind}"
      cache = op.outputs[0]
      c4 = g.shape(cache)
      if len(c4) != 4 or c4[0] != 1:
        raise RuntimeError(f"{where}: cache shape {c4}")
      cons = sorted(set(g.consumers.get(cache, [])))
      if not cons:
        # prefill: the last layer's attention reads nothing the signature returns (no logits), so the graph has no op
        # after its KV write (that cache is a signature output only); gpu_graph_retrofit.py then writes its
        # odml.cache_update without odml.runtime_bmm, which is right (kv_pass checks the count)
        stat[f"cache_unread_{kind}"] += 1
        continue
      if len(cons) != 1 or g.opcode(sg.operators[cons[0]]) != OP.RESHAPE:
        raise RuntimeError(f"{where}: the cache is read by {[G.BN.get(g.opcode(sg.operators[c])) for c in cons]}, "
                           "not by one RESHAPE")
      ri = cons[0]
      c3 = sg.operators[ri].outputs[0]
      if g.shape(c3) != c4[1:] or c3 in outs:
        raise RuntimeError(f"{where}: cache RESHAPE writes {g.shape(c3)} (not {c4[1:]}) or is a graph output")
      bcons = sorted(set(g.consumers.get(c3, [])))
      if len(bcons) != 1:
        raise RuntimeError(f"{where}: the rank-3 cache has {len(bcons)} readers")
      bi = bcons[0]
      bop = sg.operators[bi]
      if g.opcode(bop) != OP.BATCH_MATMUL or len(bop.inputs) != 2 or bop.inputs[1] != c3:
        raise RuntimeError(f"{where}: the rank-3 cache is read by {G.BN.get(g.opcode(bop))} (input slot "
                           f"{list(bop.inputs).index(c3)}), not as a BATCH_MATMUL's rhs")
      o = bop.builtinOptions
      if o is None or not o.adjY or o.adjX:
        raise RuntimeError(f"{where}: BATCH_MATMUL {bi} is not (adjX=False, adjY=True)")
      a3 = bop.inputs[0]
      sa = g.shape(a3)
      if len(sa) != 3 or sa[0] != c4[1] or sa[2] != c4[3]:
        raise RuntimeError(f"{where}: lhs {sa} does not match the cache {c4}")
      out3 = bop.outputs[0]
      so = g.shape(out3)
      if so != [c4[1], sa[1], c4[2]] or out3 in outs:
        raise RuntimeError(f"{where}: BATCH_MATMUL output {so}, expected {[c4[1], sa[1], c4[2]]}")
      ocons = sorted(set(g.consumers.get(out3, [])))
      if len(ocons) != 1 or g.opcode(sg.operators[ocons[0]]) != OP.RESHAPE or sg.operators[ocons[0]].inputs[0] != out3:
        raise RuntimeError(f"{where}: the BATCH_MATMUL output is read by "
                           f"{[G.BN.get(g.opcode(sg.operators[c])) for c in ocons]}, not by one RESHAPE")
      a4shape = [1] + sa
      pa = g.producer.get(a3)
      acons = sorted(set(g.consumers.get(a3, [])))
      if pa is not None and g.opcode(sg.operators[pa]) == OP.RESHAPE and acons == [bi] and a3 not in outs:
        pop = sg.operators[pa]
        if len(pop.inputs) >= 2:
          pop.inputs = [pop.inputs[0], shape_const(a4shape)] + list(pop.inputs[2:])
        opt = pop.builtinOptions
        if opt is not None and getattr(opt, "newShape", None) is not None and len(opt.newShape):
          opt.newShape = list(a4shape)
        elif len(pop.inputs) < 2:
          raise RuntimeError(f"{where}: lhs RESHAPE {pa} has neither a shape input nor newShape")
        set_shape(sg.tensors[a3], a4shape)
        a4 = a3
        stat["lhs_reshape_retargeted"] += 1
      else:
        a4 = G.add_tensor(model, sg, f"{key}_r3_rank4_lhs_{bi}", sg.tensors[a3].type, a4shape)
        new_before[bi].append(G.make_op(model, OP.RESHAPE, [a3, shape_const(a4shape)], [a4]))
        stat["lhs_reshape_inserted"] += 1
      bop.inputs = [a4, cache]
      set_shape(sg.tensors[out3], [1] + so)
      remove.add(ri)
      stat["bmm_rank4"] += 1
      layers[kind].add(li)
      shapes_seen[f"{kind}: lhs {a4shape} rhs {c4} out {[1] + so}"] += 1
    ops = []
    for oi, op in enumerate(sg.operators):
      ops.extend(new_before.get(oi, []))
      if oi in remove:
        continue
      ops.append(op)
    sg.operators = ops
    ok, bad = G.check_topo(model, sg)
    if not ok:
      raise RuntimeError(f"{key}: not topologically ordered after the pre-pass at {bad}")
    rep["signatures"][key] = {"layers_k": len(layers["k"]), "layers_v": len(layers["v"]),
                              "cache_reshapes_dropped": len(remove), "shape_constants_added": len(shapes),
                              "shapes": dict(shapes_seen), **dict(stat)}
  return rep


def kv_pass(bundle, rep):
  """The rank-4 pre-pass + gpu_graph_retrofit.retrofit (KV_ARGS), in memory; then pack -> reload into bundle.model."""
  path, begin = bundle.path, bundle.begin

  def buffer_bytes(model_, bi, src_path):  # an external buffer lives in the bundle at the section offset
    b = model_.buffers[bi]
    if b.data is not None and len(b.data):
      return bytes(b.data)
    if b.offset is not None and b.offset > 1:
      with open(path, "rb") as f:
        f.seek(begin + b.offset)
        return f.read(b.size)
    return b""

  G.buffer_bytes = buffer_bytes
  before = census(bundle.model)
  pre = rank4_prepass(bundle.model)
  kv = G.retrofit(bundle.model, path, **KV_ARGS)
  for key, s in kv["signatures"].items():
    unread = pre["signatures"].get(key, {}).get("cache_unread_k", 0)
    bpl = s["bmm_per_layer"]
    if set(bpl) - {"0", "2"} or bpl.get("0", 0) != unread:
      raise RuntimeError(f"{key}: attention BATCH_MATMULs taken per layer {bpl}, layers whose cache has no reader "
                         f"{unread}: a layer with a cache reader was left without odml.runtime_bmm")
  packed = G.pack_model(bundle.model)
  m = fb.ModelT.InitFromPackedBuf(bytearray(packed), 0)
  normalize(m)
  bundle.model = m
  rep["kv"] = {"args": KV_ARGS, "rank4_prepass": pre, "census_before": before, "census_after": census(m),
               "structure_bytes": len(packed), "report": kv}
  return len(packed)


# ----------------------------------------------------------------------------------------------------------------------
# 2-4. odml.rms_norm, odml.qkv_norm_rope, the router


def code_of(model, op):
  c = model.operatorCodes[op.opcodeIndex]
  return max(c.builtinCode, c.deprecatedBuiltinCode)


class View:
  def __init__(self, bundle, sg):
    self.b, self.model, self.sg = bundle, bundle.model, sg
    self.refresh()

  def refresh(self):
    self.producer, self.consumers = {}, collections.defaultdict(list)
    for oi, op in enumerate(self.sg.operators):
      for t in op.outputs:
        self.producer[t] = oi
      for t in op.inputs:
        if t >= 0:
          self.consumers[t].append(oi)

  def code(self, oi):
    return code_of(self.model, self.sg.operators[oi])

  def name(self, oi):
    c = self.code(oi)
    if c == OP.STABLEHLO_COMPOSITE:
      return "COMPOSITE:" + G.name_of(self.sg.operators[oi].builtinOptions2.name)
    if c == OP.CUSTOM:
      return "CUSTOM:" + G.name_of(self.model.operatorCodes[self.sg.operators[oi].opcodeIndex].customCode)
    return BN.get(c, str(c))

  def shape(self, t):
    return [int(x) for x in self.sg.tensors[t].shape]

  def cons(self, t):
    return sorted(set(self.consumers.get(t, [])))

  def prod(self, t):
    return self.producer.get(t)

  def is_const(self, t):
    return G.is_const(self.model, self.sg, t)

  def const(self, t):
    tt = self.sg.tensors[t]
    if tt.type not in NP:
      return None  # packed or quantized types (int4 weights): not read here
    raw = self.b.buffer_bytes(tt.buffer)
    return np.frombuffer(raw, dtype=NP[tt.type]).reshape([int(x) for x in tt.shape]) if raw else None

  def op(self, oi):
    return self.sg.operators[oi]


def other_input(op, t):
  ins = list(op.inputs)
  if len(ins) != 2 or t not in ins:
    return None
  return ins[1] if ins[0] == t else ins[0]


def canon(x):
  """A comparable form of an options object (or value)."""
  if x is None or isinstance(x, (bool, int, float, str, bytes)):
    return x
  if isinstance(x, np.ndarray):
    return ("nd", str(x.dtype), tuple(x.shape), x.tobytes())
  if isinstance(x, (list, tuple)):
    return tuple(canon(v) for v in x)
  if hasattr(x, "__dict__"):
    return (type(x).__name__, tuple(sorted((k, canon(v)) for k, v in vars(x).items())))
  return repr(x)


class NoMatch(Exception):
  pass


def need(cond, why):
  if not cond:
    raise NoMatch(why)


def match_rmsnorm(v, r):
  """RSQRT op r -> dict(x, w, out, ops, eps, axis) of x * rsqrt(mean(x*x) + eps) * w."""
  outs = set(v.sg.outputs)
  rop = v.op(r)
  ae = rop.inputs[0]
  a = v.prod(ae)
  need(a is not None and v.code(a) == OP.ADD, "rsqrt input not ADD")
  aop = v.op(a)
  ms = next((t for t in aop.inputs if not v.is_const(t)), None)
  eps_t = next((t for t in aop.inputs if v.is_const(t)), None)
  need(ms is not None and eps_t is not None, "ADD not (x, const)")
  eps = v.const(eps_t)
  need(eps is not None and eps.size == 1, "eps not a scalar constant")
  m = v.prod(ms)
  need(m is not None and v.code(m) == OP.MEAN, "not MEAN")
  mop = v.op(m)
  sq, ax_t = mop.inputs[0], mop.inputs[1]
  need(v.is_const(ax_t), "MEAN axis not constant")
  s = v.prod(sq)
  need(s is not None and v.code(s) == OP.MUL, "not MUL(x, x)")
  sop = v.op(s)
  need(len(sop.inputs) == 2 and sop.inputs[0] == sop.inputs[1], "MUL inputs differ")
  x = sop.inputs[0]
  rank = len(v.shape(x))
  ax = v.const(ax_t).reshape(-1).tolist()
  need(ax in ([rank - 1], [-1]), f"MEAN axis {ax}")
  need(mop.builtinOptions is not None and mop.builtinOptions.keepDims, "MEAN keep_dims false")
  rs = rop.outputs[0]
  c1 = v.cons(rs)
  need(len(c1) == 1 and v.code(c1[0]) == OP.MUL and other_input(v.op(c1[0]), rs) == x, "rsqrt not into MUL(x, .)")
  p1 = c1[0]
  xr = v.op(p1).outputs[0]
  c2 = v.cons(xr)
  need(len(c2) == 1 and v.code(c2[0]) == OP.MUL, "MUL(x, rs) not into one MUL")
  p2 = c2[0]
  w = other_input(v.op(p2), xr)
  need(w is not None and v.is_const(w) and v.shape(w) == [v.shape(x)[-1]], "weight not a constant [D]")
  for t, c in ((sq, m), (ms, a), (ae, r)):
    need(v.cons(t) == [c] and t not in outs, "an intermediate with another reader")
  need(xr not in outs and rs not in outs, "an intermediate is an output")
  for oi in (s, m, a, r, p1, p2):
    o = v.op(oi).builtinOptions
    if o is not None and hasattr(o, "fusedActivationFunction"):
      need(o.fusedActivationFunction == 0, "a fused activation")
  return {"x": x, "w": w, "out": v.op(p2).outputs[0], "ops": [s, m, a, r, p1, p2], "eps": float(eps.reshape(-1)[0]),
          "rank": rank}


def match_head(v, nm):
  """A head RMSNorm match nm (x rank 4) -> the q / k chain FC out -> ... -> RoPE out, or NoMatch."""
  outs = set(v.sg.outputs)
  x = nm["x"]
  rsh = v.prod(x)
  need(rsh is not None and v.code(rsh) == OP.RESHAPE, "head norm input not RESHAPE")
  fc_out = v.op(rsh).inputs[0]
  sx = v.shape(x)
  sf = v.shape(fc_out)
  need(len(sx) == 4 and sx[0] == 1 and len(sf) == 3 and sf == [1, sx[1], sx[2] * sx[3]], f"reshape {sf} -> {sx}")
  T, n, hd = sx[1], sx[2], sx[3]
  need(sorted(v.cons(x)) == sorted([nm["ops"][0], nm["ops"][4]]) and x not in outs, "head x has other readers")
  c = v.cons(nm["out"])
  need(len(c) == 1 and v.code(c[0]) == OP.TRANSPOSE, "norm out not into one TRANSPOSE")
  tr = c[0]
  perm = v.const(v.op(tr).inputs[1])
  need(perm is not None and perm.reshape(-1).tolist() == [0, 2, 1, 3], "transpose perm")
  xt = v.op(tr).outputs[0]
  need(v.shape(xt) == [1, n, T, hd] and xt not in outs, "transpose out shape")
  cs = v.cons(xt)
  need(len(cs) == 3, f"rope input read by {len(cs)} ops")
  half = hd // 2
  sl_lo = sl_hi = mcos = None
  for oi in cs:
    cc = v.code(oi)
    if cc in (OP.SLICE, OP.STRIDED_SLICE):
      begin = v.const(v.op(oi).inputs[1]).reshape(-1).tolist()
      need(begin[:3] == [0, 0, 0] and begin[3] in (0, half), f"slice begin {begin}")
      need(v.shape(v.op(oi).outputs[0]) == [1, n, T, half], "slice shape")
      if begin[3] == 0:
        sl_lo = oi
      else:
        sl_hi = oi
    elif cc == OP.MUL:
      mcos = oi
  need(sl_lo is not None and sl_hi is not None and mcos is not None, "rope readers")
  cos_t = other_input(v.op(mcos), xt)
  need(cos_t is not None and not v.is_const(cos_t) and v.shape(cos_t) == [1, 1, T, hd], "cos")
  x1 = v.op(sl_lo).outputs[0]
  x2 = v.op(sl_hi).outputs[0]
  c2 = v.cons(x2)
  need(len(c2) == 1 and v.code(c2[0]) == OP.MUL, "x2 not into MUL(-1)")
  mneg = c2[0]
  neg_t = other_input(v.op(mneg), x2)
  need(neg_t is not None and v.is_const(neg_t) and v.const(neg_t).reshape(-1).tolist() == [-1.0], "MUL -1")
  nx2 = v.op(mneg).outputs[0]
  c3 = v.cons(nx2)
  need(len(c3) == 1 and v.code(c3[0]) == OP.CONCATENATION and v.cons(x1) == c3, "concat")
  cat = c3[0]
  need(list(v.op(cat).inputs) == [nx2, x1] and v.op(cat).builtinOptions.axis in (3, -1), "concat order / axis")
  rot = v.op(cat).outputs[0]
  c4 = v.cons(rot)
  need(len(c4) == 1 and v.code(c4[0]) == OP.MUL, "rot not into MUL(sin)")
  msin = c4[0]
  sin_t = other_input(v.op(msin), rot)
  need(sin_t is not None and not v.is_const(sin_t) and v.shape(sin_t) == [1, 1, T, hd], "sin")
  a_t, b_t = v.op(mcos).outputs[0], v.op(msin).outputs[0]
  ca, cb = v.cons(a_t), v.cons(b_t)
  need(len(ca) == 1 and ca == cb and v.code(ca[0]) == OP.ADD, "the two rope products not into one ADD")
  add = ca[0]
  out = v.op(add).outputs[0]
  need(v.shape(out) == [1, n, T, hd], "rope out shape")
  for t in (x1, x2, nx2, rot, a_t, b_t):
    need(t not in outs, "an intermediate is an output")
  for oi in (mneg, mcos, msin, add, cat):
    o = v.op(oi).builtinOptions
    if o is not None and hasattr(o, "fusedActivationFunction"):
      need(o.fusedActivationFunction == 0, "a fused activation")
  ops = [rsh] + nm["ops"] + [tr, sl_lo, sl_hi, mneg, cat, mcos, msin, add]
  return {"fc_out": fc_out, "w": nm["w"], "out": out, "ops": sorted(ops), "last": add, "T": T, "n": n, "hd": hd,
          "eps": nm["eps"], "cos": cos_t, "sin": sin_t}


def table_ops(v, roots, stop):
  """Ops that produce `roots` back to the tensor `stop` (exclusive), constants as leaves."""
  ops, todo = set(), list(roots)
  while todo:
    t = todo.pop()
    if t == stop or t < 0 or v.is_const(t):
      continue
    p = v.prod(t)
    need(p is not None, f"rope table tensor {t} has no producer and is not {stop}")
    need(v.code(p) in PURE, f"rope table op {BN.get(v.code(p))} is not pure")
    if p in ops:
      continue
    ops.add(p)
    todo.extend(x for x in v.op(p).inputs if x >= 0)
  return sorted(ops)


def chain_signature(v, op_idxs, boundary):
  """Structure of a chain: ops, options, constant bytes, shapes; boundary tensors by position."""
  where = {t: ("in", k) for k, t in enumerate(boundary)}
  desc = []
  for j, oi in enumerate(op_idxs):
    op = v.op(oi)
    ins = []
    for t in op.inputs:
      if t < 0:
        ins.append(("none",))
      elif t in where:
        ins.append(where[t])
      elif v.is_const(t):
        tt = v.sg.tensors[t]
        ins.append(("c", tt.type, tuple(v.shape(t)), hashlib.sha256(v.b.buffer_bytes(tt.buffer)).hexdigest()))
      else:
        raise NoMatch(f"chain input {t} is neither a boundary, a constant nor produced earlier in the chain")
    for k, t in enumerate(op.outputs):
      where[t] = ("v", j, k)
    desc.append((v.code(oi), canon(op.builtinOptions), canon(op.builtinOptions2), tuple(ins),
                 tuple(tuple(v.shape(t)) for t in op.outputs), tuple(v.sg.tensors[t].type for t in op.outputs)))
  return tuple(desc)


def extract(v, op_idxs, inputs, outputs, name):
  """A new SubGraphT running ops op_idxs (copied) on the subgraph inputs `inputs` [(main tensor, shape or None)]."""
  sg = v.sg
  d = fb.SubGraphT()
  d.tensors, d.operators, d.name = [], [], name
  tmap = {}

  def new_t(t, shape=None, const=False):
    src = sg.tensors[t]
    if blockwise_refs(src):
      raise RuntimeError(f"{name}: tensor {t} has blockwise quantization details (not carried into a decomposition)")
    nt = fb.TensorT()
    nt.name = f"{name}/t{len(d.tensors)}"
    nt.type = src.type
    shp = list(shape) if shape is not None else [int(x) for x in src.shape]
    nt.shape = shp
    if shape is None and src.shapeSignature is not None and len(src.shapeSignature):
      nt.shapeSignature = [int(x) for x in src.shapeSignature]
    else:
      nt.shapeSignature = list(shp)
    nt.buffer = src.buffer if const else 0
    nt.hasRank = True
    nt.quantization = copy.deepcopy(src.quantization) if src.quantization is not None else fb.QuantizationParametersT()
    d.tensors.append(nt)
    return len(d.tensors) - 1

  for t, shp in inputs:
    tmap[t] = new_t(t, shp)
  d.inputs = [tmap[t] for t, _ in inputs]
  for oi in op_idxs:
    op = v.op(oi)
    nop = copy.deepcopy(op)
    ins = []
    for t in op.inputs:
      if t < 0:
        ins.append(t)
        continue
      if t not in tmap:
        if not v.is_const(t):
          raise RuntimeError(f"{name}: op {oi} reads tensor {t} from outside the chain")
        tmap[t] = new_t(t, const=True)
      ins.append(tmap[t])
    outs = []
    for t in op.outputs:
      tmap[t] = new_t(t)
      outs.append(tmap[t])
    nop.inputs, nop.outputs = ins, outs
    d.operators.append(nop)
  d.outputs = [tmap[t] for t in outputs]
  return d


def dead_ops(v):
  """Remove pure ops whose outputs nothing reads (iteratively). -> Counter of removed op types."""
  sg = v.sg
  removed = collections.Counter()
  while True:
    used = set(sg.outputs)
    for op in sg.operators:
      used.update(t for t in op.inputs if t >= 0)
    keep, gone = [], []
    for op in sg.operators:
      if code_of(v.model, op) in PURE and not any(t in used for t in op.outputs):
        gone.append(op)
      else:
        keep.append(op)
    if not gone:
      break
    for op in gone:
      removed[BN.get(code_of(v.model, op))] += 1
    sg.operators = keep
  v.refresh()
  return removed


def blockwise_refs(t):
  """Tensor indices held by a tensor's quantization details (BlockwiseQuantization scales / zeroPoints; -1 = none)."""
  q = t.quantization
  if q is None or q.detailsType != BLOCKWISE or q.details is None:
    return []
  return [int(x) for x in (q.details.scales, q.details.zeroPoints) if x is not None and int(x) >= 0]


def compact(model, si):
  """Drop the tensors of subgraph si that nothing references; remap ops (inputs, outputs, intermediates), subgraph I/O,
  SignatureDefs and the tensor indices inside quantization details (BlockwiseQuantization scales / zeroPoints: the int4
  experts' scales; left at the old indices, the runtime refuses the file: `Invalid scales tensor index in blockwise
  quantization`)."""
  sg = model.subgraphs[si]
  live = set(sg.inputs) | set(sg.outputs)
  for op in sg.operators:
    live.update(t for t in op.inputs if t >= 0)
    live.update(t for t in op.outputs if t >= 0)
    if op.intermediates is not None:
      live.update(int(t) for t in op.intermediates if t >= 0)
  sds = [sd for sd in model.signatureDefs if sd.subgraphIndex == si]
  for sd in sds:
    live.update(tm.tensorIndex for tm in list(sd.inputs) + list(sd.outputs))
  todo = list(live)
  while todo:  # a live tensor keeps the tensors its quantization details name
    for r in blockwise_refs(sg.tensors[todo.pop()]):
      if r not in live:
        live.add(r)
        todo.append(r)
  order = sorted(live)
  remap = {o: n for n, o in enumerate(order)}
  before = len(sg.tensors)
  sg.tensors = [sg.tensors[o] for o in order]
  for op in sg.operators:
    op.inputs = [remap[t] if t >= 0 else t for t in op.inputs]
    op.outputs = [remap[t] if t >= 0 else t for t in op.outputs]
    if op.intermediates is not None and len(op.intermediates):
      op.intermediates = [remap[int(t)] if t >= 0 else int(t) for t in op.intermediates]
  for t in sg.tensors:
    q = t.quantization
    if q is not None and q.detailsType == BLOCKWISE and q.details is not None:
      if q.details.scales is not None and int(q.details.scales) >= 0:
        q.details.scales = remap[int(q.details.scales)]
      if q.details.zeroPoints is not None and int(q.details.zeroPoints) >= 0:
        q.details.zeroPoints = remap[int(q.details.zeroPoints)]
  sg.inputs = [remap[t] for t in sg.inputs]
  sg.outputs = [remap[t] for t in sg.outputs]
  for sd in sds:
    for tm in list(sd.inputs) + list(sd.outputs):
      tm.tensorIndex = remap[tm.tensorIndex]
  return before, len(sg.tensors)


def blockwise_snapshot(model):
  """{signature key: {blockwise tensor name: (scales tensor name, scales buffer, zeroPoints name or None)}} before the
  rewrite = what blockwise_check compares the rewritten model with (by names, which the rewrite never changes)."""
  out = {}
  for sd in model.signatureDefs:
    sg = model.subgraphs[sd.subgraphIndex]
    m = {}
    for t in sg.tensors:
      q = t.quantization
      if q is not None and q.detailsType == BLOCKWISE and q.details is not None:
        s, z = int(q.details.scales), (int(q.details.zeroPoints) if q.details.zeroPoints is not None else -1)
        m[G.name_of(t.name)] = (G.name_of(sg.tensors[s].name), int(sg.tensors[s].buffer),
                                G.name_of(sg.tensors[z].name) if z >= 0 else None)
    out[G.name_of(sd.signatureKey)] = m
  return out


def blockwise_check(model, snap):
  """-> {key: {"blockwise": n, "same": n, "wrong": [...]}}: every blockwise tensor of the snapshot still present, its
  details pointing at a tensor of the same name and buffer; no tensor index anywhere out of range."""
  out = {}
  for sd in model.signatureDefs:
    key = G.name_of(sd.signatureKey)
    sg = model.subgraphs[sd.subgraphIndex]
    n = len(sg.tensors)
    now = {}
    wrong = []
    for t in sg.tensors:
      q = t.quantization
      if q is not None and q.detailsType == BLOCKWISE and q.details is not None:
        s, z = int(q.details.scales), (int(q.details.zeroPoints) if q.details.zeroPoints is not None else -1)
        if not 0 <= s < n or z >= n:
          wrong.append((G.name_of(t.name), f"index out of range: scales {s}, zeroPoints {z}, tensors {n}"))
          continue
        now[G.name_of(t.name)] = (G.name_of(sg.tensors[s].name), int(sg.tensors[s].buffer),
                                  G.name_of(sg.tensors[z].name) if z >= 0 else None)
    for nm, ref in snap.get(key, {}).items():
      if now.get(nm) != ref:
        wrong.append((nm, f"{ref} -> {now.get(nm)}"))
    for op_i, op in enumerate(sg.operators):
      for t in list(op.inputs) + list(op.outputs) + list(op.intermediates if op.intermediates is not None else []):
        if t >= n or t < -1:
          wrong.append((f"op {op_i}", f"tensor index {t} of {n}"))
    out[key] = {"blockwise": len(snap.get(key, {})), "same": sum(1 for nm, ref in snap.get(key, {}).items()
                                                                  if now.get(nm) == ref), "wrong": wrong[:10],
                "wrong_count": len(wrong)}
  return out


def census(model):
  out = {}
  for sd in model.signatureDefs:
    key = G.name_of(sd.signatureKey)
    sg = model.subgraphs[sd.subgraphIndex]
    c = collections.Counter()
    for op in sg.operators:
      code = code_of(model, op)
      nm = BN.get(code, str(code))
      if code == OP.STABLEHLO_COMPOSITE and op.builtinOptions2 is not None:
        nm = "STABLEHLO_COMPOSITE:" + G.name_of(op.builtinOptions2.name)
      elif code == OP.CUSTOM:
        nm = "CUSTOM:" + G.name_of(model.operatorCodes[op.opcodeIndex].customCode)
      c[nm] += 1
    out[key] = {"ops": len(sg.operators), "tensors": len(sg.tensors), "by_type": dict(sorted(c.items()))}
  return out


def theta_of(inv_row, hd):
  """rope_theta from the writer's inv_freq constant [1, hd/2]: inv[j] = theta^(-2j/hd)."""
  inv = np.asarray(inv_row, np.float64).reshape(-1)
  est = [inv[j] ** (-hd / (2.0 * j)) for j in range(1, len(inv))]
  return float(np.median(est)), float(np.max(np.abs(np.array(est) / np.median(est) - 1)))


def scalar_or_vec(v, t):
  """The int values of a small constant, scalar constants included (shape None)."""
  tt = v.sg.tensors[t]
  return np.frombuffer(v.b.buffer_bytes(tt.buffer), dtype=NP[tt.type]).reshape(-1).tolist()


def match_router(v, oi):
  """Router at TOPK_V2 op oi: FC logits [.., E] -> SOFTMAX (beta 1) -> TOPK_V2(k) -> values read by exactly SUM(values,
  last axis, keepDims) and DIV(values, sum); the SUM read only by the DIV (the writer's `norm_topk_prob` routing).
  The rewrite is TOPK_V2(logits) -> SOFTMAX(values): p_i / sum_topk p_j =
  e^z_i / sum_topk e^z_j, and softmax keeps the order, so the k indices are the same."""
  topk = v.op(oi)
  sm = v.prod(topk.inputs[0])
  need(sm is not None and v.code(sm) == OP.SOFTMAX, "the TOPK input is not a SOFTMAX output")
  need(v.cons(topk.inputs[0]) == [oi], "the SOFTMAX output has other readers")
  smop = v.op(sm)
  need(smop.builtinOptions is not None and abs(float(smop.builtinOptions.beta) - 1.0) < 1e-12, "SOFTMAX beta != 1")
  logits = smop.inputs[0]
  p = v.prod(logits)
  need(p is not None and v.code(p) == OP.FULLY_CONNECTED, "the SOFTMAX input is not an FC output")
  values = topk.outputs[0]
  readers = sorted(v.cons(values))
  sums = [r for r in readers if v.code(r) == OP.SUM]
  divs = [r for r in readers if v.code(r) == OP.DIV]
  need(len(readers) == 2 and len(sums) == 1 and len(divs) == 1, f"TOPK values read by {[v.name(r) for r in readers]}")
  su, dv = sums[0], divs[0]
  suop, dvop = v.op(su), v.op(dv)
  rank = len(v.shape(values))
  need(suop.inputs[0] == values and suop.builtinOptions is not None and suop.builtinOptions.keepDims, "SUM form")
  need(scalar_or_vec(v, suop.inputs[1]) in ([rank - 1], [-1]), "SUM axis is not the last")
  need(v.cons(suop.outputs[0]) == [dv], "the SUM output has other readers")
  need(list(dvop.inputs) == [values, suop.outputs[0]], "DIV operands")
  need(dvop.builtinOptions is None or dvop.builtinOptions.fusedActivationFunction == 0, "DIV activation")
  need(v.shape(dvop.outputs[0]) == v.shape(values), "DIV shape")
  k = scalar_or_vec(v, topk.inputs[1])
  need(len(k) == 1, "TOPK k")
  return {"logits": logits, "softmax": sm, "values": values, "sum": su, "div": dv, "div_out": dvop.outputs[0],
          "k": int(k[0]), "E": v.shape(logits)[-1], "T": v.shape(logits)[0] if rank == 2 else None}


def rewrite(bundle, kinds, theta_expect=1000000.0):
  model = bundle.model
  rep = {"kinds": sorted(kinds), "signatures": {}, "composites_added": collections.Counter(),
         "decompositions_added": 0}
  comp_code = G.opcode_index(model, OP.STABLEHLO_COMPOSITE)
  template = None
  for sg in model.subgraphs:
    template = next((op for op in sg.operators if op.opcodeIndex == comp_code), None)
    if template is not None:
      break
  if template is None:
    raise RuntimeError("no STABLEHLO_COMPOSITE op to copy (kv_pass adds them first)")
  for sd in model.signatureDefs:
    key = G.name_of(sd.signatureKey)
    if not (key.startswith("prefill") or key == "decode"):
      continue
    si = sd.subgraphIndex
    sg = model.subgraphs[si]
    v = View(bundle, sg)
    srep = {"hidden_norms": 0, "head_paths": collections.Counter(), "skipped": [], "decompositions": []}
    norms = []
    for oi in range(len(sg.operators)):
      if v.code(oi) != OP.RSQRT:
        continue
      try:
        norms.append(match_rmsnorm(v, oi))
      except NoMatch as e:
        srep["skipped"].append(f"rsqrt op {oi}: {e}")
    hidden = [m for m in norms if m["rank"] == 3]
    heads = []
    for m in [m for m in norms if m["rank"] == 4]:
      try:
        heads.append(match_head(v, m))
      except NoMatch as e:
        srep["skipped"].append(f"head norm at op {m['ops'][0]}: {e}")
    srep["matched"] = {"hidden_norms": len(hidden), "head_paths": len(heads)}
    replace, remove = {}, set()
    insert_after = collections.defaultdict(list)
    shared = {}  # (kind, chain signature) -> decomposition subgraph index

    def decomposition(kindkey, op_idxs, boundary, inputs, outputs, name):
      sig = (kindkey, chain_signature(v, op_idxs, boundary))
      if sig not in shared:
        model.subgraphs.append(extract(v, op_idxs, inputs, outputs, name))
        shared[sig] = len(model.subgraphs) - 1
        srep["decompositions"].append({"name": name, "subgraph": shared[sig], "ops": len(op_idxs)})
        rep["decompositions_added"] += 1
      return shared[sig]

    if "rms" in kinds:
      for m in hidden:
        T, D = v.shape(m["x"])[1], v.shape(m["x"])[2]
        di = decomposition(("rms", T, D), m["ops"], [m["x"], m["w"]], [(m["x"], None), (m["w"], None)], [m["out"]],
                           f"r4.odml.rms_norm.impl_{key}_{T}x{D}_{len(srep['decompositions'])}")
        replace[m["ops"][-1]] = G.composite_op(model, template, "odml.rms_norm", di, {"epsilon": m["eps"]},
                                               [m["x"], m["w"]], [m["out"]])
        remove.update(m["ops"][:-1])
        srep["hidden_norms"] += 1
        rep["composites_added"]["odml.rms_norm"] += 1
    if "qkv" in kinds and heads:
      # position [1, T] f32 from the CAST of input_pos that the RoPE tables start from
      pos_in = next(tm.tensorIndex for tm in sd.inputs if G.name_of(tm.name) == "input_pos")
      casts = [oi for oi in v.cons(pos_in) if v.code(oi) == OP.CAST]
      need(len(casts) == 1, f"{key}: {len(casts)} CAST ops read input_pos")
      cast = casts[0]
      pos_f = v.op(cast).outputs[0]
      T = heads[0]["T"]
      need(v.shape(pos_f) == [T] and v.sg.tensors[pos_f].type == TT.FLOAT32, f"{key}: CAST out {v.shape(pos_f)}")
      tabs = {}
      for h in heads:
        for t in (h["cos"], h["sin"]):
          if t not in tabs:
            tabs[t] = table_ops(v, [t], pos_f)
      tab_ops = sorted(set().union(*tabs.values()))
      for oi in tab_ops:
        for t in v.op(oi).inputs:
          if t == pos_f:
            need(v.code(oi) == OP.RESHAPE, f"{key}: the CAST output is read by {v.name(oi)} in the tables")
      thetas = []
      for oi in tab_ops:
        for t in v.op(oi).inputs:
          if t >= 0 and v.is_const(t) and v.shape(t) == [1, heads[0]["hd"] // 2]:
            thetas.append(theta_of(v.const(t), heads[0]["hd"]))
      need(len(thetas) >= 1, f"{key}: no inv_freq constant in the RoPE tables")
      theta, spread = thetas[0]
      need(abs(theta / theta_expect - 1) < 1e-4, f"{key}: rope theta {theta} (expected {theta_expect})")
      srep["rope"] = {"theta_from_inv_freq": theta, "theta_rel_spread": spread, "theta_used": theta_expect,
                      "table_ops": [v.name(oi) for oi in tab_ops]}
      pos_1t = G.add_tensor(model, sg, f"{key}_r4_position_1xT", TT.FLOAT32, [1, T])
      shp = G.add_tensor(model, sg, f"{key}_r4_position_shape", TT.INT32, [2], np.array([1, T], np.int32))
      insert_after[cast].append(G.make_op(model, OP.RESHAPE, [pos_f, shp], [pos_1t]))
      for h in heads:
        chain = sorted(set(tabs[h["cos"]]) | set(tabs[h["sin"]]) | set(h["ops"]))
        di = decomposition(("qkv", h["n"], h["T"], h["hd"]), chain, [h["fc_out"], pos_f, h["w"]],
                           [(h["fc_out"], None), (pos_f, [1, T]), (h["w"], None)], [h["out"]],
                           f"r4.odml.qkv_norm_rope.impl_{key}_n{h['n']}_{len(srep['decompositions'])}")
        attrs = {"num_heads": h["n"], "num_kv_heads": 0, "head_dim": h["hd"], "min_timescale": 1.0,
                 "max_timescale": theta_expect, "proportion": 1.0, "epsilon": h["eps"], "has_v_norm": False}
        replace[h["last"]] = G.composite_op(model, template, "odml.qkv_norm_rope", di, attrs,
                                            [h["fc_out"], pos_1t, h["w"]], [h["out"]])
        remove.update(set(h["ops"]) - {h["last"]})
        srep["head_paths"][f"n{h['n']}"] += 1
        rep["composites_added"]["odml.qkv_norm_rope"] += 1
    if "router" in kinds:
      srep["routers"] = 0
      for oi in range(len(sg.operators)):
        if v.code(oi) != OP.TOPK_V2:
          continue
        try:
          m = match_router(v, oi)
        except NoMatch as e:
          srep["skipped"].append(f"topk op {oi}: {e}")
          continue
        need(oi not in replace and m["div"] not in replace and not ({oi, m["div"], m["softmax"], m["sum"]} & remove),
             f"{key}: router op {oi} overlaps another rewrite")
        nt = copy.deepcopy(v.op(oi))
        nt.inputs = [m["logits"]] + list(v.op(oi).inputs[1:])
        replace[oi] = nt
        ns = copy.deepcopy(v.op(m["softmax"]))  # SOFTMAX, beta 1, over the k values (last axis)
        ns.inputs, ns.outputs = [m["values"]], [m["div_out"]]
        replace[m["div"]] = ns
        remove.update({m["softmax"], m["sum"]})
        srep["routers"] += 1
        srep.setdefault("router_shapes", collections.Counter())[f"E{m['E']} k{m['k']}"] += 1
      if "router_shapes" in srep:
        srep["router_shapes"] = dict(srep["router_shapes"])
      rep["routers_rewritten"] = rep.get("routers_rewritten", 0) + srep["routers"]
    ops = []
    for oi, op in enumerate(sg.operators):
      if oi not in remove:
        ops.append(replace.get(oi, op))
      ops.extend(insert_after.get(oi, []))
    sg.operators = ops
    v.refresh()
    srep["dead_ops_removed"] = dict(dead_ops(v))
    ok, bad = G.check_topo(model, sg)
    if not ok:
      raise RuntimeError(f"{key}: not topologically ordered at {bad}")
    srep["tensors"] = compact(model, si)
    srep["head_paths"] = dict(srep["head_paths"])
    rep["signatures"][key] = srep
  rep["composites_added"] = dict(rep["composites_added"])
  return rep


# ----------------------------------------------------------------------------------------------------------------------
# writing in place


def buffer_identity_bundle(src, dst_path, dst_begin):
  """Every buffer of src's TFLite section at the same index in dst's, inline bytes equal, offset and size equal."""
  with open(dst_path, "rb") as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    B = fb.Model.GetRootAsModel(mm, dst_begin)
    res = {"src_buffers": len(src.model.buffers), "dst_buffers": B.BuffersLength(), "inline_same": 0,
           "external_same": 0, "empty": 0, "mismatch": []}
    for i, a in enumerate(src.model.buffers):
      b = B.Buffers(i)
      if a.offset is not None and a.offset > 1:
        if b.Offset() == a.offset and b.Size() == a.size:
          res["external_same"] += 1
        else:
          res["mismatch"].append(i)
      elif a.data is not None and len(a.data):
        if b.DataLength() == len(a.data) and bytes(b.DataAsNumpy()) == bytes(a.data):
          res["inline_same"] += 1
        else:
          res["mismatch"].append(i)
      else:
        res["empty"] += 1
        if b.DataLength() or b.Offset() > 1:
          res["mismatch"].append(i)
    ds, _ = G.external_region(B)
    mm.close()
  res["dst_data_start"] = ds
  res["ok"] = not res["mismatch"] and ds == src.data_start
  return res


def hash3(path, begin, region_start):
  """One sequential read: sha256 of the whole file, of [0, begin) and of [region_start, EOF)."""
  hf, hh, hr = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
  pos = 0
  with open(path, "rb") as f:
    while True:
      b = f.read(1 << 24)
      if not b:
        break
      hf.update(b)
      if pos < begin:
        hh.update(b[:begin - pos])
      lo = region_start - pos
      if lo < len(b):
        hr.update(b[max(lo, 0):])
      pos += len(b)
  return {"full": hf.hexdigest(), "head": hh.hexdigest(), "region": hr.hexdigest(), "bytes": pos}


def clone(src, dst):
  if sys.platform == "darwin":
    subprocess.run(["cp", "-c", src, dst], check=True)
  else:
    subprocess.run(["cp", "--reflink=auto", src, dst], check=True)


def build(src, dst, packed, rep, src_sha):
  t0 = time.time()
  if os.path.exists(dst):
    raise RuntimeError(f"{dst} exists")
  if len(packed) > src.data_start:
    raise RuntimeError(f"the new structure ({len(packed)} B) does not fit before the weights ({src.data_start} B)")
  if src.end != src.size:
    raise RuntimeError("the TFLite section does not end at EOF")
  region = src.begin + src.data_start
  hs = hash3(src.path, src.begin, region)
  rep["src_hashes"] = hs
  if src_sha and hs["full"] != src_sha:
    raise RuntimeError(f"source sha256 {hs['full']} != {src_sha}")
  tmp = dst + ".tmp"
  if os.path.exists(tmp):
    os.remove(tmp)
  clone(src.path, tmp)
  try:
    with open(tmp, "r+b") as f:
      f.seek(src.begin)
      f.write(packed)
      f.write(b"\0" * (src.data_start - len(packed)))
      f.flush()
      os.fsync(f.fileno())
    hd = hash3(tmp, src.begin, region)
    rep["dst_hashes"] = hd
    chk = {"size_same": hd["bytes"] == src.size and os.path.getsize(tmp) == src.size,
           "head_same": hd["head"] == hs["head"], "weights_region": [region, src.end],
           "weights_same": hd["region"] == hs["region"], "sections_same": header_sections(tmp) == src.secs,
           "buffer_identity": buffer_identity_bundle(src, tmp, src.begin), "structure_bytes": len(packed),
           "structure_sha256": hashlib.sha256(packed).hexdigest(), "padding_bytes": src.data_start - len(packed)}
    rep["asserts"] = chk
    bad = [k for k in ("size_same", "head_same", "weights_same", "sections_same") if not chk[k]]
    if not chk["buffer_identity"]["ok"]:
      bad.append("buffer_identity")
    if bad:
      raise RuntimeError(f"asserts failed: {bad}")
    rep["dst_sha256"] = hd["full"]
    os.replace(tmp, dst)
  except Exception:
    if os.path.exists(tmp):
      os.remove(tmp)
    raise
  rep["build_seconds"] = round(time.time() - t0, 1)


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("src")
  ap.add_argument("dst", nargs="?")
  ap.add_argument("--rope-theta", type=float, required=True,
                  help="the source checkpoint's rope_theta (odml.qkv_norm_rope's max_timescale)")
  ap.add_argument("--dry", action="store_true", help="the rewrite in memory only: sizes, counts, the structure sha256")
  ap.add_argument("--src-sha256", help="expected sha256 of SRC (checked during the asserts' read)")
  ap.add_argument("--report", help="write a JSON report here")
  a = ap.parse_args()
  if not a.dry and not a.dst:
    raise SystemExit("give DST, or --dry")
  t0 = time.time()
  bundle = Bundle(a.src)
  rep = {"src": a.src, "section_begin": bundle.begin, "section_end": bundle.end, "src_data_start": bundle.data_start,
         "src_bytes": bundle.size, "sections": bundle.secs, "census_src": census(bundle.model),
         "src_structure_bytes_repacked": len(G.pack_model(bundle.model))}
  kv_bytes = kv_pass(bundle, rep)
  snap = blockwise_snapshot(bundle.model)
  rw = rewrite(bundle, {"rms", "qkv", "router"}, theta_expect=a.rope_theta)
  rw["blockwise_refs"] = blockwise_check(bundle.model, snap)
  rep["rewrite"] = rw
  bad_refs = {k: v for k, v in rw["blockwise_refs"].items() if v["wrong_count"]}
  packed = G.pack_model(bundle.model)
  rep.update({"kv_structure_bytes": kv_bytes, "new_structure_bytes": len(packed),
              "new_structure_sha256": hashlib.sha256(packed).hexdigest(),
              "fits_before_weights": len(packed) <= bundle.data_start, "census_after": census(bundle.model),
              "subgraphs_after": len(bundle.model.subgraphs), "buffers_after": len(bundle.model.buffers),
              "external_buffers": len(ext_buffers(bundle.model)),
              "tools": {"gpu_rewrite.py": file_sha256(os.path.abspath(__file__)),
                        "gpu_graph_retrofit.py": file_sha256(PUBLIC)}, "args": vars(a)})
  if bad_refs:
    raise SystemExit(f"tensor references broken by the rewrite, nothing written: {json.dumps(bad_refs)[:2000]}")
  if not a.dry:
    build(bundle, a.dst, packed, rep, a.src_sha256)
  rep["seconds"] = round(time.time() - t0, 1)
  if a.report:
    with open(a.report, "w") as f:
      f.write(json.dumps(rep, indent=1, default=str) + "\n")
  print("structure", rep["src_structure_bytes_repacked"], "-> kv", kv_bytes, "->", len(packed), "B of data_start",
        bundle.data_start, "(fits)" if rep["fits_before_weights"] else "(DOES NOT FIT)", "sha256",
        rep["new_structure_sha256"])
  print("composites", rw["composites_added"], "routers", rw.get("routers_rewritten", 0), "decompositions",
        rw["decompositions_added"], "| blockwise references (same / blockwise / wrong):",
        {k: (v["same"], v["blockwise"], v["wrong_count"]) for k, v in rw["blockwise_refs"].items()})
  for k, v in rep["census_after"].items():
    b = rep["census_src"][k]
    bt = v["by_type"]
    print(f"{k}: ops {b['ops']} -> {v['ops']}; cache_update {bt.get('STABLEHLO_COMPOSITE:odml.cache_update')} "
          f"runtime_bmm {bt.get('STABLEHLO_COMPOSITE:odml.runtime_bmm')} rms_norm "
          f"{bt.get('STABLEHLO_COMPOSITE:odml.rms_norm')} qkv_norm_rope "
          f"{bt.get('STABLEHLO_COMPOSITE:odml.qkv_norm_rope')}")
  if "asserts" in rep:
    c = rep["asserts"]
    print("asserts:", {k: c[k] for k in ("size_same", "head_same", "weights_same", "sections_same")},
          "buffer identity", c["buffer_identity"]["ok"], "| DST sha256", rep["dst_sha256"])
  return 0


if __name__ == "__main__":
  sys.exit(main())
