#!/usr/bin/env python3
"""decode_rewrite.py: fewer GPU dispatches per token for a published Qwen3-30B-A3B-family file, written in place.

  python decode_rewrite.py SRC.litertlm DST.litertlm [--src-sha256 HEX] [--report R.json]
  python decode_rewrite.py SRC.litertlm --dry [--report R.json]

SRC is an output of `gpu_rewrite.py`, with its attention and lm_head weights int8 or, after `int4_attention.py`, int4.

Changes, in every prefill and decode signature:
  1. The attention core of each layer (RESHAPE q, odml.runtime_bmm(q, K), RESHAPE, MUL 1/sqrt(hd), ADD mask, SOFTMAX,
     RESHAPE, odml.runtime_bmm(p, V), RESHAPE) becomes one odml.sdpa_transposed(q, K, V, mask, param_tensor) with
     k_ts_idx 2, v_ts_idx 3, is_causal true, from_cache_update true. The op takes q already scaled, so 1/sqrt(hd) is
     folded into the weight of the q path's odml.qkv_norm_rope (the norm and RoPE are linear in that weight). The
     decomposition, which the CPU runs, is the old chain without the MUL, with plain BATCH_MATMULs.
  2. A layer's q, k and v projections become one FULLY_CONNECTED [Rq + Rk + Rv, C] (rows in the order q, k, v; int8 per
     channel: the scale and zero-point vectors concatenated; int4 blockwise: codes and scales concatenated) and its two
     odml.qkv_norm_rope one odml.qkv_norm_rope in its 4-input, 3-output form: inputs (qkv, position, q weight, k
     weight), outputs (q, k, v), num_kv_heads set. V passes through the composite (has_v_norm false) in place of the V
     path's RESHAPE and TRANSPOSE. The file holds k, v, q in that order with small constants (norm weights, the
     decompositions' tables) between them, so each layer's span is rewritten in place: q, k, v as one region, then those
     constants, their bytes unchanged at new offsets. A layer of a prefill signature without a q projection (the last
     layer: nothing reads its attention) keeps its k and v FCs, whose buffers move to their new place in the region.
The tensors and subgraphs this script adds carry the tag `r11` in their names.

Writing: nothing is unpacked. DST = a copy of SRC (`cp -c`, an APFS clone, on macOS; `cp --reflink=auto` elsewhere) with
the new structure and zero padding over [section begin, section begin + data_start) and the rewritten regions. Then the
asserts, read back from disk (DST is renamed into place only when all hold): the same size, header bytes and section
table; every byte outside the rewritten ranges equal (both files read once); every source buffer not in the changed
list at the same index with the same inline bytes or offset and size; no live tensor outside the changed list reads a
rewritten range; each merged tensor read back through DST's own flatbuffer equal to SRC's q | k | v bytes; each moved
buffer's bytes at its new offset equal to SRC's at its old one.

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
import re
import sys
import time

import numpy as np
from flatbuffers import flexbuffers

PUBLIC_REWRITE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpu_rewrite.py")


def load_by_path(name, path):
  spec = importlib.util.spec_from_file_location(name, path)
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


W = load_by_path("gpu_rewrite_public", PUBLIC_REWRITE)
G, fb, OP, TT = W.G, W.fb, W.OP, W.TT
View, NoMatch, need = W.View, W.NoMatch, W.need
BLOCKWISE = fb.QuantizationDetails.BlockwiseQuantization
PROJ_RX = re.compile(r"model\.layers\.(\d+)\.self_attn\.([qkv])_proj\.weight$")


def comp_name(v, oi):
  op = v.op(oi)
  if v.code(oi) != OP.STABLEHLO_COMPOSITE or op.builtinOptions2 is None:
    return None
  return G.name_of(op.builtinOptions2.name)


def comp_attrs(op):
  raw = op.builtinOptions2.compositeAttributes
  return flexbuffers.Loads(bytes(bytearray(raw))) if raw is not None and len(raw) else {}


def single_cons(v, t, code, why):
  c = v.cons(t)
  need(len(c) == 1 and v.code(c[0]) == code, f"{why}: readers {[v.name(x) for x in c]}")
  return c[0]


# ----------------------------------------------------------------------------------------------------------------------
# sdpa


def match_attention(v, a_oi, mask_t, param_t):
  A = v.op(a_oi)
  need(comp_name(v, a_oi) == "odml.runtime_bmm", "not runtime_bmm")
  need(not comp_attrs(A).get("is_src", False), "is_src")
  need(len(A.inputs) == 3 and A.inputs[2] == param_t, "q.K inputs")
  qp, kc = A.inputs[0], A.inputs[1]
  r0 = v.prod(qp)
  need(r0 is not None and v.code(r0) == OP.RESHAPE, "q RESHAPE")
  q = v.op(r0).inputs[0]
  need(v.cons(qp) == [a_oi], "packed q read elsewhere")
  B1, H, T, hd = v.shape(q)
  _, KV, gT, hd2 = v.shape(qp)
  need(B1 == 1 and hd2 == hd and KV * gT == H * T and H % KV == 0 and gT == (H // KV) * T, f"q {v.shape(q)} {v.shape(qp)}")
  kshape = v.shape(kc)
  need(kshape[:2] == [1, KV] and kshape[3] == hd, f"K cache {kshape}")
  C = kshape[2]
  kp = v.prod(kc)
  need(kp is not None and comp_name(v, kp) == "odml.cache_update" and v.op(kp).outputs[0] == kc, "K from cache_update")
  s1 = A.outputs[0]
  r1 = single_cons(v, s1, OP.RESHAPE, "q.K out")
  need(v.shape(v.op(r1).outputs[0]) == [1, H, T, C], "scores RESHAPE")
  m = single_cons(v, v.op(r1).outputs[0], OP.MUL, "scores")
  sc = W.other_input(v.op(m), v.op(r1).outputs[0])
  need(sc is not None and v.is_const(sc) and v.const(sc).size == 1, "scale")
  scale = float(v.const(sc).reshape(-1)[0])
  need(abs(scale * np.sqrt(hd) - 1) < 1e-6, f"scale {scale} for hd {hd}")
  a = single_cons(v, v.op(m).outputs[0], OP.ADD, "scaled scores")
  need(W.other_input(v.op(a), v.op(m).outputs[0]) == mask_t, "ADD with the mask input")
  need(v.shape(mask_t) == [1, 1, T, C], f"mask {v.shape(mask_t)}")
  sm = single_cons(v, v.op(a).outputs[0], OP.SOFTMAX, "masked scores")
  need(abs(v.op(sm).builtinOptions.beta - 1.0) == 0, "softmax beta")
  r2 = single_cons(v, v.op(sm).outputs[0], OP.RESHAPE, "probs")
  need(v.shape(v.op(r2).outputs[0]) == [1, KV, gT, C], "probs RESHAPE")
  b_oi = single_cons(v, v.op(r2).outputs[0], OP.STABLEHLO_COMPOSITE, "packed probs")
  Bo = v.op(b_oi)
  need(comp_name(v, b_oi) == "odml.runtime_bmm" and comp_attrs(Bo).get("is_src", False), "p.V runtime_bmm")
  need(len(Bo.inputs) == 3 and Bo.inputs[2] == param_t, "p.V inputs")
  vc = Bo.inputs[1]
  need(v.prod(vc) == kp and v.op(kp).outputs[1] == vc, "V from the same cache_update")
  need(v.shape(vc) == [1, KV, hd, C], f"V cache {v.shape(vc)}")
  r3 = single_cons(v, Bo.outputs[0], OP.RESHAPE, "p.V out")
  out = v.op(r3).outputs[0]
  need(v.shape(out) == [1, H, T, hd], "out RESHAPE")
  qc = v.prod(q)
  need(qc is not None and comp_name(v, qc) == "odml.qkv_norm_rope", f"q is not a qkv_norm_rope output: {v.name(qc)}")
  need(v.cons(q) == [r0], "q read elsewhere")
  return {"q": q, "q_comp": qc, "qp": qp, "kc": kc, "vc": vc, "scale": scale, "out": out, "H": H, "KV": KV, "T": T,
          "hd": hd, "C": C, "ops": [r0, a_oi, r1, m, a, sm, r2, b_oi, r3], "r0": r0, "A": a_oi, "r1": r1, "m": m,
          "a": a, "sm": sm, "r2": r2, "B": b_oi, "r3": r3}


def decomp_bmm_op(v, comp_oi):
  si = v.op(comp_oi).builtinOptions2.decompositionSubgraphIndex
  d = v.model.subgraphs[si]
  bmms = [op for op in d.operators if W.code_of(v.model, op) == OP.BATCH_MATMUL]
  need(len(bmms) == 1, f"runtime_bmm decomposition {si}: {len(bmms)} BATCH_MATMUL ops ({len(d.operators)} ops)")
  return bmms[0]


def sdpa_decomposition(v, m, name):
  """A new SubGraphT: inputs (q scaled, K, V, mask, param), the chain of m without the MUL, plain BATCH_MATMULs."""
  sg = v.sg
  d = fb.SubGraphT()
  d.tensors, d.operators, d.name = [], [], name
  tmap = {}

  def copy_t(t, const=False):
    src = sg.tensors[t]
    nt = fb.TensorT()
    nt.name = f"{name}/t{len(d.tensors)}"
    nt.type = src.type
    shp = [int(x) for x in src.shape]
    nt.shape = shp
    nt.shapeSignature = list(shp)
    nt.buffer = src.buffer if const else 0
    nt.hasRank = True
    nt.quantization = copy.deepcopy(src.quantization) if src.quantization is not None else fb.QuantizationParametersT()
    d.tensors.append(nt)
    return len(d.tensors) - 1

  for t in (m["q"], m["kc"], m["vc"], m["mask"], m["param"]):
    tmap[t] = copy_t(t)
  d.inputs = [tmap[m["q"]], tmap[m["kc"]], tmap[m["vc"]], tmap[m["mask"]], tmap[m["param"]]]

  def emit(op, ins, outs):
    nop = copy.deepcopy(op)
    ni = []
    for t in ins:
      if t < 0:
        ni.append(t)
        continue
      if t not in tmap:
        need(v.is_const(t), f"{name}: tensor {t} is not a chain tensor nor a constant")
        tmap[t] = copy_t(t, const=True)
      ni.append(tmap[t])
    no = []
    for t in outs:
      tmap[t] = copy_t(t)
      no.append(tmap[t])
    nop.inputs, nop.outputs = ni, no
    d.operators.append(nop)

  r0, r1, a, sm, r2, r3 = (v.op(m[k]) for k in ("r0", "r1", "a", "sm", "r2", "r3"))
  emit(r0, list(r0.inputs), list(r0.outputs))
  emit(decomp_bmm_op(v, m["A"]), [m["qp"], m["kc"]], list(v.op(m["A"]).outputs))
  emit(r1, list(r1.inputs), list(r1.outputs))
  s_unscaled = r1.outputs[0]
  a_ins = [s_unscaled if t == v.op(m["m"]).outputs[0] else t for t in a.inputs]  # the MUL is gone: q is scaled
  emit(a, a_ins, list(a.outputs))
  emit(sm, list(sm.inputs), list(sm.outputs))
  emit(r2, list(r2.inputs), list(r2.outputs))
  emit(decomp_bmm_op(v, m["B"]), [r2.outputs[0], m["vc"]], list(v.op(m["B"]).outputs))
  emit(r3, list(r3.inputs), list(r3.outputs))
  d.outputs = [tmap[m["out"]]]
  return d


def sdpa_pass(v, sd, key, srep, template, rep, scaled_weights):
  model, sg = v.model, v.sg
  mask_t = next((tm.tensorIndex for tm in sd.inputs if G.name_of(tm.name) == "mask"), None)
  param_t = next((tm.tensorIndex for tm in sd.inputs if G.name_of(tm.name) == "param_tensor"), None)
  need(mask_t is not None and param_t is not None, f"{key}: mask / param_tensor inputs")
  atts = []
  for oi in range(len(sg.operators)):
    if comp_name(v, oi) == "odml.runtime_bmm" and not comp_attrs(v.op(oi)).get("is_src", False):
      try:
        m = match_attention(v, oi, mask_t, param_t)
        m["mask"], m["param"] = mask_t, param_t
        atts.append(m)
      except NoMatch as e:
        srep["skipped"].append(f"runtime_bmm op {oi}: {e}")
  srep["attention_matched"] = len(atts)
  replace, remove = {}, set()
  di, first = None, None
  for m in atts:
    consts = [bytes(v.b.buffer_bytes(v.sg.tensors[v.op(m[k]).inputs[1]].buffer)) for k in ("r0", "r1", "r2", "r3")]
    shape_key = (m["H"], m["KV"], m["T"], m["hd"], m["C"], m["scale"])
    if di is None:
      model.subgraphs.append(sdpa_decomposition(v, m, f"r11.odml.sdpa_transposed.impl_{key}"))
      di = len(model.subgraphs) - 1
      first = (shape_key, consts)
      rep["decompositions_added"] += 1
      srep["sdpa_decomposition"] = {"subgraph": di, "ops": len(model.subgraphs[di].operators)}
    else:
      need((shape_key, consts) == first, f"{key}: attention at op {m['A']} differs from the first one's shapes")
      need(decomp_bmm_op(v, m["A"]).builtinOptions.adjY == decomp_bmm_op(v, atts[0]["A"]).builtinOptions.adjY
           and decomp_bmm_op(v, m["B"]).builtinOptions.adjY == decomp_bmm_op(v, atts[0]["B"]).builtinOptions.adjY,
           f"{key}: BMM options differ")
    # the scale goes into the q path's norm weight (one new inline buffer per source weight, shared by the signatures)
    qc = v.op(m["q_comp"])
    wt = qc.inputs[2]
    src_t = sg.tensors[wt]
    need(v.is_const(wt) and src_t.type == TT.FLOAT32 and v.shape(wt) == [m["hd"]], f"{key}: q weight {v.shape(wt)}")
    if int(src_t.buffer) not in scaled_weights:
      w = v.const(wt).astype(np.float32)
      scaled = (w * np.float32(m["scale"])).astype(np.float32)
      scaled_weights[int(src_t.buffer)] = G.new_buffer(model, scaled)
    nt = copy.deepcopy(src_t)
    nt.name = (G.name_of(src_t.name) or f"{key}_qw_{wt}") + "_r11_scaled"
    nt.buffer = scaled_weights[int(src_t.buffer)]
    sg.tensors.append(nt)
    qc.inputs = [qc.inputs[0], qc.inputs[1], len(sg.tensors) - 1]
    attrs = {"k_ts_idx": 2, "v_ts_idx": 3, "is_causal": True, "from_cache_update": True}
    replace[m["r3"]] = G.composite_op(model, template, "odml.sdpa_transposed", di, attrs,
                                      [m["q"], m["kc"], m["vc"], m["mask"], m["param"]], [m["out"]])
    remove.update(set(m["ops"]) - {m["r3"]})
    rep["added"]["odml.sdpa_transposed"] += 1
  ops = []
  for oi, op in enumerate(sg.operators):
    if oi not in remove:
      ops.append(replace.get(oi, op))
  sg.operators = ops
  v.refresh()
  srep["dead_ops_after_sdpa"] = dict(W.dead_ops(v))


# ----------------------------------------------------------------------------------------------------------------------
# qkv


def align64(x):
  return -(-x // 64) * 64


def qkv_decomposition(v, q_op, k_op, vres, vtr, T, nq, nkv, hd, name):
  """(qkv [1, T, (nq + 2 nkv) hd], position [1, T], q weight, k weight) -> (q, k, v): SLICEs, then the two Q-only
  decompositions' ops, then the V path's RESHAPE + TRANSPOSE."""
  model, sg = v.model, v.sg
  dq = model.subgraphs[q_op.builtinOptions2.decompositionSubgraphIndex]
  dk = model.subgraphs[k_op.builtinOptions2.decompositionSubgraphIndex]
  d = fb.SubGraphT()
  d.tensors, d.operators, d.name = [], [], name

  def new(ttype, shape, buffer=0, q=None):
    nt = fb.TensorT()
    nt.name = f"{name}/t{len(d.tensors)}"
    nt.type = ttype
    nt.shape = list(shape)
    nt.shapeSignature = list(shape)
    nt.buffer = buffer
    nt.hasRank = True
    nt.quantization = copy.deepcopy(q) if q is not None else fb.QuantizationParametersT()
    d.tensors.append(nt)
    return len(d.tensors) - 1

  rq, rkv = nq * hd, nkv * hd
  x = new(TT.FLOAT32, [1, T, rq + 2 * rkv])
  pos = new(TT.FLOAT32, [1, T])
  wq = new(TT.FLOAT32, [hd])
  wk = new(TT.FLOAT32, [hd])
  d.inputs = [x, pos, wq, wk]

  def const_i32(vals):
    return new(TT.INT32, [len(vals)], G.new_buffer(model, np.array(vals, np.int32)))

  def slice_of(begin, size):
    out = new(TT.FLOAT32, [1, T, size])
    op = G.make_op(model, OP.SLICE, [x, const_i32([0, 0, begin]), const_i32([1, T, size])], [out],
                   fb.BuiltinOptions.SliceOptions, fb.SliceOptionsT())
    d.operators.append(op)
    return out

  xq, xk, xv = slice_of(0, rq), slice_of(rq, rkv), slice_of(rq + rkv, rkv)

  def clone(sub, ins):
    tmap = dict(zip([int(t) for t in sub.inputs], ins))
    for op in sub.operators:
      nop = copy.deepcopy(op)
      ni = []
      for t in op.inputs:
        t = int(t)
        if t < 0:
          ni.append(t)
          continue
        if t not in tmap:
          st = sub.tensors[t]
          tmap[t] = new(st.type, [int(z) for z in st.shape], st.buffer, st.quantization)
        ni.append(tmap[t])
      no = []
      for t in op.outputs:
        t = int(t)
        st = sub.tensors[t]
        tmap[t] = new(st.type, [int(z) for z in st.shape], 0, st.quantization)
        no.append(tmap[t])
      nop.inputs, nop.outputs = ni, no
      d.operators.append(nop)
    return tmap[int(sub.outputs[0])]

  oq = clone(dq, [xq, pos, wq])
  ok = clone(dk, [xk, pos, wk])
  v4 = new(TT.FLOAT32, [1, T, nkv, hd])
  rop = copy.deepcopy(v.op(vres))
  rop.inputs, rop.outputs = [xv, const_i32([1, T, nkv, hd])], [v4]
  d.operators.append(rop)
  ov = new(TT.FLOAT32, [1, nkv, T, hd])
  top = copy.deepcopy(v.op(vtr))
  top.inputs, top.outputs = [v4, const_i32([0, 2, 1, 3])], [ov]
  d.operators.append(top)
  d.outputs = [oq, ok, ov]
  return d


def tensor_form(sg, t):
  """-> ('int8', None) for per-channel int8, ('int4', scales tensor index) for blockwise int4, else NoMatch."""
  q = t.quantization
  if t.type == TT.INT8 and q is not None and q.scale is not None and len(q.scale) == int(t.shape[0]) \
      and (q.detailsType or 0) == 0 and (q.quantizedDimension or 0) == 0:
    return "int8", None
  if t.type == TT.INT4 and q is not None and q.detailsType == BLOCKWISE and q.details is not None:
    need(int(q.details.zeroPoints) == -1 and int(q.details.blockSize) == 32, "blockwise details")
    return "int4", int(q.details.scales)
  raise NoMatch(f"tensor {G.name_of(t.name)}: type {t.type}, details {q.detailsType if q is not None else None}")


def layer_plan(bundle, sg, L, ts, ext):
  """The bytes of one layer's merged weights (shared by the signatures): ts = {'q': tensor, 'k': .., 'v': ..}; ext =
  sorted (offset, size, buffer index) of the source's external buffers. The span [first projection, end of the last) is
  rewritten as: merged codes, merged scales (int4), then every other buffer that lay in the span (the small constants
  between the projections), its bytes unchanged at a new 64-aligned offset. In an int4 file a projection's region is
  still its int8 size (codes, scales, a free rest)."""
  model = bundle.model
  forms = {p: tensor_form(sg, ts[p]) for p in "qkv"}
  kinds = {f[0] for f in forms.values()}
  need(len(kinds) == 1, f"layer {L}: mixed forms {forms}")
  kind = kinds.pop()
  C = int(ts["q"].shape[1])
  need(all(int(ts[p].shape[1]) == C for p in "qkv"), f"layer {L}: input sizes differ")
  rows = {p: int(ts[p].shape[0]) for p in "qkv"}
  buf = {p: int(ts[p].buffer) for p in "qkv"}
  for p in "qkv":
    b = model.buffers[buf[p]]
    need(b.offset is not None and b.offset > 1, f"layer {L} {p}: not an external buffer")
  codes = {p: bundle.buffer_bytes(buf[p]) for p in "qkv"}
  plan = {"L": L, "kind": kind, "rows": rows, "C": C, "buffers": buf, "src_sha": {}, "writes": []}
  start = min(int(model.buffers[buf[p]].offset) for p in "qkv")
  end = max(int(model.buffers[buf[p]].offset) + rows[p] * C for p in "qkv")
  own = set(buf.values())
  merged = b"".join(codes[p] for p in "qkv")
  plan["writes"].append((start, merged))
  plan["codes_off"], plan["codes_len"] = start, len(merged)
  nb = fb.BufferT()
  nb.offset, nb.size = start, len(merged)
  model.buffers.append(nb)
  plan["buffer"] = len(model.buffers) - 1
  moves = {}
  off = start
  for p in "qkv":
    moves[buf[p]] = (off, len(codes[p]))
    off += len(codes[p])
  plan["src_sha"]["codes"] = hashlib.sha256(merged).hexdigest()
  cursor = start + len(merged)
  if kind == "int4":
    sbuf = {p: int(sg.tensors[forms[p][1]].buffer) for p in "qkv"}
    own |= set(sbuf.values())
    scales = {p: bundle.buffer_bytes(sbuf[p]) for p in "qkv"}
    sm = b"".join(scales[p] for p in "qkv")
    soff = align64(cursor)
    plan["writes"].append((soff, sm))
    plan["scales_off"], plan["scales_len"] = soff, len(sm)
    ns = fb.BufferT()
    ns.offset, ns.size = soff, len(sm)
    model.buffers.append(ns)
    plan["scales_buffer"] = len(model.buffers) - 1
    plan["src_scale_buffers"] = sbuf
    off = soff
    for p in "qkv":
      moves[sbuf[p]] = (off, len(scales[p]))
      off += len(scales[p])
    plan["src_sha"]["scales"] = hashlib.sha256(sm).hexdigest()
    cursor = soff + len(sm)
  others = [(o, s, i) for o, s, i in ext if i not in own and o < end and start < o + s]
  for o, s, i in others:
    need(start <= o and o + s <= end, f"layer {L}: buffer {i} straddles the span's edge")
    cursor = align64(cursor)
    plan["writes"].append((cursor, bundle.buffer_bytes(i)))
    moves[i] = (cursor, s)
    cursor += s
  need(cursor <= end, f"layer {L}: the merged weights and the other {len(others)} buffers do not fit the span")
  plan["moves"] = moves
  plan["own"] = own
  plan["span"] = (start, end)
  plan["others"] = len(others)
  return plan


def merged_tensor(sg, plan, tq, tk, tv, name):
  """The merged weight tensor for this subgraph (appended, with its scales tensor when int4) -> tensor index."""
  R, C = sum(plan["rows"].values()), plan["C"]
  t = copy.deepcopy(tq)
  t.name = name
  t.shape = [R, C]
  if t.shapeSignature is not None and len(t.shapeSignature):
    t.shapeSignature = [R, C]
  t.buffer = plan["buffer"]
  if plan["kind"] == "int8":
    qp = fb.QuantizationParametersT()
    qp.quantizedDimension = 0
    qp.scale = np.concatenate([np.asarray(x.quantization.scale, np.float32) for x in (tq, tk, tv)]).tolist()
    qp.zeroPoint = np.concatenate([np.asarray(x.quantization.zeroPoint, np.int64) for x in (tq, tk, tv)]).tolist()
    t.quantization = qp
    sg.tensors.append(t)
    return len(sg.tensors) - 1
  st = fb.TensorT()
  st.name = name + "_scales"
  st.type = TT.FLOAT16
  st.shape = [R * C // 32]
  st.buffer = plan["scales_buffer"]
  sg.tensors.append(st)
  si = len(sg.tensors) - 1
  qp = fb.QuantizationParametersT()
  qp.quantizedDimension = 0
  d = fb.BlockwiseQuantizationT()
  d.scales, d.zeroPoints, d.blockSize = si, -1, 32
  qp.detailsType = BLOCKWISE
  qp.details = d
  t.quantization = qp
  sg.tensors.append(t)
  return len(sg.tensors) - 1


def qkv_pass(bundle, v, sd, key, srep, template, rep, plans, merged_names, ext):
  model, sg = v.model, v.sg
  fcs = collections.defaultdict(dict)
  for oi, op in enumerate(sg.operators):
    if v.code(oi) != OP.FULLY_CONNECTED:
      continue
    m = PROJ_RX.search(G.name_of(sg.tensors[op.inputs[1]].name) or "")
    if m:
      fcs[int(m.group(1))][m.group(2)] = oi
  replace, remove = {}, set()
  shared = {}
  srep["qkv_layers"] = 0
  for L, d in sorted(fcs.items()):
    if set(d) != {"q", "k", "v"}:
      srep.setdefault("qkv_kept", []).append({"layer": L, "projections": sorted(d)})
      continue
    qf, kf, vf = v.op(d["q"]), v.op(d["k"]), v.op(d["v"])
    x = qf.inputs[0]
    need(kf.inputs[0] == x and vf.inputs[0] == x, f"{key} layer {L}: the FCs read different inputs")
    for op in (qf, kf, vf):
      need(len(op.inputs) == 2 or op.inputs[2] == -1, f"{key} layer {L}: a bias")
    need(W.canon(qf.builtinOptions) == W.canon(kf.builtinOptions) == W.canon(vf.builtinOptions), f"{key} {L}: FC options")
    q_out, k_out, v_out = qf.outputs[0], kf.outputs[0], vf.outputs[0]
    qc = single_cons(v, q_out, OP.STABLEHLO_COMPOSITE, f"{key} {L} q FC out")
    kc = single_cons(v, k_out, OP.STABLEHLO_COMPOSITE, f"{key} {L} k FC out")
    need(comp_name(v, qc) == "odml.qkv_norm_rope" and comp_name(v, kc) == "odml.qkv_norm_rope", f"{key} {L}: composites")
    qop, kop = v.op(qc), v.op(kc)
    need(len(qop.inputs) == 3 and len(kop.inputs) == 3 and qop.inputs[1] == kop.inputs[1], f"{key} {L}: composite inputs")
    qa, ka = comp_attrs(qop), comp_attrs(kop)
    for name in ("head_dim", "min_timescale", "max_timescale", "proportion", "epsilon", "has_v_norm"):
      need(qa[name] == ka[name], f"{key} {L}: attribute {name}: {qa[name]} vs {ka[name]}")
    nq, nkv, hd = int(qa["num_heads"]), int(ka["num_heads"]), int(qa["head_dim"])
    vr = single_cons(v, v_out, OP.RESHAPE, f"{key} {L} v FC out")
    v4 = v.op(vr).outputs[0]
    vt = single_cons(v, v4, OP.TRANSPOSE, f"{key} {L} v RESHAPE out")
    perm = v.const(v.op(vt).inputs[1])
    need(perm is not None and perm.reshape(-1).tolist() == [0, 2, 1, 3], f"{key} {L}: v perm")
    v_t = v.op(vt).outputs[0]
    T = v.shape(q_out)[1]
    need(v.shape(v4) == [1, T, nkv, hd] and v.shape(v_t) == [1, nkv, T, hd], f"{key} {L}: V shapes")
    need(v.shape(q_out) == [1, T, nq * hd] and v.shape(k_out) == [1, T, nkv * hd], f"{key} {L}: q, k shapes")
    tq, tk, tv = (sg.tensors[o.inputs[1]] for o in (qf, kf, vf))
    if L not in plans:
      plans[L] = layer_plan(bundle, sg, L, {"q": tq, "k": tk, "v": tv}, ext)
    plan = plans[L]
    name = f"model.layers.{L}.self_attn.qkv_proj.weight"
    merged_names.setdefault(key, set()).update(G.name_of(t.name) for t in (tq, tk, tv))
    w_i = merged_tensor(sg, plan, tq, tk, tv, name)
    out_t = copy.deepcopy(sg.tensors[q_out])
    out_t.name = f"{key}_r11_qkv_{L}"
    out_t.shape = [1, T, (nq + 2 * nkv) * hd]
    if out_t.shapeSignature is not None and len(out_t.shapeSignature):
      out_t.shapeSignature = list(out_t.shape)
    sg.tensors.append(out_t)
    qkv_t = len(sg.tensors) - 1
    fc = copy.deepcopy(kf)
    fc.inputs = [x, w_i] + list(kf.inputs[2:])
    fc.outputs = [qkv_t]
    sig = (T, nq, nkv, hd, qop.builtinOptions2.decompositionSubgraphIndex, kop.builtinOptions2.decompositionSubgraphIndex)
    if sig not in shared:
      model.subgraphs.append(qkv_decomposition(v, qop, kop, vr, vt, T, nq, nkv, hd,
                                               f"r11.odml.qkv_norm_rope.impl_{key}_{nq}_{nkv}"))
      shared[sig] = len(model.subgraphs) - 1
      rep["decompositions_added"] += 1
      srep.setdefault("qkv_decompositions", []).append({"subgraph": shared[sig],
                                                        "ops": len(model.subgraphs[shared[sig]].operators)})
    attrs = {"num_heads": nq, "num_kv_heads": nkv, "head_dim": hd, "min_timescale": float(qa["min_timescale"]),
             "max_timescale": float(qa["max_timescale"]), "proportion": float(qa["proportion"]),
             "epsilon": float(qa["epsilon"]), "has_v_norm": False}
    comp = G.composite_op(model, template, "odml.qkv_norm_rope", shared[sig], attrs,
                          [qkv_t, qop.inputs[1], qop.inputs[2], kop.inputs[2]], [qop.outputs[0], kop.outputs[0], v_t])
    first_fc = min(d.values())
    replace[first_fc] = fc
    last_pos = max(kc, 0)
    replace[last_pos] = comp
    remove.update({d["q"], d["k"], d["v"], qc, kc, vr, vt} - {first_fc, last_pos})
    rep["added"]["odml.qkv_norm_rope (4, 3)"] += 1
    srep["qkv_layers"] += 1
  ops = []
  for oi, op in enumerate(sg.operators):
    if oi not in remove:
      ops.append(replace.get(oi, op))
  sg.operators = ops
  v.refresh()
  srep["dead_ops_after_qkv"] = dict(W.dead_ops(v))


# ----------------------------------------------------------------------------------------------------------------------


def rewrite(bundle):
  model = bundle.model
  rep = {"signatures": {}, "added": collections.Counter(), "decompositions_added": 0}
  comp_code = G.opcode_index(model, OP.STABLEHLO_COMPOSITE)
  template = None
  for sg in model.subgraphs:
    template = next((op for op in sg.operators if op.opcodeIndex == comp_code), None)
    if template is not None:
      break
  if template is None:
    raise RuntimeError("no STABLEHLO_COMPOSITE op to copy")
  plans, merged_names, scaled_weights = {}, {}, {}
  ext = sorted((int(b.offset), int(b.size), i) for i, b in enumerate(model.buffers) if b.offset is not None and b.offset > 1)
  for sd in model.signatureDefs:
    key = G.name_of(sd.signatureKey)
    if not (key.startswith("prefill") or key == "decode"):
      continue
    si = sd.subgraphIndex
    sg = model.subgraphs[si]
    v = View(bundle, sg)
    srep = {"skipped": []}
    sdpa_pass(v, sd, key, srep, template, rep, scaled_weights)
    qkv_pass(bundle, v, sd, key, srep, template, rep, plans, merged_names, ext)
    ok, bad = G.check_topo(model, sg)
    if not ok:
      raise RuntimeError(f"{key}: not topologically ordered at {bad}")
    srep["tensors"] = W.compact(model, si)
    rep["signatures"][key] = srep
  # the old q / k / v buffers: moved when a live tensor still reads them, emptied otherwise; every other buffer of a span moves
  live = set()
  for sg in model.subgraphs:
    for t in sg.tensors:
      live.add(int(t.buffer))
  moved, emptied = 0, 0
  changed = set()
  for p in plans.values():
    for bi, (off, size) in p["moves"].items():
      b = model.buffers[bi]
      changed.add(bi)
      if bi in p["own"] and bi not in live:
        b.offset, b.size, b.data = 0, 0, None
        emptied += 1
      else:
        b.offset, b.size = off, size
        moved += 1
  rep["qkv_buffers"] = {"moved": moved, "emptied": emptied, "layers": len(plans),
                        "others_in_spans": sum(p["others"] for p in plans.values())}
  rep["added"] = dict(rep["added"])
  return rep, plans, merged_names, changed


def stream_compare(src, dst, begin, data_start, holes, end):
  """One pass over both files: full sha256 of each, and whether every byte of [begin + data_start, end) outside the
  holes (section-relative ranges) is equal."""
  holes = sorted(holes)
  hs, hd = hashlib.sha256(), hashlib.sha256()
  bad, compared = [], 0
  hi = 0
  pos = 0
  with open(src, "rb") as a, open(dst, "rb") as b:
    while True:
      x = a.read(1 << 24)
      y = b.read(1 << 24)
      if not x and not y:
        break
      hs.update(x)
      hd.update(y)
      if len(x) != len(y):
        bad.append(("length", pos))
        break
      lo = max(pos, begin + data_start)
      up = pos + len(x)
      while lo < up:
        while hi < len(holes) and begin + holes[hi][1] <= lo:
          hi += 1
        if hi < len(holes) and begin + holes[hi][0] <= lo:
          lo = begin + holes[hi][1]
          continue
        nxt = min(up, begin + holes[hi][0]) if hi < len(holes) else up
        if x[lo - pos:nxt - pos] != y[lo - pos:nxt - pos]:
          bad.append(("bytes", lo))
        compared += nxt - lo
        lo = nxt
      pos = up
  return {"src_sha256": hs.hexdigest(), "dst_sha256": hd.hexdigest(), "bytes": pos, "compared": compared, "bad": bad[:10],
          "bad_count": len(bad)}


def build(bundle, dst, packed, plans, changed, rep, src_sha):
  t0 = time.time()
  if os.path.exists(dst):
    raise RuntimeError(f"{dst} exists")
  if len(packed) > bundle.data_start:
    raise RuntimeError(f"the new structure ({len(packed)} B) does not fit before the weights ({bundle.data_start} B)")
  tmp = dst + ".tmp"
  if os.path.exists(tmp):
    os.remove(tmp)
  W.clone(bundle.path, tmp)
  try:
    holes = []
    with open(tmp, "r+b") as f:
      f.seek(bundle.begin)
      f.write(packed)
      f.write(b"\0" * (bundle.data_start - len(packed)))
      for p in plans.values():
        for off, data in p["writes"]:
          f.seek(bundle.begin + off)
          f.write(data)
          holes.append((off, off + len(data)))
      f.flush()
      os.fsync(f.fileno())
    chk = {"size_same": os.path.getsize(tmp) == bundle.size, "sections_same": W.header_sections(tmp) == bundle.secs}
    with open(bundle.path, "rb") as a, open(tmp, "rb") as b:
      chk["head_same"] = a.read(bundle.begin) == b.read(bundle.begin)
    cmp_ = stream_compare(bundle.path, tmp, bundle.begin, bundle.data_start, holes, bundle.end)
    chk["stream"] = cmp_
    chk["complement_same"] = cmp_["bad_count"] == 0
    chk["hole_bytes"] = sum(b - a for a, b in holes)
    if src_sha and cmp_["src_sha256"] != src_sha:
      raise RuntimeError(f"source sha256 {cmp_['src_sha256']} != {src_sha}")
    with open(tmp, "rb") as f:
      mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
      M = fb.Model.GetRootAsModel(mm, bundle.begin)
      ds, _ = G.external_region(M)
      chk["data_start_same"] = ds == bundle.data_start
      # (1) every source buffer that is not in the changed list: same index, same bytes / offset / size
      src_model = W.Bundle(bundle.path).model
      ok_b, bad_b = 0, []
      for i, a in enumerate(src_model.buffers):
        if i in changed:
          continue
        y = M.Buffers(i)
        if a.offset is not None and a.offset > 1:
          good = y.Offset() == a.offset and y.Size() == a.size
        elif a.data is not None and len(a.data):
          good = y.DataLength() == len(a.data) and bytes(y.DataAsNumpy()) == bytes(a.data)
        else:
          good = y.DataLength() == 0 and y.Offset() <= 1
        ok_b += good
        if not good:
          bad_b.append(i)
      chk["buffers_same"], chk["buffers_differ"] = ok_b, bad_b[:20]
      chk["buffers_src"], chk["buffers_dst"] = len(src_model.buffers), M.BuffersLength()
      # (2) no live tensor outside the changed and merged buffers reads a rewritten range
      merged_bufs = set()
      for p in plans.values():
        merged_bufs.add(p["buffer"])
        if "scales_buffer" in p:
          merged_bufs.add(p["scales_buffer"])
      hs = sorted(holes)
      overlap = []
      for si in range(M.SubgraphsLength()):
        sg = M.Subgraphs(si)
        for ti in range(sg.TensorsLength()):
          bi = sg.Tensors(ti).Buffer()
          if bi in changed or bi in merged_bufs:
            continue
          b = M.Buffers(bi)
          if b.Offset() > 1:
            lo, up = b.Offset(), b.Offset() + b.Size()
            if any(lo < y and x < up for x, y in hs):
              overlap.append((si, ti, bi))
      chk["live_overlap"] = overlap[:10]
      # (3) merged tensors read back through the new structure
      rb_ok, rb_bad = 0, []
      for si in range(M.SubgraphsLength()):
        sg = M.Subgraphs(si)
        for ti in range(sg.TensorsLength()):
          t = sg.Tensors(ti)
          nm = (t.Name() or b"").decode()
          m = re.search(r"model\.layers\.(\d+)\.self_attn\.qkv_proj\.weight$", nm)
          if not m:
            continue
          p = plans[int(m.group(1))]
          R, C = sum(p["rows"].values()), p["C"]
          b = M.Buffers(t.Buffer())
          code_bytes = mm[bundle.begin + b.Offset(): bundle.begin + b.Offset() + b.Size()]
          good = list(t.ShapeAsNumpy()) == [R, C] and hashlib.sha256(code_bytes).hexdigest() == p["src_sha"]["codes"]
          q = t.Quantization()
          if p["kind"] == "int8":
            good = good and t.Type() == TT.INT8 and q.ScaleLength() == R and q.ZeroPointLength() == R
          else:
            d = fb.BlockwiseQuantization()
            d.Init(q.Details().Bytes, q.Details().Pos)
            st = sg.Tensors(d.Scales())
            sb = M.Buffers(st.Buffer())
            sbytes = mm[bundle.begin + sb.Offset(): bundle.begin + sb.Offset() + sb.Size()]
            good = (good and t.Type() == TT.INT4 and st.Type() == TT.FLOAT16 and
                    (st.Name() or b"").decode() == nm + "_scales" and d.BlockSize() == 32 and d.ZeroPoints() == -1 and
                    hashlib.sha256(sbytes).hexdigest() == p["src_sha"]["scales"])
          rb_ok += good
          if not good:
            rb_bad.append((si, ti, nm))
      chk["readback_ok"], chk["readback_bad"] = rb_ok, rb_bad[:10]
      # (4) every moved buffer: the source's bytes at its old offset = the written bytes at its new offset
      src_b = W.Bundle(bundle.path)
      mv_ok, mv_bad = 0, []
      for p in plans.values():
        for bi, (off, size) in p["moves"].items():
          y = M.Buffers(bi)
          if y.Offset() <= 1:  # emptied: no reader left
            continue
          good = (y.Offset() == off and y.Size() == size and
                  bytes(mm[bundle.begin + off: bundle.begin + off + size]) == bytes(src_b.buffer_bytes(bi)))
          mv_ok += good
          if not good:
            mv_bad.append(bi)
      chk["moved_ok"], chk["moved_bad"] = mv_ok, mv_bad[:10]
      mm.close()
    rep["asserts"] = chk
    bad = [k for k in ("size_same", "sections_same", "head_same", "complement_same", "data_start_same") if not chk[k]]
    if bad_b:
      bad.append("buffers_differ")
    if overlap:
      bad.append("live_overlap")
    if rb_bad:
      bad.append("readback_bad")
    if mv_bad:
      bad.append("moved_bad")
    if bad:
      raise RuntimeError(f"asserts failed: {bad}: {json.dumps(chk, default=str)[:3000]}")
    rep["dst_sha256"] = cmp_["dst_sha256"]
    rep["src_sha256"] = cmp_["src_sha256"]
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
  ap.add_argument("--dry", action="store_true", help="the rewrite in memory only: sizes, counts, the structure sha256")
  ap.add_argument("--src-sha256", help="expected sha256 of SRC (checked during the asserts' read)")
  ap.add_argument("--report", help="write a JSON report here")
  a = ap.parse_args()
  if not a.dry and not a.dst:
    raise SystemExit("give DST, or --dry")
  b = W.Bundle(a.src)
  before = W.census(b.model)
  snap = W.blockwise_snapshot(b.model)
  rep, plans, merged_names, changed = rewrite(b)
  for k, names in merged_names.items():
    for n in names:
      snap.get(k, {}).pop(n, None)
  rep["blockwise"] = W.blockwise_check(b.model, snap)
  bad_bw = {k: x for k, x in rep["blockwise"].items() if x["wrong_count"]}
  after = W.census(b.model)
  rep["census"] = {k: {"ops_before": before[k]["ops"], "ops_after": after[k]["ops"],
                       "delta": {n: after[k]["by_type"].get(n, 0) - before[k]["by_type"].get(n, 0)
                                 for n in sorted(set(before[k]["by_type"]) | set(after[k]["by_type"]))
                                 if after[k]["by_type"].get(n, 0) != before[k]["by_type"].get(n, 0)}}
                   for k in after if k in before}
  packed = G.pack_model(b.model)
  rep.update({"src": a.src, "structure_bytes": len(packed), "data_start": b.data_start,
              "fits": len(packed) <= b.data_start, "structure_sha256": hashlib.sha256(packed).hexdigest(),
              "tools": {"decode_rewrite.py": W.file_sha256(os.path.abspath(__file__)),
                        "gpu_rewrite.py": W.file_sha256(PUBLIC_REWRITE)},
              "subgraphs": len(b.model.subgraphs), "buffers": len(b.model.buffers)})
  print(f"added {rep['added']}, decompositions {rep['decompositions_added']}, structure {len(packed)} / {b.data_start} B "
        f"sha256 {rep['structure_sha256']}, qkv buffers {rep['qkv_buffers']}")
  for k, x in rep["census"].items():
    print(f"  {k}: ops {x['ops_before']} -> {x['ops_after']} {x['delta']}")
  for k, s in rep["signatures"].items():
    if s.get("skipped"):
      print(f"  {k}: skipped {len(s['skipped'])}: {s['skipped'][:3]}")
  if bad_bw:
    raise SystemExit(f"blockwise references changed, nothing written: {json.dumps(bad_bw, default=str)[:1500]}")
  if not rep["fits"]:
    raise SystemExit("the new structure does not fit before the weights")
  if not a.dry:
    build(b, a.dst, packed, plans, changed, rep, a.src_sha256)
    print(f"  wrote {a.dst} sha256 {rep['dst_sha256']} asserts ok ({rep['build_seconds']} s)")
  if a.report:
    with open(a.report, "w") as f:
      json.dump(rep, f, indent=1, default=str)
  return 0


if __name__ == "__main__":
  sys.exit(main())
