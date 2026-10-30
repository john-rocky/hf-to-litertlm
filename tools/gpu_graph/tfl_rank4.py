#!/usr/bin/env python3
"""tfl_rank4.py — rewrite a .tflite so the ML Drift GPU front end places every operand of the flagged ops the same
way, and cut sub-graphs out of a .tflite for a CPU-vs-GPU bisect.

Placement (LiteRT tflite/delegates/gpu/common/model_builder_helper.cc ExtractTensorShape, model_builder.cc
ParseInputsWithConstTensorImpl): a runtime tensor of rank 2 [a,b] is BHWC(a,1,1,b), rank 3 [a,b,c] is BHWC(a,1,b,c);
a non-scalar, non-linear constant of rank 2 [a,b] is HWC(1,a,b), rank 3 [a,b,c] is HWC(a,b,c). So a runtime [1024,16,d]
times a constant [1024,1,d] (RoPE) reads the constant at H = 0 for every token, and [1024,C] + const [1024,C] (a
position table) reads row 0. Rules (each switchable):
  elem    binary elementwise op, one runtime operand, one constant that is neither scalar nor per-channel: the runtime
          operand is reshaped to rank 4 with leading 1s, the constant is stored at the full rank-4 output shape
          (broadcast materialized), the op writes a rank-4 tensor, a RESHAPE restores the original output shape
  bmm     BATCH_MATMUL of two runtime rank-3 operands -> [1,B,M,K] x [1,B,K,N], RESHAPE back
  mean    MEAN / SUM with keep_dims false whose output placement differs from the reduced input placement ->
          keep_dims true, RESHAPE to the original output shape
  sslice  STRIDED_SLICE / SLICE of a rank <= 3 input with a leading dim != 1 that slices axis 0 -> rank 4 input
  softmax SOFTMAX of a rank-4 input with a leading dim != 1 -> [1, B*H, W, C], RESHAPE back
Every inserted RESHAPE keeps the data order (row-major), so the rewrite computes the same function on the CPU.

The file keeps the external-buffer layout (> 2 GB graphs): the flatbuffer is rebuilt, every referenced constant is
copied after it at a 64-byte boundary, new constants are appended the same way.

  rewrite IN OUT [--rules elem,bmm,mean,sslice,softmax] [--report R.json]
  slice   IN OUT [--inputs T,...] --outputs T,...      T = tensor index or op:K (first output of op K)
  probe   IN OUT --ops A:B                              every op output in [A,B) becomes a graph output
  (slice / probe may be combined with --rules: the rewrite runs first, op numbers refer to the input file)
"""
import argparse
import json
import mmap
import os
import sys

import flatbuffers
import numpy as np
from ai_edge_litert import schema_py_generated as s

OPNAMES = {v: k for k, v in s.BuiltinOperator.__dict__.items() if not k.startswith("_")}
ELEMWISE = {"ADD", "SUB", "MUL", "DIV", "SQUARED_DIFFERENCE", "MAXIMUM", "MINIMUM"}
NP = {s.TensorType.FLOAT32: np.float32, s.TensorType.INT32: np.int32, s.TensorType.FLOAT16: np.float16,
      s.TensorType.INT64: np.int64, s.TensorType.INT8: np.int8, s.TensorType.UINT8: np.uint8}
ALIGN = 64


class Graph:
    def __init__(self, path):
        self.f = open(path, "rb")
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        self.m = s.ModelT.InitFromPackedBuf(self.mm, 0)
        self.sg = self.m.subgraphs[0]
        assert len(self.m.subgraphs) == 1, "one subgraph only"
        self.new_data = {}  # buffer index -> bytes of a constant this tool created
        for t in self.sg.tensors:
            t.shape = [int(x) for x in t.shape] if t.shape is not None else []
            t.shapeSignature = None
        for op in self.sg.operators:
            op.inputs = [int(x) for x in op.inputs]
            op.outputs = [int(x) for x in op.outputs]
        self.sg.inputs = [int(x) for x in self.sg.inputs]
        self.sg.outputs = [int(x) for x in self.sg.outputs]

    def opname(self, op):
        oc = self.m.operatorCodes[op.opcodeIndex]
        return OPNAMES.get(max(oc.builtinCode, oc.deprecatedBuiltinCode), str(oc.builtinCode))

    def opcode(self, name):
        code = getattr(s.BuiltinOperator, name)
        for i, oc in enumerate(self.m.operatorCodes):
            if max(oc.builtinCode, oc.deprecatedBuiltinCode) == code:
                return i
        oc = s.OperatorCodeT()
        oc.builtinCode = code
        oc.deprecatedBuiltinCode = min(code, 127)
        oc.version = 1
        self.m.operatorCodes.append(oc)
        return len(self.m.operatorCodes) - 1

    def is_const(self, ti):
        t = self.sg.tensors[ti]
        if t.buffer in self.new_data:
            return True
        b = self.m.buffers[t.buffer]
        return b.offset > 1 or (b.data is not None and len(b.data) > 0)

    def const(self, ti):
        t = self.sg.tensors[ti]
        dt = NP[t.type]
        if t.buffer in self.new_data:
            raw = self.new_data[t.buffer]
        else:
            b = self.m.buffers[t.buffer]
            raw = self.mm[b.offset:b.offset + b.size] if b.offset > 1 else bytes(np.asarray(b.data, np.uint8))
        return np.frombuffer(raw, dtype=dt).reshape(t.shape)

    def add_buffer(self, arr):
        b = s.BufferT()
        b.offset, b.size = 2, arr.nbytes  # placeholder, laid out at write time
        self.m.buffers.append(b)
        self.new_data[len(self.m.buffers) - 1] = np.ascontiguousarray(arr).tobytes()
        return len(self.m.buffers) - 1

    def add_tensor(self, shape, ttype, name, buffer=0):
        t = s.TensorT()
        t.shape = [int(x) for x in shape]
        t.type = ttype
        t.name = name.encode() if isinstance(name, str) else name
        t.buffer = buffer
        t.hasRank = True
        self.sg.tensors.append(t)
        return len(self.sg.tensors) - 1

    def add_const(self, arr, name):
        ttype = {np.dtype(np.float32): s.TensorType.FLOAT32, np.dtype(np.int32): s.TensorType.INT32}[arr.dtype]
        return self.add_tensor(arr.shape, ttype, name, self.add_buffer(arr))

    def reshape_op(self, src, dst):
        shape = np.array(self.sg.tensors[dst].shape, np.int32)
        st = self.add_const(shape, self.sg.tensors[dst].name + b"/r4_shape")
        op = s.OperatorT()
        op.opcodeIndex = self.opcode("RESHAPE")
        op.inputs, op.outputs = [src, st], [dst]
        op.builtinOptionsType = s.BuiltinOptions.ReshapeOptions
        op.builtinOptions = s.ReshapeOptionsT()
        op.builtinOptions.newShape = [int(x) for x in shape]
        return op


def rank4(shape):
    shape = list(shape)
    assert len(shape) <= 4, shape
    return [1] * (4 - len(shape)) + shape


def bcast(*shapes):
    return list(np.broadcast_shapes(*[tuple(x) for x in shapes]))


def runtime_bhwc(shape):
    r = len(shape)
    if r == 0:
        return (1, 1, 1, 1)
    if r == 1:
        return (shape[0], 1, 1, 1)
    if r == 2:
        return (shape[0], 1, 1, shape[1])
    if r == 3:
        return (shape[0], 1, shape[1], shape[2])
    return tuple(shape) if r == 4 else None


def rewrite(g, rules, report):
    sg = g.sg
    out_ops = []
    cache = {}  # (tensor, shape) -> reshaped tensor, inserted once at the first use
    ccache = {}  # (constant tensor, shape) -> materialized rank-4 constant (RoPE tables are shared by every layer)

    def lifted(ti, shape, pre):
        key = (ti, tuple(shape))
        if key not in cache:
            t = sg.tensors[ti]
            nt = g.add_tensor(shape, t.type, t.name + b"/r4")
            pre.append(g.reshape_op(ti, nt))
            cache[key] = nt
        return cache[key]

    for k, op in enumerate(sg.operators):
        name = g.opname(op)
        pre, post, site = [], [], None
        ins = op.inputs
        if name in ELEMWISE and "elem" in rules and len(ins) == 2:
            c = [g.is_const(i) for i in ins]
            if c[0] != c[1]:
                ci, ri = (0, 1) if c[0] else (1, 0)
                cs, rs = sg.tensors[ins[ci]].shape, sg.tensors[ins[ri]].shape
                n = int(np.prod(cs)) if cs else 1
                linear = len(cs) >= 1 and all(x == 1 for x in cs[:-1])
                if n > 1 and not linear:
                    out = op.outputs[0]
                    oshape = sg.tensors[out].shape
                    o4 = rank4(oshape)
                    assert rank4(rs) == o4, f"op {k}: runtime operand {rs} is broadcast to {oshape}"
                    ckey = (ins[ci], tuple(o4))
                    if ckey not in ccache:
                        carr = np.broadcast_to(g.const(ins[ci]).reshape(rank4(cs)), o4)
                        ccache[ckey] = g.add_const(np.ascontiguousarray(carr), sg.tensors[ins[ci]].name + b"/r4")
                    newc = ccache[ckey]
                    newr = lifted(ins[ri], o4, pre)
                    o4t = g.add_tensor(o4, sg.tensors[out].type, sg.tensors[out].name + b"/r4")
                    op.inputs = [newc, newr] if ci == 0 else [newr, newc]
                    op.outputs = [o4t]
                    post.append(g.reshape_op(o4t, out))
                    site = {"rule": "elem", "runtime": rs, "const": cs, "out4": o4}
        elif name == "BATCH_MATMUL" and "bmm" in rules:
            shapes = [sg.tensors[i].shape for i in ins]
            if all(not g.is_const(i) for i in ins) and all(len(x) == 3 for x in shapes):
                out = op.outputs[0]
                op.inputs = [lifted(i, rank4(sh), pre) for i, sh in zip(ins, shapes)]
                o4t = g.add_tensor(rank4(sg.tensors[out].shape), sg.tensors[out].type, sg.tensors[out].name + b"/r4")
                op.outputs = [o4t]
                post.append(g.reshape_op(o4t, out))
                site = {"rule": "bmm", "shapes": shapes}
        elif name in ("MEAN", "SUM") and "mean" in rules:
            opt = op.builtinOptions
            ishape = sg.tensors[ins[0]].shape
            out = op.outputs[0]
            if opt is not None and not opt.keepDims and g.is_const(ins[1]):
                axes = [int(a) % len(ishape) for a in np.atleast_1d(g.const(ins[1]))]
                kshape = [1 if i in axes else d for i, d in enumerate(ishape)]
                b_in = runtime_bhwc(kshape)
                b_out = runtime_bhwc(sg.tensors[out].shape)
                if b_in != b_out:
                    opt.keepDims = True
                    kt = g.add_tensor(kshape, sg.tensors[out].type, sg.tensors[out].name + b"/r4")
                    op.outputs = [kt]
                    post.append(g.reshape_op(kt, out))
                    site = {"rule": "mean", "in": ishape, "out": sg.tensors[out].shape, "keep": kshape}
        elif name == "STRIDED_SLICE" and "sslice" in rules:
            ishape = sg.tensors[ins[0]].shape
            if len(ishape) <= 3 and ishape[0] != 1:
                b, e, st = (g.const(i) for i in ins[1:4])
                opt = op.builtinOptions
                if not (opt.ellipsisMask or opt.newAxisMask or opt.shrinkAxisMask):
                    pad = 4 - len(ishape)
                    nb = g.add_const(np.concatenate([np.zeros(pad, np.int32), b]).astype(np.int32), b"r4_begin")
                    ne = g.add_const(np.concatenate([np.ones(pad, np.int32), e]).astype(np.int32), b"r4_end")
                    ns = g.add_const(np.concatenate([np.ones(pad, np.int32), st]).astype(np.int32), b"r4_strides")
                    opt.beginMask = int(opt.beginMask) << pad
                    opt.endMask = int(opt.endMask) << pad
                    out = op.outputs[0]
                    o4t = g.add_tensor(rank4(sg.tensors[out].shape), sg.tensors[out].type,
                                       sg.tensors[out].name + b"/r4")
                    op.inputs = [lifted(ins[0], rank4(ishape), pre), nb, ne, ns]
                    op.outputs = [o4t]
                    post.append(g.reshape_op(o4t, out))
                    site = {"rule": "sslice", "in": ishape}
        elif name == "SOFTMAX" and "softmax" in rules:
            ishape = sg.tensors[ins[0]].shape
            if len(ishape) == 4 and ishape[0] != 1:
                m4 = [1, ishape[0] * ishape[1], ishape[2], ishape[3]]
                out = op.outputs[0]
                op.inputs = [lifted(ins[0], m4, pre)]
                mt = g.add_tensor(m4, sg.tensors[out].type, sg.tensors[out].name + b"/r4")
                op.outputs = [mt]
                post.append(g.reshape_op(mt, out))
                site = {"rule": "softmax", "in": ishape}
        if site is not None:
            site.update({"op": k, "name": name})
            report.append(site)
        out_ops.extend(pre + [op] + post)
    sg.operators = out_ops


def resolve(g, specs, opmap):
    res = []
    for x in specs:
        if x.startswith("op:"):
            res.append(opmap[int(x[3:])])
        else:
            res.append(int(x))
    return res


def cut(g, inputs, outputs):
    """Keep only the ops that compute `outputs` from `inputs` (+ constants); renumber tensors and buffers."""
    sg = g.sg
    producer = {}
    for k, op in enumerate(sg.operators):
        for o in op.outputs:
            producer[o] = k
    need, keep = set(outputs), set()
    stack = list(outputs)
    stop = set(inputs)
    while stack:
        t = stack.pop()
        if t in stop or t not in producer:
            continue
        k = producer[t]
        if k in keep:
            continue
        keep.add(k)
        for i in sg.operators[k].inputs:
            if i >= 0 and i not in need:
                need.add(i)
                stack.append(i)
    ops = [op for k, op in enumerate(sg.operators) if k in keep]
    for op in ops:
        for i in op.inputs:
            if i >= 0 and not g.is_const(i) and i not in stop and i not in producer:
                raise SystemExit(f"tensor {i} ({sg.tensors[i].name}) is neither an input nor computed")
    sg.operators = ops
    sg.inputs, sg.outputs = list(inputs), list(outputs)


def compact(g):
    """Drop unused tensors and buffers; renumber. Signature defs follow the subgraph inputs / outputs."""
    sg = g.sg
    used = []
    seen = set()
    for t in list(sg.inputs) + [i for op in sg.operators for i in list(op.inputs) + list(op.outputs)] + list(sg.outputs):
        if t >= 0 and t not in seen:
            seen.add(t)
            used.append(t)
    tmap = {t: n for n, t in enumerate(used)}
    old_t = sg.tensors
    sg.tensors = [old_t[t] for t in used]
    for op in sg.operators:
        op.inputs = [tmap[i] if i >= 0 else -1 for i in op.inputs]
        op.outputs = [tmap[i] for i in op.outputs]
    sg.inputs = [tmap[i] for i in sg.inputs]
    sg.outputs = [tmap[i] for i in sg.outputs]
    bused = [0]
    for t in sg.tensors:
        if t.buffer and t.buffer not in bused:
            bused.append(t.buffer)
    meta = [md.buffer for md in (g.m.metadata or [])]
    for b in meta:
        if b not in bused:
            bused.append(b)
    bmap = {b: n for n, b in enumerate(bused)}
    old_b = g.m.buffers
    g.m.buffers = [old_b[b] for b in bused]
    g.new_data = {bmap[b]: d for b, d in g.new_data.items() if b in bmap}
    for t in sg.tensors:
        t.buffer = bmap[t.buffer]
    for md in (g.m.metadata or []):
        md.buffer = bmap[md.buffer]
    g.m.metadataBuffer = None
    sig = s.SignatureDefT()
    old = g.m.signatureDefs[0] if g.m.signatureDefs else None
    sig.signatureKey = old.signatureKey if old else b"serving_default"
    sig.subgraphIndex = 0
    names_in = {tm.tensorIndex: tm.name for tm in (old.inputs if old else [])}
    names_out = {tm.tensorIndex: tm.name for tm in (old.outputs if old else [])}
    rev = {n: t for t, n in tmap.items()}
    sig.inputs, sig.outputs = [], []
    for kind, lst, names in (("in", sg.inputs, names_in), ("out", sg.outputs, names_out)):
        for n, ti in enumerate(lst):
            tm = s.TensorMapT()
            tm.tensorIndex = ti
            tm.name = names.get(rev[ti]) or f"{kind}{n}_t{rev[ti]}".encode()
            (sig.inputs if kind == "in" else sig.outputs).append(tm)
    g.m.signatureDefs = [sig]


def write(g, out):
    """Flatbuffer first, then every external / new constant at a 64-byte boundary (offsets absolute)."""
    bufs = g.m.buffers
    ext = [i for i, b in enumerate(bufs) if i in g.new_data or b.offset > 1]
    src_ext = {i: (bufs[i].offset, bufs[i].size) for i in ext if i not in g.new_data}
    for i in ext:
        bufs[i].offset = 2  # same packed size for any non-zero value
        bufs[i].data = None

    def pack():
        b = flatbuffers.Builder(1 << 20)
        b.Finish(g.m.Pack(b), file_identifier=b"TFL3")
        return bytes(b.Output())

    hdr = pack()
    pos = (len(hdr) + 4095) // 4096 * 4096
    layout = {}
    for i in ext:
        size = len(g.new_data[i]) if i in g.new_data else src_ext[i][1]
        layout[i] = pos
        bufs[i].offset, bufs[i].size = pos, size
        pos = (pos + size + ALIGN - 1) // ALIGN * ALIGN
    hdr2 = pack()
    assert len(hdr2) == len(hdr), (len(hdr2), len(hdr))
    tmp = out + ".part"
    with open(tmp, "wb") as f:
        f.write(hdr2)
        for i in ext:
            f.seek(layout[i])
            if i in g.new_data:
                f.write(g.new_data[i])
            else:
                o, n = src_ext[i]
                step = 1 << 26
                for a in range(0, n, step):
                    f.write(g.mm[o + a:o + min(n, a + step)])
        f.truncate(pos)
    os.replace(tmp, out)
    return len(hdr2), pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["rewrite", "slice", "probe"])
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--rules", default="")
    ap.add_argument("--inputs", default="")
    ap.add_argument("--outputs", default="")
    ap.add_argument("--ops", default="")
    ap.add_argument("--report")
    a = ap.parse_args()
    g = Graph(a.src)
    opmap = {k: op.outputs[0] for k, op in enumerate(g.sg.operators)}
    names = {k: g.opname(op) for k, op in enumerate(g.sg.operators)}
    rules = set(x for x in a.rules.split(",") if x)
    if a.cmd == "rewrite" and not rules:
        rules = {"elem", "bmm", "mean", "sslice", "softmax"}
    report = []
    if rules:
        rewrite(g, rules, report)
    if a.cmd == "slice":
        ins = resolve(g, [x for x in a.inputs.split(",") if x], opmap) or list(g.sg.inputs)
        cut(g, ins, resolve(g, a.outputs.split(","), opmap))
    elif a.cmd == "probe":
        lo, hi = (int(x) for x in a.ops.split(":"))
        outs = [opmap[k] for k in range(lo, hi)]
        cut(g, list(g.sg.inputs), outs)
    compact(g)
    hlen, size = write(g, a.dst)
    by_rule = {}
    for r in report:
        by_rule[r["rule"]] = by_rule.get(r["rule"], 0) + 1
    print(f"wrote {a.dst}: ops {len(g.sg.operators)} tensors {len(g.sg.tensors)} header {hlen} B file {size} B; "
          f"sites {by_rule}", flush=True)
    if a.report:
        json.dump({"src": a.src, "dst": a.dst, "rules": sorted(rules), "sites": report,
                   "probe_ops": ({k: names[k] for k in range(*map(int, a.ops.split(":")))} if a.cmd == "probe" else None)},
                  open(a.report, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
