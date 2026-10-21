#!/usr/bin/env python3
"""bf16to32.py <in.litertlm> <out.litertlm> [<report.json>] — rewrite every BFLOAT16 tensor of a .litertlm's TFLite
sections as FLOAT32. litert-torch's `model_dtype="bfloat16"` export computes in FLOAT32 and puts
BFLOAT16 on the KV cache, on the composites' inputs / outputs and on some constants, joined by CAST ops; LiteRT-LM's
GPU delegate refuses BFLOAT16 tensors and XNNPACK leaves the CASTs to the reference kernels. Per TFLite section and
subgraph (the signatures and the composites' decomposition subgraphs):
  1. a BFLOAT16 tensor -> FLOAT32; a BFLOAT16 constant -> the same values in float32 (u32 = u16 << 16, exact);
  2. a CAST that now reads and writes FLOAT32 is removed: its readers (ops, the subgraph's and the signature's outputs)
     read its input, and its output tensor, read by nothing any more, is dropped;
  3. the model is written again in the converter's layout: the flatbuffer, then each buffer over 512 B outside it
     (offset / size from the model's first byte, 16-byte aligned, in buffer order), smaller ones inline.
The container keeps the input's header byte for byte except the sections' begin / end offsets (16 KiB blocks); every
other section is copied byte for byte. It refuses a table field this schema does not know (the rewrite would drop it)
and a buffer shared by a BFLOAT16 and another tensor type. Needs ai-edge-litert, litert-lm-builder, flatbuffers, numpy.
"""
import collections
import hashlib
import json
import mmap
import os
import re
import sys
import time

import flatbuffers
import numpy as np
from ai_edge_litert import schema_py_generated as S
from ai_edge_litert.tools.flatbuffer_utils import update_packed_buffer
from litert_lm_builder import litertlm_core as C
from litert_lm_builder import litertlm_header_schema_py_generated as H

BF16, F32 = S.TensorType.BFLOAT16, S.TensorType.FLOAT32
INLINE_MAX, ALIGN, CHUNK = 512, 16, 1 << 26
TYPED = {S.CastOptionsT: ("inDataType", "outDataType"), S.ArgMaxOptionsT: ("outputType",),   # options holding a
         S.ArgMinOptionsT: ("outputType",), S.ShapeOptionsT: ("outType",), S.UniqueOptionsT: ("idxOutType",)}   # TensorType
up = lambda x, a: (x + a - 1) // a * a


def known_slots():
    """per schema class, the last vtable slot its accessors read"""
    known, cls = {}, None
    for line in open(S.__file__, encoding="utf-8"):
        m = re.match(r"class (\w+)\(object\):", line)
        cls = m.group(1) if m else cls
        m = re.search(r"self\._tab\.Offset\((\d+)\)", line)
        if m:
            known[cls] = max(known.get(cls, 0), int(m.group(1)))
    return known


def unknown_fields(fb):
    """tables that carry a field past the slots this schema knows: {class: count}"""
    known, bad = known_slots(), collections.Counter()
    u16, s32 = flatbuffers.number_types.VOffsetTFlags, flatbuffers.number_types.SOffsetTFlags
    names = {u: {v: k for k, v in vars(u).items() if not k.startswith("_")}
             for u in (S.BuiltinOptions, S.BuiltinOptions2, S.QuantizationDetails)}

    def chk(obj, name):
        t = obj._tab
        vt = t.Pos - t.Get(s32, t.Pos)
        if any(t.Get(u16, vt + o) for o in range(max(known.get(name, 0) + 2, 4), t.Get(u16, vt), 2)):
            bad[name] += 1

    def union(typ, tab, u):
        if typ and tab is not None:
            o = getattr(S, names[u][typ])()
            o.Init(tab.Bytes, tab.Pos)
            chk(o, names[u][typ])
    chk(fb, "Model")
    for get, n, name in ((fb.OperatorCodes, fb.OperatorCodesLength(), "OperatorCode"), (fb.Buffers, fb.BuffersLength(),
                         "Buffer"), (fb.Metadata, fb.MetadataLength(), "Metadata")):
        for i in range(n):
            chk(get(i), name)
    for i in range(fb.SignatureDefsLength()):
        sd = fb.SignatureDefs(i)
        chk(sd, "SignatureDef")
        for tm in [sd.Inputs(j) for j in range(sd.InputsLength())] + [sd.Outputs(j) for j in range(sd.OutputsLength())]:
            chk(tm, "TensorMap")
    for si in range(fb.SubgraphsLength()):
        sg = fb.Subgraphs(si)
        chk(sg, "SubGraph")
        for ti in range(sg.TensorsLength()):
            t = sg.Tensors(ti)
            chk(t, "Tensor")
            q = t.Quantization()
            if q is not None:
                chk(q, "QuantizationParameters")
                union(q.DetailsType(), q.Details(), S.QuantizationDetails)
        for oi in range(sg.OperatorsLength()):
            o = sg.Operators(oi)
            chk(o, "Operator")
            union(o.BuiltinOptionsType(), o.BuiltinOptions(), S.BuiltinOptions)
            union(o.BuiltinOptions2Type(), o.BuiltinOptions2(), S.BuiltinOptions2)
    return dict(bad)


def qref(t):
    """the tensor indices a tensor's quantization details name (blockwise / multi-axis: scales, zero points)"""
    d = t.quantization.details if t.quantization is not None else None
    if d is None or not isinstance(getattr(d, "scales", None), int):
        return ()
    return (d.scales,) + ((d.zeroPoints,) if isinstance(d.zeroPoints, int) else ())


def has_bf16(fb):
    return any(fb.Subgraphs(si).Tensors(ti).Type() == BF16 for si in range(fb.SubgraphsLength())
               for ti in range(fb.Subgraphs(si).TensorsLength()))


def to_f32(m, view, rep):
    """steps 1 and 2 on the object tree; returns {buffer index: float32 bytes}"""
    cast = {i for i, c in enumerate(m.operatorCodes) if max(c.builtinCode, c.deprecatedBuiltinCode) == S.BuiltinOperator.CAST}
    users = collections.defaultdict(set)
    for sg in m.subgraphs:
        for t in sg.tensors:
            if t.buffer:
                users[t.buffer].add(t.type)
    conv = sorted(b for b, ts in users.items() if BF16 in ts)
    if any(len(users[b]) > 1 for b in conv):
        sys.exit("refused: a buffer is shared by a BFLOAT16 tensor and a tensor of another type")
    sigs = collections.defaultdict(list)
    for sd in m.signatureDefs or []:
        sigs[sd.subgraphIndex].append(sd)
    for si, sg in enumerate(m.subgraphs):
        was = [t.type == BF16 for t in sg.tensors]
        if not any(was):
            continue
        rep["subgraphs"] += 1
        rep["bf16_tensors"] += sum(was)
        for t in sg.tensors:
            t.type = F32 if t.type == BF16 else t.type
        repl, keep = {}, []
        for op in sg.operators:
            ins = [int(x) for x in (op.inputs if op.inputs is not None else [])]
            outs = [int(x) for x in (op.outputs if op.outputs is not None else [])]
            for k in TYPED.get(type(op.builtinOptions), ()):
                if getattr(op.builtinOptions, k) == BF16:
                    setattr(op.builtinOptions, k, F32)
                    rep["typed_options"] += 1
            if (op.opcodeIndex in cast and len(ins) == 1 and len(outs) == 1 and (was[ins[0]] or was[outs[0]])
                    and sg.tensors[ins[0]].type == F32 and sg.tensors[outs[0]].type == F32):
                repl[outs[0]] = ins[0]
                rep["casts_removed"]["BFLOAT16->FLOAT32" if was[ins[0]] else "FLOAT32->BFLOAT16"] += 1
                continue
            keep.append(op)

        def res(x):
            while x in repl:
                x = repl[x]
            return x
        sg.operators = keep
        maps = [(tm, "out") for sd in sigs[si] for tm in sd.outputs] + [(tm, "in") for sd in sigs[si] for tm in sd.inputs]
        if any(d == "in" and tm.tensorIndex in repl for tm, d in maps):
            sys.exit(f"refused: subgraph {si}: a signature input is the output of a removed CAST")
        ins_sg = [int(x) for x in sg.inputs]
        outs_sg = [res(int(x)) for x in sg.outputs]
        rep["outputs_now_inputs"] += sum(1 for x in outs_sg if x in ins_sg)
        used = set(ins_sg) | set(outs_sg) | {res(tm.tensorIndex) for tm, _ in maps}
        for op in keep:
            op.inputs = [res(int(x)) if x >= 0 else int(x) for x in op.inputs]
            used |= set(op.inputs) | set(int(x) for x in op.outputs) | set(int(x) for x in (op.intermediates if op.intermediates is not None else []))
        drop = set(repl) - used
        if len(drop) != len(repl):
            sys.exit(f"refused: subgraph {si}: {len(repl) - len(drop)} removed CAST outputs are still read")
        if drop & {x for t in sg.tensors for x in qref(t)}:   # an omitted zero point reads as tensor 0
            sys.exit(f"refused: subgraph {si}: a quantization's scale / zero-point index names a removed CAST output")
        names = lambda: [(t.name, tuple(sg.tensors[x].name if x >= 0 else None for x in qref(t))) for t in sg.tensors
                         if qref(t)]
        before = names()
        new = {old: i for i, old in enumerate(x for x in range(len(sg.tensors)) if x not in drop)}
        mp = lambda x: new[x] if x >= 0 else x
        sg.tensors = [t for x, t in enumerate(sg.tensors) if x not in drop]
        for op in keep:
            op.inputs = [mp(x) for x in op.inputs]
            op.outputs = [mp(int(x)) for x in op.outputs]
            if op.intermediates is not None:
                op.intermediates = [mp(int(x)) for x in op.intermediates]
        sg.inputs, sg.outputs = [mp(x) for x in ins_sg], [mp(x) for x in outs_sg]
        for tm, _ in maps:
            tm.tensorIndex = mp(res(tm.tensorIndex))
        for t in sg.tensors:   # blockwise / multi-axis quantization name their scale / zero-point tensors by index
            d = t.quantization.details if t.quantization is not None else None
            if qref(t):
                d.scales = mp(d.scales)
                if d.zeroPoints is not None:
                    d.zeroPoints = mp(d.zeroPoints)
        n = len(sg.tensors)
        idx = [x for op in keep for x in list(op.inputs) + list(op.outputs) + list(op.intermediates or [])]
        idx += list(sg.inputs) + list(sg.outputs) + [tm.tensorIndex for tm, _ in maps] + [x for t in sg.tensors for x in qref(t)]
        if any(x >= n or x < -1 for x in idx) or names() != before:
            sys.exit(f"refused: subgraph {si}: a tensor index left out of place after the rewrite (a bug of this tool)")
        rep["tensors_dropped"] += len(drop)
    out = {}
    for b in conv:
        bu = m.buffers[b]
        raw = (bytes(bu.data) if bu.data is not None and len(bu.data) else
               bytes(view[bu.offset:bu.offset + bu.size]) if bu.size else b"")
        if raw:
            out[b] = (np.frombuffer(raw, "<u2").astype("<u4") << 16).tobytes()
            rep["buffers_converted"] += 1
            rep["buffer_bytes"][0] += len(raw)
            rep["buffer_bytes"][1] += len(out[b])
    return out


def pack_buffer(self, builder):
    """BufferT.Pack with the schema's force_align: 16 on the inline data (the generated code aligns to 1)"""
    data = None
    if self.data is not None:
        payload = bytes(self.data)
        builder.StartVector(1, len(payload), ALIGN)
        builder.head -= len(payload)
        builder.Bytes[builder.head:builder.head + len(payload)] = payload
        builder.vectorNumElems = len(payload)
        data = builder.EndVector()
    S.BufferStart(builder)
    if data is not None:
        S.BufferAddData(builder, data)
    S.BufferAddOffset(builder, self.offset)
    S.BufferAddSize(builder, self.size)
    return S.BufferEnd(builder)


def write_tflite(view, src, base, out, rep):
    """steps 1-3 for one TFLite section (view = its bytes, base = its offset in the open input file src)"""
    fb = S.Model.GetRootAs(view, 0)
    bad = unknown_fields(fb)
    if bad:
        sys.exit(f"refused: fields this schema does not know {bad}: the rewrite would drop them")
    m = S.ModelT.InitFromObj(fb)
    if (m.externalBuffers or m.externalBufferGroups
            or any(op.largeCustomOptionsOffset for sg in m.subgraphs for op in sg.operators)
            or any(t.externalBuffer for sg in m.subgraphs for t in sg.tensors)):
        sys.exit("refused: external buffers or large custom options (offsets this relayout would leave stale)")
    outside0 = [bool(b.size) and not (b.data is not None and len(b.data)) for b in m.buffers]
    conv = to_f32(m, view, rep)
    plan = []   # (buffer index, size, converted bytes or None = copy from the input)
    for bi, bu in enumerate(m.buffers):
        data = conv.get(bi)
        size = len(data) if data is not None else (bu.size if outside0[bi] else 0)
        if outside0[bi] or (data is not None and any(outside0) and size > INLINE_MAX):
            plan.append((bi, size, data, bu.offset if data is None else None))
            bu.data, bu.offset, bu.size = None, 1, size   # offset 1: a placeholder that keeps the field in the table
            rep["moved_outside"] += int(data is not None and not outside0[bi])
        elif data is not None:
            bu.data = np.frombuffer(data, np.uint8)
    S.BufferT.Pack = pack_buffer
    b = flatbuffers.Builder(1 << 24)
    b.Finish(m.Pack(b), file_identifier=b"TFL3")
    flat = bytearray(b.Output())
    packed = S.Model.GetRootAs(flat, 0)
    pos, offs = up(len(flat), ALIGN), []
    for bi, size, _, _ in plan:
        update_packed_buffer(packed.Buffers(bi), offset=pos)
        offs.append(pos)
        pos = up(pos + size, ALIGN)
    start, h, buf = out.tell(), hashlib.sha256(), bytearray(CHUNK)

    def put(x):
        out.write(x)
        h.update(x)
    put(bytes(flat))
    for (bi, size, data, src_off), off in zip(plan, offs):
        put(b"\0" * (start + off - out.tell()))
        if data is not None:
            put(data)
            continue
        src.seek(base + src_off)
        while size:
            n = src.readinto(memoryview(buf)[:min(CHUNK, size)])
            put(memoryview(buf)[:n])
            size -= n
    rep["flatbuffer_bytes"].append(len(flat))
    return h.hexdigest()


def main(src_path, dst_path, rep_path=None):
    t0 = time.time()
    rep = {"input": src_path, "output": dst_path, "subgraphs": 0, "bf16_tensors": 0, "tensors_dropped": 0,
           "casts_removed": collections.Counter(), "typed_options": 0, "outputs_now_inputs": 0,
           "buffers_converted": 0, "buffer_bytes": [0, 0], "moved_outside": 0, "flatbuffer_bytes": [], "sections": []}
    src = open(src_path, "rb")
    mm = mmap.mmap(src.fileno(), 0, access=mmap.ACCESS_READ)
    hend = int.from_bytes(mm[C.HEADER_END_LOCATION_BYTE_OFFSET:C.HEADER_END_LOCATION_BYTE_OFFSET + 8], "little")
    md = H.LiteRTLMMetaData.GetRootAs(bytearray(mm[C.HEADER_BEGIN_BYTE_OFFSET:hend]), 0)
    objs = [md.SectionMetadata().Objects(i) for i in range(md.SectionMetadata().ObjectsLength())]
    head = bytearray(mm[:objs[0].BeginOffset()])
    with open(dst_path + ".part", "wb") as out:
        out.write(head)
        for so in objs:
            b, e, typ = so.BeginOffset(), so.EndOffset(), C.any_section_data_type_to_string(so.DataType())
            start = up(out.tell(), C.BLOCK_SIZE)
            out.write(b"\0" * (start - out.tell()))
            view = memoryview(mm)[b:e]
            if typ == "TFLiteModel" and has_bf16(S.Model.GetRootAs(view, 0)):
                sha = write_tflite(view, src, b, out, rep)
                how = "rewritten"
            else:
                h, buf, size = hashlib.sha256(), bytearray(CHUNK), e - b
                src.seek(b)
                while size:
                    n = src.readinto(memoryview(buf)[:min(CHUNK, size)])
                    out.write(memoryview(buf)[:n])
                    h.update(memoryview(buf)[:n])
                    size -= n
                sha, how = h.hexdigest(), "copied"
            end = out.tell()
            for slot, v in ((6, start), (8, end)):   # SectionObject.begin_offset / end_offset (uint64, present)
                p = C.HEADER_BEGIN_BYTE_OFFSET + so._tab.Pos + so._tab.Offset(slot)
                head[p:p + 8] = v.to_bytes(8, "little")
            rep["sections"].append({"type": typ, "in": [b, e], "out": [start, end], "how": how, "sha256": sha})
        out.seek(0)
        out.write(head)
    os.replace(dst_path + ".part", dst_path)
    rep["seconds"] = round(time.time() - t0, 2)
    rep["casts_removed"] = dict(rep["casts_removed"])
    print(json.dumps({k: v for k, v in rep.items() if k != "sections"}))
    for s in rep["sections"]:
        print(f"  {s['type']:22s} {s['how']:9s} [{s['in'][0]}, {s['in'][1]}) -> [{s['out'][0]}, {s['out'][1]}) {s['sha256'][:16]}")
    if rep_path:
        json.dump(rep, open(rep_path, "w"), indent=1)


def tflite_sections(path):
    f = open(path, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    hend = int.from_bytes(mm[C.HEADER_END_LOCATION_BYTE_OFFSET:C.HEADER_END_LOCATION_BYTE_OFFSET + 8], "little")
    sm = H.LiteRTLMMetaData.GetRootAs(bytearray(mm[C.HEADER_BEGIN_BYTE_OFFSET:hend]), 0).SectionMetadata()
    return f, {i: S.Model.GetRootAs(memoryview(mm)[sm.Objects(i).BeginOffset():sm.Objects(i).EndOffset()], 0)
               for i in range(sm.ObjectsLength()) if C.any_section_data_type_to_string(sm.Objects(i).DataType()) == "TFLiteModel"}


def out_of_range(m):
    """every tensor index the model names (ops, subgraph and signature inputs / outputs, quantization details), checked
    against its subgraph's tensor count: the count of offenders"""
    sigs = collections.defaultdict(list)
    for sd in m.signatureDefs or []:
        sigs[sd.subgraphIndex] += list(sd.inputs) + list(sd.outputs)
    bad = 0
    for si, sg in enumerate(m.subgraphs):
        n = len(sg.tensors)
        idx = [int(x) for op in sg.operators for a in (op.inputs, op.outputs, op.intermediates) if a is not None for x in a]
        idx += [int(x) for x in list(sg.inputs) + list(sg.outputs)] + [tm.tensorIndex for tm in sigs[si]]
        bad += sum(1 for x in idx if x >= n or x < -1) + sum(1 for t in sg.tensors for x in qref(t) if x >= n or x < -1)
    return bad


def renamed(o, m):
    """the references of the rewritten model m that name another tensor than the input o names at the same place, with
    the removed CASTs walked through (an op's input that read a removed CAST's output names that CAST's input): ops in
    order, their inputs / outputs / intermediates, the subgraphs' and signatures' inputs / outputs, and per tensor (the
    kept ones in order) its name, type (BFLOAT16 -> FLOAT32), shape and the tensors its quantization details name"""
    cast = {i for i, c in enumerate(o.operatorCodes) if max(c.builtinCode, c.deprecatedBuiltinCode) == S.BuiltinOperator.CAST}
    f32 = lambda ty: F32 if ty == BF16 else ty
    bad = collections.Counter()
    resolve = {}
    if len(o.subgraphs) != len(m.subgraphs):
        bad["subgraphs"] += 1
    for si, (go, gm) in enumerate(zip(o.subgraphs, m.subgraphs)):
        ty = [t.type for t in go.tensors]
        repl, ops = {}, []
        resolve[si] = repl
        for op in go.operators:
            ins, outs = [int(x) for x in op.inputs], [int(x) for x in op.outputs]
            if (op.opcodeIndex in cast and len(ins) == 1 and len(outs) == 1 and BF16 in (ty[ins[0]], ty[outs[0]])
                    and f32(ty[ins[0]]) == F32 and f32(ty[outs[0]]) == F32):
                repl[outs[0]] = ins[0]
            else:
                ops.append(op)

        def no(x):
            while x in repl:
                x = repl[x]
            return go.tensors[x].name if x >= 0 else None
        nm = lambda x: gm.tensors[x].name if 0 <= x < len(gm.tensors) else (None if x == -1 else "<out of range>")
        if len(ops) != len(gm.operators):
            bad["op count"] += 1
            continue
        for a, b in zip(ops, gm.operators):
            bad["op code"] += a.opcodeIndex != b.opcodeIndex
            for k in ("inputs", "outputs", "intermediates"):
                va, vb = getattr(a, k), getattr(b, k)
                bad[f"op {k}"] += [no(int(x)) for x in (va if va is not None else [])] != [nm(int(x)) for x in (vb if vb is not None else [])]
        for k in ("inputs", "outputs"):
            bad[f"subgraph {k}"] += [no(int(x)) for x in getattr(go, k)] != [nm(int(x)) for x in getattr(gm, k)]
        kept = [t for x, t in enumerate(go.tensors) if x not in repl]
        if len(kept) != len(gm.tensors):
            bad["tensor count"] += 1
            continue
        for a, b in zip(kept, gm.tensors):
            bad["tensor name / type / shape"] += (a.name, f32(a.type), list(a.shape if a.shape is not None else [])) != \
                (b.name, b.type, list(b.shape if b.shape is not None else []))
            bad["quantization references"] += [go.tensors[x].name if x >= 0 else None for x in qref(a)] != [nm(x) for x in qref(b)]
    for sa, sb in zip(o.signatureDefs or [], m.signatureDefs or []):
        ga, gb, repl = o.subgraphs[sa.subgraphIndex], m.subgraphs[sb.subgraphIndex], resolve.get(sa.subgraphIndex, {})

        def walk(x):
            while x in repl:
                x = repl[x]
            return x
        for k in ("inputs", "outputs"):
            ta, tb = getattr(sa, k), getattr(sb, k)
            bad[f"signature {k}"] += [(x.name, ga.tensors[walk(x.tensorIndex)].name) for x in ta] != \
                [(x.name, gb.tensors[x.tensorIndex].name if x.tensorIndex < len(gb.tensors) else None) for x in tb]
    return {k: v for k, v in bad.items() if v}


def check(path, orig=None):
    """--check <file> [<the input it was written from>]: per TFLite section, the BFLOAT16 tensors, the table fields
    this schema does not know, the tensor references out of range, and (with the input) the references that name
    another tensor than the input's (read-only). Exit 1 when any of these is found."""
    keep, out = tflite_sections(path)
    keep0, src = tflite_sections(orig) if orig else (None, {})
    ok = True
    for i, fb in out.items():
        n = sum(fb.Subgraphs(s).Tensors(t).Type() == BF16 for s in range(fb.SubgraphsLength())
                for t in range(fb.Subgraphs(s).TensorsLength()))
        unk = unknown_fields(fb)
        m = S.ModelT.InitFromObj(fb)
        oor = out_of_range(m)
        line = f"section {i}: BFLOAT16 tensors {n}, fields this schema does not know {unk or 'none'}, tensor references out of range {oor}"
        if i in src:
            ren = renamed(S.ModelT.InitFromObj(src[i]), m)
            line += f", references naming another tensor than the input's {ren or 'none'}"
            ok = ok and not ren
        print(line)
        ok = ok and not n and not unk and not oor
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(check(*sys.argv[2:4]) if sys.argv[1:2] == ["--check"] else main(*sys.argv[1:]))
