#!/usr/bin/env python3
"""tfl_swapcheck.py A.litertlm B.litertlm SECTION OUT.json, after tfl_diff.py: is the difference between two
TFLiteModel sections only an ORDER difference? (1) the contents of every buffer of A vs every buffer of B for the buffers that
differ in place (A's buffer i == B's buffer j?), (2) per subgraph the tensor table (name, type, shape, the content hash of its
buffer) and the operator list (opcode, inputs / outputs as tensor names) compared as tables, so a renumbering of buffers shows up
as "same tensors, same contents, different indices". Read-only, mmap; needs ai-edge-litert."""
import hashlib
import json
import mmap
import sys

from ai_edge_litert import schema_py_generated as schema

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from tfl_diff import section  # noqa: E402


def load(path, idx):
    _, b, e = section(path, idx)
    f = open(path, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    view = memoryview(mm)[b:e]
    return schema.Model.GetRootAsModel(view, 0), view


def buf_bytes(fb, view, bi):
    b = fb.Buffers(bi)
    if b.DataLength():
        off, size = b._tab.Vector(b._tab.Offset(4)), b.DataLength()
    elif b.Size():
        off, size = b.Offset(), b.Size()
    else:
        return b""
    return bytes(view[off:off + size]) if size <= (1 << 20) else hashlib.sha256(view[off:off + size]).hexdigest().encode()


def tables(fb, view):
    hcache = {}

    def h(bi):
        if bi not in hcache:
            hcache[bi] = hashlib.sha256(buf_bytes(fb, view, bi)).hexdigest()[:16]
        return hcache[bi]

    codes = [fb.OperatorCodes(i) for i in range(fb.OperatorCodesLength())]
    out = []
    for si in range(fb.SubgraphsLength()):
        sg = fb.Subgraphs(si)
        names = [bytes(sg.Tensors(i).Name() or b"").decode(errors="replace") for i in range(sg.TensorsLength())]
        tens = []
        for i in range(sg.TensorsLength()):
            t = sg.Tensors(i)
            tens.append((names[i], t.Type(), tuple(t.ShapeAsNumpy().tolist()) if t.ShapeLength() else (), h(t.Buffer())))
        ops = []
        for oi in range(sg.OperatorsLength()):
            op = sg.Operators(oi)
            c = codes[op.OpcodeIndex()]
            code = max(c.BuiltinCode(), c.DeprecatedBuiltinCode())
            ins = tuple(names[x] if x >= 0 else None for x in (op.InputsAsNumpy().tolist() if op.InputsLength() else []))
            outs = tuple(names[x] for x in (op.OutputsAsNumpy().tolist() if op.OutputsLength() else []))
            ops.append((code, ins, outs))
        out.append({"name": bytes(sg.Name() or b"").decode(errors="replace"), "tensors": tens, "ops": ops})
    return out


def main(a, b, idx, out_path):
    fa, va = load(a, idx)
    fb_, vb = load(b, idx)
    res = {"a": a, "b": b, "section": idx, "buffers": [fa.BuffersLength(), fb_.BuffersLength()]}
    # (1) the in-place differing small buffers: cross-match by content
    def size_of(fb, bi):
        x = fb.Buffers(bi)
        return x.DataLength() or x.Size()
    small = [i for i in range(min(fa.BuffersLength(), fb_.BuffersLength())) if size_of(fa, i) <= 4096 and size_of(fb_, i) <= 4096]
    diff = [i for i in small if buf_bytes(fa, va, i) != buf_bytes(fb_, vb, i)]
    cross = {}
    for i in diff:
        ai = buf_bytes(fa, va, i)
        cross[i] = [j for j in diff if buf_bytes(fb_, vb, j) == ai]
    res["inplace_differing_small_buffers"] = diff
    res["a_buffer_equals_b_buffer"] = {str(k): v for k, v in cross.items()}
    # (2) tables
    ta, tb = tables(fa, va), tables(fb_, vb)
    rows = []
    for sa, sb in zip(ta, tb):
        ra = {"name": sa["name"], "tensors_same_list": sa["tensors"] == sb["tensors"],
              "tensors_same_set": sorted(sa["tensors"]) == sorted(sb["tensors"]),
              "ops_same_list": sa["ops"] == sb["ops"], "n_tensors": [len(sa["tensors"]), len(sb["tensors"])],
              "n_ops": [len(sa["ops"]), len(sb["ops"])]}
        if not ra["tensors_same_set"]:
            only_a = sorted(set(sa["tensors"]) - set(sb["tensors"]))[:5]
            only_b = sorted(set(sb["tensors"]) - set(sa["tensors"]))[:5]
            ra["only_a"] = [list(map(str, x)) for x in only_a]
            ra["only_b"] = [list(map(str, x)) for x in only_b]
        if not ra["ops_same_list"]:
            k = next(i for i, (x, y) in enumerate(zip(sa["ops"], sb["ops"])) if x != y) if len(sa["ops"]) == len(sb["ops"]) else None
            ra["first_op_diff"] = None if k is None else {"index": k, "a": str(sa["ops"][k])[:300], "b": str(sb["ops"][k])[:300]}
        rows.append(ra)
    res["subgraphs"] = rows
    json.dump(res, open(out_path, "w"), indent=1)
    print(json.dumps({"buffers": res["buffers"], "inplace_differing_small_buffers": diff,
                      "a_buffer_equals_b_buffer": res["a_buffer_equals_b_buffer"]}))
    for r in rows:
        print(f"{r['name']}: tensors same list {r['tensors_same_list']} same set {r['tensors_same_set']}, ops same list "
              f"{r['ops_same_list']} {r['n_tensors']} {r['n_ops']}" + (f" first op diff {r['first_op_diff']}" if r.get("first_op_diff") else "")
              + (f" only_a {r['only_a'][:2]} only_b {r['only_b'][:2]}" if r.get("only_a") else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]))
