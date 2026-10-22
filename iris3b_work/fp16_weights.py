"""Stores every FULLY_CONNECTED weight of the 20 float32 graphs as float16 behind a DEQUANTIZE, under the published names.

    python fp16_weights.py --out OUT

Activations, biases, norm weights and every other constant stay float32, so the graphs still compute in float32 (the
GPU runs them at FP32 precision). The recipe is litert-torch's full_fp16_recipe() translated for ai-edge-quantizer,
applied to each float32 file's bytes with translate_recipe.quantize_model: the call that
litert_torch.convert(..., quant_config=full_fp16_recipe()) makes on its exported bytes.

  OUT/fp32/te/iris_te_c<NN>_f32.tflite  -> OUT/model/iris_te_c<NN>_f16w.tflite
  OUT/fp32/dit/dit_<name>.tflite        -> OUT/model/iris_dit_<name>_f16w_dit.tflite

Checks of every written file (a failure exits 4): the signature (key, input and output names, shapes, dtypes) equals
the float32 file's; the FULLY_CONNECTED count equals the float32 file's and each FC reads a FLOAT16 constant through a
DEQUANTIZE; every other op count is unchanged; the float16 data is numpy's cast of the float32 weight; no weight is
inf. OUT/model_fp16w.json records per file: bytes, sha256, the FC weight elements the cast changed, the largest change,
the non-zero weights that became 0 and the weights that were not bfloat16 values (low 16 bits non-zero).
A file that is already there and recorded with the same size is skipped.
"""

import argparse
import hashlib
import json
import mmap
import os
import sys
import time
from pathlib import Path

import numpy as np

DIT_NAMES = ["adapter", "dual0", "dual1", "dual2", "single0", "single1", "single2", "pixel"]


def items(root):
    """[(published name, float32 source)] in the order they are written (the small TE c00 first)."""
    out = [(f"iris_te_c{i:02d}_f16w.tflite", root / "fp32" / "te" / f"iris_te_c{i:02d}_f32.tflite") for i in range(12)]
    out += [(f"iris_dit_{n}_f16w_dit.tflite", root / "fp32" / "dit" / f"dit_{n}.tflite") for n in DIT_NAMES]
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def fp16_recipe():
    """litert-torch's full_fp16_recipe() translated for ai-edge-quantizer, as convert(quant_config=) does it."""
    from litert_torch.generative.quantize import quant_recipes as qrs
    from litert_torch.quantize import translate_recipe as tr

    return tr.translate_to_ai_edge_recipe(qrs.full_fp16_recipe().generative_recipe)


class Flat:
    """A .tflite read through a memory map: op codes, operators, tensors and constant data of subgraph 0."""

    def __init__(self, path):
        from ai_edge_litert import schema_py_generated as S

        self.fh = open(path, "rb")
        self.buf = mmap.mmap(self.fh.fileno(), 0, access=mmap.ACCESS_READ)
        self.m = S.Model.GetRootAs(self.buf, 0)
        names = {v: k for k, v in S.BuiltinOperator.__dict__.items() if not k.startswith("_")}
        self.ttype = {v: k for k, v in S.TensorType.__dict__.items() if not k.startswith("_")}
        self.codes = [names.get(max(self.m.OperatorCodes(i).BuiltinCode(), self.m.OperatorCodes(i).DeprecatedBuiltinCode()))
                      for i in range(self.m.OperatorCodesLength())]
        self.g = self.m.Subgraphs(0)

    def ops(self):
        """[(op index, op name, [input tensor ids], [output tensor ids])]."""
        out = []
        for oi in range(self.g.OperatorsLength()):
            op = self.g.Operators(oi)
            out.append((oi, self.codes[op.OpcodeIndex()],
                        [int(op.Inputs(j)) for j in range(op.InputsLength())],
                        [int(op.Outputs(j)) for j in range(op.OutputsLength())]))
        return out

    def op_counts(self):
        c = {}
        for _, name, _, _ in self.ops():
            c[name] = c.get(name, 0) + 1
        return dict(sorted(c.items()))

    def ttype_of(self, ti):
        return self.ttype.get(self.g.Tensors(ti).Type())

    def data(self, ti):
        """The constant data of tensor ti as a numpy array of its type and shape (a view of the map), else None."""
        t = self.g.Tensors(ti)
        b = self.m.Buffers(t.Buffer())
        if b is None:
            return None
        size = b.Size() if hasattr(b, "Size") else 0
        if size:                                            # data stored after the flatbuffer (offset layout)
            raw = memoryview(self.buf)[b.Offset():b.Offset() + size]
        elif b.DataLength() > 0:
            raw = b.DataAsNumpy()
        else:
            return None
        dt = {"FLOAT32": np.float32, "FLOAT16": np.float16, "INT32": np.int32, "INT8": np.int8}[self.ttype[t.Type()]]
        shape = [int(t.Shape(j)) for j in range(t.ShapeLength())]
        return np.frombuffer(raw, dtype=dt).reshape(shape)

    def close(self):
        self.m = self.g = None
        try:
            self.buf.close()
        except BufferError:                                 # a live numpy view still holds the map; the GC frees it
            pass
        self.fh.close()


def fc_weights(flat):
    """[(FC op index, producer op name or None, the constant tensor id)] in op order: in a float16-weight graph the FC
    reads a DEQUANTIZE output whose input is the constant; in a float32 graph it reads the constant directly."""
    ops = flat.ops()
    producer = {}
    for oi, name, ins, outs in ops:
        for t in outs:
            producer[t] = (name, ins)
    out = []
    for oi, name, ins, _ in ops:
        if name != "FULLY_CONNECTED":
            continue
        w = ins[1]
        if w in producer:
            pname, pins = producer[w]
            out.append((oi, pname, pins[0] if pins else None))
        else:
            out.append((oi, None, w))
    return out


def signature(path):
    """Signature 0: key, inputs and outputs as (name, shape, dtype)."""
    from ai_edge_litert import schema_py_generated as S

    types = {v: k for k, v in S.TensorType.__dict__.items() if not k.startswith("_")}
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    try:
        m = S.Model.GetRootAs(mm, 0)
        sd = m.SignatureDefs(0)
        sg = m.Subgraphs(sd.SubgraphIndex())

        def tensor(t):
            x = sg.Tensors(int(t.TensorIndex()))
            return [t.Name().decode(), [int(x.Shape(j)) for j in range(x.ShapeLength())],
                    types.get(x.Type(), str(x.Type())).lower()]

        return {"signature": sd.SignatureKey().decode(),
                "inputs": [tensor(sd.Inputs(i)) for i in range(sd.InputsLength())],
                "outputs": [tensor(sd.Outputs(i)) for i in range(sd.OutputsLength())]}
    finally:
        del m
        mm.close()


def check(src, dst):
    """Every FC weight of dst (float16 constant -> DEQUANTIZE) against the same FC's float32 weight in src."""
    a, b = Flat(src), Flat(dst)
    try:
        fa, fb = fc_weights(a), fc_weights(b)
        problems = []
        if len(fa) != len(fb):
            problems.append(f"FC count {len(fa)} -> {len(fb)}")
        tot = {"fc": len(fb), "elements": 0, "changed": 0, "max_abs_delta": 0.0, "max_rel_delta_normal": 0.0,
               "to_zero": 0, "inf": 0, "not_bf16": 0, "max_abs_w": 0.0}
        for (oa, pa, ca), (ob, pb, cb) in zip(fa, fb):
            if pa is not None or a.data(ca) is None:
                problems.append(f"float32 FC op {oa}: the weight is not a constant")
                continue
            if pb != "DEQUANTIZE" or b.ttype_of(cb) != "FLOAT16":
                problems.append(f"FC op {ob}: weight made by {pb}, constant type {b.ttype_of(cb)}")
                continue
            w32 = np.asarray(a.data(ca), np.float32)
            w16 = b.data(cb)
            if w16 is None or w32.shape != w16.shape:
                problems.append(f"FC op {oa}/{ob}: shapes {w32.shape} {None if w16 is None else w16.shape}")
                continue
            if not np.array_equal(w16, w32.astype(np.float16)):
                problems.append(f"FC op {ob}: the float16 data is not numpy's cast of the float32 weight")
            back = w16.astype(np.float32)
            d = back != w32
            n_d = int(np.count_nonzero(d))
            tot["elements"] += int(w32.size)
            tot["changed"] += n_d
            if n_d:
                dd = np.abs(back[d] - w32[d])
                tot["max_abs_delta"] = max(tot["max_abs_delta"], float(dd.max()))
                normal = np.abs(w32[d]) >= np.float32(6.103515625e-05)        # float16's smallest normal
                if normal.any():
                    tot["max_rel_delta_normal"] = max(tot["max_rel_delta_normal"],
                                                      float((dd[normal] / np.abs(w32[d][normal])).max()))
            tot["to_zero"] += int(np.count_nonzero((back == 0) & (w32 != 0)))
            tot["inf"] += int(np.count_nonzero(np.isinf(back)))
            tot["not_bf16"] += int(np.count_nonzero(w32.view(np.uint32) & np.uint32(0xFFFF)))
            tot["max_abs_w"] = max(tot["max_abs_w"], float(np.abs(w32).max()))
        ca_, cb_ = a.op_counts(), b.op_counts()
        dq = cb_.get("DEQUANTIZE", 0) - ca_.get("DEQUANTIZE", 0)
        if dq != tot["fc"]:
            problems.append(f"DEQUANTIZE added {dq}, FC {tot['fc']}")
        other = {k: (ca_.get(k, 0), cb_.get(k, 0)) for k in set(ca_) | set(cb_)
                 if k != "DEQUANTIZE" and ca_.get(k, 0) != cb_.get(k, 0)}
        if other:
            problems.append(f"op counts changed {other}")
        if tot["inf"]:
            problems.append(f"{tot['inf']} inf in the float16 weights")
        sa, sb = signature(src), signature(dst)
        if sa != sb:
            problems.append(f"signature changed: {sa} -> {sb}")
        return tot, cb_, problems
    finally:
        a.close()
        b.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out", help="output root (default: out)")
    args = ap.parse_args()
    root = Path(args.out)
    from importlib.metadata import version

    from litert_torch.quantize.translate_recipe import quantize_model

    recipe = fp16_recipe()
    model_dir = root / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    log_path = root / "model_fp16w.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {}
    log["recipe"] = json.loads(json.dumps(recipe, default=str))
    log["versions"] = {p: version(p) for p in ("ai-edge-quantizer", "litert-torch", "ai-edge-litert", "numpy")}
    files = log.setdefault("files", {})
    print("RECIPE", json.dumps(log["recipe"]), flush=True)
    t0 = time.time()
    bad = []
    for name, src in items(root):
        dst = model_dir / name
        ent = files.get(name)
        if dst.exists() and ent and ent.get("bytes") == dst.stat().st_size and not ent.get("problems"):
            print("SKIP", name, flush=True)
            continue
        tq = time.time()
        out = bytes(quantize_model(src.read_bytes(), recipe))
        tmp = model_dir / (name + ".part")
        tmp.write_bytes(out)
        os.replace(tmp, dst)
        del out
        q_s = time.time() - tq
        stats, ops, problems = check(src, dst)
        ent = {"src": str(src.relative_to(root)), "src_bytes": src.stat().st_size, "src_sha256": sha256(src),
               "bytes": dst.stat().st_size, "sha256": sha256(dst), "seconds": round(q_s, 1), "weights": stats,
               "ops": ops, "problems": problems}
        files[name] = ent
        log_path.write_text(json.dumps(log, indent=1))
        print("WROTE", json.dumps({"file": name, "bytes": ent["bytes"], "sha256": ent["sha256"],
                                   "changed": stats["changed"], "elements": stats["elements"],
                                   "max_abs_delta": stats["max_abs_delta"], "not_bf16": stats["not_bf16"],
                                   "problems": problems}), flush=True)
        if problems:
            bad.append(name)
    print("FP16W_DONE", json.dumps({"files": len(files), "bad": bad, "s": round(time.time() - t0, 1)}), flush=True)
    if bad:
        sys.exit(4)


if __name__ == "__main__":
    main()
