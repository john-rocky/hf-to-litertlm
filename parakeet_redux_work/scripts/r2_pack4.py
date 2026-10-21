#!/usr/bin/env python3
"""enc30_int4x (venv .venv-q: ai-edge-quantizer 0.10.0 + ai-edge-litert 2.3.0, no model execution).

1. Every FULLY_CONNECTED weight of the fp32 encoder file is matched to a redux module by the sha1 of its fp32 bytes
   against the expanded ternary weights (unpack3) and the dense redux tensors: a weight the converter changed (a scale
   folded in) matches nothing and stops the script (exit 2).
2. Only the FCs that hold a ternary weight get int4: dequantized_weight_recovery, BLOCKWISE_128, symmetric, INTEGER
   compute (the CPU path quantizes the activation at run time); the recipe regex is the exact alternation of those
   ops' scopes (aeq: output tensor names joined by ';' plus ';'). Every other weight stays FLOAT32.
3. Each int4 scale tensor is overwritten with redux's own fp16 scales (aeq rounds blockwise scales through bf16;
   redux's scales are all normal and non-zero, so the 17,225 all-zero groups get a non-zero scale with a zero payload).
4. Check per module: payload == code - 1, scale == redux fp16 scale, payload x scale == the expanded weight
   (max|d| and the count of differing elements), zero / subnormal (< 2^-14) scales. relative_k_proj (24 modules) has a
   constant input (the position table), so the converter folds it into a constant: each folded table is compared with
   the float64 product pos @ W^T in the layouts the attention can use.
CLI: r2_pack4.py <fp32 src> <dst> --json results/r2_int4x.json
"""
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import int4_check as C  # noqa: E402
import r2_fb  # noqa: E402
import unpack3  # noqa: E402
from ai_edge_litert import schema_py_generated as S  # noqa: E402
from ai_edge_litert.tools import mmap_utils  # noqa: E402
from ai_edge_quantizer import qtyping, quantizer  # noqa: E402
from ai_edge_quantizer.utils import tfl_flatbuffer_utils as TFU  # noqa: E402

SUB = np.float32(2.0 ** -14)


def sha1(a):
    return hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()


def fc_ops(path):
    model = TFU.read_model(Path(path))
    sg = model.subgraphs[0]
    out = []
    for i, op in enumerate(sg.operators):
        if TFU.get_op_name_by_index(model, 0, i) != "FULLY_CONNECTED":
            continue
        wi = op.inputs[1]
        t = sg.tensors[wi]
        d = model.buffers[t.buffer].data
        out.append({"op": i, "scope": TFU.get_op_scope(op, sg.tensors), "wi": wi, "shape": list(t.shape),
                    "dtype": int(t.type), "data": d})
    return model, out


def rel_pos(t, d):
    inv = 1.0 / (10000.0 ** (np.arange(0, d, 2, dtype=np.float64) / d))
    p = np.arange(t - 1, -t, -1, dtype=np.float64)
    f = p[:, None] * inv[None]
    return np.stack([np.sin(f), np.cos(f)], -1).reshape(2 * t - 1, d)


def main(argv):
    src, dst = argv[1], argv[2]
    out_json = argv[argv.index("--json") + 1]
    t0 = time.time()
    meta = json.load(open("hf/redux/ternary.json"))
    gs = meta["quant"]["group_size"]
    tens = unpack3._read_safetensors("hf/redux/model.safetensors")
    mods = {m["name"]: m for m in meta["quantized_modules"]}
    by_hash = {}
    for n, m in mods.items():
        _, w = unpack3.unpack_module(tens[n + ".qweight"], tens[n + ".scales"], m["in_features"], gs)
        by_hash[sha1(w.astype(np.float32))] = n
    dense = {k: v.astype(np.float32) for k, v in tens.items()
             if v.dtype == np.float16 and v.ndim == 2 and not k.startswith("vad_head")}
    for k, v in dense.items():
        by_hash.setdefault(sha1(v), "dense:" + k)
    _, fcs = fc_ops(src)
    rep = {"src": src, "dst": dst, "fc_ops": len(fcs), "matched_ternary": 0, "matched_dense": [], "unmatched": []}
    tern = {}
    for f in fcs:
        if f["dtype"] != S.TensorType.FLOAT32:
            rep["unmatched"].append({"scope": f["scope"][-160:], "dtype": f["dtype"], "shape": f["shape"]})
            continue
        h = sha1(np.frombuffer(f["data"], np.float32))
        n = by_hash.get(h)
        if n is None:
            rep["unmatched"].append({"scope": f["scope"][-160:], "shape": f["shape"]})
        elif n.startswith("dense:"):
            rep["matched_dense"].append({"tensor": n[6:], "shape": f["shape"]})
        else:
            tern[f["scope"]] = n
    # FCs the converter scaled (0.5 of the FFN residual into linear2, the attention scale into k_proj): the output
    # names carry fused locations (a k FC's first name can be q_proj), so the module is found by value: candidates =
    # the not-yet-matched ternary modules of the same shape in the blocks the scope names; the least-squares factor
    # w_graph = f x w_ternary with the smallest residual wins (residual <= 1e-6 required)
    factor = {n: 1.0 for n in tern.values()}
    still = []
    for f in [x for x in fcs if x["dtype"] == S.TensorType.FLOAT32 and x["scope"] not in tern
              and by_hash.get(sha1(np.frombuffer(x["data"], np.float32)), "").startswith("dense:") is False]:
        blocks = sorted({int(b) for b in re.findall(r"Block_(\d+)", f["scope"])})
        cands = [n for n, m in mods.items() if n not in tern.values() and [m["out_features"], m["in_features"]] == f["shape"]
                 and int(n.split(".")[2]) in blocks]
        best = None
        wg = np.frombuffer(f["data"], np.float32).reshape(f["shape"]).astype(np.float64)
        for n in cands:
            mm = mods[n]
            _, wt = unpack3.unpack_module(tens[n + ".qweight"], tens[n + ".scales"], mm["in_features"], gs)
            wt = wt.astype(np.float64)
            fac = float((wg * wt).sum() / (wt * wt).sum())
            res = float(np.abs(wg - fac * wt).max() / np.abs(wg).max())
            if best is None or res < best[2]:
                best = (n, fac, res)
        if best is None or best[2] > 1e-6:
            still.append({"scope": f["scope"][-160:], "shape": f["shape"], "blocks": blocks,
                          "best": list(best) if best else None})
            continue
        tern[f["scope"]] = best[0]
        factor[best[0]] = best[1]
    rep["folded_factors"] = {}
    for n, fac in factor.items():
        key = "1" if fac == 1.0 else ("0.5" if fac == 0.5 else f"{fac:.9g}")
        rep["folded_factors"][key] = rep["folded_factors"].get(key, 0) + 1
    rep["unmatched"] = still
    rep["matched_ternary"] = len(tern)
    missing = sorted(set(mods) - set(tern.values()))
    rep["ternary_not_fc"] = {"count": len(missing), "kinds": sorted({re.sub(r"layers\.\d+\.", "layers.N.", x)
                                                                      for x in missing})}
    print(json.dumps({k: rep[k] for k in ("fc_ops", "matched_ternary", "matched_dense", "unmatched",
                                          "ternary_not_fc")}), flush=True)
    if rep["unmatched"] or len(tern) + len(missing) != len(mods):
        rep["stop"] = "an FC weight matches neither a ternary module nor a dense redux tensor"
        json.dump(rep, open(out_json, "w"), indent=1)
        return 2
    regex = "^(?:" + "|".join(re.escape(s) for s in sorted(tern)) + ")$"
    qt = quantizer.Quantizer(src)
    qt.update_quantization_recipe(
        regex=regex, operation_name=qtyping.TFLOperationName.FULLY_CONNECTED,
        op_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=qtyping.TensorQuantizationConfig(
                num_bits=4, symmetric=True, granularity=qtyping.QuantGranularity.BLOCKWISE_128,
                dtype=qtyping.TensorDataType.INT),
            compute_precision=qtyping.ComputePrecision.INTEGER, explicit_dequantize=False),
        algorithm_key=quantizer.AlgorithmName.DEQUANTIZED_WEIGHT_RECOVERY)
    t1 = time.time()
    res = qt.quantize()
    res.export_model(dst, overwrite=True)
    rep["aeq_s"] = round(time.time() - t1, 1)
    # scale write-back and per-module check
    data = bytearray(open(dst, "rb").read())
    m = S.Model.GetRootAs(data, 0)
    sg = m.Subgraphs(0)
    names = {i: sg.Tensors(i).Name().decode() for i in range(sg.TensorsLength())}
    rows = []
    int4_fc = float_fc = 0
    for i in range(sg.OperatorsLength()):
        op = sg.Operators(i)
        oc = m.OperatorCodes(op.OpcodeIndex())
        if C.BO.get(max(oc.BuiltinCode(), oc.DeprecatedBuiltinCode())) != "FULLY_CONNECTED":
            continue
        scope = ";".join(names[op.Outputs(j)] for j in range(op.OutputsLength())) + ";"
        wt = sg.Tensors(op.Inputs(1))
        if C.TT.get(wt.Type()) != "INT4":
            float_fc += 1
            if scope in tern:
                rows.append({"module": tern[scope], "error": f"weight left {C.TT.get(wt.Type())}"})
            continue
        int4_fc += 1
        n = tern.get(scope)
        if n is None:
            rows.append({"scope": scope[-160:], "error": "int4 FC that is not a ternary module"})
            continue
        q = wt.Quantization()
        bq = S.BlockwiseQuantization()
        tab = q.Details()
        bq.Init(tab.Bytes, tab.Pos)
        st = sg.Tensors(bq.Scales())
        mm = mods[n]
        o, k = mm["out_features"], mm["in_features"]
        codes, w = unpack3.unpack_module(tens[n + ".qweight"], tens[n + ".scales"], k, gs)
        fac = factor[n]
        sc16 = (tens[n + ".scales"].astype(np.float64) * fac).astype(np.float16)
        arr = C.buf_view_f16(m, data, st.Buffer())
        before = arr.reshape(o, -1).astype(np.float32).copy()
        arr[:] = sc16.reshape(-1)
        payload = C.unpack_int4(C.buf_bytes(m, data, wt.Buffer()), o * k).reshape(o, k)
        sf = sc16.astype(np.float32)
        w_hat = payload.astype(np.float32) * np.repeat(sf, gs, axis=1)[:, :k]
        w_f = (w.astype(np.float64) * fac).astype(np.float32)
        rows.append({"module": n, "factor": fac, "block": int(bq.BlockSize()), "payload_eq_code_minus_1":
                     bool(np.array_equal(payload, codes.astype(np.int8) - 1)),
                     "max_abs": float(np.abs(w_hat - w_f).max()), "n_diff": int((w_hat != w_f).sum()),
                     "max_rel_scale": float(np.abs(sf.astype(np.float64) - tens[n + ".scales"].astype(np.float64) * fac).max()
                                            / (np.abs(tens[n + ".scales"].astype(np.float64)).max() * abs(fac))),
                     "aeq_scales_ne_redux": int((before != sf).sum()), "zero_scales": int((sf == 0).sum()),
                     "subnormal_scales": int(((sf > 0) & (sf < SUB)).sum())})
    open(dst, "wb").write(data)
    ok = [r for r in rows if "error" not in r and r["payload_eq_code_minus_1"] and r["zero_scales"] == 0
          and r["subnormal_scales"] == 0 and r["block"] == gs]
    rep["modules_value_exact"] = sum(1 for r in ok if r["max_abs"] == 0.0 and r["n_diff"] == 0)
    rep["modules_scale_rounded"] = sorted({f"{r['factor']:.9g}" for r in ok if r["n_diff"] > 0})
    rep["worst_scale_rel_rounding"] = max((r["max_rel_scale"] for r in ok), default=None)
    rep.update({"int4_fc": int4_fc, "float_fc": float_fc, "modules_exact": len(ok), "module_rows": rows,
                "errors": [r for r in rows if "error" in r],
                "aeq_scales_ne_redux_total": int(sum(r.get("aeq_scales_ne_redux", 0) for r in rows)),
                "bytes": os.path.getsize(dst), "sha256": r2_fb_sha(dst)})
    # folded relative_k_proj tables
    pos = rel_pos(375, 1024)
    cands = []
    srcm = S.Model.GetRootAs(mmap_utils.get_file_contents(src), 0)
    ssg = srcm.Subgraphs(0)
    for i in range(ssg.TensorsLength()):
        t = ssg.Tensors(i)
        if C.TT.get(t.Type()) == "FLOAT32" and int(np.prod(t.ShapeAsNumpy())) == 749 * 1024 \
                and r2_fb._buf_len(srcm, t.Buffer()) > 0:
            cands.append((i, t.ShapeAsNumpy().tolist(), np.frombuffer(
                C.buf_bytes(srcm, mmap_utils.get_file_contents(src), t.Buffer()), np.float32).astype(np.float64)))
    folded = []
    for n in missing:
        mm = mods[n]
        _, w = unpack3.unpack_module(tens[n + ".qweight"], tens[n + ".scales"], mm["in_features"], gs)
        rk = pos @ w.astype(np.float64).T
        views = {"[749,1024]": rk, "[8,128,749]": rk.reshape(749, 8, 128).transpose(1, 2, 0),
                 "[8,749,128]": rk.reshape(749, 8, 128).transpose(1, 0, 2)}
        best = None
        for ci, shp, c in cands:
            for vk, v in views.items():
                for fk, fac in (("1", 1.0), ("1/sqrt(128)", 128 ** -0.5)):
                    ref = (v * fac).reshape(-1)
                    rel = float(np.abs(c - ref).max() / np.abs(ref).max())
                    if best is None or rel < best["max_rel"]:
                        best = {"tensor": ci, "shape": shp, "layout": vk, "factor": fk, "max_rel": rel}
            if best and best["max_rel"] < 1e-5:
                break
        folded.append({"module": n, **(best or {"error": "no 749x1024 constant"})})
    rep["folded_relative_k"] = folded
    rep["folded_worst_max_rel"] = max((x.get("max_rel", 1.0) for x in folded), default=None)
    rep["total_s"] = round(time.time() - t0, 1)
    rep["census"] = r2_fb.census(dst)
    json.dump(rep, open(out_json, "w"), indent=1)
    print(json.dumps({k: rep[k] for k in ("int4_fc", "float_fc", "modules_exact", "modules_value_exact",
                                          "modules_scale_rounded", "worst_scale_rel_rounding", "folded_factors",
                                          "aeq_scales_ne_redux_total", "bytes", "folded_worst_max_rel", "total_s")}),
          flush=True)
    good = (len(ok) == len(tern) == int4_fc and not rep["errors"] and float_fc == len(rep["matched_dense"]))
    return 0 if good else 3


def r2_fb_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
