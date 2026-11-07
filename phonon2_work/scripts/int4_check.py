#!/usr/bin/env python3
"""int4 container test, part 2 (venv .venv-q: ai-edge-quantizer 0.10.0 + ai-edge-litert 2.3.0).

For each fp32 one-op file of int4_micro.py and each recipe
  a   = recipes/gptqrec_int4_block128.json entry 1: dequantized_weight_recovery, BLOCKWISE_128, int4 sym, INTEGER
  b   = min_max_uniform_quantize, BLOCKWISE_128, int4 sym, INTEGER
  c   = min_max_uniform_quantize, BLOCKWISE_32, int4 sym, INTEGER
the int4 payload and the scale tensor are read back from the file and dequantized (q * scale[row, col // block]);
compared with the expanded fp32 weight: max|d| and the count of differing elements. Zero / subnormal scales are
counted. a' = the a file with its scale tensor overwritten by redux's own fp16 scales (payload untouched).
Each file runs once on the ai-edge-litert CPU interpreter (XNNPACK) on a fixed random input (seed 0); its output is
compared with the fp32 product x @ w.T (float64) and with the product through the file's own dequantized weight.
-> results/int4_check.json, out/int4/<name>.<recipe>.tflite
"""
import copy
import json
import os
import shutil
import sys
import traceback

import numpy as np

from ai_edge_litert import schema_py_generated as S
from ai_edge_litert.interpreter import Interpreter
from ai_edge_quantizer import quantizer

D = "out/int4"
NAMES = ["ff1_l1", "ff1_l2", "pw1", "pw1_lin"]
SUB = np.float32(2.0 ** -14)


def recipe_mm(gran):
    return [{"regex": ".*", "operation": "*", "algorithm_key": "min_max_uniform_quantize",
             "op_config": {"weight_tensor_config": {"num_bits": 4, "symmetric": True, "granularity": gran,
                                                    "dtype": "INT"},
                           "compute_precision": "INTEGER", "explicit_dequantize": False, "skip_checks": False,
                           "min_weight_elements": 0}}]


def recipes():
    rec = json.load(open("scripts/recipes/gptqrec_int4_block128.json"))
    return {"a": [rec[0]], "b": recipe_mm("BLOCKWISE_128"), "c": recipe_mm("BLOCKWISE_32")}


TT = {v: k for k, v in S.TensorType.__dict__.items() if not k.startswith("_")}
BO = {v: k for k, v in S.BuiltinOperator.__dict__.items() if not k.startswith("_")}


def unpack_int4(raw, n):
    b = np.frombuffer(raw, np.uint8)
    lo = (b & 0x0F).astype(np.int8)
    hi = (b >> 4).astype(np.int8)
    lo[lo > 7] -= 16
    hi[hi > 7] -= 16
    out = np.empty(b.size * 2, np.int8)
    out[0::2] = lo
    out[1::2] = hi
    return out[:n]


def buf_bytes(m, data, bidx):
    """bytes of buffer bidx: inline data, or out-of-band (Buffer.offset / size, the layout aeq writes)."""
    buf = m.Buffers(bidx)
    if buf.DataLength():
        return bytes(buf.DataAsNumpy().tobytes())
    off, size = buf.Offset(), buf.Size()
    return bytes(data[off:off + size]) if size else b""


def buf_view_f16(m, data, bidx):
    buf = m.Buffers(bidx)
    if buf.DataLength():
        return buf.DataAsNumpy().view(np.float16)
    return np.ndarray(buf.Size() // 2, dtype=np.float16, buffer=data, offset=buf.Offset())


def find_weight(path):
    """-> dict for the first op whose input 1 is a constant: op name, weight dtype/shape, payload, scales."""
    data = bytearray(open(path, "rb").read())
    m = S.Model.GetRootAsModel(data, 0)
    sg = m.Subgraphs(0)
    ops = {}
    found = None
    for i in range(sg.OperatorsLength()):
        op = sg.Operators(i)
        oc = m.OperatorCodes(op.OpcodeIndex())
        name = BO.get(max(oc.BuiltinCode(), oc.DeprecatedBuiltinCode()))
        ops[name] = ops.get(name, 0) + 1
        if found is None and op.InputsLength() > 1 and name in ("FULLY_CONNECTED", "CONV_2D", "BATCH_MATMUL"):
            ti = op.Inputs(1)
            t = sg.Tensors(ti)
            found = {"op": name, "tensor_index": int(ti), "dtype": TT.get(t.Type()), "shape": t.ShapeAsNumpy().tolist(),
                     "raw": buf_bytes(m, data, t.Buffer())}
            q = t.Quantization()
            found["qdetails"] = None
            if q is not None:
                found["qdetails"] = int(q.DetailsType())
                if q.DetailsType() == S.QuantizationDetails.BlockwiseQuantization:
                    bq = S.BlockwiseQuantization()
                    tab = q.Details()
                    bq.Init(tab.Bytes, tab.Pos)
                    st = sg.Tensors(bq.Scales())
                    found.update({"block": int(bq.BlockSize()), "scales_tensor": int(bq.Scales()),
                                  "scales_dtype": TT.get(st.Type()), "scales_shape": st.ShapeAsNumpy().tolist(),
                                  "zero_points": int(bq.ZeroPoints()),
                                  "scales_raw": buf_bytes(m, data, st.Buffer())})
                elif q.ScaleLength():
                    found.update({"perchannel_scales": q.ScaleAsNumpy().astype(np.float32),
                                  "qdim": int(q.QuantizedDimension())})
    found["ops"] = ops
    return found


def patch_scales(src, dst, new_f16):
    data = bytearray(open(src, "rb").read())
    m = S.Model.GetRootAsModel(data, 0)
    sg = m.Subgraphs(0)
    n = 0
    for i in range(sg.TensorsLength()):
        q = sg.Tensors(i).Quantization()
        if q is None or q.DetailsType() != S.QuantizationDetails.BlockwiseQuantization:
            continue
        bq = S.BlockwiseQuantization()
        tab = q.Details()
        bq.Init(tab.Bytes, tab.Pos)
        st = sg.Tensors(bq.Scales())
        arr = buf_view_f16(m, data, st.Buffer())
        assert arr.size == new_f16.size, (arr.size, new_f16.size)
        arr[:] = new_f16.reshape(-1)
        n += 1
    open(dst, "wb").write(data)
    return n


def cpu_run(path, x):
    try:
        it = Interpreter(model_path=path, num_threads=4)
        it.allocate_tensors()
        inp = it.get_input_details()[0]
        it.set_tensor(inp["index"], x)
        it.invoke()
        out = it.get_output_details()[0]
        return it.get_tensor(out["index"]).astype(np.float64), None
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


def main():
    ref = np.load(os.path.join(D, "ref.npz"))
    recs = recipes()
    rng = np.random.default_rng(0)
    out = {"files": {}}
    for name in NAMES:
        fp32 = os.path.join(D, name + ".tflite")
        if not os.path.exists(fp32):
            continue
        w = ref["w_" + name]
        codes = ref["codes_" + name]
        sc16 = ref["scales_" + name]
        o, i = w.shape
        conv = name == "pw1"
        x = rng.standard_normal((1, i, 16) if conv else (1, 16, i)).astype(np.float32)
        y_ref = (np.einsum("oi,bit->bot", w.astype(np.float64), x.astype(np.float64)) if conv
                 else x.astype(np.float64) @ w.astype(np.float64).T)
        ymax = float(np.abs(y_ref).max())
        f = {"shape": [o, i]}
        yf, err = cpu_run(fp32, x)
        f["fp32_file_cpu"] = {"error": err} if err else {"max_abs_vs_fp64": float(np.abs(yf - y_ref).max()),
                                                          "rel": float(np.abs(yf - y_ref).max() / ymax)}
        variants = {}
        for key, rc in recs.items():
            dst = os.path.join(D, f"{name}.{key}.tflite")
            v = {}
            try:
                qt = quantizer.Quantizer(fp32, copy.deepcopy(rc))
                res = qt.quantize()
                res.export_model(dst, overwrite=True)
            except Exception as e:  # noqa: BLE001
                v["error"] = f"{type(e).__name__}: {e}"[:600]
                v["trace"] = traceback.format_exc()[-1200:]
                variants[key] = v
                print(name, key, "ERROR", v["error"][:200], flush=True)
                continue
            variants[key] = v
            v["bytes"] = os.path.getsize(dst)
            fw = find_weight(dst)
            v.update({k: fw.get(k) for k in ("op", "dtype", "shape", "block", "scales_dtype", "scales_shape",
                                              "zero_points", "qdetails", "ops")})
            if fw.get("dtype") == "INT4" and fw.get("block"):
                bs = fw["block"]
                q = unpack_int4(fw["raw"], o * i).reshape(o, i)
                s = np.frombuffer(fw["scales_raw"], np.float16).reshape(o, i // bs)
                w_hat = q.astype(np.float32) * np.repeat(s.astype(np.float32), bs, axis=1)
                d = np.abs(w_hat - w)
                v["levels"] = sorted(int(t) for t in np.unique(q))
                v["max_abs_diff"] = float(d.max())
                v["n_diff"] = int((w_hat != w).sum())
                v["max_rel_diff_per_group_scale"] = float((d / np.repeat(sc16.astype(np.float32), 128, axis=1)).max())
                sf = s.astype(np.float32)
                v["scales"] = {"count": int(sf.size), "zero": int((sf == 0).sum()),
                               "subnormal": int(((sf > 0) & (sf < SUB)).sum()), "min": float(sf.min()),
                               "max": float(sf.max())}
                if bs == 128:
                    mult = 1.0 if key == "a" else 7.0
                    v["scales"]["n_ne_redux_scale"] = int((sf * mult != sc16.astype(np.float32)).sum())
                    v["payload_eq_code_minus_1"] = bool(np.array_equal(q, codes.astype(np.int8) - 1)) if key == "a" else None
                y, err = cpu_run(dst, x)
                if err:
                    v["cpu"] = {"error": err[:600]}
                else:
                    y_hat = (np.einsum("oi,bit->bot", w_hat.astype(np.float64), x.astype(np.float64)) if conv
                             else x.astype(np.float64) @ w_hat.astype(np.float64).T)
                    v["cpu"] = {"max_abs_vs_fp32_weights": float(np.abs(y - y_ref).max()),
                                "rel_vs_fp32_weights": float(np.abs(y - y_ref).max() / ymax),
                                "max_abs_vs_file_weights": float(np.abs(y - y_hat).max()),
                                "rel_vs_file_weights": float(np.abs(y - y_hat).max() / ymax)}
                if key == "a":
                    dstp = os.path.join(D, f"{name}.a_patched.tflite")
                    npatch = patch_scales(dst, dstp, sc16)
                    fwp = find_weight(dstp)
                    qp = unpack_int4(fwp["raw"], o * i).reshape(o, i)
                    spf = np.frombuffer(fwp["scales_raw"], np.float16).reshape(o, i // bs).astype(np.float32)
                    w_p = qp.astype(np.float32) * np.repeat(spf, bs, axis=1)
                    vp = {"patched_scale_tensors": npatch, "bytes": os.path.getsize(dstp),
                          "max_abs_diff": float(np.abs(w_p - w).max()), "n_diff": int((w_p != w).sum()),
                          "scales": {"zero": int((spf == 0).sum()), "subnormal": int(((spf > 0) & (spf < SUB)).sum())}}
                    y2, err2 = cpu_run(dstp, x)
                    if err2:
                        vp["cpu"] = {"error": err2[:600]}
                    else:
                        vp["cpu"] = {"max_abs_vs_fp32_weights": float(np.abs(y2 - y_ref).max()),
                                     "rel_vs_fp32_weights": float(np.abs(y2 - y_ref).max() / ymax),
                                     "bit_equal_to_unpatched_cpu": bool(y is not None and np.array_equal(y, y2))}
                    variants["a_patched"] = vp
            print(name, key, {k: v.get(k) for k in ("op", "dtype", "block", "max_abs_diff", "n_diff")},
                  v.get("cpu"), flush=True)
        f["variants"] = variants
        out["files"][name] = f
    import ai_edge_litert
    import ai_edge_quantizer
    out["versions"] = {"ai_edge_quantizer": getattr(ai_edge_quantizer, "__version__", "?"),
                       "ai_edge_litert": getattr(ai_edge_litert, "__version__", "?"), "numpy": np.__version__}
    json.dump(out, open("results/int4_check.json", "w"), indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
