#!/usr/bin/env python3
"""The decoder in int8 (venv .venv-q: ai-edge-quantizer 0.10.0, ai-edge-litert 2.3.0). Copied from
parakeet_redux_work/scripts/r4_dec8.py: out/r1 and results/r1 paths, U_ii instead of B_ii, the int8 file only (the
joint-head-only variant of redux is not made).

make   out/r1/dec.tflite (predict / joint, fp32) -> out/r1/dec_int8.tflite = ai-edge-quantizer dynamic_wi8_afp32 (weights
       int8 per channel, activations fp32 = XNNPACK's dynamically quantized int8 inputs, "qd8") on every op the recipe
       takes. Bytes, sha256, census, the ops whose weight became int8 with the max |w_int8 x scale - w| / max |w| per
       op. -> results/r1/dec_int8.json
run    S32's 83 windows (out/s32_ii order): mel30 + enc30_fp32 on the CPU (4 threads) once, then the TDT loop
       (redux CONTRACT rule 4) per decoder file (fp32, int8) x decoder threads (4, 1), each in a fresh interpreter: 1
       warm-up window, then every window timed (the whole loop: predict / joint calls, their writes and reads, the
       argmax). Tokens against U_ii (out/s32_ii npz tokens) and (token, frame, duration) against U_ii's emissions (npz
       frames / durs). Timing: run it alone inside a measurement window. -> results/r1/dec8_ms.json
CLI: u1_dec8.py make | run
"""
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import u1_drive as DR  # noqa: E402

SRC = "out/r1/dec.tflite"
OUT = {"int8": "out/r1/dec_int8.tflite"}
JOINT_HEAD = "joint_logits_output;"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def int8_ops(path_fp32, path_q):
    """ops whose weight is INT8 in path_q, with the max relative error of the dequantized weight vs the fp32 file"""
    import int4_check as C
    from ai_edge_litert import schema_py_generated as S
    from ai_edge_litert.tools import mmap_utils
    out = []
    dq = mmap_utils.get_file_contents(path_q)
    df = mmap_utils.get_file_contents(path_fp32)
    mq, mf = S.Model.GetRootAs(dq, 0), S.Model.GetRootAs(df, 0)
    for si in range(mq.SubgraphsLength()):
        sq, sf = mq.Subgraphs(si), mf.Subgraphs(si)
        fp32_w = {}
        for i in range(sf.OperatorsLength()):
            op = sf.Operators(i)
            oc = mf.OperatorCodes(op.OpcodeIndex())
            name = C.BO.get(max(oc.BuiltinCode(), oc.DeprecatedBuiltinCode()))
            if name in ("FULLY_CONNECTED", "EMBEDDING_LOOKUP"):
                wi = op.Inputs(1) if name == "FULLY_CONNECTED" else op.Inputs(1)
                t = sf.Tensors(wi)
                key = ";".join(sf.Tensors(op.Outputs(j)).Name().decode() for j in range(op.OutputsLength()))
                fp32_w[key] = np.frombuffer(C.buf_bytes(mf, df, t.Buffer()), np.float32).reshape(t.ShapeAsNumpy())
        for i in range(sq.OperatorsLength()):
            op = sq.Operators(i)
            oc = mq.OperatorCodes(op.OpcodeIndex())
            name = C.BO.get(max(oc.BuiltinCode(), oc.DeprecatedBuiltinCode()))
            if name not in ("FULLY_CONNECTED", "EMBEDDING_LOOKUP"):
                continue
            key = ";".join(sq.Tensors(op.Outputs(j)).Name().decode() for j in range(op.OutputsLength()))
            t = sq.Tensors(op.Inputs(1))
            dt = C.TT.get(t.Type())
            row = {"subgraph": sq.Name().decode(), "op": name, "scope": key[-80:], "weight_dtype": dt,
                   "shape": t.ShapeAsNumpy().tolist()}
            if dt == "INT8":
                q = t.Quantization()
                sc = q.ScaleAsNumpy().astype(np.float64)
                zp = q.ZeroPointAsNumpy()
                ax = q.QuantizedDimension()
                w8 = np.frombuffer(C.buf_bytes(mq, dq, t.Buffer()), np.int8).reshape(t.ShapeAsNumpy()).astype(np.float64)
                shp = [1] * w8.ndim
                shp[ax] = -1
                wd = (w8 - zp.reshape(shp)) * sc.reshape(shp)
                w = fp32_w.get(key)
                row.update({"channels": int(sc.size), "axis": int(ax), "zero_points_all_0": bool(np.all(zp == 0))})
                if w is not None and w.shape == wd.shape:
                    row["max_rel_err"] = float(np.abs(wd - w).max() / np.abs(w).max())
            out.append(row)
    return out


def make():
    from ai_edge_quantizer import qtyping, quantizer, recipe
    import r2_fb
    rep = {"src": SRC, "src_sha256": sha256(SRC), "src_bytes": os.path.getsize(SRC), "files": {}}
    for tag, dst in OUT.items():
        t0 = time.time()
        qt = quantizer.Quantizer(SRC)
        if tag == "int8":
            qt.load_quantization_recipe(recipe.dynamic_wi8_afp32())
        else:
            qt.update_quantization_recipe(
                regex="^" + JOINT_HEAD.replace(";", ";") + "$", operation_name=qtyping.TFLOperationName.FULLY_CONNECTED,
                op_config=qtyping.OpQuantizationConfig(
                    weight_tensor_config=qtyping.TensorQuantizationConfig(
                        num_bits=8, symmetric=True, granularity=qtyping.QuantGranularity.CHANNELWISE,
                        dtype=qtyping.TensorDataType.INT),
                    compute_precision=qtyping.ComputePrecision.INTEGER, explicit_dequantize=False),
                algorithm_key=quantizer.AlgorithmName.MIN_MAX_UNIFORM_QUANT)
        tmp = dst[:-7] + ".tmp.tflite"
        qt.quantize().export_model(tmp, overwrite=True)
        os.replace(tmp, dst)
        ops = int8_ops(SRC, dst)
        rep["files"][tag] = {"path": dst, "bytes": os.path.getsize(dst), "sha256": sha256(dst),
                             "seconds": round(time.time() - t0, 1), "census": r2_fb.census(dst),
                             "int8_ops": [o for o in ops if o["weight_dtype"] == "INT8"],
                             "other_weight_ops": [o for o in ops if o["weight_dtype"] != "INT8"]}
        print(tag, json.dumps({"bytes": rep["files"][tag]["bytes"], "int8_ops": len(rep["files"][tag]["int8_ops"]),
                               "worst": max((o.get("max_rel_err", 0) for o in rep["files"][tag]["int8_ops"]),
                                            default=None)}), flush=True)
    json.dump(rep, open("results/r1/dec_int8.json", "w"), indent=1, default=str)
    return 0 if all(rep["files"][t]["int8_ops"] for t in OUT) else 3


def loop(pred, joint, enc, valid_t):
    """CONTRACT v2 rule 4 (r4_drive.Pipeline.loop), returning the (token, frame, duration) emissions"""
    h = np.zeros((2, 1, DR.D), np.float32)
    c = np.zeros((2, 1, DR.D), np.float32)
    o = pred(token=np.array([[DR.BLANK]], np.int32), h=h, c=c)
    dec_out, h, c = o["dec_out"], o["h_out"], o["c_out"]
    t = n = on = 0
    em = []
    while t < valid_t and n < DR.MAX_STEPS:
        n += 1
        lg = joint(enc_frame=enc[t:t + 1], dec_out=dec_out)["logits"][0]
        tok = int(np.argmax(lg[:DR.VOCAB]))
        d = int(np.argmax(lg[DR.VOCAB:]))
        if tok == DR.BLANK and d == 0:
            d = 1
        if tok != DR.BLANK:
            em.append((tok, t, d))
            if tok == DR.EOS:
                break
            o = pred(token=np.array([[tok]], np.int32), h=h, c=c)
            dec_out, h, c = o["dec_out"], o["h_out"], o["c_out"]
        if d == 0:
            if tok != DR.BLANK:
                on += 1
            if on >= DR.CAP:
                d = 1
        if d > 0:
            on = 0
        t += d
    return em, n


def run():
    man = DR.read_jsonl(DR.SETS["s32"][0])
    paths = {r["id"]: r["path"] for r in man}
    order = {r["id"]: k for k, r in enumerate(man)}
    files = sorted(glob.glob("out/s32_ii/*.npz"),
                   key=lambda f: (order.get(os.path.basename(f)[:-4].split("__w")[0], 1e9), f))
    mel = DR.CpuGraph(DR.mel_path(30), 4)
    enc_g = DR.CpuGraph(DR.enc_path(30, "fp32"), 4)
    wins = []
    for f in files:
        stem = os.path.basename(f)[:-4]
        uid, k = (stem.split("__w") + ["0"])[:2]
        z = np.load(f)
        audio = DR.load_wav(paths[uid])
        s0, n = DR.windows(len(audio))[int(k)]
        x = np.zeros((1, DR.n_audio(30)), np.float32)
        x[0, :n] = audio[s0:s0 + n]
        mo = mel("serving_default", audio=x, n_valid=np.array([n], np.int32))
        enc = np.array(enc_g("serving_default", features=mo["features"], mel_mask=mo["mel_mask"])["enc"][0], copy=True)
        wins.append({"stem": stem, "enc": enc, "valid_t": DR.enc_valid_frames(n), "bii_tokens": z["tokens"].tolist(),
                     "bii_em": list(zip(z["tokens"].tolist(), z["frames"].tolist(), z["durs"].tolist()))})
    del enc_g, mel
    rep = {"windows": len(wins), "time_start": time.strftime("%H:%M:%S"), "loadavg_before": os.getloadavg(), "configs": {}}
    decs = {"fp32": SRC, **{k: v for k, v in OUT.items() if os.path.exists(v)}}
    for tag, path in decs.items():
        for th in (4, 1):
            g = DR.CpuGraph(path, th)
            pred, joint = g.runners["predict"], g.runners["joint"]
            loop(pred, joint, wins[0]["enc"], wins[0]["valid_t"])
            ms, tok_eq, em_eq, steps, ntok = [], 0, 0, 0, 0
            for w in wins:
                t0 = time.perf_counter()
                em, n = loop(pred, joint, w["enc"], w["valid_t"])
                ms.append(1e3 * (time.perf_counter() - t0))
                toks = [e[0] for e in em]
                tok_eq += toks == w["bii_tokens"]
                em_eq += [tuple(e) for e in em] == [tuple(e) for e in w["bii_em"]]
                steps += n
                ntok += len(toks)
            rep["configs"][f"{tag}_t{th}"] = {
                "dec": path, "threads": th, "load_s": g.info["load_s"], "window_ms_median": round(float(np.median(ms)), 3),
                "window_ms_p90": round(float(np.quantile(ms, 0.9)), 3), "total_ms": round(float(sum(ms)), 1),
                "steps": steps, "tokens": ntok, "ms_per_step": round(float(sum(ms)) / steps, 4),
                "windows_tokens_eq_bii": tok_eq, "windows_emissions_eq_bii": em_eq, "window_ms": [round(m, 3) for m in ms]}
            print(tag, th, json.dumps({k: v for k, v in rep["configs"][f"{tag}_t{th}"].items() if k != "window_ms"}),
                  flush=True)
            del g
    rep["loadavg_after"] = os.getloadavg()
    rep["time_end"] = time.strftime("%H:%M:%S")
    rep["env"] = DR.env_record()
    json.dump(rep, open("results/r1/dec8_ms.json", "w"), indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(make() if sys.argv[1:] == ["make"] else run() if sys.argv[1:] == ["run"] else 2)
