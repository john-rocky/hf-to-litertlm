#!/usr/bin/env python3
"""Export port6.py (the audio path written for the GPU and NPU delegates) as one single-signature TFLite file for a
window of W seconds, turn its batched matrix products back into rank-4 ones, and store the weights in fp16: the audio
encoder of granite-speech-5.0-2b_ekv2048.litertlm (W = 24).

  signature `encode`: input `audio` f32 [1, 100 W, 160] (W s of 16 kHz PCM, frame = hop = 160 samples)
                      -> `features` f32 [1, 10 W, 2048] + `mask` uint8 [1, 10 W]     (W = 24: [1, 2400, 160] -> 240)

  OUT/granite_speech_audio6_<F>f_fp32_raw.tflite  litert-torch export of port6.GraniteSpeechAudioGraph6 (weights
                                                  BF16 -> fp32); deleted after the next step unless --keep-raw
  OUT/granite_speech_audio6_<F>f_fp32.tflite      tools/gpu_graph/tfl_rank4.py rewrite --rules bmm --unique-names:
                                                  the converter writes each matrix product of two batch-1 rank-4
                                                  operands as a rank-3 BATCH_MATMUL; the rule makes it a rank-4
                                                  [1, B, M, K] x [1, B, K, N] product again, with a RESHAPE on each
                                                  side (the same function on the CPU), so the GPU front end places
                                                  both operands the same way (a graph with rank-3 attention products
                                                  was computed wrong on the GPU: LiteRT#8619)
  OUT/granite_speech_audio6_<F>f_fp16.tflite      ai-edge-quantizer FLOAT_CASTING of every supported weight to fp16,
                                                  explicit DEQUANTIZE, compute in fp32 (export_encoder.py's recipe)
  OUT/rank4_report.json                           the rewrite's sites
  OUT/export_encoder6.json                        per file: bytes, sha256, operator counts, the highest rank of a
                                                  tensor an operator reads or writes, the counts of GATHER, GATHER_ND,
                                                  BROADCAST_TO, SELECT, SELECT_V2, DIV and CUSTOM (Flex included),
                                                  the signature; with --check-wav the window-by-window comparison below

The rewritten fp32 file and the fp16 file must have none of those seven operators and no tensor above rank 4 (exit 1
otherwise). With --check-wav the clip is framed and cut into windows as LiteRT-LM 0.19.1 feeds a streaming audio
encoder (160-sample frames with the last one zero padded, windows of 100 W frames without overlap, the last one zero
padded), and every window goes through each file (ai-edge-litert interpreter, CPU), through a fresh eager port6
module and through the eager port.py module (fp32 torch): the valid count (last non-zero entry of `mask` + 1, as the
runtime counts it) and the mask equal the eager module's, the features are finite, and over the valid rows
cos >= 0.9999999 and max|d| <= 1e-3 for fp32, cos >= 0.9998 and max|d| <= 0.05 for fp16, and port6 against port.py
cos >= 0.9999999 and max|d| <= 1e-3 (exit 1 otherwise).

  python export_encoder6.py --src <snapshot dir> --out <dir> [--win-s 24] [--check-wav <snapshot>/multilingual_sample.wav]
"""
import argparse
import collections
import json
import mmap
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
from export_encoder import interp, op_hist, quantize_fp16, sig_io  # noqa: E402

RANK4 = HERE.parent / "tools" / "gpu_graph" / "tfl_rank4.py"
WATCH = ("GATHER", "GATHER_ND", "BROADCAST_TO", "SELECT", "SELECT_V2", "DIV", "CUSTOM")
BARS = {"fp32": (0.9999999, 1e-3), "fp16": (0.9998, 0.05), "port6_vs_port": (0.9999999, 1e-3)}  # (cos >=, max|d| <=)


def export_fp32_raw(src, out, wf):
    import torch
    import litert_torch
    import port6 as P6  # imported as the top-level module `port6`: the module path is part of every tensor name
    g, n = P6.load_graph6(str(src), win_frames=wf)
    t0 = time.time()
    edge = litert_torch.signature("encode", g, sample_kwargs={"audio": torch.zeros(1, wf, C.HOP)}).convert()
    edge.export(str(out))
    return round(time.time() - t0, 1), n


def rank4(src, out, report):
    p = subprocess.run([sys.executable, str(RANK4), "rewrite", str(src), str(out), "--rules", "bmm", "--unique-names",
                        "--report", str(report)], capture_output=True, text=True, stdin=subprocess.DEVNULL)
    tail = (p.stdout + p.stderr)[-2000:]
    if p.returncode != 0:
        raise SystemExit(f"tfl_rank4.py failed (rc {p.returncode}):\n{tail}")
    return tail.strip()


def audit(path):
    """Operator counts and the highest rank of a tensor an operator reads or writes, from the flatbuffer."""
    from ai_edge_litert import schema_py_generated as S
    names = {v: k for k, v in S.BuiltinOperator.__dict__.items() if isinstance(v, int)}
    hist, max_rank = collections.Counter(), 0
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)  # freed with the last view of it, on return
    m = S.Model.GetRootAsModel(memoryview(mm), 0)
    for si in range(m.SubgraphsLength()):
        g = m.Subgraphs(si)
        for i in range(g.OperatorsLength()):
            op = g.Operators(i)
            code = m.OperatorCodes(op.OpcodeIndex())
            hist[names.get(max(code.BuiltinCode(), code.DeprecatedBuiltinCode()), "?")] += 1
            for ti in [op.Inputs(j) for j in range(op.InputsLength())] + \
                      [op.Outputs(j) for j in range(op.OutputsLength())]:
                if ti >= 0:
                    max_rank = max(max_rank, g.Tensors(ti).ShapeLength())
    return {"n_ops": sum(hist.values()), "max_rank": max_rank, "watch": {k: hist.get(k, 0) for k in WATCH}}


def runtime_windows(x, wf):
    """[1, wf, 160] float32 windows as LiteRT-LM feeds a streaming audio encoder."""
    n = len(x)
    nfr = -(-n // C.HOP)
    buf = np.zeros(nfr * C.HOP, np.float32)
    buf[:n] = x
    fr = buf.reshape(nfr, C.HOP)
    out = []
    for pos in range(0, nfr, wf):
        w = np.zeros((wf, C.HOP), np.float32)
        seg = fr[pos:pos + wf]
        w[:len(seg)] = seg
        out.append(w.reshape(1, wf, C.HOP))
    return out


def cmp(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    cos = float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b)))
    return {"cos": cos, "max_abs": float(np.abs(a - b).max())}


def eager_refs(src, wav, wf):
    """Per window: the eager port6 module's mask / features (a fresh module, not the one the export traced) and the
    eager port.py module's, with the comparison of the two."""
    import torch
    import port
    import port6 as P6
    wins = runtime_windows(C.wav_float(wav), wf)
    eager6, _ = P6.load_graph6(str(src), win_frames=wf)
    eager1, _ = port.load_graph(str(src), win_frames=wf, feature_scale=1.0)
    refs = []
    with torch.no_grad():
        for w in wins:
            o6 = eager6(torch.from_numpy(w))
            o1 = eager1(torch.from_numpy(w))
            m6, f6 = o6["mask"][0].numpy().copy(), o6["features"][0].numpy().copy()
            m1, f1 = o1["mask"][0].numpy().copy(), o1["features"][0].numpy().copy()
            n = int(m6.sum())
            r = {"tokens": n, "port_tokens": int(m1.sum()), "mask_equal_port": bool(np.array_equal(m6, m1))}
            r["port6_vs_port"] = cmp(f6[:n], f1[:n]) if n else None
            refs.append((w, m6, f6, r))
    del eager6, eager1
    return refs


def check(path, refs, threads, bars):
    run = interp(path, threads).get_signature_runner("encode")
    rows = []
    for j, (w, m6, f6, _) in enumerate(refs):
        o = run(audio=w)
        m = o["mask"][0]
        nz = np.nonzero(m)[0]
        valid = int(nz[-1]) + 1 if len(nz) else 0
        n = int(m6.sum())
        feats = o["features"][0]
        r = {"window": j, "valid": valid, "eager_tokens": n, "mask_equal_eager": bool(np.array_equal(m, m6)),
             "finite": bool(np.isfinite(feats).all())}
        r["vs_eager"] = cmp(feats[:n], f6[:n]) if n else None
        r["pass"] = (valid == n and r["mask_equal_eager"] and r["finite"]
                     and (n == 0 or (r["vs_eager"]["cos"] >= bars[0] and r["vs_eager"]["max_abs"] <= bars[1])))
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="snapshot of ibm-granite/granite-speech-5.0-2b (config.json, model.safetensors)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--win-s", type=int, default=24, help="window in seconds (a multiple of 12)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--check-wav", default="", help="a clip to compare the files with the eager modules on")
    ap.add_argument("--keep-raw", action="store_true", help="keep the export before the rank-4 rewrite")
    args = ap.parse_args()
    import torch
    torch.set_num_threads(args.threads)
    wf = args.win_s * C.SR // C.HOP
    nt = wf // 10
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    raw = out / f"granite_speech_audio6_{wf}f_fp32_raw.tflite"
    fp32 = out / f"granite_speech_audio6_{wf}f_fp32.tflite"
    fp16 = out / f"granite_speech_audio6_{wf}f_fp16.tflite"
    want_io = {"inputs": {"audio": {"dtype": "float32", "shape": [1, wf, C.HOP]}},
               "outputs": {"features": {"dtype": "float32", "shape": [1, nt, 2048]},
                           "mask": {"dtype": "uint8", "shape": [1, nt]}}}
    import importlib.metadata as md
    rep = {"src": str(args.src), "win_s": args.win_s, "win_frames": wf, "out_tokens": nt, "threads": args.threads,
           "versions": {p: md.version(p) for p in ("litert-torch", "ai-edge-quantizer", "ai-edge-litert",
                                                   "litert-converter", "torch", "torchaudio")}}
    ok = True

    rep["export_s"], rep["graph_tensors"] = export_fp32_raw(args.src, raw, wf)
    rep["raw"] = {"bytes": raw.stat().st_size, "sha256": C.sha256(raw), **audit(raw)}
    print(f"exported {raw.name} {rep['raw']['bytes']:,} B sha256 {rep['raw']['sha256']} in {rep['export_s']} s "
          f"({rep['graph_tensors']} checkpoint tensors)", flush=True)
    rep["rank4"] = rank4(raw, fp32, out / "rank4_report.json")
    print("tfl_rank4.py:", rep["rank4"].splitlines()[-1], flush=True)
    if not args.keep_raw:
        raw.unlink()
    rep["quantize_fp16_s"] = quantize_fp16(fp32, fp16)

    refs = eager_refs(args.src, args.check_wav, wf) if args.check_wav else []
    if refs:
        rep["check_wav"] = Path(args.check_wav).name
        rep["eager"] = [r for (_, _, _, r) in refs]
        for j, r in enumerate(rep["eager"]):
            c = r["port6_vs_port"]
            r["pass"] = (r["tokens"] == r["port_tokens"] and r["mask_equal_port"]
                         and (c is None or (c["cos"] >= BARS["port6_vs_port"][0]
                                            and c["max_abs"] <= BARS["port6_vs_port"][1])))
            ok &= r["pass"]
            print(f"window {j}: eager port6 vs eager port.py: {json.dumps(r)}", flush=True)

    rep["files"] = {}
    for form, p in (("fp32", fp32), ("fp16", fp16)):
        f = {"bytes": p.stat().st_size, "sha256": C.sha256(p), **audit(p), "ops": op_hist(p)}
        f["signature_encode"] = sig_io(interp(p).get_signature_runner("encode"))
        f["io_ok"] = f["signature_encode"] == want_io
        f["graph_ok"] = f["max_rank"] <= 4 and not any(f["watch"].values())
        ok &= f["io_ok"] and f["graph_ok"]
        print(f"{p.name} {f['bytes']:,} B sha256 {f['sha256']} ops {f['n_ops']} max rank {f['max_rank']} "
              f"{f['watch']} signature {'OK' if f['io_ok'] else f['signature_encode']}", flush=True)
        if refs:
            f["check"] = check(p, refs, args.threads, BARS[form])
            ok &= all(r["pass"] for r in f["check"])
            for r in f["check"]:
                print(f"  window {r['window']}: {json.dumps(r)}", flush=True)
        rep["files"][p.name] = f
    rep["pass"] = bool(ok)
    (out / "export_encoder6.json").write_text(json.dumps(rep, indent=1))
    print("EXPORT_ENCODER6_DONE", "PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
