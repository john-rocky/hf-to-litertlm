"""Export one length bucket of the d1-Omni-2B audio encoder (d1_audio.D1Audio: downsampler + 17-layer LFM2 encoder +
projector) with litert-torch, cast its FULLY_CONNECTED weights to float16 (the decision graph's recipe: float_casting,
channelwise; convolutions, norms, biases and activations stay float32), check both files on the CPU against the module,
scan them and dump their elementwise operand ranks. One bucket a process.

  PYTHONHASHSEED=0 $EXPORT conversion/export_audio.py --T 1000 [--keep-fp32]

Check inputs: the provider feature extractor's input_features of the synthetic clips that
`conversion/audio_check.py --synthetic --save-feats out/feats` saved (the export venv has no provider code): the
longest clip that fits the bucket and a 2.5 s clip padded into it (the padding: only the clip's ceil(F / 8) tokens are
compared, as the host reads them).

Writes out/graphs/d1-omni-2b_audio_T{T}_{fp32,fp16fc}.tflite (the float32 file is deleted after the fp16 one unless
--keep-fp32) and results/export_audio_T{T}.json. Existing outputs are never overwritten.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import resource
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
CLIPS = ("syn_0.3s", "syn_2.5s", "syn_5.05s", "syn_7.7s", "syn_10s", "syn_13.3s", "syn_20s", "syn_26.6s", "syn_30s")


def check_clips(T):
    """[(name, features [128, F])]: the longest saved clip with F <= T, then syn_2.5s."""
    import numpy as np

    feats = {}
    for n in CLIPS:
        p = os.path.join(WORK, "out/feats", f"{n}.npy")
        if os.path.exists(p):
            feats[n] = np.load(p)
    fit = [n for n in CLIPS if n in feats and feats[n].shape[1] <= T]
    assert fit and "syn_2.5s" in feats, "run audio_check.py --synthetic --save-feats out/feats first"
    names = [max(fit, key=lambda n: feats[n].shape[1])]
    if names[0] != "syn_2.5s":
        names.append("syn_2.5s")
    return [(n, feats[n]) for n in names]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, required=True, help="mel frames, a multiple of 8")
    ap.add_argument("--keep-fp32", action="store_true")
    a = ap.parse_args()
    import numpy as np
    import torch

    import d1_audio as A
    from export_decide import fp16fc, sha256_file
    from rank_dump import dump
    from tflite_scan_d1 import scan

    T = a.T
    assert T % 8 == 0
    tag = f"T{T}"
    gdir = os.path.join(WORK, "out/graphs")
    os.makedirs(gdir, exist_ok=True)
    p32 = os.path.join(gdir, f"d1-omni-2b_audio_{tag}_fp32.tflite")
    p16 = os.path.join(gdir, f"d1-omni-2b_audio_{tag}_fp16fc.tflite")
    rec_path = os.path.join(WORK, f"results/export_audio_{tag}.json")
    for p in (p16, rec_path):
        assert not os.path.exists(p), f"never overwrite {p}"
    torch.set_num_threads(8)
    t0 = time.time()
    model = A.build(T).eval().requires_grad_(False)
    clips = check_clips(T)
    samples = []
    for name, f in clips:
        mel, valid = A.make_inputs(f, T)
        s = {"mel": torch.from_numpy(mel), "valid": torch.from_numpy(valid)}
        with torch.inference_mode():
            ref = model(**s)["features"].numpy()
        n = A.tokens_for(f.shape[1])
        samples.append((name, f.shape[1], n, s, ref[0, :n]))
    sample = samples[0][3]
    rec = {"T": T, "tag": tag, "graph": "d1_audio.D1Audio", "signature": "audio",
           "inputs": {k: list(v.shape) for k, v in sample.items()}, "outputs": {"features": [1, T // 8, 2048]},
           "check_clips": [{"clip": n, "frames": fr, "tokens": t} for n, fr, t, _, _ in samples],
           "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
           "versions": {p: importlib.metadata.version(p) for p in
                        ("torch", "litert-torch", "litert-converter", "ai-edge-litert", "ai-edge-quantizer", "numpy")},
           "load_build_s": round(time.time() - t0, 1)}
    import litert_torch

    t1 = time.time()
    if not os.path.exists(p32):
        lrt = litert_torch.signature("audio", model, sample_kwargs=sample).convert()
        rec["convert_s"] = round(time.time() - t1, 1)
        lrt.export(p32)
        del lrt
    else:
        rec["fp32_reused"] = True
    rec["fp32"] = {"file": os.path.relpath(p32, WORK), "bytes": os.path.getsize(p32), "sha256": sha256_file(p32)}
    from ai_edge_litert.interpreter import Interpreter

    def cpu_check(path, key):
        it = Interpreter(model_path=path, num_threads=8)
        run = it.get_signature_runner("audio")
        rows = []
        for name, fr, n, s, ref in samples:
            out = run(**{k: v.numpy() for k, v in s.items()})["features"][0, :n]
            d = np.abs(out - ref)
            rows.append({"clip": name, "max_abs": float(d.max()), "mean_abs": float(d.mean()),
                         "ref_max_abs": float(np.abs(ref).max())})
        del it
        rec[f"{key}_cpu_vs_module"] = rows
        return max(r["max_abs"] for r in rows)
    t3 = time.time()
    rec["fp32_cpu_vs_module_max_abs"] = cpu_check(p32, "fp32")
    rec["fp32_cpu_s"] = round(time.time() - t3, 1)
    t4 = time.time()
    rec["recipe"] = fp16fc(p32, p16)
    rec["quant_s"] = round(time.time() - t4, 1)
    rec["fp16fc"] = {"file": os.path.relpath(p16, WORK), "bytes": os.path.getsize(p16), "sha256": sha256_file(p16)}
    rec["fp16fc_cpu_vs_module_max_abs"] = cpu_check(p16, "fp16fc")
    rec["scan_fp32"] = scan(p32, with_sha=False)
    rec["scan_fp16fc"] = scan(p16, with_sha=False)
    rows = dump(p16, show_all=True, quiet=True)
    rec["rank_dump_fp16fc"] = rows
    rec["risk_r3_c4_ops"] = sum(r["count"] for r in rows if r["risk"].startswith("RISK"))
    rec["mixed_const_ops"] = sum(r["count"] for r in rows if r["mixed_const"])
    if not a.keep_fp32:
        os.remove(p32)
        rec["fp32"]["deleted"] = True
    rec["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rec["wall_s"] = round(time.time() - t0, 1)
    json.dump(rec, open(rec_path, "w"), indent=1)
    s16 = rec["scan_fp16fc"]
    print(json.dumps({"tag": tag, "fp16fc_bytes": rec["fp16fc"]["bytes"],
                      "fp32_cpu_vs_module_max_abs": rec["fp32_cpu_vs_module_max_abs"],
                      "fp16fc_cpu_vs_module_max_abs": rec["fp16fc_cpu_vs_module_max_abs"],
                      "ops": s16["operator_count"], "fc_weight_dtypes": s16["fc_weight_dtypes"],
                      "custom": s16["custom_ops"], "rank_gt4": s16["rank_gt4_tensor_count"],
                      "int64": s16["int64_tensor_count"], "risk_r3_c4": rec["risk_r3_c4_ops"],
                      "mixed_const": rec["mixed_const_ops"],
                      "peak_rss_gb": round(rec["peak_rss_bytes"] / 2**30, 1), "wall_s": rec["wall_s"]}))


if __name__ == "__main__":
    main()
