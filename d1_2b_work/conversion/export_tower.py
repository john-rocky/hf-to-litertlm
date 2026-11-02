"""Export one canvas of the d1-2B picture tower (d1_tower.D1Tower: vision tower + projector) with litert-torch, then
cast its FULLY_CONNECTED weights to float16 (the decision graph's recipe; the depthwise 3x3 convolutions, norms and
biases stay float32), then check both files on the CPU against the module and scan them. One canvas a process.

  PYTHONHASHSEED=0 $EXPORT conversion/export_tower.py --canvas 32x48 [--qblock 1152] [--keep-fp32]

The check input is a real picture (the COCO card picture, processor pixel values) placed on the canvas, so the
float32 file's difference from the module is the conversion's own on real activations.

Writes out/graphs/d1-2b_tower_G{GH}x{GW}{_qb<Q>}_{fp32,fp16fc}.tflite and results/export_tower_G...json.
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


def coco_pixels():
    """The processor's pixel values of the card's COCO picture, saved by conversion/save_pixels.py (no torch model
    needed here): out/pixels/<name>.npz with pixel_values [n, 1536] and grid (gh, gw)."""
    import numpy as np

    p = os.path.join(WORK, "out/pixels/coco_val2017_000000039769.npz")
    z = np.load(p)
    return z["pixel_values"], tuple(int(x) for x in z["grid"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canvas", required=True, help="GHxGW in patches, both even")
    ap.add_argument("--qblock", type=int, default=0)
    ap.add_argument("--keep-fp32", action="store_true")
    a = ap.parse_args()
    import numpy as np
    import torch

    import d1_tower as T
    from export_decide import fp16fc, sha256_file
    from tflite_scan_d1 import scan

    GH, GW = (int(x) for x in a.canvas.split("x"))
    tag = f"G{GH}x{GW}" + (f"_qb{a.qblock}" if a.qblock else "")
    gdir = os.path.join(WORK, "out/graphs")
    os.makedirs(gdir, exist_ok=True)
    p32 = os.path.join(gdir, f"d1-2b_tower_{tag}_fp32.tflite")
    p16 = os.path.join(gdir, f"d1-2b_tower_{tag}_fp16fc.tflite")
    rec_path = os.path.join(WORK, f"results/export_tower_{tag}.json")
    for p in (p16, rec_path):
        assert not os.path.exists(p), f"never overwrite {p}"
    torch.set_num_threads(8)
    t0 = time.time()
    vcfg = T.read_vcfg()
    W = T.load_vision_weights(dtype=torch.float32)
    model = T.D1Tower(W, vcfg, GH, GW, a.qblock or None, torch.float32).eval().requires_grad_(False)
    pv, (gh, gw) = coco_pixels()
    if gh > GH or gw > GW:          # the card picture does not fit: a random picture-sized block instead
        gh, gw = GH - GH % 2, GW - GW % 2
        pv = np.random.default_rng(0).uniform(-1, 1, (gh * gw, 1536)).astype(np.float32)
        pv = pv.reshape(gh * gw, 3, 2, 256)
        pv[:, :, 1] = pv[:, :, 0]
        pv = pv.reshape(gh * gw, 1536)
    patches, valid = T.to_canvas(pv, gh, gw, GH, GW)
    sample = {"patches": torch.from_numpy(patches), "valid": torch.from_numpy(valid)}
    with torch.inference_mode():
        ref = model(**sample)["features"].numpy()
    ref_pic = T.from_canvas(ref, gh, gw, GH, GW)
    rec = {"canvas": [GH, GW], "qblock": a.qblock or None, "tag": tag, "graph": "d1_tower.D1Tower",
           "signature": "tower", "inputs": {k: list(v.shape) for k, v in sample.items()},
           "check_picture_grid": [gh, gw], "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
           "versions": {p: importlib.metadata.version(p) for p in
                        ("torch", "litert-torch", "litert-converter", "ai-edge-litert", "ai-edge-quantizer", "numpy")},
           "load_build_s": round(time.time() - t0, 1)}
    import litert_torch

    t1 = time.time()
    if not os.path.exists(p32):
        lrt = litert_torch.signature("tower", model, sample_kwargs=sample).convert()
        rec["convert_s"] = round(time.time() - t1, 1)
        lrt.export(p32)
        del lrt
    else:
        rec["fp32_reused"] = True
    rec["fp32"] = {"file": os.path.relpath(p32, WORK), "bytes": os.path.getsize(p32), "sha256": sha256_file(p32)}
    from ai_edge_litert.interpreter import Interpreter

    def cpu_run(path):
        it = Interpreter(model_path=path, num_threads=8)
        out = it.get_signature_runner("tower")(**{k: v.numpy() for k, v in sample.items()})["features"]
        del it
        return T.from_canvas(out, gh, gw, GH, GW)
    t3 = time.time()
    got = cpu_run(p32)
    rec["fp32_cpu_vs_module_max_abs"] = float(np.abs(got - ref_pic).max())
    rec["fp32_cpu_s"] = round(time.time() - t3, 1)
    rec["module_ref_max_abs"] = float(np.abs(ref_pic).max())
    t4 = time.time()
    rec["recipe"] = fp16fc(p32, p16)
    rec["quant_s"] = round(time.time() - t4, 1)
    rec["fp16fc"] = {"file": os.path.relpath(p16, WORK), "bytes": os.path.getsize(p16), "sha256": sha256_file(p16)}
    got16 = cpu_run(p16)
    rec["fp16fc_cpu_vs_module_max_abs"] = float(np.abs(got16 - ref_pic).max())
    rec["fp16fc_cpu_vs_module_mean_abs"] = float(np.abs(got16 - ref_pic).mean())
    rec["scan_fp32"] = scan(p32, with_sha=False)
    rec["scan_fp16fc"] = scan(p16, with_sha=False)
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
                      "int64": s16["int64_tensor_count"], "largest": s16["largest_tensors"][:3],
                      "peak_rss_gb": round(rec["peak_rss_bytes"] / 2**30, 1), "wall_s": rec["wall_s"]}))


if __name__ == "__main__":
    main()
