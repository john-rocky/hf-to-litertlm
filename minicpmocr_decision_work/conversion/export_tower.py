#!/usr/bin/env python3
"""Export one patch-count bucket of the tower (tower_graph.TowerRow), cast its FULLY_CONNECTED weights to float16, read both files,
and check them on the CPU against the module and the author's rows. One bucket a process.

    PYTHONHASHSEED=0 $EXPORT conversion/export_tower.py --N 576 [--qblock 1024] [--keep-fp32]

Writes out/graphs/tower_<tag>_{fp32,fp16fc}.tflite and results/export_tower_<tag>.json. Never overwrites."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import resource
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(os.path.abspath(__file__)).parent
WORK = HERE.parent        # the work directory: snapshot/, out/, results/
sys.path.insert(0, str(HERE))
import fp16fc as Q  # noqa: E402
import lrt  # noqa: E402
from export_decide import footprint_sampler, sha256_file  # noqa: E402


def pictures(N: int):
    """The fixture pictures that fit N, nearest to N first: (tag, n, npz)."""
    rows = []
    for p in (WORK / "out/fix").glob("pic_*.npz"):
        with np.load(p) as z:
            n = int(z["pixel_values"].shape[0])
        if n <= N:
            rows.append((n, p))
    rows.sort(key=lambda x: -x[0])
    if not rows:
        return [("blank", N, blank_picture(N))]
    return [(p.stem[4:], n, np.load(p)) for n, p in rows]


def blank_picture(N: int) -> dict:
    """No fixture: N patches of seeded random values. It only carries the shapes through the export; no author's numbers in it."""
    g = np.random.default_rng(0)
    return {"pixel_values": g.uniform(-1.0, 1.0, (N, 1536)).astype(np.float32), "pos": g.normal(0.0, 0.02, (N, 1024)).astype(np.float32),
            "cos": np.ones((N, 64), np.float32), "sin": np.zeros((N, 64), np.float32)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--qblock", type=int, default=0)
    ap.add_argument("--keep-fp32", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--checks", type=int, default=3, help="fixture pictures checked on the CPU in this process")
    ap.add_argument("--variant", default="", choices=("", "f16s", "xb"), help="other forms of the graph: files tower_<tag>_<variant>32 / <variant>fc, record export_tower_<tag>_<variant>.json; xb = the plain module, another bucket set kept apart")
    a = ap.parse_args()
    import torch

    import tower_graph as T

    torch.set_num_threads(a.threads)
    torch.set_grad_enabled(False)
    tag = T.tag(a.N, a.qblock)
    gdir = WORK / "out/graphs"
    gdir.mkdir(parents=True, exist_ok=True)
    v = a.variant
    p32 = gdir / (f"tower_{tag}_{v}32.tflite" if v else f"tower_{tag}_fp32.tflite")
    p16 = gdir / (f"tower_{tag}_{v}fc.tflite" if v else f"tower_{tag}_fp16fc.tflite")
    rec_path = WORK / (f"results/export_tower_{tag}_{v}.json" if v else f"results/export_tower_{tag}.json")
    for p in (p16, rec_path):
        assert not p.exists(), f"never overwrite {p}"
    t0 = time.time()
    rec = {"N": a.N, "qblock": a.qblock, "tag": tag, "signature": "tower", "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
           "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "versions": {p: importlib.metadata.version(p) for p in ("torch", "litert-torch", "litert-converter", "ai-edge-litert", "ai-edge-quantizer", "numpy")},
           "scripts": {f.name: sha256_file(f)[:16] for f in sorted(HERE.glob("*.py"))}}
    stop = threading.Event()
    watch = {}
    threading.Thread(target=footprint_sampler, args=(stop, watch), daemon=True).start()
    sd = T.load_visual_state(WORK / "snapshot")
    rec["visual_params"] = int(sum(v.numel() for v in sd.values()))
    # the position table is not in the graph: the host reads it
    table = sd.pop("pos_embed.weight")
    assert list(table.shape) == [2304, 1024], table.shape
    assert not [k for k in sd if k.startswith("deepstack")], "no deepstack in this tower"
    rec["variant"] = v
    model = T.graph(sd, a.N, a.qblock, f16s=v == "f16s")
    pics = pictures(a.N)
    name0, n0, z0 = pics[0]
    feed0 = T.pad_inputs(z0["pixel_values"], z0["pos"], z0["cos"], z0["sin"], a.N)
    sample = {k: torch.from_numpy(v) for k, v in feed0.items()}
    ts = time.time()
    y_mod = model(**sample)["rows"][0].numpy()
    rec["sample"] = {"picture": name0, "n": n0, "module_forward_s": round(time.time() - ts, 1)}
    if "rows" in z0:
        rec["sample"]["module_vs_author"] = lrt.stats(y_mod[:n0 // 4], z0["rows"])
    print(json.dumps({"tag": tag, "sample": rec["sample"]}), flush=True)
    rec["memory_after_load"] = lrt.memory()
    if not p32.exists():
        import litert_torch

        t1 = time.time()
        conv = litert_torch.signature("tower", model, sample_kwargs=sample)
        out = conv.convert()
        rec["convert_s"] = round(time.time() - t1, 1)
        t2 = time.time()
        out.export(str(p32))
        rec["write_s"] = round(time.time() - t2, 1)
        del out, conv
    else:
        rec["fp32_reused"] = True
    rec["fp32"] = {"file": str(p32.relative_to(WORK)), "bytes": p32.stat().st_size, "sha256": sha256_file(p32)}
    s32, gen = Q.fc_scan(p32), Q.general_scan(p32)
    rec["scan_fp32"] = {**s32, **gen}
    t3 = time.time()
    rec["fp16fc_recipe"] = Q.fp16fc(p32, p16)
    rec["quant_s"] = round(time.time() - t3, 1)
    rec["fp16fc"] = {"file": str(p16.relative_to(WORK)), "bytes": p16.stat().st_size, "sha256": sha256_file(p16)}
    s16, gen16 = Q.fc_scan(p16), Q.general_scan(p16)
    rec["scan_fp16fc"] = {**s16, **gen16}
    rec["cpu_checks"] = {}
    for form, path in (("fp32", p32), ("fp16fc", p16)):
        g = lrt.Graph(path, "tower", "cpu", threads=a.threads)
        rows = []
        for name, n, z in pics[:a.checks]:
            feed = T.pad_inputs(z["pixel_values"], z["pos"], z["cos"], z["sin"], a.N)
            out = g(**feed)["rows"][0]
            ref = z["rows"] if "rows" in z else y_mod[:n // 4]     # the blank picture: against the module
            rows.append({"picture": name, "n": n, "ms": round(g.last_ms, 1), **lrt.stats(out[:n // 4], ref)})
        g.close()
        rec["cpu_checks"][form] = rows
    NEG = ("COS", "SIN", "GATHER", "GATHER_ND", "BROADCAST_TO", "SELECT_V2", "EMBEDDING_LOOKUP")
    checks = {
        "fp32_forbidden_0": all(gen["op_histogram"].get(k, 0) == 0 for k in NEG),
        "fp32_int64_0": gen["int64_tensor_count"] == 0, "fp32_custom_0": s32["custom_op_count"] == 0, "fp32_rank_gt4_0": gen["rank_gt4_tensor_count"] == 0,
        "fp32_all_constant_input_ops_0": gen["all_constant_input_ops"] == 0,
        "fp32_fc_weights_constant": s32["fc_weight_tensor_dtype_direct"] == {"FLOAT32": s32["fully_connected"]},
        "signature": Q.signature_ok(gen["signatures"], T.contract(a.N, a.qblock)),
        "fp16_fc_count": s16["fully_connected"] == s32["fully_connected"],
        "fp16_fc_weight_all_FLOAT16_via_DEQUANTIZE": s16["fc_weight_source_dtype"] == {"FLOAT16": s32["fully_connected"]},
        "fp16_dequantize_one_per_input_fc_buffer": s16["dequantize_in_out"] == {"FLOAT16->FLOAT32": s32["fc_weight_buffers"]},
        "fp16_op_histogram_input_plus_dequantize": {k: v for k, v in s16["op_histogram"].items() if k != "DEQUANTIZE"} == s32["op_histogram"],
    }
    rec.update(checks=checks, checks_pass=all(checks.values()), failing=[k for k, v in checks.items() if not v])
    if not a.keep_fp32:
        p32.unlink()
        rec["fp32"]["deleted"] = True
    stop.set()
    rec.update(watch)
    rec["memory_end"] = lrt.memory()
    rec["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rec["wall_s"] = round(time.time() - t0, 1)
    rec["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    rec_path.parent.mkdir(parents=True, exist_ok=True)
    rec_path.write_text(json.dumps(rec, indent=1, default=str) + "\n")
    print("EXPORT_DONE " + json.dumps({"tag": tag, "checks_pass": rec["checks_pass"], "failing": rec["failing"], "fp16_bytes": rec["fp16fc"]["bytes"],
                                       "ops": s16["operator_count"], "fc": s16["fully_connected"], "dq": s16["dequantize_count"],
                                       "cpu": {f: [(r["picture"], float(f"{r['rel_rms']:.3g}"), r["ms"]) for r in rows] for f, rows in rec["cpu_checks"].items()},
                                       "footprint_max_gib": round(rec.get("footprint_max", 0) / 2**30, 1), "peak_rss_gib": round(rec["peak_rss_bytes"] / 2**30, 1),
                                       "wall_s": rec["wall_s"]}), flush=True)
    return 0 if rec["checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
