#!/usr/bin/env python3
"""Round 3b: the reading of the four tower files' runtime check (scripts/r3_tower_rt.py parts and the rows it saved)
that the driver takes before the image e2e — no model is opened, NumPy only.

    venv-ref/bin/python scripts/r3b_rt_verdict.py        -> results/r3b_rt_verdict.json, exit 0 / 3 / 4

Registered in the round record before the numbers (the reading of the tower's values):
  compared with the reference's image rows (ref/npz/<id>__<grid>.npz image_embeds = the author's fp32 rows);
  mismatch (exit 3, the stop "the real tower does not match the reference rows on Metal fp32"): on some run of
    some file the Metal fp32 rows hold a non-finite value, or their cos < 0.999 while the same file's CPU rows on that
    run have cos >= 0.999; a Metal fp32 part that failed (status FAIL) counts the same;
  judgment needed (exit 4): the CPU rows of a file have cos < 0.999 or a non-finite value on some run, or a file's
    Metal fp32 delegation is not every node in one partition (r3_e2e.py refuses such a model on Metal);
  recorded only: rel_rms and its Metal / CPU ratio, Metal-vs-CPU rows (cos, rel_rms), the Metal default-precision part.
The e2e parity line decides the rest (no element bar)."""
import json
import sys
from pathlib import Path

import numpy as np

K = Path(__file__).resolve().parents[1]
GRIDS = ("g256", "g448")
FORMS = ("fp32", "v2_fp16fc")
ACCELS = ("cpu", "gpu_f32", "gpu_default")
COS_MIN = 0.999


def pair(a: np.ndarray, b: np.ndarray) -> dict:
    """a vs b (b = the comparison rows): cos, rel_rms, max |a - b|."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    d = a - b
    return {"cos": float(a.ravel() @ b.ravel() / (np.linalg.norm(a) * np.linalg.norm(b))),
            "rel_rms": float(np.linalg.norm(d) / np.linalg.norm(b)), "max_abs": float(np.abs(d).max())}


def one_partition(dlg: dict | None) -> bool:
    rep = (dlg or {}).get("replacing") or []
    return bool(rep) and all(x["delegated"] == x["total"] and x["partitions"] == 1 for x in rep)


def main() -> int:
    out = {"what": "round 3b: the tower runtime reading before the image e2e (pre-registered)",
           "cos_min": COS_MIN, "files": {}, "mismatch": [], "judgment": []}
    for grid in GRIDS:
        for form in FORMS:
            stem = f"clef_tower_{grid}_{form}"
            parts = {}
            for accel in ACCELS:
                p = K / f"results/r3_tower_rt_{stem}_{accel}.json"
                parts[accel] = json.loads(p.read_text()) if p.exists() else None
            f = {"parts": {a: (None if d is None else {"status": d.get("status"), "error": d.get("error"),
                                                      "compile_seconds": d.get("compile_seconds"),
                                                      "is_fully_accelerated": d.get("is_fully_accelerated"),
                                                      "delegation": d.get("delegation"),
                                                      "repeat_bit_equal": d.get("repeat_bit_equal"),
                                                      "summary": d.get("summary")})
                           for a, d in parts.items()}}
            cpu, gpu = parts["cpu"], parts["gpu_f32"]
            if cpu is None or cpu.get("status") != "OK" or not cpu.get("rows"):
                out["judgment"].append(f"{stem}: the CPU part is missing or failed")
            if gpu is None or gpu.get("status") != "OK" or not gpu.get("rows"):
                out["mismatch"].append(f"{stem}: the Metal fp32 part is missing or failed "
                                       f"({None if gpu is None else gpu.get('error')})")
            if gpu is not None and not one_partition(gpu.get("delegation")):
                out["judgment"].append(f"{stem}: Metal fp32 is not every node in one partition {gpu.get('delegation')}")
            per_run = []
            cpu_rows = {r["id"]: r for r in (cpu or {}).get("rows", [])}
            for r in (gpu or {}).get("rows", []):
                rid = r["id"]
                c = cpu_rows.get(rid)
                item = {"id": rid, "metal_vs_author": r["vs_author"], "cpu_vs_author": c["vs_author"] if c else None}
                ym = np.load(K / f"cache/r3/rt/{stem}_gpu_f32/{rid}__{grid}.npy")
                cp = K / f"cache/r3/rt/{stem}_cpu/{rid}__{grid}.npy"
                if cp.exists():
                    item["metal_vs_cpu"] = pair(ym, np.load(cp))
                dp = K / f"cache/r3/rt/{stem}_gpu_default/{rid}__{grid}.npy"
                if dp.exists() and cp.exists():
                    item["metal_default_vs_cpu"] = pair(np.load(dp), np.load(cp))
                m_bad = r["vs_author"]["nonfinite"] > 0 or (
                    r["vs_author"]["cos"] < COS_MIN and (c is None or c["vs_author"]["cos"] >= COS_MIN))
                if m_bad:
                    out["mismatch"].append(f"{stem} {rid}: Metal fp32 cos {r['vs_author']['cos']:.6f} "
                                           f"nonfinite {r['vs_author']['nonfinite']} (CPU cos "
                                           f"{None if c is None else round(c['vs_author']['cos'], 6)})")
                per_run.append(item)
            for rid, c in cpu_rows.items():
                if c["vs_author"]["nonfinite"] > 0 or c["vs_author"]["cos"] < COS_MIN:
                    out["judgment"].append(f"{stem} {rid}: CPU cos {c['vs_author']['cos']:.6f} "
                                           f"nonfinite {c['vs_author']['nonfinite']}")
            if per_run:
                mx = lambda key, sub: max(x[sub][key] for x in per_run if x.get(sub))  # noqa: E731
                mn = lambda key, sub: min(x[sub][key] for x in per_run if x.get(sub))  # noqa: E731
                f["summary"] = {
                    "runs": len(per_run),
                    "metal_vs_author": {"min_cos": mn("cos", "metal_vs_author"), "max_rel_rms": mx("rel_rms", "metal_vs_author")},
                    "cpu_vs_author": ({"min_cos": mn("cos", "cpu_vs_author"), "max_rel_rms": mx("rel_rms", "cpu_vs_author")}
                                      if all(x.get("cpu_vs_author") for x in per_run) else None),
                    "metal_vs_cpu": ({"min_cos": mn("cos", "metal_vs_cpu"), "max_rel_rms": mx("rel_rms", "metal_vs_cpu")}
                                     if all(x.get("metal_vs_cpu") for x in per_run) else None),
                    "metal_default_vs_cpu": ({"min_cos": mn("cos", "metal_default_vs_cpu"),
                                              "max_rel_rms": mx("rel_rms", "metal_default_vs_cpu")}
                                             if all(x.get("metal_default_vs_cpu") for x in per_run) else None)}
                s = f["summary"]
                if s["cpu_vs_author"]:
                    s["ratio_max_rel_rms_metal_over_cpu"] = s["metal_vs_author"]["max_rel_rms"] / s["cpu_vs_author"]["max_rel_rms"]
            f["runs"] = per_run
            out["files"][stem] = f
    out["verdict"] = "mismatch" if out["mismatch"] else ("judgment" if out["judgment"] else "match")
    (K / "results/r3b_rt_verdict.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"verdict": out["verdict"], "mismatch": out["mismatch"][:10], "judgment": out["judgment"][:10],
                      "summary": {k: v.get("summary", {}).get("metal_vs_author") for k, v in out["files"].items()}},
                     indent=1))
    return {"match": 0, "mismatch": 3, "judgment": 4}[out["verdict"]]


if __name__ == "__main__":
    raise SystemExit(main())
