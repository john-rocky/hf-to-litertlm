#!/usr/bin/env python3
"""Round 3 acceptance 4 (files): the tower + merger graph of each fixed grid in two forms, then the static scans.

    TMPDIR=<K>/tmp ~/venvs/lt094dev/bin/python scripts/r3_tower_tfl.py     (after scripts/r3_tower_torch.py)
        -> out/r3/clef_tower_g256_fp32.tflite, out/r3/clef_tower_g448_fp32.tflite (litert-torch default path),
           out/r3/clef_tower_g256_v2_fp16fc.tflite, out/r3/clef_tower_g448_v2_fp16fc.tflite (scripts/r3_fp16fc.py),
           results/r3_tower_files.json, results/r3_prelint_<stem>.json, results/files_r3.json (the ledger rows)
        A grid whose two files and records exist is skipped (resumable); nothing is overwritten.

1. Guard: results/r3_tower_torch.json exists and its float64 identity holds (rewrite vs the author <= 1e-10 rel_rms).
2. Per grid: ClefTower(the checkpoint's model.visual tensors bf16 -> float32, cache/r3/consts_<grid>.npz) on the
   reference's img_01/<grid> pixel_values; the module's rows here (torch 2.13.0) vs the torch check's rows of the same
   run (cache/r3/torch/img_01__<grid>.npy, torch 2.9.0): recorded (float32 noise of two torch builds).
3. litert_torch.convert(tower, sample_kwargs={"patches": [1, N, 1536]}) -> export; then the fp16-FC form.
4. Scan each file (scripts/clef_scan.py full_scan = tflite_scan + all-constant-input ops): GATHER_ND / SELECT_V2 /
   BROADCAST_TO / GATHER = 0, INT64 tensors 0, CUSTOM 0, rank > 4 0, StableHLO 0, all-constant-input ops 0, signature
   patches [1, N, 1536] -> rows [1, N/4, 4096]; ../tools/rank3_prelint.py (warnings recorded); the fp16-FC checks; the
   share of FC weights whose fp16 cast is exact (LayerNorm gamma folded in, scripts/r3_fp16fc.py fold_census).
Memory (ru_maxrss / phys_footprint) and seconds per part are recorded."""
import hashlib
import importlib.metadata
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

K = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(K / "scripts"))
import r3_tower as RT  # noqa: E402
import r3_fp16fc as RQ  # noqa: E402
from clef_export import memory  # noqa: E402
from clef_scan import full_scan  # noqa: E402

REV = "f0b7d3cd65296abfe459e6c228b618f5d8b26d2f"
SNAP = K / "hf/hub/models--Cloudflare--clef-flash/snapshots" / REV
EXP_PY = Path(sys.executable)   # the lt094dev environment that runs this script
FORBID = ("GATHER_ND", "SELECT_V2", "BROADCAST_TO")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def scan_checks(s: dict, n: int, form: dict, fp16fc: bool = False) -> dict:
    """fp16fc = the FLOAT_CASTING file: its all-constant-input ops are exactly the DEQUANTIZE ops (fp16 constant ->
    fp32), one per FC weight buffer, and its FC weights are those DEQUANTIZE outputs (the fp16fc checks of
    scripts/r3_fp16fc.py verify the dtypes); the fp32 file has neither (attempt 1 applied the fp32 checks to both)."""
    sig = s["signatures"]
    want_in = sorted((i["name"], i["shape"], i["dtype"].upper()) for i in RT.contract({256: "g256", 784: "g448"}[n], form)["inputs"])
    sig_ok = (len(sig) == 1 and sorted((i["name"], i["shape"], i["dtype"]) for i in sig[0]["inputs"]) == want_in
              and [(o["name"], o["shape"], o["dtype"]) for o in sig[0]["outputs"]] == [("rows", [1, n // 4, RT.OUT], "FLOAT32")])
    lw = s["lightweight"]
    c = {"forbidden_0": all(s["forbidden_counts"].get(k, 0) == 0 for k in FORBID), "gather_0": s["gather_count"] == 0,
         "int64_0": s["int64_tensor_count"] == 0, "custom_0": s["custom_op_count"] == 0,
         "rank_gt4_0": s["rank_gt4_tensor_count"] == 0, "stablehlo_0": not s["stablehlo_ops"],
         "signature_matches_contract": sig_ok}
    if fp16fc:
        deq = s["op_histogram"].get("DEQUANTIZE", 0)
        c["all_constant_input_ops_are_the_dequantize_ops"] = lw["all_constant_input_ops"] == deq > 0
        c["fc_weights_from_dequantize"] = lw["fc_weight_not_constant"] == s["fully_connected_count"] and deq > 0
    else:
        c["all_constant_input_ops_0"] = lw["all_constant_input_ops"] == 0
        c["fc_weights_constant_with_data"] = lw["fc_weight_not_constant"] == 0 and lw["fc_weight_constant_without_data"] == 0
    return c


def brief(s: dict) -> dict:
    return {k: s[k] for k in ("bytes", "sha256", "operator_count", "op_histogram", "signatures", "forbidden_counts",
                              "int64_tensor_count", "rank_gt4_tensor_count", "custom_op_count", "stablehlo_ops",
                              "gather_count", "embedding_lookup_count", "tensor_rank_histogram", "batch_matmul_count",
                              "fully_connected_count")} | {"all_constant_input_ops": s["lightweight"]["all_constant_input_ops"],
                                                          "constant_bytes_by_dtype": s["lightweight"]["constant_bytes_by_dtype"]}


def prelint(path: Path) -> dict:
    out = K / f"results/r3_prelint_{path.stem}.json"
    if not out.exists():
        r = subprocess.run([str(EXP_PY), str(K.parent / "tools/rank3_prelint.py"), str(path), "--json", str(out), "--quiet"],
                           capture_output=True, text=True)
        if r.returncode not in (0, 1) or not out.exists():
            return {"rc": r.returncode, "stderr_tail": r.stderr[-600:]}
    d = json.loads(out.read_text())
    keys = ("total", "warnings", "summary", "by_rule", "counts")
    return {"file": str(out.relative_to(K)), **{k: d[k] for k in keys if k in d}}


def ledger_rows(paths: list) -> None:
    led = K / "results/files_r3.json"
    doc = json.loads(led.read_text()) if led.exists() else {
        "what": "round 3 files outside git (sha256 at creation; the round's own ledger, merged into files.json later)",
        "files": {}}
    for p in paths:
        doc["files"][str(p.relative_to(K))] = {"bytes": p.stat().st_size, "sha256": sha256_file(p),
                                               "created": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(p.stat().st_mtime))}
    doc["total_bytes"] = sum(v["bytes"] for v in doc["files"].values())
    led.write_text(json.dumps(doc, indent=1) + "\n")


def main() -> int:
    rec_path = K / "results/r3_tower_files.json"
    assert not rec_path.exists(), f"refusing to overwrite {rec_path}"
    tc = json.loads((K / "results/r3_tower_torch.json").read_text())
    assert tc["summary"]["fp64"]["identity_max_rel_rms"] <= 1e-10, tc["summary"]["fp64"]
    import litert_torch

    torch.set_grad_enabled(False)
    torch.set_num_threads(8)
    t_all = time.time()
    doc = {"what": "round 3: tower + merger graphs (fp32 and fp16-FC) of g256 / g448, static scans",
           "versions": {p: importlib.metadata.version(p) for p in ("torch", "litert-torch", "litert-" + "conv" + "erter",
                                                                    "ai-edge-litert", "ai-edge-quantizer")},
           "r3_tower_py_sha256": sha256_file(K / "scripts/r3_tower.py"), "memory_start": memory(), "grids": {}}
    fp = K / "cache/r3/tower_form.json"
    form = json.loads(fp.read_text())["form"] if fp.exists() else dict(RT.DEFAULT_FORM)
    doc["form"] = form
    doc["form_source"] = str(fp.relative_to(K)) if fp.exists() else "r3_tower.DEFAULT_FORM"
    t0 = time.time()
    sd = RT.load_visual_state(SNAP)
    doc["load_seconds"] = round(time.time() - t0, 1)
    doc["visual_params"] = int(sum(v.numel() for v in sd.values()))
    ref = json.loads((K / "ref/records_ref.json").read_text())
    made = []
    for grid, side in RT.GRIDS.items():
        n = side * side
        f32 = K / f"out/r3/clef_tower_{grid}_fp32.tflite"
        f16 = K / f"out/r3/clef_tower_{grid}_v2_fp16fc.tflite"
        g = {"n_patches": n, "rows": n // 4, "files": {}}
        c = RT.load_consts(K / f"cache/r3/consts_{grid}.npz")
        g["consts"] = {"file": f"cache/r3/consts_{grid}.npz", "sha256": sha256_file(K / f"cache/r3/consts_{grid}.npz"),
                       "transformers": str(c["npz"]["transformers"]), "torch": str(c["npz"]["torch"])}
        tower = RT.ClefTower(sd, c["pos_embeds"], c["cos"], c["sin"], form=form).eval()
        r0 = next(r for r in ref["rows"] if r["id"] == "img_01" and r["arm"] == grid)
        pv = np.load(K / f"ref/npz/img_01__{grid}.npz")["pixel_values"]
        sample = torch.from_numpy(pv)[None]
        feed = tower.feed(sample)
        y = tower(**feed)["rows"][0].numpy()
        if form != RT.DEFAULT_FORM:              # the form must not change the function (the torch check's form)
            y_def = RT.ClefTower(sd, c["pos_embeds"], c["cos"], c["sin"]).eval()(sample)["rows"][0].numpy()
            g["form_vs_default_form_equal"] = bool(np.array_equal(y, y_def))
            g["form_vs_default_form_rel_rms"] = float(np.linalg.norm(y.astype(np.float64) - y_def) / np.linalg.norm(y_def))
            if form.get("merge") == "split4":    # reassociated fc1 sum: float32 rounding only
                assert g["form_vs_default_form_rel_rms"] <= 1e-6, g["form_vs_default_form_rel_rms"]
            else:
                assert g["form_vs_default_form_equal"], "the chosen form changes the values in torch"
        prev = np.load(K / f"cache/r3/torch/img_01__{grid}.npy")
        g["sample"] = {"run": f"{r0['id']}/{grid}", "shape": list(sample.shape),
                       "max_abs_vs_torch_check": float(np.abs(y - prev).max()),
                       "bit_equal_torch_check": bool(np.array_equal(y, prev)),
                       "rel_rms_vs_torch_check": float(np.linalg.norm(y.astype(np.float64) - prev) / np.linalg.norm(prev))}
        if not f32.exists():
            t0 = time.perf_counter()
            edge = litert_torch.convert(tower, sample_kwargs=feed)
            g["convert_seconds"] = round(time.perf_counter() - t0, 1)
            t1 = time.perf_counter()
            edge.export(str(f32))
            g["write_seconds"] = round(time.perf_counter() - t1, 1)
            del edge
            g["memory_after_export"] = memory()
            made.append(f32)
        s32 = full_scan(f32)
        g["files"]["fp32"] = {"file": str(f32.relative_to(K)), **brief(s32), "checks": scan_checks(s32, n, form),
                              "prelint": prelint(f32)}
        fc = RQ.fold_census(f32)
        g["fold_census_fp32"] = {k: v for k, v in fc.items() if k != "rows"} | {"rows_first8": fc["rows"][:8]}
        if not f16.exists():
            q = RQ.fp16fc(f32, f16)
            made.append(f16)
        else:
            q = {"note": "file existed (made by an earlier invocation)"}
        s16 = full_scan(f16)
        g["files"]["v2_fp16fc"] = {"file": str(f16.relative_to(K)), **brief(s16), "checks": scan_checks(s16, n, form, fp16fc=True),
                                   "fp16fc": q, "prelint": prelint(f16)}
        g["checks_pass"] = (all(g["files"]["fp32"]["checks"].values()) and all(g["files"]["v2_fp16fc"]["checks"].values())
                            and q.get("checks_pass", True))
        doc["grids"][grid] = g
        print(json.dumps({grid: {"fp32": {k: g["files"]["fp32"][k] for k in ("bytes", "operator_count", "forbidden_counts")},
                                 "fp16fc_bytes": g["files"]["v2_fp16fc"]["bytes"], "checks_pass": g["checks_pass"],
                                 "sample": g["sample"]}}), flush=True)
        del tower
    ledger_rows(made)
    doc.update(memory_end=memory(), peak_rss_bytes_getrusage=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               tmpdir_after=sorted(p.name for p in Path(os.environ.get("TMPDIR", "/nonexistent")).glob("*"))[:50],
               seconds=round(time.time() - t_all, 1), contract={grid: RT.contract(grid, form) for grid in RT.GRIDS},
               checks_pass=all(g["checks_pass"] for g in doc["grids"].values()))
    rec_path.write_text(json.dumps(doc, indent=1, default=str) + "\n")
    print(json.dumps({"checks_pass": doc["checks_pass"], "seconds": doc["seconds"],
                      "peak_rss_gb": round(doc["peak_rss_bytes_getrusage"] / 1e9, 2)}), flush=True)
    return 0 if doc["checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
