#!/usr/bin/env python3
"""Round 3 acceptance 4 (runtime) and the tower's ms: the four tower files (g256 / g448 x fp32 / fp16-FC) through
ai-edge-litert CompiledModel on the Mac CPU (XNNPACK 8 threads), Metal fp32 precision and Metal default precision
(recorded only), on the 14 reference runs of each grid.

    TMPDIR=<K>/tmp ~/venvs/lt094dev/bin/python scripts/r3_tower_rt.py [--deadline-s 1300]
        -> results/r3_tower_rt_<stem>_<accel>.json per (file, accel) (resumable: an existing part is skipped),
           results/r3_tower_rt.json when all 12 parts exist, cache/r3/rt/<stem>_<accel>/<id>__<grid>.npy (the rows),
           logs/r3_rt_<stem>_<accel>.runtime.log (the runtime's delegation lines)

Per (file, accel): compile seconds, is_fully_accelerated, the runtime's `Replacing N out of M node(s) ... yielding P
partitions` / `Partitioned subgraph<i>, selected N ops, from a total of M ops` lines, a refused compile with its text.
Per run: the host's patches (scripts/r3_host.py tile_patches on the fixture image; asserted bit-equal to the reference
npz pixel_values) -> rows [N/4, 4096] vs the author's rows (npz image_embeds) and vs the torch rewrite
(cache/r3/torch): cos, rel_rms, max |diff|, min row cos, non-finite count; the first run repeated (bit-equal expected).
ms: the img_01 run, 5 warm-up calls then 20 timed calls: `call` = write the patches + run + read the rows (what the
host does per picture) and `run` = run_by_name alone; median / min / max / p90, load average before / after; the
measurement lock's line at the start and the end (the window must be this run's own). No element bar: the e2e decides.
Round 3b: every part runs in a fresh process (`--only <stem>:<accel>`, started by the parent), so its compile seconds,
phys_footprint and lifetime maximum carry no other model's residue; a part process that dies without writing leaves a
CRASH part with the runtime log's tail (a crash is a result)."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

K = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(K / "scripts"))
import r3_host as RH  # noqa: E402
import r3_lrt as RL  # noqa: E402
from clef_export import memory  # noqa: E402

ACCELS = ("cpu", "gpu_f32", "gpu_default")
FORMS = ("fp32", "v2_fp16fc")
LOCK = Path(os.environ.get("MEASUREMENT_LOCK_FILE", "/dev/null"))   # the machine-wide measurement lock (its line is recorded)


def stats(got, want) -> dict:
    g = np.asarray(got, np.float64)
    w = np.asarray(want, np.float64)
    d = g - w
    rows = (g * w).sum(-1) / np.maximum(np.linalg.norm(g, axis=-1) * np.linalg.norm(w, axis=-1), 1e-300)
    return {"cos": float(g.ravel() @ w.ravel() / (np.linalg.norm(g) * np.linalg.norm(w))),
            "rel_rms": float(np.linalg.norm(d) / np.linalg.norm(w)), "max_abs": float(np.abs(d).max()),
            "min_row_cos": float(rows.min()), "nonfinite": int((~np.isfinite(g)).sum())}


def feed(m, patches: np.ndarray, pos: np.ndarray) -> dict:
    """The tower's inputs by the compiled signature: patches [1, N, 1536] (+ pos [1, N, C] when the form has it)."""
    f = {"patches": patches[None]}
    if "pos" in m.in_det:
        f["pos"] = pos[None]
    return f


def lock_line() -> str:
    try:
        return LOCK.read_text().strip()
    except OSError:
        return ""


def one(stem: str, grid: str, accel: str, runs: list, deadline: float | None, pos: np.ndarray) -> dict:
    path = K / f"out/r3/{stem}.tflite"
    cache = K / f"cache/r3/rt/{stem}_{accel}"
    cache.mkdir(parents=True, exist_ok=True)
    log = K / f"logs/r3_rt_{stem}_{accel}.runtime.log"
    doc = {"file": str(path.relative_to(K)), "bytes": path.stat().st_size, "grid": grid, "accel": accel,
           "runtime_log": str(log.relative_to(K)), "files_next_to_model_before": RL.listing(path.parent),
           "tmpdir_before": RL.listing(Path(os.environ.get("TMPDIR", "/nonexistent"))), "lock_at_start": lock_line(),
           "memory_before": memory()}
    rl = RL.RuntimeLog(log)
    rows = []
    try:
        with rl:
            m = RL.Model(path, accel)
            doc.update(m.info())
            doc["memory_after_compile"] = memory()
            first = None
            for rid, pv, emb in runs:
                y = m.run(feed(m, pv, pos))["rows"][0]
                np.save(cache / f"{rid}__{grid}.npy", y)
                if first is None:
                    first = (rid, pv, y.copy())
                tc = np.load(K / f"cache/r3/torch/{rid}__{grid}.npy")
                rows.append({"id": rid, "vs_author": stats(y, emb), "vs_torch_rewrite": stats(y, tc)})
            again = m.run(feed(m, first[1], pos))["rows"][0]
            doc["repeat_bit_equal"] = bool(np.array_equal(again, first[2]))
            # ms on the first run (img_01)
            if deadline is None or time.monotonic() < deadline:
                x = first[1][None]
                la0 = os.getloadavg()
                call, run = [], []
                for i in range(25):
                    t = time.perf_counter()
                    m.write(feed(m, x[0], pos))
                    tr = time.perf_counter()
                    m.invoke()
                    tr2 = time.perf_counter()
                    m.read("rows")
                    t2 = time.perf_counter()
                    if i >= 5:
                        call.append((t2 - t) * 1e3)
                        run.append((tr2 - tr) * 1e3)
                doc["ms"] = {"run_id": first[0], "warmup": 5, "timed": 20, "call": RL.ms_stats(call), "run": RL.ms_stats(run),
                             "call_all": [round(v, 3) for v in call], "loadavg_before": list(la0),
                             "loadavg_after": list(os.getloadavg()), "lock_during": lock_line()}
            doc["memory_end"] = memory()
            m.close()
        doc["status"] = "OK"
    except BaseException as e:  # a refused compile or a runtime error is a result
        doc["status"] = "FAIL"
        doc["error"] = f"{type(e).__name__}: {e}"
        import traceback
        doc["traceback_tail"] = traceback.format_exc()[-1500:]
    doc["delegation"] = rl.delegation
    doc["runtime_log_key_lines"] = rl.key_lines
    doc["runtime_log_lines"] = rl.lines
    doc["files_next_to_model_after"] = RL.listing(path.parent)
    doc["new_files_next_to_model"] = sorted(set(doc["files_next_to_model_after"]) - set(doc["files_next_to_model_before"]))
    doc["tmpdir_after"] = RL.listing(Path(os.environ.get("TMPDIR", "/nonexistent")))
    doc["lock_at_end"] = lock_line()
    doc["rows"] = rows
    if rows:
        def w(key, sub, fn):
            r = fn(rows, key=lambda x: x[sub][key])
            return {"value": r[sub][key], "id": r["id"]}
        doc["summary"] = {"runs": len(rows),
                          "vs_author": {"min_cos": w("cos", "vs_author", min), "max_rel_rms": w("rel_rms", "vs_author", max),
                                        "max_abs": w("max_abs", "vs_author", max), "min_row_cos": w("min_row_cos", "vs_author", min)},
                          "vs_torch_rewrite": {"min_cos": w("cos", "vs_torch_rewrite", min),
                                               "max_rel_rms": w("rel_rms", "vs_torch_rewrite", max)},
                          "nonfinite_total": sum(r["vs_author"]["nonfinite"] for r in rows)}
    return doc


def runs_of(grid: str) -> list:
    """The 14 reference runs of a grid: (id, the host's patches, the author's rows), img_01 first (the ms run)."""
    ref = json.loads((K / "ref/records_ref.json").read_text())
    recs = {r["id"]: r for r in json.loads((K / "fixtures/records.json").read_text())["records"]}
    out = []
    for r in ref["rows"]:
        if r["arm"] == grid:
            z = np.load(K / f"ref/npz/{r['id']}__{r['arm']}.npz")
            pv = RH.tile_patches(K / "fixtures/images" / recs[r["id"]]["image_files"][0], RH.TILES[r["arm"]])
            assert np.array_equal(pv.view(np.uint32), z["pixel_values"].view(np.uint32)), (r["id"], r["arm"])
            out.append((r["id"], pv, z["image_embeds"]))
    out.sort(key=lambda x: (x[0] != "img_01", x[0]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deadline-s", type=float, default=None)
    ap.add_argument("--only", default=None, help="<stem>:<accel> = that one part in this process (round 3b: the parent "
                    "starts every part in a fresh process, so compile seconds and phys_footprint are the part's own)")
    a = ap.parse_args()
    t_start = time.monotonic()
    deadline = t_start + a.deadline_s if a.deadline_s else None
    files = json.loads((K / "results/r3_tower_files.json").read_text())
    assert files["checks_pass"], "the tower files did not pass their scans"
    if a.only:
        stem, accel = a.only.split(":")
        grid = stem.split("_")[2]
        assert grid in RH.TILES and accel in ACCELS, a.only
        d = one(stem, grid, accel, runs_of(grid), deadline, np.load(K / f"cache/r3/consts_{grid}.npz")["pos_embeds"])
        d["process"] = "fresh process for this part (round 3b)"
        (K / f"results/r3_tower_rt_{stem}_{accel}.json").write_text(json.dumps(d, indent=1, default=str) + "\n")
        s = d.get("summary", {})
        print(f"{stem} {accel}: {d['status']} {d.get('error', '')} compile {d.get('compile_seconds')} s "
              f"deleg {d['delegation']} min cos {s.get('vs_author', {}).get('min_cos')} "
              f"ms {d.get('ms', {}).get('call', {}).get('median')}", flush=True)
        return 0
    done = []
    for grid in RH.TILES:
        for form in FORMS:
            stem = f"clef_tower_{grid}_{form}"
            for accel in ACCELS:
                part = K / f"results/r3_tower_rt_{stem}_{accel}.json"
                if part.exists():
                    done.append(part)
                    continue
                if deadline is not None and time.monotonic() > deadline:
                    print(f"DEADLINE before {stem} {accel}", flush=True)
                    return 0
                cmd = [sys.executable, str(Path(__file__).resolve()), "--only", f"{stem}:{accel}"]
                if deadline is not None:
                    cmd += ["--deadline-s", f"{max(deadline - time.monotonic(), 1.0):.0f}"]
                t0 = time.monotonic()
                rc = subprocess.run(cmd, stdin=subprocess.DEVNULL).returncode
                if not part.exists():       # the process died before writing (a crash in the runtime is a result too)
                    log = K / f"logs/r3_rt_{stem}_{accel}.runtime.log"
                    tail = log.read_text(errors="replace")[-3000:] if log.exists() else ""
                    part.write_text(json.dumps({"file": f"out/r3/{stem}.tflite", "grid": grid, "accel": accel,
                                                "status": "CRASH", "error": f"part process exit code {rc}",
                                                "seconds": round(time.monotonic() - t0, 1), "runtime_log_tail": tail,
                                                "delegation": RL.delegation_of(tail), "rows": []}, indent=1) + "\n")
                    print(f"{stem} {accel}: CRASH rc {rc}", flush=True)
                done.append(part)
    parts = {p.stem.replace("r3_tower_rt_", ""): json.loads(p.read_text()) for p in done}
    merged = {"what": "round 3: the tower files through CompiledModel CPU / Metal fp32 / Metal default (14 runs each)",
              "parts": {k: {kk: v.get(kk) for kk in ("file", "accel", "status", "error", "compile_seconds",
                                                    "is_fully_accelerated", "delegation", "repeat_bit_equal", "summary",
                                                    "new_files_next_to_model", "lock_at_start", "lock_at_end")}
                        | {"ms_call_median": v.get("ms", {}).get("call", {}).get("median"),
                           "ms_run_median": v.get("ms", {}).get("run", {}).get("median"),
                           "ms_call": v.get("ms", {}).get("call"), "ms_loadavg": [v.get("ms", {}).get("loadavg_before"),
                                                                                  v.get("ms", {}).get("loadavg_after")],
                           "memory_before": v.get("memory_before"), "memory_after_compile": v.get("memory_after_compile"),
                           "memory_end": v.get("memory_end"), "process": v.get("process")} for k, v in parts.items()},
              "seconds_this_invocation": round(time.monotonic() - t_start, 1)}
    (K / "results/r3_tower_rt.json").write_text(json.dumps(merged, indent=1, default=str) + "\n")
    print(json.dumps({k: (v["status"], v["delegation"], v["ms_call_median"]) for k, v in merged["parts"].items()}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
