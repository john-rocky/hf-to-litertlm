#!/usr/bin/env python3
"""One timed cell (conversion/sets/ladder_workload.json): one form x one accelerator x one workload, one process, on a quiet machine. N warm-ups, then M timed requests through the host; median, min, max of the whole request and of each part.

    <python> conversion/speed.py <work> --form fp16fc --acc gpu_f32 --workload text [--warm 5] [--timed 20] [--no-rows-cache]

Parts: pictures (decode, resize, patches, position rows, rotary), tower run per picture, decide run, the rest = host. Compile = the wall of each
CompiledModel.from_file on its first use, and the first request's wall. Memory = phys_footprint after the first request and its lifetime max at
the end. Writes results/speed/<form>__<acc>__<workload>[suffix].json, every value kept. Refuses to overwrite."""
import argparse
import importlib.metadata
import json
import os
import statistics
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("work", type=Path, help="the work directory")
ap.add_argument("--form", default="fp16fc")
ap.add_argument("--tower-form", default="")
ap.add_argument("--acc", default="gpu_f32")
ap.add_argument("--workload", required=True, choices=("text", "formula", "table"))
ap.add_argument("--warm", type=int, default=5)
ap.add_argument("--timed", type=int, default=20)
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--sharing", action="store_true")
ap.add_argument("--multi", default="", help="the decision graph from one multi-signature file (path from the work directory)")
ap.add_argument("--multi-tower", default="", help="the tower from one multi-signature file (path from the work directory)")
ap.add_argument("--tower-acc", default="", help="the tower's accelerator when it differs from --acc")
ap.add_argument("--weight-cache-dir", default="", help="CPU: XNNPACK weight cache files here (the first process writes them)")
ap.add_argument("--graphs", default="out/graphs")
ap.add_argument("--suffix", default="")
ap.add_argument("--lengths", default="")
ap.add_argument("--patches", default="")
a = ap.parse_args()
WORK = a.work.resolve()
sys.path.insert(0, str(WORK / "host"))
sys.path.insert(0, str(WORK / "conversion"))
import lrt  # noqa: E402
from decide_host import host as H  # noqa: E402
from decide_host.litert_backends import LiteRTDecide, LiteRTMultiDecide, LiteRTMultiTower, LiteRTTower, discover  # noqa: E402

SNAP, BENCH = WORK / "snapshot", WORK / "bench"
out_dir = WORK / "results/speed"
out_dir.mkdir(parents=True, exist_ok=True)
tag = f"{a.form}__{a.acc}__{a.workload}{a.suffix}"
out_path = out_dir / f"{tag}.json"
assert not out_path.exists(), f"refusing to overwrite {out_path}"


W = json.loads((WORK / "conversion/sets/ladder_workload.json").read_text())
rows_all = {r["item_id"]: r for r in map(json.loads, (BENCH / "data/jevocrbench.jsonl").read_text(encoding="utf-8").splitlines())}


def rec_paths(i):
    r = rows_all[i]
    return str(BENCH / r["image_path"]), str(BENCH / r["render_image_path"]), str(r["ocr_text"] or "")


sizes_l = {int(x) for x in a.lengths.split(",")} if a.lengths else None
sizes_n = {int(x) for x in a.patches.split(",")} if a.patches else None
d_files = discover(WORK / a.graphs, "decide", a.form, sizes_l)
t_files = discover(WORK / a.graphs, "tower", a.tower_form or a.form, sizes_n)
doc = {"form": a.form, "tower_form": a.tower_form or a.form, "acc": a.acc, "tower_acc": a.tower_acc or a.acc, "workload": a.workload, "threads": a.threads, "sharing": a.sharing, "warm": a.warm, "timed": a.timed,
       "decide_files": {k: v.name for k, v in d_files.items()}, "tower_files": {k: v.name for k, v in t_files.items()},
       "ai_edge_litert": importlib.metadata.version("ai-edge-litert"), "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "loadavg_before": os.getloadavg(),
       "memory_before": lrt.memory()}
wc = str(WORK / a.weight_cache_dir) if a.weight_cache_dir else ""
if wc:
    Path(wc).mkdir(parents=True, exist_ok=True)
if a.multi_tower:
    tower = LiteRTMultiTower(WORK / a.multi_tower, a.tower_acc or a.acc, a.threads, a.sharing, str(Path(wc) / (Path(a.multi_tower).stem + ".xnnpack_cache")) if wc else "")
    doc["tower_files"] = {"multi": a.multi_tower, "signatures": list(tower.sigs.values())}
else:
    tower = LiteRTTower(t_files, a.tower_acc or a.acc, a.threads, keep=8, sharing=a.sharing, weight_cache_dir=wc)
if a.multi:
    decide = LiteRTMultiDecide(WORK / a.multi, a.acc, a.threads, a.sharing, str(Path(wc) / (Path(a.multi).stem + ".xnnpack_cache")) if wc else "")
    doc["decide_files"] = {"multi": a.multi, "signatures": list(decide.sigs.values())}
else:
    decide = LiteRTDecide(d_files, a.acc, a.threads, keep=4, sharing=a.sharing, weight_cache_dir=wc)
doc["weight_cache_dir"] = a.weight_cache_dir
host = H.Host(tower, decide, SNAP)


def one_request():
    """-> {"total", "pictures", "tower": [..], "decide": [..], "host"} in ms, and the probs."""
    t0 = time.perf_counter()
    parts = {"pictures": 0.0, "tower": [], "decide": [], "tower_io": [], "decide_io": []}
    run_tower, run_decide = tower.__call__, decide.__call__

    def t_call(*x):
        r = run_tower(*x)
        parts["tower"].append(tower.last["run"])
        parts["tower_io"].append(tower.last["write"] + tower.last["read"])
        return r

    def d_call(*x):
        r = run_decide(*x)
        parts["decide"].append(decide.last["run"])
        parts["decide_io"].append(decide.last["write"] + decide.last["read"])
        return r

    host.tower, host.decide = t_call, d_call     # plain functions: the host reads no limit from them
    probs = []
    img, rend, text = rec_paths(W[a.workload]["item_id"])
    ts = time.perf_counter()
    pa, pb = host.picture(img), host.picture(rend)
    parts["pictures"] += (time.perf_counter() - ts) * 1000
    r = host._score(pa, pb, text)
    assert r["status"] == "ok", r
    probs.append(r["prob_1"])
    parts["total"] = (time.perf_counter() - t0) * 1000
    parts["host"] = parts["total"] - parts["pictures"] - sum(parts["tower"]) - sum(parts["decide"])
    host.tower, host.decide = tower, decide
    return parts, probs


t0 = time.perf_counter()
first, probs0 = one_request()
doc["first_request_ms"] = first
doc["first_request_wall_s"] = round(time.perf_counter() - t0, 2)
doc["compile"] = {"tower": tower.load_log, "decide": decide.load_log}
doc["memory_after_first"] = lrt.memory()
for _ in range(a.warm - 1):
    one_request()
vals = []
for _ in range(a.timed):
    p, probs = one_request()
    assert probs == probs0 or max(abs(x - y) for x, y in zip(probs, probs0)) < 1e-3, (probs, probs0)
    vals.append(p)
med = lambda v: round(statistics.median(v), 2)  # noqa: E731
doc["total_ms"] = {"median": med([v["total"] for v in vals]), "min": round(min(v["total"] for v in vals), 2), "max": round(max(v["total"] for v in vals), 2), "values": [round(v["total"], 1) for v in vals]}
doc["parts_median_ms"] = {"pictures": med([v["pictures"] for v in vals]), "tower_run": med([sum(v["tower"]) for v in vals]), "decide_run": med([sum(v["decide"]) for v in vals]),
                          "host_rest": med([v["host"] for v in vals]), "tower_io": med([sum(v["tower_io"]) for v in vals]), "decide_io": med([sum(v["decide_io"]) for v in vals])}
doc["per_call_median_ms"] = {"tower": [med([v["tower"][i] for v in vals]) for i in range(len(vals[0]["tower"]))], "decide": [med([v["decide"][i] for v in vals]) for i in range(len(vals[0]["decide"]))]}
doc["prob_1"] = probs0
doc["prob_1_last"] = probs
doc["buckets_last"] = {"tower": tower.last_bucket, "decide": decide.last_bucket}
doc["memory_end"] = lrt.memory()
doc["loadavg_after"] = os.getloadavg()
doc["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
tower.close()
decide.close()
out_path.write_text(json.dumps(doc, indent=1) + "\n")
print("SPEED_DONE " + json.dumps({"tag": tag, "median_ms": doc["total_ms"]["median"], "parts": doc["parts_median_ms"], "first_wall_s": doc["first_request_wall_s"],
                                  "footprint_gib": round(doc["memory_end"].get("lifetime_max_phys_footprint", 0) / 2**30, 2)}), flush=True)
