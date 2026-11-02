"""Speed of the PyTorch-free host on the LiteRT graphs, the card's workloads (set before any number): One question,
3 questions, 5K-token state, 384 px picture, Box, Boxes, 64 states in one call (64 requests, the host calls the graph
once per request). Run alone on the machine (the conversion ran each timing process in a measurement window that no
other heavy job shared):

  $EXPORT conversion/bench_host.py --accel gpu --precision fp32 --out results/bench_host_gpu_fp32.json

Per workload: the graphs it needs are loaded fresh (compile = load seconds of each), 5 warm-up calls, 20 timed calls
(median, p10, p90 of the wall time of host.system_one / the 64 calls), the split (tower / decide / host) of the median
call, the bucket and canvas used, and the process's phys_footprint (now and lifetime max, proc_pid_rusage) after the
timed calls. Graphs are closed between workloads.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(WORK, "host"))
sys.path.insert(0, HERE)
SNAP = os.path.join(WORK, "snapshot")
WORKLOADS = ("one_question", "3_questions", "5k_state", "384px", "box", "boxes", "64_states")


class RusageInfoV4(ctypes.Structure):
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [(n, ctypes.c_uint64) for n in (
        "ri_user_time", "ri_system_time", "ri_pkg_idle_wkups", "ri_interrupt_wkups", "ri_pageins", "ri_wired_size",
        "ri_resident_size", "ri_phys_footprint", "ri_proc_start_abstime", "ri_proc_exit_abstime",
        "ri_child_user_time", "ri_child_system_time", "ri_child_pkg_idle_wkups", "ri_child_interrupt_wkups",
        "ri_child_pageins", "ri_child_elapsed_abstime", "ri_diskio_bytesread", "ri_diskio_byteswritten",
        "ri_cpu_time_qos_default", "ri_cpu_time_qos_maintenance", "ri_cpu_time_qos_background",
        "ri_cpu_time_qos_utility", "ri_cpu_time_qos_legacy", "ri_cpu_time_qos_user_initiated",
        "ri_cpu_time_qos_user_interactive", "ri_billed_system_time", "ri_serviced_system_time",
        "ri_logical_writes", "ri_lifetime_max_phys_footprint", "ri_instructions", "ri_cycles", "ri_billed_energy",
        "ri_serviced_energy", "ri_interval_max_phys_footprint", "ri_runnable_time")]


def footprint() -> dict:
    lib = ctypes.CDLL("/usr/lib/libproc.dylib")
    info = RusageInfoV4()
    rc = lib.proc_pid_rusage(os.getpid(), 4, ctypes.byref(info))
    if rc != 0:
        return {"error": rc}
    return {"phys_footprint_gb": round(info.ri_phys_footprint / 2 ** 30, 2),
            "lifetime_max_phys_footprint_gb": round(info.ri_lifetime_max_phys_footprint / 2 ** 30, 2),
            "resident_gb": round(info.ri_resident_size / 2 ** 30, 2)}


def load_records():
    d = json.load(open(os.path.join(WORK, "set/requests.json")))
    return {r["id"]: r for r in (d["records"] if isinstance(d, dict) else d)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--accel", choices=("cpu", "gpu"), required=True)
    ap.add_argument("--precision", choices=("fp32", "default"), default="fp32")
    ap.add_argument("--form", default="fp16fc")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--warm", type=int, default=5)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--workloads", default=",".join(WORKLOADS))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import numpy as np
    from PIL import Image

    from d1_host import D1Host
    from lrt_backend import LiteRTDecide, LiteRTTower, graph_paths

    recs = load_records()
    media = json.load(open(os.path.join(WORK, "set/media.json")))
    dec_paths, tow_paths = graph_paths(os.path.join(WORK, "out/graphs"), a.form)

    def request(rid):
        req = recs[rid]["request"]
        items = []
        for mid in req.get("media", []) or []:
            p = os.path.join(WORK, media[mid]["path"])
            assert hashlib.sha256(open(p, "rb").read()).hexdigest() == media[mid]["sha256"]
            items.append(Image.open(p))
        return req.get("state"), req["questions"], items

    work = {"one_question": ["wl_one_question"], "3_questions": ["wl_3_questions"], "5k_state": ["wl_5k_state"],
            "384px": ["wl_384px"], "box": ["wl_box"], "boxes": ["wl_boxes"],
            "64_states": [f"wl_64_states_{i:02d}" for i in range(64)]}
    doc = {"accelerator": a.accel, "precision": a.precision, "form": a.form, "threads": a.threads,
           "warm": a.warm, "reps": a.reps, "host": "host/d1_host.py", "workloads": {},
           "start_footprint": footprint(), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    for wl in a.workloads.split(","):
        decide = LiteRTDecide(dec_paths, a.accel, a.precision, a.threads)
        tower = LiteRTTower(tow_paths, a.accel, a.precision, a.threads)
        host = D1Host(SNAP, decide, tower, embed_table=os.path.join(WORK, "tables/embed_table.npy"))
        reqs = [request(rid) for rid in work[wl]]
        # first call: loads (compiles) the graphs it needs
        t0 = time.perf_counter()
        for st, qs, items in reqs:
            first = host.system_one(st, qs, items)
        first_s = time.perf_counter() - t0
        compile_s = {f"decide_L{L}": round(g.load_s, 2) for L, g in decide.graphs.items()}
        compile_s.update({f"tower_G{c[0]}x{c[1]}": round(g.load_s, 2) for c, g in tower.graphs.items()})
        for _ in range(a.warm):
            for st, qs, items in reqs:
                host.system_one(st, qs, items)
        times, splits = [], []
        for _ in range(a.reps):
            t1 = time.perf_counter()
            agg = {"tower_s": 0.0, "decide_s": 0.0, "host_s": 0.0, "tokens": 0, "decide_calls": 0}
            for st, qs, items in reqs:
                host.system_one(st, qs, items)
                tm = host.last["timing"]
                for k in agg:
                    agg[k] += tm.get(k, 0)
            times.append(time.perf_counter() - t1)
            splits.append(agg)
        order = np.argsort(times)
        mid = splits[int(order[len(order) // 2])]
        row = {"records": len(reqs), "first_call_s": round(first_s, 3), "compile_s": compile_s,
               "median_s": round(statistics.median(times), 4), "p10_s": round(float(np.percentile(times, 10)), 4),
               "p90_s": round(float(np.percentile(times, 90)), 4), "min_s": round(min(times), 4),
               "median_split": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in mid.items()},
               "buckets": sorted(decide.graphs), "canvases": [list(c) for c in tower.graphs],
               "footprint": footprint(), "answers_first": json.loads(json.dumps(first["answers"], default=str))}
        doc["workloads"][wl] = row
        print(json.dumps({"workload": wl, **{k: row[k] for k in ("median_s", "p10_s", "p90_s", "compile_s",
                                                                  "buckets", "canvases", "footprint")}}), flush=True)
        decide.close()
        tower.close()
        del host
        json.dump(doc, open(os.path.join(WORK, a.out), "w"), indent=1)
    doc["torch_imported"] = "torch" in sys.modules
    json.dump(doc, open(os.path.join(WORK, a.out), "w"), indent=1)
    print("DONE torch_imported", doc["torch_imported"])


if __name__ == "__main__":
    main()
