"""Speed of the PyTorch-free host on the LiteRT graphs of d1-Omni-2B, the card's Omni workloads (set before any number):
One question, 3 questions, 5K-token state, 384 px picture, 10 s audio clip (state None + 1 noul; the time includes the
host's mel features), 64 states in one call (64 requests, the host calls the graph once per request). The d1-2B
timing script with the Omni graphs, the audio backend and the set's loader (records.py). Run alone on the machine
(the conversion ran each timing process in a measurement window that no other heavy job shared):

  $EXPORT conversion/bench_host.py --accel gpu --precision fp32 --out results/bench_host_gpu_fp32.json

Per workload: the graphs it needs are loaded fresh (compile = load seconds of each), 5 warm-up calls, 20 timed calls
(median, p10, p90 of the wall time of host.system_one / the 64 calls), the split (every timing key the host reports)
of the median call, the buckets, canvases and audio buckets used, and the process's phys_footprint (now and lifetime
max, proc_pid_rusage) after the timed calls. Graphs are closed between workloads. `--multi` = the seven-signature
decide file with GpuOptions.constant_tensor_sharing on the GPU (lrt_multi.py).
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WORKDIR = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WORKDIR, "host"))
SNAP = os.path.join(WORKDIR, "snapshot")
TABLE = os.path.join(WORKDIR, "tables/embed_table.npy")
WORKLOADS = ("one_question", "3_questions", "5k_state", "384px", "audio_10s", "64_states")
WORK = {"one_question": ["wl_one_question"], "3_questions": ["wl_3_questions"], "5k_state": ["wl_5k_state"],
        "384px": ["wl_384px"], "audio_10s": ["wl_audio_10s"],
        "64_states": [f"wl_64_states_{i:02d}" for i in range(64)]}
FILES = ["set/requests.json", "set/requests_audio.json"]
MEDIA = ["set/media.json", "set/media_audio.json"]


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--accel", choices=("cpu", "gpu"), required=True)
    ap.add_argument("--precision", choices=("fp32", "default"), default="fp32")
    ap.add_argument("--form", default="fp16fc")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--warm", type=int, default=5)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--workloads", default=",".join(WORKLOADS))
    ap.add_argument("--multi", default="", help="the seven-signature decide file (sharing on the GPU)")
    ap.add_argument("--weight-cache", default="", help="CpuOptions.xnnpack_weight_cache_path for --multi on the CPU")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import numpy as np

    from d1_host import D1Host
    from lrt_backend import LiteRTAudio, LiteRTDecide, LiteRTTower, audio_paths, graph_paths
    from records import load_records, open_request

    recs, media = load_records(FILES, MEDIA)
    recs = {r["id"]: r for r in recs}
    gdir = os.path.join(WORKDIR, "out/graphs")
    dec_paths, tow_paths = graph_paths(gdir, a.form)
    aud_paths = audio_paths(gdir, a.form)
    out_path = os.path.join(WORKDIR, a.out)
    prev = json.load(open(out_path)) if os.path.exists(out_path) else None  # a later step adds its workloads
    doc = {"accelerator": a.accel, "precision": a.precision, "form": a.form, "threads": a.threads,
           "multi": a.multi or None, "weight_cache": a.weight_cache or None, "warm": a.warm, "reps": a.reps,
           "host": "host/d1_host.py", "workloads": dict(prev["workloads"]) if prev else {},
           "start_footprint": footprint(), "time": time.strftime("%Y-%m-%d %H:%M:%S"),
           **({"earlier_steps": [{k: prev[k] for k in ("time", "start_footprint") if k in prev}]} if prev else {})}
    for wl in a.workloads.split(","):
        if a.multi:
            from lrt_multi import LiteRTMultiDecide, SharedAudio, SharedTower

            share = a.accel == "gpu"
            wc = os.path.join(WORKDIR, a.weight_cache) if a.weight_cache else ""
            if wc:
                os.makedirs(os.path.dirname(wc), exist_ok=True)
            decide = LiteRTMultiDecide(os.path.join(WORKDIR, a.multi), a.accel, a.precision, a.threads, sharing=share,
                                       weight_cache=wc)
            tower = (SharedTower(tow_paths, a.accel, a.precision, a.threads) if share else
                     LiteRTTower(tow_paths, a.accel, a.precision, a.threads))
            audio = (SharedAudio(aud_paths, a.accel, a.precision, a.threads) if share else
                     LiteRTAudio(aud_paths, a.accel, a.precision, a.threads))
        else:
            decide = LiteRTDecide(dec_paths, a.accel, a.precision, a.threads)
            tower = LiteRTTower(tow_paths, a.accel, a.precision, a.threads)
            audio = LiteRTAudio(aud_paths, a.accel, a.precision, a.threads)
        host = D1Host(SNAP, decide, tower, embed_table=TABLE, audio=audio)
        reqs = [open_request(recs[rid], media) for rid in WORK[wl]]
        # first call: loads (compiles) the graphs it needs
        t0 = time.perf_counter()
        used = set()
        for st, qs, items in reqs:
            first = host.system_one(st, qs, items)
            used.add(decide.last.get("L"))
        first_s = time.perf_counter() - t0
        compile_s = {}
        if a.multi:
            compile_s.update({f"decide_multi7_{x.get('file', '')}": x["load_s"] for x in decide.loads})
        else:
            compile_s.update({f"decide_L{L}": round(g.load_s, 2) for L, g in decide.graphs.items()})
        compile_s.update({f"tower_G{c[0]}x{c[1]}": round(getattr(g, "load_s", getattr(getattr(g, "m", None),
                                                                                       "load_s", 0.0)), 2)
                          for c, g in tower.graphs.items()})
        compile_s.update({f"audio_T{x['T']}": x["load_s"] for x in audio.loads})
        for _ in range(a.warm):
            for st, qs, items in reqs:
                host.system_one(st, qs, items)
        times, splits = [], []
        for _ in range(a.reps):
            t1 = time.perf_counter()
            agg = {}
            for st, qs, items in reqs:
                host.system_one(st, qs, items)
                for k, v in (host.last.get("timing") or {}).items():
                    if isinstance(v, (int, float)):
                        agg[k] = agg.get(k, 0) + v
            times.append(time.perf_counter() - t1)
            splits.append(agg)
        order = np.argsort(times)
        mid = splits[int(order[len(order) // 2])]
        row = {"records": len(reqs), "first_call_s": round(first_s, 3), "compile_s": compile_s,
               "median_s": round(statistics.median(times), 4), "p10_s": round(float(np.percentile(times, 10)), 4),
               "p90_s": round(float(np.percentile(times, 90)), 4), "min_s": round(min(times), 4),
               "median_split": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in mid.items()},
               "buckets": sorted(decide.sigs) if a.multi else sorted(decide.graphs),
               "canvases": [list(c) for c in tower.graphs], "audio_buckets": sorted(audio.graphs),
               "footprint": footprint(), "answers_first": json.loads(json.dumps(first["answers"], default=str))}
        if a.multi:
            row["buckets_used"] = sorted(used)
        doc["workloads"][wl] = row
        print(json.dumps({"workload": wl, **{k: row[k] for k in ("median_s", "p10_s", "p90_s", "compile_s",
                                                                  "buckets", "canvases", "audio_buckets",
                                                                  "footprint")}}), flush=True)
        decide.close()
        tower.close()
        audio.close()
        del host
        json.dump(doc, open(out_path, "w"), indent=1)
    doc["torch_imported"] = "torch" in sys.modules
    json.dump(doc, open(out_path, "w"), indent=1)
    print("DONE torch_imported", doc["torch_imported"])


if __name__ == "__main__":
    main()
