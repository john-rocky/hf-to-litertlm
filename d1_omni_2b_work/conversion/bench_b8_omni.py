"""Speed rows of d1-Omni-2B's eight-signature file and B = 8 single file (set before any number): the card's
d1-Omni-2B workloads, bench_host.py's protocol (per workload the graphs loaded fresh = load seconds, 5 warm-up + 20
timed requests, median / p10 / p90 / min of the wall, the split of the median request, phys_footprint now and lifetime
max), one arm per process, run alone on the machine:

  --arm m8        decide = host/multi8_omni.Multi8Decide over ONE CompiledModel of the Omni multi8 file: Metal
                  float32 + GpuOptions.constant_tensor_sharing (--accel gpu) or XNNPACK 8 threads + the XNNPACK weight
                  cache file (--accel cpu --weight-cache <file>); routing (set before any number): a provider batch of
                  >= 8 sequences that all fit 128 tokens -> decide_b8_L128 eight a call, else the smallest bucket
  --arm b8single  decide = host/b8_single_omni.B8SingleDecide over the Omni B = 8 single file (no sharing, no weight
                  cache = the d1-2B single-file arm); the 64-states workload only
  tower / audio = the single files (lrt_backend.LiteRTTower / LiteRTAudio, no sharing: the phone's form)
  host = host/d1_batch_omni.D1HostBatchedOmni (the Omni host line for line outside the decide call)
  workloads: one_question, 3_questions, 5k_state, 384px, audio_10s = one host.system_one a request (the time includes
                  the host's mel features for the clip); 64_states = the 64 requests in ONE host.system_one_batch (one
                  provider batch of 64 sequences -> 8 calls of decide_b8_L128)

  $EXPORT conversion/bench_b8_omni.py --arm b8single --accel gpu --out results/bench_r8_b8single_gpu32.json
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WORKDIR = os.path.normpath(os.path.join(HERE, ".."))
SNAP = os.path.join(WORKDIR, "snapshot")
TABLE = os.path.join(WORKDIR, "tables/embed_table.npy")
M8 = "out/graphs/d1-omni-2b_decide_multi8_fp16fc.tflite"
B8 = "out/graphs/d1-omni-2b_decide_b8_L128_fp16fc.tflite"
WORKLOADS = ("one_question", "3_questions", "5k_state", "384px", "audio_10s", "64_states")
WORK = {"one_question": ["wl_one_question"], "3_questions": ["wl_3_questions"], "5k_state": ["wl_5k_state"],
        "384px": ["wl_384px"], "audio_10s": ["wl_audio_10s"],
        "64_states": [f"wl_64_states_{i:02d}" for i in range(64)]}           # conversion/bench_host.py's WORK
FILES = ["set/requests.json", "set/requests_audio.json"]
MEDIA = ["set/media.json", "set/media_audio.json"]                  # conversion/bench_host.py's FILES / MEDIA


def load_by_path(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("m8", "b8single"), required=True)
    ap.add_argument("--accel", choices=("cpu", "gpu"), required=True)
    ap.add_argument("--multi8", default=M8)
    ap.add_argument("--b8", default=B8)
    ap.add_argument("--weight-cache", default="", help="CPU m8: CpuOptions.xnnpack_weight_cache_path (a file)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--warm", type=int, default=5)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--workloads", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import numpy as np

    hb = load_by_path("d1_batch_omni", os.path.join(WORKDIR, "host/d1_batch_omni.py"))
    m8 = load_by_path("multi8_omni", os.path.join(WORKDIR, "host/multi8_omni.py"))
    bs = load_by_path("b8_single_omni", os.path.join(WORKDIR, "host/b8_single_omni.py"))
    import lrt_backend as _b
    from lrt_backend import LiteRTAudio, LiteRTTower, audio_paths, graph_paths
    from records import load_records, open_request
    assert os.path.samefile(_b.__file__, os.path.join(WORKDIR, "host/lrt_backend.py")), _b.__file__
    footprint = load_by_path("r4_fp", os.path.join(WORKDIR, "conversion/fp.py")).footprint

    recs, media = load_records(FILES, MEDIA)
    recs = {r["id"]: r for r in recs}
    gdir = os.path.join(WORKDIR, "out/graphs")
    _, tow_paths = graph_paths(gdir, "fp16fc")
    aud_paths = audio_paths(gdir, "fp16fc")
    gpu = a.accel == "gpu"
    wc = os.path.join(WORKDIR, a.weight_cache) if a.weight_cache else ""
    if a.arm == "m8" and not gpu and not (wc and os.path.exists(wc)):
        raise SystemExit("the CPU m8 arm reads the multi8 weight cache written by conversion/multi8_check_omni.py")
    workloads = a.workloads.split(",") if a.workloads else (list(WORKLOADS) if a.arm == "m8" else ["64_states"])
    doc = {"model": "LiquidAI/d1-Omni-2B@431f1d66", "arm": a.arm, "accelerator": a.accel, "precision": "fp32",
           "form": ("multi8 (" + a.multi8 + ")") if a.arm == "m8" else ("B=8 single file (" + a.b8 + ")"),
           "options": (("constant_tensor_sharing" if gpu else f"xnnpack_weight_cache_path {a.weight_cache}")
                       if a.arm == "m8" else "none (no sharing, no weight cache)"),
           "threads": a.threads, "warm": a.warm, "reps": a.reps,
           "host": "host/d1_batch_omni.py (D1HostBatchedOmni on host/d1_host.py)",
           "backend": "host/multi8_omni.py (batch_min 8)" if a.arm == "m8" else "host/b8_single_omni.py",
           "tower_audio": "the single files (7 canvases, 4 audio buckets), no sharing", "workloads": {},
           "start_footprint": footprint(), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    out = os.path.join(WORKDIR, a.out)
    for wl in workloads:
        t0 = time.perf_counter()
        if a.arm == "m8":
            decide = m8.Multi8Decide(os.path.join(WORKDIR, a.multi8), a.accel, "fp32", a.threads, sharing=gpu,
                                     weight_cache=wc, batch_min=8)
        else:
            assert wl == "64_states", "the B=8 single file answers the 64-states workload only (<= 128 tokens)"
            decide = bs.B8SingleDecide(os.path.join(WORKDIR, a.b8), a.accel, "fp32", a.threads)
        tower = LiteRTTower(tow_paths, a.accel, "fp32", a.threads)
        audio = LiteRTAudio(aud_paths, a.accel, "fp32", a.threads)
        host = hb.D1HostBatchedOmni(SNAP, decide, tower, embed_table=TABLE, audio=audio)
        reqs = [open_request(recs[rid], media) for rid in WORK[wl]]
        batch = wl == "64_states"
        if batch:
            assert all(not items for _, _, items in reqs)

        def once():
            if batch:
                return host.system_one_batch([(st, qs) for st, qs, _ in reqs])
            return [host.system_one(st, qs, items) for st, qs, items in reqs]
        first = once()
        first_s = time.perf_counter() - t0
        load = {("multi8" if a.arm == "m8" else "b8single"): round(decide.load_s, 2)}
        load.update({f"tower_G{c[0]}x{c[1]}": round(g.load_s, 2) for c, g in tower.graphs.items()})
        load.update({f"audio_T{x['T']}": x["load_s"] for x in audio.loads})
        for _ in range(a.warm):
            once()
        times, splits, calls = [], [], []
        for _ in range(a.reps):
            n0 = len(decide.calls)
            t1 = time.perf_counter()
            once()
            times.append(time.perf_counter() - t1)
            tm = host.last["timing"]
            splits.append({k: tm.get(k, 0) for k in ("prep_s", "feature_s", "tower_s", "audio_s", "decide_s",
                                                      "readout_s", "host_s", "decide_calls", "tower_calls",
                                                      "audio_calls", "tokens")})
            calls.append(decide.calls[n0:])
        order = np.argsort(times)
        mi = int(order[len(order) // 2])
        mid = splits[mi]
        row = {"records": len(reqs), "first_call_s": round(first_s, 3), "load_s": load,
               "median_ms": round(statistics.median(times) * 1e3, 2),
               "p10_ms": round(float(np.percentile(times, 10)) * 1e3, 2),
               "p90_ms": round(float(np.percentile(times, 90)) * 1e3, 2), "min_ms": round(min(times) * 1e3, 2),
               "median_split_ms": {(k[:-2] + "_ms" if k.endswith("_s") else k): (round(v * 1e3, 2) if k.endswith("_s") else v)
                                   for k, v in mid.items()},
               "graph_calls_per_request": len(calls[mi]),
               "calls_by_signature": dict(collections.Counter(c["sig"] for c in calls[mi])),
               "rows_per_call": [c["rows"] for c in calls[mi]],
               "canvases": [list(c) for c in tower.graphs], "audio_buckets": sorted(audio.graphs),
               "footprint": footprint(), "fully_accelerated": decide.fully_accelerated,
               "answers_first": json.loads(json.dumps([r["answers"] for r in first], default=str))[:4]}
        doc["workloads"][wl] = row
        print(json.dumps({"workload": wl, **{k: row[k] for k in ("median_ms", "p10_ms", "p90_ms", "load_s",
                                                                  "median_split_ms", "calls_by_signature", "footprint")}}),
              flush=True)
        decide.close()
        tower.close()
        audio.close()
        del host
        json.dump(doc, open(out, "w"), indent=1)
    doc["torch_imported"] = "torch" in sys.modules
    json.dump(doc, open(out, "w"), indent=1)
    print("DONE torch_imported", doc["torch_imported"])


if __name__ == "__main__":
    main()
