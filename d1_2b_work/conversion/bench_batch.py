"""Speed rows of the batch graphs and of other decide files, bench_host.py's protocol (the graphs loaded fresh per
workload = load seconds, 5 warm-up + 20 timed, median / p10 / p90 / min, the split of the median, phys_footprint now and
lifetime max), one arm per process, run alone on the machine:

  --mode quant --decide <L256 file>     workloads "One question (L256 bucket)" and "3 questions (L256 bucket)":
                                        wl_one_question / wl_3_questions with the decide backend = that one file
  --mode batch --arm single|b8|b64      workload "64 states in one call": the 64 requests wl_64_states_00..63 in ONE
                                        host.system_one_batch (host/d1_batch.D1HostBatched); single = the L128 file
                                        one sequence a call (64 calls), b8 / b64 = the batch graph (8 / 1 calls)

  $EXPORT conversion/bench_batch.py --mode batch --arm b8 --accel gpu --out results/bench_batch_b8_gpu32.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(WORK, "host"))
sys.path.insert(2, HERE)
SNAP = os.path.join(WORK, "snapshot")
SINGLE128 = "out/graphs/d1-2b_decide_L128_fp16fc.tflite"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("quant", "batch"), required=True)
    ap.add_argument("--decide", default="")
    ap.add_argument("--arm", choices=("single", "b8", "b64"), default="single")
    ap.add_argument("--accel", choices=("cpu", "gpu"), required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--warm", type=int, default=5)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import numpy as np

    from d1_batch import D1HostBatched, LiteRTDecideBatch
    from fp import footprint
    from lrt_backend import LiteRTDecide

    reqs = {r["id"]: r["request"] for r in json.load(open(os.path.join(WORK, "set/requests.json")))["records"]}
    table = os.path.join(WORK, "tables/embed_table.npy")
    if a.mode == "quant":
        work = {"one_question (L256 bucket)": ["wl_one_question"], "3_questions (L256 bucket)": ["wl_3_questions"]}
    else:
        work = {"64_states_in_one_call": [f"wl_64_states_{i:02d}" for i in range(64)]}
    doc = {"mode": a.mode, "arm": a.arm if a.mode == "batch" else None, "decide": a.decide or None,
           "accelerator": a.accel, "precision": "fp32", "threads": a.threads, "warm": a.warm, "reps": a.reps,
           "host": "host/d1_batch.py (D1HostBatched; D1Host's code line for line outside the batch call)",
           "start_footprint": footprint(), "time": time.strftime("%Y-%m-%d %H:%M:%S"), "workloads": {}}
    out = os.path.join(WORK, a.out)
    for wl, rids in work.items():
        t0 = time.perf_counter()   # first call = load (the single-file backend loads on first use) + the first answer
        if a.mode == "quant":
            decide = LiteRTDecide({256: os.path.join(WORK, a.decide)}, a.accel, "fp32", a.threads)
        elif a.arm == "single":
            decide = LiteRTDecide({128: os.path.join(WORK, SINGLE128)}, a.accel, "fp32", a.threads)
        else:
            B = int(a.arm[1:])
            decide = LiteRTDecideBatch(os.path.join(WORK, f"out/graphs/d1-2b_decide_b{B}_L128_fp16fc.tflite"),
                                       a.accel, "fp32", a.threads)
        host = D1HostBatched(SNAP, decide, None, embed_table=table)
        batch = [(reqs[r]["state"], reqs[r]["questions"]) for r in rids]
        first = host.system_one_batch(batch)
        first_s = time.perf_counter() - t0
        if hasattr(decide, "graphs"):
            load = {f"decide_L{L}": round(g.load_s, 2) for L, g in decide.graphs.items()}
        else:
            load = {os.path.basename(decide.path): round(decide.load_s, 2)}
        for _ in range(a.warm):
            host.system_one_batch(batch)
        times, splits = [], []
        calls0 = len(decide.calls) if hasattr(decide, "calls") else None   # the batch backend logs each graph call
        for _ in range(a.reps):
            t1 = time.perf_counter()
            host.system_one_batch(batch)
            times.append(time.perf_counter() - t1)
            tm = host.last["timing"]
            splits.append({k: tm.get(k, 0) for k in ("decide_s", "host_s", "prep_s", "readout_s", "decide_calls",
                                                      "tokens")})
        order = np.argsort(times)
        mid = splits[int(order[len(order) // 2])]
        row = {"records": len(rids), "first_call_s": round(first_s, 3), "load_s": load,
               "median_ms": round(statistics.median(times) * 1e3, 2),
               "p10_ms": round(float(np.percentile(times, 10)) * 1e3, 2),
               "p90_ms": round(float(np.percentile(times, 90)) * 1e3, 2), "min_ms": round(min(times) * 1e3, 2),
               "median_split_ms": {(k[:-2] + "_ms" if k.endswith("_s") else k): (round(v * 1e3, 2) if k.endswith("_s") else v)
                                   for k, v in mid.items()},
               # graph calls per request: the batch backend's own log; the single-file backend makes one call a sequence
               # (the host's decide_calls counts sequences on both)
               "graph_calls_per_request": ((len(decide.calls) - calls0) / a.reps if calls0 is not None
                                           else mid.get("decide_calls")),
               "rows_per_call_last": ([c["rows"] for c in decide.calls[-int((len(decide.calls) - calls0) / a.reps):]]
                                      if calls0 is not None and len(decide.calls) > calls0 else None),
               "footprint": footprint(), "fully_accelerated": getattr(decide, "fully_accelerated", None),
               "answers_first": json.loads(json.dumps([r["answers"] for r in first], default=str))[:4]}
        if hasattr(decide, "graphs"):
            row["fully_accelerated"] = {L: g.fully_accelerated for L, g in decide.graphs.items()}
        doc["workloads"][wl] = row
        print(json.dumps({"workload": wl, **{k: row[k] for k in ("median_ms", "p10_ms", "p90_ms", "load_s",
                                                                  "median_split_ms", "graph_calls_per_request", "footprint")}}),
              flush=True)
        decide.close()
        del host
        json.dump(doc, open(out, "w"), indent=1)
    doc["torch_imported"] = "torch" in sys.modules
    json.dump(doc, open(out, "w"), indent=1)
    print("DONE torch_imported", doc["torch_imported"])


if __name__ == "__main__":
    main()
