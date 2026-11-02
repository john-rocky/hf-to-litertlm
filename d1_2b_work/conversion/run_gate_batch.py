"""The gate of the batch path: the records of --ids-file through the batch host (host/d1_batch.D1HostBatched) on one
batch graph (or, with --single, the single L128 file one sequence a call through the same host), up to --group requests
per `host.run` (the provider's `_batches` puts them in one batch; the batch graph takes B of its sequences a call,
padding rows for the rest), written in run_gate.py's candidate form for score.py: per record `answers`, `probs`,
`usage`, `seqs`, `graphs` (the group's calls), `timing` (the group's, shared), `time_s` (the group's wall / its
records), `error`; slot hidden states in <out>_slots.npz as `<id>::seq<j>`.

  $EXPORT conversion/run_gate_batch.py --accel gpu --batch-file out/graphs/d1-2b_decide_b8_L128_fp16fc.tflite \
      --ids-file conversion/ids_L128.json --files set/requests.json set/red_arms.json --out out/gate/L128_b8_gpu32.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(WORK, "host"))
sys.path.insert(1, HERE)
SNAP = os.path.join(WORK, "snapshot")


def jsonable(x):
    import numpy as np

    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.integer):
        return int(x)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--accel", choices=("cpu", "gpu"), required=True)
    ap.add_argument("--batch-file", default="")
    ap.add_argument("--single", default="", help="the single L128 file instead (one sequence a call, same host)")
    ap.add_argument("--ids-file", required=True)
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--media", default="set/media.json")
    ap.add_argument("--group", type=int, default=64)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import hashlib

    import numpy as np
    from PIL import Image

    from d1_batch import D1HostBatched, LiteRTDecideBatch
    from lrt_backend import LiteRTDecide, LiteRTTower, graph_paths

    if bool(a.batch_file) == bool(a.single):
        ap.error("exactly one of --batch-file / --single")
    decide = (LiteRTDecideBatch(os.path.join(WORK, a.batch_file), a.accel, "fp32", 8) if a.batch_file else
              LiteRTDecide({128: os.path.join(WORK, a.single)}, a.accel, "fp32", 8))
    _, tow_paths = graph_paths(os.path.join(WORK, "out/graphs"), "fp16fc")
    tower = LiteRTTower(tow_paths, a.accel, "fp32", 8)
    host = D1HostBatched(SNAP, decide, tower, embed_table=os.path.join(WORK, "tables/embed_table.npy"))
    media = json.load(open(os.path.join(WORK, a.media)))
    want = json.load(open(os.path.join(WORK, a.ids_file)))["ids"]
    recs = {}
    for p in a.files:
        doc = json.load(open(os.path.join(WORK, p)))
        for r in (doc if isinstance(doc, list) else doc["records"]):
            recs[r["id"]] = r
    ids = [i for i in sorted(want) if i in recs]

    def open_media(mids):
        items = []
        for mid in mids or []:
            m = media[mid]
            p = os.path.join(WORK, m["path"])
            if m.get("sha256") and hashlib.sha256(open(p, "rb").read()).hexdigest() != m["sha256"]:
                raise SystemExit(f"media {mid}: sha256 mismatch")
            items.append(Image.open(p))
        return items
    out_path = os.path.join(WORK, a.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    doc = {"records": {}, "groups": []}
    arrays = {}
    t_all = time.time()
    for g0 in range(0, len(ids), a.group):
        gids = ids[g0:g0 + a.group]
        reqs, named = [], []
        for rid in gids:
            req = recs[rid]["request"]
            nm = host.named(req["questions"])
            named.append(nm)
            reqs.append((req.get("state"), list(nm.values()), open_media(req.get("media"))))
        if hasattr(decide, "calls"):
            decide.calls = []
        t1 = time.time()
        done = host.run(reqs)
        dt = time.time() - t1
        seqs = host.last.get("seqs", [])
        hidden = host.last.get("slot_hidden", [])
        calls = list(getattr(decide, "calls", []))
        doc["groups"].append({"records": len(gids), "sequences": len(seqs), "graph_calls": len(calls) or len(seqs),
                              "rows_per_call": [c["rows"] for c in calls], "wall_s": round(dt, 3),
                              "timing": jsonable(host.last.get("timing"))})
        for i, rid in enumerate(gids):
            probs, read = done[i]
            state = recs[rid]["request"].get("state")
            answers = {name: host.answer(state, q, p) for (name, q), p in zip(named[i].items(), probs)}
            own = [j for j, s in enumerate(seqs) if s["request"] == i]
            row = {"answers": jsonable(answers), "probs": jsonable(probs),
                   "usage": {"input_tokens": read, "output_tokens": 0},
                   "seqs": jsonable([{k: seqs[j][k] for k in ("first_question", "ids", "state_len", "question", "ends",
                                                              "image_grid_thw")} for j in own]),
                   "graphs": {"group": len(doc["groups"]) - 1}, "timing": None, "time_s": round(dt / len(gids), 4),
                   "error": None}
            for n, j in enumerate(own):
                arrays[f"{rid}::seq{n}"] = np.asarray(hidden[j], np.float32)
            doc["records"][rid] = row
        print(f"group {len(doc['groups'])}: {len(gids)} records, {len(seqs)} sequences, "
              f"{doc['groups'][-1]['graph_calls']} calls, {dt:.2f}s", flush=True)
    doc["run"] = {"accelerator": a.accel, "precision": "fp32", "batch_file": a.batch_file or None,
                  "single": a.single or None, "group": a.group, "records": len(ids), "host": "host/d1_batch.py",
                  "loads": getattr(decide, "loads", None), "torch_imported": "torch" in sys.modules,
                  "wall_s": round(time.time() - t_all, 2)}
    json.dump(doc, open(out_path, "w"))
    np.savez(out_path.replace(".json", "_slots.npz"), **arrays)
    print("DONE", json.dumps(doc["run"]))


if __name__ == "__main__":
    main()
