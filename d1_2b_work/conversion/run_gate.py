"""Run the question set through the PyTorch-free host (host/d1_host.py) on the LiteRT graphs of one form, on one
accelerator, and write the candidate in the reference's form (conversion/ref_run.py), for conversion/score.py.

  $EXPORT conversion/run_gate.py --accel gpu --precision fp32 --form fp16fc \
      --files set/requests.json set/red_arms.json --media set/media.json --out out/gate/gpu_fp32.json

`--multi out/graphs/d1-2b_decide_multi7_fp16fc.tflite` runs every bucket on the seven-signature file (one
CompiledModel, conversion/lrt_multi.py); `--sharing` turns GpuOptions.constant_tensor_sharing on (with --multi).

Per record: `answers` (host.answer on host.run's values, as ref_run does it), `probs` (the input of api.answer),
`usage`, `seqs` (ids, state_len, question tags, slot positions, picture grids), `graphs` (the bucket / canvas each
sequence / picture ran on), `timing` (host.last), `time_s`, `error`. The slot hidden states go to
<out>_slots.npz as `<id>::seq<j>` (sequence j in the host's order = the provider's `_sequences` order). Resumes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(WORK, "host"))
sys.path.insert(0, HERE)
SNAP = os.path.join(WORK, "snapshot")


def jsonable(x):
    import numpy as np

    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def load_records(paths):
    recs = []
    for p in paths:
        doc = json.load(open(os.path.join(WORK, p)))
        recs.extend(doc if isinstance(doc, list) else doc["records"])
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--accel", choices=("cpu", "gpu"), required=True)
    ap.add_argument("--precision", choices=("fp32", "default"), default="fp32")
    ap.add_argument("--form", default="fp16fc")
    ap.add_argument("--graphs", default="out/graphs")
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--media", default="set/media.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--ids", default=None)
    ap.add_argument("--min-bucket", type=int, default=0, help="use only decision graphs of L >= this")
    ap.add_argument("--multi", default="", help="the seven-signature decide file (one CompiledModel, all buckets)")
    ap.add_argument("--sharing", action="store_true", help="GpuOptions.constant_tensor_sharing (with --multi)")
    a = ap.parse_args()
    if a.sharing and not a.multi:
        ap.error("--sharing is implemented for --multi runs only")
    import numpy as np
    from PIL import Image

    from d1_host import D1Host
    from lrt_backend import LiteRTDecide, LiteRTTower, graph_paths

    dec_paths, tow_paths = graph_paths(os.path.join(WORK, a.graphs), a.form)
    dec_paths = {L: p for L, p in dec_paths.items() if L >= a.min_bucket}
    if a.multi:
        from lrt_multi import LiteRTMultiDecide, SharedTower

        decide = LiteRTMultiDecide(os.path.join(WORK, a.multi), a.accel, a.precision, a.threads, sharing=a.sharing)
        dec_paths = dict(decide.paths)
        tower = ((SharedTower(tow_paths, a.accel, a.precision, a.threads) if a.sharing else
                  LiteRTTower(tow_paths, a.accel, a.precision, a.threads)) if tow_paths else None)
    else:
        decide = LiteRTDecide(dec_paths, a.accel, a.precision, a.threads)
        tower = LiteRTTower(tow_paths, a.accel, a.precision, a.threads) if tow_paths else None
    host = D1Host(SNAP, decide, tower, embed_table=os.path.join(WORK, "tables/embed_table.npy"))
    media = json.load(open(os.path.join(WORK, a.media))) if a.media else {}
    recs = load_records(a.files)
    if a.ids:
        want = set(a.ids.split(","))
        recs = [r for r in recs if r["id"] in want]
    # one load per bucket: records in the order of the bucket of their longest sequence (the set's token counts)
    tc = json.load(open(os.path.join(WORK, "set/token_counts.json")))
    longest = {**{k: v.get("longest") or 0 for k, v in tc["records"].items()},
               **{k: v.get("longest") or 0 for k, v in tc.get("arms", {}).items()}}

    def bucket_of(r):
        n = longest.get(r["id"], 0)
        return next((L for L in sorted(dec_paths) if L >= n), 10 ** 9)
    recs.sort(key=lambda r: (bucket_of(r), r["id"]))
    out_path = os.path.join(WORK, a.out)
    hid_path = out_path.replace(".json", "_slots.npz")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    doc = json.load(open(out_path)) if os.path.exists(out_path) else {"records": {}}
    arrays = dict(np.load(hid_path)) if os.path.exists(hid_path) else {}
    doc["run"] = {"accelerator": a.accel, "precision": a.precision, "form": a.form, "threads": a.threads,
                  "decide_graphs": {str(L): os.path.relpath(p, WORK) for L, p in sorted(dec_paths.items())},
                  "tower_graphs": {f"{c[0]}x{c[1]}": os.path.relpath(p, WORK) for c, p in sorted(tow_paths.items())},
                  "host": "host/d1_host.py", "torch_imported": "torch" in sys.modules,
                  **({"multi": {"file": a.multi, "signatures": {str(L): s for L, s in sorted(decide.sigs.items())},
                                "sharing": a.sharing, "loads": decide.loads}} if a.multi else {})}

    def open_media(ids):
        items = []
        for mid in ids or []:
            m = media[mid]
            p = os.path.join(WORK, m["path"])
            data = open(p, "rb").read()
            if m.get("sha256") and hashlib.sha256(data).hexdigest() != m["sha256"]:
                raise SystemExit(f"media {mid}: sha256 mismatch")
            items.append(Image.open(p))
        return items

    calls = {"decide": [], "tower": []}
    dec_call, tow_call = decide.__call__, (tower.__call__ if tower else None)

    def decide_logged(embeds, state_len, question):
        h = dec_call(embeds, state_len, question)
        calls["decide"].append(dict(decide.last))
        return h

    def tower_logged(pv, grid):
        f = tow_call(pv, grid)
        calls["tower"].append({k: list(v) if isinstance(v, tuple) else v for k, v in tower.last.items()})
        return f
    host.decide = decide_logged
    host.decide.max_len = decide.max_len if hasattr(decide, "max_len") else None
    if tower:
        host.tower = tower_logged
    t0 = time.time()
    for n, rec in enumerate(recs):
        rid = rec["id"]
        if rid in doc["records"]:
            continue
        req = rec["request"]
        state, questions = req.get("state"), req["questions"]
        calls["decide"].clear()
        calls["tower"].clear()
        t1 = time.time()
        try:
            items = open_media(req.get("media"))
            named = host.named(questions)
            done = host.run([(state, list(named.values()), items)])
            probs, read = done[0]
            answers = {name: host.answer(state, q, p) for (name, q), p in zip(named.items(), probs)}
            err = None
        except Exception as e:  # a refusal is a result too
            probs, read, answers, err = None, None, None, f"{type(e).__name__}: {e}"
        dt = time.time() - t1
        seqs = [{k: s[k] for k in ("first_question", "ids", "state_len", "question", "ends", "image_grid_thw")}
                for s in host.last.get("seqs", [])]
        row = {"answers": jsonable(answers), "probs": jsonable(probs), "usage": {"input_tokens": read,
                                                                                  "output_tokens": 0},
               "seqs": jsonable(seqs), "graphs": {"decide": jsonable(calls["decide"]), "tower": jsonable(calls["tower"])},
               "timing": jsonable(host.last.get("timing")), "time_s": round(dt, 4), "error": err}
        if err is None:
            for j, h in enumerate(host.last.get("slot_hidden", [])):
                arrays[f"{rid}::seq{j}"] = np.asarray(h, np.float32)
        doc["records"][rid] = row
        if n % 20 == 0 or n == len(recs) - 1:
            json.dump(doc, open(out_path + ".tmp", "w"))
            os.replace(out_path + ".tmp", out_path)
            np.savez(hid_path + ".tmp.npz", **arrays)
            os.replace(hid_path + ".tmp.npz", hid_path)
        print(f"{rid} {dt:.2f}s tokens={read} err={err}", flush=True)
    doc["wall_s"] = round(time.time() - t0, 2)
    doc["run"]["torch_imported"] = "torch" in sys.modules
    doc["run"]["decide_loads"] = decide.loads
    doc["run"]["tower_loads"] = {f"{c[0]}x{c[1]}": {"load_s": round(g.load_s, 2), "fully_accelerated": g.fully_accelerated}
                                 for c, g in (tower.graphs.items() if tower else [])}
    json.dump(doc, open(out_path + ".tmp", "w"))
    os.replace(out_path + ".tmp", out_path)
    np.savez(hid_path + ".tmp.npz", **arrays)
    os.replace(hid_path + ".tmp.npz", hid_path)
    print("DONE", json.dumps(doc["run"]))


if __name__ == "__main__":
    main()
