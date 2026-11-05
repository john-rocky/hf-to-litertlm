"""The PyTorch-free host over a run's graph outputs (replay backends, replay.py): every run of one pass in a folder (its report DONE, its out_<tag>.f32 there) answers the host's graph
calls for the inputs it ran; rows as run_set.py writes them, hidden[pooled] beside.

  $EXPORT conversion/run_replay.py --round-dir <R> [--round-dir ...] --prefix <tag prefix> [--prefix ...] --tag <name> [--set set_device | --ids 1,2,3]

A run's rows file is read from <R>/gen/, else from out/rows. A call whose input bytes no run ran is the record's error (KeyError: the device never ran this input), written to
<name>.skipped.jsonl and listed, never a silent answer. Writes results/replay/<name>.jsonl, <name>.skipped.jsonl, <name>.run.json and out/replay/<name>/<id>.npy (numpy host, no model load).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WORK, "host"))
from decide_host import host as H  # noqa: E402
import replay as RP  # noqa: E402

SNAP, BENCH = os.path.join(WORK, "snapshot"), os.path.join(WORK, "bench")
ROWS = os.path.join(WORK, "out/rows")


def legs_of(round_dir, prefixes):
    dec, tow, used = [], [], []
    for name in sorted(os.listdir(round_dir)):
        if not name.endswith(".json") or not any(name.startswith(p) for p in prefixes):
            continue
        tag = name[:-5]
        try:
            rep = json.load(open(os.path.join(round_dir, name)))
        except Exception:
            continue
        if not isinstance(rep, dict) or rep.get("status") != "DONE" or rep.get("mode", "generic") != "generic":
            continue
        out = os.path.join(round_dir, f"out_{tag}.f32")
        if not os.path.exists(out):
            continue
        rows = rep["rows_file"]
        rp = os.path.join(round_dir, "gen", rows)
        rp = rp if os.path.exists(rp) else os.path.join(ROWS, rows)
        n = (rep.get("summary") or {}).get("count") or 0
        src = RP.GenericRows(rp, out, rows_limit=n)
        (dec if src.doc.get("graph_kind") == "decide" else tow).append(src)
        used.append({"tag": tag, "rows": rows, "rows_path": os.path.relpath(rp, WORK), "out": os.path.relpath(out, WORK), "rows_ran": n, "rows_in_file": len(src.rows),
                     "graph": rep.get("graph"), "accel": rep.get("accel", rep.get("acc")), "precision": rep.get("precision"), "signature": rep.get("signature"),
                     "sharing": rep.get("gpu_constant_tensor_sharing", rep.get("sharing"))})
    return dec, tow, used


def label_int(x):
    """score.py label_to_int; an unknown label stays None (run_set.py)."""
    return {"1": 1, "pos": 1, "positive": 1, "0": 0, "neg": 0, "negative": 0}.get(str(x).strip().lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round-dir", required=True, action="append")
    ap.add_argument("--prefix", required=True, action="append")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--set", default="set_device")
    ap.add_argument("--ids", default="")
    a = ap.parse_args()
    dec_src, tow_src, used = [], [], []
    for rd in a.round_dir:
        d, t, u = legs_of(rd, a.prefix)
        dec_src += d
        tow_src += t
        used += u
    decide, tower = RP.ReplayDecide(dec_src), RP.ReplayTower(tow_src)
    host = H.Host(tower, decide, SNAP)
    rows_all = {r["item_id"]: r for r in map(json.loads, open(os.path.join(BENCH, "data/jevocrbench.jsonl"), encoding="utf-8"))}
    dist = {r["item_id"]: r for r in map(json.loads, open(os.path.join(WORK, "conversion/sets/record_sizes.jsonl")))}
    ids = [int(x) for x in a.ids.split(",")] if a.ids else json.load(open(os.path.join(WORK, f"conversion/sets/{a.set}.json")))["item_ids"]
    res = os.path.join(WORK, "results/replay")
    hdir = os.path.join(WORK, "out/replay", a.tag)
    os.makedirs(res, exist_ok=True)
    os.makedirs(hdir, exist_ok=True)
    keep = ("prob_1", "logits", "prob_1_prev", "logits_prev", "T", "N", "grids", "pooled_id", "ids_sha")
    t0 = time.time()
    n_ok, errs = 0, {}
    with open(os.path.join(res, f"{a.tag}.jsonl"), "w") as fo, open(os.path.join(res, f"{a.tag}.skipped.jsonl"), "w") as fs:
        for i in ids:
            r = rows_all[i]
            try:
                out = host.score(os.path.join(BENCH, r["image_path"]), os.path.join(BENCH, r["render_image_path"]), str(r["ocr_text"] or ""))
            except KeyError as e:
                errs[i] = str(e)
                fs.write(json.dumps({"item_id": i, "category": r["category"], "status": "not_run_on_this_device", "why": str(e)[:160]}) + "\n")
                continue
            if out["status"] != "ok":
                errs[i] = out["status"]
                fs.write(json.dumps({"item_id": i, "category": r["category"], **{k: v for k, v in out.items() if k != "hidden_pooled"}}) + "\n")
                continue
            hid = out.pop("hidden_pooled")
            np.save(os.path.join(hdir, f"{i}.npy"), hid)
            row = {k: out[k] for k in keep}
            row.update(item_id=i, category=r["category"], label=label_int(r["label"]), family=dist[i]["family"], lang=dist[i]["lang"], arm="base",
                       hidden_sha=hashlib.sha256(hid.tobytes()).hexdigest()[:16], L=decide.last.get("L"), decide_rows=decide.last.get("rows"))
            fo.write(json.dumps(row) + "\n")
            n_ok += 1
    doc = {"tag": a.tag, "round_dirs": [os.path.relpath(r, WORK) for r in a.round_dir], "prefixes": a.prefix, "legs": used, "records": len(ids), "answered": n_ok, "errors": errs,
           "rows_used": {"decide": len(set(decide.used)), "tower": len(set(tower.used))}, "torch_imported": "torch" in sys.modules, "wall_s": round(time.time() - t0, 1)}
    with open(os.path.join(res, f"{a.tag}.run.json"), "w") as fh:
        json.dump(doc, fh, indent=1)
    print("DONE", json.dumps({k: doc[k] for k in ("tag", "records", "answered", "rows_used", "torch_imported", "wall_s")}), "errors", len(errs), sorted(errs)[:12])


if __name__ == "__main__":
    main()
