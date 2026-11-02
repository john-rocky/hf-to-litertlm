"""The PyTorch-free host (host/d1_host.py) over a run's graph outputs (replay backends, replay.py): every generic leg
of one pass in a phone's pulled folder (its report DONE, its out_<tag>.f32 pulled) answers the host's graph calls for
the inputs it ran; the candidate is written in run_gate.py's form for score.py. The legs' reports and outputs are
written by the phones' test app, which is not published.

  $EXPORT conversion/run_replay.py --round-dir <R> --prefix <tag prefix> [--prefix ...] \
      --rows-dir out/rows --set conversion/set_ids.json --out out/replay/<name>.json

A leg's rows file is read from --rows-dir, or from <R>/gen/ (the prep legs' rows). A call whose inputs no leg ran is
the record's error (KeyError: the device never ran this input), never a silent answer.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WORK, "host"))
sys.path.insert(0, HERE)
sys.path.insert(0, HERE)
SNAP = os.path.join(WORK, "snapshot")


def legs_of(round_dir, rows_dir, prefixes):
    from replay import GenericRows

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
        rp = rp if os.path.exists(rp) else os.path.join(rows_dir, rows)
        n = (rep.get("summary") or {}).get("count") or 0
        src = GenericRows(rp, out, rows_limit=n)
        kind = src.doc.get("graph_kind")
        (dec if kind == "d1-2b_decide" else tow).append(src)
        used.append({"tag": tag, "rows": rows, "rows_path": os.path.relpath(rp, WORK), "out": os.path.relpath(out, WORK),
                     "rows_ran": n, "rows_in_file": len(src.rows), "graph": rep.get("graph"),
                     "accel": rep.get("accel"), "precision": rep.get("precision"), "signature": rep.get("signature")})
    return dec, tow, used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round-dir", required=True, nargs="+")
    ap.add_argument("--prefix", required=True, action="append")
    ap.add_argument("--rows-dir", default=os.path.join(WORK, "out/rows"))
    ap.add_argument("--set", default=os.path.join(WORK, "conversion/set_ids.json"))
    ap.add_argument("--ids", default=None, help="comma-separated record ids (default: the set's records + extra)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import numpy as np

    from d1_host import D1Host
    from make_rows import load_records, open_media, MEDIA
    from replay import ReplayDecide, ReplayTower
    from run_gate import jsonable

    dec_src, tow_src, used = [], [], []
    for rd in a.round_dir:
        d, t, u = legs_of(rd, a.rows_dir, a.prefix)
        dec_src += d
        tow_src += t
        used += u
    decide = ReplayDecide(dec_src)
    tower = ReplayTower(tow_src) if tow_src else None
    host = D1Host(SNAP, decide, tower, embed_table=os.path.join(WORK, "tables/embed_table.npy"))
    sel = json.load(open(a.set))
    ids = a.ids.split(",") if a.ids else sel["records"] + sel["extra_canvas"]
    recs = load_records()
    media = json.load(open(os.path.join(WORK, MEDIA)))
    doc = {"records": {}, "run": {"replay_of": [os.path.relpath(r, WORK) for r in a.round_dir], "prefixes": a.prefix,
                                  "legs": used, "host": "host/d1_host.py", "backends": "conversion/replay.py"}}
    arrays = {}
    calls = {"decide": [], "tower": []}
    dcall, tcall = decide.__call__, (tower.__call__ if tower else None)

    def decide_logged(embeds, state_len, question):
        h = dcall(embeds, state_len, question)
        calls["decide"].append(dict(decide.last))
        return h

    def tower_logged(pv, grid):
        f = tcall(pv, grid)
        calls["tower"].append({k: list(v) if isinstance(v, tuple) else v for k, v in tower.last.items()})
        return f
    host.decide = decide_logged
    host.decide.max_len = decide.max_len
    if tower:
        host.tower = tower_logged
    t0 = time.time()
    for rid in ids:
        req = recs[rid]["request"]
        calls["decide"].clear()
        calls["tower"].clear()
        t1 = time.time()
        try:
            items = open_media(media, req.get("media"))
            named = host.named(req["questions"])
            done = host.run([(req.get("state"), list(named.values()), items)])
            probs, read = done[0]
            answers = {name: host.answer(req.get("state"), q, p) for (name, q), p in zip(named.items(), probs)}
            err = None
        except Exception as e:
            probs, read, answers, err = None, None, None, f"{type(e).__name__}: {e}"
        seqs = [{k: s[k] for k in ("first_question", "ids", "state_len", "question", "ends", "image_grid_thw")}
                for s in host.last.get("seqs", [])]
        doc["records"][rid] = {"answers": jsonable(answers), "probs": jsonable(probs),
                               "usage": {"input_tokens": read, "output_tokens": 0}, "seqs": jsonable(seqs),
                               "graphs": {"decide": jsonable(calls["decide"]), "tower": jsonable(calls["tower"])},
                               "time_s": round(time.time() - t1, 4), "error": err}
        if err is None:
            for j, h in enumerate(host.last.get("slot_hidden", [])):
                arrays[f"{rid}::seq{j}"] = np.asarray(h, np.float32)
        print(f"{rid} err={err}", flush=True)
    doc["wall_s"] = round(time.time() - t0, 2)
    doc["run"]["torch_imported"] = "torch" in sys.modules
    doc["run"]["rows_used"] = {"decide": len(set(decide.used)), "tower": len(set(tower.used)) if tower else 0}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(doc, open(a.out + ".tmp", "w"))
    os.replace(a.out + ".tmp", a.out)
    hp = a.out.replace(".json", "_slots.npz")
    np.savez(hp + ".tmp.npz", **arrays)
    os.replace(hp + ".tmp.npz", hp)
    errs = {r: v["error"] for r, v in doc["records"].items() if v["error"]}
    print("DONE", json.dumps({"records": len(ids), "errors": len(errs), "legs": len(used),
                              "rows_used": doc["run"]["rows_used"], "torch_imported": doc["run"]["torch_imported"]}))
    for r, e in errs.items():
        print("ERROR", r, e)


if __name__ == "__main__":
    main()
