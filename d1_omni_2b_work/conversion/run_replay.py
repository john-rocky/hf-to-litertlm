"""The PyTorch-free host (host/d1_host.py) over a run's graph outputs (replay backends, replay.py): every generic leg
of one pass in a phone's pulled folder (its report DONE, its out_<tag>.f32 pulled) answers the host's graph calls for
the inputs it ran; the candidate is written in run_gate.py's form for score.py. The d1-2B replay runner for
d1-Omni-2B: the Omni snapshot, table and loader (records.py open_request: pictures and clips, spoken questions), and
the audio legs as a third backend. The legs' reports and outputs are written by the phones' test app, which is not
published.

  $EXPORT conversion/run_replay.py --round-dir <R> [<R2> ...] --prefix <tag prefix> [--prefix ...] \
      --rows-dir out/rows --set conversion/set_ids.json --out out/replay/<name>.json

A leg's rows file is read from <R>/gen/ (the prep legs' rows), else from --rows-dir. A call whose inputs no leg ran is
the record's error (KeyError: the device never ran this input), never a silent answer. For an input two legs ran, the
first leg in file-name order (round dirs in the order given) answers.
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
SNAP = os.path.join(WORK, "snapshot")
EMBED = os.path.join(WORK, "tables/embed_table.npy")


def legs_of(round_dir, rows_dir, prefixes):
    from replay import KIND_AUDIO, KIND_DECIDE, KIND_TOWER, GenericRows

    dec, tow, aud, used = [], [], [], []
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
        {KIND_DECIDE: dec, KIND_TOWER: tow, KIND_AUDIO: aud}[kind].append(src)
        used.append({"tag": tag, "rows": rows, "rows_path": os.path.relpath(rp, WORK), "out": os.path.relpath(out, WORK),
                     "rows_ran": n, "rows_in_file": len(src.rows), "graph": rep.get("graph"), "kind": kind,
                     "accel": rep.get("accel"), "precision": rep.get("precision"), "signature": rep.get("signature")})
    return dec, tow, aud, used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round-dir", required=True, nargs="+")
    ap.add_argument("--prefix", required=True, action="append")
    ap.add_argument("--rows-dir", default=os.path.join(WORK, "out/rows"))
    ap.add_argument("--set", default=os.path.join(WORK, "conversion/set_ids.json"))
    ap.add_argument("--ids", default=None, help="comma-separated record ids (default: the set's records)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import numpy as np

    from d1_host import D1Host
    from records import CARD_FILES, TEXTPIC_FILES, TEXTPIC_MEDIA, AUDIO_FILES, AUDIO_MEDIA, load_records, open_request
    from replay import ReplayAudio, ReplayDecide, ReplayTower
    from run_gate import jsonable

    dec_src, tow_src, aud_src, used = [], [], [], []
    for rd in a.round_dir:
        d, t, au, u = legs_of(rd, a.rows_dir, a.prefix)
        dec_src += d
        tow_src += t
        aud_src += au
        used += u
    decide = ReplayDecide(dec_src)
    tower = ReplayTower(tow_src) if tow_src else None
    audio = ReplayAudio(aud_src) if aud_src else None
    host = D1Host(SNAP, decide, tower, embed_table=EMBED, audio=audio)
    sel = json.load(open(a.set))
    ids = a.ids.split(",") if a.ids else sel["records"]
    recs, media = load_records(TEXTPIC_FILES + AUDIO_FILES + CARD_FILES, TEXTPIC_MEDIA + AUDIO_MEDIA)
    by = {r["id"]: r for r in recs}
    doc = {"records": {}, "run": {"replay_of": [os.path.relpath(r, WORK) for r in a.round_dir], "prefixes": a.prefix,
                                  "legs": used, "host": "host/d1_host.py", "backends": "conversion/replay.py"}}
    arrays = {}
    calls = {"decide": [], "tower": [], "audio": []}
    dcall = decide.__call__
    tcall = tower.__call__ if tower else None
    acall = audio.__call__ if audio else None

    def decide_logged(embeds, state_len, question):
        h = dcall(embeds, state_len, question)
        calls["decide"].append(dict(decide.last))
        return h

    def tower_logged(pv, grid):
        f = tcall(pv, grid)
        calls["tower"].append({k: list(v) if isinstance(v, tuple) else v for k, v in tower.last.items()})
        return f

    def audio_logged(x):
        f = acall(x)
        calls["audio"].append(dict(audio.last))
        return f
    host.decide = decide_logged
    host.decide.max_len = decide.max_len
    if tower:
        host.tower = tower_logged
    if audio:
        host.audio = audio_logged
        host.audio.max_frames = audio.max_frames
    t0 = time.time()
    for rid in ids:
        for v in calls.values():
            v.clear()
        t1 = time.time()
        try:
            st, qs, items = open_request(by[rid], media)
            named = host.named(qs)
            done = host.run([(st, list(named.values()), items)])
            probs, read = done[0]
            answers = {name: host.answer(st, q, p) for (name, q), p in zip(named.items(), probs)}
            err = None
        except Exception as e:
            probs, read, answers, err = None, None, None, f"{type(e).__name__}: {e}"
        seqs = [{k: s.get(k) for k in ("first_question", "ids", "state_len", "question", "ends", "image_grid_thw",
                                        "audio")}   # conversion/run_gate.py's keys
                for s in host.last.get("seqs", [])]
        doc["records"][rid] = {"answers": jsonable(answers), "probs": jsonable(probs),
                               "usage": {"input_tokens": read, "output_tokens": 0}, "seqs": jsonable(seqs),
                               "graphs": {k: jsonable(v) for k, v in calls.items()},
                               "time_s": round(time.time() - t1, 4), "error": err}
        if err is None:
            for j, h in enumerate(host.last.get("slot_hidden", [])):
                arrays[f"{rid}::seq{j}"] = np.asarray(h, np.float32)
        print(f"{rid} err={err}", flush=True)
    doc["wall_s"] = round(time.time() - t0, 2)
    doc["run"]["torch_imported"] = "torch" in sys.modules
    doc["run"]["rows_used"] = {"decide": len(set(decide.used)), "tower": len(set(tower.used)) if tower else 0,
                               "audio": len(set(audio.used)) if audio else 0}
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
