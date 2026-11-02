"""The phone's line of the batch signature decide_b8_L128: the batch host (host/d1_batch.D1HostBatched) answers the
64-states workload's 64 requests in ONE `run` (the grouping make_rows_b8.py recorded), every decide call answered by
a phone leg's output for the very same input bytes: the call is packed as host/multi8.Multi8Decide packs it (eight
sequences a call, padding rows), the sha256 of each input's bytes finds the row of the leg's rows file (b8s64.json;
each slice checked against the row's sha256 when the file is opened), and the row's output [8, 128, 2048] gives each
real sequence its rows [b, :T]. A call no leg ran is an error (the replay never answers for an input the phone did not
see). Written in run_gate.py's candidate form for score.py.

  $EXPORT conversion/replay_b8.py --round-dir device/p6 --prefix m8_b8 --rows-dir out/rows_b8 \
      --out out/replay/p6_m8_b8.json
then
  $REF conversion/score.py --ref out/ref/ref_fp32.json --cand out/replay/p6_m8_b8.json \
      --files set/requests.json --ids <the 64 ids> --out results/line_p6_m8_b8.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(WORK, "host"))
sys.path.insert(1, HERE)
SNAP = os.path.join(WORK, "snapshot")
B, L = 8, 128


def load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


RP = load_by_path("r5_replay", os.path.join(WORK, "conversion/replay.py"))   # GenericRows, sha
MR = load_by_path("r6_make_rows_b8", os.path.join(WORK, "conversion/make_rows_b8.py"))   # pack (Multi8Decide's layout)


class ReplayDecideB8:
    names = ("embeds", "sees_q", "sees_k")
    max_len = L

    def __init__(self, sources):
        self.index = {}
        for src in sources:
            assert src.out is not None and tuple(s["name"] for s in src.inputs) == self.names, src.path
            for key, pos in src.by_sha().items():
                self.index.setdefault(key, (src, pos))
        self.used, self.last, self.calls = [], {}, []

    def many(self, seqs):
        seqs = [(np.asarray(e, np.float32), int(s), list(q)) for e, s, q in seqs]
        if len(seqs) < 8 or any(e.shape[0] > L for e, _, _ in seqs):
            raise KeyError("not a batch the b8 rows hold (fewer than 8 sequences or one longer than 128 tokens)")
        out = []
        for i in range(0, len(seqs), B):
            chunk = seqs[i:i + B]
            arrays = MR.pack(chunk)
            key = tuple(RP.sha(arrays[n]) for n in self.names)
            hit = self.index.get(key)
            if hit is None:
                raise KeyError(f"no replayed row has these bytes ({key[0][:12]}...): the phone never ran this input")
            src, pos = hit
            h = src.output(pos, "hidden")
            self.used.append((os.path.basename(src.path), src.rows[pos]["key"]))
            self.calls.append({"rows": os.path.basename(src.path), "key": src.rows[pos]["key"], "real_rows": len(chunk)})
            out.extend(np.array(h[b, : chunk[b][0].shape[0]]) for b in range(len(chunk)))
        self.last = {"calls": -(-len(seqs) // B)}
        return out

    def __call__(self, embeds, state_len, question):
        raise KeyError("the b8 replay answers batches only")


def legs_of(round_dirs, rows_dir, prefixes):
    srcs, used = [], []
    for rd in round_dirs:
        for name in sorted(os.listdir(rd)):
            if not name.endswith(".json") or not any(name.startswith(p) for p in prefixes):
                continue
            tag = name[:-5]
            try:
                rep = json.load(open(os.path.join(rd, name)))
            except Exception:
                continue
            if not isinstance(rep, dict) or rep.get("status") != "DONE" or rep.get("mode", "generic") != "generic":
                continue
            out = os.path.join(rd, f"out_{tag}.f32")
            if not os.path.exists(out):
                continue
            rows = rep["rows_file"]
            n = (rep.get("summary") or {}).get("count") or 0
            src = RP.GenericRows(os.path.join(rows_dir, rows), out, rows_limit=n)
            srcs.append(src)
            used.append({"tag": tag, "rows": rows, "out": os.path.relpath(out, WORK), "rows_ran": n,
                         "graph": rep.get("graph"), "accel": rep.get("accel"), "precision": rep.get("precision"),
                         "signature": rep.get("signature")})
    return srcs, used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round-dir", required=True, nargs="+")
    ap.add_argument("--prefix", required=True, action="append")
    ap.add_argument("--rows-dir", default=os.path.join(WORK, "out/rows_b8"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from d1_batch import D1HostBatched

    def jsonable(x):
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
    srcs, used = legs_of(a.round_dir, a.rows_dir, a.prefix)
    dec = ReplayDecideB8(srcs)
    host = D1HostBatched(SNAP, dec, None, embed_table=os.path.join(WORK, "tables/embed_table.npy"))
    recs = {r["id"]: r for r in json.load(open(os.path.join(WORK, "set/requests.json")))["records"]}
    ids = [f"wl_64_states_{i:02d}" for i in range(64)]
    named = [host.named(recs[r]["request"]["questions"]) for r in ids]
    t0 = time.time()
    doc = {"records": {}, "run": {"replay_of": [os.path.relpath(r, WORK) for r in a.round_dir], "prefixes": a.prefix,
                                  "legs": used, "host": "host/d1_batch.py (one run of the 64 requests)",
                                  "backend": "conversion/replay_b8.py"}}
    arrays = {}
    try:
        done = host.run([(recs[r]["request"]["state"], list(nm.values()), []) for r, nm in zip(ids, named)])
        err = None
    except Exception as e:  # noqa: BLE001
        done, err = None, f"{type(e).__name__}: {e}"
    seqs = host.last.get("seqs", [])
    hidden = host.last.get("slot_hidden", [])
    for i, rid in enumerate(ids):
        if err:
            doc["records"][rid] = {"answers": None, "probs": None, "usage": None, "seqs": [], "graphs": {},
                                   "time_s": None, "error": err}
            continue
        probs, read = done[i]
        state = recs[rid]["request"]["state"]
        answers = {name: host.answer(state, q, p) for (name, q), p in zip(named[i].items(), probs)}
        own = [j for j, s in enumerate(seqs) if s["request"] == i]
        doc["records"][rid] = {"answers": jsonable(answers), "probs": jsonable(probs),
                               "usage": {"input_tokens": read, "output_tokens": 0},
                               "seqs": jsonable([{k: seqs[j][k] for k in ("first_question", "ids", "state_len", "question",
                                                                          "ends", "image_grid_thw")} for j in own]),
                               "graphs": {"decide": dec.calls}, "time_s": None, "error": None}
        for n, j in enumerate(own):
            arrays[f"{rid}::seq{n}"] = np.asarray(hidden[j], np.float32)
    doc["wall_s"] = round(time.time() - t0, 2)
    doc["run"]["torch_imported"] = "torch" in sys.modules
    doc["run"]["rows_used"] = len(set(dec.used))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(doc, open(a.out + ".tmp", "w"))
    os.replace(a.out + ".tmp", a.out)
    hp = a.out.replace(".json", "_slots.npz")
    np.savez(hp + ".tmp.npz", **arrays)
    os.replace(hp + ".tmp.npz", hp)
    print("DONE", json.dumps({"records": len(ids), "error": err, "legs": len(used), "rows_used": doc["run"]["rows_used"],
                              "torch_imported": doc["run"]["torch_imported"]}))


if __name__ == "__main__":
    main()
