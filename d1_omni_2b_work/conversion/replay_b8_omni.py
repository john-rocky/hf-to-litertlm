"""The phone's line of the batch signature decide_b8_L128 for d1-Omni-2B (replay_b8.py of the d1-2B conversion on the
Omni host): the batch host (host/d1_batch_omni.D1HostBatchedOmni) answers the 64-states workload's 64 requests in ONE
`system_one_batch`-shaped run (named -> run, the grouping make_rows_b8_omni.py recorded), every decide call answered
by a leg's output for the very same input bytes: the call is packed as host/multi8_omni.Multi8Decide._batch packs it
(eight sequences a call, padding rows; its own `_sees_batch`), the sha256 of each input's bytes finds the row of the
leg's rows file (b8s64.json; each slice checked against the row's sha256 when the file is opened), and the row's
output [8, 128, 2048] gives each real sequence its rows [b, :T]. A call no leg ran is an error (the replay never
answers for an input the phone did not see). Written in run_gate_m8.py's candidate form for score.py.

  $EXPORT conversion/replay_b8_omni.py --round-dir device/p8 --prefix m8_b8 \
      --rows-dir out/rows_b8 --out out/replay/p8_m8_b8.json
then
  $REF conversion/score.py --ref out/ref/ref_fp32.json --cand out/replay/p8_m8_b8.json \
      --files <the six set files> --ids-file conversion/ids_L128_64states.json --out results/line_p8_m8_b8.json
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
SNAP = os.path.join(WORK, "snapshot")
TABLE = os.path.join(WORK, "tables/embed_table.npy")
FILES = ["set/requests.json", "set/requests_audio.json"]
MEDIA = ["set/media.json", "set/media_audio.json"]                  # conversion/bench_b8_omni.py's FILES / MEDIA
STATES = [f"wl_64_states_{i:02d}" for i in range(64)]
B, L = 8, 128


def load_by_path(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


HB = load_by_path("d1_batch_omni", os.path.join(WORK, "host/d1_batch_omni.py"))
M8 = load_by_path("multi8_omni", os.path.join(WORK, "host/multi8_omni.py"))     # _sees_batch (the ship packing)
RP = load_by_path("r7_replay", os.path.join(WORK, "conversion/replay.py"))           # GenericRows, sha (Omni rows)


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


def pack(chunk):
    """Multi8Decide._batch's arrays of one call (embeds zero past each T and on the padding rows)."""
    e = np.zeros((B, L, chunk[0][0].shape[1]), np.float32)
    for b, (embeds, _, _) in enumerate(chunk):
        e[b, : embeds.shape[0]] = embeds
    q, k = M8._sees_batch([(s, qq) for _, s, qq in chunk], B, L)
    return {"embeds": e, "sees_q": q, "sees_k": k}


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
            arrays = pack(chunk)
            key = tuple(RP.sha(arrays[n]) for n in self.names)
            hit = self.index.get(key)
            if hit is None:
                raise KeyError(f"no replayed row has these bytes ({key[0][:12]}...): the phone never ran this input")
            src, pos = hit
            h = src.output(pos, "hidden")
            self.used.append((os.path.basename(src.path), src.rows[pos]["key"]))
            self.calls.append({"sig": "decide_b8_L128", "rows": len(chunk), "rows_file": os.path.basename(src.path),
                               "key": src.rows[pos]["key"]})
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
    from records import load_records, open_request

    srcs, used = legs_of(a.round_dir, a.rows_dir, a.prefix)
    dec = ReplayDecideB8(srcs)
    host = HB.D1HostBatchedOmni(SNAP, dec, None, embed_table=TABLE, audio=None)
    recs, media = load_records(FILES, MEDIA)
    recs = {r["id"]: r for r in recs}
    t0 = time.time()
    doc = {"records": {}, "groups": [], "run": {"replay_of": [os.path.relpath(r, WORK) for r in a.round_dir],
                                                "prefixes": a.prefix, "legs": used,
                                                "host": "host/d1_batch_omni.py (one run of the 64 requests)",
                                                "backend": "conversion/replay_b8_omni.py"}}
    arrays = {}
    named, states, reqs = [], [], []
    try:
        for rid in STATES:
            st, qs, items = open_request(recs[rid], media)
            nm = host.named(qs)
            named.append(nm)
            states.append(st)
            reqs.append((st, list(nm.values()), items))
        done, err = host.run(reqs), None
    except Exception as e:  # noqa: BLE001
        done, err = None, f"{type(e).__name__}: {e}"
    seqs = host.last.get("seqs", []) if err is None else []
    hidden = host.last.get("slot_hidden", []) if err is None else []
    doc["groups"].append({"records": len(STATES), "sequences": len(seqs), "graph_calls": len(dec.calls),
                          "rows_per_call": [c["rows"] for c in dec.calls]})
    for i, rid in enumerate(STATES):
        if err is not None:
            doc["records"][rid] = {"answers": None, "probs": None, "usage": {"input_tokens": None, "output_tokens": 0},
                                   "seqs": [], "graphs": {"group": 0}, "timing": None, "time_s": None, "error": err}
            continue
        probs, read = done[i]
        answers = {name: host.answer(states[i], q, p) for (name, q), p in zip(named[i].items(), probs)}
        own = [j for j, s in enumerate(seqs) if s["request"] == i]
        doc["records"][rid] = {"answers": jsonable(answers), "probs": jsonable(probs),
                               "usage": {"input_tokens": read, "output_tokens": 0},
                               "seqs": jsonable([{k: seqs[j].get(k) for k in ("first_question", "ids", "state_len",
                                                                              "question", "ends", "image_grid_thw",
                                                                              "audio")} for j in own]),
                               "graphs": {"group": 0, "decide": None, "tower": None, "audio": None},
                               "timing": None, "time_s": None, "error": None}
        for n, j in enumerate(own):
            arrays[f"{rid}::seq{n}"] = np.asarray(hidden[j], np.float32)
    doc["run"]["calls"] = dec.calls
    doc["run"]["rows_used"] = len(set(dec.used))
    doc["run"]["torch_imported"] = "torch" in sys.modules
    doc["wall_s"] = round(time.time() - t0, 2)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(doc, open(a.out + ".tmp", "w"))
    os.replace(a.out + ".tmp", a.out)
    hp = a.out.replace(".json", "_slots.npz")
    np.savez(hp + ".tmp.npz", **arrays)
    os.replace(hp + ".tmp.npz", hp)
    print("DONE", json.dumps({"records": len(STATES), "error": err, "legs": len(used), "rows_used": doc["run"]["rows_used"],
                              "calls": len(dec.calls), "torch_imported": doc["run"]["torch_imported"]}))


if __name__ == "__main__":
    main()
