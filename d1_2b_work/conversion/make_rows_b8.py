"""The device rows of the eight-signature file's batch signature decide_b8_L128: the 64-states workload's 64 requests
(wl_64_states_00..63, set/requests.json) through the batch host (host/d1_batch.D1HostBatched) in ONE `run` (the
provider's `_batches` puts the 64 sequences in one batch), its decide calls recorded by a backend that packs them
exactly as host/multi8.Multi8Decide does (eight sequences a call of [8, 128, ...], padding rows past the last real
one: embeds 0, sees_q [0, 1, 0..], sees_k 0), so the phone runs the very bytes the Mac host sends.

  $EXPORT conversion/make_rows_b8.py --out-dir out/rows_b8

Writes <out-dir>/b8s64.json (the test app's generic contract, replay.py write_rows: signature decide_b8_L128, inputs
embeds [8,128,2048] / sees_q [8,128,66] / sees_k [8,128,66], output hidden [8,128,2048]; every row = one call with
`slots` = the real rows' (record, sequence index in the host's order, T, state_len); one timing set "64_states" = the 8
rows as one request) and its stacked input files, and appends to <out-dir>/SHA256SUMS. The replay of a phone leg of this
file is replay_b8.py (the same packing, rows found by the sha256 of every input). numpy only (no torch).
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
EMBED = os.path.join(WORK, "tables/embed_table.npy")
B, L, HIDDEN = 8, 128, 2048
SIG = "decide_b8_L128"


def load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _sha(a):
    import hashlib

    return hashlib.sha256(np.ascontiguousarray(a, np.float32).tobytes()).hexdigest()


def pack(chunk):
    """Multi8Decide._batch's layout of one call (host/multi8.py)."""
    from d1_layout import SEES, make_sees

    e = np.zeros((B, L, HIDDEN), np.float32)
    q = np.zeros((B, L, SEES), np.float32)
    k = np.zeros((B, L, SEES), np.float32)
    q[:, :, 1] = 1.0
    for b, (embeds, st, tags) in enumerate(chunk):
        e[b, : embeds.shape[0]] = embeds
        qb, kb = make_sees(st, tags, L)
        q[b], k[b] = qb[0], kb[0]
    return {"embeds": e, "sees_q": q, "sees_k": k}


class RecordingBatch:
    """D1HostBatched's decide backend for the rows: `many` keeps the packed calls (batch_min 8, as the ship host)."""
    max_len = L

    def __init__(self):
        self.calls = []
        self.last = {}

    def many(self, seqs):
        seqs = [(np.asarray(e, np.float32), int(s), list(q)) for e, s, q in seqs]
        assert len(seqs) >= 8 and all(e.shape[0] <= L for e, _, _ in seqs), "the 64-states batch goes through b8"
        for i in range(0, len(seqs), B):
            chunk = seqs[i:i + B]
            self.calls.append({"arrays": pack(chunk), "slots": [(_sha(e), e.shape[0], s) for e, s, _ in chunk]})
        self.last = {"calls": -(-len(seqs) // B)}
        return [np.zeros_like(e) for e, _, _ in seqs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(WORK, "out/rows_b8"))
    a = ap.parse_args()
    from d1_batch import D1HostBatched
    rp = load_by_path("r5_replay", os.path.join(WORK, "conversion/replay.py"))   # write_rows

    t0 = time.time()
    recs = {r["id"]: r for r in json.load(open(os.path.join(WORK, "set/requests.json")))["records"]}
    ids = [f"wl_64_states_{i:02d}" for i in range(64)]
    dec = RecordingBatch()
    host = D1HostBatched(SNAP, dec, None, embed_table=EMBED)
    host.system_one_batch([(recs[r]["request"]["state"], recs[r]["request"]["questions"]) for r in ids])
    seqs = host.last["seqs"]
    assert len(seqs) == 64 and len(dec.calls) == 8, (len(seqs), len(dec.calls))
    stem = "b8s64"
    ins = [{"name": "embeds", "dtype": "float32", "shape": [B, L, HIDDEN], "file": f"{stem}__embeds.f32"},
           {"name": "sees_q", "dtype": "float32", "shape": [B, L, 66], "file": f"{stem}__sees_q.f32"},
           {"name": "sees_k", "dtype": "float32", "shape": [B, L, 66], "file": f"{stem}__sees_k.f32"}]
    outs = [{"name": "hidden", "dtype": "float32", "shape": [B, L, HIDDEN]}]
    # the host order of the sequences: each real row found by the sha256 of its embeds (the table rows of its ids)
    by_sha = {}
    for n, s in enumerate(seqs):
        by_sha.setdefault(_sha(host.embed.rows(s["ids"])), []).append(n)
    rows = []
    for c, call in enumerate(dec.calls):
        slots = []
        for h, T, st in call["slots"]:
            n = by_sha[h].pop(0)
            assert len(seqs[n]["ids"]) == T and int(seqs[n]["state_len"]) == st, (n, T, st)
            slots.append({"record": ids[seqs[n]["request"]], "seq": n, "T": T, "state_len": st})
        rows.append({"key": f"{stem}/c{c}", "record": "64_states", "call": c, "real_rows": len(slots), "slots": slots})
    sets = [{"name": "64_states", "kind": "request", "rows": [r["key"] for r in rows]}]
    written = rp.write_rows(os.path.join(a.out_dir, stem + ".json"), "d1-2b_decide", SIG, ins, outs, rows,
                            [c["arrays"] for c in dec.calls],
                            {"sets": sets, "made_by": "conversion/make_rows_b8.py",
                             "requests": ids, "host": "host/d1_batch.py system_one_batch (one run of 64 requests)"})
    with open(os.path.join(a.out_dir, "SHA256SUMS"), "a") as fh:
        for s, f in written:
            fh.write(f"{s}  {f}\n")
    print(json.dumps({"rows": len(rows), "real_rows": [r["real_rows"] for r in rows], "T_max": [max(s["T"] for s in r["slots"]) for r in rows],
                      "files": [f for _, f in written], "wall_s": round(time.time() - t0, 1),
                      "torch_imported": "torch" in sys.modules}))


if __name__ == "__main__":
    main()
