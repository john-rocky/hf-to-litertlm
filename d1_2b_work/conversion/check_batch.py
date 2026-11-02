"""The batch graphs against the single L128 file on the host's own sequences (the hidden of each batch row at its real
positions [:T] vs the same sequence through the single L128 fp16-FC file, on the same accelerator; a difference above
1e-3 stops the checks):

  set "states64" = the 64 sequences of the "64 states in one call" workload (wl_64_states_00..63, 58-77 tokens)
  set "mixed64"  = the first sequence of 64 records of conversion/ids_L128.json spread over their lengths (43-128 tokens)
  per accelerator (cpu = XNNPACK 8 threads, gpu = Metal float32), one model loaded at a time:
    single L128 (64 calls a set) -> B=8 (8 calls a set) -> B=64 (1 call a set); plus "partial": the first 5 sequences of
    states64 alone (B=8: 3 padding rows, B=64: 59) = the padding rows must not move the real rows (bit compare with
    the same sequences in the full batch, and vs single).
The sequences come from the host itself (host/d1_batch.D1HostBatched with a capturing backend: the exact embeds,
state lengths and question tags the gate and the timing pass). numpy + ai-edge-litert only. multi8_check.py imports
its sequences() for the eight-signature file's checks.

  $EXPORT conversion/check_batch.py --accel gpu --out results/check_batch_gpu32.json
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
sys.path.insert(2, HERE)
SNAP = os.path.join(WORK, "snapshot")
SINGLE = "out/graphs/d1-2b_decide_L128_fp16fc.tflite"


class Capture:
    max_len = 128

    def __init__(self):
        self.seqs = []

    def many(self, seqs):
        import numpy as np

        self.seqs.extend((np.array(e, np.float32), int(s), list(q)) for e, s, q in seqs)
        self.last = {"calls": 1}
        return [np.zeros_like(e) for e, _, _ in seqs]


def sequences():
    """(states64, mixed64): the host's (embeds, state_len, question) of each set."""
    from d1_batch import D1HostBatched

    reqs = {r["id"]: r for r in json.load(open(os.path.join(WORK, "set/requests.json")))["records"]}
    ids128 = json.load(open(os.path.join(WORK, "conversion/ids_L128.json")))["ids"]
    tc = json.load(open(os.path.join(WORK, "set/token_counts.json")))["records"]
    cap = Capture()
    host = D1HostBatched(SNAP, cap, None, embed_table=os.path.join(WORK, "tables/embed_table.npy"))
    work = [reqs[f"wl_64_states_{i:02d}"]["request"] for i in range(64)]
    host.system_one_batch([(w["state"], w["questions"]) for w in work])
    states64 = list(cap.seqs)
    assert len(states64) == 64, len(states64)
    pool = sorted((i for i in ids128 if i in reqs and not i.startswith("wl_64_states")), key=lambda i: (tc[i]["longest"], i))
    pick = [pool[int(round(j * (len(pool) - 1) / 63))] for j in range(64)]
    mixed = []
    for rid in pick:
        cap.seqs = []
        r = reqs[rid]["request"]
        host.system_one_batch([(r["state"], r["questions"])])
        mixed.append(cap.seqs[0])
    return states64, mixed, pick


def footprint():
    from fp import footprint as fp   # conversion/fp.py (proc_pid_rusage)

    return fp()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--accel", choices=("cpu", "gpu"), required=True)
    ap.add_argument("--Bs", default="8,64")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import numpy as np

    from d1_batch import LiteRTDecideBatch
    from lrt_backend import Graph
    from d1_layout import make_sees

    states64, mixed64, pick = sequences()
    sets = {"states64": states64, "mixed64": mixed64}
    doc = {"accelerator": a.accel, "precision": "fp32", "single": SINGLE, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
           "sets": {n: {"sequences": len(s), "T_min": min(len(q) for _, _, q in s), "T_max": max(len(q) for _, _, q in s)}
                    for n, s in sets.items()},
           "mixed64_records": pick, "results": {}}
    t0 = time.time()
    g = Graph(os.path.join(WORK, SINGLE), "decide", a.accel, "fp32", 8)
    doc["single_load_s"], doc["single_fully_accelerated"] = round(g.load_s, 2), g.fully_accelerated
    single = {}
    for n, s in sets.items():
        outs = []
        for e, st, q in s:
            T = e.shape[0]
            emb = np.zeros((1, 128, 2048), np.float32)
            emb[0, :T] = e
            sq, sk = make_sees(st, q, 128)
            outs.append(g(embeds=emb, sees_q=sq, sees_k=sk)["hidden"][0, :T].copy())
        single[n] = outs
    doc["single_footprint"] = footprint()
    g.close()
    del g
    doc["single_s"] = round(time.time() - t0, 1)
    for B in (int(x) for x in a.Bs.split(",")):
        path = os.path.join(WORK, f"out/graphs/d1-2b_decide_b{B}_L128_fp16fc.tflite")
        dec = LiteRTDecideBatch(path, a.accel, "fp32", 8)
        row = {"file": os.path.relpath(path, WORK), "load_s": round(dec.load_s, 2),
               "fully_accelerated": dec.fully_accelerated, "sets": {}}
        full = {}
        for n, s in sets.items():
            dec.calls = []
            hs = dec.many(s)
            full[n] = hs
            per = [float(np.abs(h - r).max()) for h, r in zip(hs, single[n])]
            row["sets"][n] = {"calls": len(dec.calls), "max_abs_vs_single": max(per),
                              "mean_of_seq_max": float(np.mean(per)), "worst_seq": int(np.argmax(per)),
                              "finite": all(bool(np.isfinite(h).all()) for h in hs),
                              "seq_max_abs": [round(x, 7) for x in per]}
        part = dec.many(states64[:5])
        row["partial5"] = {"rows": 5, "padding_rows": B - 5,
                           "vs_full_batch_max_abs": max(float(np.abs(h - f).max()) for h, f in zip(part, full["states64"][:5])),
                           "vs_single_max_abs": max(float(np.abs(h - r).max()) for h, r in zip(part, single["states64"][:5]))}
        row["footprint"] = footprint()
        dec.close()
        del dec
        doc["results"][f"B{B}"] = row
        print(json.dumps({f"B{B}": {"load_s": row["load_s"], "fully_accelerated": row["fully_accelerated"],
                                     **{n: {k: v for k, v in r.items() if k != "seq_max_abs"} for n, r in row["sets"].items()},
                                     "partial5": row["partial5"]}}), flush=True)
        json.dump(doc, open(os.path.join(WORK, a.out), "w"), indent=1)
    doc["max_abs_vs_single_all"] = max(r["sets"][n]["max_abs_vs_single"] for r in doc["results"].values()
                                       for n in r["sets"])
    doc["stop_condition_1e-3"] = doc["max_abs_vs_single_all"] > 1e-3
    doc["torch_imported"] = "torch" in sys.modules
    json.dump(doc, open(os.path.join(WORK, a.out), "w"), indent=1)
    print("DONE max_abs_vs_single_all", doc["max_abs_vs_single_all"], "torch_imported", doc["torch_imported"])


if __name__ == "__main__":
    main()
