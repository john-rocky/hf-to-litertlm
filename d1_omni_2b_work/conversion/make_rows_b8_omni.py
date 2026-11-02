"""The device rows of the d1-Omni-2B eight-signature file (the test app's generic mode): the card's six Omni workloads
as the decide calls the ship host makes, packed by the ship backend's own code, so the phone runs the very bytes the
Mac host sends (make_rows_b8.py's rule of the d1-2B conversion, on the Omni host).

The host = host/d1_batch_omni.D1HostBatchedOmni (the timing's host), its decide backend = an instance of
host/multi8_omni.Multi8Decide (batch_min 8, the timing's) whose graph is a recorder: every call of `_one` (the
smallest bucket, [1, L, ...]) and `_batch` (decide_b8_L128, [8, 128, ...], padding rows past the last real one) keeps
the arrays it would hand to the CompiledModel. The tower and the audio encoder are replay.py's recording backends
(zeros out): the timing rows time the graph's work, which does not depend on the values, so the picture and clip
positions of the 384 px and 10 s audio rows hold zeros (written in the file as `media_features: zeros`, as the d1-2B
picture timing rows). Requests = bench_b8_omni.py's WORK, opened as it opens them (records.py, set/requests.json +
set/requests_audio.json, set/media.json + set/media_audio.json), one `system_one` a request, the 64 states in one
`system_one_batch`.

  $EXPORT conversion/make_rows_b8_omni.py --out-dir out/rows_b8

Writes (replay.write_rows, the generic contract; graph_kind d1-omni-2b_decide):
  tmL128q.json   decide_L128          sets one_question, 3_questions (one row each, kind single)
  tmL256.json    decide_L256          sets 384px, audio_10s (one row each)
  tmL6144.json   decide_L6144_qb1024  set 5k_state (one row)
  b8s64.json     decide_b8_L128       8 rows of [8, 128, ...] (each row's `slots` = its real sequences: record, the
                                      host's sequence index, T, state_len), set 64_states (kind request = the 8 rows)
and appends their files' sha256 to <out-dir>/SHA256SUMS (the conversion linked the decL128 rows of out/rows beside
them). The replay of a phone leg of b8s64.json is replay_b8_omni.py. numpy, tokenizers, Pillow; no torch.
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORKDIR = os.path.normpath(os.path.join(HERE, ".."))
SNAP = os.path.join(WORKDIR, "snapshot")
TABLE = os.path.join(WORKDIR, "tables/embed_table.npy")
FILES = ["set/requests.json", "set/requests_audio.json"]
MEDIA = ["set/media.json", "set/media_audio.json"]                  # conversion/bench_b8_omni.py's FILES / MEDIA
SINGLE = {"one_question": "wl_one_question", "3_questions": "wl_3_questions", "5k_state": "wl_5k_state",
          "384px": "wl_384px", "audio_10s": "wl_audio_10s"}
STATES = [f"wl_64_states_{i:02d}" for i in range(64)]
STEM = {"decide_L128": "tmL128q", "decide_L256": "tmL256", "decide_L6144_qb1024": "tmL6144"}
B, BL, HIDDEN, SEES = 8, 128, 2048, 66


def load_by_path(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class RecordingGraph:
    """MultiGraph's call API: keeps (signature, arrays) of every call; returns zeros of the output shape."""

    def __init__(self):
        self.calls = []
        self.tag = None

    def __call__(self, sig, **arrays):
        self.calls.append({"tag": self.tag, "sig": sig,
                           "arrays": {k: np.array(v, np.float32, copy=True) for k, v in arrays.items()}})
        return {"hidden": np.zeros_like(arrays["embeds"])}


def packing_decide(m8):
    """A Multi8Decide over the recorder (its __init__ minus the CompiledModel): the routing and packing code is the
    module's own."""
    d = object.__new__(m8.Multi8Decide)
    d.path, d.accelerator, d.precision = "recorder", "none", "fp32"
    d.m = RecordingGraph()
    d.sigs = {128: "decide_L128", 256: "decide_L256", 512: "decide_L512", 1024: "decide_L1024", 2048: "decide_L2048",
              4096: "decide_L4096_qb1024", 6144: "decide_L6144_qb1024"}       # the multi8 file's signature table
    d.bsig, d.B, d.BL = "decide_b8_L128", B, BL
    d.paths = {L: "recorder" for L in d.sigs}
    d.max_len, d.batch_min = max(d.sigs), 8
    d.sharing, d.weight_cache, d.load_s, d.fully_accelerated = False, "", 0.0, True
    d.loads, d.calls, d.last, d.graphs = [], [], {}, {}
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(WORKDIR, "out/rows_b8"))
    a = ap.parse_args()
    t0 = time.time()
    hb = load_by_path("d1_batch_omni", os.path.join(WORKDIR, "host/d1_batch_omni.py"))
    m8 = load_by_path("multi8_omni", os.path.join(WORKDIR, "host/multi8_omni.py"))
    rp = load_by_path("r7_replay", os.path.join(WORKDIR, "conversion/replay.py"))      # Omni kinds, specs, write_rows
    from records import load_records, open_request
    import records as _rec
    assert os.path.samefile(_rec.__file__, os.path.join(WORKDIR, "conversion/records.py")), _rec.__file__

    recs, media = load_records(FILES, MEDIA)
    recs = {r["id"]: r for r in recs}
    dec = packing_decide(m8)
    order = []                     # every `many` call's sequences: [(embeds [T, d], state_len)]
    _many = dec.many

    def many(seqs_in):
        order.append([(np.array(e, np.float32, copy=True), int(st)) for e, st, _ in seqs_in])
        return _many(seqs_in)
    dec.many = many                # the host calls getattr(decide, "many")
    tow, aud = rp.RecordingTower(), rp.RecordingAudio()
    host = hb.D1HostBatchedOmni(SNAP, dec, tow, embed_table=TABLE, audio=aud)
    calls_of = collections.OrderedDict()
    for wl, rid in SINGLE.items():
        dec.m.tag = rid
        n0 = len(dec.m.calls)
        st, qs, items = open_request(recs[rid], media)
        host.system_one(st, qs, items)
        got = dec.m.calls[n0:]
        assert len(got) == 1 and got[0]["sig"] != dec.bsig, (wl, [c["sig"] for c in got])
        got[0]["T"] = len(host.last["seqs"][0]["ids"])
        got[0]["state_len"] = int(host.last["seqs"][0]["state_len"])
        calls_of[wl] = got[0]
    tower_calls, audio_calls = len(tow.calls), len(aud.calls)
    # the 64 states: one provider batch of 64 sequences -> 8 calls of decide_b8_L128 (bench_b8_omni's 64_states)
    dec.m.tag = "64_states"
    n0, n_many0 = len(dec.m.calls), len(order)
    reqs = [open_request(recs[rid], media) for rid in STATES]
    assert all(not items for _, _, items in reqs)
    host.system_one_batch([(st, qs) for st, qs, _ in reqs])
    bcalls = dec.m.calls[n0:]
    seqs = host.last["seqs"]
    assert len(seqs) == 64 and len(bcalls) == 8 and all(c["sig"] == dec.bsig for c in bcalls), (len(seqs), len(bcalls))
    os.makedirs(a.out_dir, exist_ok=True)
    made = "conversion/make_rows_b8_omni.py"
    written = []
    by_sig = collections.OrderedDict()
    for wl, c in calls_of.items():
        by_sig.setdefault(c["sig"], []).append((wl, c))
    for sig, items in by_sig.items():
        stem = STEM[sig]
        L = int(items[0][1]["arrays"]["embeds"].shape[1])
        ins, outs = rp.decide_spec(L, stem)
        rows = [{"key": f"{c['tag']}/c0", "record": c["tag"], "call": 0, "T": c["T"], "state_len": c["state_len"]}
                for _, c in items]
        sets = [{"name": wl, "kind": "single", "rows": [f"{c['tag']}/c0"]} for wl, c in items]
        written += rp.write_rows(os.path.join(a.out_dir, stem + ".json"), rp.KIND_DECIDE, sig, ins, outs, rows,
                                 [c["arrays"] for _, c in items],
                                 {"sets": sets, "media_features": "zeros", "made_by": made,
                                  "host": "host/d1_batch_omni.py system_one (bench_b8_omni's path)",
                                  "packing": "host/multi8_omni.Multi8Decide._one"})
    # b8 rows: the sequences in the order `many` got them (each call = the next 8), each found among the host's
    # sequences by the sha256 of its embeddings (text only: the table rows of its ids; the d1-2B rule)
    stem = "b8s64"
    ins = [{"name": "embeds", "dtype": "float32", "shape": [B, BL, HIDDEN], "file": f"{stem}__embeds.f32"},
           {"name": "sees_q", "dtype": "float32", "shape": [B, BL, SEES], "file": f"{stem}__sees_q.f32"},
           {"name": "sees_k", "dtype": "float32", "shape": [B, BL, SEES], "file": f"{stem}__sees_k.f32"}]
    outs = [{"name": "hidden", "dtype": "float32", "shape": [B, BL, HIDDEN]}]
    by_sha = {}
    for n, s in enumerate(seqs):
        by_sha.setdefault(rp.sha(host.embed.rows(s["ids"])), []).append(n)
    chunks = [batch[i:i + B] for batch in order[n_many0:] for i in range(0, len(batch), B)]
    assert len(chunks) == len(bcalls), (len(chunks), len(bcalls))
    rows = []
    for ci, (chunk, c) in enumerate(zip(chunks, bcalls)):
        slots = []
        for b, (e, st) in enumerate(chunk):
            T = e.shape[0]
            assert np.array_equal(c["arrays"]["embeds"][b, :T], e) and not c["arrays"]["embeds"][b, T:].any()
            n = by_sha[rp.sha(e)].pop(0)
            assert len(seqs[n]["ids"]) == T and int(seqs[n]["state_len"]) == st, (n, T, st)
            slots.append({"record": STATES[seqs[n]["request"]], "seq": n, "T": T, "state_len": st})
        assert not c["arrays"]["embeds"][len(chunk):].any()      # padding rows past the last real one: embeds 0
        rows.append({"key": f"{stem}/c{ci}", "record": "64_states", "call": ci, "real_rows": len(slots),
                     "slots": slots})
    assert sum(r["real_rows"] for r in rows) == 64 and not any(by_sha.values()), [r["real_rows"] for r in rows]
    sets = [{"name": "64_states", "kind": "request", "rows": [r["key"] for r in rows]}]
    written += rp.write_rows(os.path.join(a.out_dir, stem + ".json"), rp.KIND_DECIDE, dec.bsig, ins, outs, rows,
                             [c["arrays"] for c in bcalls],
                             {"sets": sets, "made_by": made, "requests": STATES,
                              "host": "host/d1_batch_omni.py system_one_batch (one run of the 64 requests)",
                              "packing": "host/multi8_omni.Multi8Decide._batch"})
    with open(os.path.join(a.out_dir, "SHA256SUMS"), "a") as fh:
        for s, f in written:
            fh.write(f"{s}  {f}\n")
    print(json.dumps({"single": {wl: {"sig": c["sig"], "T": c["T"], "state_len": c["state_len"]}
                                 for wl, c in calls_of.items()},
                      "tower_calls": tower_calls, "audio_calls": audio_calls,
                      "b8_rows": len(rows), "real_rows": [r["real_rows"] for r in rows],
                      "T_max": [max(s["T"] for s in r["slots"]) for r in rows],
                      "files": [f for _, f in written], "wall_s": round(time.time() - t0, 1),
                      "torch_imported": "torch" in sys.modules}))


if __name__ == "__main__":
    main()
