"""The b8 input sets of the eight-signature file's identity check (multi8_check_omni.py), check_batch.sequences() of the
d1-2B conversion rebuilt on the Omni host and table: the host's own (embeds, state_len, question) of

  states64  the 64-states workload's 64 sequences (wl_64_states_00..63, one `system_one_batch` of 64 requests)
  mixed64   64 records' first sequences spread over 43..128 tokens: the text records of conversion/ids_L128.json (no
            media; the 64-states records excluded) sorted by (longest sequence, id), every (n - 1) / 63-th one (the
            d1-2B pick rule)

captured by a stand-in decide backend (returns zeros; no graph runs) through host/d1_batch_omni.D1HostBatchedOmni.
numpy + the host only (no torch).
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.normpath(os.path.join(HERE, ".."))
SNAP = os.path.join(WORK, "snapshot")
TABLE = os.path.join(WORK, "tables/embed_table.npy")
FILES = ["set/requests.json", "set/extra_g96x72.json", "set/red_arms.json", "set/requests_audio.json",
         "set/red_arms_audio.json", "set/card_omni.json"]
MEDIA = ["set/media.json", "set/media_extra.json", "set/media_audio.json"]


def load_by_path(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


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
    """(states64, mixed64, pick): the host's (embeds, state_len, question) of each set and mixed64's record ids."""
    hb = load_by_path("d1_batch_omni", os.path.join(WORK, "host", "d1_batch_omni.py"))
    from records import load_records, open_request   # records.py of this folder

    recs, media = load_records(FILES, MEDIA)
    recs = {r["id"]: r for r in recs}
    ref = json.load(open(os.path.join(WORK, "out/ref/ref_fp32.json")))["records"]
    sets = json.load(open(os.path.join(HERE, "ids_L128.json")))
    longest = {i: max(len(s["ids"]) for s in ref[i]["seqs"]) for i in sets["ids"]}
    cap = Capture()
    host = hb.D1HostBatchedOmni(SNAP, cap, None, embed_table=TABLE, audio=None)
    work = [open_request(recs[f"wl_64_states_{i:02d}"], media) for i in range(64)]
    assert all(not items for _, _, items in work)
    host.system_one_batch([(st, qs) for st, qs, _ in work])
    states64 = list(cap.seqs)
    assert len(states64) == 64, len(states64)
    pool = sorted((i for i in sets["ids"] if not i.startswith("wl_64_states") and i not in sets["media_records"]),
                  key=lambda i: (longest[i], i))
    pick = [pool[int(round(j * (len(pool) - 1) / 63))] for j in range(64)]
    mixed = []
    for rid in pick:
        cap.seqs = []
        st, qs, items = open_request(recs[rid], media)
        assert not items, rid
        host.system_one_batch([(st, qs)])
        mixed.append(cap.seqs[0])
    return states64, mixed, pick
