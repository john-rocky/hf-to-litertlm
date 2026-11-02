"""The record set of the eight-sequence graph's checks, written before any number: the d1-Omni-2B reference-answered
records whose every sequence is <= 128 tokens, counted twice and required to agree: (1) the provider's own sequences in
the reference (out/ref/ref_fp32.json, each record's `seqs[].ids`), (2) the PyTorch-free host's sequences
(host/d1_host.py D1Host.sequences: no backend runs; pictures and clips are read and featurized, which fixes their token
counts). Then the two rows of the gate (text + picture = conversion/ids_textpic.json, audio = conversion/ids_audio.json)
restricted to the set, and the chosen control arms (conversion/red_arms_chosen_arms.json,
red_arms_audio_chosen_arms.json) whose arm AND base both fit. Record kinds (text / picture / audio state / spoken
question) are counted for the record.

  $HOST conversion/make_ids_L128.py
      -> conversion/ids_L128.json (+ the lists ids_L128_textpic.json, ids_L128_audio.json, ids_L128_64states.json),
         conversion/red_arms_L128.json, conversion/red_arms_audio_L128.json
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WORK, "host"))
SNAP = os.path.join(WORK, "snapshot")
TABLE = os.path.join(WORK, "tables/embed_table.npy")
FILES = ["set/requests.json", "set/extra_g96x72.json", "set/red_arms.json", "set/requests_audio.json",
         "set/red_arms_audio.json", "set/card_omni.json"]          # the gate runs' files
MEDIA = ["set/media.json", "set/media_extra.json", "set/media_audio.json"]
L = 128


def sha(p):
    return hashlib.sha256(open(os.path.join(WORK, p), "rb").read()).hexdigest()


def main():
    import d1_host
    from records import load_records, open_request

    assert os.path.samefile(d1_host.__file__, os.path.join(WORK, "host/d1_host.py")), d1_host.__file__
    ref = json.load(open(os.path.join(WORK, "out/ref/ref_fp32.json")))["records"]
    recs, media = load_records(FILES, MEDIA)
    recs = {r["id"]: r for r in recs}
    host = d1_host.D1Host(SNAP, None, None, embed_table=TABLE, audio=None)
    ref_longest, host_longest, kinds, mismatch = {}, {}, {}, []
    for rid, r in sorted(ref.items()):
        if r.get("error"):
            continue
        ref_longest[rid] = max(len(s["ids"]) for s in r["seqs"])
        st, qs, items = open_request(recs[rid], media)
        seqs = host.sequences(st, qs, items)
        host_longest[rid] = max(len(s["ids"]) for s in seqs)
        if host_longest[rid] != ref_longest[rid]:
            mismatch.append((rid, ref_longest[rid], host_longest[rid]))
        pics = sum(len(s["image_grid_thw"] or []) for s in seqs)
        clips = sum(len(s["clips"]) for s in seqs)
        spoken = any(isinstance(q.get("instructions"), dict) for q in recs[rid]["request"]["questions"].values()
                     if isinstance(q, dict))
        kinds[rid] = ("picture" if pics else "") + ("+" if pics and clips else "") + (
            ("spoken" if spoken and not (recs[rid]["request"].get("media")) else "audio") if clips else "") or "text"
    ids = sorted(k for k, v in ref_longest.items() if v <= L)
    textpic = set(json.load(open(os.path.join(WORK, "conversion/ids_textpic.json"))))
    audio = set(json.load(open(os.path.join(WORK, "conversion/ids_audio.json"))))
    doc = {"what": f"d1-Omni-2B reference-answered records whose every sequence is <= {L} tokens (written before any number)",
           "bucket": L, "count": len(ids), "ids": ids,
           "textpic_ids": [i for i in ids if i in textpic], "audio_ids": [i for i in ids if i in audio],
           "kinds": dict(collections.Counter(kinds[i] for i in ids)),
           "media_records": {i: kinds[i] for i in ids if kinds[i] != "text"},
           "wl_64_states": sum(1 for i in ids if i.startswith("wl_64_states_")),
           "longest_host_vs_ref_mismatch": mismatch, "answered": len(ref_longest),
           "sources": {"ref": "out/ref/ref_fp32.json", "ref_sha256": sha("out/ref/ref_fp32.json"),
                       "files": FILES, "media": MEDIA, "host": "host/d1_host.py",
                       "host_sha256": sha("host/d1_host.py")}}
    doc["textpic_count"], doc["audio_count"] = len(doc["textpic_ids"]), len(doc["audio_ids"])
    out = {}
    for name, src in (("red_arms_L128.json", "conversion/red_arms_chosen_arms.json"),
                      ("red_arms_audio_L128.json", "conversion/red_arms_audio_chosen_arms.json")):
        arms = json.load(open(os.path.join(WORK, src)))
        keep = [a for a in arms if ref_longest.get(a["id"], 10 ** 9) <= L and ref_longest.get(a["base_id"], 10 ** 9) <= L]
        json.dump(keep, open(os.path.join(HERE, name), "w"), indent=1)
        out[name] = {"from": src, "kept": [a["id"] for a in keep],
                     "all": {a["id"]: [ref_longest.get(a["id"]), ref_longest.get(a["base_id"])] for a in arms}}
    doc["red_arms"] = out
    json.dump(doc, open(os.path.join(HERE, "ids_L128.json"), "w"), indent=0)
    # the --ids-file lists of conversion/score.py (a JSON list each): the two rows and the 64-states records alone
    for name, lst in (("ids_L128_textpic.json", doc["textpic_ids"]), ("ids_L128_audio.json", doc["audio_ids"]),
                      ("ids_L128_64states.json", [i for i in ids if i.startswith("wl_64_states_")])):
        json.dump(lst, open(os.path.join(HERE, name), "w"), indent=0)
    print(json.dumps({k: doc[k] for k in ("count", "textpic_count", "audio_count", "kinds", "wl_64_states",
                                          "longest_host_vs_ref_mismatch", "answered")}))
    print(json.dumps({k: v["kept"] for k, v in out.items()}))
    print("host imports torch:", "torch" in sys.modules)


if __name__ == "__main__":
    main()
