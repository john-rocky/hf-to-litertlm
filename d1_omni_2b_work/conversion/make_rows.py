"""The device rows for d1-Omni-2B (the phones' test app reads them in its generic mode; the app is not published), made
by the PyTorch-free host (host/d1_host.py) with recording backends (replay.py): every graph call the host makes for a
request, laid out as the LiteRT backends lay it out (host/lrt_backend.py), becomes a row of the bucket's / canvas's /
audio bucket's rows file. The d1-2B rows script with the Omni snapshot, table and loader (records.py), the audio
encoder's rows, and one prep for every media kind.

  $EXPORT conversion/make_rows.py static --set conversion/set_ids.json --out-dir out/rows
      tower rows per canvas (twG<GH>x<GW>.json), audio rows per length bucket (auT<T>.json), the text requests' decide
      rows per bucket (decL<L>.json; a text request = no picture and no clip), the timing rows of the tower and audio
      workloads (tmG24x24.json = 384 px, tmG32x48.json = Box, tmT1000.json = the 10 s audio clip), media_index.json
      (each media request: the buckets its sequences take, the canvases of its pictures, the audio buckets of its
      clips), SHA256SUMS, static_summary.json
  $EXPORT conversion/make_rows.py prep --kind pic|aud|pa --rows-dir <rows> --round-dir <R> --src-tag <leg> --bucket L \
      --name <rows file name> --out-dir <dir> [--sha-file <file>] [--allow-missing]
      the bucket's decide rows of the media requests of the kind (pic = pictures and no clip, aud = a clip (with or
      without pictures), pa = every media request), their embeddings holding the tower and audio encoder outputs of
      the round's legs (every DONE tower leg whose tag starts with <stem>tw and every DONE audio leg whose tag starts
      with <stem>au, <stem> = --src-tag up to "tw" or "au"): the phone's decision graph reads the phone's own picture
      and clip features; kinds pic and pa put the bucket's static text rows first (decL<L>.json)

A media request's decide rows are never static: its embeddings hold the tower's / the audio encoder's output. The
timing rows time the graph's work, which does not depend on the values.
Run with $EXPORT (numpy, tokenizers, Pillow; no torch).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(WORK, "host"))
sys.path.insert(0, os.path.join(WORK, "host"))
SNAP = os.path.join(WORK, "snapshot")
EMBED = os.path.join(WORK, "tables/embed_table.npy")
# the timing workloads of the towers and the audio encoder (the 384 px picture, the 10 s clip and wl_box for the
# Box tower): workload -> its record; each is one tower or audio call
TIMING = {"384px": "wl_384px", "box": "wl_box", "audio_10s": "wl_audio_10s"}


def records_and_media():
    from records import CARD_FILES, TEXTPIC_FILES, TEXTPIC_MEDIA, AUDIO_FILES, AUDIO_MEDIA, load_records

    recs, media = load_records(TEXTPIC_FILES + AUDIO_FILES + CARD_FILES, TEXTPIC_MEDIA + AUDIO_MEDIA)
    return {r["id"]: r for r in recs}, media


def make_host(dec, tow, aud):
    from d1_host import D1Host

    return D1Host(SNAP, dec, tow, embed_table=EMBED, audio=aud)


def run_records(host, backends, by, media, ids):
    """Each request through the host once (the gate's path: open_request, named, run); the recording backends keep the
    calls under the id."""
    from records import open_request

    dec = backends[0]
    for rid in ids:
        for b in backends:
            b.tag = rid
        n0 = len(dec.calls)
        st, qs, items = open_request(by[rid], media)
        named = host.named(qs)
        host.run([(st, list(named.values()), items)])
        for n, c in enumerate(dec.calls[n0:]):
            c["call"] = n


def grouped(calls, key):
    out = {}
    for c in calls:
        out.setdefault(key(c), []).append(c)
    return out


def sums_append(path, pairs):
    if not path:
        return
    with open(path, "a") as fh:
        for s, f in pairs:
            fh.write(f"{s} {f}\n")


def cmd_static(a):
    from replay import (DECIDE_SIG, KIND_AUDIO, KIND_DECIDE, KIND_TOWER, RecordingAudio, RecordingDecide,
                        RecordingTower, audio_spec, decide_spec, tower_spec, write_rows)

    by, media = records_and_media()
    sel = json.load(open(os.path.join(WORK, a.set)))
    ids = sel["records"]
    dec, tow, aud = RecordingDecide(), RecordingTower(), RecordingAudio()
    host = make_host(dec, tow, aud)
    t0 = time.time()
    run_records(host, (dec, tow, aud), by, media, ids)
    media_ids = sorted({c["tag"] for c in tow.calls} | {c["tag"] for c in aud.calls})
    os.makedirs(a.out_dir, exist_ok=True)
    made = "conversion/make_rows.py static"
    written = []
    for c, calls in sorted(grouped(tow.calls, lambda c: c["canvas"]).items()):
        stem = f"twG{c[0]}x{c[1]}"
        ins, outs = tower_spec(c, stem)
        rows = [{"key": f"{x['tag']}/p{j}", "record": x["tag"], "grid": list(x["grid"])} for j, x in enumerate(calls)]
        written += write_rows(os.path.join(a.out_dir, stem + ".json"), KIND_TOWER, "tower", ins, outs, rows,
                              [x["arrays"] for x in calls], {"made_by": made})
    for T, calls in sorted(grouped(aud.calls, lambda c: c["T"]).items()):
        stem = f"auT{T}"
        ins, outs = audio_spec(T, stem)
        rows = [{"key": f"{x['tag']}/a{j}", "record": x["tag"], "frames": x["F"], "tokens": -(x["F"] // -8)}
                for j, x in enumerate(calls)]
        written += write_rows(os.path.join(a.out_dir, stem + ".json"), KIND_AUDIO, "audio", ins, outs, rows,
                              [x["arrays"] for x in calls], {"made_by": made})
    text = [c for c in dec.calls if c["tag"] not in media_ids]
    for L, calls in sorted(grouped(text, lambda c: c["L"]).items()):
        stem = f"decL{L}"
        ins, outs = decide_spec(L, stem)
        rows = [{"key": f"{x['tag']}/c{x['call']}", "record": x["tag"], "call": x["call"], "T": x["T"],
                 "state_len": x["state_len"]} for x in calls]
        written += write_rows(os.path.join(a.out_dir, stem + ".json"), KIND_DECIDE, DECIDE_SIG[L], ins, outs,
                              rows, [x["arrays"] for x in calls], {"made_by": made})
    index = {}
    for x in dec.calls:
        if x["tag"] in media_ids:
            index.setdefault(x["tag"], {"buckets": [], "canvases": [], "audio": []})["buckets"].append(x["L"])
    for x in tow.calls:
        index[x["tag"]]["canvases"].append(list(x["canvas"]))
    for x in aud.calls:
        index[x["tag"]]["audio"].append(x["T"])
    for v in index.values():
        v["kind"] = "both" if v["canvases"] and v["audio"] else ("pic" if v["canvases"] else "aud")
    json.dump({"records": index, "made_by": made}, open(os.path.join(a.out_dir, "media_index.json"), "w"), indent=1)
    # the timing rows: one call each (a tower canvas or an audio bucket), sets of kind single
    wdec, wtow, waud = RecordingDecide(), RecordingTower(), RecordingAudio()
    whost = make_host(wdec, wtow, waud)
    run_records(whost, (wdec, wtow, waud), by, media, list(TIMING.values()))
    timing = {}
    for wl, rid in TIMING.items():
        tc = [c for c in wtow.calls if c["tag"] == rid]
        ac = [c for c in waud.calls if c["tag"] == rid]
        assert len(tc) + len(ac) == 1, (wl, len(tc), len(ac))
        if tc:
            c = tc[0]
            stem = f"tmG{c['canvas'][0]}x{c['canvas'][1]}"
            ins, outs = tower_spec(c["canvas"], stem)
            row = {"key": f"{rid}/p0", "record": rid, "grid": list(c["grid"])}
            kind, sig = KIND_TOWER, "tower"
        else:
            c = ac[0]
            stem = f"tmT{c['T']}"
            ins, outs = audio_spec(c["T"], stem)
            row = {"key": f"{rid}/a0", "record": rid, "frames": c["F"], "tokens": -(c["F"] // -8)}
            kind, sig = KIND_AUDIO, "audio"
        timing.setdefault(stem, (kind, sig, ins, outs, [], [], []))
        timing[stem][4].append(row)
        timing[stem][5].append(c["arrays"])
        timing[stem][6].append({"name": wl, "kind": "single", "rows": [row["key"]]})
    for stem, (kind, sig, ins, outs, rows, arrays, sets) in sorted(timing.items()):
        written += write_rows(os.path.join(a.out_dir, stem + ".json"), kind, sig, ins, outs, rows, arrays,
                              {"sets": sets, "made_by": made})
    with open(os.path.join(a.out_dir, "SHA256SUMS"), "w") as fh:
        for s, f in sorted(written, key=lambda t: t[1]):
            fh.write(f"{s}  {f}\n")
    summary = {"records": len(ids), "media_records": media_ids, "decide_calls": len(dec.calls),
               "text_decide_calls": len(text), "tower_calls": len(tow.calls), "audio_calls": len(aud.calls),
               "per_bucket_text": {str(L): len(v) for L, v in sorted(grouped(text, lambda c: c["L"]).items())},
               "per_bucket_media": {str(L): len(v) for L, v in sorted(grouped(
                   [c for c in dec.calls if c["tag"] in media_ids], lambda c: c["L"]).items())},
               "per_canvas": {f"{c[0]}x{c[1]}": len(v) for c, v in sorted(grouped(tow.calls, lambda c: c["canvas"]).items())},
               "per_audio_bucket": {str(T): len(v) for T, v in sorted(grouped(aud.calls, lambda c: c["T"]).items())},
               "timing": {stem: [s["name"] for s in v[6]] for stem, v in timing.items()},
               "files": len(written), "bytes": sum(os.path.getsize(os.path.join(a.out_dir, f)) for _, f in written),
               "wall_s": round(time.time() - t0, 1)}
    json.dump(summary, open(os.path.join(a.out_dir, "static_summary.json"), "w"), indent=1)
    print(json.dumps(summary))


def leg_sources(rows_dir, round_dir, prefix, rows_head):
    """The round's DONE generic legs whose tag starts with `prefix` and whose rows file starts with `rows_head` (twG /
    auT): {rows file: (tag, GenericRows with its out file)}; a leg stopped before its last row is refused."""
    from replay import GenericRows

    out = {}
    for name in sorted(os.listdir(round_dir)):
        if not (name.startswith(prefix) and name.endswith(".json")):
            continue
        tag = name[:-5]
        try:
            doc = json.load(open(os.path.join(round_dir, name)))
        except Exception:
            continue
        if not isinstance(doc, dict) or doc.get("status") != "DONE" or not str(doc.get("rows_file", "")).startswith(rows_head):
            continue
        o = os.path.join(round_dir, f"out_{tag}.f32")
        if not os.path.exists(o):
            continue
        rows = doc["rows_file"]
        limit = (doc.get("summary") or {}).get("count") or 0
        src = GenericRows(os.path.join(rows_dir, rows), o, rows_limit=limit)
        if limit != len(src.rows):
            raise SystemExit(f"leg {tag}: {limit} of {len(src.rows)} rows ran (stopped early?)")
        out.setdefault(rows, (tag, src))
    return out


def cmd_prep(a):
    import numpy as np

    from replay import (DECIDE_SIG, KIND_DECIDE, GenericRows, RecordingAudio, RecordingDecide, RecordingTower,
                        ReplayAudio, ReplayTower, decide_spec, write_rows)

    t0 = time.time()
    index = json.load(open(os.path.join(a.rows_dir, "media_index.json")))["records"]
    kinds = {"pic": ("pic",), "aud": ("aud", "both"), "pa": ("pic", "aud", "both")}[a.kind]
    want = sorted(r for r, v in index.items() if a.bucket in v["buckets"] and v["kind"] in kinds)
    cut = min([i for i in (a.src_tag.find("tw"), a.src_tag.find("au")) if i >= 0], default=-1)
    if cut < 0:
        raise SystemExit(f"--src-tag {a.src_tag}: no 'tw' or 'au' in the tag (the stem of the round's legs)")
    stem = a.src_tag[:cut]
    tsrc = leg_sources(a.rows_dir, a.round_dir, stem + "tw", "twG")
    asrc = leg_sources(a.rows_dir, a.round_dir, stem + "au", "auT")
    need_t = sorted({f"twG{c[0]}x{c[1]}.json" for r in want for c in index[r]["canvases"]})
    need_a = sorted({f"auT{T}.json" for r in want for T in index[r]["audio"]})
    missing = [n for n in need_t if n not in tsrc] + [n for n in need_a if n not in asrc]
    dropped = []
    if missing and not a.allow_missing:
        raise SystemExit(f"bucket {a.bucket}: no DONE leg of stem {stem} for {missing} in {a.round_dir}")
    if missing:   # the requests whose pictures / clips no leg ran leave the file (the replay then refuses them, by id)
        dropped = [r for r in want if any(f"twG{c[0]}x{c[1]}.json" in missing for c in index[r]["canvases"])
                   or any(f"auT{T}.json" in missing for T in index[r]["audio"])]
        want = [r for r in want if r not in dropped]
        print(f"bucket {a.bucket}: no leg for {missing}: dropped {dropped}", file=sys.stderr)
    used_t = sorted({f"twG{c[0]}x{c[1]}.json" for r in want for c in index[r]["canvases"]})
    used_a = sorted({f"auT{T}.json" for r in want for T in index[r]["audio"]})
    rt = ReplayTower([tsrc[n][1] for n in used_t]) if used_t else None
    ra = ReplayAudio([asrc[n][1] for n in used_a]) if used_a else None
    by, media = records_and_media()
    dec, tow, aud = RecordingDecide(), RecordingTower(rt), RecordingAudio(ra)
    host = make_host(dec, tow, aud)
    run_records(host, (dec, tow, aud), by, media, want)
    calls = [c for c in dec.calls if c["L"] == a.bucket]
    name_stem = a.name[:-5] if a.name.endswith(".json") else a.name
    ins, outs = decide_spec(a.bucket, name_stem)
    rows = [{"key": f"{x['tag']}/c{x['call']}", "record": x["tag"], "call": x["call"], "T": x["T"],
             "state_len": x["state_len"]} for x in calls]
    arrays = [x["arrays"] for x in calls]
    n_text = 0
    if a.kind in ("pic", "pa"):   # the bucket's text rows first (the static decL<L>.json, its slices read and checked)
        tp = os.path.join(a.rows_dir, f"decL{a.bucket}.json")
        if os.path.exists(tp):
            t = GenericRows(tp)
            t_rows, t_arr = [], []
            for r in t.rows:
                d = {}
                for spec in t.inputs:
                    n = int(np.prod(spec["shape"]))
                    mm = np.memmap(os.path.join(t.dir, spec["file"]), dtype="<f4", mode="r")
                    d[spec["name"]] = np.asarray(mm[r["index"] * n:(r["index"] + 1) * n]).reshape(spec["shape"])
                t_arr.append(d)
                t_rows.append({k: v for k, v in r.items() if k not in ("index", "sha256")})
            rows, arrays, n_text = t_rows + rows, t_arr + arrays, len(t_rows)
    if not rows:
        raise SystemExit(f"bucket {a.bucket}: no row of kind {a.kind}")
    os.makedirs(a.out_dir, exist_ok=True)
    written = write_rows(os.path.join(a.out_dir, name_stem + ".json"), KIND_DECIDE, DECIDE_SIG[a.bucket], ins, outs,
                         rows, arrays,
                         {"kind_of_rows": a.kind, "tower_legs": {n: tsrc[n][0] for n in used_t},
                          "audio_legs": {n: asrc[n][0] for n in used_a}, "text_rows_first": n_text,
                          "media_missing": missing, "requests_dropped": dropped, "round_dir": a.round_dir,
                          "made_by": "conversion/make_rows.py prep"})
    sums_append(a.sha_file, written)
    print(json.dumps({"bucket": a.bucket, "kind": a.kind, "name": name_stem + ".json", "rows": len(rows),
                      "text_rows": n_text, "records": want, "tower_legs": {n: tsrc[n][0] for n in used_t},
                      "audio_legs": {n: asrc[n][0] for n in used_a}, "dropped": dropped,
                      "replayed": {"tower": len(rt.used) if rt else 0, "audio": len(ra.used) if ra else 0},
                      "wall_s": round(time.time() - t0, 1)}))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("static")
    s.add_argument("--set", default="conversion/set_ids.json")
    s.add_argument("--out-dir", default=os.path.join(WORK, "out/rows"))
    p = sub.add_parser("prep")
    p.add_argument("--kind", choices=("pic", "aud", "pa"), required=True)
    p.add_argument("--rows-dir", required=True)
    p.add_argument("--round-dir", required=True)
    p.add_argument("--src-tag", required=True)
    p.add_argument("--bucket", type=int, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--sha-file", default="")
    p.add_argument("--allow-missing", action="store_true", help="drop the requests whose canvas / clip no leg ran")
    a = ap.parse_args()
    {"static": cmd_static, "prep": cmd_prep}[a.cmd](a)


if __name__ == "__main__":
    main()
