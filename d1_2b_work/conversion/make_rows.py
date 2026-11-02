"""The device rows for d1-2B (the phones' test app reads them in its generic mode; the app is not published), made by
the PyTorch-free host (host/d1_host.py) with recording backends (replay.py): every graph call the host makes for a
request, laid out as the LiteRT backends lay it out (host/lrt_backend.py), becomes a row of the bucket's / canvas's
rows file.

  $EXPORT conversion/make_rows.py static --set conversion/set_ids.json --out-dir out/rows
      tower rows per canvas (twG<GH>x<GW>.json), the text requests' decide rows per bucket (decL<L>.json), the timing
      rows of the card's workloads (tmL<L>.json, tmG<GH>x<GW>.json) and pic_index.json (the picture requests: the
      canvases their pictures and the buckets their sequences take), SHA256SUMS
  $EXPORT conversion/make_rows.py pic --rows-dir <rows> --round-dir <R> --src-tag <a tower leg> --bucket L \
      --name <rows file name> --out-dir <dir> [--sha-file <file>]
      the picture requests' decide rows of bucket L, their embeddings holding the tower output of the round's tower
      legs (every DONE leg in R whose tag starts like --src-tag up to "tw": one per canvas), so the phone's decision
      graph reads the phone's own picture features

A picture request's decide rows are never static: its embeddings hold the tower's output. The timing rows are the one
exception: they time the graph's work, which does not depend on the values, so their picture positions hold zeros
(written in the file as `picture_features: zeros`).
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
sys.path.insert(0, os.path.join(WORK, "host"))   # the host modules (replay.py is in this folder)
sys.path.insert(0, os.path.join(WORK, "host"))
SNAP = os.path.join(WORK, "snapshot")
EMBED = os.path.join(WORK, "tables/embed_table.npy")
RECORD_FILES = ["set/requests.json", "set/red_arms.json", "set/extra_g96x72.json"]
MEDIA = "set/media_all.json"
WORKLOADS = {"one_question": ["wl_one_question"], "3_questions": ["wl_3_questions"], "5k_state": ["wl_5k_state"],
             "384px": ["wl_384px"], "box": ["wl_box"], "boxes": ["wl_boxes"],
             "64_states": [f"wl_64_states_{i:02d}" for i in range(64)]}   # conversion/bench_host.py


def load_records():
    out = {}
    for p in RECORD_FILES:
        doc = json.load(open(os.path.join(WORK, p)))
        for r in (doc if isinstance(doc, list) else doc["records"]):
            out[r["id"]] = r
    return out


def open_media(media, ids):
    from PIL import Image

    items = []
    for mid in ids or []:
        m = media[mid]
        p = os.path.join(WORK, m["path"])
        data = open(p, "rb").read()
        if m.get("sha256") and hashlib.sha256(data).hexdigest() != m["sha256"]:
            raise SystemExit(f"media {mid}: sha256 mismatch")
        items.append(Image.open(p))
    return items


def run_records(host, dec, tow, recs, ids, media):
    """Each request through the host once (run_gate.py's path); the recording backends keep the calls under the id."""
    for rid in ids:
        req = recs[rid]["request"]
        dec.tag = tow.tag = rid
        n0 = len(dec.calls)
        items = open_media(media, req.get("media"))
        named = host.named(req["questions"])
        host.run([(req.get("state"), list(named.values()), items)])
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
    from d1_host import D1Host
    from replay import (DECIDE_SIG, RecordingDecide, RecordingTower, decide_spec, tower_spec, write_rows)

    recs = load_records()
    media = json.load(open(os.path.join(WORK, MEDIA)))
    sel = json.load(open(os.path.join(WORK, a.set)))
    ids = sel["records"] + sel["extra_canvas"]
    dec, tow = RecordingDecide(), RecordingTower()
    host = D1Host(SNAP, dec, tow, embed_table=EMBED)
    t0 = time.time()
    run_records(host, dec, tow, recs, ids, media)
    pic_ids = sorted({c["tag"] for c in tow.calls})
    os.makedirs(a.out_dir, exist_ok=True)
    written = []
    # tower rows per canvas (every picture of the set)
    for c, calls in sorted(grouped(tow.calls, lambda c: c["canvas"]).items()):
        stem = f"twG{c[0]}x{c[1]}"
        ins, outs = tower_spec(c, stem)
        rows = [{"key": f"{x['tag']}/p{j}", "record": x["tag"], "grid": list(x["grid"])}
                for j, x in enumerate(calls)]
        written += write_rows(os.path.join(a.out_dir, stem + ".json"), "d1-2b_tower", "tower", ins, outs, rows,
                              [x["arrays"] for x in calls], {"made_by": "conversion/make_rows.py static"})
    # decide rows of the text requests per bucket
    text = [c for c in dec.calls if c["tag"] not in pic_ids]
    for L, calls in sorted(grouped(text, lambda c: c["L"]).items()):
        stem = f"decL{L}"
        ins, outs = decide_spec(L, stem)
        rows = [{"key": f"{x['tag']}/c{x['call']}", "record": x["tag"], "call": x["call"], "T": x["T"],
                 "state_len": x["state_len"]} for x in calls]
        written += write_rows(os.path.join(a.out_dir, stem + ".json"), "d1-2b_decide", DECIDE_SIG[L], ins, outs,
                              rows, [x["arrays"] for x in calls], {"made_by": "conversion/make_rows.py static"})
    pic = {}
    for x in dec.calls:
        if x["tag"] in pic_ids:
            pic.setdefault(x["tag"], {"buckets": [], "canvases": []})["buckets"].append(x["L"])
    for x in tow.calls:
        pic[x["tag"]]["canvases"].append(list(x["canvas"]))
    json.dump({"records": pic, "made_by": "conversion/make_rows.py static"},
              open(os.path.join(a.out_dir, "pic_index.json"), "w"), indent=1)
    # the card's workloads (timing; picture positions hold zeros: the tower is timed on its own canvas)
    wdec, wtow = RecordingDecide(), RecordingTower()
    whost = D1Host(SNAP, wdec, wtow, embed_table=EMBED)
    wl_ids = [r for v in WORKLOADS.values() for r in v]
    run_records(whost, wdec, wtow, recs, wl_ids, media)
    by_rec = {wl: [r for r in v] for wl, v in WORKLOADS.items()}
    sets_dec, sets_tow = {}, {}
    for wl, rids in by_rec.items():
        dcalls = [c for c in wdec.calls if c["tag"] in rids]
        tcalls = [c for c in wtow.calls if c["tag"] in rids]
        for c in dcalls:
            sets_dec.setdefault(c["L"], {}).setdefault(wl, []).append(c)
        for c in tcalls:
            sets_tow.setdefault(c["canvas"], {}).setdefault(wl, []).append(c)
    for L, by_wl in sorted(sets_dec.items()):
        stem = f"tmL{L}"
        ins, outs = decide_spec(L, stem)
        calls = [c for wl in by_wl for c in by_wl[wl]]
        rows = [{"key": f"{x['tag']}/c{x['call']}", "record": x["tag"], "call": x["call"], "T": x["T"]} for x in calls]
        sets = [{"name": wl, "kind": "request" if len(v) > 1 else "single",
                 "rows": [f"{x['tag']}/c{x['call']}" for x in v]} for wl, v in by_wl.items()]
        written += write_rows(os.path.join(a.out_dir, stem + ".json"), "d1-2b_decide", DECIDE_SIG[L], ins, outs, rows,
                              [x["arrays"] for x in calls], {"sets": sets, "picture_features": "zeros",
                                                              "made_by": "conversion/make_rows.py static"})
    for c, by_wl in sorted(sets_tow.items()):
        stem = f"tmG{c[0]}x{c[1]}"
        ins, outs = tower_spec(c, stem)
        calls = [x for wl in by_wl for x in by_wl[wl]]
        rows = [{"key": f"{x['tag']}/p{j}", "record": x["tag"], "grid": list(x["grid"])} for j, x in enumerate(calls)]
        keys = {id(x): r["key"] for x, r in zip(calls, rows)}
        sets = [{"name": wl, "kind": "single", "rows": [keys[id(x)] for x in v]} for wl, v in by_wl.items()]
        written += write_rows(os.path.join(a.out_dir, stem + ".json"), "d1-2b_tower", "tower", ins, outs, rows,
                              [x["arrays"] for x in calls], {"sets": sets, "made_by": "conversion/make_rows.py static"})
    with open(os.path.join(a.out_dir, "SHA256SUMS"), "w") as fh:
        for s, f in sorted(written, key=lambda t: t[1]):
            fh.write(f"{s}  {f}\n")
    summary = {"records": len(ids), "picture_records": pic_ids, "tower_calls": len(tow.calls),
               "decide_calls": len(dec.calls), "text_decide_calls": len(text),
               "per_bucket_text": {str(L): len(v) for L, v in sorted(grouped(text, lambda c: c["L"]).items())},
               "per_bucket_picture": {str(L): len(v) for L, v in sorted(grouped(
                   [c for c in dec.calls if c["tag"] in pic_ids], lambda c: c["L"]).items())},
               "per_canvas": {f"{c[0]}x{c[1]}": len(v) for c, v in sorted(grouped(tow.calls, lambda c: c["canvas"]).items())},
               "timing_sets": {f"L{L}": {wl: len(v) for wl, v in by_wl.items()} for L, by_wl in sets_dec.items()},
               "timing_tower_sets": {f"G{c[0]}x{c[1]}": {wl: len(v) for wl, v in by_wl.items()}
                                     for c, by_wl in sets_tow.items()},
               "files": len(written), "bytes": sum(os.path.getsize(os.path.join(a.out_dir, f)) for _, f in written),
               "wall_s": round(time.time() - t0, 1)}
    json.dump(summary, open(os.path.join(a.out_dir, "static_summary.json"), "w"), indent=1)
    print(json.dumps(summary))


def tower_sources(rows_dir, round_dir, prefix):
    """The round's DONE tower legs whose tag starts with `prefix` (one per canvas): GenericRows with their out file."""
    from replay import GenericRows

    out = {}
    for name in sorted(os.listdir(round_dir)):
        if not (name.startswith(prefix) and name.endswith(".json")) or name.endswith(".verdict"):
            continue
        tag = name[:-5]
        rep = os.path.join(round_dir, name)
        try:
            doc = json.load(open(rep))
        except Exception:
            continue
        if doc.get("status") != "DONE" or not str(doc.get("rows_file", "")).startswith("twG"):
            continue
        o = os.path.join(round_dir, f"out_{tag}.f32")
        if not os.path.exists(o):
            continue
        rows = doc["rows_file"]
        limit = (doc.get("summary") or {}).get("count") or 0
        src = GenericRows(os.path.join(rows_dir, rows), o, rows_limit=limit)
        if limit != len(src.rows):
            raise SystemExit(f"tower leg {tag}: {limit} of {len(src.rows)} rows ran (stopped early?)")
        out[rows] = (tag, src)
    return out


def cmd_pic(a):
    from d1_host import D1Host
    from replay import DECIDE_SIG, RecordingDecide, RecordingTower, ReplayTower, decide_spec, write_rows

    t0 = time.time()
    pi = json.load(open(os.path.join(a.rows_dir, "pic_index.json")))["records"]
    want = sorted(r for r, v in pi.items() if a.bucket in v["buckets"])
    if not want:
        raise SystemExit(f"no picture request has a sequence in bucket {a.bucket}")
    prefix = a.src_tag[:a.src_tag.index("tw") + 2] if "tw" in a.src_tag else a.src_tag
    srcs = tower_sources(a.rows_dir, a.round_dir, prefix)
    need = sorted({f"twG{c[0]}x{c[1]}.json" for r in want for c in pi[r]["canvases"]})
    missing = [n for n in need if n not in srcs]
    dropped = []
    if missing and not a.allow_missing:
        raise SystemExit(f"bucket {a.bucket}: no DONE tower leg of prefix {prefix} for {missing} in {a.round_dir}")
    if missing:   # the requests whose pictures no tower leg ran leave the file (the replay then refuses them, by id)
        dropped = [r for r in want if any(f"twG{c[0]}x{c[1]}.json" in missing for c in pi[r]["canvases"])]
        want = [r for r in want if r not in dropped]
        need = [n for n in need if n not in missing]
        print(f"bucket {a.bucket}: no tower leg for {missing}: dropped {dropped}", file=sys.stderr)
    replay = ReplayTower([srcs[n][1] for n in need]) if need else None
    recs = load_records()
    media = json.load(open(os.path.join(WORK, MEDIA)))
    dec, tow = RecordingDecide(), RecordingTower(replay)
    host = D1Host(SNAP, dec, tow, embed_table=EMBED)
    run_records(host, dec, tow, recs, want, media)
    calls = [c for c in dec.calls if c["L"] == a.bucket]
    stem = a.name[:-5] if a.name.endswith(".json") else a.name
    ins, outs = decide_spec(a.bucket, stem)
    rows = [{"key": f"{x['tag']}/c{x['call']}", "record": x["tag"], "call": x["call"], "T": x["T"],
             "state_len": x["state_len"]} for x in calls]
    arrays = [x["arrays"] for x in calls]
    n_text = 0
    if a.with_text:   # the bucket's text rows first (the static decL<L>.json, its slices read and checked)
        import numpy as np

        from replay import GenericRows

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
    os.makedirs(a.out_dir, exist_ok=True)
    written = write_rows(os.path.join(a.out_dir, stem + ".json"), "d1-2b_decide", DECIDE_SIG[a.bucket], ins, outs,
                         rows, arrays,
                         {"picture_features": {n: srcs[n][0] for n in need}, "text_rows_first": n_text,
                          "pictures_missing": missing, "requests_dropped": dropped,
                          "round_dir": a.round_dir, "made_by": "conversion/make_rows.py pic"})
    sums_append(a.sha_file, written)
    print(json.dumps({"bucket": a.bucket, "name": stem + ".json", "rows": len(rows), "text_rows": n_text, "records": want,
                      "tower_legs": {n: srcs[n][0] for n in need}, "replayed": len(replay.used) if replay else 0, "dropped": dropped,
                      "wall_s": round(time.time() - t0, 1)}))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("static")
    s.add_argument("--set", default="conversion/set_ids.json")
    s.add_argument("--out-dir", default=os.path.join(WORK, "out/rows"))
    p = sub.add_parser("pic")
    p.add_argument("--rows-dir", required=True)
    p.add_argument("--round-dir", required=True)
    p.add_argument("--src-tag", required=True)
    p.add_argument("--bucket", type=int, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--sha-file", default="")
    p.add_argument("--with-text", action="store_true", help="the bucket's static text rows first, in the same file")
    p.add_argument("--allow-missing", action="store_true", help="drop the requests whose canvas no tower leg ran")
    a = ap.parse_args()
    {"static": cmd_static, "pic": cmd_pic}[a.cmd](a)


if __name__ == "__main__":
    main()
