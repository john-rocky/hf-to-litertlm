"""The rows a phone runs, made by the PyTorch-free host (host/decide_host) with recording backends (replay.py): every tower call the host makes for a record, laid out as the LiteRT backends lay it
out, becomes a row of the bucket's rows file. A decide row is never static: its embeddings hold the run's own tower output (the phone's decide reads the phone's picture features), so decide rows are
made after the tower runs, from their outputs (`pic`).

  $EXPORT conversion/make_rows.py static [--out-dir out/rows] [--ids 1,2,3]
      twN<N>.json + inputs: the device set's pictures per patch bucket; tmN<N>.json: the fixed workload's pictures per bucket with their timing sets;
      records.npz + records.json (ids, grids, the rows' keys); SHA256SUMS
  $EXPORT conversion/make_rows.py pic --rows-dir <rows> --round-dir <R> --src-prefix <tower run prefix> --bucket L --name gdL512.json --out-dir <dir> [--sha-file <file>] [--ids 1,2,3]
      the decide rows of bucket L of the records whose both pictures a DONE tower run (tag starts with the prefix) answered
  $EXPORT conversion/make_rows.py timing --rows-dir <rows> --round-dir <R> --src-prefix <tower run prefix> --out-dir <dir> [--sha-file <file>]
      tmL<L>.json with the workload's timing sets, the picture positions holding that tower run's output (timing is about the graph's work; the file says which run)
Numbers only, under out/: they hold the pictures' values, so they never go into a repository. Needs numpy, tokenizers, Pillow; no torch."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WORK, "host"))
from decide_host import formulas as F  # noqa: E402
from decide_host import host as H  # noqa: E402
import replay as RP  # noqa: E402

SNAP, BENCH = os.path.join(WORK, "snapshot"), os.path.join(WORK, "bench")


def bench_rows():
    return {r["item_id"]: r for r in map(json.loads, open(os.path.join(BENCH, "data/jevocrbench.jsonl"), encoding="utf-8"))}


def device_ids():
    return json.load(open(os.path.join(WORK, "conversion/sets/set_device.json")))["item_ids"]


def workload():
    """The fixed workload (conversion/sets/ladder_workload.json): {name: [item ids]}."""
    w = json.load(open(os.path.join(WORK, "conversion/sets/ladder_workload.json")))
    return {"text": [w["text"]["item_id"]], "formula": [w["formula"]["item_id"]], "table": [w["table"]["item_id"]]}


def record_inputs(host, tow, dec, rows_all, i):
    """One record through the host with recording backends -> its record dict; the calls stay in tow / dec."""
    r = rows_all[i]
    tow.tag = dec.tag = i
    out = host.score(os.path.join(BENCH, r["image_path"]), os.path.join(BENCH, r["render_image_path"]), str(r["ocr_text"] or ""))
    assert out["status"] == "ok", (i, out)
    ids = host.token_row(str(r["ocr_text"] or ""), out["N"][0] // 4, out["N"][1] // 4)
    return {"item_id": i, "T": out["T"], "N": out["N"], "grids": out["grids"], "ids": ids, "ids_sha": out["ids_sha"],
            "category": r["category"], "label": r["label"]}


def tower_rows_file(calls, N, stem, out_dir, extra=None, keys=None):
    ins, outs = RP.tower_spec(N, stem)
    rows = [{"key": keys[j] if keys else f"{c['tag']}/p{c['pic']}", "record": c["tag"], "picture": c["pic"], "n": c["n"], "T": c["n"] // 4}
            for j, c in enumerate(calls)]
    return RP.write_rows(os.path.join(out_dir, stem + ".json"), "tower", RP.tower_sig(N), ins, outs, rows, [c["arrays"] for c in calls], extra)


def cmd_static(a):
    t0 = time.time()
    rows_all = bench_rows()
    tow, dec = RP.RecordingTower(), RP.RecordingDecide()
    host = H.Host(tow, dec, SNAP)
    written = []
    out_dir = a.out_dir
    os.makedirs(out_dir, exist_ok=True)
    wl = workload()
    wl_ids = sorted({i for v in wl.values() for i in v})
    ids = [int(x) for x in a.ids.split(",")] if a.ids else device_ids()
    records = {}
    per_bucket = {}
    # the device set, one bucket's rows at a time is not possible (a record's two pictures fall in two buckets): the calls of all records
    # are kept, 2.5 GB
    for i in ids:
        n0 = len(tow.calls)
        records[i] = record_inputs(host, tow, dec, rows_all, i)
        for j, c in enumerate(tow.calls[n0:]):
            c["pic"] = j
    device_calls = tow.calls
    for N in RP.N_BUCKETS:
        calls = [c for c in device_calls if c["N"] == N]
        per_bucket[f"N{N}"] = len(calls)
        if calls:
            written += tower_rows_file(calls, N, f"twN{N}", out_dir, {"made_by": "conversion/make_rows.py static"})
    # the fixed workload: timing rows per bucket
    wtow, wdec = RP.RecordingTower(), RP.RecordingDecide()
    whost = H.Host(wtow, wdec, SNAP)
    wrec = {}
    for i in wl_ids:
        n0 = len(wtow.calls)
        wrec[i] = record_inputs(whost, wtow, wdec, rows_all, i)
        for j, c in enumerate(wtow.calls[n0:]):
            c["pic"] = j
    wcalls = list(wtow.calls)
    sets_tow = {}
    for name, members in wl.items():
        for c in wcalls:
            if c["tag"] in members:
                sets_tow.setdefault(c["N"], {}).setdefault(name, []).append(f"{c['tag']}/p{c['pic']}")
    for N, by in sorted(sets_tow.items()):
        calls = [c for c in wcalls if c["N"] == N]
        sets = [{"name": name, "kind": "request" if len(v) > 1 else "single", "rows": v} for name, v in by.items()]
        written += tower_rows_file(calls, N, f"tmN{N}", out_dir, {"sets": sets, "made_by": "conversion/make_rows.py static"})
    all_recs = {**records, **wrec}
    np.savez(os.path.join(out_dir, "records.npz"), **{f"ids_{i}": r["ids"] for i, r in all_recs.items()})
    json.dump({"device": ids, "workload": wl, "records": {str(i): {k: v for k, v in r.items() if k != "ids"} for i, r in all_recs.items()}},
              open(os.path.join(out_dir, "records.json"), "w"), indent=0)
    summary = {"records": len(ids), "workload_records": wl_ids, "tower_calls": len(device_calls), "per_bucket": per_bucket,
               "T_buckets": {f"L{L}": sum(RP.bucket_of(RP.L_BUCKETS, r["T"]) == L for r in records.values()) for L in RP.L_BUCKETS},
               "timing_tower_sets": {f"N{N}": list(by) for N, by in sets_tow.items()}, "files": len(written),
               "bytes": sum(os.path.getsize(os.path.join(out_dir, f)) for _, f in written), "wall_s": round(time.time() - t0, 1),
               "torch_imported": "torch" in sys.modules}
    json.dump(summary, open(os.path.join(out_dir, "static_summary.json"), "w"), indent=1)
    with open(os.path.join(out_dir, "SHA256SUMS"), "w") as fh:
        for s, f in sorted(written, key=lambda t: t[1]):
            fh.write(f"{s}  {f}\n")
    print(json.dumps(summary))


def tower_sources(rows_dir, round_dir, prefix, check_inputs=False):
    """Every DONE tower leg of `round_dir` whose tag starts with `prefix`: {row key: (GenericRows, position)}, and the legs' tags."""
    index, legs = {}, []
    for name in sorted(os.listdir(round_dir)):
        if not (name.startswith(prefix) and name.endswith(".json")):
            continue
        tag = name[:-5]
        try:
            doc = json.load(open(os.path.join(round_dir, name)))
        except Exception:
            continue
        if not isinstance(doc, dict) or doc.get("status") != "DONE":
            continue
        out = os.path.join(round_dir, f"out_{tag}.f32")
        if not os.path.exists(out):
            continue
        limit = (doc.get("summary") or {}).get("count") or 0
        rp = os.path.join(rows_dir, doc.get("rows_file", ""))
        if not os.path.isfile(rp):     # a decide leg's rows live in the round's gen/: not a tower leg
            continue
        src = RP.GenericRows(rp, out, check_inputs=check_inputs, rows_limit=limit)
        if src.doc.get("graph_kind") != "tower":
            continue
        legs.append(tag)
        for pos in range(limit):
            index.setdefault(src.rows[pos]["key"], (src, pos))
    return index, legs


def decide_arrays(rec, ids, table, rows_a, rows_b):
    """A record's decide inputs at its bucket, the same as the backend makes them from the same tower rows."""
    T, L = rec["T"], RP.bucket_of(RP.L_BUCKETS, rec["T"])
    embeds = F.fill_embeds(ids, T, table, [rows_a, rows_b])[0][0]
    pos3 = F.mrope_planes(ids, [tuple(rec["grids"][0]), tuple(rec["grids"][1])])
    cos, sin = F.mrope_tables(pos3)
    return RP.decide_inputs(embeds, cos[0], sin[0], pos3)


def cmd_pic(a):
    t0 = time.time()
    meta = json.load(open(os.path.join(a.rows_dir, "records.json")))
    arrs = np.load(os.path.join(a.rows_dir, "records.npz"))
    want = [int(x) for x in a.ids.split(",")] if a.ids else meta["device"]
    recs = {i: meta["records"][str(i)] for i in want}
    index, legs = tower_sources(a.rows_dir, a.round_dir, a.src_prefix)
    table = F.Bf16Table(os.path.join(SNAP, "model.safetensors"), H.EMBED_KEY)
    rows, arrays, dropped = [], [], []
    for i in sorted(recs, key=lambda i: (recs[i]["T"], i)):
        rec = recs[i]
        if RP.bucket_of(RP.L_BUCKETS, rec["T"]) != a.bucket:
            continue
        ka, kb = f"{i}/p0", f"{i}/p1"
        if ka not in index or kb not in index:
            dropped.append(i)
            continue
        (sa, pa), (sb, pb) = index[ka], index[kb]
        ra = sa.output(pa, "rows")[0, :rec["N"][0] // 4]
        rb = sb.output(pb, "rows")[0, :rec["N"][1] // 4]
        L, arr = decide_arrays(rec, arrs[f"ids_{i}"], table, ra, rb)
        assert L == a.bucket
        rows.append({"key": f"{i}/d", "record": i, "T": rec["T"], "N": rec["N"], "tower_legs": [os.path.basename(sa.out_path), os.path.basename(sb.out_path)]})
        arrays.append(arr)
    if not rows:
        raise SystemExit(f"bucket {a.bucket}: no record whose pictures a DONE leg of prefix {a.src_prefix} answered (dropped {dropped})")
    stem = a.name[:-5] if a.name.endswith(".json") else a.name
    ins, outs = RP.decide_spec(a.bucket, stem)
    written = RP.write_rows(os.path.join(a.out_dir, stem + ".json"), "decide", RP.decide_sig(a.bucket), ins, outs, rows, arrays,
                            {"picture_features": legs, "records_dropped": dropped, "round_dir": a.round_dir, "made_by": "conversion/make_rows.py pic"})
    if a.sha_file:
        with open(a.sha_file, "a") as fh:
            for s, f in written:
                fh.write(f"{s} {f}\n")
    print(json.dumps({"bucket": a.bucket, "name": stem + ".json", "rows": len(rows), "dropped": dropped, "tower_legs": legs, "wall_s": round(time.time() - t0, 1)}))


def cmd_timing(a):
    """tmL<L>.json: the workload's decide calls, the picture positions from a tower run of the same rows."""
    t0 = time.time()
    meta = json.load(open(os.path.join(a.rows_dir, "records.json")))
    arrs = np.load(os.path.join(a.rows_dir, "records.npz"))
    wl = meta["workload"]
    index, legs = tower_sources(a.rows_dir, a.round_dir, a.src_prefix)
    table = F.Bf16Table(os.path.join(SNAP, "model.safetensors"), H.EMBED_KEY)
    by_bucket = {}
    for name, members in wl.items():
        for i in members:
            rec = meta["records"][str(i)]
            (sa, pa), (sb, pb) = index[f"{i}/p0"], index[f"{i}/p1"]
            ra = sa.output(pa, "rows")[0, :rec["N"][0] // 4]
            rb = sb.output(pb, "rows")[0, :rec["N"][1] // 4]
            L, arr = decide_arrays(rec, arrs[f"ids_{i}"], table, ra, rb)
            by_bucket.setdefault(L, []).append((name, i, rec, arr))
    written = []
    groups = [(f"tmL{L}", L, items) for L, items in sorted(by_bucket.items())]
    for stem, L, items in groups:
        if not items:
            continue
        ins, outs = RP.decide_spec(L, stem)
        rows = [{"key": f"{i}/d", "record": i, "T": rec["T"], "N": rec["N"]} for _, i, rec, _ in items]
        sets = {}
        for name, i, _, _ in items:
            sets.setdefault(name, []).append(f"{i}/d")
        written += RP.write_rows(os.path.join(a.out_dir, stem + ".json"), "decide", RP.decide_sig(L), ins, outs, rows, [x[3] for x in items],
                                 {"sets": [{"name": n, "kind": "request" if len(v) > 1 else "single", "rows": v} for n, v in sets.items()],
                                  "picture_features": legs, "made_by": "conversion/make_rows.py timing"})
    if a.sha_file:
        with open(a.sha_file, "a") as fh:
            for s, f in written:
                fh.write(f"{s}  {f}\n")
    print(json.dumps({"buckets": sorted(by_bucket), "tower_legs": legs, "wall_s": round(time.time() - t0, 1)}))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("static")
    s.add_argument("--out-dir", default=os.path.join(WORK, "out/rows"))
    s.add_argument("--ids", default="", help="comma list: these records in place of the device set")
    p = sub.add_parser("pic")
    p.add_argument("--rows-dir", required=True)
    p.add_argument("--round-dir", required=True)
    p.add_argument("--src-prefix", required=True)
    p.add_argument("--bucket", type=int, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--sha-file", default="")
    p.add_argument("--ids", default="")
    t = sub.add_parser("timing")
    t.add_argument("--rows-dir", required=True)
    t.add_argument("--round-dir", required=True)
    t.add_argument("--src-prefix", required=True)
    t.add_argument("--out-dir", required=True)
    t.add_argument("--sha-file", default="")
    a = ap.parse_args()
    {"static": cmd_static, "pic": cmd_pic, "timing": cmd_timing}[a.cmd](a)


if __name__ == "__main__":
    main()
