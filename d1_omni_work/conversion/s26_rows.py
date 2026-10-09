"""Round 5: the input files of the S26 gate app (android/d1omni_gate), cut from the oracle (ref/records_ref.json
version 2: its ids, markers, K, question type and prefix length per question; the media prefixes from ref/npz/<id>.npz
`prefix`). No tokenizer run: the oracle already holds the provider's ids.

    cd d1_omni_work; ~/venvs/lt094dev/bin/python scripts/s26_rows.py          # -> device/

Files (device/; a file that already exists with other content is refused):
  rows_L{128,256,512,2048,4096}.json  every question whose smallest bucket is L (P + n <= L, the host's bucket_for over
      128..4096 = the graph the host sends the row to), in oracle order; L1024 holds no row (no file):
      {"L", "hidden": 1024, "pad_id": 0, "rows": [{"key": "<record>/<qid>", "ids": [...], "markers": [...],
       "qtype": 0 | 1 | 2, "P": <prefix rows>, "prefix_file": "prefix_<record>.f32" | null, "K": <options>}]}
  rows_L256_sub.json  the L256 gate rows for the phone: the first 100 rows of rows_L256.json (it has 95 = all of them),
      then the control row red_arm_000/answer, the oracle's near-tie questions and the media questions that fit L256
      (P + n <= 256) but whose smallest bucket is 128 (the L256 graph runs any row of up to 256 positions)
  rows_L128_sub.json  the first 100 rows of rows_L128.json (+ red_arm_000/answer when it is not among them)
  timing_rows.json  the card's workloads per graph L (the app runs the sets whose L equals the graph's L):
      L128: (a) card_text/refund, (b) card_text's 3 questions (a request);
      L256: (a) and (b) again, (c) own_fiveq_09's 5 questions (a request; 132-148 ids = smallest bucket 256),
            (e) img_dogs_01/count (image prefix P 84), (f) aud_01/topic (audio prefix P 121);
      L4096: (d) long_3400/component (3,467 ids)
  prefix_<record>.f32  little-endian float32 [P, 1024] = the oracle npz's `prefix` (the vision / audio graph output the
      provider fed its text encoder), one file per media record that a rows file names
  rows_manifest.json  bytes + sha256 of every file above and of the sources; the row-count table per bucket

Round 9 (`--r9`; the files above are only read):
    cd d1_omni_work; ~/venvs/lt094dev/bin/python scripts/s26_rows.py --r9                 # -> device/ (r9 files)
  timing_rows_r9.json  the text timing sets of round 9 per graph L: L128 (a) (b); L256 (a) (b) (c); L512 (g)
      tv4_009/answer; L2048 (g) own_long_log_10/resolved; L4096 (d) long_3400/component ((g) = round 4's one call of the
      bucket's median row)
  generic rows files (the gate app's mode=generic / generic_timing; app GenericCodec): {kind "generic", graph_kind,
      signature, inputs [{name, dtype, shape, file}], outputs [{name, dtype, shape}], rows [{key, index, ...}],
      sets [{name, kind, rows: [key]}]}; an input file holds one little-endian slice per row index, back to back:
    g9_vt.json + g9_vt_{pixels,pos,mask}.f32  the vision tower (signature vision_tower): the 13 crops of the 7 image
      records (oracle order, tiles row-major then the thumbnail), host/d1_vision_host.py tower_inputs; every crop's
      pixels / pos / mask bit-equal to the oracle npz's pixel_values / pos_resized / pixel_attention_mask (= the
      provider's preprocess, float resize path) and its grid to spatial_shapes, else the script stops;
      set v_one_crop = img_dogs_01/c0
    g9_pjt.json + g9_pjt_soft.f32  the projector (signature projector), timing only: img_dogs_01/c0's soft = the
      oracle's tower features (npz tower_last_hidden_state, real rows) through the host unshuffle; set p_one_crop.
      The projector's gate rows are made during the phone run from the phone's own tower features
      (scripts/s26_score.py soft)
    g9_au.json + g9_au_{mel,mel_valid,v1,v2,v3}.f32  the audio graph at T1001 (signature audio_1001): the 6 clips of
      at most 10 s (aud_01..03, aud_reservation_01, aud_weather_02, aud_food_03), host/d1_audio_host.py prepare
      (float32 mel); frames and P equal to the oracle's, the mel's distance to the oracle's recorded; set f_audio_clip
      = aud_01
    g9_au2.json + g9_au2_*.f32  the audio graph at T2001 (signature audio_2001): card_topic (10.435 s)
  rows_manifest_r9.json  bytes + sha256 of every r9 file, the checks above
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

K = Path(__file__).resolve().parents[1]
ORACLE = K / "ref/records_ref.json"
NPZ = K / "ref/npz"
OUT = K / "device"
BUCKETS = (128, 256, 512, 1024, 2048, 4096)
D = 1024
PAD_ID = 0
QTYPES = {"choice": 0, "score": 1, "noul": 2}
SUB_HEAD = 100
CONTROL = "red_arm_000/answer"
TIMING = {
    128: [("a_one_question", "single", ["card_text/refund"]),
          ("b_three_questions", "request", ["card_text/refund", "card_text/team", "card_text/urgency"])],
    256: [("a_one_question", "single", ["card_text/refund"]),
          ("b_three_questions", "request", ["card_text/refund", "card_text/team", "card_text/urgency"]),
          ("c_five_questions", "request", ["own_fiveq_09/signed", "own_fiveq_09/where", "own_fiveq_09/damaged",
                                           "own_fiveq_09/request_followed", "own_fiveq_09/photo"]),
          ("e_image_question", "single", ["img_dogs_01/count"]),
          ("f_audio_question", "single", ["aud_01/topic"])],
    4096: [("d_state_3k4", "single", ["long_3400/component"])],
}
PROTOCOL = ("warm-up calls, then timed rounds (default 5 + 20; a request set = its rows back to back per round); every "
            "call's [device wall clock ms at its start, ms write + run + read, ms run only]; write = the six inputs, "
            "read = the whole scores output")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 24):
            h.update(block)
    return h.hexdigest()


def write_bytes(path: Path, data: bytes) -> Path:
    if path.exists():
        assert path.read_bytes() == data, f"{path} exists with other content (move it away first)"
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)
    return path


def write_json(path: Path, doc) -> Path:
    return write_bytes(path, (json.dumps(doc, separators=(",", ":")) + "\n").encode())


def smallest_bucket(n: int) -> int:
    for L in BUCKETS:
        if n <= L:
            return L
    raise ValueError(f"a row of {n} positions is longer than the largest bucket")


def main() -> int:
    oracle = json.loads(ORACLE.read_text())
    assert oracle.get("version") == 2, f"oracle version {oracle.get('version')} (round 5 reads version 2)"
    oracle_sha = sha256(ORACLE)
    rows, info = [], {}
    for rec in oracle["records"]:
        for q in rec["questions"]:
            P, n = int(q.get("prefix") or 0), len(q["ids"])
            assert int(rec.get("prefix") or 0) == P, (rec["id"], q["qid"])
            assert len(q["markers"]) >= int(q["K"]) and all(0 <= m < n for m in q["markers"][: int(q["K"])]), (rec["id"], q["qid"])
            key = f"{rec['id']}/{q['qid']}"
            row = {"key": key, "ids": [int(x) for x in q["ids"]], "markers": [int(m) for m in q["markers"]],
                   "qtype": QTYPES[q["type"]], "P": P, "prefix_file": f"prefix_{rec['id']}.f32" if P else None,
                   "K": int(q["K"])}
            assert PAD_ID not in row["ids"][1:] and row["ids"][0] == 1, f"{key}: bos first, no pad id inside the row"
            rows.append(row)
            info[key] = {"id": rec["id"], "mode": rec["mode"], "bucket": smallest_bucket(P + n), "near_tie": bool(q["near_tie"]),
                         "oracle_bucket": int(q["bucket"])}
            assert info[key]["bucket"] == info[key]["oracle_bucket"], (key, info[key])
    by_bucket = {L: [r for r in rows if info[r["key"]]["bucket"] == L] for L in BUCKETS}
    head = {"hidden": D, "pad_id": PAD_ID, "source": {"oracle": "ref/records_ref.json", "version": 2, "sha256": oracle_sha}}
    written = {}

    def put(name, L, rlist, extra=None):
        doc = {"L": L, **head, **(extra or {}), "rows": rlist}
        written[name] = (write_json(OUT / name, doc), len(rlist))

    for L in BUCKETS:
        if by_bucket[L]:
            put(f"rows_L{L}.json", L, by_bucket[L])
    # the L256 gate rows of the phone
    r256 = by_bucket[256]
    sub = list(r256[:SUB_HEAD])
    have = {r["key"] for r in sub}
    added = {"control": [], "near_tie": [], "media": []}
    fits256 = [r for r in rows if r["P"] + len(r["ids"]) <= 256]
    for r in fits256:
        k = r["key"]
        if k in have:
            continue
        why = ("control" if k == CONTROL else "near_tie" if info[k]["near_tie"] else
               "media" if info[k]["mode"] != "text" else None)
        if why:
            sub.append(r)
            have.add(k)
            added[why].append(k)
    assert CONTROL in have
    put("rows_L256_sub.json", 256, sub, {"subset": (
        f"the first {SUB_HEAD} rows of rows_L256.json ({min(SUB_HEAD, len(r256))} of {len(r256)}), then the rows that fit "
        f"L256 but whose smallest bucket is 128: the control row, the oracle's near-tie questions, the media questions"),
        "added": added})
    r128 = by_bucket[128]
    sub128 = list(r128[:SUB_HEAD])
    if CONTROL not in {r["key"] for r in sub128}:
        sub128.append(next(r for r in r128 if r["key"] == CONTROL))
    put("rows_L128_sub.json", 128, sub128, {"subset": f"the first {SUB_HEAD} rows of rows_L128.json + {CONTROL}"})
    # timing sets
    by_key = {r["key"]: r for r in rows}
    sets = []
    for L, defs in TIMING.items():
        for name, kind, keys in defs:
            rl = [by_key[k] for k in keys]
            assert all(r["P"] + len(r["ids"]) <= L for r in rl), (name, L)
            sets.append({"name": name, "L": L, "kind": kind,
                         "smallest_bucket": sorted({info[k]["bucket"] for k in keys}), "rows": rl})
    written["timing_rows.json"] = (write_json(OUT / "timing_rows.json", {**head, "protocol": PROTOCOL, "sets": sets}), len(sets))
    # prefix files of every media record a rows file names
    need = {}
    for name, (p, _) in list(written.items()):
        doc = json.loads(p.read_text())
        for r in (doc.get("rows") or [x for s in doc.get("sets", []) for x in s["rows"]]):
            if r["prefix_file"]:
                need.setdefault(r["prefix_file"], (info[r["key"]]["id"], r["P"]))
    prefix_files = {}
    for fname, (rid, P) in sorted(need.items()):
        npz = NPZ / f"{rid}.npz"
        with np.load(npz) as z:
            pre = np.asarray(z["prefix"])
        assert pre.dtype == np.float32 and pre.shape == (P, D), (rid, pre.dtype, pre.shape, P)
        assert np.isfinite(pre).all(), rid
        p = write_bytes(OUT / fname, pre.astype("<f4").tobytes())
        prefix_files[fname] = {"record": rid, "P": P, "bytes": p.stat().st_size, "sha256": sha256(p),
                               "npz": f"ref/npz/{rid}.npz", "npz_sha256": sha256(npz)}
    fit = {str(L): sum(r["P"] + len(r["ids"]) <= L for r in rows) for L in BUCKETS}
    table = {str(L): {"rows_smallest_bucket": len(by_bucket[L]),
                      "text": sum(info[r["key"]]["mode"] == "text" for r in by_bucket[L]),
                      "image": sum(info[r["key"]]["mode"] == "image" for r in by_bucket[L]),
                      "audio": sum(info[r["key"]]["mode"] == "audio" for r in by_bucket[L]),
                      "near_tie": sum(info[r["key"]]["near_tie"] for r in by_bucket[L]), "rows_fit": fit[str(L)]}
             for L in BUCKETS}
    assert sum(v["rows_smallest_bucket"] for v in table.values()) == len(rows) == 448
    manifest = {
        "step": "round 5: S26 gate app inputs (scripts/s26_rows.py)",
        "sources": {"ref/records_ref.json": {"bytes": ORACLE.stat().st_size, "sha256": oracle_sha, "version": 2}},
        "table": table,
        "rows_L256_sub_added": added,
        "timing_sets": {f"{s['name']}@L{s['L']}": {"kind": s["kind"], "rows": len(s["rows"]),
                                                    "keys": [r["key"] for r in s["rows"]],
                                                    "positions": [r["P"] + len(r["ids"]) for r in s["rows"]],
                                                    "smallest_bucket": s["smallest_bucket"]} for s in sets},
        "files": {name: {"rows_or_sets": n, "bytes": p.stat().st_size, "sha256": sha256(p)} for name, (p, n) in written.items()},
        "prefix_files": prefix_files,
    }
    (OUT / "rows_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({"table": table, "files": {k: v["rows_or_sets"] for k, v in manifest["files"].items()},
                      "rows_L256_sub_added": {k: len(v) for k, v in added.items()},
                      "prefix_files": {k: [v["P"], v["bytes"]] for k, v in prefix_files.items()},
                      "timing_sets": {k: [v["rows"], v["positions"]] for k, v in manifest["timing_sets"].items()}}, indent=1))
    return 0


# ---------------------------------------------------------------- round 9

R9_TIMING = {
    128: [("a_one_question", "single", ["card_text/refund"]),
          ("b_three_questions", "request", ["card_text/refund", "card_text/team", "card_text/urgency"])],
    256: [("a_one_question", "single", ["card_text/refund"]),
          ("b_three_questions", "request", ["card_text/refund", "card_text/team", "card_text/urgency"]),
          ("c_five_questions", "request", ["own_fiveq_09/signed", "own_fiveq_09/where", "own_fiveq_09/damaged",
                                           "own_fiveq_09/request_followed", "own_fiveq_09/photo"])],
    512: [("g_one_call", "single", ["tv4_009/answer"])],
    2048: [("g_one_call", "single", ["own_long_log_10/resolved"])],
    4096: [("d_state_3k4", "single", ["long_3400/component"])],
}
IMAGE_IDS = ("card_cats", "img_dogs_01", "img_cat_02", "img_bike_03", "img_01", "img_02", "img_03")
AUDIO_SETS = {"g9_au": (1001, ("aud_01", "aud_02", "aud_03", "aud_reservation_01", "aud_weather_02", "aud_food_03"), "aud_01"),
              "g9_au2": (2001, ("card_topic",), "card_topic")}


def write_stacked(path: Path, slices) -> Path:
    """float32 / int32 arrays of one shape -> one file, each slice little-endian, back to back (GenericCodec)."""
    a = [np.ascontiguousarray(s) for s in slices]
    assert a and all(x.shape == a[0].shape and x.dtype == a[0].dtype for x in a), [x.shape for x in a]
    assert a[0].dtype in (np.float32, np.int32), a[0].dtype
    return write_bytes(path, b"".join(x.astype(a[0].dtype.newbyteorder("<")).tobytes() for x in a))


def spec(name, arr, file=None):
    d = {"name": name, "dtype": {np.dtype(np.float32): "float32", np.dtype(np.int32): "int32"}[np.asarray(arr).dtype],
         "shape": [int(v) for v in np.asarray(arr).shape]}
    if file:
        d["file"] = file
    return d


def main_r9() -> int:
    sys.path.insert(0, str(K / "scripts"))
    sys.path.insert(0, str(K / "host"))
    import d1_src as S
    import d1_vision_host as V
    import d1_audio_host as AH

    oracle = json.loads(ORACLE.read_text())
    assert oracle.get("version") == 2 and oracle["load"]["resize_path"]["path"] == "float", "oracle v2 (float resize path)"
    oracle_sha = sha256(ORACLE)
    fx = {r["id"]: r for r in json.loads((K / "fixtures/requests.json").read_text())["records"]}
    rows_by_key, bucket_of = {}, {}
    for rec in oracle["records"]:
        for q in rec["questions"]:
            P = int(q.get("prefix") or 0)
            key = f"{rec['id']}/{q['qid']}"
            rows_by_key[key] = {"key": key, "ids": [int(x) for x in q["ids"]], "markers": [int(m) for m in q["markers"]],
                                "qtype": QTYPES[q["type"]], "P": P, "prefix_file": f"prefix_{rec['id']}.f32" if P else None,
                                "K": int(q["K"])}
            bucket_of[key] = smallest_bucket(P + len(q["ids"]))
    head = {"source": {"oracle": "ref/records_ref.json", "version": 2, "sha256": oracle_sha}}
    written, checks = {}, {}

    # text timing sets of round 9
    sets = []
    for L, defs in R9_TIMING.items():
        for name, kind, keys in defs:
            rl = [rows_by_key[k] for k in keys]
            assert all(r["P"] + len(r["ids"]) <= L for r in rl), (name, L)
            sets.append({"name": name, "L": L, "kind": kind, "smallest_bucket": sorted({bucket_of[k] for k in keys}), "rows": rl})
    written["timing_rows_r9.json"] = (write_json(OUT / "timing_rows_r9.json", {"hidden": D, "pad_id": PAD_ID, **head,
                                                                               "protocol": PROTOCOL, "sets": sets}), len(sets))
    checks["timing_sets"] = {f"{s['name']}@L{s['L']}": {"keys": [r["key"] for r in s["rows"]],
                                                       "positions": [r["P"] + len(r["ids"]) for r in s["rows"]],
                                                       "smallest_bucket": s["smallest_bucket"]} for s in sets}

    # vision tower: 13 crops, the host's inputs bit-equal to the provider's preprocess (oracle npz)
    table = V.read_position_table(S.WEIGHTS)
    vt_rows, pixels, pos, mask, vchk, records = [], [], [], [], [], {}
    for rid in IMAGE_IDS:
        rec = fx[rid]
        path = K / rec["media"]["ref"]
        assert S.sha256_file(path) == rec["media"]["sha256"], path
        crops, plan = V.crops_of(V.load_image(path))
        with np.load(NPZ / f"{rid}.npz") as z:
            npz = {k: np.asarray(z[k]) for k in ("pixel_values", "pixel_attention_mask", "pos_resized", "spatial_shapes", "prefix")}
        assert len(crops) == npz["pixel_values"].shape[0], (rid, len(crops))
        P = 0
        for i, c in enumerate(crops):
            crop = V.to_patches(c)
            x = V.tower_inputs(crop, table)
            h, w = crop["grid"]
            eq = {"pixels": bool(np.array_equal(x["pixels"][0], npz["pixel_values"][i]) and x["pixels"].dtype == np.float32),
                  "pos": bool(np.array_equal(x["pos"][0], npz["pos_resized"][i]) and x["pos"].dtype == np.float32),
                  "mask": bool(np.array_equal(x["mask"][0], npz["pixel_attention_mask"][i].astype(np.float32))),
                  "grid": [h, w] == [int(v) for v in npz["spatial_shapes"][i]]}
            assert all(eq.values()), (rid, i, eq)
            key = f"{rid}/c{i}"
            vt_rows.append({"key": key, "index": len(vt_rows), "record": rid, "crop": i, "grid": [h, w], "patches": h * w,
                            "prefix_rows": (h // 2) * (w // 2)})
            pixels.append(x["pixels"])
            pos.append(x["pos"])
            mask.append(x["mask"])
            vchk.append({"key": key, **eq})
            P += (h // 2) * (w // 2)
        assert P == npz["prefix"].shape[0], (rid, P)
        records[rid] = {"P": P, "crops": len(crops), "tiled": bool(plan["tiled"])}
    assert len(vt_rows) == 13, len(vt_rows)
    for name, arr in (("pixels", pixels), ("pos", pos), ("mask", mask)):
        written[f"g9_vt_{name}.f32"] = (write_stacked(OUT / f"g9_vt_{name}.f32", arr), len(arr))
    vt_doc = {"kind": "generic", "graph_kind": "vision_tower", "signature": "vision_tower",
              "inputs": [spec("pixels", pixels[0], "g9_vt_pixels.f32"), spec("pos", pos[0], "g9_vt_pos.f32"),
                         spec("mask", mask[0], "g9_vt_mask.f32")],
              "outputs": [{"name": "features", "dtype": "float32", "shape": [1, 1024, 768]}],
              "rows": vt_rows, "records": records,
              "sets": [{"name": "v_one_crop", "kind": "single", "rows": ["img_dogs_01/c0"]}], **head,
              "made_by": "scripts/s26_rows.py --r9 (host/d1_vision_host.py tower_inputs)"}
    written["g9_vt.json"] = (write_json(OUT / "g9_vt.json", vt_doc), len(vt_rows))
    checks["vision_tower_inputs_bit_equal_provider_preprocess"] = {
        "crops": len(vchk), "all_bit_equal": all(all(v for k, v in c.items() if k != "key") for c in vchk), "per_crop": vchk}

    # projector, timing only: img_dogs_01/c0's soft from the oracle's tower features
    with np.load(NPZ / "img_dogs_01.npz") as z:
        feat = np.asarray(z["tower_last_hidden_state"][0])
        gh, gw = (int(v) for v in z["spatial_shapes"][0])
    soft = V.projector_input(V.pixel_unshuffle(feat[: gh * gw], (gh, gw)))
    written["g9_pjt_soft.f32"] = (write_stacked(OUT / "g9_pjt_soft.f32", [soft]), 1)
    pj_doc = {"kind": "generic", "graph_kind": "projector", "signature": "projector",
              "inputs": [spec("soft", soft, "g9_pjt_soft.f32")],
              "outputs": [{"name": "prefix", "dtype": "float32", "shape": [1, 256, 1024]}],
              "rows": [{"key": "img_dogs_01/c0", "index": 0, "record": "img_dogs_01", "crop": 0, "grid": [gh, gw],
                        "prefix_rows": (gh // 2) * (gw // 2), "soft_from": "oracle npz tower_last_hidden_state"}],
              "sets": [{"name": "p_one_crop", "kind": "single", "rows": ["img_dogs_01/c0"]}], **head,
              "made_by": "scripts/s26_rows.py --r9 (host/d1_vision_host.py pixel_unshuffle + projector_input)"}
    written["g9_pjt.json"] = (write_json(OUT / "g9_pjt.json", pj_doc), 1)

    # audio: host float32 mel + the four masks per clip
    import audio_graph as G   # clip_samples: the round 7 reader (stdlib wave / soundfile)

    checks["audio"] = {}
    for stem, (T_b, rids, timing_rid) in AUDIO_SETS.items():
        xs, au_rows, achk = {n: [] for n in G.INPUT_NAMES}, [], []
        for rid in rids:
            x16, src = G.clip_samples(rid)
            x, info = AH.prepare(x16, bucket=T_b)
            with np.load(NPZ / f"{rid}.npz") as z:
                omel, oframes, oP = np.asarray(z["mel"]), int(z["frames"][0]), int(z["prefix"].shape[0])
            assert info["frames"] == oframes and info["P"] == oP and info["T_b"] == T_b, (rid, info, oframes, oP)
            assert x["mel"].shape == (1, AH.N_MELS, T_b) and omel.shape == (AH.N_MELS, info["T"]), (rid, x["mel"].shape, omel.shape)
            d = np.abs(x["mel"][0, :, : info["T"]].astype(np.float64) - omel)
            for n in G.INPUT_NAMES:
                assert x[n].dtype == np.float32, (rid, n)
                xs[n].append(x[n])
            au_rows.append({"key": rid, "index": len(au_rows), "record": rid, "source": src, "frames": info["frames"],
                            "T": info["T"], "T_b": T_b, "P": info["P"], "L123": info["L123"]})
            achk.append({"key": rid, "frames": info["frames"], "P": info["P"], "mel_vs_oracle_max_abs": float(d.max()),
                         "mel_vs_oracle_mean_abs": float(d.mean())})
        files = {}
        for n in G.INPUT_NAMES:
            fname = f"{stem}_{n}.f32"
            written[fname] = (write_stacked(OUT / fname, xs[n]), len(xs[n]))
            files[n] = fname
        T3 = AH.dims(T_b)[2]
        doc = {"kind": "generic", "graph_kind": "audio", "signature": f"audio_{T_b}",
               "inputs": [spec(n, xs[n][0], files[n]) for n in G.INPUT_NAMES],
               "outputs": [{"name": "prefix", "dtype": "float32", "shape": [1, T3, D]}],
               "rows": au_rows, "sets": [{"name": "f_audio_clip", "kind": "single", "rows": [timing_rid]}], **head,
               "made_by": "scripts/s26_rows.py --r9 (host/d1_audio_host.py prepare, float32 mel)"}
        written[f"{stem}.json"] = (write_json(OUT / f"{stem}.json", doc), len(au_rows))
        checks["audio"][stem] = {"T_b": T_b, "clips": achk}

    manifest = {
        "step": "round 9: S26 gate app inputs, round 9 additions (scripts/s26_rows.py --r9)",
        "sources": {"ref/records_ref.json": {"bytes": ORACLE.stat().st_size, "sha256": oracle_sha, "version": 2},
                    "position_table": f"{S.WEIGHTS.name} vision.tower.vision_model.embeddings.position_embedding.weight"},
        "files": {name: {"rows_or_slices": n, "bytes": p.stat().st_size, "sha256": sha256(p)} for name, (p, n) in written.items()},
        "checks": checks,
    }
    write_bytes(OUT / "rows_manifest_r9.json", (json.dumps(manifest, indent=1) + "\n").encode())
    print(json.dumps({"files": {k: [v["rows_or_slices"], v["bytes"]] for k, v in manifest["files"].items()},
                      "vision_bit_equal": checks["vision_tower_inputs_bit_equal_provider_preprocess"]["all_bit_equal"],
                      "audio_mel_max_abs": {s: max(c["mel_vs_oracle_max_abs"] for c in v["clips"]) for s, v in checks["audio"].items()},
                      "timing_sets": {k: v["positions"] for k, v in checks["timing_sets"].items()}}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main_r9() if sys.argv[1:] == ["--r9"] else main())
