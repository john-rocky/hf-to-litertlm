"""Round 8 acceptance 5: Mac wall time of a request with pictures on the host's path (host/d1_litert.py D1Host,
item 9), per part, through the CompiledModel API.

    $EXPORT \
        scripts/d1_image_timing.py --accel gpu_f32 --sets one_picture,split_picture \
        --out cache/real/image/timing_gpu_f32.json
    $EXPORT \
        scripts/d1_image_timing.py --accel cpu --sets one_picture --out cache/real/image/timing_cpu.json
    $EXPORT scripts/d1_image_timing.py --merge cache/real/image/timing_gpu_f32.json \
        cache/real/image/timing_cpu.json --out results/timing_mac_image.json

Run it only inside the measurement window (quiet_hold.py; label with `timing`). Sets = requests of
results/real_vision_e2e_ref.json: one_picture = one_tile (card_cats_001/cats + the card's COCO photo: 1 tile, 271
tokens, the L512 row graph) = the card's picture column; split_picture = split_thumbnail (tv4x_qnli_07 + the synthetic
s1280x853: 2 x 3 tiles + thumbnail = 7 tower calls and 7 projector calls, 1,856 tokens, L2048). Files = the shipping
forms: exports/real_vision_tower_v2_fp16fc, real_projector_v2_fp16fc, real_rowprefill_embeds_L{L}_v2e_fp16fc.
A round answers the request the way D1Host.decide does (the same functions, in its order), with a timer per part:
  preprocess  host CPU: open and decode the picture file, cap_pixels, preprocess (resize, normalise, patches), the
              render and the ids (row_ids), the position table resized to each tile's grid (tower_inputs)
  tower       the tower graph, once per tile (write + run + read-back of the whole output)
  unshuffle   host CPU: the real patches -> cells -> the projector input, per tile
  projector   the projector graph, once per tile (write + run + read-back)
  embed       host CPU: the table's rows (bfloat16 -> float32), the picture tokens at the <image> positions, padding
  row_graph   the embeds row graph of the bucket, one call (write + run + read-back of the whole hidden output)
  readout     host CPU: the read-out and the answer
  total       the round's wall time
--warmup rounds (5) then --reps timed rounds (20): median / min / max per part. load1 before and after each set; top,
swap and the window lock line before and after the run; each graph's compile seconds and is_fully_accelerated; the
first round's probabilities must be finite and equal, bit for bit, to D1Host.decide's on the same request.
Outputs (never overwritten): --out json; --merge writes one table (set x accelerator x part) and the card columns.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

K = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(K / "host"))
import d1_litert as H  # noqa: E402
import d1_vision as V  # noqa: E402

REF = K / "results/real_vision_e2e_ref.json"
TABLES = K / "cache/real/tables"
TOWER = "exports/real_vision_tower_v2_fp16fc.tflite"
PROJECTOR = "exports/real_projector_v2_fp16fc.tflite"
TEXT = "exports/real_rowprefill_embeds_L{L}_v2e_fp16fc.tflite"
SETS = {"one_picture": "one_tile", "split_picture": "split_thumbnail"}
PARTS = ("preprocess", "tower", "unshuffle", "projector", "embed", "row_graph", "readout", "total")


def stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def machine_lines() -> dict:
    top = subprocess.run(["top", "-l", "1", "-n", "0"], capture_output=True, text=True).stdout.splitlines()
    swap = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout.strip()
    lock = Path(os.environ.get("D1_GPU_LOCK", str(K / "gpu.lock")))
    return {"top": [ln for ln in top if ln.startswith(("Load Avg", "CPU usage", "PhysMem"))], "swap": swap,
            "gpu_lock": lock.read_text().strip() if lock.exists() else None, "at": stamp()}


def stats(xs) -> dict:
    a = np.asarray(xs, dtype=np.float64)
    return {"median": round(float(np.median(a)), 3), "min": round(float(a.min()), 3), "max": round(float(a.max()), 3),
            "n": int(a.size)}


def load1() -> float:
    return round(os.getloadavg()[0], 2)


def one_round(req: dict, g: dict, tok, ids_map: dict, pos_table, table, rt, pad_id: int) -> tuple[dict, list]:
    """One request through the host's functions (D1Host.rows / picture_tokens / hidden_at_slot / readout), timed."""
    ms = {}
    perf = time.perf_counter
    t_all = t = perf()
    q = H.as_question(req["question"])
    pics = [V.preprocess(V.cap_pixels(V.open_picture(p))) for p in req["images"]]
    text = H.prefix_text(req["state"], V.image_markup(len(pics))) + H.suffix_text(tok, q)
    ids = V.row_ids(tok.encode, text, pics, ids_map)
    groups = H.readout_ids(tok, q)
    tiles = [tile for p in pics for tile in p.tiles]
    feeds = [V.tower_inputs(tile, pos_table) for tile in tiles]
    ms["preprocess"] = perf() - t
    ms["tower"] = ms["unshuffle"] = ms["projector"] = 0.0
    toks = []
    for tile, fd in zip(tiles, feeds):
        t = perf()
        feat = g["tower"](**fd)[0]
        ms["tower"] += perf() - t
        t = perf()
        h, w = tile.grid
        cells = V.pixel_unshuffle(feat[: h * w], tile.grid)
        soft = V.projector_input(cells)
        ms["unshuffle"] += perf() - t
        t = perf()
        mm = g["projector"](soft=soft)[0]
        ms["projector"] += perf() - t
        toks.append(mm[: cells.shape[0]])
    t = perf()
    tokens = np.concatenate(toks).astype(np.float32)
    emb = V.insert_embeddings(ids, table.rows, tokens, ids_map["image"])
    n = len(ids)
    text_g = g["text"]
    x = np.empty((1, text_g.L, emb.shape[1]), dtype=np.float32)
    x[0, :n] = emb
    x[0, n:] = table.rows([pad_id])
    valid = np.zeros((1, text_g.L), dtype=np.float32)
    valid[0, :n] = 1.0
    ms["embed"] = perf() - t
    t = perf()
    hidden = text_g(x, valid)
    ms["row_graph"] = perf() - t
    t = perf()
    probs = H.readout(hidden[0, n - 1], rt, groups)
    H.answer(q, probs)
    ms["readout"] = perf() - t
    ms["total"] = perf() - t_all
    return {k: v * 1000 for k, v in ms.items()}, probs


def run(a) -> int:
    out = K / a.out
    assert not out.exists(), f"refusing to overwrite {out}"
    ref = {r["name"]: r for r in json.loads(REF.read_text())["requests"]}
    contract = json.loads((K / "host/contract.json").read_text())
    ids_map = V.token_ids(contract)
    pad_id = int(contract["token_ids"]["pad"])
    tok = H.D1Tokenizer(K / "hf_small" / H.TOKENIZER_FILE)
    table = H.EmbedTable(TABLES / "embed_table.safetensors")
    rt = H.ReadoutTable.from_file(TABLES / "readout_table.safetensors")
    pos_table = V.load_position_table(TABLES / "vision_position_table.safetensors")
    accel, precision = ("cpu", "fp32") if a.accel == "cpu" else ("gpu", "fp32" if a.accel == "gpu_f32" else "default")
    doc = {"what": "round 8: Mac wall time of a request with pictures on the host's path, per part (ms; graph calls = "
                   "write + run + read-back)", "accel": a.accel, "threads": a.threads if accel == "cpu" else None,
           "versions": {p: importlib.metadata.version(p) for p in ("ai-edge-litert", "numpy", "pillow", "tokenizers")},
           "warmup": a.warmup, "reps": a.reps, "machine_before": machine_lines(), "load1_before": load1(), "sets": []}
    t0 = time.time()
    g, compiles = {}, {}
    for name, f in (("tower", TOWER), ("projector", PROJECTOR)):
        t = time.perf_counter()
        g[name] = V.LiteRTGraph(K / f, accel, precision, a.threads)
        compiles[name] = {"file": f, "compile_s": round(time.perf_counter() - t, 3),
                          "fully_accelerated": g[name].fully_accelerated}
    doc["graphs"] = compiles
    for sname in a.sets.split(","):
        r = ref[SETS[sname]]
        req = {"state": r["request"]["state"], "question": r["request"]["question"],
               "images": [K / p for p in r["pictures"]]}
        L = H.pick_L(r["tokens"])
        t = time.perf_counter()
        g["text"] = H.LiteRTEmbedsGraph(K / TEXT.format(L=L), accel, precision, a.threads)
        text_info = {"file": TEXT.format(L=L), "compile_s": round(time.perf_counter() - t, 3),
                     "fully_accelerated": g["text"].fully_accelerated}
        vision = V.VisionPath(g["tower"], g["projector"], pos_table, ids_map)
        host = H.D1Host(tok, {}, rt, pad_id, embeds_graphs={L: g["text"]}, embed_table=table, vision=vision)
        qn = r["question"]
        decided = host.decide({"state": req["state"], "questions": {qn: req["question"]}, "images": req["images"]})
        rec = {"name": sname, "request": r["name"], "record": r["record"], "question": qn, "pictures": r["pictures"],
               "tokens": r["tokens"], "tiles": r["tiles"], "L": L, "row_graph": text_info,
               "calls": {"tower": r["tiles"], "projector": r["tiles"], "row_graph": 1},
               "load1_before": load1(), "started_at": stamp()}
        first_ms, first_probs = one_round(req, g, tok, ids_map, pos_table, table, rt, pad_id)
        q = H.as_question(req["question"])
        want = decided["answers"][qn]
        want_probs = [want["noul"]] if q.type == "noul" else list(want["probabilities"].values())
        rec["first_round_finite"] = bool(np.all(np.isfinite(first_probs)))
        rec["first_round_equals_decide"] = [float(x) for x in first_probs[:len(want_probs)]] == [float(x) for x in want_probs]
        rec["first_round_probs"] = first_probs
        for _ in range(max(0, a.warmup - 1)):
            one_round(req, g, tok, ids_map, pos_table, table, rt, pad_id)
        per = {k: [] for k in PARTS}
        for _ in range(a.reps):
            ms, _ = one_round(req, g, tok, ids_map, pos_table, table, rt, pad_id)
            for k in PARTS:
                per[k].append(ms[k])
        rec.update(ms={k: stats(v) for k, v in per.items()}, load1_after=load1(), finished_at=stamp(),
                   sum_of_part_medians=round(sum(float(np.median(per[k])) for k in PARTS if k != "total"), 3))
        doc["sets"].append(rec)
        print(json.dumps({"set": sname, "accel": a.accel, "total_median_ms": rec["ms"]["total"]["median"],
                          "parts": {k: rec["ms"][k]["median"] for k in PARTS}, "load1": [rec["load1_before"], rec["load1_after"]],
                          "equal_decide": rec["first_round_equals_decide"]}), flush=True)
        g["text"].close()
        del g["text"]
    for k in ("tower", "projector"):
        g[k].close()
    doc.update(machine_after=machine_lines(), load1_after=load1(), seconds_wall=round(time.time() - t0, 1), finished_at=stamp())
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1) + "\n")
    ok = all(s["first_round_finite"] and s["first_round_equals_decide"] for s in doc["sets"])
    return 0 if ok else 1


def merge(a) -> int:
    out = K / a.out
    assert not out.exists(), f"refusing to overwrite {out}"
    rows, sources = [], []
    for src in a.merge:
        d = json.loads((K / src).read_text())
        sources.append({"file": src, "accel": d["accel"], "threads": d["threads"], "window": d["machine_before"]["gpu_lock"],
                        "started": d["machine_before"]["at"], "finished": d["finished_at"], "graphs": d["graphs"],
                        "load1_before": d["load1_before"], "load1_after": d["load1_after"]})
        for s in d["sets"]:
            rows.append({"set": s["name"], "accel": d["accel"], "threads": d["threads"], "request": s["request"],
                         "tokens": s["tokens"], "tiles": s["tiles"], "L": s["L"], "calls": s["calls"],
                         "row_graph_file": s["row_graph"]["file"], "ms": s["ms"],
                         "sum_of_part_medians": s["sum_of_part_medians"], "load1": [s["load1_before"], s["load1_after"]],
                         "first_round_equals_decide": s["first_round_equals_decide"],
                         "window": d["machine_before"]["gpu_lock"], "source": src})
    card = {}
    for key, (sname, accel) in {"one_picture_one_question_gpu_f32": ("one_picture", "gpu_f32"),
                                "split_picture_one_question_gpu_f32": ("split_picture", "gpu_f32"),
                                "one_picture_one_question_cpu": ("one_picture", "cpu")}.items():
        hit = [r for r in rows if (r["set"], r["accel"]) == (sname, accel)]
        card[key] = None if not hit else {"total_median_ms": hit[0]["ms"]["total"]["median"],
                                          "parts_median_ms": {k: v["median"] for k, v in hit[0]["ms"].items() if k != "total"},
                                          "min_ms": hit[0]["ms"]["total"]["min"], "max_ms": hit[0]["ms"]["total"]["max"],
                                          "n": hit[0]["ms"]["total"]["n"], "load1": hit[0]["load1"],
                                          "window": hit[0]["window"]}
    doc = {"what": "round 8: Mac timing of a request with pictures (tower v2 + projector v2 + v2e row graph, host CPU parts; "
                   "median of the timed rounds, ms)", "sources": sources, "card_columns": card, "rows": rows}
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print("| set | accel | tiles | L | " + " | ".join(PARTS) + " | min / max total | n | load1 |")
    print("|---|---|---:|---:|" + "---:|" * len(PARTS) + "---|---:|---|")
    for r in rows:
        print(f"| {r['set']} | {r['accel']} | {r['tiles']} | {r['L']} | "
              + " | ".join(str(r["ms"][k]["median"]) for k in PARTS)
              + f" | {r['ms']['total']['min']} / {r['ms']['total']['max']} | {r['ms']['total']['n']} | {r['load1'][0]} / {r['load1'][1]} |")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--accel", choices=["cpu", "gpu_f32", "gpu_default"], default="gpu_f32")
    ap.add_argument("--sets", default="one_picture,split_picture")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--out", default="")
    ap.add_argument("--merge", nargs="+", default=[])
    a = ap.parse_args()
    assert a.out, "--out is required"
    return merge(a) if a.merge else run(a)


if __name__ == "__main__":
    sys.exit(main())
