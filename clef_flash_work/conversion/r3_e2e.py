#!/usr/bin/env python3
"""Round 3 acceptance 5-7: image records end to end on the Mac GPU (Metal fp32 precision) — the host's pixels -> the
tower graph -> the image rows into the row graph's `embeds` -> round 2's V2 row graph -> the hidden states (scored by
scripts/r3_score.py with the author's head in venv-ref) — plus the image request's ms.

    TMPDIR=<K>/tmp ~/venvs/lt094dev/bin/python scripts/r3_e2e.py --tag L1024_v2 --L 1024 \
        --row out/r2/clef_rowprefill_L1024_v2_fp16fc.tflite --tower v2_fp16fc --arms g256,g448,zero,native \
        [--deadline-s 1300] [--ms]
        -> cache/r3/e2e/<tag>/<id>__<arm>.npy (hidden at the real positions, float32 [T, 4096]; red arms
           <id>__<arm>__zero.npy), cache/r3/e2e/<tag>/DONE when every selected run has its file,
           results/r3_e2e_<tag>_part<k>.json per invocation (resumable: a run whose file exists is skipped),
           results/r3_ms_<tag>.json (with --ms, after DONE, inside the same compile)

Guard: the row graph is used only when round 2 recorded its Metal fp32 gate as passed
(results/r2_score_v2_fp16fc_L<L>_gpu_f32.json gate_pass true) and the file is the one round 2 checked (path or its
realpath, bytes equal to round 2's check json); otherwise the script stops before compiling. One recorded extra row:
round 1's fp32 L1024 file on the CPU (--accel cpu --row out/r1/clef_rowprefill_L1024_fp32.tflite), gated by round 1's
CPU parity (results/score_cpu_L1024.json) — the image path on the CPU, not the Metal fp32 gate.
Arms (each run = one reference row; T <= L only, the others are listed as not fitting):
  g256 / g448  ids = the reference's ids; pixels = scripts/r3_host.py tile_patches(fixture image, tile) (asserted
               bit-equal to the reference pixel_values) -> tower graph (Metal fp32) -> rows -> image_row_inputs
  zero         the red arm: img_07/g448, img_08/g448, img_01/g256 with the image rows set to 0 (Core AI P4)
  native       the reference's own image_embeds (fp32, npz) as the rows (the row graph alone; no native tower)
  native_bucket  the same, only the native runs whose smallest row graph is this L (round 3b: L2048 gets the 6 runs
               that do not fit L1024)
Round 3b: with --accel cpu the V2 file is gated by round 2's CPU score (results/r2_score_v2_fp16fc_L<L>_cpu.json).
cos / sin = the host's M-RoPE tables of the closed-form planes (scripts/r3_host.py; equal to the reference's planes on
42/42 runs, results/r3_pixpos.json). Delegation: the runtime's lines per compiled model (row graph, each tower); on
Metal the line is all nodes in one partition, recorded per model (`metal_one_partition`); a miss ends the run
(status DELEGATION_FAIL, exit 4). Per run: tower rows vs the author's (cos, rel_rms), hidden max |diff| vs the
reference's last_hidden (recorded only), non-finite counts, the gate loop's ms.
ms (--ms): photo_01 (the largest fixture photo, 960 x 1280) as g256 and as g448: 5 warm-up + 20 timed requests, each =
host preprocessing (open the PNG, RGB, BICUBIC to the tile, normalize, patchify) + tower call (write, run, read) +
the row inputs (table rows, image rows, cos / sin) + row graph call (write, run, read); the head is not included.
Median / min / max / p90 of the request and of each part, the load average and the lock line before / after."""
import argparse
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

K = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(K / "scripts"))
import clef_host as CH  # noqa: E402
import r3_host as RH  # noqa: E402
import r3_lrt as RL  # noqa: E402
from clef_export import memory  # noqa: E402

SNAP = K / "hf/hub/models--Cloudflare--clef-flash/snapshots/f0b7d3cd65296abfe459e6c228b618f5d8b26d2f"
LOCK = Path(os.environ.get("MEASUREMENT_LOCK_FILE", "/dev/null"))   # the machine-wide measurement lock (its line is recorded)
ZERO_RUNS = (("img_07", "g448"), ("img_08", "g448"), ("img_01", "g256"))
SUFFIX = {"v2_fp16fc": "t16", "fp32": "t32"}      # hidden-file suffix of a tower form that is not the main one
MS_RUN = "photo_01"


_POS = {}


def tower_feed(m, patches: np.ndarray, grid: str) -> dict:
    """A tower's inputs by its compiled signature: patches [1, N, 1536] (+ the grid's position rows [1, N, C] when
    the form takes them as an input; cache/r3/consts_<grid>.npz pos_embeds)."""
    f = {"patches": patches[None]}
    if "pos" in m.in_det:
        if grid not in _POS:
            _POS[grid] = np.load(K / f"cache/r3/consts_{grid}.npz")["pos_embeds"][None]
        f["pos"] = _POS[grid]
    return f


def lock_line() -> str:
    try:
        return LOCK.read_text(errors="replace").strip()
    except OSError:
        return ""


def stats(got, want) -> dict:
    g = np.asarray(got, np.float64)
    w = np.asarray(want, np.float64)
    d = g - w
    return {"cos": float(g.ravel() @ w.ravel() / (np.linalg.norm(g) * np.linalg.norm(w))),
            "rel_rms": float(np.linalg.norm(d) / np.linalg.norm(w)), "max_abs": float(np.abs(d).max())}


def one_partition(dlg: dict) -> bool:
    rep = dlg.get("replacing") or []
    return bool(rep) and all(x["delegated"] == x["total"] and x["partitions"] == 1 for x in rep)


R1_ROW = "out/r1/clef_rowprefill_L1024_fp32.tflite"


def r1_cpu_gate(row: str) -> dict:
    """Round 1's fp32 L1024 file on the CPU: its CPU parity (results/score_cpu_L1024.json line_pass) and its bytes
    (results/export_L1024_fp32.json) — a recorded CPU row of the image path, not the Metal fp32 gate."""
    sc = json.loads((K / "results/score_cpu_L1024.json").read_text())
    ex = json.loads((K / "results/export_L1024_fp32.json").read_text())
    if sc["summary"].get("line_pass") is not True or ex.get("status") != "EXPORTED":
        raise SystemExit("STOP: round 1's CPU gate of the fp32 L1024 file is not a PASS")
    if (K / row).stat().st_size != ex["bytes"]:
        raise SystemExit(f"STOP: {row} is not round 1's file ({ex['bytes']} B)")
    return {"score": "results/score_cpu_L1024.json", "verdict": "PASS: fp32 L1024 cpu (round 1)", "tflite": row,
            "bytes": ex["bytes"], "sha256": ex["sha256"], "summary": sc["summary"]}


def r2_gate(L: int, row: str, accel: str = "gpu_f32") -> dict:
    """Round 2's gate of this V2 file on the backend the e2e uses (round 3b: the CPU e2e reads the CPU gate)."""
    backend = "cpu" if accel == "cpu" else "gpu_f32"
    sc = K / f"results/r2_score_v2_fp16fc_L{L}_{backend}.json"
    if not sc.exists():
        raise SystemExit(f"STOP: no round-2 {backend} score for L{L} ({sc.name})")
    d = json.loads(sc.read_text())
    if d.get("gate_pass") is not True:
        raise SystemExit(f"STOP: round 2's {backend} gate of L{L} is not a PASS: {d.get('verdict')}")
    chk = json.loads((K / d["check"]).read_text())
    p = K / row
    real = p.resolve()
    same = (row == d["tflite"]) or (str(real) == str(Path(d.get("tflite_realpath") or "").resolve()))
    if not same:
        raise SystemExit(f"STOP: {row} is not the file round 2 gated ({d['tflite']}, {d.get('tflite_realpath')})")
    if p.stat().st_size != chk["tflite_bytes"]:
        raise SystemExit(f"STOP: {row} has {p.stat().st_size} B, round 2 checked {chk['tflite_bytes']} B")
    return {"score": str(sc.relative_to(K)), "verdict": d["verdict"], "check": d["check"], "tflite": d["tflite"],
            "tflite_realpath": d.get("tflite_realpath"), "bytes": chk["tflite_bytes"], "summary": d.get("summary")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--row", required=True, help="K-relative row graph file (round 2's V2 of this L)")
    ap.add_argument("--towers", default="v2_fp16fc", help="tower forms, comma-separated; the first is the main one "
                    "(its hidden files carry no suffix), the others add <id>__<arm>__t16 / __t32 files")
    ap.add_argument("--arms", default="g256,g448,zero,native")
    ap.add_argument("--accel", default="gpu_f32", choices=["gpu_f32", "gpu_default", "cpu"])
    ap.add_argument("--deadline-s", type=float, default=None)
    ap.add_argument("--ms", action="store_true")
    ap.add_argument("--dry", action="store_true", help="print the gate and the selected runs, open no model, write nothing")
    a = ap.parse_args()
    t_start = time.monotonic()
    deadline = t_start + a.deadline_s if a.deadline_s else None
    arms = a.arms.split(",")
    towers = a.towers.split(",")
    assert towers and all(t in SUFFIX for t in towers), towers
    cache = K / f"cache/r3/e2e/{a.tag}"
    cache.mkdir(parents=True, exist_ok=True)
    k = 1
    while (K / f"results/r3_e2e_{a.tag}_part{k}.json").exists():
        k += 1
    part_path = K / f"results/r3_e2e_{a.tag}_part{k}.json"
    gate = r1_cpu_gate(a.row) if (a.row == R1_ROW and a.accel == "cpu") else r2_gate(a.L, a.row, a.accel)
    ref = json.loads((K / "ref/records_ref.json").read_text())
    recs = {r["id"]: r for r in json.loads((K / "fixtures/records.json").read_text())["records"]}
    sel, not_fitting = [], []
    for r in ref["rows"]:
        if r["arm"] == "text":
            continue
        want = []
        if r["arm"] in RH.TILES and r["arm"] in arms:
            want += [("" if i == 0 else SUFFIX[f], f) for i, f in enumerate(towers)]
        if r["arm"] == "native" and ("native" in arms or ("native_bucket" in arms and RH.bucket(r["tokens"]) == a.L)):
            want.append(("", None))
        if "zero" in arms and (r["id"], r["arm"]) in ZERO_RUNS:
            want.append(("zero", None))
        for w, f in want:
            if r["tokens"] > a.L:
                not_fitting.append(f"{r['id']}/{r['arm']}{'/' + w if w else ''} {r['tokens']}")
                continue
            sel.append((r, w, f))
    if a.dry:
        from collections import Counter
        print(json.dumps({"gate": gate["verdict"], "selected": len(sel),
                          "by_arm_variant": dict(Counter(f"{r['arm']}/{w or 'base'}" for r, w, f in sel)),
                          "native_ids": sorted(r["id"] for r, w, f in sel if r["arm"] == "native"),
                          "not_fitting_L": not_fitting,
                          "already_done": sum((cache / (f"{r['id']}__{r['arm']}" + (f"__{w}" if w else "") + ".npy")).exists()
                                              for r, w, f in sel)}, indent=1))
        return 0
    doc = {"what": f"round 3: image e2e {a.tag}", "tag": a.tag, "L": a.L, "row": a.row, "towers": towers, "arms": arms,
           "accel": a.accel, "part": k, "r2_gate": gate, "selected": len(sel), "not_fitting_L": not_fitting,
           "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "lock_at_start": lock_line(),
           "ai_edge_litert": importlib.metadata.version("ai-edge-litert"), "memory_start": memory(),
           "loadavg_start": list(os.getloadavg())}
    rows, status, exc_text = [], "OK", None
    models = {}
    try:
        need = sorted({(r["arm"], f) for r, w, f in sel if f is not None} | ({(g, towers[0]) for g in RH.TILES} if a.ms else set()))
        for name, path in [("row", K / a.row)] + [(f"{g}:{f}", K / f"out/r3/clef_tower_{g}_{f}.tflite") for g, f in need]:
            rl = RL.RuntimeLog(K / f"logs/r3_e2e_{a.tag}_part{k}_{name.replace(':', '_')}.runtime.log")
            with rl:
                models[name] = RL.Model(path, a.accel)
            info = models[name].info() | {"delegation": rl.delegation, "runtime_log_key_lines": rl.key_lines[:12],
                                          "memory_after_compile": memory()}
            info["metal_one_partition"] = one_partition(rl.delegation) if a.accel != "cpu" else None
            doc.setdefault("models", {})[name] = info
            print(f"compiled {name}: {info['compile_seconds']} s deleg {rl.delegation}", flush=True)
            if a.accel != "cpu" and not info["metal_one_partition"]:
                status = "DELEGATION_FAIL"
        if status == "OK":
            rlog = RL.RuntimeLog(K / f"logs/r3_e2e_{a.tag}_part{k}_runs.runtime.log")
            with rlog:
                table = CH.EmbedTable(SNAP)
                row_m = models["row"]
                Lm, D = row_m.out_shape["hidden"][1:]
                assert Lm == a.L, (Lm, a.L)
                for r, w, f in sel:
                    rid, arm = r["id"], r["arm"]
                    name = f"{rid}__{arm}" + (f"__{w}" if w else "")
                    if (cache / f"{name}.npy").exists():
                        continue
                    if deadline is not None and time.monotonic() > deadline:
                        status = "DEADLINE"
                        break
                    z = np.load(K / f"ref/npz/{rid}__{arm}.npz")
                    T, P = r["tokens"], r["token_offset"]
                    gt, gh, gw = r["grid_thw"]
                    H, W = gh // 2, gw // 2
                    item = {"id": rid, "arm": arm, "variant": w or "base", "tower": f, "tokens": T, "N": H * W}
                    t0 = time.perf_counter()
                    if w == "zero":
                        rows_img = np.zeros((H * W, D), np.float32)
                    elif arm == "native":
                        rows_img = z["image_embeds"].astype(np.float32)
                    else:
                        pv = RH.tile_patches(K / "fixtures/images" / recs[rid]["image_files"][0], RH.TILES[arm])
                        assert np.array_equal(pv.view(np.uint32), z["pixel_values"].view(np.uint32)), name
                        t1 = time.perf_counter()
                        rows_img = models[f"{arm}:{f}"].run(tower_feed(models[f"{arm}:{f}"], pv, arm))["rows"][0]
                        item["ms_tower"] = round((time.perf_counter() - t1) * 1e3, 2)
                        item["tower_vs_author"] = stats(rows_img, z["image_embeds"])
                        item["tower_nonfinite"] = int((~np.isfinite(rows_img)).sum())
                    embeds, valid, cos, sin = RH.image_row_inputs(r["ids"], a.L, table, rows_img, P, H, W)
                    t2 = time.perf_counter()
                    row_m.write({"embeds": embeds, "valid": valid, "cos": cos, "sin": sin})
                    t3 = time.perf_counter()
                    row_m.invoke()
                    t4 = time.perf_counter()
                    h = row_m.read("hidden")[0]
                    t5 = time.perf_counter()
                    hr = np.ascontiguousarray(h[:T])
                    tmp = cache / f"{name}.tmp.npy"
                    np.save(tmp, hr)
                    os.replace(tmp, cache / f"{name}.npy")
                    item.update(ms_total=round((t5 - t0) * 1e3, 1), ms_row_write=round((t3 - t2) * 1e3, 1),
                                ms_row_run=round((t4 - t3) * 1e3, 1), ms_row_read=round((t5 - t4) * 1e3, 1),
                                nonfinite_real=int((~np.isfinite(hr)).sum()), nonfinite_all=int((~np.isfinite(h)).sum()),
                                hidden_max_abs_vs_reference=(float(np.abs(hr - z["last_hidden"]).max()) if w != "zero" else None))
                    rows.append(item)
                    print(f"{len(rows)} {name} T={T} row {item['ms_row_run']:.0f} ms "
                          + (f"tower cos {item['tower_vs_author']['cos']:.8f} " if "tower_vs_author" in item else "")
                          + (f"max|dh| {item['hidden_max_abs_vs_reference']:.3e}" if item["hidden_max_abs_vs_reference"] is not None else ""),
                          flush=True)
                done = all((cache / (f"{r['id']}__{r['arm']}" + (f"__{w}" if w else "") + ".npy")).exists() for r, w, f in sel)
                if done and status == "OK":
                    (cache / "DONE").write_text(json.dumps({"runs": len(sel), "tag": a.tag, "at": time.strftime("%H:%M:%S")}) + "\n")
                # ms of the image request
                ms_path = K / f"results/r3_ms_{a.tag}.json"
                if a.ms and done and status == "OK" and not ms_path.exists():
                    est = np.median([x["ms_total"] for x in rows]) / 1e3 if rows else 3.0
                    if deadline is None or time.monotonic() + 2 * 25 * est * 1.3 + 20 < deadline:
                        ms_path.write_text(json.dumps(image_request_ms(models, table, ref, recs, a), indent=1) + "\n")
                    else:
                        doc["ms_deferred"] = f"not enough time left for 50 requests at {est:.2f} s"
            doc["runs_runtime_log_key_lines"] = rlog.key_lines[:20]
        for m in models.values():
            m.close()
    except BaseException as e:
        status = "FAIL"
        import traceback
        exc_text = traceback.format_exc()[-2000:]
        print(exc_text, flush=True)
    doc.update(status=status, error=exc_text, rows=rows, runs_this_part=len(rows), memory_end=memory(),
               lock_at_end=lock_line(), loadavg_end=list(os.getloadavg()),
               seconds_wall=round(time.monotonic() - t_start, 1), finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    if rows:
        doc["ms_row_run_gate_loop"] = RL.ms_stats([x["ms_row_run"] for x in rows])
    part_path.write_text(json.dumps(doc, indent=1, default=str) + "\n")
    print(json.dumps({kk: doc.get(kk) for kk in ("status", "selected", "runs_this_part", "seconds_wall")}), flush=True)
    return {"OK": 0, "DEADLINE": 0, "DELEGATION_FAIL": 4}.get(status, 5)


def image_request_ms(models, table, ref, recs, a) -> dict:
    out = {"what": "round 3: one image request (host preprocessing + tower + row inputs + row graph, no head)",
           "tag": a.tag, "L": a.L, "row": a.row, "tower": a.towers.split(",")[0], "accel": a.accel, "run": MS_RUN,
           "warmup": 5, "timed": 20, "lock_before": lock_line(), "loadavg_before": list(os.getloadavg()), "grids": {}}
    row_m = models["row"]
    for grid in ("g256", "g448"):
        r = next(x for x in ref["rows"] if x["id"] == MS_RUN and x["arm"] == grid)
        img = K / "fixtures/images" / recs[MS_RUN]["image_files"][0]
        T, P = r["tokens"], r["token_offset"]
        H = W = RH.tile_grid(RH.TILES[grid])
        parts = {k: [] for k in ("request", "host_pixels", "tower_call", "row_inputs", "row_call")}
        for i in range(25):
            t0 = time.perf_counter()
            pv = RH.tile_patches(img, RH.TILES[grid])
            t1 = time.perf_counter()
            tm = models[f"{grid}:{a.towers.split(',')[0]}"]
            tm.write(tower_feed(tm, pv, grid))
            tm.invoke()
            rows_img = tm.read("rows")[0]
            t2 = time.perf_counter()
            embeds, valid, cos, sin = RH.image_row_inputs(r["ids"], a.L, table, rows_img, P, H, W)
            t3 = time.perf_counter()
            row_m.write({"embeds": embeds, "valid": valid, "cos": cos, "sin": sin})
            row_m.invoke()
            h = row_m.read("hidden")
            t4 = time.perf_counter()
            if i >= 5:
                for kk, v in (("request", t4 - t0), ("host_pixels", t1 - t0), ("tower_call", t2 - t1),
                              ("row_inputs", t3 - t2), ("row_call", t4 - t3)):
                    parts[kk].append(v * 1e3)
        out["grids"][grid] = {"tokens": T, "image_rows": H * W, "finite": bool(np.isfinite(h[0, :T]).all()),
                              **{kk: RL.ms_stats(v) for kk, v in parts.items()},
                              "request_all": [round(v, 2) for v in parts["request"]]}
    out.update(lock_after=lock_line(), loadavg_after=list(os.getloadavg()))
    return out


if __name__ == "__main__":
    raise SystemExit(main())
