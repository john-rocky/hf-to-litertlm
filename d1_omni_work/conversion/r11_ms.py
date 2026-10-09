"""Round 11 step 6: Mac ms of the ship files (ship/, the files of the repository), CPU XNNPACK 8 threads and Metal at
fp32 precision, with round 4 / 6 / 7's definitions. Exporter venv (lt094dev), from K:

    cd d1_omni_work; Q=~/code/standup/tools/quiet
    ~/venvs/lt094dev/bin/python scripts/r11_ms.py --wait-quiet          # outside any window: wait for a quiet machine
    $Q/quiet_hold.py d1c-r11-ms-cpu -- ~/venvs/lt094dev/bin/python scripts/r11_ms.py --backend cpu
    $Q/quiet_hold.py d1c-r11-ms-gpu -- ~/venvs/lt094dev/bin/python scripts/r11_ms.py --backend gpu_fp32
    ~/venvs/lt094dev/bin/python scripts/r11_ms.py --aggregate           # no run: -> results/timing_mac_r11.json

Window file results/timing_mac_r11_<backend>[_take<N>].json (never overwritten). Before every set: timing_mac.py's
contention gate (no process outside this process tree above 120 % CPU and CPU idle >= 50 % on the second
`top -l 2 -s 1` sample; the 1-minute load and `top` recorded). Every set = one fresh child process (one model set per
process: memory figures carry no other model's residue):
  text L<L>  timing_mac.py's child, unchanged, with --file = ship/d1-omni-600M_decide_L<L>_fp16.tflite: compile,
             memory, per workload 5 warm-up then 20 timed calls (a request workload: 20 requests of its rows), 1 call =
             write the six inputs + run + read `scores` back; workloads = round 4's (a) (b) (c) (d) (e) (e_L256) (f) (g);
             the first timed call of each row read out against the reference.
  vision     round 6's protocol on the ship tower / projector files: img_dogs_01's crop, 5 + 20 calls of each (1 call
             = write the inputs + run + read the output); the host steps of round 6 (load_image, layout + resize,
             patches, positions, unshuffle; 2 + 20 runs, this process); and (e) measured end to end: load_image ->
             crops -> tower -> unshuffle -> projector -> text inputs -> decide_128 -> readout on img_dogs_01/count.
  audio      round 7's protocol on the ship T1001 file: aud_01 (9.6 s), 5 + 20 calls; the host mel + inputs of the
             5 s cut / 10 s / 27 s clips (5 + 20 runs); and (f) measured end to end: host mel -> audio T1001 -> prefix
             rows -> text inputs -> decide_256 -> readout on aud_01/topic.
A CPU set whose median is more than 20 % above its min is taken once more in the same window (round 4's rule).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
K = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(K / "host"))
import timing_mac as TM  # noqa: E402

SHIP = K / "ship"
MODEL = "d1-omni-600M"
TMP = K / "out/r11_timing_tmp"
TEXT_L = (128, 256, 512, 1024, 2048, 4096)
WARMUP, REPS = TM.WARMUP, TM.REPS
IMAGE_ROW, AUDIO_ROW = ("img_dogs_01", "count"), ("aud_01", "topic")


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def ship_file(name):
    p = SHIP / name
    assert p.is_file(), p
    return p


def public_row(mode, rid, qname):
    doc = json.loads((SHIP / "fixtures" / f"public_{mode}.json").read_text())
    rec = next(r for r in doc["records"] if r["id"] == rid)
    e = next(x for x in rec["expected"] if x["name"] == qname)
    return rec, e


# ---------------------------------------------------------------- children

def _timed_graph(graph, feeds):
    t = time.perf_counter()
    out = graph(**feeds)
    return (time.perf_counter() - t) * 1000.0, out


def child_vision(a):
    import d1_host as H
    import d1_prompt as Pm
    import d1_vision_host as V
    import litert_run as R

    accel = "cpu" if a.backend == "cpu" else "gpu"
    doc = {"kind": "vision", "backend": a.backend, "pid": os.getpid(), "started": now(), "status": "FAIL"}
    try:
        tf, pf = ship_file(f"{MODEL}_vision_tower_fp16.tflite"), ship_file(f"{MODEL}_projector_fp16.tflite")
        doc["files"] = {"tower": str(tf.relative_to(K)), "projector": str(pf.relative_to(K))}
        doc["memory_before_compile"] = R.memory()
        t = time.perf_counter()
        tower = V.LiteRTGraph(tf, accel, "fp32", threads=8)
        doc["compile_seconds_tower"] = round(time.perf_counter() - t, 3)
        t = time.perf_counter()
        proj = V.LiteRTGraph(pf, accel, "fp32", threads=8)
        doc["compile_seconds_projector"] = round(time.perf_counter() - t, 3)
        doc["memory_after_compile"] = R.memory()
        doc["fully_accelerated"] = {"tower": tower.fully_accelerated, "projector": proj.fully_accelerated}
        table = V.load_position_table(SHIP / "host/vision_position_table.npy")
        rec, e = public_row("image", *IMAGE_ROW)
        path = SHIP / "fixtures" / rec["media"]["file"]
        crops, plan = V.crops_of(V.load_image(path))
        crop = V.to_patches(crops[0])
        h, w = crop["grid"]
        doc["crop"] = {"image": rec["media"]["file"], "crops": len(crops), "grid": [h, w], "plan": str(plan)}
        x = {k: np.ascontiguousarray(v, np.float32) for k, v in V.tower_inputs(crop, table).items()}
        feat = tower(**x)[0]
        soft = {"soft": np.ascontiguousarray(V.projector_input(V.pixel_unshuffle(feat[: h * w], (h, w))))}
        for name, graph, feeds in (("tower", tower, x), ("projector", proj, soft)):
            warm, timed, out = [], [], None
            for i in range(WARMUP + REPS):
                ms, out = _timed_graph(graph, feeds)
                (warm if i < WARMUP else timed).append(round(ms, 3))
            doc[name] = {"what": f"1 call = write the inputs + run + read the output ({name}, img_dogs_01's crop)",
                         "warmup_ms": warm, "timed_ms": timed, "ms_write_run_read": TM.stats(timed),
                         "finite": bool(np.isfinite(out).all())}
        # (e) end to end on the ship files: image -> prefix -> decide_128 -> readout
        tcm, desc = R.open_compiled(ship_file(f"{MODEL}_decide_L128_fp16.tflite"), accel, "fp32", threads=8)
        trun = R.Runner(tcm, "decide_128")
        q = Pm.as_question(rec["questions"][IMAGE_ROW[1]])
        temps = json.loads((SHIP / "contract.json").read_text())["temperatures"]

        def one():
            t0 = time.perf_counter()
            img = V.load_image(path)
            cs = [V.to_patches(c) for c in V.crops_of(img)[0]]
            ins = [{k: np.ascontiguousarray(v, np.float32) for k, v in V.tower_inputs(c, table).items()} for c in cs]
            t1 = time.perf_counter()
            feats = [tower(**i)[0] for i in ins]
            t2 = time.perf_counter()
            rows = []
            for c, f in zip(cs, feats):
                gh, gw = c["grid"]
                cells = V.pixel_unshuffle(f[: gh * gw], (gh, gw))
                rows.append(proj(soft=V.projector_input(cells))[0][: cells.shape[0]])
            prefix = np.concatenate(rows).astype(np.float32)
            t3 = time.perf_counter()
            xt = H.build_inputs(e["ids"], prefix, 128)
            xt["qtype_onehot"] = H.qtype_onehot(q)
            xt = {k: np.ascontiguousarray(v) for k, v in xt.items()}
            t4 = time.perf_counter()
            s, _, _, _ = trun.timed(xt)
            t5 = time.perf_counter()
            p = H.readout_f64(s, prefix.shape[0], e["markers"], q, False, temps)
            t6 = time.perf_counter()
            return p, [(t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3, (t4 - t3) * 1e3, (t5 - t4) * 1e3,
                       (t6 - t5) * 1e3, (t6 - t0) * 1e3]
        for _ in range(WARMUP):
            one()
        parts, probs = [], None
        for _ in range(REPS):
            p, ms = one()
            parts.append([round(v, 3) for v in ms])
            probs = probs or p
        cols = ["image_host_load_crops_patches_positions", "tower", "unshuffle_projector", "text_inputs",
                "text_graph_L128", "readout", "total"]
        doc["pipeline_e"] = {"row": "/".join(IMAGE_ROW), "what": "384 px image, 1 question, end to end on the ship "
                                                                 "files (one process, graphs compiled once)",
                             "columns": cols, "calls_ms": parts,
                             "ms": {c: TM.stats([r[i] for r in parts]) for i, c in enumerate(cols)},
                             "probs": probs, "probs_reference": e["probs"],
                             "max_abs_dp_vs_reference": max(abs(u - v) for u, v in zip(probs, e["probs"]))}
        trun.close()
        doc["memory_end"] = R.memory()
        if a.backend == "cpu":
            doc["host_steps"] = host_steps(V, path, table)
        tower.close()
        proj.close()
        doc["status"] = "OK"
    except BaseException as ex:  # recorded
        import traceback

        doc["error"], doc["traceback"] = f"{type(ex).__name__}: {ex}", traceback.format_exc()[-3000:]
    doc["finished"] = now()
    Path(a.out).write_text(json.dumps(doc, indent=1, default=str) + "\n")
    return 0 if doc["status"] == "OK" else 1


def host_steps(V, path, table):
    """Round 6's host steps on img_dogs_01 (2 warm-up + 20 runs, this process, one Python thread)."""
    steps = {"load_image": [], "layout_and_resize": [], "patches": [], "positions": [], "unshuffle_and_pad": []}
    g = np.random.default_rng(0)
    feats = None
    for i in range(2 + REPS):
        t = time.perf_counter()
        img = V.load_image(path)
        t1 = time.perf_counter()
        crops, _ = V.crops_of(img)
        t2 = time.perf_counter()
        cs = [V.to_patches(c) for c in crops]
        t3 = time.perf_counter()
        [V.tower_inputs(c, table) for c in cs]
        t4 = time.perf_counter()
        if feats is None:
            feats = [g.standard_normal((1024, 768)).astype(np.float32) for _ in cs]
        [V.projector_input(V.pixel_unshuffle(f[: c["grid"][0] * c["grid"][1]], c["grid"])) for f, c in zip(feats, cs)]
        t5 = time.perf_counter()
        if i >= 2:
            for k, (u, v) in zip(steps, ((t, t1), (t1, t2), (t2, t3), (t3, t4), (t4, t5))):
                steps[k].append((v - u) * 1000.0)
    tot = [sum(x) for x in zip(*steps.values())]
    return {"image": "img_dogs_01", "ms": {k: TM.stats(v) for k, v in steps.items()}, "ms_total": TM.stats(tot)}


def child_audio(a):
    import d1_audio_host as A
    import d1_host as H
    import d1_prompt as Pm
    import d1_vision_host as V
    import litert_run as R

    accel = "cpu" if a.backend == "cpu" else "gpu"
    doc = {"kind": "audio", "backend": a.backend, "pid": os.getpid(), "started": now(), "status": "FAIL"}
    try:
        af = ship_file(f"{MODEL}_audio_T1001_fp16.tflite")
        doc["files"] = {"audio": str(af.relative_to(K))}
        doc["memory_before_compile"] = R.memory()
        t = time.perf_counter()
        g = V.LiteRTGraph(af, accel, "fp32", threads=8)
        doc["compile_seconds_audio"] = round(time.perf_counter() - t, 3)
        doc["memory_after_compile"] = R.memory()
        doc["fully_accelerated"] = g.fully_accelerated
        rec, e = public_row("audio", *AUDIO_ROW)
        path = SHIP / "fixtures" / rec["media"]["file"]
        x16 = A.read_audio(path)
        x, info = A.prepare(x16)
        assert info["T_b"] == 1001, info
        x = {k: np.ascontiguousarray(v) for k, v in x.items()}
        warm, timed, out = [], [], None
        for i in range(WARMUP + REPS):
            ms, out = _timed_graph(g, x)
            (warm if i < WARMUP else timed).append(round(ms, 3))
        doc["audio_graph"] = {"what": "audio T1001, 1 call = write the 5 inputs + run + read the prefix [1, 126, 1024]",
                              "clip": rec["media"]["file"], "seconds": round(x16.shape[0] / 16000, 3), "P": info["P"],
                              "warmup_ms": warm, "timed_ms": timed, "ms_write_run_read": TM.stats(timed),
                              "finite": bool(np.isfinite(out).all())}
        doc["host_mel"] = {}
        others = [A.read_audio(SHIP / "fixtures/media" / f"aud_0{i}.wav") for i in (2, 3)]
        for label, clip in (("5 s", x16[: 5 * 16000]), ("10 s", x16), ("27 s", np.concatenate([x16] + others))):
            for _ in range(WARMUP):
                A.prepare(clip)
            ms = []
            for _ in range(REPS):
                t = time.perf_counter()
                A.prepare(clip)
                ms.append((time.perf_counter() - t) * 1000.0)
            doc["host_mel"][label] = {"seconds": round(clip.shape[0] / 16000, 3), "ms": TM.stats(ms),
                                      "what": "d1_audio_host.prepare: waveform + float32 mel + the 5 inputs (numpy)"}
        tcm, desc = R.open_compiled(ship_file(f"{MODEL}_decide_L256_fp16.tflite"), accel, "fp32", threads=8)
        trun = R.Runner(tcm, "decide_256")
        q = Pm.as_question(rec["questions"][AUDIO_ROW[1]])
        temps = json.loads((SHIP / "contract.json").read_text())["temperatures"]

        def one():
            t0 = time.perf_counter()
            xa, inf = A.prepare(x16)
            t1 = time.perf_counter()
            pre = A.prefix_rows(g(**{k: np.ascontiguousarray(v) for k, v in xa.items()}), inf)
            t2 = time.perf_counter()
            xt = H.build_inputs(e["ids"], pre, 256)
            xt["qtype_onehot"] = H.qtype_onehot(q)
            xt = {k: np.ascontiguousarray(v) for k, v in xt.items()}
            t3 = time.perf_counter()
            s, _, _, _ = trun.timed(xt)
            t4 = time.perf_counter()
            p = H.readout_f64(s, pre.shape[0], e["markers"], q, False, temps)
            t5 = time.perf_counter()
            return p, [(t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3, (t4 - t3) * 1e3, (t5 - t4) * 1e3,
                       (t5 - t0) * 1e3]
        for _ in range(WARMUP):
            one()
        parts, probs = [], None
        for _ in range(REPS):
            p, ms = one()
            parts.append([round(v, 3) for v in ms])
            probs = probs or p
        cols = ["mel_and_inputs", "audio_graph_T1001", "text_inputs", "text_graph_L256", "readout", "total"]
        doc["pipeline_f"] = {"row": "/".join(AUDIO_ROW), "what": "10 s audio, 1 question, end to end on the ship files "
                                                                 "(one process, graphs compiled once)",
                             "columns": cols, "calls_ms": parts,
                             "ms": {c: TM.stats([r[i] for r in parts]) for i, c in enumerate(cols)},
                             "probs": probs, "probs_reference": e["probs"],
                             "max_abs_dp_vs_reference": max(abs(u - v) for u, v in zip(probs, e["probs"]))}
        trun.close()
        doc["memory_end"] = R.memory()
        g.close()
        doc["status"] = "OK"
    except BaseException as ex:  # recorded
        import traceback

        doc["error"], doc["traceback"] = f"{type(ex).__name__}: {ex}", traceback.format_exc()[-3000:]
    doc["finished"] = now()
    Path(a.out).write_text(json.dumps(doc, indent=1, default=str) + "\n")
    return 0 if doc["status"] == "OK" else 1


# ---------------------------------------------------------------- window

def spawn(cmd_args, tag, script):
    TMP.mkdir(parents=True, exist_ok=True)
    out = TMP / f"{tag}.json"
    if out.exists():
        out.unlink()
    log = K / f"logs/r11_timing_{tag}.child.log"
    cmd = [sys.executable, str(script), "--child", "--out", str(out)] + cmd_args
    t0 = time.time()
    with open(log, "w") as fo:
        child = subprocess.Popen(cmd, stdout=fo, stderr=subprocess.STDOUT, cwd=str(K))
        _, status, ru = os.wait4(child.pid, 0)
    rc = os.waitstatus_to_exitcode(status)
    doc = json.loads(out.read_text()) if out.exists() else {"status": "FAIL", "error": "child wrote no record"}
    doc.update(child_returncode=rc, child_ru_maxrss_bytes=int(ru.ru_maxrss), child_seconds_wall=round(time.time() - t0, 1),
               child_log=str(log.relative_to(K)), child_record=str(out.relative_to(K)))
    if rc != 0:
        doc["child_log_tail"] = log.read_text(errors="replace").splitlines()[-30:]
    return doc


def spreads(child):
    out = {}
    for name, w in (child.get("workloads") or {}).items():
        for key in ("ms_write_run_read", "request_ms_write_run_read"):
            st = w.get(key)
            if st and st["min"]:
                out[f"{name}.{key}"] = round((st["median"] - st["min"]) / st["min"], 4)
    for name in ("tower", "projector", "audio_graph"):
        st = (child.get(name) or {}).get("ms_write_run_read")
        if st and st["min"]:
            out[f"{name}.ms_write_run_read"] = round((st["median"] - st["min"]) / st["min"], 4)
    for name in ("pipeline_e", "pipeline_f"):
        st = ((child.get(name) or {}).get("ms") or {}).get("total")
        if st and st["min"]:
            out[f"{name}.total"] = round((st["median"] - st["min"]) / st["min"], 4)
    return {"median_over_min_minus_1": out, "over": {k: v for k, v in out.items() if v > TM.SPREAD_MAX},
            "rule": f"CPU: retake once when > {TM.SPREAD_MAX:g}"}


def window_path(backend):
    p = K / f"results/timing_mac_r11_{backend}.json"
    n = 2
    while p.exists():
        p = K / f"results/timing_mac_r11_{backend}_take{n}.json"
        n += 1
    return p


def window_main(a):
    out = window_path(a.backend)
    sets = [(f"text_L{L}", ["--backend", a.backend, "--L", str(L), "--form", "ship_fp16",
                           "--file", f"ship/{MODEL}_decide_L{L}_fp16.tflite"], HERE / "timing_mac.py") for L in TEXT_L]
    sets += [("vision", ["--backend", a.backend, "--kind", "vision"], Path(__file__).resolve()),
             ("audio", ["--backend", a.backend, "--kind", "audio"], Path(__file__).resolve())]
    if a.only:
        sets = [s for s in sets if s[0] in a.only.split(",")]
    doc = {"what": f"round 11 Mac timing of the ship files, backend {a.backend}, one quiet window",
           "backend": a.backend, "protocol": __doc__.split("\n\n", 2)[2], "window_lock_line_at_start": TM.lock_line(),
           "started": now(), "sets": {}}
    rc = 0
    for name, args, script in sets:
        gate = TM.load_gate()
        st = {"load_gate": gate}
        if not gate["ok"]:
            st["status"] = "discarded: contention for the whole wait"
            doc["sets"][name] = st
            rc = 3
            continue
        tag = f"{a.backend}_{name}"
        child = spawn(args, tag, script)
        st["load_after"] = {"load": TM.load_avg(), "at": now()}
        sp = spreads(child)
        if a.backend == "cpu" and child.get("status") == "OK" and sp["over"]:
            st["first_attempt"] = {"result": child, "spread": sp}
            st["retake_gate"] = TM.load_gate()
            child = spawn(args, f"{tag}_retake", script)
            st["load_after_retake"] = {"load": TM.load_avg(), "at": now()}
            sp = spreads(child)
            st["retake_spread_still_over"] = sp["over"]
        st.update(spread=sp, result=child,
                  status="measured" if child.get("status") == "OK" else f"failed: {child.get('error')}")
        doc["sets"][name] = st
        print(f"{name}: {st['status']}", flush=True)
    doc["finished"] = now()
    doc["window_lock_line_at_end"] = TM.lock_line()
    out.write_text(json.dumps(doc, indent=1, default=str) + "\n")
    print(json.dumps({"out": str(out.relative_to(K)), "sets": {k: v["status"] for k, v in doc["sets"].items()}}))
    return rc


def wait_quiet(max_s=1800):
    """Outside any window: poll timing_mac.contention() until the machine is quiet (round 8's pre-check)."""
    t0 = time.time()
    while True:
        c = TM.contention()
        if c["ok"]:
            print(json.dumps({"quiet_after_s": round(time.time() - t0, 1), "idle_pct": c["idle_pct"]}))
            return 0
        if time.time() - t0 > max_s:
            print(json.dumps({"gave_up_s": round(time.time() - t0, 1), "contention": c}))
            return 3
        time.sleep(10)


# ---------------------------------------------------------------- aggregate

def latest(backend):
    ps = sorted(K.glob(f"results/timing_mac_r11_{backend}*.json"), key=lambda p: p.stat().st_mtime)
    return ps[-1] if ps else None


def pct(new, old):
    return None if new is None or old in (None, 0) else round((new / old - 1.0) * 100.0, 2)


def aggregate():
    """results/timing_mac_r11.json: per backend every median of the latest window, next to the same definition in
    rounds 4 (text, results/timing_mac_r4_<backend>_fp16.json), 6 (vision, results/timing_mac_r6.json, fp16 rows) and
    7 (audio, results/timing_mac_r7.json, fp16), with the difference in %."""
    r6 = {r["backend"]: r for r in json.loads((K / "results/timing_mac_r6.json").read_text())["rows"]
          if r["form"] == "fp16"}
    r7 = json.loads((K / "results/timing_mac_r7.json").read_text())
    out = {"what": "round 11 Mac ms of the ship files next to rounds 4 / 6 / 7 (same definitions; round 4 timed the text "
                   "files before the fp16-safe rewrite, rounds 6 / 7 the same vision / audio files under out/)",
           "written": now(), "unit": "ms, median of 20 timed calls after 5 warm-up (min, max, n kept)", "backends": {}}
    for b in ("cpu", "gpu_fp32"):
        p = latest(b)
        if p is None:
            continue
        d = json.loads(p.read_text())
        r4 = json.loads((K / f"results/timing_mac_r4_{b}_fp16.json").read_text())["by_L"]
        row = {"window_file": str(p.relative_to(K)), "window_started": d["started"], "window_finished": d.get("finished"),
               "lock_line": d.get("window_lock_line_at_start"), "text": {}, "conditions": {}}
        for name, st in d["sets"].items():
            res = st.get("result") or {}
            g = st["load_gate"]
            row["conditions"][name] = {"set_started": g.get("at"), "load_at_start": g.get("load_at_start"),
                                       "idle_pct": (g.get("contention") or {}).get("idle_pct"),
                                       "peers_above_120": (g.get("contention") or {}).get("peers_above_120"),
                                       "waited_s": g.get("waited_s"), "status": st["status"],
                                       "retake": "first_attempt" in st, "spread_over": (st.get("spread") or {}).get("over"),
                                       "child_record": res.get("child_record")}
            if name.startswith("text_L"):
                L = name[6:]
                for wn, w in (res.get("workloads") or {}).items():
                    old = ((r4.get(L) or {}).get("workloads") or {}).get(wn) or {}
                    e = {"ms": w["ms_write_run_read"], "round4_ms": (old.get("ms_write_run_read") or {}).get("median"),
                         "diff_pct": pct(w["ms_write_run_read"]["median"], (old.get("ms_write_run_read") or {}).get("median")),
                         "readout_check": w.get("readout_check")}
                    if w.get("request_ms_write_run_read"):
                        rq = (old.get("request_ms_write_run_read") or {}).get("median")
                        e.update(request_ms=w["request_ms_write_run_read"], round4_request_ms=rq,
                                 request_diff_pct=pct(w["request_ms_write_run_read"]["median"], rq))
                    row["text"][f"L{L}.{wn}"] = e
                row["text"][f"L{L}.compile_s"] = {"now": res.get("compile_seconds"),
                                                  "round4": (r4.get(L) or {}).get("compile_seconds")}
                row["text"][f"L{L}.memory_after_compile_phys_footprint"] = (res.get("memory_after_compile") or {}).get("phys_footprint")
            elif name == "vision" and res.get("status") == "OK":
                o6 = r6.get(b) or {}
                tw, pj = res["tower"]["ms_write_run_read"], res["projector"]["ms_write_run_read"]
                host = (res.get("host_steps") or {}).get("ms_total")
                row["vision"] = {"tower_ms": tw, "round6_tower_ms": o6.get("tower_ms"),
                                 "tower_diff_pct": pct(tw["median"], o6.get("tower_ms")),
                                 "projector_ms": pj, "round6_projector_ms": o6.get("projector_ms"),
                                 "projector_diff_pct": pct(pj["median"], o6.get("projector_ms")),
                                 "one_crop_tower_plus_projector_ms": round(tw["median"] + pj["median"], 3),
                                 "host_steps_ms": host, "round6_host_ms": o6.get("host_ms_img_dogs_01"),
                                 "pipeline_e": res["pipeline_e"]["ms"],
                                 "pipeline_e_max_abs_dp": res["pipeline_e"]["max_abs_dp_vs_reference"],
                                 "compile_s": [res["compile_seconds_tower"], res["compile_seconds_projector"]],
                                 "memory_after_compile_phys_footprint": (res.get("memory_after_compile") or {}).get("phys_footprint")}
            elif name == "audio" and res.get("status") == "OK":
                a7 = (r7.get("audio_graph") or {}).get(b, {}).get("fp16", {}).get("1001", {}).get("ms_write_run_read") or {}
                p7 = (r7.get("pipeline_10s_question") or {}).get(b, {}).get("fp16", {})
                m7 = (r7.get("host_mel") or {}).get(b, {}).get("fp16", {})
                ag = res["audio_graph"]["ms_write_run_read"]
                pf = res["pipeline_f"]["ms"]
                row["audio"] = {"T1001_ms": ag, "round7_T1001_ms": a7.get("median"), "T1001_diff_pct": pct(ag["median"], a7.get("median")),
                                "host_mel": res["host_mel"],
                                "round7_host_mel_ms": {k: (v.get("ms") or {}).get("median") for k, v in m7.items()},
                                "pipeline_f": pf, "round7_pipeline_f_ms": p7.get("ms_median"),
                                "pipeline_f_diff_pct": pct(pf["total"]["median"], (p7.get("ms_median") or {}).get("total")),
                                "pipeline_f_max_abs_dp": res["pipeline_f"]["max_abs_dp_vs_reference"],
                                "compile_s": res["compile_seconds_audio"],
                                "memory_after_compile_phys_footprint": (res.get("memory_after_compile") or {}).get("phys_footprint")}
        v, t = row.get("vision"), row["text"]
        if v and v.get("host_steps_ms") is None and b != "cpu":
            cpu_host = (out["backends"].get("cpu") or {}).get("vision", {}).get("host_steps_ms")
            v["host_steps_ms"], v["host_steps_from"] = cpu_host, "the CPU window (host steps are numpy, backend-free)"
        if v and v.get("host_steps_ms") and "L128.e" in t and "L256.e_L256" in t:
            hs = v["host_steps_ms"]["median"]
            gr = v["tower_ms"]["median"] + v["projector_ms"]["median"]
            v["e_sum_of_parts_ms"] = {"L128": round(hs + gr + t["L128.e"]["ms"]["median"], 3),
                                      "L256_round6_definition": round(hs + gr + t["L256.e_L256"]["ms"]["median"], 3),
                                      "round6_total_L256": (r6.get(b) or {}).get("total_one_question_384px_ms"),
                                      "parts": "host steps + tower + projector + the text call of (e)"}
        out["backends"][b] = row
    (K / "results/timing_mac_r11.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(json.dumps({b: {"window": v["window_file"]} for b, v in out["backends"].items()}, indent=1))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=("cpu", "gpu_fp32"))
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--kind", choices=("vision", "audio"))
    ap.add_argument("--out")
    ap.add_argument("--only", help="comma-separated set names (text_L128, ..., vision, audio)")
    ap.add_argument("--wait-quiet", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args()
    if a.wait_quiet:
        return wait_quiet()
    if a.aggregate:
        return aggregate()
    if a.child:
        return child_vision(a) if a.kind == "vision" else child_audio(a)
    return window_main(a)


if __name__ == "__main__":
    sys.exit(main())
