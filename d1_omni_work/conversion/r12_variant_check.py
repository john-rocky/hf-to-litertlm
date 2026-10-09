"""Round 12 (launch 2026-10-08-d1c-d1-omni-litert-opus-r12.md): the Mac gates and the Mac ms of the audio / vision
graph variants (scripts/audio_graph.py --variant f16safe | clast, scripts/vision_graph.py --variant f16safe), against
the round 6 / 7 files and stores of the same backend and form and against the oracle (ref/records_ref.json v2).
Runs in the exporter venv (lt094dev): ai-edge-litert 2.2.0 CompiledModel, numpy, torch (the host read-out).

    cd d1_omni_work; Q=~/code/standup/tools/quiet; PY=~/venvs/lt094dev/bin/python
    $Q/quiet_wait.py -- $PY scripts/r12_variant_check.py audio --variant f16safe --form fp16 --backend cpu
    $Q/quiet_hold.py d1c-r12-gpu-... -- $PY scripts/r12_variant_check.py audio --variant f16safe --form fp16 \
        --backend gpu --precision fp32|default          # one GPU run = one window (the run is a child process)
    $Q/quiet_wait.py -- $PY scripts/r12_variant_check.py vision --form fp16 --backend cpu   (gpu as above)
    $Q/quiet_hold.py d1c-r12-ms -- $PY scripts/r12_variant_check.py ms           # the Mac ms window (step 4)

audio: every clip of the buckets whose variant file exists, host float32 mel (host/d1_audio_host.prepare): T1001 = the 6
  clips of at most 10 s, T2001 = card_topic (10.4 s), T501 = aud_01 cut to 5 s, T3001 = aud_01 + 02 + 03 (27.2 s);
  pad-content (the invalid mel frames replaced by noise: the valid prefix rows must not move). Prefix rows ->
  out/r12_runs/audio_<label>_<variant>_<form>.npz (+ .json: delegation from the runtime's VERBOSE log, compile s).
  Score -> results/audio_parity_<label>_<variant>_<form>.json: per clip vs round 7's store of the same backend / form
  (out/r7_runs/audio_<label>_<form>[_T3001].npz = the current file, same host mel: bit-equal or max |d|), vs the
  oracle's prefix (the 7 oracle clips), vs the provider's own prefix (the 5 s cut, the 27.2 s clip); end to end = the
  7 prefixes (T1001 + T2001) through out/d1omni_decide_L256_fp16.tflite on the CPU (8 threads) -> the host read-out
  -> the 19 audio rows vs the oracle (litert_gate.compare, FACTS §7), with round 7's e2e of the current file beside it.
vision: the 13 crops of the 7 image records (host/d1_vision_host.tower_inputs): the variant tower and the current tower
  (the same form, the same backend, one process), the current projector of the same form; per crop the features of
  both towers (bit-equal / max |d| on the real patches), per record the prefix (variant tower -> host unshuffle ->
  projector) vs the oracle, round 6's store of the same backend / form (out/r6_runs/<tag>/<id>.npy) and the current
  tower's chain run here; end to end = the 16 image rows through the fp16 text graphs on the CPU (the smallest of L256 /
  512 / 1024 / 2048 / 4096 that holds the row, round 6's rule) -> results/vision_parity_<label>_f16safe_<form>.json.
ms: one window, a fresh child per set, 5 warm-up + 20 timed calls (write + run + read back), the round 4 load gate
  before each set -> results/timing_mac_r12.json (sets: the audio T1001 current / f16safe / clast files and the vision
  tower current / f16safe files on Metal fp32 precision; the f16safe files also on Metal default precision).
Output files are never overwritten (a second run of a tag stops).
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
import d1_src as S  # noqa: E402  (stdlib only)
import d1_audio_host as AH  # noqa: E402  (numpy only)

RUNS = K / "out/r12_runs"
R7 = K / "out/r7_runs"
R6 = K / "out/r6_runs"
NPZ = K / "ref/npz"
AUDIO_CLIPS = ("aud_01", "aud_02", "aud_03", "aud_reservation_01", "aud_weather_02", "aud_food_03", "card_topic")
BUCKET_OF = {"card_topic": 2001}
CUT = ("aud_01", 5.0)
LONG_REF = K / "out/r7_eager_prefix_T3001_long.npz"
CUT_REF = K / "out/r7_eager_prefix_T501_aud01cut5.npz"
WARMUP, REPS = 5, 20


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def write_new(path: Path, doc) -> Path:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, default=str) + "\n")
    os.replace(tmp, path)
    return path


def diff(a, b) -> dict:
    a64, b64 = np.asarray(a, np.float64), np.asarray(b, np.float64)
    assert a64.shape == b64.shape, (a64.shape, b64.shape)
    d = np.abs(a64 - b64)
    rr = float(np.sqrt((b64 ** 2).mean())) if b64.size else 0.0
    return {"max_abs": float(np.nanmax(d)) if d.size else 0.0, "mean_abs": float(np.nanmean(d)) if d.size else 0.0,
            "ref_absmax": float(np.abs(b64).max()) if b64.size else 0.0,
            "rel_rms": float(np.sqrt(np.nanmean(d ** 2)) / rr) if rr else None,
            "bit_equal": bool(np.array_equal(np.asarray(a), np.asarray(b))),
            "nonfinite": int((~np.isfinite(a64)).sum())}


def label_of(backend, precision):
    return "cpu" if backend == "cpu" else f"gpu_{precision}"


def child_parent(argv_child, record_path: Path, log: Path):
    """Run this file with argv_child in a child process (a GPU delegate abort ends only the child); a child that wrote
    no record of its own pid gets a failure record here."""
    import signal

    t0 = time.time()
    with open(log, "a") as fo:
        fo.write(f"--- {now()} {' '.join(argv_child)}\n")
        fo.flush()
        child = subprocess.Popen([sys.executable, str(Path(__file__).resolve())] + argv_child, stdout=fo,
                                 stderr=subprocess.STDOUT, cwd=str(K))
        _, st, ru = os.wait4(child.pid, 0)
    rc = os.waitstatus_to_exitcode(st)
    rec = json.loads(record_path.read_text()) if record_path.exists() else {}
    if rec.get("pid") != child.pid:
        rec = {"pid": child.pid, "status": "GPU_FAIL", "returncode": rc,
               "signal": signal.Signals(-rc).name if rc < 0 else None, "seconds_wall": round(time.time() - t0, 1),
               "child_stdio_tail": log.read_text(errors="replace").splitlines()[-40:]}
    rec["child_ru_maxrss_bytes"] = int(ru.ru_maxrss)
    rec["child_returncode"] = rc
    record_path.write_text(json.dumps(rec, indent=1, default=str) + "\n")
    print(f"child pid={child.pid} rc={rc} status={rec.get('status')} seconds={time.time() - t0:.1f}", flush=True)
    return rc


# ---------------------------------------------------------------- audio

def audio_file(T, variant, form):
    return K / (f"out/d1omni_audio_T{T}_{variant}_{form}.tflite" if variant else f"out/d1omni_audio_T{T}_{form}.tflite")


def audio_clip_inputs(T):
    """-> [(store key, inputs, info)] of the bucket's clips (host float32 mel)."""
    import audio_graph as G

    out = []
    if T == 1001:
        for rid in AUDIO_CLIPS:
            if BUCKET_OF.get(rid, 1001) != 1001:
                continue
            x16, _ = G.clip_samples(rid)
            x, info = AH.prepare(x16, bucket=1001)
            out.append((f"{rid}__host", x, info))
    elif T == 2001:
        x16, _ = G.clip_samples("card_topic")
        x, info = AH.prepare(x16, bucket=2001)
        out.append(("card_topic__host", x, info))
    elif T == 501:
        x16, _ = G.clip_samples(CUT[0], seconds=CUT[1])
        x, info = AH.prepare(x16)
        assert info["T_b"] == 501
        out.append(("aud_01_cut5__T501", x, info))
    elif T == 3001:
        x16, _ = G.long_samples()
        x, info = AH.prepare(x16)
        assert info["T_b"] == 3001
        out.append(("long__host", x, info))
    return out


def audio_run(a) -> int:
    import importlib.metadata as md
    import resource

    import audio_graph as G
    import litert_run as R

    label = label_of(a.backend, a.precision)
    tag = f"audio_{label}_{a.variant}_{a.form}{a.suffix}"
    RUNS.mkdir(parents=True, exist_ok=True)
    rec_path = RUNS / f"{tag}.json"
    if a.backend == "gpu" and not a.child:
        assert not rec_path.exists(), f"refusing to overwrite {rec_path}"
        argv = ["audio", "--variant", a.variant, "--form", a.form, "--backend", "gpu", "--precision", a.precision,
                "--buckets", a.buckets, "--child"] + (["--suffix", a.suffix] if a.suffix else [])
        rc = child_parent(argv, rec_path, K / f"logs/r12_{tag}.child_stdio.log")
        return rc if rc else audio_score(a)
    assert a.child or not rec_path.exists(), f"refusing to overwrite {rec_path}"
    log = K / f"logs/r12_{tag}.runtime.log"
    info = {"tag": tag, "started": now(), "pid": os.getpid(), "backend": a.backend, "precision": a.precision,
            "variant": a.variant, "form": a.form, "ai_edge_litert": md.version("ai-edge-litert"), "files": {},
            "compile_seconds": {}, "is_fully_accelerated": {}, "status": "FAIL"}
    store, clips = {}, {}
    rng = np.random.default_rng(12)
    t0 = time.time()
    with R.capture_fd2(log):
        try:
            info["logger"] = R.runtime_log_verbose()
            for T in [int(t) for t in a.buckets.split(",")]:
                path = audio_file(T, a.variant, a.form)
                if not path.exists():
                    info["files"][str(T)] = {"file": str(path.relative_to(K)), "missing": True}
                    continue
                info["files"][str(T)] = {"file": str(path.relative_to(K)), "bytes": path.stat().st_size,
                                         "sha256": R.sha256_file(path)}
                t1 = time.time()
                cm, desc = R.open_compiled(path, a.backend, a.precision, threads=8)
                info["compile_seconds"][str(T)] = round(time.time() - t1, 2)
                info["options"] = desc
                try:
                    info["is_fully_accelerated"][str(T)] = bool(cm.is_fully_accelerated())
                except Exception as e:  # informational
                    info["is_fully_accelerated"][str(T)] = f"unavailable: {type(e).__name__}: {e}"
                run = G.AudioRunner(cm, f"audio_{T}")
                for key, x, hi in audio_clip_inputs(T):
                    P, frames = hi["P"], hi["frames"]
                    out = run(x)
                    y = {k: v.copy() for k, v in x.items()}
                    y["mel"][0, :, frames:] = rng.standard_normal((128, T - frames)).astype(np.float32) * 5.0
                    out_p = run(y)
                    store[key] = out[:P].astype(np.float32)
                    clips[key] = {"T_b": T, "P": P, "frames": frames, "nonfinite_all_rows": int((~np.isfinite(out)).sum()),
                                  "pad_content_valid_rows_bit_equal": bool(np.array_equal(out_p[:P], out[:P]))}
                run.close()
                if hasattr(cm, "close"):
                    cm.close()
            info["status"] = "OK"
        except BaseException as e:  # recorded
            import traceback

            info["error"] = f"{type(e).__name__}: {e}"
            traceback.print_exc()
    info["clips"] = clips
    info["seconds_wall"] = round(time.time() - t0, 1)
    info["ru_maxrss_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    info["delegation"] = R.delegation_from_log(log)
    np.savez(RUNS / f"{tag}.npz", **store)
    rec_path.write_text(json.dumps(info, indent=1, default=str) + "\n")
    print(json.dumps({"tag": tag, "status": info["status"], "error": info.get("error"), "clips": list(clips),
                      "replacing": info["delegation"]["replacing"], "compile_seconds": info["compile_seconds"]}, indent=1))
    if info["status"] != "OK":
        return 1
    return 0 if a.child else audio_score(a)


def audio_store(label, form, key):
    """Round 7's prefix of the current file (same backend label and form, same host mel) for a store key."""
    if key == "long__host":
        p = R7 / f"audio_{label}_{form}_T3001.npz"
    else:
        p = R7 / f"audio_{label}_{form}.npz"
    if not p.exists():
        return None, str(p.relative_to(K))
    with np.load(p) as z:
        return (np.asarray(z[key]) if key in z.files else None), str(p.relative_to(K))


def audio_score(a) -> int:
    import audio_check as AC
    import litert_gate as LG
    import litert_run as R

    label = label_of(a.backend, a.precision)
    tag = f"audio_{label}_{a.variant}_{a.form}{a.suffix}"
    out_path = K / f"results/audio_parity_{label}_{a.variant}_{a.form}{a.suffix}.json"
    rec = json.loads((RUNS / f"{tag}.json").read_text())
    with np.load(RUNS / f"{tag}.npz") as z:
        st = {k: np.asarray(z[k]) for k in z.files}
    clips = {}
    for key, pre in st.items():
        c = dict(rec["clips"][key])
        ref, src = audio_store(label, a.form, key)
        c["vs_current_file_store"] = {"store": src, **(diff(pre, ref) if ref is not None else {"missing": True})}
        rid = key.split("__")[0]
        if rid in AUDIO_CLIPS:
            c["vs_oracle_prefix"] = diff(pre, AC.npz_of(rid)["prefix"])
        elif key == "aud_01_cut5__T501":
            c["vs_provider_prefix"] = diff(pre, np.load(CUT_REF)["provider"])
        elif key == "long__host":
            c["vs_provider_prefix"] = diff(pre, np.load(LONG_REF)["provider_prefix"])
        clips[key] = c
    doc = {"step": f"round 12: the audio graph variant {a.variant} ({a.form}) on the Mac, {label}", "scored_at": now(),
           "tag": tag, "run": {k: v for k, v in rec.items() if k != "clips"}, "clips": clips,
           "bar": {"store": "bit-equal to the current file's prefix (round 7 store, same backend / form) or max |d| <= 1e-5 "
                            "(f16safe); clast: max |d| <= 1e-4", "e2e": LG.BAR}}
    have = {k.split("__")[0] for k in st if k.endswith("__host")}
    filled = []
    if a.variant == "clast" and set(AUDIO_CLIPS) - have:
        # the clast variant exists at T1001 only: the clips of other buckets (card_topic, T2001) come from round 7's
        # store of the current file of the same backend / form, named in the record
        for rid in sorted(set(AUDIO_CLIPS) - have):
            ref, src = audio_store(label, a.form, f"{rid}__host")
            if ref is not None:
                st[f"{rid}__host"] = ref
                filled.append({"clip": rid, "from": src})
        have = {k.split("__")[0] for k in st if k.endswith("__host")}
    if set(AUDIO_CLIPS) <= have:
        rows, ometa = AC.oracle_audio_rows()
        ref = {r["key"]: (r["probs"], r["logits_raw"]) for r in rows}
        sources = LG.record_sources()
        text_file = K / "out/d1omni_decide_L256_fp16.tflite"
        cm, desc = R.open_compiled(text_file, "cpu", threads=8)
        trun = R.Runner(cm, next(iter(cm.get_signature_list())))
        lit = {}
        for r in rows:
            s = trun(AC.text_inputs(r, st[f"{r['id']}__host"], 256))
            lit[r["key"]] = LG.readout(s[:r["P"] + r["n"]], r)
        trun.close()
        e = LG.compare(rows, lit, ref, sources)
        keep = ("rows_compared", "max_abs_dp", "p95_abs_dp", "mean_abs_dp", "max_abs_dlogit", "argmax",
                "cutoff_crossings", "nonfinite_rows", "bar_pass", "top10_by_dp")
        r7 = K / f"results/audio_parity_{label}_{a.form}.json"
        r7s = json.loads(r7.read_text()).get("summary", {}) if r7.exists() else {}
        doc["e2e"] = {"text_graph": {"file": str(text_file.relative_to(K)), "backend": desc, "L": 256}, "oracle": ometa,
                      "clips_from_round7_store": filled,
                      "host_mel": {k: e[k] for k in keep},
                      "round7_current_file": {"file": str(r7.relative_to(K)), "e2e_host_mel_max_abs_dp": r7s.get("e2e_host_mel_max_abs_dp"),
                                              "e2e_bar_pass": r7s.get("e2e_bar_pass")},
                      "per_row": [{"key": r["key"], "probs": lit[r["key"]][0], "probs_oracle": r["probs"]} for r in rows]}
    cl = list(clips.values())
    stores = [c["vs_current_file_store"] for c in cl if not c["vs_current_file_store"].get("missing")]
    doc["summary"] = {
        "status": rec.get("status"), "clips": len(cl), "buckets": sorted({c["T_b"] for c in cl}),
        "vs_current_store_bit_equal": f"{sum(s['bit_equal'] for s in stores)}/{len(stores)}",
        "vs_current_store_max_abs": max((s["max_abs"] for s in stores), default=None),
        "vs_oracle_prefix_max_abs": max((c["vs_oracle_prefix"]["max_abs"] for c in cl if "vs_oracle_prefix" in c), default=None),
        "vs_provider_prefix_max_abs": max((c["vs_provider_prefix"]["max_abs"] for c in cl if "vs_provider_prefix" in c),
                                          default=None),
        "nonfinite": sum(c["nonfinite_all_rows"] for c in cl),
        "pad_content_bit_equal": f"{sum(c['pad_content_valid_rows_bit_equal'] for c in cl)}/{len(cl)}",
        "e2e_max_abs_dp": (doc.get("e2e") or {}).get("host_mel", {}).get("max_abs_dp"),
        "e2e_mean_abs_dp": (doc.get("e2e") or {}).get("host_mel", {}).get("mean_abs_dp"),
        "e2e_argmax": (doc.get("e2e") or {}).get("host_mel", {}).get("argmax"),
        "e2e_bar_pass": (doc.get("e2e") or {}).get("host_mel", {}).get("bar_pass"),
        "replacing": rec.get("delegation", {}).get("replacing"), "is_fully_accelerated": rec.get("is_fully_accelerated"),
        "compile_seconds": rec.get("compile_seconds")}
    write_new(out_path, doc)
    print(json.dumps({"out": str(out_path.relative_to(K)), **{k: v for k, v in doc["summary"].items() if k != "replacing"},
                      "replacing": doc["summary"]["replacing"][:6] if doc["summary"]["replacing"] else None}, indent=1,
                     default=str))
    return 0


# ---------------------------------------------------------------- vision

def vision_run(a) -> int:
    import importlib.metadata as md
    import resource

    import d1_vision_host as V
    import litert_run as R
    import vision_check as VC

    label = label_of(a.backend, a.precision)
    tag = f"vision_{label}_f16safe_{a.form}"
    rdir = RUNS / tag
    rec_path = RUNS / f"{tag}.json"
    if a.backend == "gpu" and not a.child:
        assert not rec_path.exists(), f"refusing to overwrite {rec_path}"
        rdir.mkdir(parents=True, exist_ok=True)
        argv = ["vision", "--form", a.form, "--backend", "gpu", "--precision", a.precision, "--child"]
        rc = child_parent(argv, rec_path, K / f"logs/r12_{tag}.child_stdio.log")
        return rc if rc else vision_score(a)
    assert a.child or not rec_path.exists(), f"refusing to overwrite {rec_path}"
    rdir.mkdir(parents=True, exist_ok=True)
    files = {"tower_variant": K / f"out/d1omni_vision_tower_f16safe_{a.form}.tflite",
             "tower_current": K / f"out/d1omni_vision_tower_{a.form}.tflite",
             "projector": K / f"out/d1omni_projector_{a.form}.tflite"}
    logs = {n: K / f"logs/r12_{tag}.{n}.log" for n in list(files) + ["runs"]}
    accel = "cpu" if a.backend == "cpu" else "gpu"
    info = {"tag": tag, "started": now(), "pid": os.getpid(), "backend": a.backend, "precision": a.precision,
            "form": a.form, "ai_edge_litert": md.version("ai-edge-litert"),
            "files": {n: {"file": str(p.relative_to(K)), "bytes": p.stat().st_size, "sha256": R.sha256_file(p)}
                      for n, p in files.items()}, "compile_seconds": {}, "status": "FAIL"}
    per, t0 = [], time.time()
    try:
        g = {}
        for n, p in files.items():
            with R.capture_fd2(logs[n]):
                if n == "tower_variant":
                    info["logger"] = R.runtime_log_verbose()
                t = time.time()
                g[n] = V.LiteRTGraph(p, accel, a.precision, threads=8)
                info["compile_seconds"][n] = round(time.time() - t, 2)
        info["is_fully_accelerated"] = {n: x.fully_accelerated for n, x in g.items()}
        table = V.read_position_table(S.WEIGHTS)
        recs, _ = VC.image_records()
        with R.capture_fd2(logs["runs"]):
            for e, rec in recs:
                rid = e["id"]
                crops, _ = V.crops_of(V.load_image(VC.image_path(rec)))
                pres, pres0, crow = [], [], []
                for i, c in enumerate(crops):
                    crop = V.to_patches(c)
                    h, w = crop["grid"]
                    n = h * w
                    x = V.tower_inputs(crop, table)
                    fv = g["tower_variant"](**x)[0]
                    f0 = g["tower_current"](**x)[0]
                    np.save(rdir / f"{rid}_c{i}_features.npy", fv[:n])
                    cells = V.pixel_unshuffle(fv[:n], (h, w))
                    pres.append(g["projector"](soft=V.projector_input(cells))[0][: cells.shape[0]])
                    cells0 = V.pixel_unshuffle(f0[:n], (h, w))
                    pres0.append(g["projector"](soft=V.projector_input(cells0))[0][: cells0.shape[0]])
                    crow.append({"crop": i, "grid": [h, w], "patches": n,
                                 "features_variant_vs_current": diff(fv[:n], f0[:n]),
                                 "features_variant_nonfinite_all_rows": int((~np.isfinite(fv)).sum())})
                pre, pre0 = np.concatenate(pres).astype(np.float32), np.concatenate(pres0).astype(np.float32)
                np.save(rdir / f"{rid}.npy", pre)
                np.save(rdir / f"{rid}__current_tower.npy", pre0)
                per.append({"id": rid, "P": int(pre.shape[0]), "crops": crow})
                print(rid, pre.shape, flush=True)
        for x in g.values():
            x.close()
        info["status"] = "OK"
    except BaseException as ex:  # recorded
        import traceback

        info["error"] = f"{type(ex).__name__}: {ex}"
        traceback.print_exc()
    info["records"] = per
    info["seconds_wall"] = round(time.time() - t0, 1)
    info["ru_maxrss_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    info["delegation"] = {n: R.delegation_from_log(p) for n, p in logs.items() if p.exists()}
    rec_path.write_text(json.dumps(info, indent=1, default=str) + "\n")
    print(json.dumps({"tag": tag, "status": info["status"], "error": info.get("error"),
                      "replacing": {n: d.get("replacing") for n, d in info["delegation"].items()}}, indent=1))
    if info["status"] != "OK":
        return 1
    return 0 if a.child else vision_score(a)


def vision_score(a) -> int:
    from types import SimpleNamespace

    import d1_host as H
    import litert_run as R
    import vision_check as VC

    label = label_of(a.backend, a.precision)
    tag = f"vision_{label}_f16safe_{a.form}"
    rdir = RUNS / tag
    out_path = K / f"results/vision_parity_{label}_f16safe_{a.form}.json"
    rec = json.loads((RUNS / f"{tag}.json").read_text())
    r6tag = f"{a.backend}_{'fp32' if a.backend == 'cpu' else a.precision}_{a.form}"
    recs, oracle = VC.image_records()
    rows = []
    for e, _ in recs:
        rid = e["id"]
        got, cur = np.load(rdir / f"{rid}.npy"), np.load(rdir / f"{rid}__current_tower.npy")
        r6 = R6 / r6tag / f"{rid}.npy"
        z = VC.npz(rid)
        tru = np.load(R6 / "truth" / f"{rid}.npy")
        rr = next(x for x in rec["records"] if x["id"] == rid)
        rows.append({"id": rid, "P": int(got.shape[0]), "vs_current_tower_same_run": diff(got, cur),
                     "vs_round6_store": {"store": str(r6.relative_to(K)), **diff(got, np.load(r6))} if r6.exists() else None,
                     "vs_oracle": diff(got, z["prefix"]), "vs_float64_truth": diff(got, tru),
                     "provider_fp32_vs_truth_rel_rms": diff(z["prefix"], tru)["rel_rms"],
                     "crops": rr["crops"]})
    temps = S.config()["temperatures"]
    qt = {"choice": 0, "score": 1, "noul": 2}
    by_L = {}
    for e, _ in recs:
        for q in e["questions"]:
            L = next(b for b in (256, 512, 1024, 2048, 4096) if q["positions"] <= b)
            by_L.setdefault(L, []).append((e, q))
    e2e_rows, graphs = [], {}
    for L, items in sorted(by_L.items()):
        cm, desc = R.open_compiled(K / f"out/d1omni_decide_L{L}_fp16.tflite", "cpu", threads=8)
        run = R.Runner(cm, next(iter(cm.get_signature_list())))
        for e, q in items:
            x = H.build_inputs(q["ids"], np.load(rdir / f"{e['id']}.npy"), L)
            oh = np.zeros((1, 3), np.float32)
            oh[0, qt[q["type"]]] = 1.0
            x["qtype_onehot"] = oh
            sc = run(x)
            pv = [float(v) for v in H.readout(sc, q["prefix"], q["markers"], SimpleNamespace(type=q["type"], options=q["K"]),
                                              bool(q["calibrate"]), temps)]
            po = q["probs"]
            srt = sorted(po, reverse=True)
            gap = srt[0] - srt[1] if len(srt) > 1 else 1.0
            e2e_rows.append({"key": f"{e['id']}/{q['qid']}", "L": L, "type": q["type"], "probs": pv, "probs_oracle": po,
                             "near_tie": gap <= 0.02, "argmax_equal": int(np.argmax(pv)) == int(np.argmax(po)),
                             "dp": [abs(u - v) for u, v in zip(pv, po)], "finite": bool(np.isfinite(sc[: q["positions"]]).all())})
        run.close()
        graphs[str(L)] = {"file": f"out/d1omni_decide_L{L}_fp16.tflite", "rows": len(items), "options": desc}
    dps = [d for r in e2e_rows for d in r["dp"]]
    main = [r for r in e2e_rows if not r["near_tie"]]
    st = {"rows": len(e2e_rows), "argmax_outside_near_tie": f"{sum(r['argmax_equal'] for r in main)}/{len(main)}",
          "near_tie_rows": len(e2e_rows) - len(main), "max_abs_dp": max(dps), "mean_abs_dp": float(np.mean(dps)),
          "p95_abs_dp": float(np.percentile(dps, 95)), "nonfinite_rows": sum(not r["finite"] for r in e2e_rows)}
    st["bar_pass"] = bool(all(r["argmax_equal"] for r in main) and st["max_abs_dp"] <= 0.02 and st["mean_abs_dp"] <= 0.002
                          and st["nonfinite_rows"] == 0)
    r6p = K / f"results/vision_parity_{r6tag}.json"
    r6e = (json.loads(r6p.read_text()).get("e2e") or {}).get("text_fp16_cpu", {}).get("summary") if r6p.exists() else None
    fe = [c["features_variant_vs_current"] for r in rows for c in r["crops"]]
    doc = {"step": f"round 12: the vision tower f16safe variant ({a.form}) on the Mac, {label}", "scored_at": now(),
           "tag": tag, "run": {k: v for k, v in rec.items() if k != "records"}, "records": rows,
           "e2e": {"text_graphs": graphs, "summary": st, "round6_current_file": {"file": str(r6p.relative_to(K)), "summary": r6e},
                   "per_row": e2e_rows},
           "bar": {"features": "bit-equal to the current tower (same run) or max |d| <= 1e-5 on the real patches",
                   "pass_fail": "e2e, FACTS §7 (16 image rows)"},
           "summary": {"status": rec.get("status"), "crops": len(fe),
                       "features_vs_current_bit_equal": f"{sum(x['bit_equal'] for x in fe)}/{len(fe)}",
                       "features_vs_current_max_abs": max(x["max_abs"] for x in fe),
                       "features_vs_current_rel_rms_max": max((x["rel_rms"] or 0) for x in fe),
                       "prefix_vs_current_tower_bit_equal": f"{sum(r['vs_current_tower_same_run']['bit_equal'] for r in rows)}/{len(rows)}",
                       "prefix_vs_round6_store_bit_equal": f"{sum(bool((r['vs_round6_store'] or {}).get('bit_equal')) for r in rows)}/{len(rows)}",
                       "prefix_vs_round6_store_max_abs": max(((r["vs_round6_store"] or {}).get("max_abs") or 0) for r in rows),
                       "prefix_vs_oracle_max_abs": max(r["vs_oracle"]["max_abs"] for r in rows),
                       "nonfinite": sum(c["features_variant_nonfinite_all_rows"] for r in rows for c in r["crops"]),
                       "e2e_max_abs_dp": st["max_abs_dp"], "e2e_mean_abs_dp": st["mean_abs_dp"],
                       "e2e_argmax": st["argmax_outside_near_tie"], "e2e_bar_pass": st["bar_pass"],
                       "replacing": {n: d.get("replacing") for n, d in rec.get("delegation", {}).items()},
                       "is_fully_accelerated": rec.get("is_fully_accelerated"), "compile_seconds": rec.get("compile_seconds")}}
    write_new(out_path, doc)
    print(json.dumps({"out": str(out_path.relative_to(K)), **doc["summary"]}, indent=1, default=str))
    return 0


# ---------------------------------------------------------------- ms (step 4)

MS_SETS = [   # name, kind, file, backend, precision (audio: the bucket's clip of round 7's ms, SPEED_CLIP)
    ("audio_T1001_current_fp16_gpu_fp32", "audio", "out/d1omni_audio_T1001_fp16.tflite", "gpu", "fp32"),
    ("audio_T1001_f16safe_fp16_gpu_fp32", "audio", "out/d1omni_audio_T1001_f16safe_fp16.tflite", "gpu", "fp32"),
    ("audio_T1001_clast_fp16_gpu_fp32", "audio", "out/d1omni_audio_T1001_clast_fp16.tflite", "gpu", "fp32"),
    ("audio_T1001_f16safe_fp16_gpu_default", "audio", "out/d1omni_audio_T1001_f16safe_fp16.tflite", "gpu", "default"),
    ("audio_T1001_current_fp16_gpu_default", "audio", "out/d1omni_audio_T1001_fp16.tflite", "gpu", "default"),
    ("vision_tower_current_fp16_gpu_fp32", "vision", "out/d1omni_vision_tower_fp16.tflite", "gpu", "fp32"),
    ("vision_tower_f16safe_fp16_gpu_fp32", "vision", "out/d1omni_vision_tower_f16safe_fp16.tflite", "gpu", "fp32"),
    ("vision_tower_f16safe_fp16_gpu_default", "vision", "out/d1omni_vision_tower_f16safe_fp16.tflite", "gpu", "default"),
] + [(f"audio_T{T}_{v}_fp16_gpu_{p}", "audio", f"out/d1omni_audio_T{T}{'_f16safe' if v == 'f16safe' else ''}_fp16.tflite",
      "gpu", p)
     for T in (501, 2001, 3001) for v, p in (("current", "fp32"), ("f16safe", "fp32"), ("f16safe", "default"))]
MS_CLIP = {501: ("aud_01", 5.0), 1001: ("aud_01", None), 2001: ("card_topic", None), 3001: ("long", None)}
# the second window (ms --window b -> results/timing_mac_r12b.json): the vision crop = tower + projector in one window
# (round 6's per-crop sum), and the 10 s audio question end to end with the f16safe audio graph on Metal default
# precision and the text graph on Metal fp32 precision (round 7's pipeline, row 77's workload)
MS_SETS_B = [
    ("vision_tower_current_fp16_gpu_fp32", "vision", "out/d1omni_vision_tower_fp16.tflite", "gpu", "fp32"),
    ("vision_tower_f16safe_fp16_gpu_fp32", "vision", "out/d1omni_vision_tower_f16safe_fp16.tflite", "gpu", "fp32"),
    ("projector_fp16_gpu_fp32", "projector", "out/d1omni_projector_fp16.tflite", "gpu", "fp32"),
    ("vision_tower_f16safe_fp16_gpu_default", "vision", "out/d1omni_vision_tower_f16safe_fp16.tflite", "gpu", "default"),
    ("projector_fp16_gpu_default", "projector", "out/d1omni_projector_fp16.tflite", "gpu", "default"),
    ("pipeline_T1001_f16safe_default_text_fp32", "pipeline", "out/d1omni_audio_T1001_f16safe_fp16.tflite", "gpu", "default"),
    ("pipeline_T1001_current_fp32_text_fp32", "pipeline", "out/d1omni_audio_T1001_fp16.tflite", "gpu", "fp32"),
]


def ms_child(a) -> int:
    """One set in a fresh process: compile, memory, 5 warm-up + 20 timed calls of the fixed input."""
    import audio_graph as G
    import d1_vision_host as V
    import litert_run as R
    import timing_mac as TM
    import vision_check as VC

    name, kind, rel, backend, precision = next(s for s in MS_SETS + MS_SETS_B if s[0] == a.set)
    path = K / rel
    doc = {"set": name, "pid": os.getpid(), "started": now(), "file": rel, "file_bytes": path.stat().st_size,
           "backend": backend, "precision": precision, "status": "FAIL"}
    try:
        doc["memory_before_compile"] = R.memory()
        t0 = time.perf_counter()
        if kind == "pipeline":
            # round 7's 10 s question (aud_01/topic) end to end in one process: host mel -> the audio graph T1001 (this
            # set's file and precision) -> the prefix rows -> the text inputs -> the text graph L256 fp16 on Metal fp32
            # precision (the text graph fails at the default precision, rounds 3 / 8) -> the host read-out
            import audio_check as AC
            import litert_gate as LG

            acm, desc = R.open_compiled(path, backend, precision, threads=8)
            doc["compile_seconds_audio"] = round(time.perf_counter() - t0, 3)
            t1 = time.perf_counter()
            tpath = K / "out/d1omni_decide_L256_fp16.tflite"
            tcm, tdesc = R.open_compiled(tpath, "gpu", "fp32", threads=8)
            doc["compile_seconds_text"] = round(time.perf_counter() - t1, 3)
            doc["compile_seconds"] = round(time.perf_counter() - t0, 3)
            desc = {"audio": desc, "text": tdesc, "text_file": str(tpath.relative_to(K))}
            arun = G.AudioRunner(acm, "audio_1001")
            trun = R.Runner(tcm, next(iter(tcm.get_signature_list())))
            rows, _ = AC.oracle_audio_rows()
            row = next(r for r in rows if r["key"] == "aud_01/topic")
            x16, _ = G.clip_samples(row["id"])
            doc["input"] = {"row": row["key"], "clip": row["id"]}
            parts_ms = []

            def call():
                wall = time.time() * 1000.0
                ta = time.perf_counter()
                xa, info = AH.prepare(x16)
                tb = time.perf_counter()
                out, _, _, _ = arun.timed({k: np.ascontiguousarray(v) for k, v in xa.items()})
                pre = out[:info["P"]]
                tc = time.perf_counter()
                xt = AC.text_inputs(row, pre, 256)
                td = time.perf_counter()
                s, _, _, _ = trun.timed(xt)
                te = time.perf_counter()
                p = LG.readout(s[:row["P"] + row["n"]], row)[0]
                tf = time.perf_counter()
                parts_ms.append([round((tb - ta) * 1e3, 3), round((tc - tb) * 1e3, 3), round((td - tc) * 1e3, 3),
                                 round((te - td) * 1e3, 3), round((tf - te) * 1e3, 3)])
                return np.asarray(p, np.float64), wall, (tf - ta) * 1e3, (tc - tb) * 1e3 + (te - td) * 1e3
        elif kind == "projector":
            g = V.LiteRTGraph(path, backend, precision, threads=8)
            doc["compile_seconds"] = round(time.perf_counter() - t0, 3)
            desc = {"accelerator": backend, "precision": precision}
            soft_fx = K / "device/r12/stage/fx_PJ/001_soft.f32"      # img_dogs_01/c0 (scripts/npu_fixtures.py write-specs)
            x = {"soft": np.fromfile(soft_fx, dtype="<f4").reshape(1, 256, 3072)}
            m = 84
            doc["input"] = {"crop": "img_dogs_01/c0", "soft": str(soft_fx.relative_to(K)), "prefix_rows": m}
            size = int(np.prod(g.output_shape))

            def call():
                wall = time.time() * 1000.0
                t = time.perf_counter()
                g.inputs["soft"].write(x["soft"])
                tr = time.perf_counter()
                g.model.run_by_name(g.signature, g.inputs, g.outputs)
                rn = (time.perf_counter() - tr) * 1000.0
                out = np.asarray(g.outputs[g.output_name].read(size, np.float32), np.float32).reshape(g.output_shape)
                return out[0, :m], wall, (time.perf_counter() - t) * 1000.0, rn
        elif kind == "audio":
            T = int(name.split("_")[1][1:])
            cm, desc = R.open_compiled(path, backend, precision, threads=8)
            doc["compile_seconds"] = round(time.perf_counter() - t0, 3)
            run = G.AudioRunner(cm, f"audio_{T}")
            rid, cut = MS_CLIP[T]
            x16, cname = G.long_samples() if rid == "long" else G.clip_samples(rid, cut)
            x, info = AH.prepare(x16, bucket=T)
            x = {k: np.ascontiguousarray(v) for k, v in x.items()}
            doc["input"] = {"clip": cname, "seconds_cut": cut, **info}
            ref = None

            def call():
                out, wall, tot, rn = run.timed(x)
                return out[:info["P"]], wall, tot, rn
        else:
            g = V.LiteRTGraph(path, backend, precision, threads=8)
            doc["compile_seconds"] = round(time.perf_counter() - t0, 3)
            desc = {"accelerator": backend, "precision": precision}
            recs, _ = VC.image_records()
            e, rec = next((e, r) for e, r in recs if e["id"] == "img_dogs_01")
            crops, _ = V.crops_of(V.load_image(VC.image_path(rec)))
            crop = V.to_patches(crops[0])
            n = crop["grid"][0] * crop["grid"][1]
            x = {k: np.ascontiguousarray(v, np.float32) for k, v in V.tower_inputs(crop, V.read_position_table(S.WEIGHTS)).items()}
            doc["input"] = {"crop": "img_dogs_01/c0", "grid": crop["grid"], "patches": n}
            size = int(np.prod(g.output_shape))

            def call():
                wall = time.time() * 1000.0
                t = time.perf_counter()
                for k2, v in x.items():
                    g.inputs[k2].write(v)
                tr = time.perf_counter()
                g.model.run_by_name(g.signature, g.inputs, g.outputs)
                rn = (time.perf_counter() - tr) * 1000.0
                out = np.asarray(g.outputs[g.output_name].read(size, np.float32), np.float32).reshape(g.output_shape)
                return out[0, :n], wall, (time.perf_counter() - t) * 1000.0, rn
        doc["options"] = desc
        doc["memory_after_compile"] = R.memory()
        warm = [[round(v, 3) for v in call()[1:]] for _ in range(WARMUP)]
        calls, finite, first = [], True, None
        for _ in range(REPS):
            out, wall, tot, rn = call()
            calls.append([round(wall, 3), round(tot, 3), round(rn, 3)])
            finite &= bool(np.isfinite(out).all())
            if first is None:
                first = out.copy()
        doc["warmup_calls"] = warm
        doc["calls"] = calls
        doc["call_columns"] = ["wall_clock_ms_at_start", "ms_write_run_read", "ms_run_only"]
        if kind == "pipeline":
            cols = ["mel_and_inputs", "audio_graph_T1001", "text_inputs", "text_graph_L256", "readout"]
            timed = parts_ms[WARMUP:]
            doc["call_columns"] = ["wall_clock_ms_at_start", "ms_total_question", "ms_audio_graph_plus_text_graph"]
            doc["parts_ms"] = {c: TM.stats([q[i] for q in timed]) for i, c in enumerate(cols)}
            doc["parts_columns"] = cols
            doc["parts_calls_ms"] = timed
            doc["probs_first_timed"] = [float(v) for v in first]
            doc["probs_oracle"] = row["probs"]
            doc["max_abs_dp_vs_oracle"] = float(max(abs(u - v) for u, v in zip(first, row["probs"])))
        doc["ms_write_run_read"] = TM.stats([c[1] for c in calls])
        doc["ms_run_only"] = TM.stats([c[2] for c in calls])
        doc["finite"] = finite
        doc["first_output_sha256"] = __import__("hashlib").sha256(np.ascontiguousarray(first).tobytes()).hexdigest()
        doc["memory_end"] = R.memory()
        doc["status"] = "OK"
    except BaseException as ex:  # recorded
        import traceback

        doc["error"] = f"{type(ex).__name__}: {ex}"
        doc["traceback"] = traceback.format_exc()[-3000:]
    doc["finished"] = now()
    Path(a.out).write_text(json.dumps(doc, indent=1, default=str) + "\n")
    return 0 if doc["status"] == "OK" else 1


def ms_window(a) -> int:
    import timing_mac as TM

    out = K / ("results/timing_mac_r12b.json" if a.window == "b" else "results/timing_mac_r12.json")
    assert not out.exists(), f"refusing to overwrite {out}"
    tmp = K / "out/r12_timing_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    doc = {"what": ("round 12 Mac timing, window b: the vision crop (tower current / f16safe + projector) and the 10 s audio "
                    "question end to end (audio T1001 f16safe on Metal default precision or the current file on Metal fp32 "
                    "precision, text L256 fp16 on Metal fp32 precision), one quiet window, a fresh child per set"
                    if a.window == "b" else
                    "round 12 Mac timing: the audio T501 / T1001 / T2001 / T3001 and vision tower files (current, f16safe, "
                    "clast) on Metal, one quiet window, a fresh child per set"),
           "protocol": f"{WARMUP} warm-up calls then {REPS} timed calls; 1 call = write the inputs + run + read the "
                       "output back; per call [wall clock ms at start, ms write + run + read, ms run only]",
           "gate_rule": "round 4: before each set no process outside this tree above 120 % CPU and CPU idle >= 50 % on "
                        "the second `top -l 2 -s 1` sample (10 s polls up to 300 s); load recorded",
           "window_lock_line_at_start": TM.lock_line(), "started": now(), "sets": {}}
    rc = 0
    sets = [s for s in (MS_SETS_B if a.window == "b" else MS_SETS) if not a.sets or s[0] in a.sets.split(",")]
    for name, kind, rel, backend, precision in sets:
        if not (K / rel).exists():
            doc["sets"][name] = {"status": f"skipped: {rel} missing"}
            continue
        gate = TM.load_gate()
        st = {"load_gate": gate}
        if not gate["ok"]:
            st["status"] = "discarded: contention for the whole wait"
            doc["sets"][name] = st
            rc = 3
            continue
        res = tmp / f"{name}.json"
        if res.exists():
            res.unlink()
        log = K / f"logs/r12_ms_{name}.child.log"
        t0 = time.time()
        with open(log, "w") as fo:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "ms-child", "--set", name, "--out",
                                      str(res)], stdout=fo, stderr=subprocess.STDOUT, cwd=str(K))
            _, status, ru = os.wait4(child.pid, 0)
        r = json.loads(res.read_text()) if res.exists() else {"status": "FAIL", "error": "child wrote no record"}
        r.update(child_returncode=os.waitstatus_to_exitcode(status), child_ru_maxrss_bytes=int(ru.ru_maxrss),
                 child_seconds_wall=round(time.time() - t0, 1), child_log=str(log.relative_to(K)))
        st.update(result=r, load_after={"load": TM.load_avg(), "at": now()},
                  status="measured" if r.get("status") == "OK" else f"failed: {r.get('error')}")
        doc["sets"][name] = st
        m = r.get("ms_write_run_read") or {}
        print(f"{name}: {st['status']} median {m.get('median')} (min {m.get('min')}, max {m.get('max')}) "
              f"compile {r.get('compile_seconds')} idle {gate.get('contention', {}).get('idle_pct')} load "
              f"{gate.get('load_at_start')}", flush=True)
    doc["finished"] = now()
    doc["window_lock_line_at_end"] = TM.lock_line()
    write_new(out, doc)
    print(json.dumps({"out": str(out.relative_to(K)), "rc": rc}))
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("audio", "audio-score"):
        p = sub.add_parser(name)
        p.add_argument("--variant", choices=("f16safe", "clast"), required=True)
        p.add_argument("--form", choices=("fp32", "fp16"), required=True)
        p.add_argument("--backend", choices=("cpu", "gpu"), required=True)
        p.add_argument("--precision", choices=("fp32", "default"), default="fp32")
        p.add_argument("--buckets", default="1001,2001,501,3001")
        p.add_argument("--suffix", default="", help="a tag suffix (the T1001-first run: _T1001first)")
        p.add_argument("--child", action="store_true")
    for name in ("vision", "vision-score"):
        p = sub.add_parser(name)
        p.add_argument("--form", choices=("fp32", "fp16"), required=True)
        p.add_argument("--backend", choices=("cpu", "gpu"), required=True)
        p.add_argument("--precision", choices=("fp32", "default"), default="fp32")
        p.add_argument("--child", action="store_true")
    p = sub.add_parser("ms")
    p.add_argument("--sets", default="")
    p.add_argument("--window", choices=("a", "b"), default="a")
    p = sub.add_parser("ms-child")
    p.add_argument("--set", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "audio":
        return audio_run(a)
    if a.cmd == "audio-score":
        return audio_score(a)
    if a.cmd == "vision":
        return vision_run(a)
    if a.cmd == "vision-score":
        return vision_score(a)
    if a.cmd == "ms":
        return ms_window(a)
    return ms_child(a)


if __name__ == "__main__":
    sys.exit(main())
