#!/usr/bin/env python3
"""r4 files, torch part (venv .venv-lt: litert-torch 0.9.4, torch 2.13, transformers 5.20.2): reference B's weights
into the r4 graphs of r4_graphs.py, written as fp32 .tflite by litert_torch.convert (no quantization, no NeMo).
Every encoder file is written as <name>_raw.tflite (+ .ready) and lifted / packed / halved by r4_post.sh (.venv-q).

Order (each file skipped when it, its lifted form or its .ready marker exists; no new file is started when less than
--min-left seconds remain before --deadline):
  enc30_fp32s   EncB(3000) with the rescale of results/r4/fix.json (skipped when fix.json says usable: false)
  mel10, mel20  MelB(1000) / MelB(2000) (final names, no lift: no BATCH_MATMUL in the mel graph)
  enc10_fp32, enc20_fp32      EncB(1000) / EncB(2000), no rescale (the int4x source, P0 of the buckets)
  enc20_fp32s, enc10_fp32s    the same with the rescale (the fp16ws GPU source)
  enc30_taps    EncB(3000, taps) no rescale: + outputs sub, b00..b23 (the per-layer GPU check)
  enc30_tapss   EncB(3000, taps) with the rescale
Before any conversion, torch checks on S32 windows (out/s32_ii, mode ii): the rescaled EncB(3000) vs the plain one
(bit-equal expected), EncB(3000) vs B's stored output, each bucket EncB / MelB vs B's stored encoder output / the
extractor mel on the first 3 S32 windows that fit the bucket. A check above its bar stops the script (exit 2).
-> results/r4/build.json (one record per run, appended)
CLI: r4_build.py [--deadline EPOCH] [--min-left 150] [--small]
  --small = dry run outside the window: r4_small.py's random model instead of B, no torch checks (not the model).
"""
import argparse
import gc
import glob
import hashlib
import json
import os
import resource
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r4_graphs as G  # noqa: E402
import redux_ref as R  # noqa: E402

import litert_torch  # noqa: E402

OUT = "out/r4"
REC = "results/r4/build.json"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def s32_windows():
    man = R.read_jsonl("data/sets/librispeech-test-clean/manifest-S32.jsonl")
    order = {r["id"]: k for k, r in enumerate(man)}
    return sorted(glob.glob("out/s32_ii/*.npz"),
                  key=lambda f: (order.get(os.path.basename(f)[:-4].split("__w")[0], 1e9), f))


def torch_checks(model, mel_filters, ln_k, res_m):
    rep = {"enc30": [], "buckets": []}
    files = s32_windows()
    plain = G.EncB(model, 3000).eval()
    fixd = G.EncB(model, 3000, ln_k=ln_k, res_m=res_m).eval() if (ln_k or res_m) else None
    for f in files[:3]:
        z = np.load(f)
        mel = torch.from_numpy(z["mel"])[None]
        l0, ve = int(z["valid_mel"]), int(z["valid_enc"])
        mm = (torch.arange(3000)[None] < l0).to(torch.float32)
        with torch.no_grad():
            a = plain(mel, mm)["enc"][0]
            b = fixd(mel, mm)["enc"][0] if fixd is not None else a
        ref = torch.from_numpy(z["enc_proj"])
        rep["enc30"].append({"npz": os.path.basename(f), "plain_vs_B": float((a[:ve] - ref[:ve]).abs().max()),
                             "fixed_vs_plain": float((b[:ve] - a[:ve]).abs().max()),
                             "fixed_bit_equal": bool(torch.equal(a, b))})
    del plain, fixd
    man = {r["id"]: r for r in R.read_jsonl("data/sets/librispeech-test-clean/manifest-S32.jsonl")}
    for bk in (10, 20):
        tm = G.BUCKETS[bk]
        enc = G.EncB(model, tm).eval()
        mel_g = G.MelB(mel_filters, tm).eval()
        n_done = 0
        for f in files:
            stem = os.path.basename(f)[:-4]
            uid, k = (stem.split("__w") + ["0"])[:2]
            audio = R.load_wav(man[uid]["path"])
            s0, n = [(s, min(G.W30, len(audio) - s)) for s in range(0, len(audio), G.W30)][int(k)]
            if G.bucket_of(n) != bk:
                continue
            z = np.load(f)
            l0, ve = int(z["valid_mel"]), int(z["valid_enc"])
            x = torch.zeros(1, G.n_audio(tm))
            x[0, :n] = torch.from_numpy(audio[s0:s0 + n])
            with torch.no_grad():
                mo = mel_g(x, torch.tensor([n], dtype=torch.int32))
                feat_iso = torch.from_numpy(z["mel"][:, :tm].copy())[None]
                mm = (torch.arange(tm)[None] < l0).to(torch.float32)
                e_iso = enc(feat_iso, mm)["enc"][0]
                e_chain = enc(mo["features"], mo["mel_mask"])["enc"][0]
            ref = torch.from_numpy(z["enc_proj"])
            rep["buckets"].append({"bucket": bk, "npz": stem, "samples": n, "valid_mel": l0, "valid_enc": ve,
                                   "mel_vs_extractor": float((mo["features"][0, :, :l0] - feat_iso[0, :, :l0])
                                                             .abs().max()),
                                   "mel_mask_ok": int(mo["mel_mask"].sum()) == l0
                                   and int(mo["frame_mask"].sum()) == ve,
                                   "enc_iso_vs_B": float((e_iso[:ve] - ref[:ve]).abs().max()),
                                   "enc_chain_vs_B": float((e_chain[:ve] - ref[:ve]).abs().max())})
            n_done += 1
            if n_done >= 3:
                break
        del enc, mel_g
    bad = [r for r in rep["enc30"] if r["plain_vs_B"] > 1e-4 or r["fixed_vs_plain"] > 1e-5]
    bad += [r for r in rep["buckets"] if r["enc_iso_vs_B"] > 1e-4 or r["mel_vs_extractor"] > 1e-3
            or not r["mel_mask_ok"]]
    bad += [] if {r["bucket"] for r in rep["buckets"]} == {10, 20} else ["a bucket with no S32 window"]
    rep["pass"] = not bad
    return rep


def done_already(name, raw):
    """raw encoder files: the lifted file (moved in whole by r4_post), its raw .ready or a .consumed marker (the taps
    files are deleted after their check); mel files: their .ready (written after the file is complete)."""
    p = os.path.join(OUT, name)
    if raw:
        return any(os.path.exists(x) for x in (p + ".tflite", p + "_raw.tflite.ready", p + ".consumed"))
    return os.path.exists(p + ".tflite.ready")


def write(name, fn, rec, raw=True):
    path = os.path.join(OUT, name + ("_raw" if raw else "") + ".tflite")
    t0 = time.time()
    em = fn()
    t1 = time.time()
    em.export(path)
    rec["files"][name] = {"path": path, "convert_s": round(t1 - t0, 1), "write_s": round(time.time() - t1, 1),
                          "bytes": os.path.getsize(path), "sha256": sha256(path),
                          "maxrss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30, 2)}
    open(path + ".ready", "w").write(time.strftime("%H:%M:%S") + "\n")
    print(name, json.dumps(rec["files"][name]), flush=True)
    del em
    gc.collect()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deadline", type=float, default=0.0)
    ap.add_argument("--min-left", type=float, default=150.0)
    ap.add_argument("--small", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    fix = json.load(open("results/r4/fix.json"))
    usable = bool(fix.get("usable"))
    ln_k, res_m = G.fix_from_json(fix) if usable else ({}, {})
    R.set_threads(8)
    t0 = time.time()
    if a.small:
        import r4_small
        model = r4_small.small_model_640()
        ln_k = {k: v for k, v in ln_k.items() if k[0] < len(model.encoder.layers)}
        res_m = {k: v for k, v in res_m.items() if k < len(model.encoder.layers)}
    else:
        model, rep = R.build("B")
    proc = R.processor()
    mf = proc.feature_extractor.mel_filters
    rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "env": R.env_record(), "litert_torch": litert_torch.__version__,
           "load_s": round(time.time() - t0, 1), "fix_usable": usable, "ln_k_sites": len(ln_k), "res_m": res_m,
           "small": a.small, "files": {}, "skipped": []}
    rec["torch_checks"] = {"pass": True, "skipped": "small"} if a.small else torch_checks(model, mf, ln_k, res_m)
    print("torch checks", json.dumps(rec["torch_checks"])[:1500], flush=True)
    if not rec["torch_checks"]["pass"]:
        with open(REC, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        print("TORCH CHECK FAIL: no conversion", flush=True)
        return 2

    def enc_kw(tm):
        return {"features": torch.zeros(1, 128, tm), "mel_mask": torch.ones(1, tm)}

    plan = []
    if usable:
        plan.append(("enc30_fp32s", True, lambda: litert_torch.convert(
            G.EncB(model, 3000, ln_k=ln_k, res_m=res_m).eval(), sample_kwargs=enc_kw(3000), enable_x64=False)))
    for bk in (10, 20):
        tm = G.BUCKETS[bk]
        plan.append((f"mel{bk}", False, (lambda tm=tm: litert_torch.convert(
            G.MelB(mf, tm).eval(), sample_kwargs={"audio": torch.zeros(1, G.n_audio(tm)),
                                                   "n_valid": torch.tensor([G.n_audio(tm) - 160], dtype=torch.int32)},
            enable_x64=False))))
    for bk in (10, 20):
        tm = G.BUCKETS[bk]
        plan.append((f"enc{bk}_fp32", True, (lambda tm=tm: litert_torch.convert(
            G.EncB(model, tm).eval(), sample_kwargs=enc_kw(tm), enable_x64=False))))
    if usable:
        for bk in (20, 10):
            tm = G.BUCKETS[bk]
            plan.append((f"enc{bk}_fp32s", True, (lambda tm=tm: litert_torch.convert(
                G.EncB(model, tm, ln_k=ln_k, res_m=res_m).eval(), sample_kwargs=enc_kw(tm), enable_x64=False))))
    plan.append(("enc30_taps", True, lambda: litert_torch.convert(
        G.EncB(model, 3000, taps=True).eval(), sample_kwargs=enc_kw(3000), enable_x64=False)))
    if usable:
        plan.append(("enc30_tapss", True, lambda: litert_torch.convert(
            G.EncB(model, 3000, ln_k=ln_k, res_m=res_m, taps=True).eval(), sample_kwargs=enc_kw(3000),
            enable_x64=False)))
    for name, raw, fn in plan:
        if done_already(name, raw):
            rec["skipped"].append({"name": name, "why": "exists"})
            continue
        if a.deadline and a.deadline - time.time() < a.min_left:
            rec["skipped"].append({"name": name, "why": "deadline"})
            continue
        write(name, fn, rec, raw=raw)
        with open(OUT + "/BUILD.progress", "a") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {name}\n")
    rec["total_s"] = round(time.time() - t0, 1)
    with open(REC, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    left = [s["name"] for s in rec["skipped"] if s["why"] == "deadline"]
    print("build done", rec["total_s"], "s; left for the next window:", left, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
