#!/usr/bin/env python3
"""parakeet-ultra r1 export, torch part (venv .venv-lt: litert-torch 0.9.4, torch 2.13, transformers 5.20.2): reference
U's weights into redux's graphs, written as fp32 .tflite by litert_torch.convert (no quantization, no NeMo). Adapted from
parakeet_redux_work/scripts/r4_build.py + r2_build.py (same graph classes, same converter calls; the weights are
ultra's): mel30 = r2_graphs.Mel30 and dec = r2_graphs.DecPredict / DecJoint (as redux r2 built them), mel10 / mel20 =
r4_graphs.MelB, the encoders = r4_graphs.EncB(t_mel) (pw convs as Linear, LayerNorm weight / bias as rank-3 constants,
float masks) with or without the LayerNorm rescale of results/r1/fix.json (fp32s).
Encoder files are written as out/r1/<name>_raw.tflite + .ready; u1_post.sh lifts them (.venv-q).

Order (each file skipped when it, its raw form or its .ready marker exists):
  mel30, mel10, mel20, melrasp, dec, enc30_fp32, enc20_fp32, enc10_fp32, enc30_fp32s, enc20_fp32s, enc10_fp32s
  (--only a,b,... restricts the list; melrasp = a 1,001-frame mel for the raspbfox alternative only)
Before any conversion, torch checks on S32 windows (out/s32_ii, mode ii): EncB(3000) vs U's stored encoder output
(<= 1e-4), the rescaled EncB(3000) vs the plain one (bit-equal expected), each bucket's EncB / MelB vs U's stored
output / the extractor mel on the first 3 S32 windows that fit the bucket, the decoder's predict / joint vs U's torch
decoder on 5 steps (<= 1e-5). A check above its bar stops the script (exit 2).
-> results/r1/build.json (one record per run, appended)
CLI: u1_build.py [--only names]
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
import r2_graphs as G2  # noqa: E402
import r4_graphs as G  # noqa: E402
import ultra_ref as R  # noqa: E402

import litert_torch  # noqa: E402

OUT = "out/r1"
REC = "results/r1/build.json"


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
    rep = {"enc30": [], "buckets": [], "dec": {}}
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
            o = model.get_audio_features(input_features=mel.transpose(1, 2), attention_mask=mm.bool(),
                                         output_attention_mask=True)
        ref = torch.from_numpy(z["enc_proj"])
        rep["enc30"].append({"npz": os.path.basename(f), "plain_vs_U_stored": float((a[:ve] - ref[:ve]).abs().max()),
                             "plain_vs_U_now": float((a[:ve] - o.pooler_output[0][:ve]).abs().max()),
                             "fixed_vs_plain": float((b[:ve] - a[:ve]).abs().max()),
                             "fixed_bit_equal": bool(torch.equal(a, b)), "finite": bool(torch.isfinite(a).all())})
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
                                   "enc_iso_vs_U": float((e_iso[:ve] - ref[:ve]).abs().max()),
                                   "enc_chain_vs_U": float((e_chain[:ve] - ref[:ve]).abs().max())})
            n_done += 1
            if n_done >= 3:
                break
        del enc, mel_g
    pred, joint = G2.DecPredict(model).eval(), G2.DecJoint(model).eval()
    d = model.decoder
    g = torch.Generator().manual_seed(2)
    h = torch.zeros(2, 1, 640)
    c = torch.zeros(2, 1, 640)
    worst = {"dec_out": 0.0, "h": 0.0, "c": 0.0, "logits": 0.0}
    with torch.no_grad():
        for tok in (R.BLANK, 3, 1000, 8191, 17):
            t = torch.tensor([[tok]], dtype=torch.int32)
            o = pred(t, h, c)
            lo, (h2, c2) = d.lstm(d.embedding(t.long()), (h, c))
            ref = d.decoder_projector(lo)[:, 0]
            worst["dec_out"] = max(worst["dec_out"], (o["dec_out"] - ref).abs().max().item())
            worst["h"] = max(worst["h"], (o["h_out"] - h2).abs().max().item())
            worst["c"] = max(worst["c"], (o["c_out"] - c2).abs().max().item())
            ef = torch.randn(1, 640, generator=g)
            lg = joint(ef, ref)["logits"]
            lref = model.joint.head(model.joint.activation(ef + ref))
            worst["logits"] = max(worst["logits"], (lg - lref).abs().max().item())
            h, c = h2, c2
    rep["dec"] = worst
    bad = [r for r in rep["enc30"] if r["plain_vs_U_stored"] > 1e-4 or r["fixed_vs_plain"] > 1e-5
           or not r["finite"]]
    # bucket guard 1e-3, not redux's 1e-4: on ultra U's own fp32 is 1.2e-4 from its float64 and EncB(1000) fp32 5.6e-4 from
    # its float64 while the two graphs agree to 6.5e-13 in float64 (results/r1/f64_floor.json, scripts/u1_f64.py) = the
    # fp32 noise floor of this model (block 0 input |x| ~ 8e3) is above 1e-4; the gate of the graph is P0 (tokens)
    bad += [r for r in rep["buckets"] if r["enc_iso_vs_U"] > 1e-3 or r["mel_vs_extractor"] > 1e-3
            or not r["mel_mask_ok"]]
    bad += [] if {r["bucket"] for r in rep["buckets"]} == {10, 20} else ["a bucket with no S32 window"]
    bad += [("dec", worst)] if max(worst.values()) > 1e-5 else []
    rep["pass"] = not bad
    return rep


def done_already(name, raw):
    p = os.path.join(OUT, name)
    if raw:
        return any(os.path.exists(x) for x in (p + ".tflite", p + "_raw.tflite.ready", p + "_raw.tflite"))
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
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    fix = json.load(open("results/r1/fix.json")) if os.path.exists("results/r1/fix.json") else {}
    usable = bool(fix.get("usable"))
    ln_k, res_m = G.fix_from_json(fix) if usable else ({}, {})
    R.set_threads(8)
    t0 = time.time()
    model, lrep = R.build("U")
    proc = R.processor()
    mf = proc.feature_extractor.mel_filters
    rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "env": R.env_record(), "litert_torch": litert_torch.__version__,
           "load_report": lrep, "load_s": round(time.time() - t0, 1), "fix_usable": usable,
           "ln_k_sites": len(ln_k), "res_m": res_m, "files": {}, "skipped": []}
    rec["torch_checks"] = torch_checks(model, mf, ln_k, res_m)
    print("torch checks", json.dumps(rec["torch_checks"])[:2000], flush=True)
    if not rec["torch_checks"]["pass"]:
        with open(REC, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        print("TORCH CHECK FAIL: no conversion", flush=True)
        return 2

    def enc_kw(tm):
        return {"features": torch.zeros(1, 128, tm), "mel_mask": torch.ones(1, tm)}

    plan = [("mel30", False, lambda: litert_torch.convert(
        G2.Mel30(mf).eval(), sample_kwargs={"audio": torch.zeros(1, G2.N_AUDIO),
                                            "n_valid": torch.tensor([G2.N_AUDIO - 160], dtype=torch.int32)},
        enable_x64=False))]
    for bk in (10, 20):
        tm = G.BUCKETS[bk]
        plan.append((f"mel{bk}", False, (lambda tm=tm: litert_torch.convert(
            G.MelB(mf, tm).eval(), sample_kwargs={"audio": torch.zeros(1, G.n_audio(tm)),
                                                   "n_valid": torch.tensor([G.n_audio(tm) - 160], dtype=torch.int32)},
            enable_x64=False))))
    # melrasp: MelB with 1,001 frames (160,000 samples + the centered frame past the end), the input length of the
    # alternative raspbfox/parakeet-ultra-litert-dynamic-int8 encoder ([1, 128, 1001]); not a file of this lane's forms
    plan.append(("melrasp", False, lambda: litert_torch.convert(
        G.MelB(mf, 1001).eval(), sample_kwargs={"audio": torch.zeros(1, G.n_audio(1001)),
                                                 "n_valid": torch.tensor([160000], dtype=torch.int32)},
        enable_x64=False)))
    pred, joint = G2.DecPredict(model).eval(), G2.DecJoint(model).eval()
    plan.append(("dec", False, lambda: litert_torch.signature(
        "predict", pred, sample_kwargs={"token": torch.tensor([[R.BLANK]], dtype=torch.int32),
                                        "h": torch.zeros(2, 1, 640), "c": torch.zeros(2, 1, 640)}).signature(
        "joint", joint, sample_kwargs={"enc_frame": torch.zeros(1, 640), "dec_out": torch.zeros(1, 640)}).convert(
        enable_x64=False)))
    for bk in (30, 20, 10):
        tm = G.BUCKETS[bk]
        plan.append((f"enc{bk}_fp32", True, (lambda tm=tm: litert_torch.convert(
            G.EncB(model, tm).eval(), sample_kwargs=enc_kw(tm), enable_x64=False))))
    if usable:
        for bk in (30, 20, 10):
            tm = G.BUCKETS[bk]
            plan.append((f"enc{bk}_fp32s", True, (lambda tm=tm: litert_torch.convert(
                G.EncB(model, tm, ln_k=ln_k, res_m=res_m).eval(), sample_kwargs=enc_kw(tm), enable_x64=False))))
    only = set(x for x in a.only.split(",") if x)
    for name, raw, fn in plan:
        if only and name not in only:
            continue
        if done_already(name, raw):
            rec["skipped"].append({"name": name, "why": "exists"})
            continue
        write(name, fn, rec, raw=raw)
        with open(OUT + "/BUILD.progress", "a") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {name}\n")
    rec["total_s"] = round(time.time() - t0, 1)
    with open(REC, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    print("build done", rec["total_s"], "s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
