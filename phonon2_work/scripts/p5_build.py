#!/usr/bin/env python3
"""The 5 s bucket's export, torch part (venv .venv-lt, cwd = phonon2_work): the 5 s bucket by scripts/p1_build.py's form (the same graph classes, converter calls and
writer; p1_build.py itself is not edited), and in its list also the decoder and the fp32 10 / 20 / 30 s encoders (p1_build.py's graphs; --only picks what to make).
  mel5        r4_graphs.MelB(mel_filters, 504): audio [1, 80640], n_valid [1] -> features [1,128,504], mel_mask [1,504], frame_mask [1,63]
  enc5_fp32   r4_graphs.EncB(model, 504): features + mel_mask -> enc [1,63,640]; raw file + .ready, lifted by scripts/r2_fb.py lift
  enc5_fp32s  the same with the LayerNorm rescale of results/r1/fix.json
The 5 s bucket is 504 frames, not 500: EncB needs t_mel % 8 == 0 (three stride-2 stages). Writes out/r5/, results/r5/build.json (one record per run).
Before any conversion the torch checks of p1_build.py on the S32 windows that fit the bucket: EncB(504) on the extractor's mel and through MelB(504) vs P's stored
encoder output, MelB(504) vs the extractor's mel, masks; a check above its bar stops the script (exit 2). CLI: p5_build.py --only a,b,...
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p5_common  # noqa: E402,F401  (K/scripts on sys.path)
import p1_build as B  # noqa: E402
import p1_ref as R  # noqa: E402
import r2_graphs as G2  # noqa: E402
import r4_graphs as G  # noqa: E402

import litert_torch  # noqa: E402

T5 = p5_common.T5
G.BUCKETS[5] = T5
B.OUT = p5_common.OUT5
B.REC = "results/r5/build.json"


def checks_5s(model, mel_filters, ln_k, res_m):
    enc = G.EncB(model, T5).eval()
    fixd = G.EncB(model, T5, ln_k=ln_k, res_m=res_m).eval()
    mel_g = G.MelB(mel_filters, T5).eval()
    man = {r["id"]: r for r in R.read_jsonl("data/sets/librispeech-test-clean/manifest-S32.jsonl")}
    rows = []
    for f in B.s32_windows():
        stem = os.path.basename(f)[:-4]
        uid, k = (stem.split("__w") + ["0"])[:2]
        audio = R.load_wav(man[uid]["path"])
        s0, n = [(s, min(G.W30, len(audio) - s)) for s in range(0, len(audio), G.W30)][int(k)]
        if n > G.n_audio(T5):
            continue
        z = np.load(f)
        l0, ve = int(z["valid_mel"]), int(z["valid_enc"])
        x = torch.zeros(1, G.n_audio(T5))
        x[0, :n] = torch.from_numpy(audio[s0:s0 + n])
        with torch.no_grad():
            mo = mel_g(x, torch.tensor([n], dtype=torch.int32))
            feat_iso = torch.from_numpy(z["mel"][:, :T5].copy())[None]
            mm = (torch.arange(T5)[None] < l0).to(torch.float32)
            e_iso = enc(feat_iso, mm)["enc"][0]
            e_chain = enc(mo["features"], mo["mel_mask"])["enc"][0]
            e_fix = fixd(mo["features"], mo["mel_mask"])["enc"][0]
        ref = torch.from_numpy(z["enc_proj"])
        rows.append({"npz": stem, "samples": n, "valid_mel": l0, "valid_enc": ve,
                     "mel_vs_extractor": float((mo["features"][0, :, :l0] - feat_iso[0, :, :l0]).abs().max()),
                     "mel_mask_ok": int(mo["mel_mask"].sum()) == l0 and int(mo["frame_mask"].sum()) == ve,
                     "enc_iso_vs_P": float((e_iso[:ve] - ref[:ve]).abs().max()),
                     "enc_chain_vs_P": float((e_chain[:ve] - ref[:ve]).abs().max()),
                     "fixed_vs_plain": float((e_fix[:ve] - e_chain[:ve]).abs().max()),
                     "fixed_bit_equal": bool(torch.equal(e_fix, e_chain)), "finite": bool(torch.isfinite(e_chain).all())})
    bad = [r for r in rows if r["enc_iso_vs_P"] > 1e-3 or r["mel_vs_extractor"] > 1e-3 or not r["mel_mask_ok"]
           or r["fixed_vs_plain"] > 1e-5 or not r["finite"]]
    return {"windows": len(rows), "rows": rows, "worst_enc_iso_vs_P": max(r["enc_iso_vs_P"] for r in rows),
            "worst_enc_chain_vs_P": max(r["enc_chain_vs_P"] for r in rows), "worst_mel": max(r["mel_vs_extractor"] for r in rows),
            "fixed_bit_equal": sum(r["fixed_bit_equal"] for r in rows), "bad": [r["npz"] for r in bad], "pass": not bad and bool(rows)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    os.makedirs(B.OUT, exist_ok=True)
    os.makedirs("results/r5", exist_ok=True)
    fix = json.load(open("results/r1/fix.json"))
    ln_k, res_m = G.fix_from_json(fix)
    R.set_threads(8)
    t0 = time.time()
    model, lrep = R.build("P")
    mf = R.processor().feature_extractor.mel_filters
    rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "env": R.env_record(), "litert_torch": litert_torch.__version__,
           "pythonhashseed": os.environ.get("PYTHONHASHSEED"), "load_report": lrep, "load_s": round(time.time() - t0, 1), "t_mel_5s": T5,
           "ln_k_sites": len(ln_k), "res_m": res_m, "files": {}, "skipped": []}
    rec["torch_checks_5s"] = checks_5s(model, mf, ln_k, res_m)
    print("torch checks 5 s", json.dumps({k: v for k, v in rec["torch_checks_5s"].items() if k != "rows"}), flush=True)
    if not rec["torch_checks_5s"]["pass"]:
        with open(B.REC, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        print("TORCH CHECK FAIL: no conversion", flush=True)
        return 2

    def enc_kw(tm):
        return {"features": torch.zeros(1, 128, tm), "mel_mask": torch.ones(1, tm)}

    pred, joint = G2.DecPredict(model).eval(), G2.DecJoint(model).eval()
    plan = [("mel5", False, lambda: litert_torch.convert(
        G.MelB(mf, T5).eval(), sample_kwargs={"audio": torch.zeros(1, G.n_audio(T5)),
                                              "n_valid": torch.tensor([G.n_audio(T5) - 160], dtype=torch.int32)},
        enable_x64=False)),
        ("dec", False, lambda: litert_torch.signature(
            "predict", pred, sample_kwargs={"token": torch.tensor([[R.BLANK]], dtype=torch.int32),
                                            "h": torch.zeros(2, 1, 640), "c": torch.zeros(2, 1, 640)}).signature(
            "joint", joint, sample_kwargs={"enc_frame": torch.zeros(1, 640), "dec_out": torch.zeros(1, 640)}).convert(
            enable_x64=False))]
    for bk in (5, 30, 20, 10):
        tm = G.BUCKETS[bk]
        plan.append((f"enc{bk}_fp32", True, (lambda tm=tm: litert_torch.convert(
            G.EncB(model, tm).eval(), sample_kwargs=enc_kw(tm), enable_x64=False))))
    plan.append(("enc5_fp32s", True, lambda: litert_torch.convert(
        G.EncB(model, T5, ln_k=ln_k, res_m=res_m).eval(), sample_kwargs=enc_kw(T5), enable_x64=False)))
    only = set(x for x in a.only.split(",") if x)
    for name, raw, fn in plan:
        if only and name not in only:
            continue
        if B.done_already(name, raw):
            rec["skipped"].append({"name": name, "why": "exists"})
            continue
        B.write(name, fn, rec, raw=raw)
        with open(B.OUT + "/BUILD.progress", "a") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {name}\n")
    rec["total_s"] = round(time.time() - t0, 1)
    with open(B.REC, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    print("build done", rec["total_s"], "s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
