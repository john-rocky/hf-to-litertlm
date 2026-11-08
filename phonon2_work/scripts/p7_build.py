#!/usr/bin/env python3
"""The 7.5, 15 and 25 s export, torch part (venv .venv-lt, cwd = phonon2_work): the 7.5, 15 and 25 s buckets by scripts/p5_build.py's form (the same
graph classes, converter calls and writer of scripts/p1_build.py; neither is edited).
  mel{7.5,15,25}        r4_graphs.MelB(mel_filters, t_mel): audio [1, 160 t_mel], n_valid [1] -> features [1,128,t_mel], mel_mask [1,t_mel], frame_mask [1,t_mel/8]
  enc{7.5,15,25}_fp32   r4_graphs.EncB(model, t_mel): features + mel_mask -> enc [1,t_mel/8,640]; raw file + .ready, lifted by scripts/r2_fb.py lift
  enc{7.5,15,25}_fp32s  the same with the LayerNorm rescale of results/r1/fix.json
t_mel = 752 / 1,504 / 2,504 (p7_common.T_NEW: the smallest multiple of 8 at or above 100 x the seconds). Writes out/r7/, results/r7/build.json (one record per run).
Before any conversion the torch checks of p5_build.py on the S32 windows each new bucket takes under the 7-bucket rule (n in (the next smaller bucket's audio input,
its own]): EncB(t_mel) on the extractor's mel and through MelB(t_mel) vs P's stored encoder output, MelB vs the extractor's mel, masks, the rescaled graph vs the
plain one; a check above its bar, or a bucket with no S32 window, stops the script (exit 2). CLI: p7_build.py [--only a,b,...]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p7_common as C  # noqa: E402  (K/scripts on sys.path)
import p1_build as B  # noqa: E402
import p1_ref as R  # noqa: E402
import r4_graphs as G  # noqa: E402

import litert_torch  # noqa: E402

G.BUCKETS.update(C.T_ALL)
B.OUT = C.OUT7
B.REC = "results/r7/build.json"


def lower_cap(b):
    """the audio input of the next smaller bucket of the 7 (0 for 5 s)"""
    smaller = [x for x in C.B7 if x < b]
    return G.n_audio(C.T_ALL[max(smaller)]) if smaller else 0


def checks_new(model, mel_filters, ln_k, res_m):
    man = {r["id"]: r for r in R.read_jsonl("data/sets/librispeech-test-clean/manifest-S32.jsonl")}
    rows = []
    graphs = {}
    for b, tm in C.T_NEW.items():
        graphs[b] = (G.EncB(model, tm).eval(), G.EncB(model, tm, ln_k=ln_k, res_m=res_m).eval(), G.MelB(mel_filters, tm).eval())
    for f in B.s32_windows():
        stem = os.path.basename(f)[:-4]
        uid, k = (stem.split("__w") + ["0"])[:2]
        audio = R.load_wav(man[uid]["path"])
        s0, n = [(s, min(G.W30, len(audio) - s)) for s in range(0, len(audio), G.W30)][int(k)]
        b = next((x for x in C.T_NEW if lower_cap(x) < n <= G.n_audio(C.T_NEW[x])), None)
        if b is None:
            continue
        tm = C.T_NEW[b]
        enc, fixd, mel_g = graphs[b]
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
            e_fix = fixd(mo["features"], mo["mel_mask"])["enc"][0]
        ref = torch.from_numpy(z["enc_proj"])
        rows.append({"npz": stem, "bucket": b, "samples": n, "valid_mel": l0, "valid_enc": ve,
                     "mel_vs_extractor": float((mo["features"][0, :, :l0] - feat_iso[0, :, :l0]).abs().max()),
                     "mel_mask_ok": int(mo["mel_mask"].sum()) == l0 and int(mo["frame_mask"].sum()) == ve,
                     "enc_iso_vs_P": float((e_iso[:ve] - ref[:ve]).abs().max()),
                     "enc_chain_vs_P": float((e_chain[:ve] - ref[:ve]).abs().max()),
                     "fixed_vs_plain": float((e_fix[:ve] - e_chain[:ve]).abs().max()),
                     "fixed_bit_equal": bool(torch.equal(e_fix, e_chain)), "finite": bool(torch.isfinite(e_chain).all())})
    bad = [r["npz"] for r in rows if r["enc_iso_vs_P"] > 1e-3 or r["mel_vs_extractor"] > 1e-3 or not r["mel_mask_ok"]
           or r["fixed_vs_plain"] > 1e-5 or not r["finite"]]
    per = {str(b): sum(1 for r in rows if r["bucket"] == b) for b in C.T_NEW}
    bad += [f"bucket {b}: no S32 window" for b, c in per.items() if c == 0]
    return {"windows": len(rows), "per_bucket": per, "rows": rows,
            "worst_enc_iso_vs_P": max((r["enc_iso_vs_P"] for r in rows), default=None),
            "worst_enc_chain_vs_P": max((r["enc_chain_vs_P"] for r in rows), default=None),
            "worst_mel": max((r["mel_vs_extractor"] for r in rows), default=None),
            "fixed_bit_equal": sum(r["fixed_bit_equal"] for r in rows), "bad": bad, "pass": not bad and bool(rows)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    os.makedirs(B.OUT, exist_ok=True)
    os.makedirs("results/r7", exist_ok=True)
    fix = json.load(open("results/r1/fix.json"))
    ln_k, res_m = G.fix_from_json(fix)
    R.set_threads(8)
    t0 = time.time()
    model, lrep = R.build("P")
    mf = R.processor().feature_extractor.mel_filters
    rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "env": R.env_record(), "litert_torch": litert_torch.__version__,
           "pythonhashseed": os.environ.get("PYTHONHASHSEED"), "load_report": lrep, "load_s": round(time.time() - t0, 1),
           "t_mel": {str(k): v for k, v in C.T_NEW.items()}, "ln_k_sites": len(ln_k), "res_m": res_m, "files": {}, "skipped": []}
    rec["torch_checks_new"] = checks_new(model, mf, ln_k, res_m)
    print("torch checks 7.5 / 15 / 25 s", json.dumps({k: v for k, v in rec["torch_checks_new"].items() if k != "rows"}), flush=True)
    if not rec["torch_checks_new"]["pass"]:
        with open(B.REC, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        print("TORCH CHECK FAIL: no conversion", flush=True)
        return 2

    def enc_kw(tm):
        return {"features": torch.zeros(1, 128, tm), "mel_mask": torch.ones(1, tm)}

    plan = []
    for bk, tm in C.T_NEW.items():
        plan.append((f"mel{bk}", False, (lambda tm=tm: litert_torch.convert(
            G.MelB(mf, tm).eval(), sample_kwargs={"audio": torch.zeros(1, G.n_audio(tm)),
                                                  "n_valid": torch.tensor([G.n_audio(tm) - 160], dtype=torch.int32)},
            enable_x64=False))))
    for bk, tm in C.T_NEW.items():
        plan.append((f"enc{bk}_fp32", True, (lambda tm=tm: litert_torch.convert(
            G.EncB(model, tm).eval(), sample_kwargs=enc_kw(tm), enable_x64=False))))
        plan.append((f"enc{bk}_fp32s", True, (lambda tm=tm: litert_torch.convert(
            G.EncB(model, tm, ln_k=ln_k, res_m=res_m).eval(), sample_kwargs=enc_kw(tm), enable_x64=False))))
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
