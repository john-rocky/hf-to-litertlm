#!/usr/bin/env python3
"""Does results/r1/fix.json hold for the 7.5, 15 and 25 s buckets? (venv .venv-ref, CPU torch fp32, cwd = phonon2_work, its scripts; no file but the record.)
p5_range.py per new bucket: the rescaled graph r4_graphs.EncB(P, t_mel, fix.json's k) with scripts/p1_diag.py's per-op hook, on the windows the
bucket takes under the 7-bucket rule (n in (the next smaller bucket's audio input, its own]): every S4 and S32 window there, and an even spread of LS test-other and
VoxPopuli windows from the shortest to the longest (--per-set per bucket), the mel from MelB(t_mel). Every op output is read over all frames (padded rows run on the
phone too). Per LayerNorm site: the max of (x 2^-k - mean)^2 and of its row mean; both must stay under 65,504 everywhere. Also every op above 2^14 outside the masked
attention terms (p1_diag.decide_fix's "other" list), fix.json's 30 s numbers and p5_range.py's 5 s numbers (results/r5/range5.json) for the side-by-side.
-> results/r7/range7.json. Exit 3 when a site reaches 65,504 in any bucket. CLI: p7_range.py [--threads 8] [--per-set 60]
"""
import argparse
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "phonon2_work", "scripts"))
import p7_common as C  # noqa: E402  (phonon2_work/scripts on sys.path)
import p1_diag as D  # noqa: E402
import p1_ref as R  # noqa: E402
import r2_drive as RD  # noqa: E402
import r4_graphs as G  # noqa: E402

LIMIT = 65504.0


def caps(b):
    smaller = [x for x in C.B7 if x < b]
    return (G.n_audio(C.T_ALL[max(smaller)]) if smaller else 0), G.n_audio(C.T_ALL[b])


def windows_of(man, tag, lo, hi, per_set=0):
    rows = []
    for r in R.read_jsonl(man):
        samples = int(round(r["seconds_written"] * 16000))
        for k, (s0, n) in enumerate(RD.windows(samples)):
            if lo < n <= hi:
                rows.append((f"{tag}:{r['id']}#{k}", r["path"], s0, n))
    if per_set and len(rows) > per_set:
        rows.sort(key=lambda x: x[3])
        step = (len(rows) - 1) / (per_set - 1)
        rows = [rows[round(i * step)] for i in range(per_set)]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--per-set", type=int, default=60)
    a = ap.parse_args()
    os.makedirs("results/r7", exist_ok=True)
    R.set_threads(a.threads)
    t0 = time.time()
    model, _ = R.build("P")
    mf = R.processor().feature_extractor.mel_filters
    fix = json.load(open("results/r1/fix.json"))
    lk, rm = G.fix_from_json(fix)
    pf = fix["post_fix"]
    per_site_r1 = pf["per_site_max_sqdiff"]
    r5 = json.load(open("results/r5/range5.json"))
    rep = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "limit": LIMIT, "per_set": a.per_set, "buckets": {},
           "r1_30s_post_fix": {"windows": pf["windows"], "max_sqdiff": pf["max_sqdiff"], "max_sqdiff_site": pf["max_sqdiff_site"], "max_row_var": pf["max_row_var"]},
           "r5_5s": {"windows": r5["windows"], "max_sqdiff": r5["max_sqdiff"], "max_sqdiff_site": r5["max_sqdiff_site"], "max_row_var": r5["max_row_var"]}}
    ok = True
    for b, tm in C.T_NEW.items():
        lo, hi = caps(b)
        enc = G.EncB(model, tm, ln_k=lk, res_m=rm).eval()
        mel_g = G.MelB(mf, tm).eval()
        wins = (windows_of("data/sets/librispeech-test-clean/manifest-S4.jsonl", "s4", lo, hi)
                + windows_of("data/sets/librispeech-test-clean/manifest-S32.jsonl", "s32", lo, hi)
                + windows_of("data/sets/librispeech-test-other/manifest.jsonl", "lo", lo, hi, a.per_set)
                + windows_of("data/sets/voxpopuli/manifest.jsonl", "vp", lo, hi, a.per_set))
        stats, counts, nonfinite, lens = {}, {}, 0, []
        tb = time.time()
        for tag, path, s0, n in wins:
            audio = R.load_wav(path)[s0:s0 + n]
            n = len(audio)
            x = torch.zeros(1, G.n_audio(tm))
            x[0, :n] = torch.from_numpy(audio)
            with torch.no_grad():
                mo = mel_g(x, torch.tensor([n], dtype=torch.int32))

            def sink(name, v):
                s = stats.setdefault(name, {"amax": 0.0})
                s["amax"] = max(s["amax"], D.amax(v))
                if name.endswith(D.MASKED):
                    s["amax_unmasked"] = max(s.get("amax_unmasked", 0.0), D.amax(v[v > -15000.0]))
            h = D.Hook("f32", sink)
            with torch.no_grad():
                out = enc(mo["features"], mo["mel_mask"], q=h)["enc"]
            nonfinite += int((~torch.isfinite(out)).sum())
            counts[tag.split(":")[0]] = counts.get(tag.split(":")[0], 0) + 1
            lens.append(n)
        sq = {k: v["amax"] for k, v in stats.items() if k.endswith(".sqdiff")}
        vr = {k: v["amax"] for k, v in stats.items() if k.endswith(".var")}
        worst = max(sq.items(), key=lambda kv: kv[1])
        over = [{"op": k, "max_abs": v.get("amax_unmasked", v["amax"])} for k, v in stats.items()
                if v.get("amax_unmasked", v["amax"]) > D.RES_LIMIT and not k.endswith((".sqdiff", ".var", ".var_eps", ".mean", ".rsqrt"))]
        ratio = sorted(((sq[k] / max(per_site_r1[k], 1e-30), k) for k in sq if k in per_site_r1), reverse=True)
        under = bool(worst[1] < LIMIT and max(vr.values()) < LIMIT)
        ok = ok and under and nonfinite == 0
        rep["buckets"][str(b)] = {"t_mel": tm, "n_audio_lo_exclusive": lo, "n_audio": hi, "windows": len(wins), "per_set": counts,
                                  "samples_min": min(lens), "samples_max": max(lens), "seconds": round(time.time() - tb, 1), "sites": len(sq),
                                  "max_sqdiff": worst[1], "max_sqdiff_site": worst[0], "max_row_var": max(vr.values()), "all_under_limit": under,
                                  "other_over_2p14": over, "nonfinite_enc_out": nonfinite, "headroom_bits": round(math.log2(LIMIT / worst[1]) / 2, 2),
                                  "top_sites_vs_r1": [{"site": k, "ratio": r, "sqdiff": sq[k], "sqdiff_r1": per_site_r1[k]} for r, k in ratio[:8]],
                                  "per_site_max_sqdiff": sq}
        print(b, json.dumps({k: v for k, v in rep["buckets"][str(b)].items() if k not in ("per_site_max_sqdiff", "top_sites_vs_r1")}), flush=True)
        del enc, mel_g
    rep["all_under_limit"] = ok
    rep["seconds"] = round(time.time() - t0, 1)
    json.dump(rep, open("results/r7/range7.json", "w"), indent=1)
    print("range7", "PASS" if ok else "FAIL", rep["seconds"], "s", flush=True)
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
