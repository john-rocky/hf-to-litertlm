#!/usr/bin/env python3
"""Does results/r1/fix.json hold for the 5 s bucket? (venv .venv-ref, CPU torch fp32, cwd = phonon2_work, its scripts; no file but the record.) The rescaled graph
r4_graphs.EncB(P, 504, fix.json's k) with scripts/p1_diag.py's per-op hook, on the windows the bucket would take (n <= 80,640 samples: the S4 and S32 windows, and a
spread of LS test-other and VoxPopuli windows from the shortest to the longest), the mel from MelB(504). Every op output is read over all frames (padded rows run on the
phone too). Per LayerNorm site: the max of (x 2^-k - mean)^2 and of its row mean; both must stay under 65,504 everywhere. Also every op above 2^14 outside
the masked attention terms (p1_diag.decide_fix's "other" list) and the same numbers of fix.json's own windows on the 30 s graph for the side-by-side.
-> results/r5/range5.json. Exit 3 when a site reaches 65,504. CLI: p5_range.py [--threads 8] [--per-set 150]
"""
import argparse
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "phonon2_work", "scripts"))
import p5_common  # noqa: E402,F401  (phonon2_work/scripts on sys.path)
import p1_diag as D  # noqa: E402
import p1_ref as R  # noqa: E402
import r2_drive as RD  # noqa: E402
import r4_graphs as G  # noqa: E402

T5 = p5_common.T5
N5 = G.n_audio(T5)
LIMIT = 65504.0


def windows_of(man, tag, per_set=0):
    """[(tag:id#k, path, s0, n)] the windows of the manifest that fit the bucket; per_set > 0: an even spread by length, the shortest and longest kept"""
    rows = []
    for r in R.read_jsonl(man):
        samples = int(round(r["seconds_written"] * 16000))
        for k, (s0, n) in enumerate(RD.windows(samples)):
            if n <= N5:
                rows.append((f"{tag}:{r['id']}#{k}", r["path"], s0, n))
    if per_set and len(rows) > per_set:
        rows.sort(key=lambda x: x[3])
        step = (len(rows) - 1) / (per_set - 1)
        rows = [rows[round(i * step)] for i in range(per_set)]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--per-set", type=int, default=150)
    a = ap.parse_args()
    os.makedirs("results/r5", exist_ok=True)
    R.set_threads(a.threads)
    t0 = time.time()
    model, _ = R.build("P")
    mf = R.processor().feature_extractor.mel_filters
    fix = json.load(open("results/r1/fix.json"))
    lk, rm = G.fix_from_json(fix)
    enc = G.EncB(model, T5, ln_k=lk, res_m=rm).eval()
    mel_g = G.MelB(mf, T5).eval()
    wins = (windows_of("data/sets/librispeech-test-clean/manifest-S4.jsonl", "s4")
            + windows_of("data/sets/librispeech-test-clean/manifest-S32.jsonl", "s32")
            + windows_of("data/sets/librispeech-test-other/manifest.jsonl", "lo", a.per_set)
            + windows_of("data/sets/voxpopuli/manifest.jsonl", "vp", a.per_set))
    stats, order = {}, None
    counts, nonfinite = {}, 0
    lens = []
    for tag, path, s0, n in wins:
        audio = R.load_wav(path)[s0:s0 + n]
        n = len(audio)
        x = torch.zeros(1, N5)
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
        order = order or list(h.order)
        counts[tag.split(":")[0]] = counts.get(tag.split(":")[0], 0) + 1
        lens.append(n)
    sq = {k: v["amax"] for k, v in stats.items() if k.endswith(".sqdiff")}
    vr = {k: v["amax"] for k, v in stats.items() if k.endswith(".var")}
    worst = max(sq.items(), key=lambda kv: kv[1])
    over = [{"op": k, "max_abs": v.get("amax_unmasked", v["amax"])} for k, v in stats.items()
            if v.get("amax_unmasked", v["amax"]) > D.RES_LIMIT and not k.endswith((".sqdiff", ".var", ".var_eps", ".mean", ".rsqrt"))]
    pf = fix["post_fix"]
    per_site_r1 = pf["per_site_max_sqdiff"]
    ratio = sorted(((sq[k] / max(per_site_r1[k], 1e-30), k) for k in sq if k in per_site_r1), reverse=True)
    rep = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "t_mel": T5, "n_audio": N5, "windows": len(wins), "per_set": counts,
           "samples_min": min(lens), "samples_max": max(lens), "seconds": round(time.time() - t0, 1), "sites": len(sq),
           "max_sqdiff": worst[1], "max_sqdiff_site": worst[0], "max_row_var": max(vr.values()), "limit": LIMIT,
           "all_under_limit": bool(worst[1] < LIMIT and max(vr.values()) < LIMIT), "other_over_2p14": over, "nonfinite_enc_out": nonfinite,
           "r1_30s_post_fix": {"windows": pf["windows"], "max_sqdiff": pf["max_sqdiff"], "max_sqdiff_site": pf["max_sqdiff_site"], "max_row_var": pf["max_row_var"]},
           "top_sites_5s_vs_r1": [{"site": k, "ratio": r, "sqdiff_5s": sq[k], "sqdiff_r1": per_site_r1[k]} for r, k in ratio[:8]],
           "headroom_bits": round(math.log2(LIMIT / worst[1]) / 2, 2),
           "per_site_max_sqdiff": sq}
    json.dump(rep, open("results/r5/range5.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("per_site_max_sqdiff", "top_sites_5s_vs_r1")}), flush=True)
    return 0 if rep["all_under_limit"] else 3


if __name__ == "__main__":
    sys.exit(main())
