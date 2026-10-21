#!/usr/bin/env python3
"""P0 of LANE §1b for the r4 forms on the S32 30 s windows (venv .venv-q, ai-edge-litert 2.3.0), any backend.

For each of the 83 windows saved by r1 (out/s32_ii/<id>[__w<k>].npz: the extractor's mode-ii mel [128,3000], valid
mel frames, B's torch encoder_projector output, B's generate() tokens of the window = B_ii), through r4_drive's
pipeline (the window goes to its bucket: 10 s n <= 160,000 samples, 20 s n <= 320,000, else 30 s; `--buckets 30`
= every window in the 30 s files):
  mel<b>    on the window audio vs the extractor mel: max|d| on valid frames, mel_mask / frame_mask sums, pad frames 0
  encoder   enc<b>_<form>(mel<b> output) vs B's torch output (chain) and enc<b>_<form>(npz mel, first t_mel frames) vs B
            (isolated); with --cmp FORM also enc<b>_<FORM> on the same isolated input (max|d|, bit-equal windows)
  tokens    mel + encoder + dec loop vs B_ii; the steps of every window -> results/r4/p0_steps_<tag>.jsonl (r4_gap.py)
CLI: r4_p0.py --form fp32 [--backend cpu] [--buckets 10,20,30] [--threads 4] [--cmp fp32] [--tag T]
     -> results/r4/p0_<tag>.json (tag default <form>_<backend>_<bkt|w30>)
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r4_drive as DR  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--form", required=True)
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--buckets", default="10,20,30")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--cmp", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--limit", type=int, default=0, help="first N windows only (dry runs)")
    a = ap.parse_args()
    buckets = tuple(sorted(int(x) for x in a.buckets.split(",")))
    tag = a.tag or f"{a.form}_{a.backend}_{DR.bk_tag(buckets)}"
    os.makedirs("results/r4", exist_ok=True)
    t_load = time.time()
    p = DR.Pipeline(a.form, a.backend, a.threads, buckets)
    load_s = round(time.time() - t_load, 1)
    cmp = {}
    if a.cmp:
        for b in buckets:
            cmp[b] = DR.CpuGraph(DR.enc_path(b, a.cmp), a.threads)
    man = DR.read_jsonl(DR.SETS["s32"][0])
    paths = {r["id"]: r["path"] for r in man}
    order = {r["id"]: k for k, r in enumerate(man)}
    files = sorted(glob.glob("out/s32_ii/*.npz"),
                   key=lambda f: (order.get(os.path.basename(f)[:-4].split("__w")[0], 1e9), f))
    if a.limit:
        files = files[:a.limit]
    rows = []
    t0 = time.time()
    steps_out = open(f"results/r4/p0_steps_{tag}.jsonl", "w")
    for f in files:
        stem = os.path.basename(f)[:-4]
        uid, k = (stem.split("__w") + ["0"])[:2]
        z = np.load(f)
        audio = DR.load_wav(paths[uid])
        s0, n = DR.windows(len(audio))[int(k)]
        win = audio[s0:s0 + n]
        l0, ve = int(z["valid_mel"]), int(z["valid_enc"])
        r = p.window(win, keep_steps=True, keep_enc=True)
        b = r["bucket"]
        tm = DR.T_MEL[b]
        mo, enc_c = r.pop("_mel"), r.pop("_enc")
        ref_mel, ref_enc = z["mel"], z["enc_proj"]
        iso = np.ascontiguousarray(ref_mel[:, :tm][None].astype(np.float32))
        mm = (np.arange(tm) < l0).astype(np.float32)[None]
        enc_i = p.encode(iso, mm, b)
        row = {"stem": stem, "samples": n, "bucket": b, "valid_mel": l0, "valid_enc": ve,
               "mel_mask_sum": int(mo["mel_mask"].sum()), "frame_mask_sum": int(mo["frame_mask"].sum()),
               "mel_max_abs_valid": float(np.abs(mo["features"][0][:, :l0] - ref_mel[:, :l0]).max()),
               "mel_pad_zero": bool(np.all(mo["features"][0][:, l0:] == 0)),
               "enc_chain_max_abs_valid": float(np.abs(enc_c[:ve] - ref_enc[:ve]).max()),
               "enc_isolated_max_abs_valid": float(np.abs(enc_i[:ve] - ref_enc[:ve]).max()),
               "enc_finite": bool(np.isfinite(enc_c[:ve]).all() and np.isfinite(enc_i[:ve]).all()),
               "argmax_rows_equal": int((enc_i[:ve].argmax(-1) == ref_enc[:ve].argmax(-1)).sum()),
               "tokens_eq": r["tokens"] == z["tokens"].tolist(), "n_tokens": len(r["tokens"]),
               "n_tokens_ref": int(len(z["tokens"])), "n_steps": r["n_steps"], "cap_hits": r["cap_hits"],
               "max_run_same_frame": r["max_run_same_frame"], "mel_ms": r["mel_ms"], "enc_ms": r["enc_ms"],
               "dec_ms": r["dec_ms"]}
        if b in cmp:
            ec = cmp[b]("serving_default", features=iso, mel_mask=mm)["enc"][0]
            row["cmp_max_abs_valid"] = float(np.abs(ec[:ve] - enc_i[:ve]).max())
            row["cmp_bit_equal_valid"] = bool(np.array_equal(ec[:ve], enc_i[:ve]))
        rows.append(row)
        steps_out.write(json.dumps({"stem": stem, "bucket": b, "steps": r["steps"], "tokens": r["tokens"]}) + "\n")
    steps_out.close()

    def worst(key, rr=rows):
        v = [x[key] for x in rr if key in x]
        return max(v) if v else None
    per_b = {}
    for b in buckets:
        rb = [x for x in rows if x["bucket"] == b]
        per_b[str(b)] = {"windows": len(rb), "tokens_equal": sum(x["tokens_eq"] for x in rb),
                         "enc_isolated_worst": worst("enc_isolated_max_abs_valid", rb),
                         "enc_chain_worst": worst("enc_chain_max_abs_valid", rb),
                         "mel_worst": worst("mel_max_abs_valid", rb), "cmp_worst": worst("cmp_max_abs_valid", rb),
                         "cmp_bit_equal": sum(1 for x in rb if x.get("cmp_bit_equal_valid")),
                         "enc_ms_median": float(np.median([x["enc_ms"] for x in rb])) if rb else None}
    summ = {"tag": tag, "form": a.form, "backend": a.backend, "buckets": list(buckets), "threads": a.threads,
            "cmp": a.cmp or None, "windows": len(rows), "tokens_equal": sum(x["tokens_eq"] for x in rows),
            "differ": [x["stem"] for x in rows if not x["tokens_eq"]], "per_bucket": per_b,
            "mel_worst_max_abs_valid": worst("mel_max_abs_valid"),
            "mel_mask_ok": all(x["mel_mask_sum"] == x["valid_mel"] and x["frame_mask_sum"] == x["valid_enc"]
                               for x in rows),
            "mel_pad_zero": all(x["mel_pad_zero"] for x in rows),
            "enc_chain_worst": worst("enc_chain_max_abs_valid"), "enc_isolated_worst": worst("enc_isolated_max_abs_valid"),
            "cmp_worst": worst("cmp_max_abs_valid"), "finite": all(x["enc_finite"] for x in rows),
            "empty_windows": sum(1 for x in rows if x["n_tokens"] == 0),
            "cap_hits": sum(x["cap_hits"] for x in rows), "max_run_same_frame": worst("max_run_same_frame"),
            "load_s": load_s, "wall_s": round(time.time() - t0, 1), "pipeline": p.info(), "env": DR.env_record()}
    json.dump({"summary": summ, "windows": rows}, open(f"results/r4/p0_{tag}.json", "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in summ.items() if k not in ("pipeline", "env")}, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
