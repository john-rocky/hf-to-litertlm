#!/usr/bin/env python3
"""The device line on one replayed pass. AUROC = the author's evaluate.py function, loaded by path, as judge.py.

    python3 -I conversion/judge_device.py <work> <replay tag> --mac <gate tag> [--ref A__base__fp32__dw] [--set set_device] [--out FILE]

(a) agreement with the Mac ship form at cutoffs 0.5 and 0.9: >= 99 % = 0 flips of 90, except that a flip of a record whose Mac value lies within cutoff ± 0.02 is recorded and allowed;
(b) AUROC(device) against AUROC(reference) on the same records: overall <= 0.2 points, each category <= 0.5 points (a one-sided drop; the absolute gap and a form above the reference
by the width are printed beside it).
Instruments, never a verdict: |Δprob_1| max / p95 / mean against the Mac form and against the reference, hidden[pooled] cos and max|Δ| against the Mac form's same rows, tokens and grids against the reference.
A number that is not a number fails by itself: a record whose device prob_1, logits or hidden[pooled] is not finite, or whose hidden[pooled] file is missing, or whose Mac or reference prob_1 is
not finite, is LINE FAIL with the record ids listed. NaN compares false with every cutoff, so a record the Mac scores negative flips at neither; the author's AUROC sorts on it and returns an order-dependent number (here None); max() over a list with
NaN can read 0.0 (the instruments list such records apart and take their statistics on the rest).
PASS means every record of the set was answered. On a subset (a phone's short buckets, or a run that did not end) the same arithmetic runs on the records the phone answered and the word is PASS_ON_SUBSET, with how many and which are missing. Numbers only."""
import argparse
import importlib.util
import json
import math
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("work", type=Path, help="the work directory")
ap.add_argument("tag")
ap.add_argument("--mac", required=True, help="results/gate/<tag>.jsonl (or results/replay/<tag>.jsonl with --mac-dir replay)")
ap.add_argument("--mac-dir", choices=("gate", "replay"), default="gate")
ap.add_argument("--ref", default="A__base__fp32__dw")
ap.add_argument("--set", default="set_device")
ap.add_argument("--band", type=float, default=0.02)
ap.add_argument("--out", default="")
a = ap.parse_args()
WORK = a.work.resolve()
spec = importlib.util.spec_from_file_location("author_eval", WORK / "author/scripts/minicpm-ocr-jev/eval/evaluate.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
auroc = ev.auroc
CATS = ("text", "formula", "table")


def read_jsonl(p):
    return {r["item_id"]: r for r in map(json.loads, p.read_text().splitlines())}


def au(rows, key="prob_1"):
    if not rows or len({r["label"] for r in rows}) < 2 or not all(math.isfinite(r[key]) for r in rows):
        return None
    x = auroc([(r[key], r["label"]) for r in rows])
    return None if x is None else 100 * x


def pct(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(math.ceil(p / 100 * len(v))) - 1)] if v else None


def diff_stats(A, B, ids, key="prob_1"):
    pairs = [(abs(A[i][key] - B[i][key]), i) for i in ids]
    nonfinite = [i for x, i in pairs if not math.isfinite(x)]
    pairs = [(x, i) for x, i in pairs if math.isfinite(x)]
    d = [x for x, _ in pairs]
    if not d:
        return {"nonfinite_records": nonfinite} if nonfinite else None
    return {"nonfinite_records": nonfinite, "max": max(d), "p95": pct(d, 95), "mean": sum(d) / len(d), "bit_equal": sum(x == 0.0 for x in d), "worst": sorted(((round(x, 7), i) for x, i in pairs), reverse=True)[:6]}


def hidden_vs(x_dir, y_dir, ids):
    import numpy as np

    cos, mx = [], []
    missing, nonfinite, zero_norm = 0, [], []
    for i in ids:
        pa, pb = x_dir / f"{i}.npy", y_dir / f"{i}.npy"
        if not pa.exists() or not pb.exists():
            missing += 1
            continue
        x, y = np.load(pa).astype(np.float64), np.load(pb).astype(np.float64)
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            nonfinite.append(i)
            continue
        nx, ny = np.linalg.norm(x), np.linalg.norm(y)
        if nx == 0 or ny == 0:
            zero_norm.append(i)
            continue
        cos.append(float(x @ y / (nx * ny)))
        mx.append(float(np.abs(x - y).max()))
    if not cos:
        return {"records": 0, "missing": missing, "nonfinite_records": nonfinite, "zero_norm_records": zero_norm}
    return {"records": len(cos), "missing": missing, "nonfinite_records": nonfinite, "zero_norm_records": zero_norm, "bit_equal": sum(m == 0.0 for m in mx), "cos_min": min(cos), "cos_p5": pct(cos, 5), "max_abs_max": max(mx), "max_abs_p95": pct(mx, 95)}


def device_hidden(d, ids):
    """-> (ids whose saved hidden[pooled] holds a non-finite value, ids with no file): the row the host's head reads, part of the verdict"""
    import numpy as np

    bad, missing = [], []
    for i in ids:
        p = d / f"{i}.npy"
        if not p.exists():
            missing.append(i)
        elif not np.isfinite(np.load(p)).all():
            bad.append(i)
    return bad, missing


dev = read_jsonl(WORK / f"results/replay/{a.tag}.jsonl")
mac_path = WORK / f"results/{a.mac_dir}/{a.mac}.jsonl"
mac = read_jsonl(mac_path)
ref = read_jsonl(WORK / f"results/ref/{a.ref}.jsonl")
want = json.loads((WORK / f"conversion/sets/{a.set}.json").read_text())["item_ids"]
ids = sorted(i for i in want if i in dev and i in mac and i in ref)
out = {"tag": a.tag, "mac": a.mac, "ref": a.ref, "set": a.set, "records_in_set": len(want), "records_compared": len(ids),
       "not_answered_by_device": sorted(i for i in want if i not in dev), "mac_missing": sorted(i for i in want if i not in mac and i in dev)}
out["ids_differ_from_ref"] = [i for i in ids if dev[i]["ids_sha"] != ref[i]["ids_sha"]]
out["grids_differ_from_ref"] = [i for i in ids if dev[i]["grids"] != ref[i].get("grids", dev[i]["grids"])]
out["pooled_ok"] = all(dev[i]["pooled_id"] == 271 for i in ids)
# the numbers themselves: a record with no number cannot agree with anything
out["nonfinite_prob_records"] = [i for i in ids if not (math.isfinite(dev[i]["prob_1"]) and all(math.isfinite(x) for x in dev[i]["logits"]))]
out["nonfinite_hidden_records"], out["hidden_missing_records"] = device_hidden(WORK / f"out/replay/{a.tag}", ids)
out["nonfinite_mac_or_ref_records"] = [i for i in ids if not (math.isfinite(mac[i]["prob_1"]) and math.isfinite(ref[i]["prob_1"]))]
out["line_finite"] = not (out["nonfinite_prob_records"] or out["nonfinite_hidden_records"] or out["hidden_missing_records"] or out["nonfinite_mac_or_ref_records"])
# (a) the verdict's agreement with the Mac form
a_ok = True
for cut in (0.5, 0.9):
    fl = [(i, dev[i]["category"], round(mac[i]["prob_1"], 6), round(dev[i]["prob_1"], 6)) for i in ids if (mac[i]["prob_1"] >= cut) != (dev[i]["prob_1"] >= cut)]
    inside = [f for f in fl if abs(f[2] - cut) <= a.band]
    outside = [f for f in fl if abs(f[2] - cut) > a.band]
    out[f"flips_{cut}"] = {"n": len(fl), "same_side_pct": 100 * (1 - len(fl) / len(ids)) if ids else None, "inside_band_recorded": inside, "outside_band": outside,
                           "by_category": {c: sum(f[1] == c for f in fl) for c in CATS}}
    a_ok = a_ok and not outside
out["line_a_agreement"] = a_ok and not out["nonfinite_prob_records"]
# (b) AUROC against the reference, same records
A, B = [ref[i] for i in ids], [dev[i] for i in ids]
au_r, au_d = au(A), au(B)
drop = None if au_r is None or au_d is None else au_r - au_d
out["auroc_ref"], out["auroc_device"], out["drop"] = au_r, au_d, drop
per_ok, per_abs_ok = True, True
out["per_category"] = {}
for c in CATS:
    x, y = [r for r in A if r["category"] == c], [r for r in B if r["category"] == c]
    if not x:
        continue
    ar, ad = au(x), au(y)
    d = None if ar is None or ad is None else ar - ad
    out["per_category"][c] = {"n": len(x), "pos": sum(r["label"] for r in x), "ref": ar, "device": ad, "drop": d}
    per_ok = per_ok and (d is None or d <= 0.5)
    per_abs_ok = per_abs_ok and (d is None or abs(d) <= 0.5)
out["line_b_auroc"] = bool(drop is not None and drop <= 0.2 and per_ok)
out["line_b_abs_gap_within_width"] = bool(drop is not None and abs(drop) <= 0.2 and per_abs_ok)
out["device_above_ref_by_the_width"] = bool((drop or 0) < -0.2 or any((v["drop"] or 0) < -0.5 for v in out["per_category"].values()))
out["complete_set"] = len(ids) == len(want)
out["LINE"] = "FAIL" if not (out["line_finite"] and out["line_a_agreement"] and out["line_b_auroc"]) else ("PASS" if out["complete_set"] else "PASS_ON_SUBSET")
# instruments
out["abs_dprob_vs_mac"] = diff_stats(mac, dev, ids)
out["abs_dprob_vs_ref"] = diff_stats(ref, dev, ids)
out["abs_dprob_mac_vs_ref"] = diff_stats(ref, mac, ids)
mac_h = WORK / ("out/gate" if a.mac_dir == "gate" else "out/replay") / a.mac
out["hidden_pooled_vs_mac"] = hidden_vs(mac_h, WORK / f"out/replay/{a.tag}", ids)
out["hidden_pooled_vs_ref"] = hidden_vs(WORK / f"out/ref/{a.ref}", WORK / f"out/replay/{a.tag}", ids)
text = json.dumps(out, indent=1, ensure_ascii=False)
if a.out:
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(text + "\n")
print(json.dumps({k: out[k] for k in ("tag", "mac", "records_compared", "LINE", "nonfinite_prob_records", "nonfinite_hidden_records", "hidden_missing_records", "not_answered_by_device", "line_a_agreement", "line_b_auroc", "auroc_ref", "auroc_device", "drop", "abs_dprob_vs_mac", "hidden_pooled_vs_mac")}))
