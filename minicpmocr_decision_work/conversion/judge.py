#!/usr/bin/env python3
"""The lines on a form's rows against the reference's rows (ref_run.py): P1 the three official examples within 0.02, P2 AUROC within 0.2 points overall and 0.5 per
category, P3 the same side of the cutoffs 0.5 and 0.9 on at least 99 %. AUROC = the author's evaluate.py function, loaded by path. Same record set, one-sided,
every flip listed. Numbers only.

    python3 -I conversion/judge.py <work> compare <form tag> [--ref A__base__fp32__dw] [--limit-set device]   P2, P3, instruments
    python3 -I conversion/judge.py <work> p1 <examples tag>                                                  P1
    python3 -I conversion/judge.py <work> red <suffix of the form, e.g. fp16fc__gpu_f32>                     the red arms, both sides
    python3 -I conversion/judge.py <work> host <tag of the torch-backbone rows> <ref tag>                    the separation test
    python3 -I conversion/judge.py <work> summary <form tag>                                                 AUROC overall, per category, family, language
Prints json; `--out FILE` writes it too."""
import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("work", type=Path, help="the work directory")
ap.add_argument("cmd")
ap.add_argument("tag")
ap.add_argument("tag2", nargs="?")
ap.add_argument("--ref", default="A__base__fp32__dw")
ap.add_argument("--limit-set", default="")
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


def load_ref(tag):
    return read_jsonl(WORK / f"results/ref/{tag}.jsonl")


def load_form(tag):
    return read_jsonl(WORK / f"results/gate/{tag}.jsonl")


def skipped(tag):
    p = WORK / f"results/gate/{tag}.skipped.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def au(rows, key="prob_1"):
    if not rows or len({r["label"] for r in rows}) < 2:
        return None
    x = auroc([(r[key], r["label"]) for r in rows])
    return None if x is None else 100 * x


def minus(x, y):
    return None if x is None or y is None else x - y


def pct(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(math.ceil(p / 100 * len(v))) - 1)] if v else None


def hidden_rows(ref_tag, form_tag, ids):
    import numpy as np

    cos, mx = [], []
    missing = bit = 0
    for i in ids:
        pa, pb = WORK / f"out/ref/{ref_tag}/{i}.npy", WORK / f"out/gate/{form_tag}/{i}.npy"
        if not pa.exists() or not pb.exists():
            missing += 1
            continue
        x, y = np.load(pa).astype(np.float64), np.load(pb).astype(np.float64)
        cos.append(float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y))))
        mx.append(float(np.abs(x - y).max()))
        bit += int(mx[-1] == 0.0)
    if not cos:
        return {"records": 0, "missing": missing}
    return {"records": len(cos), "missing": missing, "bit_equal": bit, "cos_min": min(cos), "cos_p5": pct(cos, 5), "max_abs_max": max(mx), "max_abs_p95": pct(mx, 95)}


def compare(form_tag, ref_tag, limit=None, key="prob_1"):
    a_, b_ = load_ref(ref_tag), load_form(form_tag)
    if limit:
        keep = set(json.loads((WORK / f"conversion/sets/{limit}.json").read_text())["item_ids"])
        a_ = {k: v for k, v in a_.items() if k in keep}
        b_ = {k: v for k, v in b_.items() if k in keep}
    ids = sorted(set(a_) & set(b_))
    A, B = [a_[i] for i in ids], [b_[i] for i in ids]
    d = [abs(x[key] - y[key]) for x, y in zip(A, B)]
    dl = [max(abs(p - q) for p, q in zip(x["logits"], y["logits"])) for x, y in zip(A, B)]
    bad_ids = [i for i, x, y in zip(ids, A, B) if x["ids_sha"] != y["ids_sha"]]
    out = {"ref": ref_tag, "form": form_tag, "limit_set": limit, "common": len(ids), "only_ref": sorted(set(a_) - set(b_)), "only_form": sorted(set(b_) - set(a_)),
           "skipped": [{k: s.get(k) for k in ("item_id", "category", "status", "n", "T")} for s in skipped(form_tag)],
           "ids_differ": bad_ids, "pooled_ok": all(y["pooled_id"] == 271 for y in B),
           "auroc_ref_common": au(A), "auroc_ref_all_rows": au(list(a_.values())), "auroc_form": au(B), "drop": minus(au(A), au(B)), "per_category": {},
           "abs_dprob": {"max": max(d), "p95": pct(d, 95), "mean": sum(d) / len(d), "worst": sorted(((round(x, 5), i) for x, i in zip(d, ids)), reverse=True)[:8]},
           "abs_dlogit": {"max": max(dl), "p95": pct(dl, 95)}, "hidden_pooled": hidden_rows(ref_tag, form_tag, ids)}
    per_ok = True
    for c in CATS:
        x = [r for r in A if r["category"] == c]
        y = [r for r in B if r["category"] == c]
        if not x:
            continue
        drop = minus(au(x), au(y))
        per_ok = per_ok and drop is not None and drop <= 0.5
        out["per_category"][c] = {"n": len(x), "ref": au(x), "form": au(y), "drop": drop, "ref_all_rows": au([r for r in a_.values() if r["category"] == c])}
    for cut in (0.5, 0.9):
        fl = [(r["item_id"], r["category"], round(r[key], 5), round(s[key], 5)) for r, s in zip(A, B) if (r[key] >= cut) != (s[key] >= cut)]
        out[f"flips_{cut}"] = {"n": len(fl), "same_side_pct": 100 * (1 - len(fl) / len(ids)), "records": fl,
                               "by_category": {c: sum(f[1] == c for f in fl) for c in CATS}}
    out["P2"] = bool(out["drop"] is not None and out["drop"] <= 0.2 and per_ok)
    out["P3"] = all(out[f"flips_{c}"]["same_side_pct"] >= 99.0 for c in (0.5, 0.9))
    out["form_above_ref_by_the_width"] = bool((out["drop"] or 0) < -0.2 or any((v["drop"] or 0) < -0.5 for v in out["per_category"].values()))
    secs = [y["seconds"] for y in B if "seconds" in y]
    out["seconds"] = {"total": round(sum(secs)), "p50": pct(secs, 50), "max": max(secs)} if secs else None
    return out


def p1(tag):
    d = json.loads((WORK / f"results/gate/{tag}.json").read_text())
    ref = {r["case"]: r for r in json.loads((WORK / "results/ref/examples__fp32__dw.json").read_text())["rows"]}
    rows = []
    for r in d["rows"]:
        rows.append({"case": r["case"], "prob_1": r["prob_1"], "observed": r["observed"], "diff_observed": r["diff"], "p1": r["p1"],
                     "ref_prob_1": ref[r["case"]]["prob_1"], "diff_ref": r["prob_1"] - ref[r["case"]]["prob_1"], "ids_equal": r["ids_sha"] == ref[r["case"]]["ids_sha"], "pooled_id": r["pooled_id"]})
    return {"tag": tag, "P1": sum(r["p1"] for r in rows), "of": len(rows), "rows": rows}


def red(suffix):
    ref_base = load_ref("A__base__fp32__dw")
    form_base = load_form(f"A__base__{suffix}")
    ids = json.loads((WORK / "conversion/sets/red_arms.json").read_text())["item_ids"]
    out = {"records": len(ids), "arms": {}}

    def arm_row(base, arm_rows, key="prob_1"):
        c = [i for i in ids if i in arm_rows and i in base]
        d = [abs(arm_rows[i][key] - base[i]["prob_1"]) for i in c]
        rows = [arm_rows[i] for i in c]
        base_rows = [base[i] for i in c]
        doc = {"n": len(c), "moved_gt_0.02": sum(x > 0.02 for x in d), "auroc": au(rows, key), "base_auroc": au(base_rows), "mean_abs": sum(d) / len(d), "max_abs": max(d)}
        doc["moved"] = bool(doc["moved_gt_0.02"] >= 12 or abs(doc["auroc"] - doc["base_auroc"]) >= 5)
        return doc

    for arm in ("R1", "R2", "R4"):
        rp, fp = WORK / f"results/ref/red__{arm}__fp32__dw.jsonl", WORK / f"results/gate/red__{arm}__{suffix}.jsonl"
        doc = {}
        if rp.exists():
            ref_arm = read_jsonl(rp)
            doc["ref"] = arm_row(ref_base, ref_arm)
        if fp.exists():
            form_arm = read_jsonl(fp)
            doc["form"] = arm_row(form_base, form_arm)
        if rp.exists() and fp.exists():
            c = [i for i in ids if i in ref_arm and i in form_arm]
            dd = [abs(ref_arm[i]["prob_1"] - form_arm[i]["prob_1"]) for i in c]
            doc["form_minus_ref"] = {"n": len(c), "abs_dprob_max": max(dd), "abs_dprob_p95": pct(dd, 95), "abs_dprob_mean": sum(dd) / len(dd),
                                     "auroc_drop": au([ref_arm[i] for i in c]) - au([form_arm[i] for i in c]),
                                     "flips_0.5": [(i, round(ref_arm[i]["prob_1"], 5), round(form_arm[i]["prob_1"], 5)) for i in c if (ref_arm[i]["prob_1"] >= 0.5) != (form_arm[i]["prob_1"] >= 0.5)],
                                     "flips_0.9": [(i, round(ref_arm[i]["prob_1"], 5), round(form_arm[i]["prob_1"], 5)) for i in c if (ref_arm[i]["prob_1"] >= 0.9) != (form_arm[i]["prob_1"] >= 0.9)]}
        out["arms"][arm] = doc
    doc = {"ref": arm_row(ref_base, ref_base, "prob_1_prev"), "form": arm_row(form_base, form_base, "prob_1_prev")}
    c = [i for i in ids if i in ref_base and i in form_base]
    dd = [abs(ref_base[i]["prob_1_prev"] - form_base[i]["prob_1_prev"]) for i in c]
    doc["form_minus_ref"] = {"n": len(c), "abs_dprob_max": max(dd), "abs_dprob_p95": pct(dd, 95), "abs_dprob_mean": sum(dd) / len(dd),
                             "auroc_drop": au([ref_base[i] for i in c], "prob_1_prev") - au([form_base[i] for i in c], "prob_1_prev"),
                             "flips_0.5": [(i, round(ref_base[i]["prob_1_prev"], 5), round(form_base[i]["prob_1_prev"], 5)) for i in c if (ref_base[i]["prob_1_prev"] >= 0.5) != (form_base[i]["prob_1_prev"] >= 0.5)],
                             "flips_0.9": [(i, round(ref_base[i]["prob_1_prev"], 5), round(form_base[i]["prob_1_prev"], 5)) for i in c if (ref_base[i]["prob_1_prev"] >= 0.9) != (form_base[i]["prob_1_prev"] >= 0.9)]}
    out["arms"]["R3"] = doc
    for arm, doc in out["arms"].items():
        if "ref" in doc and "form" in doc:
            doc["ref_moved_and_form_moved"] = (not doc["ref"]["moved"]) or doc["form"]["moved"]
    return out


def summary(tag):
    """AUROC of one form's rows: overall, per category, family, language (the all-records check of the card's figure)."""
    R = list(load_form(tag).values())
    out = {"tag": tag, "n": len(R), "skipped": len(skipped(tag)), "auroc": au(R), "pos": sum(r["label"] for r in R), "neg": sum(1 - r["label"] for r in R)}
    for c in CATS:
        x = [r for r in R if r["category"] == c]
        out[c] = {"n": len(x), "auroc": au(x)}
    for f in sorted({r["family"] for r in R}):
        x = [r for r in R if r["family"] == f]
        out[f] = {"n": len(x), "auroc": au(x)}
    for la in sorted({r["lang"] for r in R}):
        x = [r for r in R if r["lang"] == la]
        out[la] = {"n": len(x), "auroc": au(x)}
    return out


def host(form_tag, ref_tag):
    a_, b_ = load_ref(ref_tag), load_form(form_tag)
    ids = sorted(set(a_) & set(b_))
    d = [abs(a_[i]["prob_1"] - b_[i]["prob_1"]) for i in ids]
    dl = [max(abs(p - q) for p, q in zip(a_[i]["logits"], b_[i]["logits"])) for i in ids]
    return {"ref": ref_tag, "rows": form_tag, "common": len(ids), "only_ref": len(set(a_) - set(b_)), "only_rows": len(set(b_) - set(a_)),
            "ids_equal": sum(a_[i]["ids_sha"] == b_[i]["ids_sha"] for i in ids), "grids_equal": sum(a_[i]["grids"] == b_[i]["grids"] for i in ids),
            "prob_bit_equal": sum(a_[i]["prob_1"] == b_[i]["prob_1"] for i in ids), "abs_dprob_max": max(d), "abs_dprob_mean": sum(d) / len(d),
            "abs_dlogit_max": max(dl), "abs_dprob_prev_max": max(abs(a_[i]["prob_1_prev"] - b_[i]["prob_1_prev"]) for i in ids),
            "PASS_le_1e-5": max(d) <= 1e-5, "worst": sorted(((round(x, 8), i) for x, i in zip(d, ids)), reverse=True)[:5],
            "hidden_pooled": hidden_rows(ref_tag, form_tag, ids)}


doc = {"compare": lambda: compare(a.tag, a.ref, a.limit_set or None), "p1": lambda: p1(a.tag), "red": lambda: red(a.tag),
       "host": lambda: host(a.tag, a.tag2), "summary": lambda: summary(a.tag)}[a.cmd]()
text = json.dumps(doc, indent=1, ensure_ascii=False)
if a.out:
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(text + "\n")
print(text)
