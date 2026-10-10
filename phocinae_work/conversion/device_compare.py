"""Step 7b: score a phone gate report (device/<tag>.json) against the publisher's reference, with the lane's bar.

    python device_compare.py device/phocinae_s26_gpu_fp32_L512_v2.json [--out device/device_parity_<tag>.json]

Same bar as litert_parity.py: all rows finite, argmax 100% on the non-near-tie test rows, max |dp| <= 0.02 and
mean |dp| <= 0.002 over every option of every row (post-temperature), the control rows move > 0.02 against the
original rows' reference, act_probability |d| reported. Flip rate of the graph on the reversed rows when present.
"""
import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from common import BAR, NEAR_TIE, W, act_probability, dump_json, probs_from_logits, temperatures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    rp = Path(a.report) if Path(a.report).is_absolute() else W / a.report
    d = json.loads(rp.read_text())
    temps = temperatures()
    oracle = json.loads((W / "oracle" / "oracle.json").read_text())
    ref = {o["id"]: o for o in oracle["rows"]}
    fx = {r["id"]: r for r in json.loads((W / "fixtures" / "rows_laya.json").read_text())["rows"]}
    recs = []
    for r in d.get("rows", []):
        o = ref[r["id"]]
        rec = dict(id=r["id"], group=fx[r["id"]]["group"], finite=bool(r["finite"]), nonfinite_real=r.get("nonfinite_real", 0), total_ms=r["total_ms"], run_ms=r["run_ms"])
        if r["finite"]:
            raw = np.asarray([float(x) for x in r["logits"]], dtype=np.float64)
            p = probs_from_logits(raw, o["qtype"], temps)
            dp = np.abs(np.asarray(p) - np.asarray(o["probs"]))
            al = np.asarray([float(x) for x in r["act_logits"]], dtype=np.float64)
            rec.update(max_abs_dlogit=float(np.abs(raw - np.asarray(o["logits"])).max()), max_abs_dp=float(dp.max()), mean_abs_dp=float(dp.mean()),
                       argmax=int(np.argmax(p)), argmax_equal=int(np.argmax(p)) == o["argmax"], near_tie=o["near_tie"],
                       act_dp=abs(act_probability(al) - o["act_probability"]))
            if rec["group"] == "control":
                base = ref[r["id"].replace("ctrl/", "", 1)]
                rec["control_move_vs_original"] = float(np.abs(np.asarray(p) - np.asarray(base["probs"])).max())
        recs.append(rec)
    fin = [x for x in recs if x["finite"]]
    nf = [x for x in recs if not x["finite"]]
    test = [x for x in fin if x["group"] == "test"]
    non_near = [x for x in test if not x["near_tie"]]
    near = [x for x in test if x["near_tie"]]
    ctrl = [x for x in fin if x["group"] == "control"]
    by_id = {x["id"]: x for x in recs}

    def agg(xs):
        if not xs:
            return {}
        return dict(rows=len(xs), argmax_equal=sum(x["argmax_equal"] for x in xs), max_abs_dp=max(x["max_abs_dp"] for x in xs),
                    mean_abs_dp=float(np.mean([x["mean_abs_dp"] for x in xs])), max_abs_dlogit=max(x["max_abs_dlogit"] for x in xs),
                    max_act_dp=max(x["act_dp"] for x in xs))

    flip = None
    rev = [x for x in fin if x["group"] == "flip" and x["id"].endswith("#rev")]
    if rev:
        n = f = 0
        for x in rev:
            b = by_id.get(x["id"].split("#")[0])
            if b is None or not b["finite"]:
                continue
            n += 1
            f += fx[x["id"]]["option_keys"][x["argmax"]] != fx[b["id"]]["option_keys"][b["argmax"]]
        flip = dict(reversed=dict(n=n, flips=f, rate=round(f / max(1, n), 4)), reference_reversed=oracle["flip"]["rev"])
    overall = agg(fin)
    summary = dict(report=str(rp.relative_to(W)), graph=d.get("graph"), accel=d.get("accel"), precision=d.get("precision"), table=d.get("table"),
                   seq=d.get("seq"), device=d.get("device"), android=d.get("android"), build=d.get("build"), litert=d.get("litert"), status=d.get("status"),
                   compile_ms=d.get("compile_ms"), uptime=dict(start=d.get("uptime_s_start"), end=d.get("uptime_s_end")),
                   rows=dict(run=len(recs), nonfinite=len(nf), nonfinite_ids=[x["id"] for x in nf][:20]),
                   overall=overall, test=agg(test), test_non_near_tie=agg(non_near), test_near_tie=agg(near), control=agg(ctrl),
                   control_moves=[dict(id=x["id"], move=x["control_move_vs_original"]) for x in ctrl], flip=flip,
                   app_summary=d.get("summary"), timing=d.get("timing"),
                   ms=dict(total_median=round(statistics.median([x["total_ms"] for x in recs]), 2), run_median=round(statistics.median([x["run_ms"] for x in recs]), 2)) if recs else None,
                   bar=dict(**BAR, near_tie_definition=f"reference top-2 gap <= {NEAR_TIE}"),
                   bar_pass=bool(recs and not nf and non_near and all(x["argmax_equal"] for x in non_near) and overall["max_abs_dp"] <= BAR["max_abs_dp"]
                                 and overall["mean_abs_dp"] <= BAR["mean_abs_dp"] and (not ctrl or max(x["control_move_vs_original"] for x in ctrl) > BAR["control_row_moves"])))
    out = Path(a.out) if a.out else rp.with_name("device_parity_" + rp.name)
    dump_json(out, dict(summary=summary, rows=recs))
    print(json.dumps({k: summary[k] for k in ("report", "accel", "precision", "status", "compile_ms", "rows", "overall", "test_non_near_tie", "test_near_tie",
                                              "control_moves", "flip", "app_summary", "timing", "ms", "bar_pass")}, indent=1, default=str))


if __name__ == "__main__":
    main()
