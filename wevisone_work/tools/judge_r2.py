#!/usr/bin/env python3
"""Round-2 verdicts of every runtime suite against the B arm, under two bars side by side:

  original  boxes IoU >= 0.75 matched >= 90 %, counts exact >= 90 %, points within 2 % >= 90 %, OCR CER <= 0.05
  bf16      a rule fixed before any number: the HF B arm in bfloat16 (r2/ref/Bbf16) scored against B with the same
            score_arms; per metric, if the bf16 mismatch d exceeds the original tolerance (10 % for boxes / counts /
            points), the tolerance becomes 2 d; CER: if the bf16 CER exceeds 0.05, the bar becomes 2 x the bf16 CER
Text gate (both bars the same; the jitter rule covers the image metrics): the 10 rows' pass / fail equal to T0's, the
same answer text count, France = "Paris"; the bf16 text arm (r2/ref/T0bf16) is listed as the oracle's own text jitter.
Counts = the author's count (score_arms count_rate, the pre-registered one).

    K/.venv-ref/bin/python -I tools/judge_r2.py [--labels a,b,...] -> r2/judge_r2.json + printed table
"""
import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

K = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("score_r2", Path(__file__).resolve().parent / "score_r2.py")
SR = importlib.util.module_from_spec(spec)
spec.loader.exec_module(SR)
SA = SR.SA
LAUNCH_TOL = {"box": 0.10, "count": 0.10, "point": 0.10}
LAUNCH_CER = 0.05


def rates(s):
    return {"box": s["box_rate"], "count": s["count_rate"], "point": s["point_rate"], "cer": s["ocr_cer"]}


def bars_from_jitter(j):
    bars, why = {}, {}
    for k, tol in LAUNCH_TOL.items():
        d = 1.0 - j[k] if j[k] is not None else 0.0
        if d > tol:
            bars[k], why[k] = 1.0 - 2 * d, f"bf16 mismatch {d:.4f} > {tol} -> tolerance 2d = {2 * d:.4f}"
        else:
            bars[k], why[k] = 1.0 - tol, f"bf16 mismatch {d:.4f} <= {tol} -> launch bar"
    c = j["cer"] or 0.0
    if c > LAUNCH_CER:
        bars["cer"], why["cer"] = 2 * c, f"bf16 CER {c:.4f} > {LAUNCH_CER} -> 2 x = {2 * c:.4f}"
    else:
        bars["cer"], why["cer"] = LAUNCH_CER, f"bf16 CER {c:.4f} <= {LAUNCH_CER} -> launch bar"
    return bars, why


def verdict(r, bars):
    ok = {k: (r[k] is None or r[k] >= bars[k]) for k in ("box", "count", "point")}
    ok["cer"] = r["cer"] is None or r["cer"] <= bars["cer"]
    return ok, all(ok.values())


def text_gate(label):
    d = K / "r2" / "suite" / label / "text"
    if not d.exists():
        return None
    rows = {}
    for p in sorted(d.glob("*.json")):
        x = json.loads(p.read_text())
        if x["row"] == "long_summary":
            continue
        rows[x["row"]] = {"pass": x["pass"], "ref_pass": x["ref_pass"], "same_text": all(x["same_text_as_T0"]),
                          "answer": x["answers"][-1]}
    if not rows:
        return None
    france = rows.get("q_france", {}).get("answer", "")
    res = {"rows": len(rows), "pass_fail_same": sum(v["pass"] == v["ref_pass"] for v in rows.values()),
           "same_text": sum(v["same_text"] for v in rows.values()), "pass": sum(v["pass"] for v in rows.values()),
           "france_paris": bool(re.fullmatch(r"\s*Paris\.?\s*", france)),
           "differs": {k: v["answer"][:90] for k, v in rows.items() if not v["same_text"]}}
    res["gate"] = res["pass_fail_same"] == res["rows"] and res["france_paris"]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="")
    ap.add_argument("--ref", default="B", help="oracle arm (B = 512, F = 768)")
    ap.add_argument("--jitter", default="Bbf16", help="the oracle's bf16 arm (Bbf16 / Fbf16)")
    ap.add_argument("--out", default="r2/judge_r2.json")
    a = ap.parse_args()
    rows = json.loads((K / "fixtures" / "fixtures.json").read_text())["rows"]
    jit, _ = SA.pair_table(a.jitter, a.ref, rows)
    j = rates(jit)
    bars_bf16, why = bars_from_jitter(j)
    bars_orig = {"box": 0.90, "count": 0.90, "point": 0.90, "cer": LAUNCH_CER}
    labels = a.labels.split(",") if a.labels else sorted(
        p.name for p in (K / "r2" / "suite").iterdir()
        if p.is_dir() and (p.name.startswith("768_") == (a.ref == "F")))    # 768 suites are judged against F only
    out = {"ref": a.ref, "jitter": {"pair": f"{a.jitter} vs {a.ref}", "summary": {k: jit[k] for k in (
        "box_matched", "box_ref", "count_equal", "count_rows", "point_matched", "point_ref", "ocr_cer",
        "ocr_max_row_cer", "exact_rows", "rows", "missing")}, "rates": j}, "bars": {"original": bars_orig,
           "bf16": bars_bf16, "why": why}, "forms": {}}
    t0b = K / "r2" / "ref" / "T0bf16"
    if t0b.exists():
        tj = [json.loads(p.read_text()) for p in sorted(t0b.glob("*.json"))]
        out["text_jitter"] = {"rows": len(tj), "pass": sum(x["pass"] for x in tj),
                              "same_text_as_T0": sum(all(x["same_text_as_T0"]) for x in tj),
                              "differs": [x["row"] for x in tj if not all(x["same_text_as_T0"])]}
    print(f"jitter {a.jitter} vs {a.ref}: box {jit['box_matched']}/{jit['box_ref']} count {jit['count_equal']}/{jit['count_rows']} "
          f"point {jit['point_matched']}/{jit['point_ref']} cer {jit['ocr_cer']} exact {jit['exact_rows']}/{jit['rows']}")
    print("bars bf16:", {k: round(v, 4) for k, v in bars_bf16.items()}, why)
    for lab in labels:
        if not any((K / "r2" / "suite" / lab).glob("*.json")):
            continue
        s, _ = SA.pair_table(lab, a.ref, rows)
        if s["rows"] < 29:
            continue
        r = rates(s)
        vo, po = verdict(r, bars_orig)
        vb, pb = verdict(r, bars_bf16)
        tg = text_gate(lab)
        out["forms"][lab] = {"rates": r, "counts": {k: s[k] for k in ("box_matched", "box_ref", "count_equal",
                                                                      "count_rows", "point_matched", "point_ref",
                                                                      "exact_rows")},
                             "original": {"per_metric": vo, "pass_image": po}, "bf16": {"per_metric": vb,
                                                                                         "pass_image": pb},
                             "text": tg}
        print(f"{lab:34s} box {s['box_matched']}/{s['box_ref']} count {s['count_equal']}/{s['count_rows']} "
              f"point {s['point_matched']}/{s['point_ref']} cer {r['cer']:.3f} exact {s['exact_rows']} | "
              f"orig {'PASS' if po else 'FAIL'} bf16 {'PASS' if pb else 'FAIL'} | text "
              f"{'-' if tg is None else ('PASS' if tg['gate'] else 'FAIL')} "
              f"{'' if tg is None else str(tg['pass_fail_same']) + '/' + str(tg['rows']) + ' same ' + str(tg['same_text'])}")
    (K / a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
