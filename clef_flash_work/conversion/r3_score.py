#!/usr/bin/env python3
"""Round 3: the hidden states of one image e2e set (scripts/r3_e2e.py) -> the author's JointSchemaHead (venv-ref,
scripts/clef_score.py's Scorer = the reference's head, threads 10) -> option probabilities -> the parity line.

    HF_HUB_OFFLINE=1 venv-ref/bin/python scripts/r3_score.py --tag L1024_v2      -> results/r3_score_<tag>.json

Reference row of a run = ref/records_ref.json (id, arm) (g256 / g448 / native). Control (every scored run, as
clef_score.py): the reference's own last_hidden through this scorer must reproduce the reference's logits within 1e-6,
else the scoring stops. Red arms (`<id>__<arm>__zero`, the image rows set to 0) are compared with the BASE run's
reference (the same id / arm with the picture): the arm moves when some option differs by > 0.02 (not in the line's
statistics; an arm that does not move is recorded only).
Line (pre-registered, unrounded probabilities), per group: g256 + g448 (the 28 fixed-grid runs through the main tower form),
native (the reference's image rows through the row graph) and fixed_grid_<t16|t32> (the same runs through another tower
form, files <id>__<arm>__t16 / __t32): every non-near-tie question keeps its argmax (near-tie = the reference's top-2
margin <= 0.02, listed apart), max |dp| over all options <= 0.02, mean over runs of each run's mean |dp| <= 0.002."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

K = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(K / "scripts"))
from clef_score import ARM_MIN, MAX_DP, MEAN_DP, NEAR_TIE, Scorer, compare  # noqa: E402


def line(runs: list) -> dict:
    if not runs:
        return {"runs": 0}
    qs = [q for r in runs for q in r["questions"]]
    nn = [q for q in qs if not q["near_tie"]]
    nt = [q for q in qs if q["near_tie"]]
    worst = max(runs, key=lambda r: r["max_abs_dp"])
    allo = [x for r in runs for x in r["option_abs_dp"]]
    s = {"runs": len(runs), "questions": len(qs), "near_tie_questions": len(nt),
         "argmax_non_near_tie": f"{sum(q['argmax_equal'] for q in nn)}/{len(nn)}",
         "argmax_near_tie": f"{sum(q['argmax_equal'] for q in nt)}/{len(nt)}",
         "near_tie_ids": [f"{r['id']}/{r['arm']}/{q['question_id']}" for r in runs for q in r["questions"] if q["near_tie"]],
         "max_abs_dp": worst["max_abs_dp"], "max_abs_dp_run": f"{worst['id']}/{worst['arm']}",
         "max_abs_dp_question": max(worst["questions"], key=lambda q: q["max_abs_dp"])["question_id"],
         "mean_of_run_means": float(np.mean([r["mean_abs_dp"] for r in runs])),
         "p95_option_abs_dp": float(np.percentile(allo, 95)),
         "flips_non_near_tie": [f"{r['id']}/{r['arm']}/{q['question_id']}" for r in runs for q in r["questions"]
                                if not q["argmax_equal"] and not q["near_tie"]]}
    s["line_pass"] = bool(all(q["argmax_equal"] for q in nn) and s["max_abs_dp"] <= MAX_DP and s["mean_of_run_means"] <= MEAN_DP)
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--threads", type=int, default=10)
    a = ap.parse_args()
    out = K / f"results/r3_score_{a.tag}.json"
    assert not out.exists(), f"refusing to overwrite {out}"
    cache = K / f"cache/r3/e2e/{a.tag}"
    assert (cache / "DONE").exists(), f"{cache}/DONE missing: the e2e set is not complete"
    ref = json.loads((K / "ref/records_ref.json").read_text())
    rows_ref = {(r["id"], r["arm"]): r for r in ref["rows"]}
    files = sorted(p for p in cache.glob("*.npy") if ".tmp" not in p.name)
    sc = Scorer(a.threads)
    runs, red, control = [], [], {"runs": 0, "logits_bit_equal": 0, "max_abs_dlogit": 0.0}
    for p in files:
        parts = p.stem.split("__")
        rid, arm, variant = parts[0], parts[1], (parts[2] if len(parts) > 2 else "base")
        row = rows_ref[(rid, arm)]
        zr = np.load(K / f"ref/npz/{rid}__{arm}.npz")
        lg_ctl = sc.logits(zr["last_hidden"], row)
        eq = all(lg.tolist() == q["logits"] for lg, q in zip(lg_ctl, row["questions"]))
        dl = max(float(np.abs(np.asarray(lg.tolist()) - np.asarray(q["logits"])).max())
                 for lg, q in zip(lg_ctl, row["questions"]))
        control["runs"] += 1
        control["logits_bit_equal"] += eq
        control["max_abs_dlogit"] = max(control["max_abs_dlogit"], dl)
        hid = np.load(p)
        pr = sc.probs(sc.logits(hid, row))
        c = compare(row, pr)
        c.update(id=rid, arm=arm, variant=variant, source=row["source"], tokens=row["tokens"], hidden_file=str(p.relative_to(K)),
                 control_logits_bit_equal=eq,
                 option_abs_dp=[abs(float(x) - float(y)) for q, pp in zip(row["questions"], pr) for x, y in zip(pp, q["probs"])])
        if variant == "zero":
            c["moves"] = c["max_abs_dp"] > ARM_MIN
            c["argmax_changed"] = [q["question_id"] for q in c["questions"] if not q["argmax_equal"]]
            red.append(c)
        else:
            runs.append(c)
    assert control["max_abs_dlogit"] <= 1e-6, f"scorer is not the reference head: {control}"
    base = [r for r in runs if r["variant"] == "base"]
    groups = {"fixed_grid": [r for r in base if r["arm"] in ("g256", "g448")],
              "g256": [r for r in base if r["arm"] == "g256"], "g448": [r for r in base if r["arm"] == "g448"],
              "native": [r for r in base if r["arm"] == "native"]}
    for v in sorted({r["variant"] for r in runs} - {"base"}):      # the other tower forms (t16 / t32)
        groups[f"fixed_grid_{v}"] = [r for r in runs if r["variant"] == v and r["arm"] in ("g256", "g448")]
    summary = {g: line(v) for g, v in groups.items()}
    red_summary = {"arms": len(red), "moved": {f"{r['id']}/{r['arm']}": round(r["max_abs_dp"], 4) for r in red},
                   "argmax_changed": {f"{r['id']}/{r['arm']}": r["argmax_changed"] for r in red},
                   "all_move": all(r["moves"] for r in red) if red else None}
    e2e_parts = sorted(K.glob(f"results/r3_e2e_{a.tag}_part*.json"))
    doc = {"what": f"round 3: image e2e {a.tag} -> author's JointSchemaHead (venv-ref) -> the parity line",
           "tag": a.tag, "e2e_parts": [str(p.relative_to(K)) for p in e2e_parts], "scorer": sc.info, "control": control,
           "line": {"near_tie": NEAR_TIE, "max_abs_dp": MAX_DP, "mean_of_run_means": MEAN_DP, "red_arm_min": ARM_MIN},
           "summary": summary, "red_arms": red_summary, "rows": runs, "red_rows": red}
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({"control": control, "summary": {g: {k: v.get(k) for k in ("runs", "argmax_non_near_tie", "argmax_near_tie",
                                                                                 "max_abs_dp", "max_abs_dp_run", "mean_of_run_means",
                                                                                 "line_pass")} for g, v in summary.items()},
                      "red_arms": red_summary}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
