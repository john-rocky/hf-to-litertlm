#!/usr/bin/env python3
"""Round 5b: the e2e line over every length of one accelerator (NumPy and the standard library only).

    venv-host/bin/python scripts/r5b_score.py --accelerator gpu [--subset all|cpu20] [--dry] [--partial]
        -> results/r5b_score_<acc>.json (results/r5b_dry_score_<acc>.json with --dry)
    exit 0 = every selected run is scored and the line holds; 2 = runs are missing (--partial: the line on what is
    there holds); 3 = the line is broken on the runs that are there (a stop)

Reads results/r5b_e2e_<acc>_L<L>.jsonl of the five lengths. The line (scripts/r5b_sets.py) over all runs
together and per group (set x arm and length); red arms: each moves > 0.02 from its base record's reference; the
response: structure (model, usage, answer keys and types, legends, field names) and decisions (non-near-tie) equal to
the reference's systemone_response, and the 4-digit values that differ (counted); respond() of the probabilities equal
to the decide() body; non-finite hidden values at the real positions."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

K = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(K / "scripts"))
import r5b_sets as S  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--accelerator", choices=["gpu", "cpu"], required=True)
    ap.add_argument("--subset", default=None, choices=["all", "cpu20"])
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--partial", action="store_true")
    a = ap.parse_args()
    subset = a.subset or ("all" if a.accelerator == "gpu" else "cpu20")
    prefix = "r5b_dry" if a.dry else "r5b_e2e"
    runs = S.load_runs()
    want = {L: [r["key"] for r in S.select(runs, L, subset)] for L in S.BUCKETS}
    rows, missing = [], {}
    for L in S.BUCKETS:
        p = K / f"results/{prefix}_{a.accelerator}_L{L}.jsonl"
        got = {}
        if p.exists():
            for ln in p.read_text().splitlines():
                if ln.strip():
                    x = json.loads(ln)
                    got[x["key"]] = x
        rows += [got[k] for k in want[L] if k in got]
        missing[L] = [k for k in want[L] if k not in got]
    by_group = defaultdict(list)
    for r in rows:
        by_group[f"{r['set']}/{r['arm']}"].append(r)
        by_group[f"L{r['length']}"].append(r)
    reds = [r for r in rows if r["set"] == "red"]
    body = {"runs": len(rows),
            "structure_equal": f"{sum(r['body_structure_equal'] for r in rows)}/{len(rows)}",
            "decisions_equal_non_near_tie": f"{sum(r['decisions_equal_non_near_tie'] for r in rows)}/{len(rows)}",
            "decision_differences": [{"run": r["key"], "question_id": q, **d} for r in rows
                                     for q, d in r["decisions"].items() if d["reference"] != d["host"]],
            "bodies_equal": f"{sum(r['body_equal'] for r in rows)}/{len(rows)}",
            "digit_fields": sum(r["digits_fields"] for r in rows),
            "digit_fields_differing": sum(r["digits_fields"] - r["digits_equal"] for r in rows),
            "runs_with_differing_digits": sum(r["digits_fields"] != r["digits_equal"] for r in rows),
            "respond_equals_decide": f"{sum(r['respond_equals_decide'] for r in rows)}/{len(rows)}"}
    total = S.line(rows)
    red = {"runs": len(reds), "moved": f"{sum(r['red']['moves_over_line'] for r in reds)}/{len(reds)}",
           "max_moves": {r["id"]: r["red"]["max_move"] for r in reds}}
    nonfinite = sum(r["nonfinite_real"] for r in rows)
    n_missing = sum(len(v) for v in missing.values())
    ok_line = total["pass"] and nonfinite == 0 and body["structure_equal"] == f"{len(rows)}/{len(rows)}" \
        and body["decisions_equal_non_near_tie"] == f"{len(rows)}/{len(rows)}" \
        and body["respond_equals_decide"] == f"{len(rows)}/{len(rows)}"
    doc = {"what": f"round 5b e2e line, {a.accelerator} ({subset})" + (" [dry: reference stand-in graphs]" if a.dry else ""),
           "accelerator": a.accelerator, "subset": subset, "complete": n_missing == 0,
           "missing": {f"L{L}": v for L, v in missing.items() if v}, "line": total, "red_arms": red, "response": body,
           "nonfinite_real": nonfinite,
           "groups": {g: {k: v for k, v in S.line(rs).items() if k != "line"} for g, rs in sorted(by_group.items())},
           "ms_decide_median_by_length": {f"L{L}": float(sorted(x["ms_decide"] for x in rows if x["length"] == L)[
               len([x for x in rows if x["length"] == L]) // 2]) for L in S.BUCKETS
               if any(x["length"] == L for x in rows)},
           "pass": ok_line and n_missing == 0}
    out = K / f"results/{'r5b_dry_score' if a.dry else 'r5b_score'}_{a.accelerator}.json"
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({"complete": doc["complete"], "missing": sum(len(v) for v in missing.values()),
                      "line": {k: total[k] for k in ("runs", "argmax_equal_non_near_tie", "near_tie", "max_abs_dp",
                                                     "max_abs_dp_at", "mean_of_run_means", "pass")},
                      "red": red["moved"], "structure": body["structure_equal"],
                      "decisions": body["decisions_equal_non_near_tie"], "digits_differing":
                          body["digit_fields_differing"], "nonfinite": nonfinite, "pass": doc["pass"]}, indent=1))
    if not ok_line and rows:
        return 3
    return 0 if n_missing == 0 else (0 if a.partial else 2)


if __name__ == "__main__":
    raise SystemExit(main())
