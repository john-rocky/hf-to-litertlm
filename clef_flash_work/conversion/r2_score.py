#!/usr/bin/env python3
"""Round 2: the hidden states of one r2_check run set -> the author's JointSchemaHead (venv-ref, scripts/clef_score.py's
Scorer = the reference's head, threads 10) -> option probabilities -> the parity line.

    HF_HUB_OFFLINE=1 venv-ref/bin/python scripts/r2_score.py --check results/r2_check_v2_fp16fc_L1024_cpu.json \
        [--compare-score results/score_cpu_L1024.json --compare-cache cache/r1/cpu --compare-tag fp32_cpu_r1]
        -> results/r2_score_<form>_L<L>_<acc>.json (refuses to overwrite)

Rows = the check's merged rows, which must be exactly the pre-registered set (scripts/r2_sets.py). Reference row per
kind: fixture = ref/records_ref.json, heldout = Core AI's held-out records (rev 17f0b0ad, same weights; re-check
results/r2_heldout_check.json), red = ref/records_ref_red.json; a red arm also against its BASE record's reference
(fixtures/red_arms.json `base_id`): it moves when some option of the base's questions differs by > 0.02.
Control (every scored run, as clef_score.py): the reference's own last_hidden through this scorer must reproduce the
reference's logits within 1e-6 (bit-equal runs counted), else the run stops.
Line (pre-registered, unrounded probabilities), over the fixture + held-out runs (red arms apart): every non-near-tie question
keeps its argmax, max |dp| <= 0.02, mean of run means <= 0.002. Reported per kind too. gate_pass = line_pass and every red
arm of the set moves (and, on Metal, the check's metal_all_nodes_one_partition).
--compare-*: the same runs of another form (its score json and hidden cache, e.g. r1's fp32 CPU run at L1024): per run
the hidden max |diff| at the real positions and per option |p - p_other|; summary = max / mean of run means / argmax
agreement (the V2-vs-fp32 row: the fp16 rounding of the weights)."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

K = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(K / "scripts"))
import r2_sets as S  # noqa: E402
from clef_score import ARM_MIN, MAX_DP, MEAN_DP, NEAR_TIE, Scorer, compare  # noqa: E402


def line_summary(runs: list) -> dict:
    qs = [q for r in runs for q in r["questions"]]
    nn = [q for q in qs if not q["near_tie"]]
    nt = [q for q in qs if q["near_tie"]]
    if not runs:
        return {"runs": 0}
    worst = max(runs, key=lambda r: r["max_abs_dp"])
    allo = [x for r in runs for x in r["option_abs_dp"]]
    s = {"runs": len(runs), "questions": len(qs), "near_tie_questions": len(nt),
         "argmax_non_near_tie": f"{sum(q['argmax_equal'] for q in nn)}/{len(nn)}",
         "argmax_near_tie": f"{sum(q['argmax_equal'] for q in nt)}/{len(nt)}",
         "max_abs_dp": worst["max_abs_dp"], "max_abs_dp_run": f"{worst['id']}/{worst['arm']}",
         "max_abs_dp_question": max(worst["questions"], key=lambda q: q["max_abs_dp"])["question_id"],
         "mean_of_run_means": float(np.mean([r["mean_abs_dp"] for r in runs])),
         "p95_option_abs_dp": float(np.percentile(allo, 95)),
         "questions_over_0p02": sum(q["max_abs_dp"] > MAX_DP for q in qs),
         "flips": [f"{r['id']}/{q['question_id']}" for r in runs for q in r["questions"] if not q["argmax_equal"]]}
    s["line_pass"] = bool(all(q["argmax_equal"] for q in nn) and s["max_abs_dp"] <= MAX_DP
                          and s["mean_of_run_means"] <= MEAN_DP)
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", required=True, help="merged results/r2_check_<tag>.json")
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--compare-score", default="")
    ap.add_argument("--compare-cache", default="")
    ap.add_argument("--compare-tag", default="")
    a = ap.parse_args()
    chk = json.loads((K / a.check).read_text())
    L, form, acc = chk["L"], chk["form"], chk["acc"]
    tag = f"{form}_L{L}_{acc}"
    out = K / f"results/r2_score_{tag}.json"
    assert not out.exists(), f"refusing to overwrite {out}"
    runs_set = S.gate_runs(L, acc)
    keys = [S.key(r) for r in runs_set]
    assert sorted(f"{x['id']}__{x['arm']}" for x in chk["rows"]) == sorted(keys), "check rows != the pre-registered set"
    cache = K / chk["cache_dir"]
    arms = {x["arm_id"]: x for x in json.loads((K / "fixtures/red_arms.json").read_text())["arms"]}
    sc = Scorer(a.threads)
    cmp_rows = {}
    if a.compare_score:
        cdoc = json.loads((K / a.compare_score).read_text())
        for r in cdoc["rows"] + cdoc.get("red_rows", []):
            cmp_rows[f"{r['id']}__{r['arm']}"] = r
    scored, red_runs, control = [], [], {"runs": 0, "logits_bit_equal": 0, "max_abs_dlogit": 0.0}
    vs = []
    for r in runs_set:
        row = S.ref_row(r)
        zr = np.load(S.ref_npz(r))
        lg_ctl = sc.logits(zr["last_hidden"], row)
        eq = all(lg.tolist() == q["logits"] for lg, q in zip(lg_ctl, row["questions"]))
        dl = max(float(np.abs(np.asarray(lg.tolist()) - np.asarray(q["logits"])).max()) for lg, q in zip(lg_ctl, row["questions"]))
        control["runs"] += 1
        control["logits_bit_equal"] += eq
        control["max_abs_dlogit"] = max(control["max_abs_dlogit"], dl)
        hid = np.load(cache / f"{S.key(r)}.npy")
        pr = sc.probs(sc.logits(hid, row))
        c = compare(row, pr)
        c.update(id=r["id"], arm=r["arm"], kind=r["kind"], source=row["source"], tokens=row["tokens"],
                 control_logits_bit_equal=eq,
                 option_abs_dp=[abs(float(x) - float(y)) for q, p in zip(row["questions"], pr) for x, y in zip(p, q["probs"])])
        if r["kind"] == "red":
            info = arms[r["id"]]
            base = next(x for x in S.fixture_rows() if x["id"] == info["base_id"])
            mv = compare(base, pr)
            ref_mv = compare(base, [q["probs"] for q in row["questions"]])
            c["vs_base_reference"] = {"base_id": info["base_id"], "changed_question": info["question_id"],
                                      "max_abs_dp": mv["max_abs_dp"], "moves": mv["max_abs_dp"] > ARM_MIN,
                                      "argmax_changed": [q["question_id"] for q in mv["questions"] if not q["argmax_equal"]],
                                      "reference_arm_vs_base_max_abs_dp": ref_mv["max_abs_dp"]}
            red_runs.append(c)
        else:
            scored.append(c)
        k = S.key(r)
        if a.compare_score and k in cmp_rows and (K / a.compare_cache / f"{k}.npy").exists():
            other = cmp_rows[k]
            dp = [abs(float(x) - float(y)) for q, qo in zip(c["questions"], other["questions"])
                  for x, y in zip(q["probs"], qo["probs"])]
            h_other = np.load(K / a.compare_cache / f"{k}.npy")
            vs.append({"key": k, "kind": r["kind"], "hidden_max_abs": float(np.abs(hid - h_other).max()),
                       "max_abs_dp": max(dp), "mean_abs_dp": float(np.mean(dp)),
                       "argmax_equal": all(q["argmax"] == qo["argmax"] for q, qo in zip(c["questions"], other["questions"]))})
    assert control["max_abs_dlogit"] <= 1e-6, f"scorer is not the reference head: {control}"
    summary = line_summary(scored)
    per_kind = {kd: line_summary([x for x in scored if x["kind"] == kd]) for kd in ("fixture", "heldout")}
    red_summary = {"arms": len(red_runs), "moved": [f"{x['id']}: {x['vs_base_reference']['max_abs_dp']:.4f}" for x in red_runs],
                   "all_move": all(x["vs_base_reference"]["moves"] for x in red_runs) if red_runs else None,
                   "vs_own_reference_max_abs_dp": max((x["max_abs_dp"] for x in red_runs), default=None)}
    metal = chk.get("metal_all_nodes_one_partition")
    gate_pass = bool(summary.get("line_pass") and (red_summary["all_move"] is not False) and (metal is not False))
    doc = {"what": f"round 2: {chk['tflite']} on {acc} -> author's JointSchemaHead (venv-ref) -> the parity line",
           "check": a.check, "tflite": chk["tflite"], "tflite_realpath": chk.get("tflite_realpath"), "L": L, "form": form,
           "acc": acc, "scorer": sc.info, "control": control,
           "line": {"near_tie": NEAR_TIE, "max_abs_dp": MAX_DP, "mean_of_run_means": MEAN_DP, "red_arm_min": ARM_MIN},
           "summary": summary, "per_kind": per_kind, "red_arms": red_summary,
           "metal_all_nodes_one_partition": metal, "gate_pass": gate_pass,
           "verdict": f"{'PASS' if gate_pass else 'FAIL'}: {form} L{L} {acc} ({chk['tflite']})",
           "rows": scored, "red_rows": red_runs}
    if vs:
        doc["vs_other"] = {"tag": a.compare_tag, "score": a.compare_score, "cache": a.compare_cache, "runs": len(vs),
                           "hidden_max_abs": max(x["hidden_max_abs"] for x in vs), "max_abs_dp": max(x["max_abs_dp"] for x in vs),
                           "mean_of_run_means": float(np.mean([x["mean_abs_dp"] for x in vs])),
                           "argmax_equal_runs": f"{sum(x['argmax_equal'] for x in vs)}/{len(vs)}", "rows": vs}
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({"verdict": doc["verdict"], "control": control, "summary": summary,
                      "per_kind": {k: {x: v.get(x) for x in ("runs", "max_abs_dp", "mean_of_run_means", "argmax_non_near_tie")}
                                   for k, v in per_kind.items()}, "red_arms": red_summary,
                      "vs_other": {k: v for k, v in doc.get("vs_other", {}).items() if k != "rows"}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
