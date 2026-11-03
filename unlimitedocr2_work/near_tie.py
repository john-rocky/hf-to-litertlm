#!/usr/bin/env python3
"""The near-tie columns of every failed gate row, and each form's count under the near-tie rule.

  python near_tie.py --set SET.json --ref ROWS_R.jsonl --spread ROWS_R_T4.jsonl --rows ROWS.jsonl [...]
      [--md OUT.md] [--json OUT.json]

For each case a form fails (score_uocr.py decides):
  step     the first step where the arm's ids leave R's = the scorer's common prefix length
  margin   R's top-1 minus top-2 logit at that step (R's own row, field top2 = the original model's fp32 raw logits)
  R2       whether the arm's id at that step is R's top-2 id. Runtime rows carry ids re-tokenized from the text
           (gen_ids_from = "retokenized"), not the runtime's generated ids; the column says so.
  spread   whether R run again with 4 CPU threads (uocr_oracle.py --threads 4, ROWS_R_T4) generated exactly R's ids
           for that case ("not run" when the case is not in that file)
Near-tie rule: a failed case does not count when the margin is below 0.1 logit, the arm took R's top-2 there, and the
4-thread re-run equals R; every other failed case counts. A form's count under the rule = its passes + the failed
cases that do not count.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
BAR = 0.1


def load(p):
  return [json.loads(l) for l in open(p) if l.strip()]


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--set", required=True, help="set.json of build_set.py")
  ap.add_argument("--ref", required=True)
  ap.add_argument("--spread", required=True)
  ap.add_argument("--rows", nargs="+", required=True)
  ap.add_argument("--md")
  ap.add_argument("--json")
  a = ap.parse_args()
  R = {r["image"]: r for r in load(a.ref) if r.get("arm") == "R" and not r.get("tag")}
  S = {r["image"]: r for r in load(a.spread)}  # the 4-thread re-run rows
  with tempfile.TemporaryDirectory() as td:
    sj = os.path.join(td, "s.json")
    subprocess.run([sys.executable, "-I", os.path.join(HERE, "score_uocr.py"), "--set", a.set, "--ref", a.ref,
                    "--rows", *a.rows, "--json", sj], check=True, stdout=subprocess.DEVNULL)
    forms = json.load(open(sj))["forms"]
  arm_rows = {}
  for p in a.rows:
    for r in load(p):
      if not r.get("tag"):
        arm_rows[(r.get("label"), r["image"])] = r
  out, md = [], ["| form | case | the line | step | R top-1 − top-2 | arm id = R's 2nd | ids | R with 4 threads = R | near-tie rule |",
                 "|---|---|---|--:|--:|---|---|---|---|"]
  summary = {}
  for form, fd in forms.items():
    label = form.split(":")[0]
    rows = fd["rows"]
    n_ok = sum(1 for x in rows if x["ok"])
    not_counted = 0
    for x in rows:
      if x["ok"]:
        continue
      case = x["case"]
      image = case[:-4] if case.endswith("_kie") else case
      ar = arm_rows.get((label, image))
      rr = R.get(image)
      rec = {"form": form, "case": case, "score": x["score"], "step": None, "margin": None, "arm_id": None,
             "R1": None, "R2": None, "is_R2": None, "ids": None, "spread": "not run", "counts": True}
      if ar is None or rr is None:
        rec["note"] = "split or missing row: not computed"
      else:
        k = x.get("prefix")
        g, t2 = ar.get("gen_ids") or [], rr.get("top2") or []
        rec["ids"] = ar.get("gen_ids_from") or "runtime"
        if k is not None and k < len(g) and k < len(t2):
          (i1, i2), (l1, l2) = t2[k]
          rec.update(step=k, margin=round(l1 - l2, 4), arm_id=g[k], R1=i1, R2=i2, is_R2=(g[k] == i2))
        sp = S.get(image)
        if sp is not None:
          rec["spread"] = "equal" if sp.get("gen_ids") == rr.get("gen_ids") else "differs"
        rec["counts"] = not (rec["margin"] is not None and rec["margin"] < BAR and rec["is_R2"]
                             and rec["spread"] == "equal")
      if not rec["counts"]:
        not_counted += 1
      out.append(rec)
      step = "—" if rec["step"] is None else str(rec["step"])
      margin = "—" if rec["margin"] is None else str(rec["margin"])
      r2 = "—" if rec["is_R2"] is None else ("yes" if rec["is_R2"] else "no")
      ids = "" if rec["arm_id"] is None else " (arm %d; R %d / %d)" % (rec["arm_id"], rec["R1"], rec["R2"])
      verdict = "counts" if rec["counts"] else "does not count"
      md.append("| %s | %s | FAIL %s | %s | %s | %s%s | %s | %s | %s |" % (
          form, case, x["score"], step, margin, r2, ids, rec["ids"] or "—", rec["spread"], verdict))
    summary[form] = {"cases": len(rows), "line_pass": n_ok, "failed": len(rows) - n_ok,
                     "failed_not_counted": not_counted, "pass_near_tie_rule": n_ok + not_counted}
  md += ["", "| form | cases | pass on the line | failed | not counted (near ties) | pass under the near-tie rule |",
         "|---|--:|--:|--:|--:|--:|"]
  for f, s in summary.items():
    md.append(f"| {f} | {s['cases']} | {s['line_pass']} | {s['failed']} | {s['failed_not_counted']} | "
              f"{s['pass_near_tie_rule']}/{s['cases']} |")
  text = "\n".join(md)
  print(text)
  if a.md:
    open(a.md, "w").write(text + "\n")
  if a.json:
    json.dump({"bar_logit": BAR, "rows": out, "summary": summary}, open(a.json, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
  main()
