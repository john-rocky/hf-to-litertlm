#!/usr/bin/env python3
"""sens_table.py - the sensitivity table from teacher_force.py records: which quantized group moves the reference's
answers, and by how much at the positions that decide them. It picks candidates; it is not a gate.

  python -I sens_table.py --dir DIR --ref ref [--label NAME=DESCRIPTION ...] [--out sens.md] [label ...]

Per label (default: every DIR/<label>.json recorded with --ref <ref>), per group of cases (e.g. thinking off / on):
  all_ref     cases whose teacher positions all keep the reference's top-1 (and among the cases marked `counted`)
  positions   positions that keep it
  max |d|     max |d logit| over the vocabulary (max / mean over positions), rms d logit
  sigma       standard deviation of the margin change on the reference's top-1 / top-2 pair, every position
  near rms    rms of that margin change only at the positions that decide an answer: counted cases, reference gap
              < --near (default 0.5), up to the case's first change. A form whose near rms is about the size of the
              smallest reference gaps flips those answers; sigma over every position does not tell (most positions
              have large gaps).
  changed     every case that changes: first position and the reference's gap there
  model       expected changed counted cases if every position took independent N(0, sigma^2) margin noise:
              sum over cases of 1 - prod Phi(gap / sigma). Check it on an all-int8 control before using it.
The parameters a group moves from fp16 to int8 come from its recipe's .info.json when DIR/<label>.info.json or
--recipes/<label>.info.json exists.
"""
import argparse
import json
import math
import os
import sys

import numpy as np


def phi(x):
  return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def expected_changed(ref_rows, sigma, group):
  if sigma <= 0:
    return 0.0
  tot = 0.0
  for r in ref_rows:
    if r["group"] != group or not r["counted"]:
      continue
    p = 1.0
    for g in r["gap"]:
      p *= phi(g / sigma)
    tot += 1.0 - p
  return tot


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("labels", nargs="*")
  ap.add_argument("--dir", required=True)
  ap.add_argument("--ref", required=True)
  ap.add_argument("--recipes", default=None)
  ap.add_argument("--label", action="append", default=[], metavar="NAME=DESCRIPTION")
  ap.add_argument("--near", type=float, default=0.5)
  ap.add_argument("--out", default=None)
  a = ap.parse_args()
  desc = dict(x.split("=", 1) for x in a.label)
  ref = json.load(open(os.path.join(a.dir, f"{a.ref}.json")))
  labels = a.labels or sorted(f[:-5] for f in os.listdir(a.dir)
                              if f.endswith(".json") and not f.endswith(".info.json") and f != f"{a.ref}.json")
  out = {"ref": a.ref, "near": a.near, "rows": []}
  groups = sorted({r["group"] for r in ref["rows"]})
  for lab in labels:
    d = json.load(open(os.path.join(a.dir, f"{lab}.json")))
    if not isinstance(d, dict) or d.get("ref") != a.ref or "summary" not in d:
      continue   # a recipe (a list), the cases file, this tool's own --out json, other records
    row = {"label": lab, "what": desc.get(lab, ""), "decoder_bytes": d["decoder_bytes"]}
    for base in (a.dir, a.recipes):
      ip = os.path.join(base, f"{lab}.info.json") if base else None
      if ip and os.path.exists(ip):
        info = json.load(open(ip))
        row["int8_params_m"] = round(sum(x["params"] for x in info["int8"]) / 1e6, 1)
        row["what"] = row["what"] or ", ".join(f"int8 {x['group']} ({x['weights']} weights)" for x in info["int8"])
    for grp in groups:
      rr = [r for r in d["rows"] if r["group"] == grp]
      s = d["summary"][grp]
      ch, near = [], []
      for r in rr:
        mc = [pm - g for pm, g in zip(r["pair_margin"], r["ref_gap"])]
        ch += mc
        if r["counted"]:
          stop = r["first_div_ref"] if r["first_div_ref"] is not None else len(mc) - 1
          near += [m for m, g in zip(mc[:stop + 1], r["ref_gap"][:stop + 1]) if g < a.near]
      sigma = float(np.std(ch)) if ch else float("nan")
      row[grp] = {"all_ref": f"{s['all_ref']}/{s['rows']}", "all_ref_counted": f"{s['all_ref_counted']}/{s['counted_rows']}",
                  "positions": f"{s['pos_eq_ref']}/{s['positions']}", "dmax_max": round(s["dmax_max"], 3),
                  "dmax_mean": round(s["dmax_mean"], 3), "drms_mean": round(s["drms_mean"], 4),
                  "sigma": round(sigma, 4), "near_rms": round(float(np.sqrt(np.mean(np.square(near)))), 4) if near else None,
                  "near_positions": len(near),
                  "model_changed": round(expected_changed(ref["rows"], sigma, grp), 2),
                  "changed": [f"{r['key'].split('|')[0]}@{r['first_div_ref']} ({r['ref_gap'][r['first_div_ref']]:.3f})"
                              for r in rr if not r["all_ref"]]}
    out["rows"].append(row)
  head = ("| label | int8 | M params | " + " | ".join(f"{g}: all / counted" for g in groups) + " | max\\|d\\| (max, mean) | "
          "sigma | near rms | model | changed (position, reference gap) |")
  lines = [head, "|" + "---|" * (8 + len(groups))]
  for r in out["rows"]:
    cells = [r["label"], r["what"], str(r.get("int8_params_m", ""))]
    cells += [f"{r[g]['all_ref']} / {r[g]['all_ref_counted']}" for g in groups]
    cells.append(" / ".join(f"{r[g]['dmax_max']}, {r[g]['dmax_mean']}" for g in groups))
    cells.append(" / ".join(str(r[g]["sigma"]) for g in groups))
    cells.append(" / ".join(str(r[g]["near_rms"]) for g in groups))
    cells.append(" / ".join(str(r[g]["model_changed"]) for g in groups))
    cells.append("; ".join(f"{g}: " + ", ".join(r[g]["changed"]) for g in groups if r[g]["changed"]) or "—")
    lines.append("| " + " | ".join(cells) + " |")
  text = "\n".join(lines) + "\n"
  if a.out:
    open(a.out, "w").write(text)
    json.dump(out, open(a.out.rsplit(".", 1)[0] + ".json", "w"), indent=1)
  print(text)
  return 0


if __name__ == "__main__":
  sys.exit(main())
