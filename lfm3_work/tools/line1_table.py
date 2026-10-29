#!/usr/bin/env python3
"""line1_table.py LINE1_JSON [ORACLE_DIR] - the line 1 table (markdown) from tools/line1_module.py's JSON.

One row per sequence and path: positions, near-tie positions left out (HF fp32 top-2 gap < 1e-3), max |dlogit| over
the counted positions (and over all), top-1 agreement; then the three summary numbers (worst max |dlogit|, top-1
agreement over all counted positions, positions left out) per path. With ORACLE_DIR, also the HF bf16 vs fp32 reference
(not a line) from hf_bf16_vs_fp32.json.
"""
import json
import os
import sys


def main(path, oracle=None):
  res = json.load(open(path))
  print(f"cache {res['cache_length']}, bar max|dlogit| <= {res['bar_max_abs_dlogit']}, near-tie gap {res['near_tie_gap']}")
  print("| sequence | path | positions | near-tie left out | max\\|dlogit\\| (all) | top-1 | verdict |")
  print("|---|---|---|---|---|---|---|")
  agg = {}
  for name, row in res["sequences"].items():
    for p, r in row.items():
      print(f"| {name} | {p} | {r['positions']} | {r['excluded_near_tie']} | {r['max_abs_dlogit']:.2e}"
            f" ({r['max_abs_dlogit_all']:.2e}) | {r['top1_agree']}/{r['top1_of']} | {'PASS' if r['pass'] else 'FAIL'} |")
      a = agg.setdefault(p, {"worst": 0.0, "agree": 0, "of": 0, "left_out": 0, "seqs": 0, "positions": 0})
      a["worst"] = max(a["worst"], r["max_abs_dlogit"])
      a["agree"] += r["top1_agree"]
      a["of"] += r["top1_of"]
      a["left_out"] += r["excluded_near_tie"]
      a["seqs"] += 1
      a["positions"] += r["positions"]
  print()
  print("| path | sequences | positions | worst max\\|dlogit\\| | top-1 | near-tie left out |")
  print("|---|---|---|---|---|---|")
  for p, a in agg.items():
    print(f"| {p} | {a['seqs']} | {a['positions']} | {a['worst']:.2e} | {a['agree']}/{a['of']} | {a['left_out']} |")
  print(f"\nline 1 verdict: {'PASS' if res['pass'] else 'FAIL'}")
  if oracle:
    ref = json.load(open(os.path.join(oracle, "hf_bf16_vs_fp32.json")))
    worst = max(r["max_abs_dlogit"] for r in ref.values())
    agree = sum(r["top1_agree"] for r in ref.values())
    of = sum(r["top1_of"] for r in ref.values())
    print(f"reference (not a line): HF bf16 vs HF fp32 worst max|dlogit| {worst:.3f}, top-1 {agree}/{of}")


if __name__ == "__main__":
  main(*sys.argv[1:])
