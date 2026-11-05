#!/usr/bin/env python3
"""The reference's own spread: the source's published bfloat16 form on the CPU against its float32 form, on the records of sets/floor_ids.json (9 records, the same records in both). A scale for reading the other differences, not a tolerance.

    $REF -I conversion/ref_run.py <work> --set floor --dtype fp32 --fast-dwconv
    $REF -I conversion/ref_run.py <work> --set floor --dtype bf16
    $REF -I conversion/reference_floor.py <work>      -> results/reference_floor.json
"""
import json
import sys
from pathlib import Path

WORK = Path(sys.argv[1]).resolve()


def load(tag):
    return {r["item_id"]: r for r in map(json.loads, (WORK / f"results/ref/{tag}.jsonl").read_text().splitlines())}


a, b = load("floor__base__fp32__dw"), load("floor__base__bf16")
ids = sorted(set(a) & set(b))
if not ids:
    raise SystemExit("no record in both forms: run ref_run.py --set floor first")
d = [abs(a[i]["prob_1"] - b[i]["prob_1"]) for i in ids]
doc = {"n": len(ids), "abs_dprob_max": max(d), "abs_dprob_mean": sum(d) / len(d), "rows": [{"item_id": i, "fp32": a[i]["prob_1"], "bf16": b[i]["prob_1"]} for i in ids], "seconds_mean_bf16": sum(b[i]["seconds"] for i in ids) / len(ids)}
(WORK / "results/reference_floor.json").write_text(json.dumps(doc, indent=1))
print(f"REFERENCE_FLOOR {doc['n']} records: bfloat16 against float32 on the CPU, |d prob_1| max {doc['abs_dprob_max']:.4f}, mean {doc['abs_dprob_mean']:.4f}")
