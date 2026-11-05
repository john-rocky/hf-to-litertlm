#!/usr/bin/env python3
"""The two fixed-size arms against the reference on the records of sets/set_device.json (90 records, category x label, 15 each): the AUROC difference (arm minus reference) with a paired bootstrap inside those cells, per the whole set and per category, and the records on the other side of each cutoff. S448 stretches both pictures to 448 x 448, P448 pads them to a square first (ref_run.py --arm). Evidence for the form of this repository: it needs no fixed picture size. Numbers only.

    $REF -I conversion/ref_run.py <work> --set device --arm S448 --dtype fp32 --fast-dwconv      (and P448; the reference's rows come from ref_run.py --set A)
    $REF -I conversion/fixed_arms.py <work>      -> results/fixed_size_arms.json
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

WORK = Path(sys.argv[1]).resolve()


def load(tag):
    return {r["item_id"]: r for r in map(json.loads, (WORK / f"results/ref/{tag}.jsonl").read_text().splitlines())}


def auroc(s, y):
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    return 100 * ((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg))


def rounded(v):
    return None if v != v else round(float(v), 2)


ref = load("A__base__fp32__dw")
rng = np.random.default_rng(20261109)
doc = {}
for arm in ("S448", "P448"):
    if not os.path.exists(WORK / f"results/ref/device__{arm}__fp32__dw.jsonl"):
        continue
    r = load(f"device__{arm}__fp32__dw")
    ids = sorted(i for i in r if i in ref)
    a, b = np.array([ref[i]["prob_1"] for i in ids]), np.array([r[i]["prob_1"] for i in ids])
    y, cat = np.array([r[i]["label"] for i in ids]), np.array([r[i]["category"] for i in ids])
    cells = defaultdict(list)
    for n, i in enumerate(ids):
        cells[(r[i]["category"], r[i]["label"])].append(n)
    groups = {"all": None, "text": "text", "formula": "formula", "table": "table"}
    point = {g: auroc(b[cat == c] if c else b, y[cat == c] if c else y) - auroc(a[cat == c] if c else a, y[cat == c] if c else y) for g, c in groups.items()}
    boot = defaultdict(list)
    for _ in range(4000):
        idx = np.concatenate([rng.choice(v, len(v)) for v in cells.values()])
        ai, bi, yi, ci = a[idx], b[idx], y[idx], cat[idx]
        for g, c in groups.items():
            m = ci == c if c else np.ones(len(idx), bool)
            boot[g].append(auroc(bi[m], yi[m]) - auroc(ai[m], yi[m]))
    doc[arm] = {"records": len(ids), "reference_auroc": rounded(auroc(a, y)), "arm_auroc": rounded(auroc(b, y)), "difference": {g: rounded(v) for g, v in point.items()},
                "interval_95": {g: [rounded(np.percentile(v, 2.5)), rounded(np.percentile(v, 97.5))] for g, v in boot.items()},
                "other_side_of_0.5": int(((a >= 0.5) != (b >= 0.5)).sum()), "other_side_of_0.9": int(((a >= 0.9) != (b >= 0.9)).sum()),
                "abs_dprob_mean": round(float(np.abs(a - b).mean()), 3), "abs_dprob_max": round(float(np.abs(a - b).max()), 3)}
if not doc:
    raise SystemExit("no arm's rows in results/ref: run ref_run.py --set device --arm S448 first")
(WORK / "results/fixed_size_arms.json").write_text(json.dumps(doc, indent=1))
print(json.dumps(doc))
