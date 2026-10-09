"""Reproduce the check of this repository in one command: the public fixtures through D1Omni, against the provider's own
float32 answers stored with them.

    python host/verify.py --repo . [--gpu] [--mode text|image|audio|all] [--threads N] [--out result.json]
        [--check-files]

--gpu runs D1Omni(accelerator="gpu"): each graph at its precision from contract.json (precision.mac_metal.graphs).

For every record of fixtures/public_<mode>.json: D1Omni.probabilities(state, questions, images / audio) and the stored
probabilities of the provider's code (PyTorch float32, CPU). Per mode it prints the rows, argmax agreement (rows whose
reference top-2 gap is <= 0.02 = near-ties, counted apart), max / p95 / mean |dp| over every option, the options that
change side of 0.5 and 0.9, and non-finite rows. Bar per mode: argmax equal on every row outside the near-ties, max
|dp| <= 0.02, mean |dp| <= 0.002, no non-finite value; and in text mode the red arm (red_arm_000 = tv4_000 with one
word of the instructions changed) must land more than 0.02 from tv4_000's reference, so the comparison can see a
one-word change. Exit code 0 = every mode run passes. --check-files adds the sha256 of every file in contract.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import d1_omni as O  # noqa: E402

BAR = {"max_abs_dp": 0.02, "mean_abs_dp": 0.002, "near_tie_gap": 0.02, "red_arm_min_dp": 0.02}
CUTOFFS = (0.5, 0.9)
MODES = ("text", "image", "audio")


def top2_gap(p):
    s = sorted(p, reverse=True)
    return s[0] - s[1] if len(s) > 1 else 1.0


def run_mode(model, repo, mode):
    doc = json.loads((repo / "fixtures" / f"public_{mode}.json").read_text())
    rows, t0 = [], time.time()
    for rec in doc["records"]:
        kw = {}
        if rec["media"]:
            path = repo / "fixtures" / rec["media"]["file"]
            if O.sha256_file(path) != rec["media"]["sha256"]:
                raise ValueError(f"{path}: sha256 differs from the fixture")
            kw = {"images": [path]} if mode == "image" else {"audio": path}
        names = [e["name"] for e in rec["expected"]]
        probs = model.probabilities(rec["state"], [rec["questions"][n] for n in names], **kw)
        for e, p in zip(rec["expected"], probs):
            rows.append({"key": f"{rec['id']}/{e['name']}", "P": e["P"], "n": len(e["ids"]), "probs": p,
                         "reference": e["probs"]})
    return doc, rows, time.time() - t0


def stats(rows):
    dps, per = [], []
    agree = n_main = agree_nt = n_nt = 0
    flips, nt_flips, nonfinite, cross = [], [], [], {str(c): 0 for c in CUTOFFS}
    for r in rows:
        p, ref = r["probs"], r["reference"]
        if not np.isfinite(p).all():
            nonfinite.append(r["key"])
            continue
        dp = [abs(a - b) for a, b in zip(p, ref)]
        dps.extend(dp)
        near = top2_gap(ref) <= BAR["near_tie_gap"]
        same = int(np.argmax(p)) == int(np.argmax(ref))
        if near:
            n_nt, agree_nt = n_nt + 1, agree_nt + same
            if not same:
                nt_flips.append(r["key"])
        else:
            n_main, agree = n_main + 1, agree + same
            if not same:
                flips.append(r["key"])
        for c in CUTOFFS:
            cross[str(c)] += sum((a >= c) != (b >= c) for a, b in zip(p, ref))
        per.append({"key": r["key"], "max_abs_dp": max(dp), "argmax_equal": same, "near_tie": near})
    d = np.asarray(dps, np.float64)
    s = {"rows": len(rows), "options": int(d.size), "argmax_outside_near_tie": f"{agree}/{n_main}",
         "near_tie": f"{agree_nt}/{n_nt}", "argmax_flips": flips, "near_tie_flips": nt_flips,
         "max_abs_dp": float(d.max()) if d.size else None, "p95_abs_dp": float(np.percentile(d, 95)) if d.size else None,
         "mean_abs_dp": float(d.mean()) if d.size else None, "cutoff_crossings": cross,
         "nonfinite_rows": nonfinite, "top5_by_dp": sorted(per, key=lambda x: -x["max_abs_dp"])[:5]}
    s["bar_pass"] = bool(rows and not nonfinite and not flips and s["max_abs_dp"] <= BAR["max_abs_dp"]
                         and s["mean_abs_dp"] <= BAR["mean_abs_dp"])
    return s


def red_arm(rows):
    by = {r["key"]: r for r in rows}
    a, b = by.get("red_arm_000/answer"), by.get("tv4_000/answer")
    if a is None or b is None:
        return {"available": False}
    d = max(abs(x - y) for x, y in zip(a["probs"], b["reference"]))
    return {"available": True, "max_abs_dp_vs_tv4_000_reference": d, "breaks_bar": d > BAR["red_arm_min_dp"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", default=str(HERE.parent))
    ap.add_argument("--gpu", action="store_true",
                    help="the GPU accelerator, each graph at its precision from contract.json (default: CPU)")
    ap.add_argument("--mode", choices=MODES + ("all",), default="all")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", help="write the result as JSON")
    ap.add_argument("--check-files", action="store_true", help="sha256 of every file in contract.json")
    a = ap.parse_args(argv)
    repo = Path(a.repo)
    modes = MODES if a.mode == "all" else (a.mode,)
    t0 = time.time()
    import importlib.metadata as md

    out = {"written": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "repo": str(repo),
           "accelerator": "gpu" if a.gpu else "cpu", "precision": None,
           "threads": None if a.gpu else a.threads, "python": sys.version.split()[0], "executable": sys.executable,
           "ai_edge_litert": md.version("ai-edge-litert"), "bar": BAR, "modes": {}}
    ok = True
    with O.D1Omni(repo, accelerator="gpu" if a.gpu else "cpu", threads=a.threads) as model:
        out["precision"] = model.graph_precision if a.gpu else None
        if a.check_files:
            files = model.check_files()
            bad = {k: v for k, v in files.items() if v is not True}
            out["files"] = {"checked": len(files), "bad": bad}
            ok &= not bad
            print(f"files: {len(files) - len(bad)}/{len(files)} sha256 and bytes as in contract.json")
        print(f"{'mode':6} {'rows':>5} {'argmax':>9} {'near-tie':>8} {'max|dp|':>9} {'p95|dp|':>9} {'mean|dp|':>9} "
              f"{'x0.5':>4} {'x0.9':>4} {'nonfin':>6} {'s':>6}  bar")
        for mode in modes:
            doc, rows, secs = run_mode(model, repo, mode)
            s = stats(rows)
            if mode == "text":
                s["red_arm"] = red_arm(rows)
                if s["red_arm"]["available"]:
                    s["bar_pass"] = s["bar_pass"] and s["red_arm"]["breaks_bar"]
            s["seconds"] = round(secs, 1)
            s["records"] = len(doc["records"])
            out["modes"][mode] = {**s, "per_row": rows}
            ok &= s["bar_pass"]
            print(f"{mode:6} {s['rows']:>5} {s['argmax_outside_near_tie']:>9} {s['near_tie']:>8} "
                  f"{s['max_abs_dp']:>9.2e} {s['p95_abs_dp']:>9.2e} {s['mean_abs_dp']:>9.2e} "
                  f"{s['cutoff_crossings']['0.5']:>4} {s['cutoff_crossings']['0.9']:>4} {len(s['nonfinite_rows']):>6} "
                  f"{secs:>6.1f}  {'PASS' if s['bar_pass'] else 'FAIL'}")
            if mode == "text" and s["red_arm"]["available"]:
                ra = s["red_arm"]
                print(f"       red arm: red_arm_000 vs tv4_000's reference max|dp| {ra['max_abs_dp_vs_tv4_000_reference']:.4f}"
                      f" ({'> 0.02: the comparison sees a one-word change' if ra['breaks_bar'] else '<= 0.02: FAIL'})")
    out["seconds"] = round(time.time() - t0, 1)
    out["PASS"] = bool(ok)
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"verify: {'PASS' if ok else 'FAIL'} ({out['accelerator']}, {', '.join(modes)}, {out['seconds']} s)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
