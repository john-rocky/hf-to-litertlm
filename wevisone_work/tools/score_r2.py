#!/usr/bin/env python3
"""Round 2 scoring with round 1's definitions, unchanged (tools/score_arms.py pair_table + tools/wv_parse.py;
the scoring definitions in score_arms.py): arms of round 2 are read from r2/ref/<arm>/<row>.json (HF arms L, S) and from
r2/suite/<label>/<row>.json (runtime outputs, tools/wv_suite.py), the round-1 arms from ref/<arm>/.

    K/.venv-ref/bin/python -I tools/score_r2.py --pair L:A1 --pair S:B --pair <suite label>:B [--out r2/score_r2.json]
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

K = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(K / "src" / "WeVisOne-VL"))

spec = importlib.util.spec_from_file_location("score_arms", Path(__file__).resolve().parent / "score_arms.py")
SA = importlib.util.module_from_spec(spec)
spec.loader.exec_module(SA)
_orig_load = SA.load


def load(arm, row_id):
    for base in (K / "r2" / "suite", K / "r2" / "ref"):
        p = base / arm / f"{row_id}.json"
        if p.exists():
            return json.loads(p.read_text())
    return _orig_load(arm, row_id)


SA.load = load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", action="append", required=True, help="TEST:REF")
    ap.add_argument("--out", default="r2/score_r2.json")
    a = ap.parse_args()
    rows = json.loads((K / "fixtures" / "fixtures.json").read_text())["rows"]
    out_path = K / a.out
    res = json.loads(out_path.read_text()) if out_path.exists() else {}
    for pair in a.pair:
        test, ref = pair.split(":")
        s, per_row = SA.pair_table(test, ref, rows)
        res[f"{test} vs {ref}"] = {"summary": s, "per_row": per_row}
        print(f"{test} vs {ref}: rows {s['rows']} missing {s['missing']} | box {s['box_matched']}/{s['box_ref']} "
              f"count {s['count_equal']}/{s['count_rows']} (stated {s['count_equal_stated']}) "
              f"point {s['point_matched']}/{s['point_ref']} cer {s['ocr_cer']} (max {s['ocr_max_row_cer']}) "
              f"none {s['none_equal']}/{s['none_rows']} exact {s['exact_rows']} trunc {s['truncated_ref']}/"
              f"{s['truncated_test']} bars {s['bars']}", flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    sys.exit(main())
