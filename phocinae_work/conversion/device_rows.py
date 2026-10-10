"""Step 7a: the rows the phone gate app runs (fixtures/device_rows_L{L}.json), per window L.

    python device_rows.py --L 512 [--groups test,control,flip_rev] [--max-rows 0]

Each row: id, ids (unpadded), markers, qtype, K. Groups: test (the 2,000 decisions), control (5 rows), flip_rev (the 600
reversed-order choice rows of the author's flip400 protocol). `timing_ids` = the fixture-median row, a 300-token row,
a 200-token row and a 128-token row (whatever fits L): the app times each one with 5 warm-ups + N calls.
"""
import argparse
import json

from common import W, dump_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--groups", default="test,control,flip_rev")
    ap.add_argument("--max-rows", type=int, default=0)
    a = ap.parse_args()
    fx = json.loads((W / "fixtures" / "rows_laya.json").read_text())["rows"]
    groups = set(a.groups.split(","))
    rows = []
    for r in fx:
        g = r["group"]
        if g == "flip":
            g = "flip_rev" if r["id"].endswith("#rev") else "flip_rand"
        if g not in groups or r["row_len"] > a.L:
            continue
        rows.append(dict(id=r["id"], ids=r["ids"], markers=r["markers"], qtype=r["qtype"], K=r["K"]))
    if a.max_rows:
        rows = rows[: a.max_rows]
    test = [r for r in fx if r["group"] == "test" and r["row_len"] <= a.L]
    lens = sorted(r["row_len"] for r in test)
    med = lens[len(lens) // 2]
    picks = {"fixture_median_row": min(test, key=lambda r: abs(r["row_len"] - med))}
    for name, target in (("row_300", 300), ("row_200", 200), ("row_128", 128)):
        if target <= a.L:
            picks[name] = min(test, key=lambda r: abs(r["row_len"] - target))
    timing_ids = []
    for name, r in picks.items():
        if r["id"] not in timing_ids:
            timing_ids.append(r["id"])
    doc = dict(L=a.L, groups=sorted(groups), rows_n=len(rows), timing_ids=timing_ids,
               timing_rows={name: dict(id=r["id"], row_len=r["row_len"], K=r["K"]) for name, r in picks.items()}, rows=rows)
    out = W / "fixtures" / f"device_rows_L{a.L}.json"
    dump_json(out, doc)
    print(json.dumps({k: v for k, v in doc.items() if k != "rows"}, indent=1), str(out.relative_to(W)))


if __name__ == "__main__":
    main()
