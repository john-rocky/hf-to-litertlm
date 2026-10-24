#!/usr/bin/env python3
"""Score the round-1 arms against each other (definitions below: boxes, counts, points, OCR) -> ref/arms_table.json + markdown.

Pairs are (test, reference): fast_vlm contract C / D / E vs B (512) and G vs F (768), repetition penalty A1 vs A0,
resolution B vs A1 and F vs A1, text T1 vs T0. Greedy metrics come from ref/<arm>/<row>.json, teacher-forced ones from
ref/tf/<row>.json (and ref/tf/text.json).

Usage: python -I score_arms.py [--md out.md]
"""
import argparse
import importlib.util
import json
from pathlib import Path

K = Path(__file__).resolve().parents[1]
REF = K / "ref"
FIXTURES = K / "fixtures" / "fixtures.json"

spec = importlib.util.spec_from_file_location("wv_parse", Path(__file__).resolve().parent / "wv_parse.py")
WP = importlib.util.module_from_spec(spec)
spec.loader.exec_module(WP)

PAIRS = [("C", "B"), ("D", "B"), ("E", "B"), ("G", "F"), ("A1", "A0"), ("B", "A1"), ("F", "A1")]
TF_KEY = {("C", "B"): "C_on_B", ("D", "B"): "D_on_B", ("E", "B"): "E_on_B", ("G", "F"): "G_on_F",
          ("B", "A1"): "B_on_A1", ("F", "A1"): "F_on_A1", ("A1", "A0"): "A1rule_on_A0"}
BAR = {"box": 0.90, "count": 0.90, "point": 0.90, "cer": 0.05}


def load(arm, row_id):
    path = REF / arm / f"{row_id}.json"
    return json.loads(path.read_text()) if path.exists() else None


def first_div(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def pair_table(test, ref, rows):
    per_row, agg = [], {"box_ref": 0, "box_matched": 0, "pt_ref": 0, "pt_matched": 0, "count_rows": 0,
                        "count_equal": 0, "count_equal_shapes": 0, "count_equal_stated": 0, "ocr_edits": 0, "ocr_len": 0, "ocr_rows": 0,
                        "ocr_max_cer": 0.0, "none_rows": 0, "none_equal": 0, "rows": 0, "exact": 0,
                        "truncated_ref": 0, "truncated_test": 0, "missing": [],
                        "tf_n": 0, "tf_agree": 0.0, "tf_coord_n": 0, "tf_coord_ok": 0.0, "tf_max_dp": 0.0,
                        "tf_worst_rank": 0}
    for row in rows:
        r, t = load(ref, row["id"]), load(test, row["id"])
        if r is None or t is None:
            agg["missing"].append(row["id"])
            continue
        rp, tp = WP.parse(r["response"], row["task"]), WP.parse(t["response"], row["task"])
        cmp = WP.compare_rows(rp, tp, row["task"])
        fd = first_div(r["new_ids"], t["new_ids"])
        rec = {"row": row["id"], "task": row["task"], "exact": fd is None, "first_div": fd,
               "n_new_ref": r["n_new"], "n_new_test": t["n_new"], "trunc_ref": r["truncated"],
               "trunc_test": t["truncated"], **cmp}
        tf = None
        tf_path = REF / "tf" / f"{row['id']}.json"
        if tf_path.exists() and TF_KEY.get((test, ref)):
            tf = json.loads(tf_path.read_text())["pairs"].get(TF_KEY[(test, ref)])
        rec["tf"] = tf
        per_row.append(rec)
        agg["rows"] += 1
        agg["exact"] += rec["exact"]
        agg["truncated_ref"] += r["truncated"]
        agg["truncated_test"] += t["truncated"]
        agg["box_ref"] += cmp["boxes"]["n_ref"]
        agg["box_matched"] += cmp["boxes"]["matched"]
        agg["pt_ref"] += cmp["points"]["n_ref"]
        agg["pt_matched"] += cmp["points"]["matched"]
        if row["task"] in ("count_box", "count_point"):
            agg["count_rows"] += 1
            agg["count_equal"] += cmp["count"]["equal"]
            agg["count_equal_shapes"] += cmp["count"]["equal_shapes"]
            agg["count_equal_stated"] += cmp["count"]["equal_stated_or_shapes"]
        if row["task"] == "ocr":
            agg["ocr_rows"] += 1
            agg["ocr_edits"] += cmp["ocr"]["edits"]
            agg["ocr_len"] += cmp["ocr"]["ref_len"]
            agg["ocr_max_cer"] = max(agg["ocr_max_cer"], cmp["ocr"]["cer"])
        if rp["is_none"] or tp["is_none"]:
            agg["none_rows"] += 1
            agg["none_equal"] += cmp["none"]["equal"]
        if tf:
            agg["tf_n"] += tf["n"]
            agg["tf_agree"] += tf["top1_agree"] * tf["n"]
            if tf["coord_within5"] is not None:
                agg["tf_coord_n"] += tf["n_coord"]
                agg["tf_coord_ok"] += tf["coord_within5"] * tf["n_coord"]
            agg["tf_max_dp"] = max(agg["tf_max_dp"], tf["max_abs_dp"])
            agg["tf_worst_rank"] = max(agg["tf_worst_rank"], tf["worst_rank"])
    s = {
        "pair": f"{test} vs {ref}", "rows": agg["rows"], "missing": agg["missing"],
        "box_rate": agg["box_matched"] / agg["box_ref"] if agg["box_ref"] else None,
        "box_matched": agg["box_matched"], "box_ref": agg["box_ref"],
        "count_rate": agg["count_equal"] / agg["count_rows"] if agg["count_rows"] else None,
        "count_equal": agg["count_equal"], "count_rows": agg["count_rows"],
        "count_rate_shapes": agg["count_equal_shapes"] / agg["count_rows"] if agg["count_rows"] else None,
        "count_rate_stated": agg["count_equal_stated"] / agg["count_rows"] if agg["count_rows"] else None,
        "count_equal_stated": agg["count_equal_stated"],
        "point_rate": agg["pt_matched"] / agg["pt_ref"] if agg["pt_ref"] else None,
        "point_matched": agg["pt_matched"], "point_ref": agg["pt_ref"],
        "ocr_cer": agg["ocr_edits"] / agg["ocr_len"] if agg["ocr_len"] else None,
        "ocr_max_row_cer": agg["ocr_max_cer"] if agg["ocr_rows"] else None, "ocr_rows": agg["ocr_rows"],
        "none_equal": agg["none_equal"], "none_rows": agg["none_rows"],
        "exact_rows": agg["exact"], "truncated_ref": agg["truncated_ref"], "truncated_test": agg["truncated_test"],
        "tf_top1": agg["tf_agree"] / agg["tf_n"] if agg["tf_n"] else None, "tf_positions": agg["tf_n"],
        "tf_coord_within5": agg["tf_coord_ok"] / agg["tf_coord_n"] if agg["tf_coord_n"] else None,
        "tf_max_abs_dp": agg["tf_max_dp"] if agg["tf_n"] else None,
        "tf_worst_rank": agg["tf_worst_rank"] if agg["tf_n"] else None,
    }
    s["bars"] = {
        "box": s["box_rate"] is None or s["box_rate"] >= BAR["box"],
        "count": s["count_rate"] is None or s["count_rate"] >= BAR["count"],
        "point": s["point_rate"] is None or s["point_rate"] >= BAR["point"],
        "cer": s["ocr_cer"] is None or s["ocr_cer"] <= BAR["cer"],
    }
    return s, per_row


def text_table():
    out = {}
    for arm in ("T0", "T1"):
        d = REF / arm
        if d.exists():
            out[arm] = {p.stem: json.loads(p.read_text()) for p in sorted(d.glob("*.json"))}
    if "T0" not in out or "T1" not in out:
        return None
    tf_path = REF / "tf" / "text.json"
    tf = json.loads(tf_path.read_text())["pairs"] if tf_path.exists() else {}
    rows = []
    for rid in out["T0"]:
        a, b = out["T0"][rid], out["T1"].get(rid)
        if b is None:
            continue
        ra, rb = a["records"][-1], b["records"][-1]
        rows.append({"row": rid, "T0_pass": a["pass"], "T1_pass": b["pass"],
                     "exact": ra["new_ids"] == rb["new_ids"] and all(
                         x["new_ids"] == y["new_ids"] for x, y in zip(a["records"], b["records"])),
                     "first_div_last_turn": first_div(ra["new_ids"], rb["new_ids"]),
                     "T0_answer": ra["answer"], "T1_answer": rb["answer"], "tf": tf.get(rid)})
    return {"rows": rows, "n": len(rows), "exact": sum(r["exact"] for r in rows),
            "T0_pass": sum(r["T0_pass"] for r in rows), "T1_pass": sum(r["T1_pass"] for r in rows),
            "tf_top1_min": min((r["tf"]["top1_agree"] for r in rows if r["tf"]), default=None),
            "tf_max_abs_dp": max((r["tf"]["max_abs_dp"] for r in rows if r["tf"]), default=None)}


def resolution_rows(rows):
    out = []
    for row in rows:
        a1, b, f = load("A1", row["id"]), load("B", row["id"]), load("F", row["id"])
        if a1 is None:
            continue
        rec = {"row": row["id"], "task": row["task"], "size": row["size"],
               "A1_processor_hw": a1.get("processor_hw"), "A1_image_tokens": a1.get("n_image_tokens")}
        for name, x in (("B", b), ("F", f)):
            if x is None:
                continue
            cmp = WP.compare_rows(WP.parse(a1["response"], row["task"]), WP.parse(x["response"], row["task"]), row["task"])
            rec[name] = {"exact": a1["new_ids"] == x["new_ids"], "box": [cmp["boxes"]["matched"], cmp["boxes"]["n_ref"]],
                         "point": [cmp["points"]["matched"], cmp["points"]["n_ref"]],
                         "count": [cmp["count"]["test"], cmp["count"]["ref"]] if row["task"].startswith("count") else None,
                         "cer": cmp["ocr"]["cer"] if row["task"] == "ocr" else None,
                         "none": [cmp["none"]["test"], cmp["none"]["ref"]]}
        out.append(rec)
    return out


def fmt(v, pct=False):
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if pct:
        return f"{100 * v:.1f} %"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def markdown(summary):
    lines = ["| pair | rows | boxes IoU≥0.75 | counts exact, author's count (Total number) | points within 2 % | OCR CER (worst row) | None equal | tokens equal | "
             "TF top-1 | TF coordinate ±5 bins | TF max \\|Δp\\| | TF worst rank | cut at the cap ref/test |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in summary["pairs"]:
        lines.append(
            f"| {s['pair']} | {s['rows']} | {fmt(s['box_rate'], True)} ({s['box_matched']}/{s['box_ref']}) | "
            f"{fmt(s['count_rate'], True)} ({s['count_equal']}/{s['count_rows']}; {s['count_equal_stated']}/{s['count_rows']}) | "
            f"{fmt(s['point_rate'], True)} ({s['point_matched']}/{s['point_ref']}) | "
            f"{fmt(s['ocr_cer'])} ({fmt(s['ocr_max_row_cer'])}) | {s['none_equal']}/{s['none_rows']} | "
            f"{s['exact_rows']}/{s['rows']} | {fmt(s['tf_top1'], True)} | {fmt(s['tf_coord_within5'], True)} | "
            f"{fmt(s['tf_max_abs_dp'])} | {fmt(s['tf_worst_rank'])} | {s['truncated_ref']}/{s['truncated_test']} |")
    return "\n".join(lines)


def tf_top1(row_id, key):
    path = REF / "tf" / f"{row_id}.json"
    if not path.exists():
        return None
    pair = json.loads(path.read_text())["pairs"].get(key)
    return None if pair is None else pair["top1_agree"]


def resolution_markdown(summary):
    lines = ["| row | task | original size W×H | A1 resize h×w / tokens | B vs A1 boxes / points / count / CER / None / TF top-1 | "
             "F vs A1 boxes / points / count / CER / None / TF top-1 |", "|---|---|---|---|---|---|"]

    def cell(x, row_id, key):
        if not x:
            return "—"
        parts = [f"{x['box'][0]}/{x['box'][1]}", f"{x['point'][0]}/{x['point'][1]}",
                 "—" if x["count"] is None else f"{x['count'][0]} vs {x['count'][1]}",
                 "—" if x["cer"] is None else f"{x['cer']:.3f}",
                 "—" if not (x["none"][0] or x["none"][1]) else f"{'none' if x['none'][0] else 'box'} vs "
                                                                f"{'none' if x['none'][1] else 'box'}"]
        t = tf_top1(row_id, key)
        parts.append("—" if t is None else f"{t:.3f}")
        return " / ".join(parts)
    for r in summary["resolution"]:
        hw = r.get("A1_processor_hw") or ["?", "?"]
        lines.append(f"| {r['row']} | {r['task']} | {r['size'][0]}×{r['size'][1]} | {hw[0]}×{hw[1]} / "
                     f"{r.get('A1_image_tokens')} | {cell(r.get('B'), r['row'], 'B_on_A1')} | "
                     f"{cell(r.get('F'), r['row'], 'F_on_A1')} |")
    return "\n".join(lines)


def rows_markdown(summary, pair="E vs B", key="E_on_B"):
    lines = [f"| row | task | boxes m/ref (test) | points m/ref (test) | author's count ref/test | Total number ref/test | CER | "
             f"None ref/test | first divergence | TF top-1 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in summary["per_row"][pair]:
        c = r["count"]
        lines.append(
            f"| {r['row']} | {r['task']} | {r['boxes']['matched']}/{r['boxes']['n_ref']} ({r['boxes']['n_test']}) | "
            f"{r['points']['matched']}/{r['points']['n_ref']} ({r['points']['n_test']}) | {c['ref']}/{c['test']} | "
            f"{c.get('ref_stated')}/{c.get('test_stated')} | "
            f"{'—' if r['task'] != 'ocr' else format(r['ocr']['cer'], '.3f')} | "
            f"{'none' if r['none']['ref'] else '-'}/{'none' if r['none']['test'] else '-'} | {r['first_div']} | "
            f"{'—' if r['tf'] is None else format(r['tf']['top1_agree'], '.3f')} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md")
    ap.add_argument("--rows-md", help="write the per-row E vs B table and the resolution table here")
    args = ap.parse_args()
    rows = json.loads(FIXTURES.read_text())["rows"]
    summary = {"definitions": "score_arms.py docstring", "bars": BAR, "pairs": [], "per_row": {}}
    for test, ref in PAIRS:
        s, per_row = pair_table(test, ref, rows)
        summary["pairs"].append(s)
        summary["per_row"][s["pair"]] = per_row
    summary["text"] = text_table()
    summary["resolution"] = resolution_rows(rows)
    (REF / "arms_table.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
    md = markdown(summary)
    if args.md:
        Path(args.md).write_text(md + "\n")
    if args.rows_md:
        Path(args.rows_md).write_text("## E vs B per row\n\n" + rows_markdown(summary) + "\n\n## resolution\n\n"
                                      + resolution_markdown(summary) + "\n")
    print(md)
    if summary["text"]:
        t = summary["text"]
        print(f"\ntext: T1 vs T0 exact {t['exact']}/{t['n']}, pass T0 {t['T0_pass']} T1 {t['T1_pass']}, "
              f"TF top-1 min {fmt(t['tf_top1_min'])}, max|dp| {fmt(t['tf_max_abs_dp'])}")


if __name__ == "__main__":
    main()
