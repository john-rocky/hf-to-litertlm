#!/usr/bin/env python3
"""Score OCR runs (rows of hf_oracle.py or of the bundle runner) against the fixtures' ground truth.

  python score_ocr.py --runs RUNS.jsonl [RUNS2.jsonl ...] --fixtures FIX.json ... [--ref-runs HF.jsonl]
                      [--md OUT.md] [--json OUT.json]

Per kind:
  text      CER = Levenshtein(norm(pred), norm(gt)) / len(norm(gt)); norm = NFKC, whitespace collapsed to one space
            and stripped (all whitespace removed when the truth holds CJK characters).
  text_ref  no truth: CER against the reference run of the same fixture in --ref-runs (mode native).
  table     the OTSL output (<fcel> text, <ecel>, <lcel>, <ucel>, <xcel>, <nl>; HTML <td> also read) as a cell grid:
            cells equal to the truth at the same (row, column) / truth cells, and CER of the grid written as
            "a|b|c\\n...".
  formula   LaTeX with whitespace and the spacing commands \\, \\; \\: \\! removed: exact (0/1) and CER.
  kie       the first {...} in the output as JSON: fields equal to the truth (stripped strings) / truth fields.
Rows with the same (name, mode) in several --runs files are scored separately (a label column names the file).
"""
import argparse
import json
import os
import re
import unicodedata

CJK = re.compile(r"[　-〿一-鿿＀-￯]")


def lev(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def norm_text(s, cjk):
    s = unicodedata.normalize("NFKC", s or "")
    return re.sub(r"\s+", "", s) if cjk else re.sub(r"\s+", " ", s).strip()


def cer(pred, gt):
    return lev(pred, gt) / max(1, len(gt))


def parse_cells(raw):
    """OTSL (or HTML) -> list of rows of cell strings (merged / empty cells as '')."""
    raw = re.sub(r"</?otsl>", "", raw or "")
    if "<td" in raw or "<th" in raw:
        rows = []
        for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", raw, flags=re.S):
            rows.append([re.sub(r"<[^>]+>", "", c).strip()
                         for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, flags=re.S)])
        return rows
    rows = []
    for line in re.split(r"<nl>", raw):
        parts = re.split(r"(<fcel>|<ecel>|<lcel>|<ucel>|<xcel>)", line)
        cells, cur_tag = [], None
        for p in parts:
            if p in ("<fcel>", "<ecel>", "<lcel>", "<ucel>", "<xcel>"):
                if cur_tag is not None:
                    cells.append(cur_text.strip() if cur_tag == "<fcel>" else "")
                cur_tag, cur_text = p, ""
            elif cur_tag is not None:
                cur_text = p
        if cur_tag is not None:
            cells.append(cur_text.strip() if cur_tag == "<fcel>" else "")
        if cells:
            rows.append(cells)
    return rows


def norm_tex(s):
    s = unicodedata.normalize("NFKC", s or "").strip()
    s = re.sub(r"^\$+|\$+$", "", s.strip())
    s = re.sub(r"^\\\[|\\\]$", "", s.strip())
    s = re.sub(r"\\[,;:!]", "", s)
    s = re.sub(r"\\ ", "", s)
    return re.sub(r"\s+", "", s)


def parse_json(s):
    m = re.search(r"\{.*\}", s or "", flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def score_row(row, item, ref_text=None):
    kind, gt = item["kind"], item["gt"]
    text = row.get("text", "")
    raw = row.get("text_raw", text)
    if kind == "text":
        c = bool(CJK.search(gt))
        return {"cer": cer(norm_text(text, c), norm_text(gt, c))}
    if kind == "text_ref":
        if ref_text is None:
            return {}
        return {"cer_vs_ref": cer(norm_text(text, False), norm_text(ref_text, False))}
    if kind == "table":
        cells = parse_cells(raw)
        hit = sum(1 for r, row_gt in enumerate(gt) for c, v in enumerate(row_gt)
                  if r < len(cells) and c < len(cells[r]) and cells[r][c] == v)
        total = sum(len(r) for r in gt)
        grid = "\n".join("|".join(r) for r in cells)
        grid_gt = "\n".join("|".join(r) for r in gt)
        return {"cells": f"{hit}/{total}", "cell_acc": hit / total, "cer": cer(grid, grid_gt),
                "shape": f"{len(cells)}x{max((len(r) for r in cells), default=0)}"}
    if kind == "formula":
        p, g = norm_tex(text), norm_tex(gt)
        return {"exact": int(p == g), "cer": cer(p, g)}
    if kind == "kie":
        j = parse_json(text)
        hit = sum(1 for k, v in gt.items() if isinstance(j, dict) and str(j.get(k, "")).strip() == v)
        return {"fields": f"{hit}/{len(gt)}", "field_acc": hit / len(gt), "json": j is not None}
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--fixtures", nargs="+", required=True)
    ap.add_argument("--ref-runs", default=None)
    ap.add_argument("--md", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    items = {}
    for p in args.fixtures:
        for it in json.load(open(p))["items"]:
            items[it["name"]] = it
    ref = {}
    if args.ref_runs:
        for line in open(args.ref_runs):
            r = json.loads(line)
            if r["mode"] == "native":
                ref[r["name"]] = r.get("text", "")
    out = []
    for path in args.runs:
        label = os.path.basename(path).rsplit(".", 1)[0]
        for line in open(path):
            r = json.loads(line)
            it = items.get(r["name"])
            if it is None:
                continue
            s = score_row(r, it, ref.get(r["name"]))
            out.append({"label": label, "name": r["name"], "mode": r["mode"], "kind": it["kind"],
                        "n_image_tokens": r.get("n_image_tokens"), "n_gen": r.get("n_gen"), "stop": r.get("stop"),
                        **s})
    lines = ["| run | fixture | mode | kind | img tok | gen | stop | score |", "|---|---|---|---|---|---|---|---|"]
    for o in out:
        sc = ", ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in o.items()
                       if k not in ("label", "name", "mode", "kind", "n_image_tokens", "n_gen", "stop"))
        lines.append(f"| {o['label']} | {o['name']} | {o['mode']} | {o['kind']} | {o['n_image_tokens']} | "
                     f"{o['n_gen']} | {o['stop']} | {sc} |")
    text = "\n".join(lines)
    print(text)
    if args.md:
        open(args.md, "w").write(text + "\n")
    if args.json:
        json.dump(out, open(args.json, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
