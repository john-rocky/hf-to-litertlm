#!/usr/bin/env python3
"""Score rows of Unlimited-OCR-2 (uocr_oracle.py or rt_uocr.py) against the reference R, case by case on the set of
build_set.py.

  python score_uocr.py --set SET.json --ref ROWS_R.jsonl [--rows ROWS.jsonl ...] [--md OUT.md] [--json OUT.json]

Reference = R (arm "R", no tag, all pages): the original model in fp32 on the original picture(s), base mode, its own
DASH attention and page2d positions, greedy (uocr_oracle.py). Every other row is a form's row, scored against R by its
case's class:
  norm_md   (1) drop the author's re_match spans (<|ref|>..<|/ref|><|det|>..<|/det|> and <|det|>label [..]<|/det|>),
            \\coloneqq -> :=, \\eqqcolon -> =:, drop <PAGE>; drop each block head <BLOCK>label[x1, y1, x2, y2]<FORK>
            and every <CHILD> mark (contents stay); (2) NFKC, whitespace collapsed to one space and stripped (all
            whitespace removed when R's text holds CJK characters).
  text <= 0.02, photo <= 0.05, page <= 0.05, multi (whole text) <= 0.05: CER = Lev(norm_md(arm), norm_md(R)) / len.
  table     cell grid (HTML <tr>/<td|th>, else a markdown | table, else OTSL): cells equal to R's at the same
            (row, col) >= ceil(0.9 x R's cells) (18/20 for 20 cells).
  formula   norm_tex(arm) == norm_tex(R) (norm_md (1) with the block heads dropped, NFKC, outer $ / \\[ \\] / \\( \\)
            dropped, \\, \\; \\: \\! and "\\ " dropped, all whitespace dropped).
  kie       the printed fields (needles) found in norm_kie(output); pass = the arm finds every field R finds.
  stop      finish == "stop" (EOS id 1) on every row; multi: generated <PAGE> (128827) count == pages.
A form passes a workload when every case of it passes: region (text, photo, table, formula, kie: 14 cases),
page (4), multi (1). Instruments (not the verdict): cer_tags (norm_md keeping the block heads), truth scores, the
common prefix of the generated ids with R's, tokens, seconds, guard / n-gram moved steps, block labels equal to R's
and the largest box coordinate difference.
Rows of one multi-page document run one page at a time (tag "page<i>", any arm) are scored as one row
"<arm>:split": texts and ids concatenated in page order, stop = every page row stopped. A row's arm key carries its
label when it has one (several bundles and backends in one call).
R's own rows are checked as a reference: (i) cut at the cap, (ii) unreadable for the class, (iii) far from the truth
where truth exists.
"""
import argparse
import html
import importlib.util
import json
import math
import os
import re
import sys
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("uocr_lev_fast", os.path.join(HERE, "lev_fast.py"))
_lev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lev)
lev = _lev.lev_fast

CJK = re.compile(r"[　-〿一-鿿＀-￯]")
REF = re.compile(r"(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)", re.S)
DET = re.compile(r"(<\|det\|>\s*([A-Za-z_][\w-]*)\s*(\[[^\]]+\])\s*<\|/det\|>)", re.S)
HEAD = re.compile(r"<BLOCK>([^\[\n]*?)\[([^\]\n]*)\]<FORK>")
PAGE_ID = 128827
LINE = {"text": 0.02, "photo": 0.05, "page": 0.05, "multi": 0.05}
WORK = {"region": ("text", "photo", "table", "formula", "kie"), "page": ("page",), "multi": ("multi",)}
NEEDLES = {"receipt": ["GREENLEAF MARKET", "2025-03-14", "17.60"],
           "receipt_jpg_kie": ["MR. D.I.Y. SDN BHD",
                               "LOT 1851-A & 1851-B, JALAN KPB 6, KAWASAN PERINDUSTRIAN BALAKONG, "
                               "43300 SERI KEMBANGAN, SELANGOR",
                               "001092886528", "18-10-17", "5.90", "0.33"]}


def strip1(s, amended=True):
    s = s or ""
    for pat in (REF, DET):
        for m in pat.findall(s):
            s = s.replace(m[0], "")
    s = s.replace("\\coloneqq", ":=").replace("\\eqqcolon", "=:").replace("<PAGE>", "\n")
    if amended:
        s = HEAD.sub("", s).replace("<CHILD>", "")
    return s


def norm_md(s, cjk, amended=True):
    s = unicodedata.normalize("NFKC", strip1(s, amended))
    return re.sub(r"\s+", "", s) if cjk else re.sub(r"\s+", " ", s).strip()


def cer(a, b):
    return lev(a, b) / max(1, len(b))


def norm_tex(s):
    s = unicodedata.normalize("NFKC", strip1(s)).strip()
    s = re.sub(r"^\$+|\$+$", "", s).strip()
    s = re.sub(r"^\\\[|\\\]$", "", s).strip()
    s = re.sub(r"^\\\(|\\\)$", "", s).strip()
    s = re.sub(r"\\[,;:!]", "", s)
    s = re.sub(r"\\ ", "", s)
    return re.sub(r"\s+", "", s)


def norm_kie(s):
    s = re.sub(r"<[^>]+>", " ", strip1(s))
    s = html.unescape(s)
    s = re.sub(r"[*#|_]", " ", s)
    s = unicodedata.normalize("NFKC", s).casefold()
    return re.sub(r"\s+", " ", s).strip()


def parse_cells(raw):
    raw = strip1(raw)

    def cell(c):
        return unicodedata.normalize("NFKC", re.sub(r"<[^>]+>", "", c)).strip()

    if "<td" in raw or "<th" in raw:
        return [[cell(c) for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, flags=re.S)]
                for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", raw, flags=re.S)]
    md = [ln.strip() for ln in raw.splitlines() if ln.strip().startswith("|")]
    if md:
        return [[cell(c) for c in ln.strip("|").split("|")] for ln in md if not re.fullmatch(r"\|[\s:|-]+\|?", ln)]
    rows = []
    for line in re.split(r"<nl>", re.sub(r"</?otsl>", "", raw)):
        parts = re.split(r"(<fcel>|<ecel>|<lcel>|<ucel>|<xcel>)", line)
        cells, tag, txt = [], None, ""
        for p in parts:
            if p in ("<fcel>", "<ecel>", "<lcel>", "<ucel>", "<xcel>"):
                if tag is not None:
                    cells.append(cell(txt) if tag == "<fcel>" else "")
                tag, txt = p, ""
            elif tag is not None:
                txt = p
        if tag is not None:
            cells.append(cell(txt) if tag == "<fcel>" else "")
        if cells:
            rows.append(cells)
    return rows


def cells_equal(a, b):
    n = sum(len(r) for r in b)
    hit = sum(1 for i, row in enumerate(b) for k, v in enumerate(row) if i < len(a) and k < len(a[i]) and a[i][k] == v)
    return hit, n


def kie_found(text, name):
    t = norm_kie(text)
    return [n for n in NEEDLES[name] if norm_kie(n) in t]


def layout(text):
    return [(m.group(1), [int(x) for x in re.findall(r"-?\d+", m.group(2))]) for m in HEAD.finditer(text or "")]


def judge(case, row, ref):
    cls, t, rt = case["class"], row["text"], ref["text"]
    if cls in LINE:
        c = bool(CJK.search(rt))
        v = cer(norm_md(t, c), norm_md(rt, c))
        out = {"ok": v <= LINE[cls], "score": f"CER {v:.4f}", "value": v}
        if cls == "multi":
            n_pg = row["gen_ids"].count(PAGE_ID)
            out["ok"] = out["ok"] and n_pg == row["n_pages_doc"]
            out["score"] += f", <PAGE> {n_pg}/{row['n_pages_doc']}"
        return out
    if cls == "table":
        hit, n = cells_equal(parse_cells(t), parse_cells(rt))
        need = math.ceil(0.9 * n) if n else 1
        return {"ok": n > 0 and hit >= need, "score": f"cells {hit}/{n} (need {need})", "value": hit / max(n, 1)}
    if cls == "formula":
        p, q = norm_tex(t), norm_tex(rt)
        return {"ok": bool(q) and p == q, "score": "tex equal" if p == q else f"tex differ (CER {cer(p, q):.3f})",
                "value": float(p == q)}
    if cls == "kie":
        fr, fa = kie_found(rt, case["name"]), kie_found(t, case["name"])
        return {"ok": all(f in fa for f in fr), "score": f"fields {len(fa)}/{len(NEEDLES[case['name']])} (R {len(fr)})",
                "value": len(fa)}
    raise ValueError(cls)


def truth_score(case, row):
    cls, t, g = case["class"], row["text"], case["truth"]
    if g is None:
        return None, None
    if cls in ("text", "photo", "page"):
        if isinstance(g, (dict, list)):
            return None, None
        c = bool(CJK.search(g))
        v = cer(norm_md(t, c), norm_md(g, c))
        return v, f"CER {v:.4f}" + ("" if case["truth_kind"] == "truth" else " (vs author .md)")
    if cls == "multi":
        c = bool(CJK.search("".join(g)))
        v = cer(norm_md(t, c), norm_md("\n".join(g), c))
        pages = t.split("<PAGE>")[1:] if "<PAGE>" in t else [t]
        per = [round(cer(norm_md(p, c), norm_md(gp, c)), 4) for p, gp in zip(pages, g)]
        return v, f"CER {v:.4f} per page {per}"
    if cls == "table":
        hit, n = cells_equal(parse_cells(t), g)
        return hit, f"cells {hit}/{n}"
    if cls == "formula":
        p, q = norm_tex(t), norm_tex(g)
        v = cer(p, q)
        return v, "tex equal" if p == q else f"tex CER {v:.3f}"
    if cls == "kie":
        f = kie_found(t, case["name"])
        return len(f), f"needles {len(f)}/{len(NEEDLES[case['name']])}"
    return None, None


def r_breaches(case, row):
    cls, t = case["class"], row["text"]
    b = []
    if row["finish"] != "stop":
        b.append(f"(i) finish={row['finish']}")
    if cls == "table" and not sum(len(r) for r in parse_cells(t)):
        b.append("(ii) table 0 cells")
    if cls == "kie" and not kie_found(t, case["name"]):
        b.append("(ii) kie 0 fields")
    if cls == "formula" and not norm_tex(t):
        b.append("(ii) empty formula")
    if cls == "multi" and row["gen_ids"].count(PAGE_ID) != row["n_pages_doc"]:
        b.append(f"(ii) <PAGE> {row['gen_ids'].count(PAGE_ID)}")
    if case["truth_kind"].startswith("truth"):
        v, s = truth_score(case, row)
        if cls in ("text", "photo", "page", "multi", "formula") and v is not None and v > 0.10:
            b.append(f"(iii) truth {s}")
        if cls == "table" and v is not None and v < 18:
            b.append(f"(iii) truth {s}")
        if cls == "kie" and v is not None and v < len(NEEDLES[case["name"]]):
            b.append(f"(iii) truth {s}")
    return b


def prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def load_rows(paths):
    rows = []
    for p in paths:
        rows += [json.loads(ln) for ln in open(p)]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, help="set.json of build_set.py")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--rows", nargs="*", default=[])
    ap.add_argument("--md")
    ap.add_argument("--json")
    a = ap.parse_args()
    setd = json.load(open(a.set))
    cases = setd["cases"]
    npg = {im["name"]: len(im["pages"]) for im in setd["images"]}
    ref = {r["image"]: r for r in load_rows([a.ref]) if r["arm"] == "R" and not r.get("tag")}
    for r in ref.values():
        r["n_pages_doc"] = npg[r["image"]]
    arms, split = {}, {}
    for r in load_rows(a.rows):
        r["n_pages_doc"] = npg[r["image"]]
        base = (r["label"] + ":" if r.get("label") else "") + r["arm"]
        if (r.get("tag") or "").startswith("page"):
            split.setdefault((base, r["image"]), []).append(r)
            continue
        key = base + (":" + r["tag"] if r.get("tag") else "")
        arms.setdefault(key, {})[r["image"]] = r
    for (base, img), rs in split.items():
        rs.sort(key=lambda x: x["pages"][0])
        if len(rs) == npg[img]:
            arms.setdefault(base, {})[img] = {
                "image": img, "arm": base + ":split", "text": "".join(x["text"] for x in rs),
                "gen_ids": sum((x["gen_ids"] for x in rs), []), "n_gen": sum(x.get("n_gen") or 0 for x in rs),
                "finish": "stop" if all(x["finish"] == "stop" for x in rs) else "+".join(x["finish"] for x in rs),
                "seconds": round(sum(x.get("seconds") or 0 for x in rs), 2),
                # runtime rows carry no guard / n-gram instrument (moved_steps is HF's)
                "moved_steps": sum((x.get("moved_steps") or [] for x in rs), []),
                "n_pages_doc": npg[img], "per_page_n_gen": [x["n_gen"] for x in rs]}
    out = {"R": [], "forms": {}}
    lines = ["| case | class | R n_gen | R stop | R vs truth | R breaches |", "|---|---|---|---|---|---|"]
    n_breach = 0
    for c in cases:
        r = ref.get(c["image"])
        if r is None:
            continue
        b = r_breaches(c, r)
        n_breach += len(b) > 0
        v, s = truth_score(c, r)
        out["R"].append({"case": c["name"], "n_gen": r["n_gen"], "finish": r["finish"], "truth": s, "breaches": b})
        lines.append(f"| {c['name']} | {c['class']} | {r['n_gen']} | {r['finish']} | {s or '—'} | {'; '.join(b) or '—'} |")
    lines.append(f"\nR rows breaching the stop condition's instruments: {n_breach}")
    out["R_breach_rows"] = n_breach
    for form, rows in sorted(arms.items()):
        res = []
        for c in cases:
            r, rr = rows.get(c["image"]), ref.get(c["image"])
            if r is None or rr is None:
                continue
            j = judge(c, r, rr)
            stop_ok = r["finish"] == "stop"
            cj = bool(CJK.search(rr["text"]))
            lay_r, lay_a = layout(rr["text"]), layout(r["text"])
            box_d = max((max(abs(x - y) for x, y in zip(p[1], q[1])) for p, q in zip(lay_a, lay_r)
                         if len(p[1]) == len(q[1]) == 4), default=None)
            res.append({"case": c["name"], "class": c["class"], "ok": j["ok"] and stop_ok, "line_ok": j["ok"],
                        "stop_ok": stop_ok, "score": j["score"], "value": j["value"], "n_gen": r["n_gen"],
                        "R_n_gen": rr["n_gen"], "finish": r["finish"],
                        "prefix": prefix(r["gen_ids"], rr["gen_ids"]), "seconds": r.get("seconds"),
                        "moved": len(r.get("moved_steps") or []),
                        "cer_tags": round(cer(norm_md(r["text"], cj, False), norm_md(rr["text"], cj, False)), 4),
                        "truth": truth_score(c, r)[1],
                        "labels_equal": [x[0] for x in lay_a] == [x[0] for x in lay_r], "box_max_abs": box_d})
        work = {}
        for w, classes in WORK.items():
            rs = [x for x in res if x["class"] in classes]
            if rs:
                work[w] = {"pass": all(x["ok"] for x in rs), "n": len(rs), "n_ok": sum(x["ok"] for x in rs),
                           "fails": [x["case"] for x in rs if not x["ok"]]}
        out["forms"][form] = {"rows": res, "work": work}
        lines.append(f"\n### {form}: " + ", ".join(f"{w} {'PASS' if v['pass'] else 'FAIL'} {v['n_ok']}/{v['n']}"
                                                   + (f" (fail: {', '.join(v['fails'])})" if v["fails"] else "")
                                                   for w, v in work.items()))
        lines.append("| case | line | stop | score | n_gen (R) | prefix | cer_tags | truth | labels= | box Δ | moved | s |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for x in res:
            lines.append(f"| {x['case']} | {'ok' if x['line_ok'] else 'FAIL'} | {x['finish']} | {x['score']} | "
                         f"{x['n_gen']} ({x['R_n_gen']}) | {x['prefix']} | {x['cer_tags']} | {x['truth'] or '—'} | "
                         f"{'y' if x['labels_equal'] else 'n'} | {x['box_max_abs']} | {x['moved']} | {x['seconds']} |")
    txt = "\n".join(lines)
    print(txt)
    if a.md:
        open(a.md, "w").write(txt + "\n")
    if a.json:
        json.dump(out, open(a.json, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
