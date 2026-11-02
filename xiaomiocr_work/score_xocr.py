#!/usr/bin/env python3
"""Score runs of Xiaomi-OCR-0 (rows of hf_xocr.py or rt_xocr.py) against the upstream model's own reading of the
original picture (HF native), case by case on the set of make_set.py.

  python score_xocr.py --set SET.json --runs A.jsonl [B.jsonl ...] [--ref-runs NATIVE.jsonl] [--md OUT.md] [--json OUT.json]

The reference of a case is its row with mode "native" and pos1d false (from --ref-runs, else from --runs). Every other
row (an arm: s{G} / p{G} / t{G} / t{G}n, POS1D, a GGUF row ...) gets the line of its case's class:
  text     CER(arm, native) <= 0.02            photo  <= 0.05            page  <= 0.05      (norm_text of score_ocr.py)
  table    cells equal to native's at the same (row, col) >= ceil(0.9 x native's cells)   (parse_cells)
  formula  norm_tex(arm) == norm_tex(native)
  kie      every schema key: str(value).strip() equal to native's (first {...} of the output as JSON)
  stop     the arm row ends with EOS (248046 / 248044), not at its cap
Instrument columns (not the verdict): the truth / the author's .md / README JSON scores, the common prefix of the
generated ids with native's, tokens, stop, seconds. A native row is flagged as a weak reference (a "breach") when it is
(i) cut at the cap, (ii) unreadable for the class (table 0 cells, kie no JSON, formula empty), or (iii) with truth:
text CER > 0.10, table < 18/20 cells, kie < all fields.
A form passes a workload when every case of it passes: region = classes text / photo / table / formula / kie, page.
GGUF rows (mode gguf_*): llama.cpp b8680's --jinja renders the generation head as `<think>\n` (enable_thinking true; the
CLI has no switch for it), the model writes `\n\n</think>\n\n` and then the answer; the scored text is what follows the
first `</think>` (the raw text stays in the row).
"""
import argparse
import importlib.util
import json
import math
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, path):
    """Import one file by its path (a peer folder's module must not shadow anything on sys.path)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


score_ocr = _load("minicpmocr_score_ocr", os.path.join(os.path.dirname(HERE), "minicpmocr_work", "score_ocr.py"))
lev_fast = _load("xiaomiocr_lev_fast", os.path.join(HERE, "lev_fast.py")).lev_fast
CJK, cer, norm_tex, norm_text = score_ocr.CJK, score_ocr.cer, score_ocr.norm_tex, score_ocr.norm_text
parse_cells, parse_json = score_ocr.parse_cells, score_ocr.parse_json
score_ocr.lev = lev_fast   # the same distance, bit-parallel (lev_fast.py checks it against the table form)

LINE = {"text": 0.02, "photo": 0.05, "page": 0.05}
REGION = ("text", "photo", "table", "formula", "kie")
SET = {"path": None}


def load_set():
    return {c["name"]: c for c in json.load(open(SET["path"]))["cases"]}


def strip_md(s):
    s = re.sub(r"(?m)^\s*#+\s*", "", s or "")
    return s.replace("**", "")


def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def judge(case, row, ref):
    """-> dict(verdict bool, score str, detail)."""
    cls = case["class"]
    text, raw = row.get("text", ""), row.get("text_raw", row.get("text", ""))
    rtext, rraw = ref.get("text", ""), ref.get("text_raw", ref.get("text", ""))
    if cls in LINE:
        c = bool(CJK.search(rtext))
        v = cer(norm_text(text, c), norm_text(rtext, c))
        return {"ok": v <= LINE[cls], "score": f"CER {v:.4f}", "value": v}
    if cls == "table":
        a, b = parse_cells(raw), parse_cells(rraw)
        n = sum(len(r) for r in b)
        hit = sum(1 for r, rowc in enumerate(b) for k, v in enumerate(rowc) if r < len(a) and k < len(a[r]) and a[r][k] == v)
        need = math.ceil(0.9 * n) if n else 1
        return {"ok": n > 0 and hit >= need, "score": f"cells {hit}/{n} (need {need})", "value": hit / max(n, 1)}
    if cls == "formula":
        p, q = norm_tex(text), norm_tex(rtext)
        return {"ok": bool(q) and p == q, "score": "tex equal" if p == q else f"tex differ (CER {cer(p, q):.3f})",
                "value": float(p == q)}
    if cls == "kie":
        ja, jb = parse_json(text), parse_json(rtext)
        keys = list(case["gt"]) if isinstance(case["gt"], dict) else []
        hit = sum(1 for k in keys if isinstance(ja, dict) and isinstance(jb, dict)
                  and str(ja.get(k, "")).strip() == str(jb.get(k, "")).strip())
        return {"ok": isinstance(jb, dict) and hit == len(keys), "score": f"fields {hit}/{len(keys)}",
                "value": hit / max(len(keys), 1)}
    raise ValueError(cls)


def instrument(case, row):
    """Scores against the truth / the author's reference (not the verdict)."""
    cls, text = case["class"], row.get("text", "")
    raw = row.get("text_raw", text)
    out = {}
    gt = case.get("gt")
    if case["ref_kind"] == "truth":
        if cls in LINE:
            c = bool(CJK.search(gt))
            out["truth"] = f"CER {cer(norm_text(strip_md(text), c), norm_text(gt, c)):.4f}"
            out["truth_value"] = cer(norm_text(strip_md(text), c), norm_text(gt, c))
        elif cls == "table":
            a = parse_cells(raw)
            hit = sum(1 for r, rowc in enumerate(gt) for k, v in enumerate(rowc) if r < len(a) and k < len(a[r]) and a[r][k] == v)
            out["truth"] = f"cells {hit}/20"
            out["truth_value"] = hit
        elif cls == "formula":
            out["truth"] = "tex equal" if norm_tex(text) == norm_tex(gt) else f"tex CER {cer(norm_tex(text), norm_tex(gt)):.3f}"
            out["truth_value"] = float(norm_tex(text) == norm_tex(gt))
        elif cls == "kie":
            j = parse_json(text)
            hit = sum(1 for k, v in gt.items() if isinstance(j, dict) and str(j.get(k, "")).strip() == v)
            out["truth"] = f"fields {hit}/{len(gt)}"
            out["truth_value"] = hit / len(gt)
    elif case["ref_kind"] == "author_md":
        c = bool(CJK.search(case["ref"]))
        v = cer(norm_text(text, c), norm_text(case["ref"], c))
        out["truth"] = f"CER vs author .md {v:.4f}"
        out["truth_value"] = v
    elif case["ref_kind"] == "author_readme_json":
        j = parse_json(text)
        hit = sum(1 for k, v in case["gt"].items() if isinstance(j, dict) and str(j.get(k, "")).strip() == v)
        out["truth"] = f"fields vs README {hit}/{len(case['gt'])}"
        out["truth_value"] = hit / len(case["gt"])
    return out


def native_breaches(case, row):
    cls, text, raw = case["class"], row.get("text", ""), row.get("text_raw", row.get("text", ""))
    b = []
    if row.get("stop") != "eos":
        b.append(f"(i) stop={row.get('stop')}")
    if cls == "table" and not parse_cells(raw):
        b.append("(ii) table unreadable")
    if cls == "kie" and parse_json(text) is None:
        b.append("(ii) no JSON")
    if cls == "formula" and not norm_tex(text):
        b.append("(ii) empty formula")
    if case["ref_kind"] == "truth":
        ins = instrument(case, row)
        v = ins.get("truth_value")
        if cls in LINE and v is not None and v > 0.10:
            b.append(f"(iii) truth CER {v:.3f}")
        if cls == "table" and v is not None and v < 18:
            b.append(f"(iii) truth cells {v}/20")
        if cls == "kie" and v is not None and v < 1:
            b.append(f"(iii) truth {ins['truth']}")
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, help="set.json of make_set.py")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--ref-runs", nargs="*", default=None)
    ap.add_argument("--md", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--by-tag", action="store_true",
                    help="keep rows of different tags apart (separate runs; leave it off when tags name workers)")
    a = ap.parse_args()
    SET["path"] = a.set
    cases = load_set()
    rows = []
    for p in a.runs:
        for line in open(p):
            r = json.loads(line)
            r["_file"] = os.path.basename(p)
            if str(r.get("mode", "")).startswith("gguf_") and "</think>" in (r.get("text") or ""):
                r["text"] = r["text"].split("</think>", 1)[1].strip()
                r["text_raw"] = r["text"]
            rows.append(r)
    refs = {}
    for p in (a.ref_runs or a.runs):
        for line in open(p):
            r = json.loads(line)
            if r["mode"] == "native" and not r.get("pos1d"):
                refs.setdefault(r["name"], r)
    out, lines = [], []
    lines.append("| form | case | class | line | verdict | stop | n_gen | prefix=native | instrument | s |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    breaches = []
    for name, r in sorted(refs.items(), key=lambda kv: list(cases).index(kv[0])):
        b = native_breaches(cases[name], r)
        ins = instrument(cases[name], r)
        lines.append(f"| native | {name} | {cases[name]['class']} | (reference) | {'BREACH ' + '; '.join(b) if b else 'ref'} | "
                     f"{r['stop']} | {r['n_gen']} | — | {ins.get('truth', '')} | {r.get('seconds')} |")
        breaches += [f"{name}: {x}" for x in b]
    forms = {}
    for r in rows:
        if r["mode"] == "native" and not r.get("pos1d"):
            continue
        name = r["name"]
        if name not in refs:
            continue
        case, ref = cases[name], refs[name]
        form = r.get("form") or (f"{r['mode']}{'+pos1d' if r.get('pos1d') else ''}"
                                 + (f" [{r['tag']}]" if a.by_tag and r.get("tag") else ""))
        j = judge(case, r, ref)
        stop_ok = r.get("stop") == "eos"
        ok = j["ok"] and stop_ok
        ins = instrument(case, r)
        pre = common_prefix(r.get("gen_ids") or [], ref.get("gen_ids") or []) if r.get("gen_ids") else None
        rec = {"form": form, "name": name, "class": case["class"], "ok": ok, "line": j["score"], "line_ok": j["ok"],
               "stop": r.get("stop"), "n_gen": r.get("n_gen"), "prefix": pre, "ref_n_gen": ref.get("n_gen"),
               "instrument": ins.get("truth", ""), "seconds": r.get("seconds"), "file": r["_file"],
               "n_image_tokens": r.get("n_image_tokens")}
        out.append(rec)
        forms.setdefault(form, []).append(rec)
        lines.append(f"| {form} | {name} | {case['class']} | {j['score']} | {'PASS' if ok else 'FAIL'} | {r.get('stop')} | "
                     f"{r.get('n_gen')} | {pre}/{ref.get('n_gen')} | {ins.get('truth', '')} | {r.get('seconds')} |")
    summ = ["", "| form | region (14) | page (4) | region fails | page fails |", "|---|---|---|---|---|"]
    for form, recs in forms.items():
        reg = [x for x in recs if x["class"] in REGION]
        pag = [x for x in recs if x["class"] == "page"]
        rf = [x["name"] for x in reg if not x["ok"]]
        pf = [x["name"] for x in pag if not x["ok"]]
        summ.append(f"| {form} | {len(reg) - len(rf)}/{len(reg)} {'PASS' if reg and not rf and len(reg) == 14 else ''} | "
                    f"{len(pag) - len(pf)}/{len(pag)} {'PASS' if pag and not pf and len(pag) == 4 else ''} | "
                    f"{', '.join(rf)} | {', '.join(pf)} |")
    text = "\n".join(lines + summ + ["", f"native instrument breaches ({len(breaches)}): " + ("; ".join(breaches) or "none")])
    print(text)
    if a.md:
        open(a.md, "w").write(text + "\n")
    if a.json:
        json.dump({"rows": out, "native_breaches": breaches}, open(a.json, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
