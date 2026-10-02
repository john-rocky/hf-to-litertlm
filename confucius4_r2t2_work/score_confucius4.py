#!/usr/bin/env python3
"""Pinned FLEURS scorer for the Confucius4-R2T2 conversion (pure python, any venv).

norm_text / norm_chars / lev and WER_FORMULA / CER_FORMULA are copied verbatim from
funasr_nano_work/score_fleurs.py, so these numbers are directly comparable with the FLEURS rows of the Fun-ASR-Nano
conversion
(same 150 clips of ~/code/coreai/_funasr_nano/fixtures, same normalisation, corpus = sum of distances / sum of
reference lengths).

  en_us        WER, words = norm_text(raw_transcription) vs norm_text(hyp)
  cmn_hans_cn  CER, chars = norm_chars(...)
  ja_jp        CER, chars = norm_chars(...)

Two uses:
  vs-reference  python3 score_confucius4.py ref --hyp out/ref_eager_vendor.jsonl:text [--hyp <omni jsonl>:text ...]
                hypothesis files are JSONL with a clip id field ("clip" or "id") and the given text field.
  hyp-vs-hyp    python3 score_confucius4.py pair --a out/ref_eager_hfprompt_crops.jsonl:text --b <omni jsonl>:text
                (the 6a gate: eager-crop transcript as the "reference" for the LiteRT-crop transcript, same norms)
"""
import argparse
import json
import os
import re
import unicodedata

FIX = os.environ.get("C4_FIXTURES", os.path.expanduser("~/code/coreai/_funasr_nano/fixtures"))

# ---- verbatim from funasr_nano_work/score_fleurs.py ------------------------------------------------------------
CER_FORMULA = ("CER = sum(levenshtein(norm(ref), norm(hyp))) / sum(len(norm(ref))); norm(s) = NFKC(s).lower() with every "
               "char of Unicode category P* or Z* and every whitespace char removed; ref = FLEURS raw_transcription")
WER_FORMULA = ("WER = sum(levenshtein_words(norm_text(ref), norm_text(hyp))) / sum(len(norm_text(ref))); norm_text = "
               "upper(), '-' -> ' ', [^A-Z0-9' ] -> ' ', split; ref = FLEURS raw_transcription")


def norm_text(s):
    s = s.upper().replace("-", " ")
    return re.sub(r"[^A-Z0-9' ]+", " ", s).split()


def norm_chars(s):
    s = unicodedata.normalize("NFKC", s).lower()
    return [ch for ch in s if not (unicodedata.category(ch)[0] in "PZ" or ch.isspace())]


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[-1]


def score(config, ref, hyp):
    if config == "en_us":
        r, h = norm_text(ref), norm_text(hyp)
    else:
        r, h = norm_chars(ref), norm_chars(hyp)
    return lev(r, h), len(r)
# ---- end verbatim ------------------------------------------------------------------------------------------------


def meta():
    return {c["name"]: c for c in json.load(open(os.path.join(FIX, "meta.json")))["clips"]}


def config_of(name, m):
    base = name.split("__")[0]  # crops are named <clip>__crop5s
    c = m.get(base)
    return c["source"]["config"] if c and c["path"].startswith("fleurs/") else None


def load_hyp(spec):
    path, field = spec.rsplit(":", 1)
    rows = {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("type") in ("header", "footer"):
            continue
        key = r.get("clip", r.get("id"))
        rows[key] = r.get(field) if r.get(field) is not None else ""
    return path, field, rows


def agg_rows(pairs, m):
    """pairs: list of (name, ref_text, hyp_text) -> per-config totals."""
    agg, rows = {}, []
    for name, ref, hyp in pairs:
        cfg = config_of(name, m)
        if cfg is None:
            continue
        e, n = score(cfg, ref, hyp)
        a = agg.setdefault(cfg, {"metric": "WER" if cfg == "en_us" else "CER", "n": 0, "err": 0, "ref_len": 0,
                                 "exact_equal": 0, "empty_hyp": 0})
        a["n"] += 1
        a["err"] += e
        a["ref_len"] += n
        a["exact_equal"] += int(ref == hyp)
        a["empty_hyp"] += int(not (hyp or "").strip())
        rows.append({"clip": name, "config": cfg, "err": e, "ref_len": n, "ref": ref, "hyp": hyp})
    for a in agg.values():
        a["rate_pct"] = round(100.0 * a["err"] / a["ref_len"], 3) if a["ref_len"] else None
    return agg, rows


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("ref")
    r.add_argument("--hyp", action="append", required=True)
    r.add_argument("--out", default="")
    p = sub.add_parser("pair")
    p.add_argument("--a", required=True, help="the side used as reference (eager-crop)")
    p.add_argument("--b", required=True, help="the side scored (LiteRT-crop)")
    p.add_argument("--out", default="")
    args = ap.parse_args()
    m = meta()
    doc = {"formulas": {"en_us": WER_FORMULA, "cmn_hans_cn": CER_FORMULA, "ja_jp": CER_FORMULA},
           "fixtures": os.path.join(FIX, "meta.json"), "scorer": "score_confucius4.py (norm/lev verbatim from "
           "funasr_nano_work/score_fleurs.py)"}
    if args.cmd == "ref":
        doc["mode"] = "vs FLEURS raw_transcription"
        doc["systems"] = {}
        for spec in args.hyp:
            path, field, hyp = load_hyp(spec)
            pairs = [(n, m[n]["reference_text"]["raw_transcription"], hyp.get(n, ""))
                     for n in m if m[n]["path"].startswith("fleurs/")]
            missing = [n for n, _, _ in pairs if n not in hyp]
            agg, rows = agg_rows(pairs, m)
            doc["systems"][spec] = {"summary": agg, "missing": missing, "rows": rows}
            print(spec, "missing", len(missing))
            for cfg, a in agg.items():
                print("  ", cfg, a["metric"], a["rate_pct"], "err", a["err"], "/", a["ref_len"], "n", a["n"],
                      "empty", a["empty_hyp"])
    else:
        pa, fa, A = load_hyp(args.a)
        pb, fb, B = load_hyp(args.b)
        doc["mode"] = "hyp vs hyp: a = reference side, b = scored side, same norms"
        doc["a"], doc["b"] = args.a, args.b
        names = sorted(set(A) | set(B))
        pairs = [(n, A.get(n, ""), B.get(n, "")) for n in names]
        agg, rows = agg_rows(pairs, m)
        doc["summary"] = agg
        doc["only_a"] = sorted(set(A) - set(B))
        doc["only_b"] = sorted(set(B) - set(A))
        doc["rows"] = rows
        for cfg, a in agg.items():
            print(cfg, a["metric"], a["rate_pct"], "err", a["err"], "/", a["ref_len"], "exact", a["exact_equal"], "/",
                  a["n"], "empty_b", a["empty_hyp"])
    if args.out:
        with open(args.out, "w") as f:
            json.dump(doc, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
