#!/usr/bin/env python3
"""The parity bar of this conversion, per reading: the bundle's text against upstream fp32's text for the same 448x448
picture (hf_oracle.py mode p448), where a difference counts only outside the spans that upstream itself reads
differently at 448 and at the picture's original size (mode native): what one 448 picture cannot resolve, so a
reading that differs there is a different guess, not a conversion error. A picture that upstream does not read at
448 (its 448 reading scored by score_ocr.py, --hf-score: character error rate above --unreadable, 0.05, against the
truth, or against its own original-size reading when the picture has no truth; this conversion: the A4 page 0.136 and
the form photo 0.305) is not judged as one picture; it is the page workload, judged on its tiles.
Both rules come from upstream's readings alone, never from the bundle's output.

  python parity_bar.py --hf-p448 hf_p448.jsonl --hf-native hf_native.jsonl --hf-score score_hf.json ROWS.jsonl ...
      [--md OUT.md]

ROWS = JSONL rows with "name" (or "case" = "r_<name>") and "text" (suite_ocr.py, device/score_dev.py). A row's verdict:
  equal      the text equals upstream p448 character for character
  pass-span  every difference lies inside a span where upstream p448 differs from upstream native
  FAIL       some difference lies outside those spans (the span, the bundle text and upstream's are printed)
  not-judged the picture is unreadable at 448 by the rule above (the verdict it would get is printed beside it)
Exit 0 when no judged row is FAIL.
"""
import argparse
import difflib
import json


def load(path):
  return {json.loads(l)["name"]: json.loads(l)["text"].strip() for l in open(path)}


def spans(a, b):
  """Spans of `a` (start, end) that differ in b; an insertion into a is the empty span at its position."""
  return [(i1, i2) for tag, i1, i2, _, _ in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
          if tag != "equal"]


def inside(span, allowed, pad=1):
  s, e = span
  return any(s >= a - pad and e <= b + pad for a, b in allowed)


def verdict(text, ref, native):
  if text == ref:
    return "equal", []
  allowed = spans(ref, native)
  bad = [sp for sp in spans(ref, text) if not inside(sp, allowed)]
  return ("pass-span" if not bad else "FAIL"), bad


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("rows", nargs="+")
  ap.add_argument("--hf-p448", required=True)
  ap.add_argument("--hf-native", required=True)
  ap.add_argument("--md", default=None)
  ap.add_argument("--hf-score", required=True, help="score_ocr.py's JSON of the upstream runs (score_hf.json)")
  ap.add_argument("--unreadable", type=float, default=0.05)
  a = ap.parse_args()
  p448, native = load(a.hf_p448), load(a.hf_native)
  score = {r["name"]: r for r in json.load(open(a.hf_score)) if r["mode"] == "p448"}
  err = {n: r.get("cer", r.get("cer_vs_ref")) for n, r in score.items()}
  unreadable = {n for n, e in err.items() if e is not None and e > a.unreadable}
  out = [f"not judged as one 448 picture (upstream's 448 reading: CER > {a.unreadable} against the truth or its own "
         "original-size reading): " + ", ".join(f"{n} {err[n]:.3f}" for n in sorted(unreadable)), "",
         "| rows | reading | verdict | outside-span differences |", "|---|---|---|---|"]
  fails = judged = passed = 0
  for path in a.rows:
    for line in open(path):
      r = json.loads(line)
      name = r.get("name") or r.get("case", "")[2:]
      if r.get("case", "r_").startswith("pg_") or name not in p448 or not r.get("text"):
        continue
      v, bad = verdict(r["text"].strip(), p448[name], native[name])
      if name in unreadable:
        v = f"not-judged ({v})"
      else:
        judged += 1
        passed += v != "FAIL"
        fails += v == "FAIL"
      tag = r.get("tag") or r.get("label") or ""
      detail = "; ".join(f"{p448[name][s - 10:e + 10]!r}" for s, e in bad[:3])
      out.append(f"| {path.split('/')[-1]} {tag} | {name} | {v} | {detail} |")
  out.append(f"\njudged {judged}, passed {passed}, failed {fails}")
  text = "\n".join(out)
  print(text)
  if a.md:
    open(a.md, "w").write(text + "\n")
  return 1 if fails else 0


if __name__ == "__main__":
  raise SystemExit(main())
