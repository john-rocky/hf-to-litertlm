#!/usr/bin/env python3
"""The bundle's text against upstream's (HF fp32, hf_xocr.py) text for the SAME fixed-size picture, in the two position
forms hf_xocr.py runs: 1-D positions (pos1d = the fast_vlm contract: every token one running position) and full
M-RoPE (the model's own). Per bundle row: equal to each (character for character after strip), CER (score_ocr's
cer on the raw strings), the first differing character, the difflib ratio on whitespace-collapsed text, and the
parity-bar verdict (minicpmocr_work/parity_bar.py's rule, rebased): reference = HF 1-D on the same picture; a
difference counts only outside the spans where that reference itself differs from HF native (what one fixed-size
picture cannot resolve) -> equal / pass-span / FAIL.
Instrument + verdict table; the gate's own verdict is score_xocr.py's line against HF native.

  python compare_hf.py ROWS.jsonl ... --hf 'hf/runs_*.jsonl' [--md OUT.md] [--json OUT.json]
"""
import argparse
import difflib
import glob
import json
import os
import re


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


def cer(a, b):
  return lev(a, b) / max(len(b), 1)


def norm(t):
  return re.sub(r"\s+", " ", t.strip())


def spans(a, b):
  return [(i1, i2) for tag, i1, i2, _, _ in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
          if tag != "equal"]


def inside(span, allowed, pad=1):
  s, e = span
  return any(s >= x - pad and e <= y + pad for x, y in allowed)


def bar(text, ref, native):
  if text == ref:
    return "equal", []
  allowed = spans(ref, native)
  bad = [sp for sp in spans(ref, text) if not inside(sp, allowed)]
  return ("pass-span" if not bad else "FAIL"), bad


def first_diff(a, b):
  for i, (x, y) in enumerate(zip(a, b)):
    if x != y:
      return i
  return None if len(a) == len(b) else min(len(a), len(b))


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("rows", nargs="+")
  ap.add_argument("--hf", required=True, help="glob of hf_xocr.py rows (quote it)")
  ap.add_argument("--md", default=None)
  ap.add_argument("--json", default=None)
  a = ap.parse_args()
  hf = {}
  for p in sorted(glob.glob(a.hf)):
    for line in open(p):
      if line.strip():
        r = json.loads(line)
        hf.setdefault((r["name"], r["mode"], bool(r.get("pos1d"))), r)
  out, lines = [], ["| rows | case | mode | = HF 1-D | = HF full | CER vs 1-D / full | first diff (1-D) | ratio 1-D / full | "
                    "bar (ref HF 1-D, spans vs native) | chars bundle / 1-D / full | stop |", "|---|" * 1 + "---|" * 10]
  counts = {}
  for path in a.rows:
    for line in open(path):
      if not line.strip():
        continue
      r = json.loads(line)
      key1, keyf, keyn = (r["name"], r["mode"], True), (r["name"], r["mode"], False), (r["name"], "native", False)
      t = (r.get("text") or "").strip()
      h1, hf_, hn = hf.get(key1), hf.get(keyf), hf.get(keyn)
      rec = {"rows": os.path.basename(path), "name": r["name"], "mode": r["mode"], "stop": r.get("stop"),
             "n_gen": r.get("n_gen")}
      for tag, h in (("1d", h1), ("full", hf_)):
        if h is None:
          rec[tag] = None
          continue
        w = h["text"].strip()
        rec[tag] = {"equal": t == w, "cer": round(cer(t, w), 4), "first_diff": first_diff(t, w),
                    "ratio": round(difflib.SequenceMatcher(None, norm(t), norm(w)).ratio(), 4), "chars": len(w),
                    "hf_stop": h.get("stop"), "hf_n_gen": h.get("n_gen")}
      if h1 is not None and hn is not None:
        v, bad = bar(t, h1["text"].strip(), hn["text"].strip())
        rec["bar"] = v
        rec["bar_outside"] = [h1["text"].strip()[max(0, s - 12):e + 12] for s, e in bad[:3]]
      else:
        rec["bar"] = "no 1-D ref"
      out.append(rec)
      c = counts.setdefault(rec["rows"], {"rows": 0, "eq1d": 0, "eqfull": 0, "bar_fail": 0, "no1d": 0})
      c["rows"] += 1
      c["eq1d"] += bool(rec["1d"] and rec["1d"]["equal"])
      c["eqfull"] += bool(rec["full"] and rec["full"]["equal"])
      c["bar_fail"] += rec["bar"] == "FAIL"
      c["no1d"] += rec["1d"] is None

      def f(tag, k):
        return "—" if rec.get(tag) is None else rec[tag][k]
      lines.append(f"| {rec['rows']} | {r['name']} | {r['mode']} | {'yes' if f('1d', 'equal') is True else f('1d', 'equal') if f('1d', 'equal') == '—' else 'no'} | "
                   f"{'yes' if f('full', 'equal') is True else f('full', 'equal') if f('full', 'equal') == '—' else 'no'} | "
                   f"{f('1d', 'cer')} / {f('full', 'cer')} | {f('1d', 'first_diff')} | {f('1d', 'ratio')} / {f('full', 'ratio')} | "
                   f"{rec['bar']}{' ' + repr(rec.get('bar_outside')) if rec.get('bar') == 'FAIL' else ''} | "
                   f"{len(t)} / {f('1d', 'chars')} / {f('full', 'chars')} | {r.get('stop')} |")
  lines.append("")
  for k, c in counts.items():
    lines.append(f"{k}: rows {c['rows']}, equal to HF 1-D {c['eq1d']}, equal to HF full {c['eqfull']}, "
                 f"bar FAIL {c['bar_fail']}, no 1-D ref {c['no1d']}")
  text = "\n".join(lines)
  print(text)
  if a.md:
    open(a.md, "w").write(text + "\n")
  if a.json:
    json.dump(out, open(a.json, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
  main()
