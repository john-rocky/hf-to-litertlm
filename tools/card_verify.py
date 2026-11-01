#!/usr/bin/env python3
"""Check every number of a model card draft against the raw run records, with code that shares nothing with the
scorer, the card writer or the table writer.

  python tools/card_verify.py --card README.draft.md --checks <checks.py> --sources <sources.json> \
      [--opt phones] [--out verify.json]

A checks file is plain Python that defines
  SOURCES = {"name": "what the path holds", ...}   the records it reads, by logical name
  def run(v): ...                                 re-derives each value from those records and calls
                                                  v.check(name, want, src)   want = the exact text the card must hold
                                                  v.record(name, found, want, src)   a check that is not one substring
--sources is a JSON object {name: path} for every name in SOURCES (relative paths are read from the JSON file's
directory). --opt NAME turns on optional parts of a checks file (v.opts), e.g. the phone sessions.
The verifier gives the checks file its readers, so the file needs nothing but the standard library:
  v.text(name) / v.json(name) / v.jsonl(name) / v.path(name)   a source's text, JSON, JSON lines, path
  v.median, v.commas, v.sha256(path)                          arithmetic and hashes
  v.bench(stderr_text)        LiteRT-LM BenchmarkInfo of one run's stderr: prefill / decode turns (tokens, tok/s),
                              vision_s, ttft_s, init_s, peak_mb ("Peak system ram usage"), nfd ("not fully delegated")
  v.thought_answer(stdout)    "[thought]...[/thought] answer" of litert_lm_main's stdout (INFO lines dropped)
  v.point_2d(text)            the first "point_2d": [x, y] of an answer
  v.mask_hit(png, x, y)       a 0-1000 point against a 1000 x 1000 truth mask (grey > 127 at [int(y), int(x)])
Exit 0 = every check found, 1 = a check failed, 2 = the card still has an open {{SLOT}}, 3 = a source is missing.
Write the checks before reading the draft's numbers; a check that fails is a number to fix in the card, or a record
to read again, never a check to loosen.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import re
import statistics
import sys


class Verifier:
  def __init__(self, card_text, sources, opts):
    self.card = card_text
    self._src = sources
    self.opts = set(opts)
    self.checks = []
    self._masks = {}

  # recording
  def check(self, name, want, src):
    self.checks.append({"check": name, "want": want, "found": want in self.card, "src": str(src)})

  def record(self, name, found, want, src):
    self.checks.append({"check": name, "want": want, "found": bool(found), "src": str(src)})

  # sources
  def path(self, name):
    if name not in self._src:
      raise KeyError(f"source {name!r} is not in --sources")
    return self._src[name]

  def text(self, name):
    return open(self.path(name), encoding="utf-8", errors="replace").read()

  def json(self, name):
    return json.load(open(self.path(name)))

  def jsonl(self, name):
    return [json.loads(line) for line in open(self.path(name)) if line.strip()]

  # helpers
  median = staticmethod(statistics.median)

  @staticmethod
  def commas(n):
    return f"{n:,}"

  @staticmethod
  def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
      for b in iter(lambda: f.read(1 << 24), b""):
        h.update(b)
    return h.hexdigest()

  @staticmethod
  def dur_s(s):
    m = re.fullmatch(r"([\d.]+)(ms|us|s)", s)
    v = float(m.group(1))
    return v / 1000 if m.group(2) == "ms" else v / 1e6 if m.group(2) == "us" else v

  @classmethod
  def bench(cls, e):
    b = e[e.find("BenchmarkInfo:"):] if "BenchmarkInfo:" in e else ""
    pre = re.findall(r"Prefill Turn \d+: Processed (\d+) tokens in [\d.]+(?:ms|us|s) duration\.\s+Prefill Speed: ([\d.]+)", b)
    dec = re.findall(r"Decode Turn \d+: Processed (\d+) tokens in [\d.]+(?:ms|us|s) duration\.\s+Decode Speed: ([\d.]+)", b)
    vis = re.findall(r"- vision_executor: ([\d.]+(?:ms|us|s))", b)
    tt = re.findall(r"Time to first token: ([\d.]+) s", b)
    it = re.findall(r"- Init Total: ([\d.]+) ms", b)
    pk = re.findall(r"Peak system ram usage: ([\d.]+)\s*MB", e)
    return {"prefill": [(int(n), float(s)) for n, s in pre], "decode": [(int(n), float(s)) for n, s in dec],
            "vision_s": cls.dur_s(vis[-1]) if vis else None, "ttft_s": float(tt[-1]) if tt else None,
            "init_s": float(it[-1]) / 1000 if it else None, "peak_mb": float(pk[-1]) if pk else None,
            "nfd": len(re.findall(r"not fully delegated", e))}

  @staticmethod
  def thought_answer(o):
    t = "\n".join(line for line in o.splitlines() if not line.startswith("INFO: ")).strip()
    m = re.match(r"\[thought\](.*?)\[/thought\]\s*(.*)\Z", t, re.S)
    return (m.group(1) if m else None), (m.group(2) if m else t).strip()

  @staticmethod
  def point_2d(ans):
    m = re.search(r'"point_2d"\s*:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]', ans or "")
    return (float(m.group(1)), float(m.group(2))) if m else None

  def mask_hit(self, png, x, y):
    from PIL import Image
    if png not in self._masks:
      self._masks[png] = Image.open(png).convert("L")
    m = self._masks[png]
    w, h = m.size
    return m.getpixel((min(w - 1, max(0, int(x))), min(h - 1, max(0, int(y))))) > 127


def load_checks(path):
  spec = importlib.util.spec_from_file_location("card_checks_" + os.path.splitext(os.path.basename(path))[0], path)
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--card", required=True)
  ap.add_argument("--checks", required=True)
  ap.add_argument("--sources", required=True)
  ap.add_argument("--opt", action="append", default=[])
  ap.add_argument("--out", default=None)
  a = ap.parse_args()
  mod = load_checks(a.checks)
  base = os.path.dirname(os.path.abspath(a.sources))
  given = json.load(open(a.sources))
  sources = {k: v if os.path.isabs(v) else os.path.normpath(os.path.join(base, v)) for k, v in given.items()}
  missing = [k for k in getattr(mod, "SOURCES", {}) if k not in sources or not os.path.exists(sources[k])]
  if missing:
    for k in missing:
      print(f"MISSING source {k}: {mod.SOURCES[k]} ({sources.get(k, 'not given')})")
    return 3
  card = open(a.card, encoding="utf-8").read()
  slots = sorted(set(re.findall(r"\{\{([A-Z0-9_]+)\}\}", card)))
  v = Verifier(card, sources, a.opt)
  mod.run(v)
  bad = [c for c in v.checks if not c["found"]]
  for c in v.checks:
    print(("ok   " if c["found"] else "FAIL ") + c["check"] + ("" if c["found"] else f"  want: {c['want']}"))
  print(f"{len(v.checks) - len(bad)}/{len(v.checks)} checks found; open slots {len(slots)}")
  if a.out:
    json.dump({"card": os.path.abspath(a.card), "card_sha256": hashlib.sha256(card.encode()).hexdigest(),
               "checks_file": os.path.abspath(a.checks), "sources": sources, "opts": sorted(v.opts),
               "open_slots": slots, "checks": v.checks}, open(a.out, "w"), indent=1, ensure_ascii=False)
  return 1 if bad else (2 if slots else 0)


if __name__ == "__main__":
  sys.exit(main())
