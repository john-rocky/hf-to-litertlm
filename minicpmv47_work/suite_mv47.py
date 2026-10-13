#!/usr/bin/env python3
"""Image suite for the MiniCPM-V-4.7-1B fast_vlm bundle (copied from qwen35vl_work/suite_gate_vl.py
and adapted; the source file is untouched): 3 fixtures x 3 prompts through
`litert-lm run --attachment`, one fresh process per case, answers to JSON, then each answer scored
against the HF oracles of vl_oracle.py (arm P = the fast_vlm contract, arm F = HF's canvas
M-RoPE on the same pixels): token-prefix first divergence and exact match on the SRC tokenizer.

The bar is GROUNDED (the answer names the fixture's content), not greedy identity: this family
paraphrases under the 1-D position contract (qwen35vl_work/FINDINGS.md). `kw` is a keyword aid per
(fixture, prompt); the verdict in GATE.md is the reading of every answer.

    python3 suite_mv47.py run <bundle> --label cpu --backend cpu [--vision-backend gpu]
        [--cases cats:0,...] [--fixtures W/fixtures --suffix _448.png] [--litert-lm RUN]
    python3 suite_mv47.py score r3/suite_<label>.json   (rescore against the current oracle files)
  (oracles are read from and results written under this directory: r3/oracle/, r3/logs/, r3/suite_*.json;
   RUN = $LITERT_LM or litert-lm on PATH, 0.18.0 for the published gate)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
K = HERE   # the scripts sit in minicpmv47_work/ itself (the private layout had them one level down)
SRC = Path.home() / (".cache/huggingface/hub/models--openbmb--MiniCPM-V-4.7-1B/snapshots/"
                     "984a89aafcd5160164f4f0d04b4be0f4eec03cb8")
RUN = os.environ.get("LITERT_LM", "litert-lm")
NAMES = ["cats", "kitchen1", "kitchen2"]
PROMPTS = ["What is in this image?",
           "Describe the main colors in this image.",
           "Where is this scene?"]
KITCHEN_OBJ = r"kitchen|pots?\b|pans?\b|stove|oven|cabinet|counter|cookware|refrigerator|fridge|utensil|\bcook|chef"
KW = {  # (fixture kind, prompt index) -> (regex, min hits)
    ("cats", 0): (r"\bcats?\b|kitten|tabby", 1),
    ("cats", 1): (r"pink|magenta|fuchsia|\bred\b", 1),
    ("cats", 2): (r"couch|sofa|\bbed\b|home|house|living room|indoor|blanket|room", 1),
    ("kitchen", 0): (KITCHEN_OBJ, 1),
    ("kitchen", 1): (r"brown|white|wood|copper|orange|yellow|beige|black|cream|\btan\b|gold|silver|gray|grey", 2),
    ("kitchen", 2): (r"kitchen", 1),
}


def kw_check(name, k, text):
  rx, need = KW[("cats" if name == "cats" else "kitchen", k)]
  hits = sorted(set(m.group(0).lower() for m in re.finditer(rx, text, re.I)))
  return {"kw_hits": hits, "kw_ok": len(hits) >= need}


def load_oracles():
  out = {}
  for arm in ("P", "F", "N"):
    p = K / "r3/oracle" / f"hf_vl_{arm}.json"
    if p.exists():
      d = json.loads(p.read_text())
      out[arm] = {(c["name"], c["prompt_index"]): c for c in d["cases"]}
  return out


def scorer():
  from tokenizers import Tokenizer
  tk = Tokenizer.from_file(str(SRC / "tokenizer.json"))

  def ids(t):
    return tk.encode(t.strip(), add_special_tokens=False).ids

  def score(got, want):
    g, w = ids(got), ids(want)
    L = min(len(g), len(w))
    first_div = next((i for i in range(L) if g[i] != w[i]), L)
    return {"first_div": first_div, "n_ref": len(w), "n_got": len(g),
            "exact": g == w, "prefix_exact": first_div == L}
  return score


def rescore(res):
  oracles = load_oracles()
  score = scorer()
  summary = {}
  for row in res["cases"]:
    row.update(kw_check(row["name"], row["prompt_index"], row["text"]))
    for arm, m in oracles.items():
      c = m.get((row["name"], row["prompt_index"]))
      if c is not None:
        row[f"vs_{arm}"] = score(row["text"], c["answer"])
  for arm in oracles:
    rows = [r for r in res["cases"] if f"vs_{arm}" in r]
    summary[arm] = {"n": len(rows), "exact": sum(r[f"vs_{arm}"]["exact"] for r in rows),
                    "first_div": [r[f"vs_{arm}"]["first_div"] for r in rows]}
  res["kw_ok"] = sum(r["kw_ok"] for r in res["cases"])
  res["score_summary"] = summary
  res["oracle_cases"] = {arm: len(m) for arm, m in oracles.items()}
  return res


def cmd_run(a):
  bundle = os.path.abspath(a.bundle)
  sha = hashlib.sha256()
  with open(bundle, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 24), b""):
      sha.update(chunk)
  cases = [(n, k) for n in NAMES for k in range(len(PROMPTS))]
  if a.cases:
    want = {(s.split(":")[0], int(s.split(":")[1])) for s in a.cases.split(",")}
    cases = [c for c in cases if c in want]
  logdir = K / "r3/logs" / f"suite_{a.label}"
  logdir.mkdir(parents=True, exist_ok=True)
  ver = subprocess.run([a.litert_lm, "--version"], capture_output=True, text=True,
                       stdin=subprocess.DEVNULL).stdout.strip()
  res = {"label": a.label, "bundle": bundle, "bundle_bytes": os.path.getsize(bundle),
         "bundle_sha256": sha.hexdigest(), "backend": a.backend, "vision_backend": a.vision_backend,
         "runtime": ver, "fixtures": os.path.abspath(a.fixtures), "suffix": a.suffix,
         "started": time.strftime("%Y-%m-%d %H:%M:%S"), "cases": []}
  for n, k in cases:
    img = os.path.join(a.fixtures, f"{n}{a.suffix}")
    cmd = [a.litert_lm, "run", bundle, "--prompt", PROMPTS[k], "--attachment", img,
           "--backend", a.backend, "--cache", "no", "--temperature", "0", "--seed", "0"]
    if a.vision_backend:
      cmd += ["--vision-backend", a.vision_backend]
    if a.verbose:
      cmd += ["--verbose"]
    t0 = time.time()
    pr = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, stdin=subprocess.DEVNULL)
    sec = round(time.time() - t0, 1)
    (logdir / f"{n}_{k}.stderr.log").write_text(pr.stderr)
    text = pr.stdout.strip()
    row = {"name": n, "prompt_index": k, "image": os.path.basename(img),
           "image_sha256": hashlib.sha256(open(img, "rb").read()).hexdigest(),
           "prompt": PROMPTS[k], "cmd": " ".join(cmd), "rc": pr.returncode, "sec": sec,
           "chars": len(text), "text": text}
    res["cases"].append(row)
    print(f"[{n} | {PROMPTS[k]}] rc={pr.returncode} {sec} s chars={len(text)}\n{text[:300]}\n", flush=True)
    out = a.out or str(K / "r3" / f"suite_{a.label}.json")
    Path(out).write_text(json.dumps(rescore(res), indent=1, ensure_ascii=False) + "\n")
  res["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
  out = a.out or str(K / "r3" / f"suite_{a.label}.json")
  Path(out).write_text(json.dumps(rescore(res), indent=1, ensure_ascii=False) + "\n")
  print("SUITE_DONE", out, json.dumps(res["score_summary"]), "kw_ok", res["kw_ok"], "/", len(res["cases"]))
  return 0


def cmd_score(a):
  p = Path(a.suite)
  res = rescore(json.loads(p.read_text()))
  p.write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
  for r in res["cases"]:
    arms = " ".join(f"{arm}:fd={r[f'vs_{arm}']['first_div']}/{r[f'vs_{arm}']['n_ref']}"
                    + ("=" if r[f"vs_{arm}"]["exact"] else "") for arm in ("P", "F", "N") if f"vs_{arm}" in r)
    print(f"{r['name']}:{r['prompt_index']} kw={r['kw_ok']} {arms}")
  print("SCORED", p, json.dumps(res["score_summary"]), "kw_ok", res["kw_ok"], "/", len(res["cases"]))
  return 0


def main():
  ap = argparse.ArgumentParser()
  sub = ap.add_subparsers(dest="cmd", required=True)
  r = sub.add_parser("run")
  r.add_argument("bundle")
  r.add_argument("--label", required=True)
  r.add_argument("--backend", default="cpu")
  r.add_argument("--vision-backend", default=None)
  r.add_argument("--cases", default=None, help="comma list of <name>:<prompt index>")
  r.add_argument("--fixtures", default=str(K / "fixtures"))
  r.add_argument("--suffix", default="_448.png")
  r.add_argument("--litert-lm", default=str(RUN))
  r.add_argument("--verbose", action="store_true")
  r.add_argument("--out", default=None)
  s = sub.add_parser("score")
  s.add_argument("suite")
  a = ap.parse_args()
  return cmd_run(a) if a.cmd == "run" else cmd_score(a)


if __name__ == "__main__":
  raise SystemExit(main())
