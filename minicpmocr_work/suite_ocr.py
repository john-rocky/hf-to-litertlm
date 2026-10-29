#!/usr/bin/env python3
"""OCR suite for the MiniCPM-OCR-1.0 fast_vlm bundle: every fixture through `litert-lm run --attachment`, one fresh
process per case (greedy: the engine's default top-k 1), rows to JSONL in the shape of hf_oracle.py, then a comparison
with the upstream rows of the same input (hf_oracle.py, same mode = the same GxG picture).

  python suite_ocr.py run BUNDLE --label L --backend cpu|gpu [--vision-backend cpu|gpu] --mode s448 \
      --fixtures FIX.json[:DIR] ... --out RUNS.jsonl [--only a,b] [--litert-lm RUN]
  python suite_ocr.py compare --runs RUNS.jsonl --hf HF.jsonl [--md OUT.md]

stdin is /dev/null for every process (the CLI otherwise waits on a terminal stdin). The attachment goes before the prompt text,
as in the upstream runs ([image, text] in one user turn).
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_ocr import cer, lev, norm_text  # noqa: E402,F401


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def load_items(specs):
    items = []
    for spec in specs:
        path, _, d = spec.partition(":")
        d = d or os.path.dirname(path)
        for it in json.load(open(path))["items"]:
            it["_dir"] = d
            items.append(it)
    return items


def cmd_run(a):
    bundle = os.path.abspath(a.bundle)
    bsha = sha256(bundle)
    ver = subprocess.run([a.litert_lm, "--version"], capture_output=True, text=True,
                         stdin=subprocess.DEVNULL).stdout.strip()
    logdir = os.path.join(os.path.dirname(os.path.abspath(a.out)), "logs", a.label)
    os.makedirs(logdir, exist_ok=True)
    done = set()
    if os.path.exists(a.out):
        done = {(json.loads(l)["name"], json.loads(l)["mode"]) for l in open(a.out)}
    only = set(a.only.split(",")) if a.only else None
    for it in load_items(a.fixtures):
        if (only and it["name"] not in only) or a.mode not in it["files"] or (it["name"], a.mode) in done:
            continue
        f = it["files"][a.mode]
        img = os.path.join(it["_dir"], f["file"])
        assert sha256(img) == f["sha256"], img
        cmd = [a.litert_lm, "run", bundle, "--prompt", it["prompt"], "--attachment", img, "--backend", a.backend,
               "--cache", "no"]
        if a.vision_backend:
            cmd += ["--vision-backend", a.vision_backend]
        t0 = time.time()
        pr = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout, stdin=subprocess.DEVNULL)
        sec = round(time.time() - t0, 1)
        open(os.path.join(logdir, f"{it['name']}_{a.mode}.stderr.log"), "w").write(pr.stderr)
        text = pr.stdout.strip("\n")
        row = {"name": it["name"], "mode": a.mode, "label": a.label, "backend": a.backend,
               "vision_backend": a.vision_backend, "runtime": ver, "bundle": bundle, "bundle_sha256": bsha,
               "file": f["file"], "sha256": f["sha256"], "rc": pr.returncode, "seconds": sec,
               "text": text, "text_raw": text, "cmd": " ".join(cmd)}
        with open(a.out, "a") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{it['name']:14s} {a.mode} rc {pr.returncode} {sec:6.1f}s | {text[:80]!r}", flush=True)


def cmd_compare(a):
    hf = {}
    for line in open(a.hf):
        r = json.loads(line)
        hf[(r["name"], r["mode"])] = r
    lines = ["| fixture | mode | label | equal to upstream | CER vs upstream | first diff (char) | chars |",
             "|---|---|---|---|---|---|---|"]
    n = eq = 0
    for line in open(a.runs):
        r = json.loads(line)
        h = hf.get((r["name"], r["mode"]))
        if h is None:
            continue
        g, w = r["text"].strip(), h["text"].strip()
        first = next((i for i, (x, y) in enumerate(zip(g, w)) if x != y), None)
        if first is None and len(g) != len(w):
            first = min(len(g), len(w))
        same = g == w
        n += 1
        eq += same
        lines.append(f"| {r['name']} | {r['mode']} | {r['label']} | {'yes' if same else 'no'} | "
                     f"{cer(g, w):.4f} | {'' if first is None else first} | {len(g)} / {len(w)} |")
    lines.append(f"\nequal to upstream: {eq} / {n}")
    text = "\n".join(lines)
    print(text)
    if a.md:
        open(a.md, "w").write(text + "\n")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("bundle")
    r.add_argument("--label", required=True)
    r.add_argument("--backend", default="cpu")
    r.add_argument("--vision-backend", default=None)
    r.add_argument("--mode", default="s448")
    r.add_argument("--fixtures", nargs="+", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--only", default=None)
    r.add_argument("--timeout", type=int, default=1800)
    r.add_argument("--litert-lm", default=os.environ.get("LITERT_LM", "litert-lm"))
    c = sub.add_parser("compare")
    c.add_argument("--runs", required=True)
    c.add_argument("--hf", required=True)
    c.add_argument("--md", default=None)
    a = ap.parse_args()
    return cmd_run(a) if a.cmd == "run" else cmd_compare(a)


if __name__ == "__main__":
    main()
