#!/usr/bin/env python3
"""set_max_tokens.py SRC.litertlm DST.litertlm --max-num-tokens N [--litert-lm CLI] [--work DIR]

Rewrite the LlmMetadata section's max_num_tokens. For Mage-VL-Cowork: 4,096 -> 3,584, the KV length the decoder holds
after cache_variant_comp.py, so the runtime no longer clamps at engine creation (`Passed in max_num_tokens (4096) is
larger than what the static model supports (3584). Using model limit.`). Only section 0 changes. The CLI path and the
checks of set_section_act.py (this directory):
  `litert-lm unpack` -> edit the unpacked LlmMetadata text (exactly one top-level `max_num_tokens:` line)
  -> `litert-lm pack` -> unpack the packed file again and read back:
  the LlmMetadata text differs from the source's in that one line only, every other section's data is the source's
  bytes (sha256), every section's header items are unchanged, the system metadata differs only in uuid /
  creation_timestamp. Any other difference fails the run and removes the output.
Same CLI traps as set_section_act.py: `unpack` writes model_type = "TF_LITE_X" that `pack` reads as "x"; `pack` exits 0
on "Error packing model" and does not overwrite an existing output. Needs Python 3.11+ (tomllib) and the litert-lm CLI
(default: litert-lm on PATH).
"""
import argparse
import difflib
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib

VOLATILE = {"uuid", "creation_timestamp"}
LINE = re.compile(r"^max_num_tokens:\s*(\d+)\s*$", re.M)


def sha256(path):
  h = hashlib.sha256()
  with open(path, "rb") as f:
    for c in iter(lambda: f.read(1 << 24), b""):
      h.update(c)
  return h.hexdigest()


def run(cmd):
  p = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True)
  if p.returncode != 0 or "Error" in (p.stdout + p.stderr):
    sys.stdout.write(p.stdout[-3000:])
    sys.stderr.write(p.stderr[-3000:])
    raise SystemExit(f"FAILED: {' '.join(cmd[:2])} rc {p.returncode}")
  return p


def read(unpack_dir):
  with open(os.path.join(unpack_dir, "model.toml"), "rb") as f:
    doc = tomllib.load(f)
  system = {e["key"]: e["value"] for e in doc.get("system_metadata", {}).get("entries", [])}
  secs = []
  for s in doc.get("section", []):
    items = {k: v for k, v in s.items() if k not in ("section_type", "data_path", "additional_metadata")}
    for e in s.get("additional_metadata", []) or []:
      items[e["key"]] = e["value"]
    p = os.path.join(unpack_dir, s["data_path"]) if "data_path" in s else None
    secs.append({"type": s.get("section_type"), "items": items, "path": p, "sha256": sha256(p) if p else None,
                 "bytes": os.path.getsize(p) if p else None})
  return system, secs


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("src")
  ap.add_argument("dst")
  ap.add_argument("--max-num-tokens", type=int, required=True)
  ap.add_argument("--litert-lm", default="litert-lm")
  ap.add_argument("--work", help="work dir (default: a temp dir next to dst)")
  a = ap.parse_args()
  work = a.work or tempfile.mkdtemp(prefix="set_max_tokens_", dir=os.path.dirname(os.path.abspath(a.dst)))
  os.makedirs(work, exist_ok=True)
  try:
    unpack = os.path.join(work, "unpack")
    run([a.litert_lm, "unpack", a.src, "--output-dir", unpack])
    src_system, src_secs = read(unpack)
    meta = [s for s in src_secs if s["type"] == "LlmMetadata"]
    if len(meta) != 1:
      raise SystemExit(f"want one LlmMetadata section, found {len(meta)}")
    src_text = open(meta[0]["path"]).read()
    hits = LINE.findall(src_text)
    if len(hits) != 1:
      raise SystemExit(f"want one top-level max_num_tokens line in {meta[0]['path']}, found {hits}")
    old = int(hits[0])
    new_text = LINE.sub(f"max_num_tokens: {a.max_num_tokens}", src_text, count=1)
    open(meta[0]["path"], "w").write(new_text)
    tp = os.path.join(unpack, "model.toml")
    toml = open(tp).read()
    toml = re.sub(r'model_type = "TF_LITE_([A-Z_]+)"', lambda m: f'model_type = "{m.group(1).lower()}"', toml)
    open(tp, "w").write(toml)
    if os.path.exists(a.dst):
      os.remove(a.dst)
    p = run([a.litert_lm, "pack", tp, "--output", a.dst])
    if not os.path.exists(a.dst) or os.path.getsize(a.dst) == 0:
      raise SystemExit(f"FAILED: pack wrote nothing: {p.stdout[-500:]} {p.stderr[-500:]}")
    shutil.rmtree(unpack)
    back = os.path.join(work, "back")
    run([a.litert_lm, "unpack", a.dst, "--output-dir", back])
    got_system, got_secs = read(back)
    errors = []
    drop = lambda d: {k: v for k, v in d.items() if k not in VOLATILE}
    if drop(got_system) != drop(src_system):
      errors.append(f"system metadata {drop(src_system)} -> {drop(got_system)}")
    if [s["type"] for s in got_secs] != [s["type"] for s in src_secs]:
      errors.append("section types changed")
    rows = []
    for k, (s, g) in enumerate(zip(src_secs, got_secs)):
      if g["items"] != s["items"]:
        errors.append(f"section {k} ({s['type']}) items {s['items']} -> {g['items']}")
      if s["type"] == "LlmMetadata":
        got_text = open(g["path"]).read()
        diff = [d for d in difflib.unified_diff(src_text.splitlines(), got_text.splitlines(), lineterm="", n=0)
                if d[:1] in "+-" and not d.startswith(("+++", "---"))]
        want = [f"-max_num_tokens: {old}", f"+max_num_tokens: {a.max_num_tokens}"]
        if diff != want:
          errors.append(f"LlmMetadata text diff {diff[:6]} != {want}")
        same = None
      else:
        same = g["sha256"] == s["sha256"]
        if not same:
          errors.append(f"section {k} ({s['type']} {s['items'].get('model_type')}) data sha256 {s['sha256']} -> "
                        f"{g['sha256']}")
      rows.append({"i": k, "type": s["type"], "model_type": s["items"].get("model_type"), "bytes": g["bytes"],
                   "sha256_src": s["sha256"], "sha256_dst": g["sha256"], "same": same})
    for r in rows:
      tail = (f"data sha256 same {r['same']} ({(r['sha256_dst'] or '')[:16]}...)" if r["same"] is not None else
              f"text: max_num_tokens {old} -> {a.max_num_tokens} only ({(r['sha256_src'] or '')[:16]}... -> "
              f"{(r['sha256_dst'] or '')[:16]}...)")
      print(f"SECTION {r['i']} {r['type']} {r['model_type']}: bytes {r['bytes']}, {tail}")
    if errors:
      os.remove(a.dst)
      raise SystemExit("FAILED, removed the output:\n  " + "\n  ".join(errors))
    print(f"OK {a.dst} {os.path.getsize(a.dst)} B sha256 {sha256(a.dst)}: LlmMetadata max_num_tokens {old} -> "
          f"{a.max_num_tokens}; the other sections' data and every section's header items unchanged (read back from the "
          f"packed file)")
  finally:
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
  main()
