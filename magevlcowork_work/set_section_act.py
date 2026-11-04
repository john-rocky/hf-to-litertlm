#!/usr/bin/env python3
"""set_section_act.py SRC.litertlm DST.litertlm --section prefill_decode --type none|fp32|fp16 [--litert-lm CLI]

Change the prefer_activation_type of ONE TFLiteModel section of a .litertlm and leave every other section's declaration
as it is. scripts/set_activation_type.py (read, not imported) writes one type into every TFLiteModel section and cannot
remove a declaration; here `--type none` removes the item from that section only, so the runtime falls back to its
default (v0.19.0 runtime/engine/engine_settings.cc MaybeOverrideActivationType: no prefer_activation_type -> the system
default, text executor F16, vision F32).

Same CLI traps as set_activation_type.py: `unpack` writes a declared item as an entry of the section's
`additional_metadata` list; `pack` exits 0 on "Error packing model" and does not overwrite an existing output. The
packed file is unpacked again and read back: the run fails (and removes the output) unless the named section reads
back as asked, every other section's header items are unchanged, every section's data is the bytes that were packed
(sha256), and the system metadata differs only in uuid / creation_timestamp. Needs Python 3.11+ (tomllib) and the
litert-lm CLI (default: litert-lm on PATH).
"""
import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib

ITEM = "prefer_activation_type"
VOLATILE = {"uuid", "creation_timestamp"}
ENTRY_LINE = re.compile(r'^\s*\{\s*key\s*=\s*"' + ITEM + r'"\s*,\s*value_type\s*=\s*"String"\s*,\s*value\s*=\s*"([^"]*)"\s*\},?\s*\n',
                        re.M)


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
    secs.append({"type": s.get("section_type"), "items": items, "sha256": sha256(p) if p else None,
                 "bytes": os.path.getsize(p) if p else None})
  return system, secs


def edit(toml, section, act):
  toml = re.sub(r'model_type = "TF_LITE_([A-Z_]+)"', lambda m: f'model_type = "{m.group(1).lower()}"', toml)
  parts = toml.split("[[section]]")
  hit = [i for i in range(1, len(parts)) if re.search(r'^section_type = "TFLiteModel"', parts[i], re.M)
         and re.search(r'^model_type = "' + re.escape(section) + r'"', parts[i], re.M)]
  if len(hit) != 1:
    raise SystemExit(f"want one TFLiteModel section {section}, found {len(hit)}")
  i = hit[0]
  b = parts[i]
  old = ENTRY_LINE.findall(b)
  if len(old) > 1 or re.search(r'^' + ITEM + r'\s*=', b, re.M):
    raise SystemExit(f"section {section}: {ITEM} is not in the one-entry additional_metadata form; edit by hand")
  b = ENTRY_LINE.sub("", b)
  b = re.sub(r'^additional_metadata = \[\s*\]\n', "", b, flags=re.M)  # the list emptied by the removal
  if act != "none":
    if re.search(r'^additional_metadata = \[\n', b, re.M):
      b = re.sub(r'^(additional_metadata = \[\n)', r'\1  { key = "' + ITEM + r'", value_type = "String", value = "' + act
                 + r'" },\n', b, count=1, flags=re.M)
    else:
      b = b.replace('section_type = "TFLiteModel"', f'section_type = "TFLiteModel"\n{ITEM} = "{act}"', 1)
  parts[i] = b
  return "[[section]]".join(parts), (old[0] if old else None)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("src")
  ap.add_argument("dst")
  ap.add_argument("--section", default="prefill_decode")
  ap.add_argument("--type", required=True, choices=["none", "fp32", "fp16"])
  ap.add_argument("--litert-lm", default="litert-lm")
  ap.add_argument("--work", help="work dir (default: a temp dir next to dst)")
  a = ap.parse_args()
  work = a.work or tempfile.mkdtemp(prefix="set_section_act_", dir=os.path.dirname(os.path.abspath(a.dst)))
  os.makedirs(work, exist_ok=True)
  try:
    unpack = os.path.join(work, "unpack")
    run([a.litert_lm, "unpack", a.src, "--output-dir", unpack])
    src_system, src_secs = read(unpack)
    tp = os.path.join(unpack, "model.toml")
    toml, old = edit(open(tp).read(), a.section, a.type)
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
      target = s["type"] == "TFLiteModel" and str(s["items"].get("model_type", "")).lower().replace("tf_lite_", "") == a.section
      want = dict(s["items"])
      if target:
        want.pop(ITEM, None)
        if a.type != "none":
          want[ITEM] = a.type
      if g["items"] != want:
        errors.append(f"section {k} ({s['type']} {s['items'].get('model_type')}) items {g['items']} != {want}")
      if g["sha256"] != s["sha256"]:
        errors.append(f"section {k} data sha256 {s['sha256']} -> {g['sha256']}")
      rows.append({"i": k, "type": s["type"], "model_type": s["items"].get("model_type"),
                   "act_src": s["items"].get(ITEM), "act_dst": g["items"].get(ITEM), "bytes": g["bytes"],
                   "sha256_same": g["sha256"] == s["sha256"], "sha256": g["sha256"]})
    for r in rows:
      print(f"SECTION {r['i']} {r['type']} {r['model_type']}: {ITEM} {r['act_src']} -> {r['act_dst']}, bytes "
            f"{r['bytes']}, data sha256 same {r['sha256_same']} ({(r['sha256'] or '')[:16]}...)")
    if errors:
      os.remove(a.dst)
      raise SystemExit("FAILED, removed the output:\n  " + "\n  ".join(errors))
    print(f"OK {a.dst} {os.path.getsize(a.dst)} B sha256 {sha256(a.dst)}: section {a.section} {ITEM} {old or 'none'} -> "
          f"{a.type}; every other section and every section's data unchanged (read back from the packed file)")
  finally:
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
  main()
