#!/usr/bin/env python3
"""Declare an activation dtype for the TFLiteModel section of a .litertlm.

The GPU executor runs activations in fp16 by default. For graphs whose
intermediates do not fit that range — the SSM/linear-attention scans are the
ones we keep hitting — the engine's own sampler reports "Invalid decode and
sample result" and every token comes out as id 0. Declaring

    prefer_activation_type = "fp32"

in the bundle's `model.toml` makes the executor keep activations in fp32.
Weights are untouched, so this is a repack, not a re-export.

    python scripts/set_activation_type.py in.litertlm out.litertlm [--type fp32]

Needs the `litert-lm` CLI (>= 0.15, for `unpack`/`pack`) on PATH, and Python 3.11+
(tomllib, for the read-back).

Three CLI traps this script absorbs (the first two measured with litert-lm 0.17.1,
2026-09-28; the third with 0.17.0 and 0.19.0, 2026-11-01):
  * `litert-lm unpack` of an older bundle writes `model_type = "TF_LITE_PREFILL_DECODE"`,
    which the same version's `pack` rejects ("'tf_lite_tf_lite_prefill_decode' is not a
    valid TfLiteModelType") -- the name is normalized to the tf-free form here.
  * `litert-lm pack` prints "Error packing model: ..." and still exits 0, so the output
    file is checked for existence instead of trusting the exit code.
  * `litert-lm unpack` of a bundle that already declares the item writes it as an entry
    of the section's `additional_metadata` list, not as a `prefer_activation_type = ...`
    key; editing only the key form leaves the old value in place and `pack` writes it
    back. Both forms are edited here, and the packed file is unpacked again and read
    back: the run fails (and removes the output) unless every TFLiteModel section now
    declares the requested type, no other header item changed, and every section's data
    is what was packed.
"""
import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile

try:
  import tomllib
except ImportError:  # Python < 3.11
  tomllib = None

ITEM = "prefer_activation_type"
# Written by `pack` itself, so they differ between any two packs of the same file.
VOLATILE_SYSTEM_KEYS = {"uuid", "creation_timestamp"}
ENTRY_RE = re.compile(
    r'(\{\s*key\s*=\s*"' + ITEM + r'"\s*,\s*value_type\s*=\s*"([^"]*)"\s*,\s*value\s*=\s*)"([^"]*)"'
)
KEY_RE = re.compile(r'^(' + ITEM + r'\s*=\s*)"([^"]*)"', re.M)


def edit_toml(toml, act_type):
  """Returns (new_toml, [(section index, old value or None)]) for every TFLiteModel section."""
  # 0.17.1 unpack emits the legacy TF_LITE_* spelling that its own pack refuses.
  toml = re.sub(
      r'model_type = "TF_LITE_([A-Z_]+)"',
      lambda m: f'model_type = "{m.group(1).lower()}"',
      toml,
  )
  parts = toml.split("[[section]]")
  changes = []
  for i in range(1, len(parts)):
    block = parts[i]
    if not re.search(r'^section_type = "TFLiteModel"', block, re.M):
      continue
    entries = ENTRY_RE.findall(block)
    keys = KEY_RE.findall(block)
    if len(entries) + len(keys) > 1:
      raise SystemExit(f"section {i - 1}: {ITEM} is declared more than once; edit model.toml by hand")
    if entries:
      _, value_type, old = entries[0]
      if value_type != "String":
        raise SystemExit(f"section {i - 1}: {ITEM} has value_type {value_type!r}, expected 'String'")
      block = ENTRY_RE.sub(lambda m: m.group(1) + f'"{act_type}"', block)
    elif keys:
      old = keys[0][1]
      block = KEY_RE.sub(lambda m: m.group(1) + f'"{act_type}"', block)
    else:
      old = None
      block = block.replace(
          'section_type = "TFLiteModel"',
          f'section_type = "TFLiteModel"\n{ITEM} = "{act_type}"', 1,
      )
    parts[i] = block
    changes.append((i - 1, old))
  if not changes:
    raise SystemExit("model.toml has no TFLiteModel section")
  return "[[section]]".join(parts), changes


def sha256(path):
  h = hashlib.sha256()
  with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 24), b""):
      h.update(chunk)
  return h.hexdigest()


def read_unpacked(unpack_dir):
  """Parses an unpacked model.toml: (system items, [section dicts]) with each
  section's header items merged from its keys and its additional_metadata list."""
  with open(os.path.join(unpack_dir, "model.toml"), "rb") as f:
    doc = tomllib.load(f)
  system = {e["key"]: e["value"] for e in doc.get("system_metadata", {}).get("entries", [])}
  sections = []
  for s in doc.get("section", []):
    items = {k: v for k, v in s.items() if k not in ("section_type", "data_path", "additional_metadata")}
    for e in s.get("additional_metadata", []):
      items[e["key"]] = e["value"]
    sections.append({"type": s.get("section_type"), "items": items,
                     "data": os.path.join(unpack_dir, s["data_path"]) if "data_path" in s else None})
  return system, sections


def run_cli(cmd):
  p = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True)
  if p.returncode != 0:
    sys.stdout.write(p.stdout)
    sys.stderr.write(p.stderr)
    raise SystemExit(f"FAILED: {' '.join(cmd[:2])} exited {p.returncode}")
  return p


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("src")
  ap.add_argument("dst")
  ap.add_argument("--type", default="fp32", choices=["fp32", "fp16"], help="fp32 or fp16 (default fp32)")
  ap.add_argument("--litert-lm", default="litert-lm")
  args = ap.parse_args()
  if tomllib is None:
    raise SystemExit("needs Python 3.11+ (tomllib) to read the packed file back")

  with tempfile.TemporaryDirectory() as td:
    unpack = os.path.join(td, "unpack")
    run_cli([args.litert_lm, "unpack", args.src, "--output-dir", unpack])
    toml_path = os.path.join(unpack, "model.toml")
    toml = open(toml_path).read()
    toml, changes = edit_toml(toml, args.type)
    open(toml_path, "w").write(toml)
    # What the packed file must read back as: the edited model.toml, every data file as unpacked.
    want_system, want_sections = read_unpacked(unpack)
    for s in want_sections:
      s["sha256"] = sha256(s["data"]) if s["data"] else None

    # `litert-lm pack` exits 0 without writing when the output already exists.
    if os.path.exists(args.dst):
      os.remove(args.dst)
    p = run_cli([args.litert_lm, "pack", toml_path, "--output", args.dst])
    # `litert-lm pack` exits 0 on "Error packing model"; the file is the proof.
    if not os.path.exists(args.dst) or os.path.getsize(args.dst) == 0:
      sys.stdout.write(p.stdout)
      sys.stderr.write(p.stderr)
      raise SystemExit(f"FAILED: {args.dst} was not written (see pack output above)")
    shutil.rmtree(unpack)

    # Read the packed file back with the same CLI.
    back = os.path.join(td, "readback")
    run_cli([args.litert_lm, "unpack", args.dst, "--output-dir", back])
    got_system, got_sections = read_unpacked(back)
    errors = []
    drop = lambda d: {k: v for k, v in d.items() if k not in VOLATILE_SYSTEM_KEYS}
    if drop(got_system) != drop(want_system):
      errors.append(f"system metadata: packed {drop(got_system)} != edited {drop(want_system)}")
    if [s["type"] for s in got_sections] != [s["type"] for s in want_sections]:
      errors.append(f"section types: packed {[s['type'] for s in got_sections]} != "
                    f"edited {[s['type'] for s in want_sections]}")
    else:
      for i, (g, w) in enumerate(zip(got_sections, want_sections)):
        if g["items"] != w["items"]:
          errors.append(f"section {i} ({w['type']}) items: packed {g['items']} != edited {w['items']}")
        if (sha256(g["data"]) if g["data"] else None) != w["sha256"]:
          errors.append(f"section {i} ({w['type']}) data differs from what was packed")
      for i, _ in changes:
        if got_sections[i]["items"].get(ITEM) != args.type:
          errors.append(f"section {i}: {ITEM} reads back {got_sections[i]['items'].get(ITEM)!r}, not {args.type!r}")
    if errors:
      os.remove(args.dst)
      raise SystemExit("FAILED: the packed file does not read back as edited, removed it:\n  " + "\n  ".join(errors))
  moved = ", ".join(f"section {i} {old or 'none'} -> {args.type}" for i, old in changes)
  print(f"OK: {args.dst} ({ITEM}: {moved}; read back from the packed file)")


if __name__ == "__main__":
  main()
