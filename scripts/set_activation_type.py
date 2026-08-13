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

Needs the `litert-lm` CLI (>= 0.15, for `unpack`/`pack`) on PATH.
"""
import argparse
import os
import re
import shutil
import subprocess
import tempfile


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("src")
  ap.add_argument("dst")
  ap.add_argument("--type", default="fp32", help="fp32 or fp16 (default fp32)")
  ap.add_argument("--litert-lm", default="litert-lm")
  args = ap.parse_args()

  with tempfile.TemporaryDirectory() as td:
    unpack = os.path.join(td, "unpack")
    subprocess.run(
        [args.litert_lm, "unpack", args.src, "--output-dir", unpack], check=True
    )
    toml_path = os.path.join(unpack, "model.toml")
    toml = open(toml_path).read()

    marker = 'section_type = "TFLiteModel"'
    if marker not in toml:
      raise SystemExit("model.toml has no TFLiteModel section")
    if "prefer_activation_type" in toml:
      toml = re.sub(
          r'prefer_activation_type = "[^"]*"',
          f'prefer_activation_type = "{args.type}"',
          toml,
      )
    else:
      toml = toml.replace(
          marker, marker + f'\nprefer_activation_type = "{args.type}"'
      )
    open(toml_path, "w").write(toml)

    # `litert-lm pack` exits 0 without writing when the output already exists.
    if os.path.exists(args.dst):
      os.remove(args.dst)
    subprocess.run([args.litert_lm, "pack", toml_path, "--output", args.dst],
                   check=True)
  print(f"OK: {args.dst} (prefer_activation_type = {args.type})")


if __name__ == "__main__":
  main()
