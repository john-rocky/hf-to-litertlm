#!/usr/bin/env python3
"""Post-export sanity check for a .litertlm: is the file the size its recipe implies?

Run it on every fresh bundle. It catches the failure that has now happened twice, both
times silently:

**The vocab table gets duplicated when a tied-embedding model is asked for two different
quantizations of one buffer.** `BOCTAV4` puts int4 on every FULLY_CONNECTED and int8 on
EMBEDDING_LOOKUP; on a tied model those are the *same* tensor, so ai-edge-quantizer
resolves the conflict by copying it — once per signature. Nothing errors. You get a file
that is 2-3x the size it should be and slower to decode (decode is bandwidth-bound), and
"int4" in the filename still looks right.

  granite-4.1-3b : 2.79 GB -> 1.01 GB once fixed (12 copies of a 154 MB table)
  Qwen2.5-Coder-1.5B : 2.53 GB, of which 1.63 GB was 7 copies of a 233 MB table

    python scripts/check_bundle_sanity.py <model.litertlm> [--params 1543714304]

Exit 0 = clean, 1 = duplication found, 2 = harness error. With --params it also reports
bytes-per-parameter, which is the number that makes a wrong recipe obvious at a glance
(int4 ~0.5-0.7, int8 ~1.1-1.3 including the embedding table).
"""
import argparse
import os
import subprocess
import sys


def grep_count(path, needle):
    """Count a literal string in the flatbuffer. Tensor names are plain strings in there,
    so this needs no parsing and takes seconds even on multi-GB files."""
    r = subprocess.run(["grep", "-c", "-a", needle, path],
                       capture_output=True, text=True, env={**os.environ, "LC_ALL": "C"})
    try:
        return int((r.stdout or "0").strip().split("\n")[0])
    except ValueError:
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("--params", type=int, default=None,
                    help="parameter count of the source model, for a bytes/param read")
    args = ap.parse_args()

    if not os.path.isfile(args.bundle):
        print(f"MISSING: {args.bundle}")
        return 2

    size = os.path.getsize(args.bundle)
    dup = grep_count(args.bundle, "_duplicated_")
    # Positive control: a zero from the line above means nothing unless the file really
    # does contain tensor-name strings.
    control = grep_count(args.bundle, "arith.constant")

    print(f"{args.bundle}")
    print(f"  size            {size:,} bytes ({size / 1e9:.2f} GB)")
    if args.params:
        print(f"  bytes/param     {size / args.params:.2f}  "
              f"(int4 ~0.5-0.7, int8 ~1.1-1.3, incl. embedding)")
    print(f"  _duplicated_    {dup}   (positive control 'arith.constant': {control})")

    if control == 0:
        print("  INCONCLUSIVE — no tensor-name strings found at all; the zero above means "
              "nothing. Check the file is really a .litertlm.")
        return 2
    if dup:
        print("  ⚠ DUPLICATED TENSORS. On a tied-embedding model this is the vocab table "
              "copied per signature. Re-export with EXTERNALIZE_EMBEDDER=1, or align the "
              "recipe so the embedder and lm_head ask for the same quantization "
              "(quantize_minicpm5.py's LM_HEAD_REGEX does the latter).")
        return 1
    print("  clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
