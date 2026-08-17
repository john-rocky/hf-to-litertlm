#!/usr/bin/env python3
"""Remove the `start_token` from a .litertlm's LlmMetadata (weights untouched).

Why this exists — measured on granite-4.1-3b, 2026-08-17:

litert-torch's metadata builder sets `start_token` from `tokenizer.bos_token`
UNCONDITIONALLY (`core/litert_lm_builder.py`, `if hasattr(tokenizer, 'bos_token')
and tokenizer.bos_token`). It never consults `add_bos_token`. For a tokenizer that
declares `add_bos_token: False` the runtime then prepends a token the model was
never fed at that position — and when that token is ALSO the EOS (granite:
bos == eos == `<|end_of_text|>` == 100257), the model reads the prompt as a
finished document and degenerates: it echoes the question back, or emits a run of
backticks, instead of answering.

This is not quantization damage, and the check that proves it costs one minute:
feed the SAME rendered prompt to the bf16 PyTorch model with and without the
leading BOS. On granite-4.1-3b, bf16 answers "There are 7 days in a week." without
it and "Answer briefly." with it. The .litertlm reproduced the with-BOS behaviour
exactly (8Q 5/8, two questions echoed).

    python scripts/strip_start_token.py in.litertlm out.litertlm

Needs the `litert-lm` CLI (>= 0.15, for `unpack`/`pack`) — pass --litert-lm to pick
the build that matches the bundle's builder version.
"""
import argparse
import os
import re
import subprocess


def strip_block(pbtext, field="start_token"):
  """Drop a top-level `field { ... }` block from a text-format proto.

  Brace-counting rather than a regex: the block contains nested braces, and the
  metadata's stop_tokens use the same shape, so a non-greedy regex would eat the
  wrong span.
  """
  m = re.search(rf"^{field}\s*\{{", pbtext, re.M)
  if not m:
    return pbtext, False
  i = pbtext.index("{", m.start())
  depth = 0
  for j in range(i, len(pbtext)):
    if pbtext[j] == "{":
      depth += 1
    elif pbtext[j] == "}":
      depth -= 1
      if depth == 0:
        end = j + 1
        while end < len(pbtext) and pbtext[end] == "\n":
          end += 1
        return pbtext[:m.start()] + pbtext[end:], True
  raise SystemExit(f"unbalanced braces in {field} block")


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("src")
  ap.add_argument("dst")
  ap.add_argument("--field", default="start_token")
  ap.add_argument("--litert-lm", default=os.path.expanduser("~/venvs/lt0160run/bin/litert-lm"))
  ap.add_argument("--work-dir", default=None,
                  help="where to unpack (default: a temp dir next to dst; the "
                       "sections are as big as the model, so keep it on a disk "
                       "with room)")
  args = ap.parse_args()

  work = args.work_dir or os.path.join(os.path.dirname(os.path.abspath(args.dst)),
                                       ".strip_start_token_tmp")
  os.makedirs(work, exist_ok=True)
  unpack = os.path.join(work, "unpack")
  subprocess.run([args.litert_lm, "unpack", args.src, "--output-dir", unpack],
                 check=True)

  pb = os.path.join(unpack, "LlmMetadataProto.pbtext")
  text = open(pb).read()
  new, found = strip_block(text, args.field)
  if not found:
    raise SystemExit(f"no `{args.field}` block in {pb} — nothing to strip")
  open(pb, "w").write(new)

  toml_path = os.path.join(unpack, "model.toml")
  if os.path.exists(args.dst):  # `litert-lm pack` exits 0 without writing otherwise
    os.remove(args.dst)
  subprocess.run([args.litert_lm, "pack", toml_path, "--output", args.dst],
                 check=True)
  print(f"OK: {args.dst} ({args.field} removed)")


if __name__ == "__main__":
  main()
