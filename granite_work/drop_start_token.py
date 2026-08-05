#!/usr/bin/env python3
"""Remove the LlmMetadata start_token from a .litertlm bundle (weights untouched).

Why: the LiteRT-LM engine prepends the metadata start_token to every rendered
prompt. Granite's official chat template has NO leading BOS, and at 350M scale
the mismatch measurably degrades greedy decoding (the model can jump to
answering only the last question of a multi-question prompt, or end replies
early). Dropping the start_token makes the on-device rendering match the HF
reference exactly. The 1b is robust either way; for the 350m this is required.

Usage: python drop_start_token.py in.litertlm out.litertlm
Needs the `litert-lm` CLI (pip install litert-lm) on PATH.
"""
import os, shutil, subprocess, sys, tempfile

LITERT_LM = shutil.which("litert-lm") or sys.exit("litert-lm CLI not on PATH (pip install litert-lm)")
src, dst = sys.argv[1], sys.argv[2]

work = tempfile.mkdtemp(prefix="dropbos_")
unpack = os.path.join(work, "u")
subprocess.run([LITERT_LM, "unpack", src, "--output-dir", unpack], check=True)

meta = os.path.join(unpack, "LlmMetadataProto.pbtext")
s = open(meta).read()
if not s.startswith("start_token {"):
    sys.exit("no leading start_token block found — nothing to do")
open(meta, "w").write(s.split("}\n", 1)[1])

if os.path.exists(dst):
    os.remove(dst)  # `litert-lm pack` silently no-ops when the output exists
subprocess.run([LITERT_LM, "pack", unpack, "--output", dst], check=True)
shutil.rmtree(work)
print("OK", dst, os.path.getsize(dst))
