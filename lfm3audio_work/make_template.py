#!/usr/bin/env python3
"""make_template.py SRC OUT - the chat template of the LFM3-Audio-2B `.litertlm` file (stdlib only).

SRC is the source repository's chat_template.jinja (LiquidAI/LFM3-Audio-2B at 9d49701c, sha256 8769fc2a...). OUT is
SRC with three edits:
  - keep_thinking = preserve_thinking (the source: preserve_thinking or loop.index0 > ns.last_user_index): every earlier
    assistant turn renders the same way wherever it sits. LiteRT-LM keeps a conversation's earlier prompt as a prefix of
    the next one, so a turn must render the same after a later user message. A conversation that ends with a user
    message renders as the source template renders it.
  - the {%- generation -%} / {%- endgeneration -%} tags are removed (LiteRT-LM 0.18.1 refuses the statement), and
  - X.get("k") is written as X["k"] (LiteRT-LM's template engine has no dict .get; a missing key reads as undefined,
    which the template treats like .get's None).
The last two keep the rendered text. LiteRT-LM rewrites .split(x)[-1], .endswith(x) and .items() itself when it loads
the template.
"""
import hashlib
import pathlib
import re
import sys

SRC_SHA256 = "8769fc2ac03684665a009b1908fee359e418329019dd83dbc57152167e448099"
KEEP_OLD = "{%- set keep_thinking = preserve_thinking or loop.index0 > ns.last_user_index -%}"
KEEP_NEW = "{%- set keep_thinking = preserve_thinking -%}"
N_GET = 13


def sha(text):
  return hashlib.sha256(text.encode()).hexdigest()


def string_preserving(t):
  n_gen = t.count("{%- generation -%}") + t.count("{%- endgeneration -%}")
  assert n_gen == 2, n_gen
  t = t.replace("{%- generation -%}\n", "").replace("{%- endgeneration -%}\n", "")
  assert "generation -%}" not in t
  t, n_get = re.subn(r'\.get\(("[^"]*")\)', r"[\1]", t)
  assert n_get == N_GET, n_get
  assert ".get(" not in t
  return t, {"generation_tags": n_gen, "get_calls": n_get}


def main():
  src_path, out_path = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
  src = src_path.read_text()
  assert sha(src) == SRC_SHA256, f"{src_path}: sha256 {sha(src)}, expected {SRC_SHA256}"
  edited, counts = string_preserving(src)
  assert edited.count(KEEP_OLD) == 1
  out = edited.replace(KEEP_OLD, KEEP_NEW)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  out_path.write_text(out)
  print(f"{out_path}: {len(out)} chars sha256 {sha(out)}; edits {counts}, keep_thinking 1")


if __name__ == "__main__":
  main()
