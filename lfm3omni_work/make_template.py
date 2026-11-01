#!/usr/bin/env python3
"""make_template.py SOURCE.jinja OUT.jinja - the chat template of LFM3-Omni-2B's .litertlm file: the source repository's
chat_template.jinja (sha256 ea74cc1d…) with three edits.

  1. Earlier assistant turns render without their thought wherever they sit (`keep_thinking = preserve_thinking`; the
     source keeps the thought of the turns after the last user message too). LiteRT-LM keeps a conversation's earlier
     prompt as a prefix of the next one, so a turn must render the same after a later user message. A conversation that
     ends with a user message renders as the source renders it.
  2. The two `{%- generation -%}` / `{%- endgeneration -%}` tags are removed (LiteRT-LM's template engine refuses them).
  3. The `.get("…")` lookups are written as `["…"]` (the engine's dicts have no `.get`; a missing key reads as
     undefined, as `.get` reads None).
Edits 2 and 3 keep the rendered text. Unlike LFM3-1.7B's template, this one defaults `enable_thinking` to false and has
the audio content parts.
"""
import hashlib
import pathlib
import re
import sys

SRC_SHA256 = "ea74cc1d14319ed436dc0f37632893004d436913113e7605267f2b01694d65e4"
KEEP_OLD = "{%- set keep_thinking = preserve_thinking or loop.index0 > ns.last_user_index -%}"
KEEP_NEW = "{%- set keep_thinking = preserve_thinking -%}"


def sha(text):
  return hashlib.sha256(text.encode()).hexdigest()


def main():
  if len(sys.argv) != 3:
    print(__doc__, file=sys.stderr)
    return 2
  src = pathlib.Path(sys.argv[1]).read_text()
  assert sha(src) == SRC_SHA256, f"{sys.argv[1]}: sha256 {sha(src)}, expected {SRC_SHA256}"
  n_gen = src.count("{%- generation -%}") + src.count("{%- endgeneration -%}")
  assert n_gen == 2, n_gen
  t = src.replace("{%- generation -%}\n", "").replace("{%- endgeneration -%}\n", "")
  assert "generation -%}" not in t
  t, n_get = re.subn(r'\.get\(("[^"]*")\)', r"[\1]", t)
  assert ".get(" not in t, "a .get( call the rule does not cover"
  assert t.count(KEEP_OLD) == 1
  t = t.replace(KEEP_OLD, KEEP_NEW)
  pathlib.Path(sys.argv[2]).write_text(t)
  print(f"{sys.argv[2]}: {len(t)} chars sha256 {sha(t)} (generation tags {n_gen}, .get calls {n_get}, keep_thinking 1)",
        flush=True)
  return 0


if __name__ == "__main__":
  sys.exit(main())
