#!/usr/bin/env python3
"""make_template.py VENDOR_JINJA OUT_JINJA - the chat template the LFM3-1.7B .litertlm files carry.

OUT_JINJA = VENDOR_JINJA (LiquidAI/LFM3-1.7B `chat_template.jinja`) with:
  history  assistant turns rendered the same way wherever they sit: keep_thinking = preserve_thinking (the vendor's
           is preserve_thinking or loop.index0 > ns.last_user_index). LiteRT-LM keeps a conversation's earlier prompt
           as a prefix of the next one, so a turn must render the same after a later user message; a conversation that
           ends with a user message renders as the vendor's (every assistant turn there sits before the last user
           message); a turn after the last user message (a tool loop) renders <think></think> + content instead of its
           reasoning.
  and two edits that keep the rendered text the same, so that LiteRT-LM's template engine (minijinja without Python
  compatibility) can run the file as it is:
           {%- generation -%} / {%- endgeneration -%} removed (training-span marks that render nothing; the runtime
           strips only the dash-free form),
           X.get("k") -> X["k"] (a missing key reads undefined, which every test the template applies treats as
           .get's None; litert-torch's packager makes the same rewrite).
The runtime's own load-time rewrites handle the rest (.split(x)[-1], .endswith(x), .items()). The template renders
bos_token as an empty string on LiteRT-LM, which adds the file's start token once. stdlib only.
"""
import hashlib
import re
import sys

KEEP_OLD = "{%- set keep_thinking = preserve_thinking or loop.index0 > ns.last_user_index -%}"
KEEP_NEW = "{%- set keep_thinking = preserve_thinking -%}"


def main(src_path, out_path):
  src = open(src_path).read()
  n_gen = src.count("{%- generation -%}") + src.count("{%- endgeneration -%}")
  assert n_gen == 2, n_gen
  t = src.replace("{%- generation -%}\n", "").replace("{%- endgeneration -%}\n", "")
  assert "generation -%}" not in t
  t, n_get = re.subn(r'\.get\(("[^"]*")\)', r"[\1]", t)
  assert n_get == 26, n_get
  assert ".get(" not in t
  assert t.count(KEEP_OLD) == 1
  t = t.replace(KEEP_OLD, KEEP_NEW)
  with open(out_path, "w") as f:
    f.write(t)
  print(f"vendor {hashlib.sha256(src.encode()).hexdigest()}")
  print(f"{out_path}: {len(t)} chars sha256 {hashlib.sha256(t.encode()).hexdigest()}")
  print(f"edits: generation tags {n_gen}, .get calls {n_get}, keep_thinking 1")


if __name__ == "__main__":
  main(sys.argv[1], sys.argv[2])
