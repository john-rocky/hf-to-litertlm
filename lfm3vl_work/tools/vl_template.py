#!/usr/bin/env python3
"""The LFM3-VL bundle's chat template = the source chat_template.jinja (@ f83d2b0a) with four edits, each counted:

  1. image parts render <|vision_start|><image_soft_token><|vision_end|> instead of <|vision_start|><|image|><|vision_end|>
     (LiteRT-LM splits the rendered prompt on <image_soft_token> and splices the adapter's (G/32)^2 embeddings there,
     fastvlm_data_processor.cc L48-66; the source processor expands <|image|> to the same count). Video parts stay.
  2. {%- generation -%} / {%- endgeneration -%} removed: the runtime's rewrite strips only the dash-free form
     (prompt_template.cc L70-72) and minijinja then stops at "unknown statement generation". The tags mark training
     spans; they render nothing.
  3. X.get("k") -> X["k"] (26 places): the runtime's minijinja has no pycompat (minijinja_template.rs create_env), so a
     map has no .get method; a missing key reads undefined either way, the same as .get's None for every test the
     template applies (is string / is mapping / or "" / if).
  4. history assistant turns rendered the same way wherever they sit: keep_thinking = preserve_thinking (source:
     preserve_thinking or loop.index0 > ns.last_user_index). LiteRT-LM keeps a conversation's earlier prompt as a
     prefix of the next one, so a turn must render the same after a later user message. A conversation that ends with
     a user message renders exactly as the source template does: every assistant turn there sits before the last user
     message.
The leading {{- bos_token -}} stays: the runtime renders bos_token = "" and adds the start token itself (session_utils.cc).

  python3 tools/vl_template.py SRC/chat_template.jinja OUT.jinja     (stdlib only)
"""
import re
import sys

SOFT = "<image_soft_token>"


def build(src_text):
  t = src_text
  img_old = '"<|vision_start|><|image|><|vision_end|>"'
  img_new = f'"<|vision_start|>{SOFT}<|vision_end|>"'
  n_img = t.count(img_old)
  assert n_img == 2, n_img
  t = t.replace(img_old, img_new)
  n_gen = t.count("{%- generation -%}") + t.count("{%- endgeneration -%}")
  assert n_gen == 2, n_gen
  t = t.replace("{%- generation -%}\n", "").replace("{%- endgeneration -%}\n", "")
  assert "generation -%}" not in t
  t, n_get = re.subn(r'\.get\(("[^"]*")\)', r"[\1]", t)
  assert n_get == 26, n_get
  keep_old = "{%- set keep_thinking = preserve_thinking or loop.index0 > ns.last_user_index -%}"
  assert t.count(keep_old) == 1
  t = t.replace(keep_old, "{%- set keep_thinking = preserve_thinking -%}")
  return t, {"image_parts": n_img, "generation_tags": n_gen, "get_calls": n_get, "keep_thinking": 1}


def main():
  if len(sys.argv) != 3:
    print(__doc__, file=sys.stderr)
    return 2
  src = open(sys.argv[1]).read()
  t, counts = build(src)
  with open(sys.argv[2], "w") as f:
    f.write(t)
  print("VL_TEMPLATE", sys.argv[2], len(src), "->", len(t), "chars", counts)
  return 0


if __name__ == "__main__":
  sys.exit(main())
