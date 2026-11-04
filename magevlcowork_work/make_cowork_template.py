#!/usr/bin/env python3
"""Write the bundle's chat template from the vendor's chat_template.jinja by listed, counted string edits.

Each edit is an exact (old, new, count) replacement on the vendor text; the run stops when an edit does not hit
exactly `count` times, so the list below is the whole difference between the two files. Why each edit exists:

  R1 image marker   the fast_vlm runtime splits the rendered prompt on the literal <image_soft_token> and injects the
                    vision embeddings there (fastvlm_data_processor.cc); the vendor's <|image_pad|> is expanded by the
                    HF processor instead. <|vision_start|> / <|vision_end|> stay (they are real tokens the model saw).
  R2 messages[0]    the runtime renders a preface with no messages (Conversation::Create with tools / extra context,
                    RenderPrefaceIntoString); an unguarded messages[0] is `undefined value` there. Two places: the
                    tools branch and the no-tools branch.
  R3 last_query     the reversed loop that finds the last real user query (`ns.last_query_index`) is removed: its
                    only use was the position-dependent assistant branch (R5), and it calls .startswith / .endswith,
                    which the runtime's minijinja (no pycompat) does not have.
  R4 split          `.split(x)[0]` / `.split(x)[-1]` / `.rstrip('\n')` / `.lstrip('\n')` -> the filters
                    `|split(x)|first` / `|split(x)|last` (minijinja builtins) and `|rstrip('\n')` / `|lstrip('\n')`
                    (filters the runtime registers, minijinja_template.rs v0.19.0). Same strings out.
  R5 history form   an assistant turn renders with its think block when it has reasoning, without one when it has
                    none, wherever it sits in the conversation. The vendor keeps reasoning only on turns after the
                    last real user query and adds an empty think block to a final assistant turn; both depend on later
                    messages, which breaks the runtime's rule that the render of the history is a prefix of the
                    render of the history plus the next message. In the agent loop the vendor's own OSWorld code
                    builds (every later user turn is a <tool_response>), the vendor renders every assistant turn with
                    its reasoning, which is this form; `.strip('\n')` / `.lstrip('\n')` become `|trim('\n')` /
                    `|lstrip('\n')`.

    python3 magevlcowork_work/make_cowork_template.py <checkpoint>/chat_template.jinja OUT.jinja [--doc REWRITES.md]

From microsoft/Mage-VL-Cowork's chat_template.jinja (5,365 B, sha256 e004c3c2...) this writes 4,623 B, sha256
5f8cdbd6d0c939f0328b98e7dbbf1f9a5181b41220bc4bf6fc8bd3ca52b23bd9 (the template of the shipped bundle). --doc also
writes the table of the edits as Markdown.
"""
import argparse
import hashlib
import os
import re

R3_OLD = """{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
    {%- set index = (messages|length - 1) - loop.index0 %}
    {%- if ns.multi_step_tool and message.role == "user" %}
        {%- set content = render_content(message.content, false) %}
        {%- if not(content.startswith('<tool_response>') and content.endswith('</tool_response>')) %}
            {%- set ns.multi_step_tool = false %}
            {%- set ns.last_query_index = index %}
        {%- endif %}
    {%- endif %}
{%- endfor %}
"""

R5_OLD = """        {%- if loop.index0 > ns.last_query_index %}
            {%- if loop.last or (not loop.last and reasoning_content) %}
                {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}
            {%- else %}
                {{- '<|im_start|>' + message.role + '\\n' + content }}
            {%- endif %}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\\n' + content }}
        {%- endif %}
"""
R5_NEW = """        {%- if reasoning_content %}
            {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content|trim('\\n') + '\\n</think>\\n\\n' + content|lstrip('\\n') }}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\\n' + content }}
        {%- endif %}
"""

EDITS = [
    ("R1", "<|vision_start|><|image_pad|><|vision_end|>", "<|vision_start|><image_soft_token><|vision_end|>", 1),
    ("R2", "{%- if messages[0].role == 'system' %}", "{%- if messages and messages[0].role == 'system' %}", 2),
    ("R3", R3_OLD, "", 1),
    ("R4a", "{%- set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}",
     "{%- set reasoning_content = content|split('</think>')|first|rstrip('\\n')|split('<think>')|last|lstrip('\\n') %}", 1),
    ("R4b", "{%- set content = content.split('</think>')[-1].lstrip('\\n') %}",
     "{%- set content = content|split('</think>')|last|lstrip('\\n') %}", 1),
    ("R5", R5_OLD, R5_NEW, 1),
]

PY_METHODS = re.compile(r"\.(get|split|strip|lstrip|rstrip|startswith|endswith|items|keys|values|replace|join)\(")


def main():
  ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  ap.add_argument("src", help="the checkpoint's chat_template.jinja")
  ap.add_argument("out", help="the bundle template to write")
  ap.add_argument("--doc", default=None, help="also write the table of the edits (Markdown)")
  a = ap.parse_args()
  src = open(a.src, encoding="utf-8").read()
  out = src
  rows = []
  for tag, old, new, count in EDITS:
    n = out.count(old)
    assert n == count, f"{tag}: expected {count} hit(s), found {n}"
    out = out.replace(old, new)
    rows.append((tag, count, old, new))
  left = sorted(set(m.group(0) for m in PY_METHODS.finditer(out)))
  assert not left, f"python methods left: {left}"
  assert "last_query_index" not in out and "loop.index0 > " not in out
  os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
  open(a.out, "w", encoding="utf-8").write(out)
  sha = lambda s: hashlib.sha256(s.encode()).hexdigest()
  if a.doc:
    with open(a.doc, "w", encoding="utf-8") as f:
      f.write("# The bundle template = the vendor's chat_template.jinja + these edits\n\n")
      f.write(f"- vendor: `{os.path.basename(a.src)}` {len(src.encode())} B, sha256 `{sha(src)}`\n")
      f.write(f"- bundle: `{os.path.basename(a.out)}` {len(out.encode())} B, sha256 `{sha(out)}`\n")
      f.write("- written by `magevlcowork_work/make_cowork_template.py` (each edit must hit exactly the listed count;"
              " the reasons are in its docstring); the result has no `.get(` / `.split(` / `.strip(` / `.lstrip(` /"
              " `.rstrip(` / `.startswith(` / `.endswith(` / `.items(` / `.replace(` / `.join(` call.\n\n")
      f.write("| edit | hits | vendor | bundle |\n|---|---|---|---|\n")
      for tag, count, old, new in rows:
        cell = lambda s: "`" + s.replace("|", "\\|").replace("\n", "⏎") + "`" if s else "(removed)"
        f.write(f"| {tag} | {count} | {cell(old)} | {cell(new)} |\n")
  print(f"WROTE {a.out} ({len(out.encode())} B, sha256 {sha(out)}) from vendor sha256 {sha(src)}; edits {len(rows)}")


if __name__ == "__main__":
  main()
