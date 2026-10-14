#!/usr/bin/env python3
"""Add tool-call / tool-result rendering to the chat template a bundle carries (metadata-only re-pack).

    python tools/chat_template_parts/add_tool_branch.py <rule> own.jinja out.jinja [--official chat_template.jinja]

own.jinja is the bundle's own template, as `litert-lm unpack in.litertlm --output-dir work --chat-template
own.jinja` writes it. Each rule is a fixed set of edits; an anchor that does not occur exactly as often as
expected stops the script, and nothing else in the template changes.

  lfm2       LFM2.5 text bundles (llm_model_type lfm2). Adds the tool rendering of LiteRT-LM's
             models/lfm2/chat_template.jinja (pass it as --official): the `tool_response` item branch of
             format_content and the macros format_tool_content and render_tool_calls, copied from it, and in
             the message loop a `tool` message turn and the tool calls of an assistant turn. Every line of
             the own template survives in order.
  granite    granite-4.0-h (llm_model_type qwen2p5): the content loop also renders `tool_response` items.
  falcon-h1  Falcon-H1 (qwen2p5): format_content also renders `tool_response` items, and the tool-call
             schema line shows the call in JSON (the runtime's JSON parser rejects single quotes).
  falcon3    Falcon3 (qwen2p5): `tojson(indent=2)` -> `tojson` (the runtime's tojson takes no argument)
             and format_content also renders `tool_response` items.

The `tool_response` rule of granite / falcon-h1 / falcon3 is the one of LiteRT-LM's models/qwen2_5 template
(mapping or sequence -> tojson, anything else -> string) with a string branch first, so that jinja2 renders
what the runtime's minijinja renders. Prints the sha256 of the input and the output; for the templates of
the files re-packed in 2026-10 it also checks the output against the template packed into them (REPRODUCE.md,
"Metadata re-packs (2026-10)"). Refuses to overwrite. Standard library only.
"""
import argparse
import hashlib
import sys
from pathlib import Path

# own template sha256 -> (rule, sha256 of the template packed into the re-packed files, files)
PACKED = {
    'e0ef38f9a7b0b552a1fabcf7a6dcc96773c1cc9efaac1b7ceba067132f61225d':
        ('lfm2', 'ba91c6c8856efb9060ead3c13e6deddd5190379a82248b6a7db9a76ddf5b677a',
         'LFM2.5-1.2B-Instruct / -JP / -Thinking'),
    'e33995e7488a650726125a57eaeefc0a667a8c812deb8af30102c7b81344ca38':
        ('lfm2', 'd97c44c1c72f4055c755ca3d080efe66e13d38719ee7b04d91f1c431926a4c94', 'LFM2.5-2.6B'),
    '9524df67b77a7b25a2dfee898f75b316a157eb9d855b51e32aeac79d7c8a83ce':
        ('granite', 'bd944db9dbbe2af7592eb3c762558412e1db01d28025dec6c806bd8b49e3d659',
         'granite-4.0-h-350m / -1b (= granite-4.0-h_tools.jinja)'),
    'afc0ce3cf7dbc7b5784f509ca74e5845c904db76263730c10dd2c41a1996fe4f':
        ('falcon-h1', '0d281b31d684a8f1159f68362be48cd4ce81a7cf12d1575851cbebd3b50f32f9', 'Falcon-H1-3B-Instruct'),
    '7f1f2d95745fa60efb904feeffd787e6d8114a265fd8469df51605e4f923159e':
        ('falcon3', 'cae52ee934736b1a88752e16190f73382d91d8c319ac1065b595a1e488af8fa7', 'Falcon3-3B-Instruct'),
}
# models/lfm2/chat_template.jinja at LiteRT-LM a484d21d5113886348fe4c56af61830e45e664a8
OFFICIAL_LFM2 = '153fa3c9775a98047ef55c37ba5bf003fa9103de284202de5e407121b6c34a37'

FC_OLD = """            {%- if item['type'] == 'text' -%}
                {{- item['text'] -}}
            {%- endif -%}
"""
FC_NEW = """            {%- if item['type'] == 'text' -%}
                {{- item['text'] -}}
            {%- elif item['type'] == 'tool_response' -%}
                {%- if item['response'] is string -%}
                    {{- item['response'] -}}
                {%- elif item['response'] is mapping or item['response'] is sequence -%}
                    {{- item['response'] | tojson -}}
                {%- else -%}
                    {{- item['response'] | string -}}
                {%- endif -%}
            {%- endif -%}
"""
GRANITE_OLD = """                    {%- set content.val = content.val + entry.text %}
                {%- endif %}
            {%- endfor %}
"""
GRANITE_NEW = """                    {%- set content.val = content.val + entry.text %}
                {%- elif entry.type == 'tool_response' %}
                    {%- if content.val != '' %}
                        {%- set content.val = content.val + '\\n' %}
                    {%- endif %}
                    {%- if entry.response is string %}
                        {%- set content.val = content.val + entry.response %}
                    {%- elif entry.response is mapping or entry.response is sequence %}
                        {%- set content.val = content.val + (entry.response | tojson) %}
                    {%- else %}
                        {%- set content.val = content.val + (entry.response | string) %}
                    {%- endif %}
                {%- endif %}
            {%- endfor %}
"""
H1_SCHEMA_OLD = "{'arguments': <args-dict>, 'name': <function-name>}"
H1_SCHEMA_NEW = '{\\"arguments\\": <args-dict>, \\"name\\": <function-name>}'

REPLACEMENTS = {
    'granite': [(GRANITE_OLD, GRANITE_NEW, 1)],
    'falcon-h1': [(FC_OLD, FC_NEW, 1), (H1_SCHEMA_OLD, H1_SCHEMA_NEW, 1)],
    'falcon3': [('tojson(indent=2)', 'tojson', 2), (FC_OLD, FC_NEW, 1)],
}


def sha256(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def replace(own, reps):
    text = own
    for old, new, count in reps:
        n = text.count(old)
        if n != count:
            raise SystemExit(f'expected {count} x {old[:40]!r}, found {n}')
        text = text.replace(old, new)
    return text


def block(text, start, end):
    """The lines of text from the line that starts with `start` (stripped) to the next line equal to `end`
    (stripped) at the same indentation as the start line, so that a nested `end` does not close it."""
    lines = text.split('\n')
    i = next(k for k, l in enumerate(lines) if l.strip().startswith(start))
    indent = len(lines[i]) - len(lines[i].lstrip())
    j = next(k for k in range(i + 1, len(lines))
             if lines[k].strip() == end and len(lines[k]) - len(lines[k].lstrip()) == indent)
    return '\n'.join(lines[i:j + 1])


def once(text, needle):
    n = text.count(needle)
    if n != 1:
        raise SystemExit(f'anchor occurs {n} times: {needle!r}')


def lfm2(own, off):
    # 1. the tool_response branch of the official format_content, re-indented to the own macro's item level
    off_branch = block(off, "{%- elif item['type'] == 'tool_response' -%}", '{%- endif -%}')
    branch_lines = off_branch.split('\n')[:-1]  # without the endif that closes the item if-chain
    branch = '\n'.join('    ' + l for l in branch_lines)
    anchor1 = "                {{- item['text'] -}}\n            {%- endif -%}"
    once(own, anchor1)
    new = own.replace(anchor1, "                {{- item['text'] -}}\n" + branch + '\n            {%- endif -%}')
    # 2. the two official macros, after the own format_content macro
    m_tool = block(off, '{%- macro format_tool_content(content) -%}', '{%- endmacro -%}')
    m_calls = block(off, '{%- macro render_tool_calls(tool_calls) -%}', '{%- endmacro -%}')
    anchor2 = '{%- endmacro -%}\n{{- bos_token -}}'
    once(new, anchor2)
    new = new.replace(anchor2, '{%- endmacro -%}\n' + m_tool + '\n' + m_calls + '\n{{- bos_token -}}')
    # 3. the message loop. Two own layouts: the 1.2B template opens each turn on its own line before the
    # content; the 2.6B template builds the whole turn in one expression at the end of the loop body.
    tool_branch = ('    {%- if message["role"] == "tool" -%}\n'
                   '        {%- set tool_content = [{"type": "text", "text": message["content"]}] if message["content"] is string else message["content"] -%}\n'
                   "        {{- '<|im_start|>tool\\n' + format_tool_content(tool_content) + '<|im_end|>\\n' -}}\n"
                   '    {%- else -%}\n')
    anchor3 = ('{%- for message in messages -%}\n    {{- "<|im_start|>" + message["role"] + "\\n" -}}\n'
               '    {%- set content = format_content(message["content"]) -%}')
    anchor3b = ('{%- for message in messages -%}\n    {%- set content = format_content(message["content"]) -%}\n'
                '    {%- if content is not string -%}\n        {%- set content = content | tojson -%}\n'
                '    {%- endif -%}\n'
                '    {{- "<|im_start|>" + message["role"] + "\\n" + content + "<|im_end|>\\n" -}}\n{%- endfor -%}')
    if new.count(anchor3) == 1:
        loop_head = ('{%- for message in messages -%}\n' + tool_branch +
                     '    {{- "<|im_start|>" + message["role"] + "\\n" -}}\n'
                     "        {%- if message['role'] == 'assistant' and message['tool_calls'] -%}\n"
                     "            {{- render_tool_calls(message['tool_calls']) -}}\n"
                     '        {%- endif -%}\n'
                     '    {%- set content = format_content(message["content"]) -%}')
        new = new.replace(anchor3, loop_head)
        anchor4 = '    {{- content + "<|im_end|>\\n" -}}\n{%- endfor -%}'
        once(new, anchor4)
        new = new.replace(anchor4, '    {{- content + "<|im_end|>\\n" -}}\n    {%- endif -%}\n{%- endfor -%}')
    elif new.count(anchor3b) == 1:
        loop = ('{%- for message in messages -%}\n' + tool_branch +
                '    {%- set content = format_content(message["content"]) -%}\n'
                '    {%- if content is not string -%}\n        {%- set content = content | tojson -%}\n'
                '    {%- endif -%}\n'
                "        {%- if message['role'] == 'assistant' and message['tool_calls'] -%}\n"
                "            {%- set content = render_tool_calls(message['tool_calls']) + content -%}\n"
                '        {%- endif -%}\n'
                '    {{- "<|im_start|>" + message["role"] + "\\n" + content + "<|im_end|>\\n" -}}\n'
                '    {%- endif -%}\n{%- endfor -%}')
        new = new.replace(anchor3b, loop)
    else:
        raise SystemExit('neither LFM2.5 message-loop layout found')
    # every own line is still there, in order
    it = iter(new.split('\n'))
    missing = [l for l in own.split('\n') if not any(l == m for m in it)]
    if missing:
        raise SystemExit(f'own lines lost: {missing[:3]}')
    # every added line of 1-2 comes from the official template
    for l in branch_lines + m_tool.split('\n') + m_calls.split('\n'):
        if l.strip() and l.strip() not in off:
            raise SystemExit(f'not from the official template: {l!r}')
    return new


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('rule', choices=['lfm2', 'granite', 'falcon-h1', 'falcon3'])
    ap.add_argument('own', type=Path, help='the template the bundle carries')
    ap.add_argument('out', type=Path)
    ap.add_argument('--official', type=Path, help="lfm2: LiteRT-LM's models/lfm2/chat_template.jinja")
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f'refusing to overwrite {a.out}')
    own = a.own.read_text(encoding='utf-8')
    print(f'own      {sha256(own)}  {len(own.encode())} B  {a.own}')
    if a.rule == 'lfm2':
        if not a.official:
            raise SystemExit('lfm2 needs --official <models/lfm2/chat_template.jinja>')
        off = a.official.read_text(encoding='utf-8')
        note = 'LiteRT-LM a484d21d' if sha256(off) == OFFICIAL_LFM2 else 'not the a484d21d file'
        print(f'official {sha256(off)}  {len(off.encode())} B  {a.official} ({note})')
        new = lfm2(own, off)
    else:
        new = replace(own, REPLACEMENTS[a.rule])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(new, encoding='utf-8')
    print(f'out      {sha256(new)}  {len(new.encode())} B  {a.out}')
    packed = PACKED.get(sha256(own))
    if packed and packed[0] == a.rule:
        same = sha256(new) == packed[1]
        print(f"{'MATCH' if same else 'DIFFERS'}: the template packed into {packed[2]} (2026-10) is {packed[1][:16]}")
        return 0 if same else 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
