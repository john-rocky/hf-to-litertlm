#!/usr/bin/env python3
"""make_template.py VENDOR.jinja MACRO.jinja OUT.jinja - the Olmo-3-7B-Think template the bundles carry
(olmo3_think_dual.jinja), built mechanically from Ai2's chat_template.jinja (allenai/Olmo-3-7B-Think at d97e442d) with every edit
asserted (each old string must occur exactly the counted number of times):
  (1) prepend the content-parts macro (nemotron_h_work/format_content_dual_tool.jinja in this repo) verbatim (text +
      tool_response parts; LiteRT-LM 0.18 hands content as typed parts, 0.17.x as a string = both render the same text);
  (2) every `message['content']` read goes through format_content(...) (system / user x2 / assistant / environment);
  (3) `message.get('x', none) is not none` -> `message['x'] is defined and message['x'] is not none` for content /
      functions (x2) / function_calls (the LiteRT-LM v0.18.0 template engine has no `.get` rewrite: minijinja
      "dict has no method named get", runtime/components/prompt_template.cc l.40-74).
Everything else is Ai2's text byte for byte: the default system message (only when no system message is given), the
` You do not currently have access to any functions. <functions></functions>` tail of a given system message, the
`functions` / `function_calls` / `environment` branches, `<|im_end|>\\n` after a non-last assistant turn and `eos_token`
after the last one (the runtime renders eos_token as "": prompt_template.h l.98), the generation prompt
`<|im_start|>assistant\\n<think>` (the thought channel opens with the prompt), no strip of the thought from history turns.
The `selectattr('role', 'equalto', 'system')` line is kept (it renders on minijinja 2.14.0).

  python olmo3_work/make_template.py chat_template.jinja nemotron_h_work/format_content_dual_tool.jinja olmo3_work/olmo3_think_dual.jinja

Writes OUT.jinja and prints its size and sha256 (2,485 B, 38dbdec2b80cf57b... for the pinned vendor file). Exit 1 when an edit does
not apply as counted."""
import hashlib
import sys

vendor_p, macro_p, out_p = sys.argv[1:4]
src = open(vendor_p, encoding="utf-8").read()
macro = open(macro_p, encoding="utf-8").read()

EDITS = [  # (old, new, expected count)
    ("{{ '<|im_start|>system\n' + message['content'] }}",
     "{{ '<|im_start|>system\n' + format_content(message['content']) }}", 1),
    ("{{ '<|im_start|>user\n' + message['content'] + '\n' + '<functions>'",
     "{{ '<|im_start|>user\n' + format_content(message['content']) + '\n' + '<functions>'", 1),
    ("{{ '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}",
     "{{ '<|im_start|>user\n' + format_content(message['content']) + '<|im_end|>\n' }}", 1),
    ("{% if message.get('content', none) is not none %}{{ message['content'] }}{% endif %}",
     "{% if message['content'] is defined and message['content'] is not none %}{{ format_content(message['content']) }}{% endif %}", 1),
    ("{{ '<|im_start|>environment\n' + message['content'] + '<|im_end|>\n' }}",
     "{{ '<|im_start|>environment\n' + format_content(message['content']) + '<|im_end|>\n' }}", 1),
    ("message.get('functions', none) is not none",
     "message['functions'] is defined and message['functions'] is not none", 2),
    ("message.get('function_calls', none) is not none",
     "message['function_calls'] is defined and message['function_calls'] is not none", 1),
]
out = src
for old, new, n in EDITS:
    c = out.count(old)
    if c != n:
        print(f"edit does not apply: {old!r} occurs {c} times, expected {n}", file=sys.stderr)
        sys.exit(1)
    out = out.replace(old, new)
for bad in (".get(", ".startswith(", ".endswith(", ".replace(", ".split(", ".join(", ".lstrip(", ".rstrip(", ".strip(",
            ".items(", "{% generation %}"):
    assert bad not in out, bad
assert "message['content'] }}" not in out.replace("format_content(message['content']) }}", "")
out = macro + out
open(out_p, "w", encoding="utf-8").write(out)
b = out.encode("utf-8")
print(f"{out_p}: {len(b)} B sha256 {hashlib.sha256(b).hexdigest()}")
