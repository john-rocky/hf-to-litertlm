#!/usr/bin/env python3
"""make_template.py VENDOR.jinja MACRO.jinja OUT.jinja - the bundle template for NVIDIA-Nemotron-Nano-9B-v2
(nemotron_nano_dual.jinja), built MECHANICALLY from the vendor prompt template (tokenizer_config.json `chat_template`, 4070 B, sha256
b7a3a520...) - every edit is one exact substring replacement asserted to occur the expected number of times; nothing else changes.

  python make_template.py <snapshot>/chat_template.jinja format_content_dual_tool.jinja nemotron_nano_dual.jinja
  (VENDOR.jinja = the `chat_template` string of the pinned tokenizer_config.json, written to a file)

Why each edit (litert-lm 0.18.0 renders the template with minijinja, hands message content as typed parts, and requires each
re-render of a conversation to extend the previous one):
  E0  prepend the dual macro (format_content_dual_tool.jinja, sha256 2a33b764...: text + tool_response parts, or a plain string) -
      the vendor reads `message['content']` as a string (`'/think' in content`, `.replace`, `.strip`), which on 0.18.0 is a list of
      parts: the `/think` test never fires and `.replace` raises.
  E1  the reasoning-switch scan: `content = format_content(message['content'])` (the `/think` / `/no_think` substring test on text).
  E2  the system turn: format_content + `| replace('/think', '') | replace('/no_think', '') | trim` (= the runtime's rewrite of
      `.replace(...)` / `.strip()` (EditTemplateForMinijinja in LiteRT-LM's prompt_template.cc), written out so no Python method
      call is left).
  E3  REMOVED: the vendor pops a trailing assistant message (`ns.last_turn_assistant_content`) and renders it after the loop as
      `<SPECIAL_11>Assistant\\n<think>\\n{content}\\n<SPECIAL_12>\\n` (gen=False) - the form of an assistant turn then depends on
      its POSITION: render([u1, a1], gen=False) ends `...Assistant\\n<think>\\n{a1}` while render([u1, a1, u2], gen=True) has
      `...Assistant\\n{a1}` -> the runtime rejects message 2 ("does not start with the previous").
      Every assistant turn now goes through the loop = the vendor's form for a past turn (thought dropped, the answer after
      `</think>`, stripped). HF never reaches the removed branch with add_generation_prompt=True and a user last (HF parity holds).
  E4  the main loop: `content = format_content(message['content'])`.
  E5  the user turn: `| replace('/think', '') | replace('/no_think', '') | trim` (the vendor's strip of the switch words).
  E6  the tool result: `format_content(message['content'])` (tool_response parts on 0.18.0).
  E7  the assistant turn: `content.split('</think>')[1].strip()` -> `(content | split('</think>'))[1] | trim`, `content.strip()`
      -> `(content | trim)` (twice).
  E8  the generation prompt: the vendor's `<SPECIAL_11>Assistant\\n` + (`<think></think>` when `/no_think` was the last switch,
      else `<think>\\n`), verbatim; the appended `ns.last_turn_assistant_content` and the whole gen=False tail branch are removed
      (E3's other half).
Checks after the edits: no Python method call is left (`.get( .strip( .split( .replace( .startswith( .endswith( .lstrip( .rstrip(
.join( .items(`), the macro is MACRO.jinja byte for byte. Writes OUT (never overwrites) and OUT.edits.json (each edit: old, new,
count). Exit 0 = PASS.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

NL = "\n"
EDITS = [
    ("E1 scan: format_content",
     "{%- for message in messages -%}{%- set content = message['content'] -%}",
     "{%- for message in messages -%}{%- set content = format_content(message['content']) -%}", 1),
    ("E2 system: format_content + replace / trim",
     "{%- set ns.non_tool_system_content = messages[0]['content'].replace('/think', '').replace('/no_think', '').strip() -%}",
     "{%- set ns.non_tool_system_content = format_content(messages[0]['content']) | replace('/think', '') | replace('/no_think', '') | trim -%}",
     1),
    ("E3 drop the trailing-assistant pop (position-dependent form)",
     "{%- if messages[-1]['role'] == 'assistant' -%}{%- set ns.last_turn_assistant_content = messages[-1]['content'].strip() -%}"
     "{%- set messages = messages[:-1] -%}{%- endif -%}",
     "", 1),
    ("E4 main loop: format_content",
     "{%- for message in messages %}{%- set content = message['content'] %}",
     "{%- for message in messages %}{%- set content = format_content(message['content']) %}", 1),
    ("E5 user: replace / trim",
     "{{- '<SPECIAL_11>User" + NL + "' + content.replace('/think', '').replace('/no_think', '').strip() + '" + NL + "' }}",
     "{{- '<SPECIAL_11>User" + NL + "' + (content | replace('/think', '') | replace('/no_think', '') | trim) + '" + NL + "' }}", 1),
    ("E6 tool result: format_content",
     "{{- message['content'] -}}",
     "{{- format_content(message['content']) -}}", 1),
    ("E7a assistant: split after </think>",
     "{%- set content = content.split('</think>')[1].strip() %}",
     "{%- set content = (content | split('</think>'))[1] | trim %}", 1),
    ("E7b assistant: trim (answer)",
     "{{- '<SPECIAL_11>Assistant" + NL + "' + content.strip() }}",
     "{{- '<SPECIAL_11>Assistant" + NL + "' + (content | trim) }}", 1),
    ("E7c assistant: trim (tool-call separator test)",
     "{%- if content.strip() != '' -%}",
     "{%- if (content | trim) != '' -%}", 1),
    ("E8a generation prompt: drop the appended trailing-assistant content",
     "{%- if ns.last_turn_assistant_content is defined and ns.last_turn_assistant_content != '' -%}"
     "{{- ns.last_turn_assistant_content -}}{%- endif -%}{%- else -%}",
     "{%- else -%}", 1),
    ("E8b drop the gen=False trailing-assistant branch",
     "{%- else -%}{%- if ns.last_turn_assistant_content is defined and ns.last_turn_assistant_content != '' -%}"
     "{{- '<SPECIAL_11>Assistant" + NL + "' -}}{%- if ns.enable_thinking is defined and ns.enable_thinking is false -%}"
     "{{- '<think></think>' -}}{%- else -%}{{- '<think>" + NL + "' -}}{%- endif -%}{{- ns.last_turn_assistant_content -}}"
     "{%- if continue_final_message is defined -%}{%- if continue_final_message is false -%}{{- '" + NL + "<SPECIAL_12>" + NL + "' -}}"
     "{%- endif -%}{%- else -%}{{- '" + NL + "<SPECIAL_12>" + NL + "' -}}{%- endif -%}{%- endif -%}{%- endif -%}",
     "{%- endif -%}", 1),
]
METHODS = (".get(", ".strip(", ".split(", ".replace(", ".startswith(", ".endswith(", ".lstrip(", ".rstrip(", ".join(", ".items(")


def main():
    if len(sys.argv) != 4:
        print(__doc__, file=sys.stderr)
        return 2
    vendor_p, macro_p, out_p = (Path(x) for x in sys.argv[1:])
    rep_p = out_p.with_name(out_p.name + ".edits.json")
    for p in (out_p, rep_p):
        if p.exists():
            print(f"refusing to overwrite {p}", file=sys.stderr)
            return 2
    vendor = vendor_p.read_text(encoding="utf-8")
    macro = macro_p.read_text(encoding="utf-8")
    src = vendor
    rows, ok = [], True
    for name, old, new, want in EDITS:
        n = src.count(old)
        good = n == want
        ok = ok and good
        rows.append({"edit": name, "count": n, "expected": want, "ok": good, "old": old, "new": new})
        if good:
            src = src.replace(old, new)
    left = [m for m in METHODS if m in src]
    # the gen=False tail must be gone and the generation prompt must remain exactly once
    gp = "{%- if add_generation_prompt -%}{{- '<SPECIAL_11>Assistant" + NL + "' -}}"
    checks = {"no_python_methods": not left, "methods_left": left,
              "generation_prompt_once": src.count(gp) == 1,
              "no_last_turn_assistant": "last_turn_assistant_content" not in src,
              "no_continue_final_message": "continue_final_message" not in src,
              "think_on_off_prompt_kept": src.count("{{- '<think></think>' -}}") == 1 and src.count("{{- '<think>" + NL + "' -}}") == 1}
    ok = ok and checks["no_python_methods"] and checks["generation_prompt_once"] and checks["no_last_turn_assistant"] \
        and checks["no_continue_final_message"] and checks["think_on_off_prompt_kept"]
    out = macro + src
    if not macro.endswith("{%- endmacro -%}" + NL):
        ok = False
        checks["macro_tail"] = repr(macro[-40:])
    rec = {"vendor": str(vendor_p), "vendor_bytes": len(vendor.encode()), "vendor_sha256": hashlib.sha256(vendor.encode()).hexdigest(),
           "macro": str(macro_p), "macro_sha256": hashlib.sha256(macro.encode()).hexdigest(),
           "out": str(out_p), "out_bytes": len(out.encode()), "out_sha256": hashlib.sha256(out.encode()).hexdigest(),
           "edits": rows, "checks": checks, "status": "PASS" if ok else "FAIL"}
    if ok:
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(out, encoding="utf-8")
    rep_p.write_text(json.dumps(rec, indent=1, ensure_ascii=False) + NL)
    print(rec["status"], json.dumps({k: rec[k] for k in ("vendor_bytes", "vendor_sha256", "out_bytes", "out_sha256")}),
          [(r["edit"][:3], r["count"]) for r in rows], checks)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
