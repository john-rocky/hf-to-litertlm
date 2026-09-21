"""Build the pinned dual-form template with the vendor default system text."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import re
import subprocess

here = Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--model',type=Path,required=True)
p.add_argument('--canonical',type=Path,default=here/'resources/qwen3_5_801acdef.jinja')
p.add_argument('--out',type=Path,default=here/'chat_template_agents_a1.jinja')
p.add_argument('--report',type=Path,default=Path('out/agents-a1-4b/results/template_source.json'))
a=p.parse_args()
canonical=a.canonical.read_text()
assert hashlib.sha256(canonical.encode()).hexdigest() == 'ba2e9a48ec90026504236cf463d0d92b06409260382183a147e7377c706be13f'
vendor=(a.model/'chat_template.jinja').read_text()
a.out.parent.mkdir(parents=True,exist_ok=True)
a.report.parent.mkdir(parents=True,exist_ok=True)
default = re.search(r"\{%- set default_system_prompt -%\}.*?\{%- endset -%\}", vendor, re.S).group(0)
macro = """{%- macro format_content(content) -%}
    {%- if content is string -%}
        {{- content -}}
    {%- elif content is sequence -%}
        {%- for item in content -%}
            {%- if item['type'] == 'text' -%}
                {{- item['text'] -}}
            {%- elif item['type'] == 'tool_response' -%}
                {%- if item['response'] is mapping or item['response'] is sequence -%}
                    {{- item['response'] | tojson -}}
                {%- else -%}
                    {{- item['response'] | string -}}
                {%- endif -%}
            {%- endif -%}
        {%- endfor -%}
    {%- endif -%}
{%- endmacro -%}"""
out = re.sub(r"\{%- macro format_content\(content\) -%\}.*?\{%- endmacro -%\}", lambda _: macro + "\n\n" + default, canonical, count=1, flags=re.S)
anchor = "{%- set is_first_system = messages[0]['role'] == 'system' -%}"
out = out.replace(anchor, anchor + "\n{%- set system_content = format_content(messages[0]['content']) if is_first_system else default_system_prompt -%}", 1)
out = out.replace("{%- if tools or is_first_system -%}", "{%- if tools or is_first_system or default_system_prompt -%}", 1)
old = """        {%- if is_first_system -%}
            {{- '\\n\\n' + format_content(messages[0]['content']) -}}
        {%- endif -%}
    {%- elif is_first_system -%}
        {{- format_content(messages[0]['content']) -}}"""
new = """        {{- '\\n\\n' + system_content -}}
    {%- else -%}
        {{- system_content -}}"""
assert old in out
out = out.replace(old, new, 1)
assert default in out and "last_query_index" not in out and "Current date: 2026-07-14" in out
a.out.write_text(out)
a.report.with_suffix(".diff").write_text("".join(difflib.unified_diff(canonical.splitlines(True), out.splitlines(True), fromfile="LiteRT-LM@801acdef", tofile="chat_template_agents_a1.jinja")))
record = {"canonical_ref": "801acdef", "canonical_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
          "template_sha256": hashlib.sha256(out.encode()).hexdigest(), "vendor_default_block_sha256": hashlib.sha256(default.encode()).hexdigest(),
          "default_block_byte_identical": re.search(r"\{%- set default_system_prompt -%\}.*?\{%- endset -%\}", out, re.S).group(0).encode() == default.encode()}
a.report.write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
