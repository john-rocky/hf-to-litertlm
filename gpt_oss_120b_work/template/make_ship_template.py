#!/usr/bin/env python3
"""make_ship_template.py: template/gptoss_litertlm.jinja = template/vendor_b5c939de.jinja with the edits listed below,
each an asserted one-time replacement (a rendered string changes only where an edit says):
  1. .items() -> |items (the runtime's minijinja has no pycompat)
  2. the training-only branch (last assistant turn without the generation prompt) removed: one inference form for
     every assistant turn without tool calls (LiteRT-LM's prefix contract)
  3. content as typed parts (LiteRT-LM 0.18+ hands templates text / tool_response parts): a content_text macro takes
     a string as is and joins the text parts; a tool message renders each tool_response part's response with the
     vendor's |tojson (a string content renders as before)
  4. messages = [] (LiteRT-LM renders a messageless preface): `messages and messages[0]...`
"""
import os
import sys

K = os.path.dirname(os.path.abspath(__file__))
s = open(os.path.join(K, "vendor_b5c939de.jinja")).read()


def rep(a, b, n=1):
    global s
    assert s.count(a) == n, (s.count(a), a[:90])
    s = s.replace(a, b)


# 1
rep("param_spec.properties.items()", "param_spec.properties|items")
rep("tool.parameters.properties.items()", "tool.parameters.properties|items")
# 2
start = s.index("        {%- elif loop.last and not add_generation_prompt %}")
end = s.index("        {%- else %}\n            {#- CoT is dropped during all previous turns")
s = s[:start] + s[end:]
# 3
rep("""{#- System Message Construction ============================================ #}
""", """{#- Content as a string or as typed parts (LiteRT-LM): the text parts joined #}
{%- macro content_text(content) -%}
    {%- if content is string -%}
        {{- content -}}
    {%- elif content is sequence -%}
        {%- for part in content -%}
            {%- if part.type == "text" -%}
                {{- part.text -}}
            {%- endif -%}
        {%- endfor -%}
    {%- endif -%}
{%- endmacro -%}

{#- System Message Construction ============================================ #}
""")
rep("""    {%- set developer_message = messages[0].content %}""",
    """    {%- set developer_message = content_text(messages[0].content) %}""")
rep("""    {%- if message.role == 'assistant' -%}
        {#- Checks to ensure the messages are being passed in the format we expect #}
        {%- if "content" in message %}
            {%- if "<|channel|>analysis<|message|>" in message.content or "<|channel|>final<|message|>" in message.content %}""",
    """    {%- if message.role == 'assistant' -%}
        {%- set message_text = content_text(message.content) %}
        {#- Checks to ensure the messages are being passed in the format we expect #}
        {%- if "content" in message %}
            {%- if "<|channel|>analysis<|message|>" in message_text or "<|channel|>final<|message|>" in message_text %}""")
rep("""            {%- if message.content and message.thinking %}""", """            {%- if message_text and message.thinking %}""")
rep("""            {%- elif message.content and not future_final_message.found %}
                {{- "<|start|>assistant<|channel|>analysis<|message|>" + message.content + "<|end|>" }}""",
    """            {%- elif message_text and not future_final_message.found %}
                {{- "<|start|>assistant<|channel|>analysis<|message|>" + message_text + "<|end|>" }}""")
rep("""            {{- "<|start|>assistant<|channel|>final<|message|>" + message.content + "<|end|>" }}""",
    """            {{- "<|start|>assistant<|channel|>final<|message|>" + message_text + "<|end|>" }}""")
rep("""        {{- "<|start|>functions." + last_tool_call.name }}
        {{- " to=assistant<|channel|>commentary<|message|>" + message.content|tojson + "<|end|>" }}""",
    """        {%- if message.content is string %}
            {{- "<|start|>functions." + last_tool_call.name }}
            {{- " to=assistant<|channel|>commentary<|message|>" + message.content|tojson + "<|end|>" }}
        {%- else %}
            {%- for part in message.content %}
                {{- "<|start|>functions." + last_tool_call.name }}
                {{- " to=assistant<|channel|>commentary<|message|>" + (part.response if part.type == "tool_response" else part.text)|tojson + "<|end|>" }}
            {%- endfor %}
        {%- endif %}""")
rep("""        {{- "<|start|>user<|message|>" + message.content + "<|end|>" }}""",
    """        {{- "<|start|>user<|message|>" + content_text(message.content) + "<|end|>" }}""")
# 4
rep("""{%- if messages[0].role == "developer" or messages[0].role == "system" %}""",
    """{%- if messages and (messages[0].role == "developer" or messages[0].role == "system") %}""")
out = os.path.join(K, "gptoss_litertlm.jinja")
open(out, "w").write(s)
print("wrote", out, len(s), "B")
