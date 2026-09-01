#!/usr/bin/env python3
"""Build lfm230_LlmMetaProto.pbtext for LFM2.5-230M.

The checkpoint's chat_template.jinja uses HF's `{% generation %}` /
`{% endgeneration %}` assistant-span markers (training-time mask helpers).
litert-lm's minijinja rejects them at engine start:
  "syntax error: unknown statement generation (in <string>:77)"
so the bundle must carry the template with those two statements stripped.
They do not affect rendered text; this script PROVES that by rendering the
original vs stripped template through HF apply_chat_template across the
conversation shapes the CLI can reach, then writes the pbtext (stops [7, 2],
bos <|startoftext|>, thought channel — same scheme as the shipped 1.2B family).
"""
import json
import re
import sys
from pathlib import Path

from transformers import AutoTokenizer

MODEL = "LiquidAI/LFM2.5-230M"
WORK = Path(__file__).resolve().parent
OUT_PBTEXT = WORK / "lfm230_LlmMetaProto.pbtext"

tok = AutoTokenizer.from_pretrained(MODEL)
original = tok.chat_template

STMT = re.compile(r"\{%-?\s*(?:end)?generation\s*-?%\}")
stripped = STMT.sub("", original)
assert not STMT.search(stripped), "generation statement survived the strip"
assert len(STMT.findall(original)) == 2, "expected exactly one generation/endgeneration pair"

# Second minijinja incompatibility (measured on the same engine, after the
# generation strip): "unknown method: map has no method named get". Rewrite
# the two dict-.get truthiness tests to plain indexing — identical semantics
# in both engines (a missing key is falsy-undefined in each).
GET = re.compile(r'(messages\[0\]|message)\.get\("content"\)')
n_get = len(GET.findall(stripped))
assert n_get == 2, f"expected 2 .get(\"content\") sites, found {n_get}"
stripped = GET.sub(r'\1["content"]', stripped)

CASES = [
    ("user_only", [{"role": "user", "content": "What is 17+25?"}], True),
    (
        "system_user",
        [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Name three primes."},
        ],
        True,
    ),
    (
        "multi_turn",
        [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello! How can I help?"},
            {"role": "user", "content": "Add 2 and 2."},
        ],
        True,
    ),
    (
        "past_think_content",
        [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "<think>secret</think>Answer1"},
            {"role": "user", "content": "Q2"},
        ],
        True,
    ),
    (
        "closed_conv_no_gen",
        [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A"},
        ],
        False,
    ),
    (
        "tools",
        [{"role": "user", "content": "weather?"}],
        True,
    ),
]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]

fails = 0
for name, msgs, add_gen in CASES:
    kw = {"tokenize": False, "add_generation_prompt": add_gen}
    if name == "tools":
        kw["tools"] = TOOLS
    tok.chat_template = original
    a = tok.apply_chat_template(msgs, **kw)
    tok.chat_template = stripped
    b = tok.apply_chat_template(msgs, **kw)
    ok = a == b
    print(f"{name}: {'OK' if ok else 'DIFF'} ({len(a)} chars)")
    if not ok:
        fails += 1
        print("  original:", json.dumps(a))
        print("  stripped:", json.dumps(b))

if fails:
    sys.exit(f"{fails} case(s) diverged — do not ship the stripped template")

def pb_escape(s: str) -> str:
    return (
        s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    )

pbtext = f'''start_token {{
  token_str: "<|startoftext|>"
}}
stop_tokens {{
  token_ids {{
    ids: 7
  }}
}}
stop_tokens {{
  token_ids {{
    ids: 2
  }}
}}
max_num_tokens: 4096
llm_model_type {{
  generic_model {{
  }}
}}
jinja_prompt_template: "{pb_escape(stripped)}"
channels {{
  channel_name: "thought"
  start: "<think>"
  end: "</think>"
}}
'''
OUT_PBTEXT.write_text(pbtext)
print(f"render equivalence: {len(CASES)}/{len(CASES)} byte-identical")
print(f"wrote {OUT_PBTEXT} ({OUT_PBTEXT.stat().st_size} B)")
