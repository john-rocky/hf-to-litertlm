"""Parse the native Qwen3.5 XML tool form from CLI text-channel content."""
import argparse
import json
from pathlib import Path
import re
import sys

BLOCK = re.compile(r"<tool_call>\n<function=([A-Za-z_][\w.-]*)>\n((?:<parameter=[A-Za-z_][\w.-]*>\n.*?\n</parameter>\n)*)</function>\n</tool_call>", re.S)
PARAM = re.compile(r"<parameter=([A-Za-z_][\w.-]*)>\n(.*?)\n</parameter>\n", re.S)


def text_channel(stdout):
    # CLI model.load_preset prints these lines before native generation.
    lines = stdout.splitlines(keepends=True)
    if lines and lines[0].startswith("Loading preset from "):
        expected = ["- Tools:\n", "  - get_weather\n", "  - multiply\n", "  - web_search\n"]
        if lines[1:5] != expected:
            raise ValueError("Unexpected preset-loader framing: " + repr(lines[:5]))
        stdout = "".join(lines[5:])
    stdout = re.sub(r"\[thought\].*?\[/thought\]", "", stdout, flags=re.S)
    if "[thought]" in stdout:
        stdout = stdout.split("[thought]", 1)[0]
    return stdout


def extract_calls(text):
    calls = []
    for match in BLOCK.finditer(text):
        params = list(PARAM.finditer(match.group(2)))
        keys = [m.group(1) for m in params]
        calls.append({"name": match.group(1), "arguments": {m.group(1): m.group(2) for m in params},
                      "duplicate_argument_keys": len(set(keys)) != len(keys),
                      "parameter_layout_valid": "".join(m.group(0) for m in params) == match.group(2),
                      "start": match.start(), "end": match.end(), "raw_block": match.group(0)})
    return calls


def evaluate(text, expected):
    calls = extract_calls(text)
    one = calls[0] if len(calls) == 1 else None
    arguments = one["arguments"] if one else {}
    plausible = False
    if expected == "get_weather":
        plausible = "tokyo" in arguments.get("location", "").lower()
        required = ["location"]
    elif expected == "multiply":
        required = ["a", "b"]
        try:
            plausible = float(arguments.get("a", "")) == 17 and float(arguments.get("b", "")) == 25
        except ValueError:
            pass
    else:
        required = ["query"]
        plausible = bool(arguments.get("query", "").strip())
    checks = {"exactly_one_block": len(calls) == 1 and text.count("<tool_call>") == 1,
              "expected_function": one is not None and one["name"] == expected,
              "required_arguments": all(k in arguments for k in required),
              "plausible_arguments": plausible,
              "no_duplicate_argument_keys": one is not None and not one["duplicate_argument_keys"],
              "parameter_layout_valid": one is not None and one["parameter_layout_valid"],
              "no_suffix": one is not None and not text[one["end"]:].strip()}
    return {"parsed_calls": calls, "checks": checks, "pass": all(checks.values()), "raw_text": text}


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture',type=Path)
    args=parser.parse_args()
    text = args.capture.read_text()
    print(json.dumps(extract_calls(text_channel(text)), indent=2, ensure_ascii=False))
