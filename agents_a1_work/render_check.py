"""Cross-engine template comparison for the recorded single-turn, tool and prefix cases."""
import argparse
import difflib
import json
from pathlib import Path
import re

here = Path(__file__).resolve().parent
user = {"role": "user", "content": "What is the capital of Japan?"}
tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get weather for a location.", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}},
         {"type": "function", "function": {"name": "add", "description": "Add two numbers.", "parameters": {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}, "required": ["a", "b"]}}}]
conversation = [user, {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "get_weather", "arguments": {"location": "Tokyo"}}}]},
                {"role": "tool", "content": [{"type": "tool_response", "name": "get_weather", "response": {"temp": "18C"}}]},
                {"role": "assistant", "content": "It is 18C."}, {"role": "user", "content": "And Osaka?"}]
cases = {
    "i_default_string": {"messages": [user]},
    "ii_thinking_false": {"messages": [user], "enable_thinking": False},
    "iii_parts": {"messages": [{"role": "user", "content": [{"type": "text", "text": user["content"]}]}]},
    "iv_explicit_system": {"messages": [{"role": "system", "content": "You are terse."}, user]},
    "v_tools": {"messages": [user], "tools": tools},
    "vi_tool_conversation": {"messages": conversation, "tools": tools},
}

def normalize_json_payloads(text):
    # Normalize only parseable JSON payloads inside tool declarations/results.
    # Any other changed character (including XML separators) remains a failure.
    def tools_block(match):
        return "<tools>\n" + "\n".join(json.dumps(json.loads(line), sort_keys=True, separators=(",", ":"), ensure_ascii=False) for line in match.group(1).strip().splitlines()) + "\n</tools>"
    text = re.sub(r"<tools>\n(.*?)\n</tools>", tools_block, text, flags=re.S)
    def response(match):
        try:
            obj = json.loads(match.group(1))
        except json.JSONDecodeError:
            return match.group(0)
        return "<tool_response>\n" + json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n</tool_response>"
    return re.sub(r"<tool_response>\n(.*?)\n</tool_response>", response, text, flags=re.S)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["hf", "minijinja"])
    ap.add_argument('--model',type=Path,default=Path('src_models/Agents-A1-4B'))
    ap.add_argument('--template',type=Path,default=here/'chat_template_agents_a1.jinja')
    ap.add_argument('--output',type=Path,default=Path('out/agents-a1-4b/results'))
    a = ap.parse_args()
    renders=a.output/'render_cases';renders.mkdir(parents=True,exist_ok=True)
    if a.mode == "hf":
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
        for name, context in cases.items():
            context = json.loads(json.dumps(context))
            for message in context["messages"]:
                if message["role"] == "tool":
                    # HF vendor supports text parts, not runtime tool_response parts.
                    message["content"] = json.dumps(message["content"][0]["response"], ensure_ascii=False)
            messages = context.pop("messages")
            rendered = tokenizer.apply_chat_template(messages, **context, tokenize=False, add_generation_prompt=True)
            (renders / f"{name}.hf.txt").write_text(rendered)
        print("HF renders saved for six cases")
        return
    import minijinja
    env = minijinja.Environment(templates={"agents": (a.template).read_text()})
    rows = []
    for name, context in cases.items():
        rendered = env.render_template("agents", **context, add_generation_prompt=True)
        expected_name = "i_default_string" if name == "iii_parts" else name
        expected = (renders / f"{expected_name}.hf.txt").read_text()
        (renders / f"{name}.minijinja.txt").write_text(rendered)
        diff = "".join(difflib.unified_diff(expected.splitlines(True), rendered.splitlines(True), fromfile="HF", tofile="minijinja"))
        (renders / f"{name}.diff").write_text(diff)
        same_outside_json = normalize_json_payloads(rendered) == normalize_json_payloads(expected)
        allowed = name.startswith(("v_", "vi_"))
        rows.append({"case": name, "identical": rendered == expected, "diff_snippet": diff[:10000],
                     "full_diff": f"results/render_cases/{name}.diff", "outside_json_identical": same_outside_json,
                     "verdict": "PASS" if rendered == expected else ("JSON_DIFF_REPORTED" if allowed and same_outside_json else "FAIL")})
    history = [{"role": "user", "content": "Hello."}, {"role": "assistant", "content": "Hi."}, {"role": "user", "content": "How are you?"}]
    for leading_system in [False, True]:
        for parts in [False, True]:
            messages = ([{"role": "system", "content": "You are terse."}] if leading_system else []) + history
            messages = json.loads(json.dumps(messages))
            appended = messages + [{"role": "assistant", "content": "Well."}]
            if parts:
                for message in appended:
                    message["content"] = [{"type": "text", "text": message["content"]}]
                messages = appended[:-1]
            old = env.render_template("agents", messages=messages, add_generation_prompt=False)
            new = env.render_template("agents", messages=appended, add_generation_prompt=False)
            gen = env.render_template("agents", messages=messages, add_generation_prompt=True)
            ok = new.startswith(old) and gen.startswith(old)
            rows.append({"case": f"vii_prefix_system{leading_system}_parts{parts}", "identical": ok,
                         "history_is_prefix_of_appended": new.startswith(old), "history_is_prefix_of_generation": gen.startswith(old),
                         "diff_snippet": "" if ok else repr((old, new, gen)), "verdict": "PASS" if ok else "FAIL"})
    result = {"verdict": "PASS" if all(x["verdict"] != "FAIL" for x in rows) else "FAIL", "cases": rows,
              "json_serialization_diffs": [x["case"] for x in rows if x["verdict"] == "JSON_DIFF_REPORTED"],
              "hf_tool_result_input": "Vendor receives equivalent JSON text; runtime template receives tool_response parts",
              "engine": "PyPI minijinja (template environment) vs transformers tokenizer.apply_chat_template"}
    (a.output / "render_check.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"verdict": result["verdict"], "cases": [{k: x[k] for k in ["case", "identical", "verdict"]} for x in rows]}, indent=2))
    raise SystemExit(0 if result["verdict"] == "PASS" else 1)

if __name__ == "__main__":
    main()
