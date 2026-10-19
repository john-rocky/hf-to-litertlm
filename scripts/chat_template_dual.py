#!/usr/bin/env python3
"""Make a chat template render on LiteRT-LM 0.18 the way transformers renders it, and prove it.

From litert-lm 0.18.0 the runtime renders the template with minijinja on native values: every
message `content` is a list of text parts, never a string (tools/chat_template_parts/README.md), and
a map has no Python methods. A template written for transformers then drops the message (`content =
''` when not a string), raises (`'...' + content`, `message.get('k')`: "map has no method named get")
or prints the list. Two edits, each applied only when the template cannot render without it:
  1. `X.get('k')` -> `X['k']` (a missing key is undefined, which the runtime's `none` test reads as
     the None that .get returns); `(X['k'] if X['k'] is defined else none)` where the result meets
     an `is none` test;
  2. the dual form: LiteRT-LM's `format_content` macro prepended and every direct content read
     (`message.content`, `message['content']`, `messages[0].content`) routed through it.

    uv run python scripts/chat_template_dual.py <template.jinja> [--out final.jinja] [--json proof.json]

The candidate is rendered the runtime's way (tools/chat_template_prefix/desk_render.py: minijinja
2.14.0, the runtime's rewrites of Python idioms, `none` = undefined, bos_token = eos_token = "", and
the messages given as minijinja literals so no Python method leaks in); the reference is
transformers' own render of the original (render_jinja_template, the same bos / eos), on c1 user,
c2 system + user, c3 user / assistant / user, c4 user with enable_thinking true and false (when the
template reads it). A template that renders parts itself (its parts render keeps every text) is
compared parts with parts; a template that needs the dual form is compared with the string render.
form = unchanged / get / dual / get+dual; pass = every case byte-identical. Exit 0 = pass, 1 = not.

    uv run python scripts/chat_template_dual.py --runtime-render <bundle.litertlm>

prints the litert-lm runtime's own render (Conversation.render_message_to_string, CPU engine, no
compile cache) of c1-c3 as JSON, for scripts/convert.py to compare with the same reference.
"""
import argparse
import hashlib
import importlib.metadata
import json
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools" / "chat_template_prefix"))
import desk_render  # noqa: E402

# LiteRT-LM a8178d7d, models/qwen3/chat_template.jinja (Apache-2.0): the macro, byte for byte.
FORMAT_CONTENT = """{%- macro format_content(content) -%}
    {%- if content is string -%}
        {{- content -}}
    {%- elif content is sequence -%}
        {%- for item in content -%}
            {%- if item['type'] == 'text' -%}
                {{- item['text'] -}}
            {%- endif -%}
        {%- endfor -%}
    {%- endif -%}
{%- endmacro -%}"""
MACRO_SOURCE = "LiteRT-LM a8178d7d models/qwen3/chat_template.jinja (format_content, Apache-2.0)"

QUOTED = r"'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\""
ACCESS = r"\b(?:messages\[0\]|message)(?:\.content|\[\s*['\"]content['\"]\s*\])"
GET = (r"(?P<obj>\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*|\[[^\[\]]*\])*)\.get\(\s*(?P<key>'[^'\\]*'|\"[^\"\\]*\")\s*\)"
       r"(?P<none>\s+is\s+(?:not\s+)?none\b)?")
# a content read of any name, for naming what a failed proof left unwrapped
ANY_ACCESS = re.compile(r"\b[A-Za-z_][\w\[\]0-9]*(?:\.content|\[\s*['\"]content['\"]\s*\])")

Q = "What is the capital of France?"
SYS = "You are a concise, helpful assistant. Preserve system marker SYS_AUDIT_314159."
ANS = "Paris is the capital of France. ASSISTANT_AUDIT_271828."
Q2 = "Which country is Paris in? USER_SECOND_161803."
LEAK = re.compile(r"\[\s*\{\s*['\"](?:type|text)['\"]\s*:")


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def edit_expressions(source, pattern, replace):
    """Apply `replace` to the matches of `pattern` inside {{ }} / {% %} only, never inside a
    comment, a string literal or the text between tags. Returns (text, count)."""
    token = re.compile(pattern + "|" + QUOTED)
    pieces, pos, count = [], 0, 0
    while pos < len(source):
        start = re.search(r"\{[{%#]", source[pos:])
        if start is None:
            pieces.append(source[pos:])
            break
        a = pos + start.start()
        pieces.append(source[pos:a])
        opener = source[a:a + 2]
        closer = {"{{": "}}", "{%": "%}", "{#": "#}"}[opener]
        quote, escaped, b = None, False, a + 2
        while b < len(source):
            char = source[b]
            if escaped:
                escaped = False
            elif quote:
                if char == "\\":
                    escaped = True
                elif char == quote:
                    quote = None
            elif opener != "{#" and char in "'\"":
                quote = char
            elif source.startswith(closer, b):
                break
            b += 1
        body = source[a + 2:b]

        def sub(match):
            nonlocal count
            if match.group(0).startswith(('"', "'")):
                return match.group(0)
            count += 1
            return replace(match)

        pieces.append(opener + (body if opener == "{#" else token.sub(sub, body)) + source[b:b + 2])
        pos = b + 2
    return "".join(pieces), count


def rewrite_get(source):
    """`X.get('k')` -> `X['k']`; before an `is none` test, the defined-or-none form."""
    counts = {"get->index": 0, "get->defined-or-none": 0}

    def rep(m):
        index = f"{m.group('obj')}[{m.group('key')}]"
        if m.group("none"):
            counts["get->defined-or-none"] += 1
            return f"({index} if {index} is defined else none){m.group('none')}"
        counts["get->index"] += 1
        return index

    text, _ = edit_expressions(source, GET, rep)
    return text, {k: v for k, v in counts.items() if v}


def wrap_content_reads(source):
    """The dual form: every direct content read through format_content, the macro in front."""
    body, n = edit_expressions(source, ACCESS, lambda m: "format_content(" + m.group(0) + ")")
    return FORMAT_CONTENT + "\n" + body, n


def cases(source):
    c = [("c1", [("user", Q)], {}),
         ("c2", [("system", SYS), ("user", Q)], {}),
         ("c3", [("user", Q), ("assistant", ANS), ("user", Q2)], {})]
    if "enable_thinking" in source:
        c += [("c4_true", [("user", Q)], {"enable_thinking": True}),
              ("c4_false", [("user", Q)], {"enable_thinking": False})]
    return c


def messages(turns, form):
    return [{"role": r, "content": t if form == "string" else [{"type": "text", "text": t}]}
            for r, t in turns]


def literal(value):
    """A minijinja literal for a JSON-like value (native maps and lists: no Python methods)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "none"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ", ".join(literal(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{json.dumps(k)}: {literal(v)}" for k, v in value.items()) + "}"
    raise TypeError(type(value).__name__)


def _result(fn):
    try:
        return {"ok": True, "text": fn()}
    except Exception as e:  # noqa: BLE001 — a template error is a result here
        return {"ok": False, "error": f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"}


def runtime_render(source, turns, form, extra):
    """The runtime's way: desk_render's settings, the messages as literals at the head."""
    def go():
        body = desk_render.none_test_to_undefined(desk_render.edit_for_minijinja(source))
        env = desk_render.make_minijinja_env()
        env.add_template("t", "{% set messages = " + literal(messages(turns, form)) + " %}" + body)
        return env.render_template("t", add_generation_prompt=True, now=desk_render.NOW_TS,
                                   bos_token="", eos_token="", **extra)
    return _result(go)


def reference_render(source, turns, form, extra):
    """transformers' own render of the template (jinja2, Python methods), bos / eos bound to ""."""
    from transformers.utils.chat_template_utils import render_jinja_template

    def go():
        out = render_jinja_template(conversations=[messages(turns, form)], chat_template=source,
                                    add_generation_prompt=True, bos_token="", eos_token="", **extra)
        rendered = out[0] if isinstance(out, tuple) else out
        return rendered[0]
    return _result(go)


def same(a, b):
    return (a["ok"] and b["ok"] and a["text"] == b["text"]) or (
        not a["ok"] and not b["ok"] and a["error"] == b["error"])


def classify(turns, ref, got):
    """equal / drop / raise / leak / differs: what the runtime render does to the reference."""
    if same(ref, got):
        return "equal"
    if not got["ok"]:
        return "raise"
    if LEAK.search(got["text"]):
        return "leak"
    if ref["ok"] and any(t in ref["text"] and t not in got["text"] for _, t in turns):
        return "drop"
    return "differs"


def retains(turns, r):
    return r["ok"] and not LEAK.search(r["text"]) and all(t in r["text"] for _, t in turns)


def prove(source):
    """(record, final template). record: form, edits, per-case classes, basis and hashes, pass."""
    record = {"original_sha256": sha(source), "form": None, "edits": {}, "final_sha256": None,
              "oracle": f"minijinja {importlib.metadata.version('minijinja')} (runtime settings, "
                        "literal messages) against transformers render_jinja_template",
              "macro": MACRO_SOURCE, "macro_sha256": sha(FORMAT_CONTENT),
              "classes": {}, "basis": {}, "cases": {}, "pass": False, "reason": None,
              "unwrapped": None}
    all_cases = cases(source)
    ref = {n: {f: reference_render(source, t, f, x) for f in ("string", "parts")} for n, t, x in all_cases}
    record["classes"] = {n: classify(t, ref[n]["string"], runtime_render(source, t, "parts", x))
                         for n, t, x in all_cases}
    # each case is compared with transformers' string render, or with its parts render when the
    # template renders parts itself (every text kept) and the string render differs from it
    basis = {n: ("string" if same(ref[n]["string"], ref[n]["parts"]) or not retains(t, ref[n]["parts"])
                 else "parts") for n, t, x in all_cases}

    def matches(text):
        out = {}
        for n, t, x in all_cases:
            got_p = runtime_render(text, t, "parts", x)
            want = ref[n][basis[n]]
            out[n] = {"basis": basis[n], "reference": _h(want), "runtime_parts": _h(got_p),
                      "equal": same(want, got_p) and want["ok"]}
        return out

    form, final = [], source
    checked = matches(final)
    if not all(c["equal"] for c in checked.values()):
        trial = [runtime_render(final, t, "parts", x) for n, t, x in all_cases]
        if any(not r["ok"] and "no method named get" in r["error"] for r in trial):
            final, counts = rewrite_get(final)
            record["edits"].update(counts)
            form.append("get")
            checked = matches(final)
    if not all(c["equal"] for c in checked.values()):
        final, n = wrap_content_reads(final)
        if n:
            record["edits"]["format_content"] = n
            form.append("dual")
            basis = {k: "string" for k in basis}  # the dual form gives the string prompt
            checked = matches(final)
            for name, t, x in all_cases:  # and the string form still renders the same
                got_s = runtime_render(final, t, "string", x)
                checked[name]["runtime_string"] = _h(got_s)
                checked[name]["equal"] = checked[name]["equal"] and same(ref[name]["string"], got_s)
    record["form"] = "+".join(form) or "unchanged"
    record["basis"] = {n: c["basis"] for n, c in checked.items()}
    record["cases"] = checked
    record["final_sha256"] = sha(final)
    record["pass"] = all(c["equal"] for c in checked.values())
    if not record["pass"]:
        bad = [n for n, c in checked.items() if not c["equal"]]
        unwrapped = [m.group(0) for m in ANY_ACCESS.finditer(final)
                     if not final[max(0, m.start() - 15):m.start()].endswith("format_content(")]
        record["unwrapped"] = unwrapped[0] if unwrapped else None
        record["reason"] = (f"the runtime's render differs from transformers' in {', '.join(bad)}"
                            + (f"; content read left unwrapped: {unwrapped[0]}" if unwrapped else ""))
    return record, final


def _h(r):
    return {"sha256": sha(r["text"]), "bytes": len(r["text"].encode())} if r["ok"] else {"error": r["error"]}


RUNTIME_CASES = ("c1", "c2", "c3")


def runtime_renders(bundle):
    """{case: the runtime's render} for c1-c3: the history goes in as the conversation's preface,
    the last user message through Conversation.render_message_to_string (litert-lm, CPU). The
    compile cache goes to a temporary dir, never next to the bundle."""
    import litert_lm
    out = {}
    with tempfile.TemporaryDirectory() as cache:
        engine = litert_lm.Engine(str(bundle), backend=litert_lm.Backend.CPU(), cache_dir=cache)
        for name, turns, _ in cases(""):
            if name not in RUNTIME_CASES:
                continue
            history = [{"role": r, "content": t} for r, t in turns[:-1]]
            conv = engine.create_conversation(messages=history or None)
            out[name] = conv.render_message_to_string({"role": "user", "content": turns[-1][1]})
        del engine
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("template", nargs="?")
    ap.add_argument("--out", help="write the final template here (when it differs from the input)")
    ap.add_argument("--json", help="write the proof here")
    ap.add_argument("--runtime-render", metavar="BUNDLE",
                    help="print the runtime's render of c1-c3 for this bundle as JSON")
    args = ap.parse_args()
    if args.runtime_render:
        print(json.dumps(runtime_renders(args.runtime_render)))
        return 0
    if not args.template:
        ap.error("a template file or --runtime-render BUNDLE")
    source = Path(args.template).read_text()
    record, final = prove(source)
    if final != source and args.out:
        Path(args.out).write_text(final)
    text = json.dumps(record, indent=2)
    print(text)
    if args.json:
        Path(args.json).write_text(text + "\n")
    return 0 if record["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
