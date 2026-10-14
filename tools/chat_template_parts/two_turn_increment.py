#!/usr/bin/env python3
"""Where does the newline after the stop token go? A desk check of a chat template's second turn
against LiteRT-LM's incremental prefill.

    python tools/chat_template_parts/two_turn_increment.py <template.jinja> <tokenizer.json> \
        [--stop "<|im_end|>"] [--q1 ...] [--a1 ...] [--q2 ...] [--parts]

LiteRT-LM prefills only the increment render(history + message) - render(history), and the stop
token the model emitted stays in the cache as the pending token of the next prefill. So after the
second message the cache holds

    ids(render([u1], generation prompt)) + ids(a1) + [stop] + ids(increment)

while the upstream prompt is ids(render([u1, a1, u2], generation prompt)). This script renders
both with the runtime's Jinja settings (tools/chat_template_prefix/desk_render.py: minijinja
rewrites, trim_blocks, lstrip_blocks, keep_trailing_newline) and prints the first difference.
The reply is assumed to tokenize the same way it was generated; the runtime's own check is three
turns on the engine.

Exit 0 when the two id lists are equal, 1 when they differ, 2 when the new render does not start
with the old one (the prefix rule; see tools/chat_template_prefix).
"""
import argparse
import os
import sys

from tokenizers import Tokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "chat_template_prefix"))
from desk_render import Renderer  # noqa: E402


def message(role, text, as_parts):
    return {"role": role, "content": [{"type": "text", "text": text}] if as_parts else text}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("template")
    ap.add_argument("tokenizer", help="tokenizer.json of the model")
    ap.add_argument("--stop", default="<|im_end|>", help="the stop token the model emits after a reply")
    ap.add_argument("--q1", default="What is the capital of France? Answer in one word.")
    ap.add_argument("--a1", default="Paris.")
    ap.add_argument("--q2", default="How many letters does it have?")
    ap.add_argument("--parts", action="store_true", help="hand content as typed parts (the 0.18 form)")
    ap.add_argument("--engine", default="jinja2", choices=["jinja2", "minijinja"])
    args = ap.parse_args()

    tok = Tokenizer.from_file(args.tokenizer)
    stop_id = tok.token_to_id(args.stop)
    if stop_id is None:
        sys.exit(f"stop token {args.stop!r} is not in the tokenizer")
    ids = lambda s: tok.encode(s, add_special_tokens=False).ids
    r = Renderer(open(args.template).read(), engine=args.engine)
    u1, a1, u2 = (message("user", args.q1, args.parts), message("assistant", args.a1, args.parts),
                  message("user", args.q2, args.parts))

    t1 = r.render([u1], True)
    old = r.render([u1, a1], False)
    new = r.render([u1, a1, u2], True)
    print(f"template {args.template}  tokenizer {args.tokenizer}  stop {args.stop!r} (id {stop_id})  engine {args.engine}")
    if not new.startswith(old):
        i = next((k for k, (a, b) in enumerate(zip(old, new)) if a != b), min(len(old), len(new)))
        print(f"PREFIX FAIL: the new render does not start with the old one (first difference at char {i}):")
        print(f"  old …{old[max(0, i - 40):i + 40]!r}")
        print(f"  new …{new[max(0, i - 40):i + 40]!r}")
        return 2
    inc = new[len(old):]
    cache = ids(t1) + ids(args.a1) + [stop_id] + ids(inc)
    upstream = ids(new)
    print(f"turn-1 prompt {len(ids(t1))} ids, reply {len(ids(args.a1))} ids + stop, increment {len(ids(inc))} ids: {inc!r}")
    print(f"cache {len(cache)} ids, upstream two-turn render {len(upstream)} ids")
    if cache == upstream:
        print("EQUAL: the cache after the second message is the upstream prompt")
        return 0
    k = next((k for k, (a, b) in enumerate(zip(cache, upstream)) if a != b), min(len(cache), len(upstream)))
    print(f"GAP: first difference at id {k}")
    print(f"  cache    {cache[k:k + 4]} = {tok.decode(cache[k:k + 4], skip_special_tokens=False)!r}")
    print(f"  upstream {upstream[k:k + 4]} = {tok.decode(upstream[k:k + 4], skip_special_tokens=False)!r}")
    missing = len(upstream) - len(cache)
    if missing > 0:
        print(f"  the cache is {missing} token(s) short: {tok.decode(upstream[k:k + missing], skip_special_tokens=False)!r} never enters it")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
