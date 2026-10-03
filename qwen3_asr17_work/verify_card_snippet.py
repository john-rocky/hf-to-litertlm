#!/usr/bin/env python3
"""Run the card's Python usage block unmodified on the released runtime (litert-lm-api 0.17.1, ~/venvs/lt0171run).
Copied from confucius4_r2t2_work/verify_card_snippet.py.

The first ```python block of the card (--card, default cards/qwen3-asr-1.7b-litert.md) is executed as is, in a
scratch directory that holds the bundle under the file name the block opens (symlink to the ship bundle
out/export/q17_30s_C_lt/model.litertlm) and `clip.wav` (a copy of one fixture), so the block's relative paths resolve.
Its printed transcript must equal the 155-clip runtime row of that clip (out/rt/q17_30s_C_lt_full155_cpu.jsonl, CPU,
same API). Then the block's transcribe() is called with the forced-language prefix its text documents, and its output
recorded. The scratch directory (the runtime writes its caches next to the model path) is deleted.

  ~/venvs/lt0171run/bin/python verify_card_snippet.py --clip zh --language Chinese  -> out/verify_card_snippet_zh.json
"""
import argparse
import contextlib
import io
import json
import os
import re
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common  # noqa: E402

CARD = os.path.join(HERE, "cards", "qwen3-asr-1.7b-litert.md")
if not os.path.exists(CARD):  # public mirror: cards/ sits next to this directory
    CARD = os.path.join(os.path.dirname(HERE), "cards", "qwen3-asr-1.7b-litert.md")
BUNDLE = os.path.join(HERE, "out", "export", "q17_30s_C_lt", "model.litertlm")  # the ship bundle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="zh")
    ap.add_argument("--card", default=CARD)
    ap.add_argument("--gate", default=os.path.join(HERE, "out", "rt", "q17_30s_C_lt_full155_cpu.jsonl"),
                    help="155-clip runtime rows of the ship bundle (CPU, the bundle's own template)")
    ap.add_argument("--language", default="Chinese")
    args = ap.parse_args()
    card = open(args.card).read()
    code = re.search(r"```python\n(.*?)```", card, re.S).group(1)
    opened = re.search(r'Engine\(\s*"([^"]+\.litertlm)"', code).group(1)
    clip = [c for c in common.load_clips() if c["name"] == args.clip][0]
    gate = {}
    for line in open(args.gate):
        r = json.loads(line)
        if r.get("clip"):
            gate[r["clip"]] = r
    tmp = tempfile.mkdtemp(prefix="q17_card_", dir=os.environ.get("TMPDIR"))
    cwd = os.getcwd()
    doc = {"card": os.path.relpath(args.card, HERE), "block_sha1": __import__("hashlib").sha1(code.encode()).hexdigest()[:12],
           "clip": args.clip, "bundle": os.path.realpath(BUNDLE), "opened_as": opened, "scratch": tmp}
    try:
        os.symlink(BUNDLE, os.path.join(tmp, opened))
        shutil.copy(clip["path"], os.path.join(tmp, "clip.wav"))
        os.chdir(tmp)
        ns = {"__name__": "__card__"}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(compile(code, "card_python_block", "exec"), ns)  # the card's block, unmodified
        printed = buf.getvalue().strip()
        doc["snippet_printed"] = printed
        doc["gate_text"] = gate[args.clip]["text"]
        doc["snippet_equals_gate"] = printed == gate[args.clip]["text"].strip()
        print("snippet printed:", repr(printed), "| = 155-clip runtime row:", doc["snippet_equals_gate"], flush=True)
        forced = ns["transcribe"]("clip.wav", prefix=f"language {args.language}<asr_text>")
        doc["forced_language"] = args.language
        doc["forced_language_text"] = forced
        print(f"transcribe(clip, language={args.language!r}):", repr(forced), flush=True)
        doc["litert_lm"] = __import__("importlib.metadata").metadata.version("litert-lm-api")
    finally:
        os.chdir(cwd)
        doc["scratch_files_at_end"] = sorted(os.listdir(tmp))
        shutil.rmtree(tmp)
    doc["ok"] = bool(doc.get("snippet_equals_gate"))
    json.dump(doc, open(os.path.join(HERE, "out", f"verify_card_snippet_{args.clip}.json"), "w"), ensure_ascii=False, indent=1)
    print("VERIFY_OK" if doc["ok"] else "VERIFY_FAILED", flush=True)


if __name__ == "__main__":
    main()
