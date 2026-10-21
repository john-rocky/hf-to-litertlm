#!/usr/bin/env python3
"""The text of redux_asr_runner_b's output: each 30 s window's token ids decoded with tokenizer.json (special tokens
skipped, repeated ids kept: the TDT decode), the non-empty window texts joined with one space. A token's window is
its "frame" // 375.
  pip install tokenizers
  python tdt_text.py tokenizer.json out.jsonl      -> one line per file: <name> <TAB> <text>
"""
import json
import sys

from tokenizers import Tokenizer


def main(tokenizer_json, runner_stdout):
    tok = Tokenizer.from_file(tokenizer_json)
    with open(runner_stdout, encoding="utf-8") as f:
        for line in f:
            if not line.startswith('{"kind":"utt"'):
                continue
            r = json.loads(line)
            wins = {}
            for t, fr in zip(r["tok"], r["frame"]):
                wins.setdefault(fr // 375, []).append(t)
            texts = [tok.decode(wins[w], skip_special_tokens=True) for w in sorted(wins)]
            print(r["id"] + "\t" + " ".join(t for t in texts if t))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
