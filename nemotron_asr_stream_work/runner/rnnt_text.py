#!/usr/bin/env python3
"""The text of nemotron_stream_runner's output: each utterance's token ids (the "tok" list of its {"kind":"utt"} line,
in emission order) decoded with tokenizer.json, special tokens skipped.
  pip install tokenizers
  python rnnt_text.py tokenizer.json out.jsonl      -> one line per file: <name> <TAB> <text>
"""
import json
import sys

from tokenizers import Tokenizer


def main(tokenizer_json, runner_stdout):
    tok = Tokenizer.from_file(tokenizer_json)
    with open(runner_stdout, encoding="utf-8") as f:
        for line in f:
            if line.startswith('{"kind":"utt"'):
                r = json.loads(line)
                print(r["id"] + "\t" + tok.decode(r["tok"], skip_special_tokens=True))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
