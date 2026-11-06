#!/usr/bin/env python3
"""Compare two transcript files (JSONL rows with `id` and `text`: transcribe.py, reference_hf.py) clip by clip: the
same text, or the words that differ (difflib on whitespace-split words). --require-identical exits 1 unless every clip
of A has the same text in B.

  python compare.py A.jsonl B.jsonl [--require-identical]
"""
import argparse
import difflib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def word_diff(a, b):
    wa, wb = a.split(), b.split()
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=wa, b=wb, autojunk=False).get_opcodes():
        if op != "equal":
            out.append(f"{' '.join(wa[i1:i2]) or '-'} -> {' '.join(wb[j1:j2]) or '-'}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--require-identical", action="store_true")
    args = ap.parse_args()
    A, B = C.read_jsonl(args.a), C.read_jsonl(args.b)
    same = 0
    for k, ra in A.items():
        rb = B.get(k)
        if rb is None:
            print(f"[{k}] missing in {args.b}")
            continue
        if ra["text"] == rb["text"]:
            same += 1
            print(f"[{k}] identical ({len(ra['text'].split())} words)")
        else:
            print(f"[{k}] differs: " + "; ".join(word_diff(ra["text"], rb["text"]) or ["(whitespace only)"]))
    print(f"IDENTICAL {same}/{len(A)} ({Path(args.a).name} vs {Path(args.b).name})")
    return 1 if args.require_identical and same != len(A) else 0


if __name__ == "__main__":
    raise SystemExit(main())
