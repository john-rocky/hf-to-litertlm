#!/usr/bin/env python3
"""Code-generation gate for a .litertlm: generate a function, then RUN it against tests.

The 8-question sanity gate asks general-knowledge questions, so for a code model it
measures nothing that matters. This asks for small functions and executes the result
against assertions — a HumanEval-shaped check, small enough to run as a ship gate.

    python scripts/eval_code.py <model.litertlm> [--backend gpu] [--json out.json]

Scoring is deliberately harsh in one direction: a task counts as passed only if the
extracted code both **imports** and **passes every assertion**. Code that looks right and
raises is a fail, which is the point — reading generated code and nodding is how a broken
quantization gets shipped.

Exit 0 = PASS (>= --min-pass), 1 = FAIL, 2 = harness error.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

DEFAULT_VERIFIER = os.path.expanduser("~/code/litert-mac-verify/.build/release/litert-mac-verify")

# (id, prompt, test source). Kept short: these run on a phone-class model, and a long
# prompt measures the harness's patience rather than the model.
TASKS = [
    ("fib",
     "Write a Python function fib(n) that returns the nth Fibonacci number, with fib(0)=0 and fib(1)=1. Code only.",
     "assert fib(0)==0\nassert fib(1)==1\nassert fib(10)==55"),
    ("reverse_words",
     "Write a Python function reverse_words(s) that returns the words of s in reverse order, space separated. Code only.",
     "assert reverse_words('a b c')=='c b a'\nassert reverse_words('hello')=='hello'"),
    ("is_prime",
     "Write a Python function is_prime(n) that returns True if n is a prime number and False otherwise. Code only.",
     "assert is_prime(2)\nassert is_prime(13)\nassert not is_prime(1)\nassert not is_prime(9)"),
    ("max_sublist",
     "Write a Python function max_sum(nums) that returns the largest sum of any contiguous sublist of the list nums. Code only.",
     "assert max_sum([1,-2,3,4])==7\nassert max_sum([-1,-2])==-1"),
    ("count_vowels",
     "Write a Python function count_vowels(s) that returns how many vowels (aeiou, case insensitive) are in s. Code only.",
     "assert count_vowels('Hello')==2\nassert count_vowels('xyz')==0"),
    ("flatten",
     "Write a Python function flatten(lst) that flattens a list of lists into a single list. Code only.",
     "assert flatten([[1,2],[3]])==[1,2,3]\nassert flatten([])==[]"),
]


def extract_code(text):
    """Prefer a fenced block; fall back to the whole reply. Models that were asked for
    'code only' still sometimes narrate, and the fence is the reliable delimiter."""
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    if m:
        return m.group(1)
    return text


def run_case(verifier, model, prompt, test_src, max_tokens, extra):
    cmd = [verifier, model, prompt, "--max-tokens", str(max_tokens)] + extra
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=400)
    except subprocess.TimeoutExpired:
        return False, "", "timeout"
    # Non-greedy: the verifier keeps logging after `OUTPUT: [<reply>]` and many log lines end
    # in `]`, so a greedy capture swallows kilobytes of warnings as if the model had said them.
    # Harmless here whenever the model fenced its code (extract_code finds the fence first),
    # which is why this survived unnoticed until eval_logic.py graded raw replies.
    m = re.search(r"^OUTPUT: \[(.*?)\]$", p.stdout + "\n" + p.stderr, re.M | re.S)
    reply = m.group(1).replace("⏎", "\n") if m else ""
    code = extract_code(reply)

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code + "\n\n" + test_src + "\nprint('PASS')\n")
        path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=20)
        ok = "PASS" in r.stdout
        err = "" if ok else (r.stderr.strip().split("\n")[-1] if r.stderr.strip() else "no PASS")
    except subprocess.TimeoutExpired:
        ok, err = False, "execution timeout (infinite loop?)"
    finally:
        os.unlink(path)
    return ok, reply, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--verifier", default=DEFAULT_VERIFIER)
    ap.add_argument("--backend", default="gpu")
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--min-pass", type=int, default=4)
    ap.add_argument("--json")
    args = ap.parse_args()

    if not os.path.exists(args.verifier):
        print(f"ERROR: verifier not found: {args.verifier}")
        return 2

    extra = ["--backend", args.backend]
    rows, passed = [], 0
    print(f"== code gate: {os.path.basename(args.model)} ({args.backend}) ==")
    for tid, prompt, test_src in TASKS:
        ok, reply, err = run_case(args.verifier, args.model, prompt, test_src,
                                  args.max_tokens, extra)
        passed += ok
        rows.append({"id": tid, "ok": ok, "error": err, "reply": reply})
        print(f"   [{'✓' if ok else '·'}] {tid:14s} {'' if ok else '— ' + err[:70]}")

    print(f"\n   passed: {passed}/{len(TASKS)}   (threshold {args.min_pass})")
    verdict = passed >= args.min_pass
    print(f"   VERDICT: {'✅ PASS' if verdict else '❌ FAIL'}")
    if args.json:
        json.dump({"model": args.model, "backend": args.backend,
                   "passed": passed, "total": len(TASKS), "rows": rows},
                  open(args.json, "w"), indent=1)
        print(f"   wrote {args.json}")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
