#!/usr/bin/env python3
"""Formal-logic gate for a .litertlm: unambiguous answers, graded by exact match.

Written for a finetune that CLAIMS formal-logic specialisation. The 8-question sanity gate
is general knowledge, so it cannot see that claim at all -- a model could be better or worse
at logic than its base and score identically. This asks questions whose correct answer is a
single fixed token or short phrase, so grading needs no judgement.

    python scripts/eval_logic.py <model.litertlm> [--backend gpu] [--json out.json]

Run it on the finetune AND on its base with the same flags, then compare. A number from one
model alone says nothing: these items are hand-written, so their absolute difficulty is
uncalibrated, and only the difference between two models on the SAME items is meaningful.

Grading is exact-match on a normalised final answer, deliberately strict about the thing that
matters (did it reach the right conclusion) and deliberately blind to everything else (how
much it reasoned first). Reasoning models emit <think> blocks -- those are stripped before
grading rather than counted against the model.

Exit 0 = ran, 1 = harness error. There is no pass threshold: this is a comparison instrument,
not a publish gate.
"""
import argparse
import json
import os
import re
import subprocess
import sys

DEFAULT_VERIFIER = os.path.expanduser("~/code/litert-mac-verify/.build/release/litert-mac-verify")

# (id, question, accepted answers, ANSWER SPACE). The answer space is per-item on purpose.
# A single shared vocabulary looks tidy and is wrong: it has to contain "a" for the A/B/C
# items, and "a" is also the English article, so on a yes/no item the grader happily reads
# the last "a" in ordinary prose as the answer "A". That mis-scored two items on the first
# run. Scanning only within THIS item's legal answers removes the whole failure mode.
TASKS = [
    ("syllogism_valid",
     "All roses are flowers. All flowers need water. Do all roses need water? Answer yes or no.",
     ["yes"], ["yes", "no"]),
    ("syllogism_invalid",
     "All cats are mammals. All dogs are mammals. Does it follow that all cats are dogs? Answer yes or no.",
     ["no"], ["yes", "no"]),
    ("modus_tollens",
     "If it is raining then the ground is wet. The ground is NOT wet. Is it raining? Answer yes or no.",
     ["no"], ["yes", "no"]),
    ("affirming_consequent",
     "If it is raining then the ground is wet. The ground IS wet. Can we conclude it is raining? Answer yes or no.",
     ["no"], ["yes", "no"]),
    ("contrapositive",
     "Statement: If a number is divisible by 4 then it is even. Which is its contrapositive: "
     "(A) if a number is even then it is divisible by 4, "
     "(B) if a number is not even then it is not divisible by 4, "
     "(C) if a number is not divisible by 4 then it is not even? Answer A, B or C.",
     ["b"], ["a", "b", "c"]),
    ("demorgan",
     "Which is equivalent to NOT (P AND Q): (A) NOT P AND NOT Q, (B) NOT P OR NOT Q, "
     "(C) P OR Q? Answer A, B or C.",
     ["b"], ["a", "b", "c"]),
    ("quantifier_negation",
     "What is the negation of 'all swans are white': (A) no swans are white, "
     "(B) at least one swan is not white, (C) all swans are not white? Answer A, B or C.",
     ["b"], ["a", "b", "c"]),
    ("disjunctive_syllogism",
     "Either the key is in the drawer or it is in the car. It is not in the drawer. "
     "Where is the key? Answer drawer or car.",
     ["car"], ["drawer", "car"]),
    ("transitive_order",
     "Ann is taller than Bob. Bob is taller than Cal. Who is shortest? Answer Ann, Bob or Cal.",
     ["cal"], ["ann", "bob", "cal"]),
    ("necessary_sufficient",
     "Being a square is sufficient for being a rectangle. Is being a rectangle sufficient for "
     "being a square? Answer yes or no.",
     ["no"], ["yes", "no"]),
    ("biconditional",
     "P if and only if Q. Q is false. What is P? Answer true or false.",
     ["false"], ["true", "false"]),
    ("vacuous_truth",
     "There are no unicorns in this room. Is the statement 'every unicorn in this room is blue' "
     "true or false? Answer true or false.",
     ["true"], ["true", "false"]),
]

THINK = re.compile(r"<think>.*?</think>", re.S | re.I)

# The verifier prints `OUTPUT: [<reply>]` and then keeps writing runtime log lines, many of
# which end in `]`. A GREEDY `\[(.*)\]$` with re.S therefore runs past the reply and captures
# ~9 KB of warnings as if the model had said it -- which on 2026-08-18 turned a correct "Yes"
# into a graded-wrong answer and produced an entirely fictitious 4/12. Non-greedy stops at the
# first `]` that ends a line, which is the reply's own closing bracket.
OUTPUT_RE = re.compile(r"^OUTPUT: \[(.*?)\]$", re.M | re.S)


def extract_output(text):
    m = OUTPUT_RE.search(text)
    return m.group(1).replace("⏎", "\n") if m else ""


def normalise(reply):
    """Strip reasoning, then take the last decisive token. Models answer as 'Yes.', '**B**',
    'The answer is B' -- all of which are the same answer and none of which should cost a
    point, since the question under test is the conclusion, not the formatting."""
    t = THINK.sub(" ", reply)
    t = re.sub(r"</?think>", " ", t, flags=re.I)
    t = t.replace("*", " ").replace("`", " ")
    words = re.findall(r"[A-Za-z]+", t)
    return [w.lower() for w in words]


def grade(reply, accepted, space):
    words = normalise(reply)
    if not words:
        return False, "empty"
    # Scan from the end: the conclusion is what the model settled on, not what it floated
    # mid-reasoning. Only tokens inside THIS item's answer space count, so neither trailing
    # prose nor a stray article can stand in for an answer.
    for w in reversed(words):
        if w in space:
            return w in accepted, w
    return False, " ".join(words[-6:])[:60]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--verifier", default=DEFAULT_VERIFIER)
    ap.add_argument("--backend", default="gpu")
    ap.add_argument("--max-tokens", type=int, default=2048,
                    help="reasoning models need room; too small reads as a wrong answer")
    ap.add_argument("--json")
    ap.add_argument("--label")
    args = ap.parse_args()

    if not os.path.exists(args.verifier):
        print(f"ERROR: verifier not found: {args.verifier}")
        return 1
    if not os.path.exists(args.model):
        print(f"ERROR: model not found: {args.model}")
        return 1

    label = args.label or os.path.basename(os.path.dirname(args.model))
    print(f"== logic gate: {label} ({args.backend}) ==", flush=True)

    rows, passed = [], 0
    for tid, q, accepted, space in TASKS:
        cmd = [args.verifier, args.model, q, "--max-tokens", str(args.max_tokens),
               "--backend", args.backend]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        except subprocess.TimeoutExpired:
            rows.append(dict(id=tid, ok=False, got="timeout"))
            print(f"   [·] {tid:24s} timeout", flush=True)
            continue
        reply = extract_output(p.stdout + "\n" + p.stderr)
        ok, got = grade(reply, accepted, space)
        passed += ok
        rows.append(dict(id=tid, ok=ok, got=got, accepted=accepted, reply=reply))
        print(f"   [{'✓' if ok else '·'}] {tid:24s} got={got!r} want={accepted}", flush=True)

    print(f"\n   {label}: {passed}/{len(TASKS)}")
    if args.json:
        json.dump(dict(model=args.model, label=label, backend=args.backend,
                       passed=passed, total=len(TASKS), rows=rows),
                  open(args.json, "w"), indent=1)
        print(f"   wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
