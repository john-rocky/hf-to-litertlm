#!/usr/bin/env python3
"""8-question text quality gate via `litert-lm run` (0.16.0 CLI), same checks
as lfm26_work/gate8q_cli.py but with a configurable CLI path for VL bundles.

    python3 gate_text.py <model.litertlm> <backend> [out.json]
"""
import json
import os
import re
import subprocess
import sys

LITERT_LM = os.environ.get(
    "LITERT_LM", "litert-lm")
MODEL, BACKEND = sys.argv[1], sys.argv[2]
OUT = sys.argv[3] if len(sys.argv) > 3 else None

SUFFIX = " Answer briefly."
QUESTIONS = [
    ("17+25=42",        "What is 17 + 25?",                                  r"\b42\b"),
    ("capital=Tokyo",   "What is the capital of Japan?",                     r"tokyo"),
    ("opp(hot)=cold",   'What is the opposite of "hot"?',                    r"\bcold\b"),
    ("days/week=7",     "How many days are in a week?",                      r"\bseven\b|\b7\b"),
    ("thanks(fr)=merci", 'How do you say "thank you" in French?',            r"merci"),
    ("8*7=56",          "What is 8 times 7?",                                r"\b56\b"),
    ("0.9>0.11",        "Which is larger: 0.9 or 0.11?",                     r"0\.9"),
    ("rhyme=blue",      'Complete the rhyme: "Roses are red, violets are ___"', r"\bblue\b"),
]


def degenerate(text):
    words = re.findall(r"\w+", text.lower())
    if len(words) >= 12:
        from collections import Counter
        top = Counter(words).most_common(1)[0][1]
        if top / len(words) > 0.5:
            return True
    return len(text.strip()) == 0


results, correct, degen = [], 0, 0
for label, q, pat in QUESTIONS:
    p = subprocess.run(
        [LITERT_LM, "run", MODEL, "--prompt", q + SUFFIX, "--backend", BACKEND,
         "--cache", "no", "--temperature", "0", "--seed", "0"],
        capture_output=True, text=True, timeout=600)
    text = p.stdout.strip()
    err = p.stderr.strip()[-300:] if p.returncode != 0 else ""
    ok = bool(re.search(pat, text, re.IGNORECASE))
    dg = degenerate(text)
    correct += ok
    degen += dg
    results.append({"label": label, "ok": ok, "degenerate": dg,
                    "rc": p.returncode, "text": text[:200], "err": err})
    print(f"[{'ok' if ok else 'NG'}{'/DEGEN' if dg else ''}] {label}: {text[:80]!r}"
          + (f" RC={p.returncode} {err[:120]}" if p.returncode else ""))

verdict = "PASS" if correct >= 6 and degen == 0 else "FAIL"
print(f"correct={correct}/8 degenerate={degen} verdict={verdict}")
if OUT:
    json.dump({"model": os.path.basename(MODEL), "backend": BACKEND,
               "correct": correct, "degenerate": degen, "verdict": verdict,
               "results": results}, open(OUT, "w"), indent=2)
sys.exit(0 if verdict == "PASS" else 1)
