#!/usr/bin/env python3
"""8-question quality gate through the litert-lm 0.17 CLI, thinking-aware.

Same 8 checks as scripts/verify_quality.py. Each question is its own process
(a thinking model would otherwise spend one budget on eight questions).
Greedy (temperature 0, top-k 1, fixed seed), --cache no, --thinking true so the
runtime's thought channel is active. The answer is scored on the text after the
last </think>; an opened-but-unclosed think block counts as unfinished.

    python3 spark_work/gate8q_cli.py <model.litertlm> <cpu|gpu> [out.json]
    env: LITERT_LM (default ~/venvs/lt0170run/bin/litert-lm), MAX_NUM_TOKENS (4096)
"""
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter

LITERT_LM = os.environ.get("LITERT_LM", os.path.expanduser("~/venvs/lt0170run/bin/litert-lm"))
MODEL, BACKEND = sys.argv[1], sys.argv[2]
OUT = sys.argv[3] if len(sys.argv) > 3 else None
MAX_NUM_TOKENS = os.environ.get("MAX_NUM_TOKENS", "4096")

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
    words = text.split()
    if len(words) >= 10:
        grams = [" ".join(words[i:i + 5]) for i in range(len(words) - 4)]
        if grams and Counter(grams).most_common(1)[0][1] >= 3:
            return True
        if len(set(words)) / len(words) < 0.30:
            return True
    if len(text) >= 40 and len(set(text)) < 15:
        return True
    return text.count("<|") >= 5


def split_think(text):
    """(answer, finished). The 0.17 CLI prints a declared channel as
    `[thought] ...reasoning... [/thought]` (litert_lm_cli/commands/run.py) and then the
    answer text; an undeclared channel leaks a literal <think>..</think> instead.
    Both spans are removed; an opened-but-unclosed span = unfinished."""
    if "[thought]" in text and "[/thought]" not in text:
        return "", False
    if "<think>" in text and "</think>" not in text:
        return "", False
    text = re.sub(r"\[thought\].*?\[/thought\]", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip(), True


results, correct, degen, unfinished = [], 0, 0, 0
t0 = time.time()
for label, q, pat in QUESTIONS:
    cmd = [LITERT_LM, "run", MODEL, "--prompt", q + SUFFIX, "--backend", BACKEND,
           "--cache", "no", "--temperature", "0", "--top-k", "1", "--seed", "0",
           "--thinking", "true", "--max-num-tokens", MAX_NUM_TOKENS]
    t1 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, stdin=subprocess.DEVNULL)
    dt = time.time() - t1
    raw = p.stdout
    m = re.search(r"\[thought\](.*?)(\[/thought\]|$)", raw, flags=re.DOTALL)
    thought = m.group(1) if m else ""
    ans, finished = split_think(raw)
    ans = ans.replace("<｜end▁of▁sentence｜>", "").strip()
    ok = bool(re.search(pat, ans, re.IGNORECASE))
    dg = degenerate(ans) if ans else False
    correct += ok
    degen += dg
    unfinished += (not finished)
    results.append({"label": label, "ok": ok, "finished": finished, "degenerate": dg,
                    "seconds": round(dt, 1), "exit": p.returncode,
                    "n_thought_chars": len(thought), "thought_head": thought[:200],
                    "answer": ans[:300], "raw_head": raw[:400],
                    "stderr_tail": p.stderr[-600:]})
    tag = "ok" if ok else "NG"
    if not finished:
        tag += "/UNFINISHED"
    if dg:
        tag += "/DEGEN"
    print(f"[{tag}] {label} ({dt:.0f}s): {ans[:90]!r}", flush=True)

verdict = "PASS" if correct >= 6 and degen == 0 else "FAIL"
print(f"correct={correct}/8 unfinished={unfinished} degenerate={degen} verdict={verdict} "
      f"backend={BACKEND} elapsed={time.time()-t0:.0f}s")
if OUT:
    json.dump({"model": MODEL, "backend": BACKEND, "litert_lm": LITERT_LM,
               "correct": correct, "unfinished": unfinished, "degenerate": degen,
               "verdict": verdict, "results": results}, open(OUT, "w"), indent=2, ensure_ascii=False)
sys.exit(0 if verdict == "PASS" else 1)
