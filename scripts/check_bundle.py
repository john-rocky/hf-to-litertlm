#!/usr/bin/env python3
"""Load a .litertlm on the pinned litert-lm CLI (CPU) and check that the user's text reaches the model.

    uv run python scripts/check_bundle.py out/<model>/<model>_int8.litertlm
                                          [--prompt P] [--expect REGEX] [--json OUT]

Runs `litert-lm run <file> --prompt P --backend cpu --top-k 1 --cache no` (greedy; `--cache no`
writes nothing next to the bundle) twice:
  * the pass/fail prompt, an echo of a word that is only in the prompt (default "Repeat this word
    exactly and say nothing else: pineapple"). PASS = the CLI loads the bundle and exits 0, and the
    answer — the CLI's channel blocks (a thinking model's thought) and a closed <think> block
    removed — is non-empty, not degenerate and contains the word, case-insensitive;
  * 17 + 25, reported as information (`arithmetic`: expected 42, the answer) and never part of the
    verdict. Arithmetic measures the model, not the conversion (SmolLM2-1.7B-Instruct answers 32 to it
    in transformers fp32 too); the echo catches a bundle whose template drops the user's text on
    litert-lm 0.18 (such a bundle answers fluently about an empty message). The wording is the one
    all three reference models echo: "Reply with exactly one word: pineapple" reads two ways, and
    gemma-4-E2B-it answered it with one word about pineapple ("Fruit").
Prints one JSON object (engine version, backend, seconds, answer heads, pass) and, on a failure, the
FAIL line of docs/one-command-lock.md on stderr (stage engine-load: the CLI failed before any output;
generate: the answer is empty, degenerate or does not carry the word).

The CLI is $LITERT_LM_CLI, else the litert-lm next to this interpreter (the pinned environment under
`uv run`), else litert-lm on PATH. Exit 0 = PASS, 1 = FAIL, 2 = harness error (no file, no CLI).
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from failure_line import failure  # noqa: E402
from verify_quality import degenerate  # noqa: E402

PROMPT = "Repeat this word exactly and say nothing else: pineapple"
EXPECT = r"pineapple"
ARITHMETIC = ("What is 17 + 25? Answer with the number only.", "42")


def resolve_cli():
    here = Path(sys.executable).parent / "litert-lm"
    return (os.environ.get("LITERT_LM_CLI")
            or (str(here) if here.exists() else None)
            or shutil.which("litert-lm"))


# The CLI prints a channel's text between "[<name>] " and " [/<name>]" (litert_lm_cli
# commands/run.py, 0.18.0: _execute_prompt and SessionState.close_channel); the answer is the rest,
# as the runtime's own response object keeps channels apart from the text.
CHANNEL = re.compile(r"\[([A-Za-z0-9_-]+)\] (.*?) \[/\1\]\n?", flags=re.DOTALL)


def strip_channels(text):
    """The CLI's output without its channel blocks (the thought of a thinking model)."""
    return CHANNEL.sub("", text)


def strip_think(text):
    """Drop a closed <think>...</think> block; an unclosed one leaves no answer."""
    if "<think>" in text and "</think>" not in text:
        return ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def run_prompt(cli, bundle, prompt, timeout):
    """(proc or None on timeout, answer, seconds) of one `litert-lm run`."""
    cmd = [cli, "run", str(bundle), "--prompt", prompt, "--backend", "cpu",
           "--top-k", "1", "--cache", "no"]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, "", round(time.time() - t0, 1)
    return proc, strip_think(strip_channels(proc.stdout)), round(time.time() - t0, 1)


def head(text):
    return " ".join(text.split())[:80]


def check(bundle, prompt=PROMPT, expect=EXPECT, timeout=900):
    cli = resolve_cli()
    result = {"file": str(bundle), "prompt": prompt, "expect": expect, "backend": "cpu",
              "engine": None, "cli": cli, "seconds": None, "exit": None,
              "answer_head": None, "pass": False, "failure": None, "arithmetic": None}
    if not cli:
        result["failure"] = failure("engine-load", "litert-lm CLI not found")
        return result
    ver = subprocess.run([cli, "--version"], capture_output=True, text=True)
    result["engine"] = (ver.stdout.strip().split("version")[-1].strip()
                        if ver.returncode == 0 else None)
    proc, answer, result["seconds"] = run_prompt(cli, bundle, prompt, timeout)
    if proc is None:
        result["failure"] = failure("generate", f"timeout after {timeout} s")
        return result
    result["exit"] = proc.returncode
    result["answer_head"] = head(answer)
    # the CLI reports a load error in its own output and can still exit 0
    load_error = re.search(r"(?im)^.*\berror\b.*$", proc.stderr) if not proc.stdout.strip() else None
    if proc.returncode != 0 or load_error:
        what = (load_error.group(0) if load_error
                else (proc.stderr.strip().splitlines() or [f"exit {proc.returncode}"])[-1])
        result["failure"] = failure("engine-load" if not proc.stdout.strip() else "generate",
                                    what, context=proc.stderr[-4000:])
        return result
    if not answer:
        result["failure"] = failure("generate", "empty answer"
                                    + (" (unclosed <think>)" if "<think>" in proc.stdout else "")
                                    + (" (the output was all channel text)" if CHANNEL.search(proc.stdout) else ""))
    elif degenerate(answer):
        result["failure"] = failure("generate", f"degenerate answer: {result['answer_head']!r}")
    elif not re.search(expect, answer, flags=re.IGNORECASE):
        result["failure"] = failure("generate", f"the answer does not carry {expect!r} from the prompt: "
                                                f"{result['answer_head']!r}")
    else:
        result["pass"] = True
    q, want = ARITHMETIC
    aproc, aanswer, aseconds = run_prompt(cli, bundle, q, timeout)
    result["arithmetic"] = {"prompt": q, "expected": want, "answer_head": head(aanswer),
                            "match": bool(aproc and re.search(rf"\b{want}\b", aanswer)),
                            "seconds": aseconds}
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("bundle")
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--expect", default=EXPECT, help="regex the answer must contain (case-insensitive)")
    ap.add_argument("--json", help="also write the result here")
    args = ap.parse_args()
    if not Path(args.bundle).is_file():
        print(f"ERROR: not a file: {args.bundle}", file=sys.stderr)
        return 2
    result = check(args.bundle, args.prompt, args.expect)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.json:
        Path(args.json).write_text(text + "\n")
    if result["failure"]:
        print(result["failure"]["line"], file=sys.stderr)
    if not result["cli"]:
        return 2
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
