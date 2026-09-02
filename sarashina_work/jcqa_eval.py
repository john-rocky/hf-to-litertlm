#!/usr/bin/env python3
"""JCommonsenseQA (JGLUE v1.3 valid split, first N rows) for the sarashina2.2 lane —
the Japanese accuracy row for the card, one harness for both the bf16 reference
and the .litertlm bundles.

Prompt: the question + the five choices, ask for the choice text. Scored by
exact match of the gold choice text in the answer, with the other four choices
rejected if they appear first (a model that lists everything scores 0). Greedy.
Numbers are comparable only within this harness (bf16 row vs bundle rows).

  # bf16 reference (HF, MPS)
  ~/venvs/ltconv040dev/bin/python sarashina_work/jcqa_eval.py hf <hf_model_or_dir> --n 100 --json-out x.json
  # bundle through the released CLI, one engine per question, explicit backend
  LITERT_LM=~/venvs/lt0160run/bin/litert-lm python3 sarashina_work/jcqa_eval.py engine <model.litertlm> --backend cpu --n 100 --json-out y.json
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "evaldata" / "jcqa_valid-v1.3.json"
LITERT_LM = os.environ.get("LITERT_LM", os.path.expanduser("~/venvs/lt0160run/bin/litert-lm"))
INSTR = "次の質問に対して、選択肢の中から最も適切なものを一つ選び、その選択肢の文字列だけを答えてください。"


def load(n):
    rows = [json.loads(l) for l in DATA.read_text(encoding="utf-8").splitlines() if l.strip()]
    return rows[:n]


def prompt_of(r):
    choices = [r[f"choice{i}"] for i in range(5)]
    return (f"{INSTR}\n\n質問: {r['question']}\n選択肢: " + " / ".join(choices))


def score(r, text):
    """Correct iff the gold choice text appears and no wrong choice appears before it."""
    gold = r[f"choice{r['label']}"]
    t = text.strip()
    pos = t.find(gold)
    if pos < 0:
        return False
    for i in range(5):
        if i == r["label"]:
            continue
        c = r[f"choice{i}"]
        if c == gold or c in gold:
            continue
        p = t.find(c)
        if 0 <= p < pos:
            return False
    return True


def ask_engine(model, backend, prompt, max_num_tokens, timeout=600):
    p = subprocess.run(
        [LITERT_LM, "run", model, "--prompt", prompt, "--backend", backend,
         "--cache", "no", "--temperature", "0", "--seed", "0",
         "--max-num-tokens", str(max_num_tokens)],
        capture_output=True, text=True, timeout=timeout)
    return p.stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["hf", "engine"])
    ap.add_argument("model")
    ap.add_argument("--backend", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--max-num-tokens", type=int, default=512)
    ap.add_argument("--tag")
    ap.add_argument("--json-out")
    args = ap.parse_args()
    rows = load(args.n)

    if args.mode == "hf":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(dev).eval()

        def ask(prompt):
            rendered = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                               tokenize=False, add_generation_prompt=True)
            ids = tok(rendered, add_special_tokens=False, return_tensors="pt")["input_ids"].to(dev)
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=args.max_new, do_sample=False,
                                     eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
            return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    else:
        def ask(prompt):
            return ask_engine(args.model, args.backend, prompt, args.max_num_tokens)

    t0, correct, results = time.time(), 0, []
    for i, r in enumerate(rows):
        text = ask(prompt_of(r))
        ok = score(r, text)
        correct += ok
        gold = r["choice%d" % r["label"]]
        head = " ".join(text.split())[:60]
        results.append({"q_id": r["q_id"], "gold": gold, "ok": ok, "answer": text[:200]})
        print("[%d/%d] gold=%s %s (%d/%d) %r" % (i + 1, len(rows), gold, "ok" if ok else "NG",
                                                correct, i + 1, head), flush=True)
    summary = {"tag": args.tag or os.path.basename(args.model), "mode": args.mode,
               "backend": args.backend if args.mode == "engine" else "hf-bf16",
               "n": len(rows), "correct": correct, "acc": round(correct / len(rows), 3),
               "seconds": round(time.time() - t0, 1)}
    print(json.dumps(summary, ensure_ascii=False))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps({"summary": summary, "results": results},
                                                  indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
