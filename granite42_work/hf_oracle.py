#!/usr/bin/env python3
"""bf16 reference answers for the 8-question gate, straight from the HF model.

granite-4.2-3b is a THINKING model (ChatML + <think> prefill), so two things differ
from the granite-4.1 oracle:
  * the budget is 1024 tokens (the model reasons before answering; 128 would cut it
    mid-think and produce a false fail — reasoning models need a big budget);
  * the regex is scored on the text AFTER the last </think>, because the reasoning
    text itself can contain the expected number (e.g. it repeats "0.9" while
    comparing) and would fake a pass.

Also an A/B on the template: `--template upstream` renders the official
chat_template.jinja (which always emits an EMPTY system block first), `--template
simple` renders the qwen3_think-style ChatML the .litertlm bundle will carry (no
system block when the app sends none). If both score 8/8 the simple template is a
faithful carrier for the gate shape.

  python granite42_work/hf_oracle.py --template upstream --out granite42_work/hf_oracle_upstream.json
  python granite42_work/hf_oracle.py --template simple  --out granite42_work/hf_oracle_simple.json
"""
import argparse
import json
import re
import sys

QUESTIONS = [
    ("17+25=42",         "What is 17 + 25?",                                       r"\b42\b"),
    ("capital=Tokyo",    "What is the capital of Japan?",                          r"tokyo"),
    ("opp(hot)=cold",    'What is the opposite of "hot"?',                         r"\bcold\b"),
    ("days/week=7",      "How many days are in a week?",                           r"\bseven\b|\b7\b"),
    ("thanks(fr)=merci", 'How do you say "thank you" in French?',                  r"merci"),
    ("8*7=56",           "What is 8 times 7?",                                     r"\b56\b"),
    ("0.9>0.11",         "Which is larger: 0.9 or 0.11?",                          r"0\.9"),
    ("rhyme=blue",       'Complete the rhyme: "Roses are red, violets are ___"',   r"\bblue\b"),
]
SUFFIX = " Answer briefly."


def render_simple(q):
  return ("<|im_start|>user\n" + q + "<|im_end|>\n"
          "<|im_start|>assistant\n<think>\n")


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--hf", default="src_models/granite-4.2-3b")
  ap.add_argument("--out", default="granite42_work/hf_oracle.json")
  ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
  ap.add_argument("--device", default="mps")
  ap.add_argument("--template", default="upstream", choices=["upstream", "simple"])
  ap.add_argument("--max-tokens", type=int, default=1024)
  args = ap.parse_args()

  import torch
  import transformers

  tok = transformers.AutoTokenizer.from_pretrained(args.hf)
  model = transformers.AutoModelForCausalLM.from_pretrained(
      args.hf, dtype=getattr(torch, args.dtype)
  ).to(args.device).eval()

  rows, correct = [], 0
  for label, q, rx in QUESTIONS:
    if args.template == "upstream":
      text = tok.apply_chat_template(
          [{"role": "user", "content": q + SUFFIX}],
          tokenize=False, add_generation_prompt=True,
      )
    else:
      text = render_simple(q + SUFFIX)
    # add_special_tokens=False: the post_processor adds nothing anyway (verified:
    # TemplateProcessing with a bare Sequence), and the render carries every marker.
    ids = tok(text, add_special_tokens=False, return_tensors="pt").to(args.device)
    with torch.no_grad():
      out = model.generate(**ids, max_new_tokens=args.max_tokens, do_sample=False)
    raw = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    # score only the answer half; the reasoning can contain the expected string
    ans = raw.rsplit("</think>", 1)[-1] if "</think>" in raw else raw
    finished = "</think>" in raw
    ok = bool(re.search(rx, ans, re.I))
    correct += ok
    rows.append({"label": label, "question": q, "answer": ans.strip(),
                 "think_chars": len(raw) - len(ans), "think_closed": finished,
                 "correct": ok})
    print(f"{'OK ' if ok else 'BAD'} {label}: think={len(raw)-len(ans)}ch "
          f"closed={finished} ans={ans.strip()[:100]!r}")

  json.dump({"model": args.hf, "dtype": args.dtype, "template": args.template,
             "max_tokens": args.max_tokens, "correct": correct,
             "total": len(QUESTIONS), "rows": rows},
            open(args.out, "w"), indent=1)
  print(f"ORACLE({args.template}) {correct}/{len(QUESTIONS)} -> {args.out}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
