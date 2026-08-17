#!/usr/bin/env python3
"""fp32/bf16 reference answers for the 8-question gate, straight from the HF model.

The oracle exists because a backend split (GPU says X, CPU says Y) is not evidence of
quantization damage until the *reference itself* has been asked the same question the
same way — twice now a "GPU bug" turned out to be the fp32 model getting it wrong
([[gates-can-lie-in-our-favour]]). Same 8 questions and same answer regexes as
scripts/verify_quality.py, same greedy decode, granite's own chat template.

  python granite41_work/hf_oracle.py --hf src_models/granite-4.1-3b --out granite41_work/hf_oracle.json
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


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--hf", default="src_models/granite-4.1-3b")
  ap.add_argument("--out", default="granite41_work/hf_oracle.json")
  ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
  ap.add_argument("--device", default="mps")
  ap.add_argument("--max-tokens", type=int, default=128)
  args = ap.parse_args()

  import torch
  import transformers

  tok = transformers.AutoTokenizer.from_pretrained(args.hf)
  model = transformers.AutoModelForCausalLM.from_pretrained(
      args.hf, dtype=getattr(torch, args.dtype)
  ).to(args.device).eval()

  rows, correct = [], 0
  for label, q, rx in QUESTIONS:
    text = tok.apply_chat_template(
        [{"role": "user", "content": q + SUFFIX}],
        tokenize=False, add_generation_prompt=True,
    )
    # add_special_tokens=False: granite's tokenizer sets add_bos_token=False, and the
    # rendered template already carries every marker the model expects.
    ids = tok(text, add_special_tokens=False, return_tensors="pt").to(args.device)
    with torch.no_grad():
      out = model.generate(**ids, max_new_tokens=args.max_tokens, do_sample=False)
    ans = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    ok = bool(re.search(rx, ans, re.I))
    correct += ok
    rows.append({"label": label, "question": q, "answer": ans, "correct": ok})
    print(f"{'OK ' if ok else 'BAD'} {label}: {ans[:100]!r}")

  json.dump({"model": args.hf, "dtype": args.dtype, "correct": correct,
             "total": len(QUESTIONS), "rows": rows},
            open(args.out, "w"), indent=1)
  print(f"ORACLE {correct}/{len(QUESTIONS)} -> {args.out}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
