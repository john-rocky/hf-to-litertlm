#!/usr/bin/env python3
"""bf16 reference answers for the 8-question gate, straight from openbmb/MiniCPM5-2B.

Renders the checkpoint's own chat_template.jinja (the bundle carries it verbatim) with
the enable_thinking knob: unset (bare assistant prompt = the model's own default, which
is what an app without a ThinkingConfig gets), on ('<think>\n' prefill), off (empty
think block). Scored on the text after the LAST </think> (the reasoning can contain the
expected string). Budget 2048 so a thinking run is never cut mid-think.

  ~/venvs/lt093ctl/bin/python minicpm_work/hf_oracle.py --thinking unset --out minicpm_work/hf_oracle_unset.json
"""
import argparse
import json
import re
import time

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
  ap.add_argument("--hf", default="openbmb/MiniCPM5-2B")
  ap.add_argument("--out", required=True)
  ap.add_argument("--thinking", choices=["on", "off", "unset"], default="unset")
  ap.add_argument("--max-tokens", type=int, default=2048)
  args = ap.parse_args()

  import torch
  import transformers
  tok = transformers.AutoTokenizer.from_pretrained(args.hf)
  model = transformers.AutoModelForCausalLM.from_pretrained(args.hf, dtype=torch.bfloat16).to("mps").eval()
  eos = [1, tok.convert_tokens_to_ids("<|im_end|>")]
  kw = {} if args.thinking == "unset" else {"enable_thinking": args.thinking == "on"}
  rows, passed = [], 0
  for label, q, rx in QUESTIONS:
    prompt = tok.apply_chat_template([{"role": "user", "content": q + SUFFIX}],
                                     tokenize=False, add_generation_prompt=True, **kw)
    ids = tok(prompt, add_special_tokens=False, return_tensors="pt").to("mps")
    t0 = time.time()
    with torch.no_grad():
      out = model.generate(**ids, max_new_tokens=args.max_tokens, do_sample=False, eos_token_id=eos)
    gen = out[0][ids["input_ids"].shape[1]:]
    raw = tok.decode(gen, skip_special_tokens=True)
    torch.mps.empty_cache()
    thought, content = (raw.rsplit("</think>", 1) if "</think>" in raw else ("", raw))
    ok = bool(re.search(rx, content, re.IGNORECASE))
    passed += ok
    rows.append({"label": label, "ok": ok, "n_gen_tokens": int(gen.shape[0]), "think_chars": len(thought),
                 "answer": content.strip()[:300], "thought_head": thought.strip()[:200], "seconds": round(time.time() - t0, 1)})
    print(f"{'OK ' if ok else 'NG '} {label:16s} gen={gen.shape[0]:5d} think={len(thought):5d} | {content.strip()[:100]!r}", flush=True)
  print(f"ORACLE thinking={args.thinking}: {passed}/8  prompt_tail={prompt[-60:]!r}")
  json.dump({"thinking": args.thinking, "passed": passed, "prompt_tail": prompt[-120:], "rows": rows}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
  main()
