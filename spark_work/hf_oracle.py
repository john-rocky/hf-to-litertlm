#!/usr/bin/env python3
"""bf16 reference answers for the 8-question gate, straight from the HF model.

Spark-X2.5 is a THINKING model (default `<think>` prefill). Two A/Bs the ship needs:

  --template upstream : vendor chat_template.jinja (default system block + <think>)
  --template simple   : templates/spark25_think.jinja rendered the same way (what
                        the .litertlm carries); both should score alike.
  --extra-bos         : prepend one more <｜start▁of▁sentence｜> (what the bundle
                        would do if start_token were left in) -- the
                        bundle prepends bos add bos false check.

Scored on the text AFTER the last </think>; a run that never closes the think
block within --max-tokens is reported as unfinished (not a wrong answer).

  ~/venvs/ltconv040dev/bin/python spark_work/hf_oracle.py --hf src_models/Spark-X2.5-1.7B \
      --template upstream --out spark_work/hf_oracle_1p7b_upstream.json
"""
import argparse
import gc
import json
import os
import re
import sys
import time

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.3")  # see gsm8k_bf16.py
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.25")

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
BOS = "<｜start▁of▁sentence｜>"


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--hf", required=True)
  ap.add_argument("--out", required=True)
  ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
  ap.add_argument("--device", default="mps")
  ap.add_argument("--attn", default="sdpa", choices=["eager", "sdpa"])
  ap.add_argument("--template", default="upstream", choices=["upstream", "simple"])
  ap.add_argument("--simple-jinja", default="templates/spark25_think.jinja")
  ap.add_argument("--extra-bos", action="store_true")
  ap.add_argument("--max-tokens", type=int, default=2048)
  args = ap.parse_args()

  import torch
  import transformers
  from jinja2 import Environment

  tok = transformers.AutoTokenizer.from_pretrained(args.hf)
  model = transformers.AutoModelForCausalLM.from_pretrained(
      args.hf, dtype=getattr(torch, args.dtype), trust_remote_code=True,
      attn_implementation=args.attn,
  ).to(args.device).eval()
  simple_tpl = None
  if args.template == "simple":
    simple_tpl = Environment().from_string(open(args.simple_jinja, encoding="utf-8").read())

  rows, correct, unfinished = [], 0, 0
  t0 = time.time()
  for label, q, rx in QUESTIONS:
    msgs = [{"role": "user", "content": q + SUFFIX}]
    if args.template == "upstream":
      text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    else:
      text = simple_tpl.render(messages=msgs, add_generation_prompt=True)
    if args.extra_bos:
      text = BOS + text
    ids = tok(text, add_special_tokens=False, return_tensors="pt").to(args.device)
    with torch.no_grad():
      out = model.generate(**ids, max_new_tokens=args.max_tokens, do_sample=False,
                           cache_implementation="static")
    gen = out[0][ids["input_ids"].shape[1]:]
    raw = tok.decode(gen, skip_special_tokens=False)
    finished = "</think>" in raw
    ans = raw.rsplit("</think>", 1)[-1] if finished else raw
    ans = ans.replace("<｜end▁of▁sentence｜>", "")
    ok = bool(re.search(rx, ans, re.I))
    correct += ok
    unfinished += (not finished)
    rows.append({"label": label, "ok": ok, "finished": finished, "n_gen": int(gen.shape[0]),
                 "answer": ans.strip()[:300], "raw_head": raw[:200]})
    print(f"[{'ok' if ok else 'NG'}{'' if finished else '/UNFINISHED'}] {label}: "
          f"n_gen={gen.shape[0]} answer={ans.strip()[:80]!r}", flush=True)
    del out, gen, ids
    gc.collect()
    if args.device == "mps":
      torch.mps.synchronize()
      torch.mps.empty_cache()
  summary = {"hf": args.hf, "dtype": args.dtype, "attn": args.attn, "template": args.template,
             "extra_bos": args.extra_bos, "max_tokens": args.max_tokens,
             "correct": correct, "unfinished": unfinished, "elapsed_s": round(time.time() - t0, 1),
             "rows": rows}
  json.dump(summary, open(args.out, "w"), indent=2, ensure_ascii=False)
  print(f"correct={correct}/8 unfinished={unfinished} template={args.template} "
        f"extra_bos={args.extra_bos} -> {args.out}")


if __name__ == "__main__":
  main()
