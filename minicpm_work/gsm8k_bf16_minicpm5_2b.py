#!/usr/bin/env python3
"""GSM8K bf16 reference for MiniCPM5-2B — the parity baseline row.

Protocol matched to the engine rows (minicpm_work/gsm8k_engine.py, which imports
minicpm_work/eval_gsm8k_api.py): same questions (evaldata/gsm8k_test.jsonl, first n),
same COT suffix, same extract()/norm(), greedy, max-new-tokens per --max-tokens.
Prompt = the checkpoint's OWN chat_template.jinja via apply_chat_template with the
`enable_thinking` knob (--thinking off = the official MiniCPM5-1B card's eval setting;
on = '<think>\n' prefill; unset = bare assistant prompt, the model's own default).
The template emits <s> itself, so the render is tokenized with add_special_tokens=False.
Content vs thinking split at the LAST </think>; extraction falls back to the thinking
text exactly like the engine harness, so the rows are comparable.

  ~/venvs/lt093ctl/bin/python minicpm_work/gsm8k_bf16.py --thinking off --n 100 \
      --jsonl minicpm_work/gsm8k_bf16_off_rows.jsonl --json-out minicpm_work/gsm8k_bf16_off.json
"""
import argparse
import importlib.util
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
_spec = importlib.util.spec_from_file_location(
    "eval_gsm8k_api", os.path.join(HERE, "eval_gsm8k_api.py"))
_api = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_api)
extract, _norm, load_q, COT = _api.extract, _api.norm, _api.load_q, _api.COT


def norm(x):
  try:
    return _norm(x)
  except (OverflowError, ValueError):
    return str(x).strip()


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--hf", default="openbmb/MiniCPM5-2B")
  ap.add_argument("--n", type=int, default=100)
  ap.add_argument("--start", type=int, default=0)
  ap.add_argument("--max-tokens", type=int, default=2048)
  ap.add_argument("--thinking", choices=["on", "off", "unset"], default="off")
  ap.add_argument("--json-out", default=None)
  ap.add_argument("--jsonl", default=None)
  args = ap.parse_args()

  import torch
  import transformers

  tok = transformers.AutoTokenizer.from_pretrained(args.hf)
  model = transformers.AutoModelForCausalLM.from_pretrained(
      args.hf, dtype=torch.bfloat16).to("mps").eval()
  eos = [1, tok.convert_tokens_to_ids("<|im_end|>")]
  kw = {} if args.thinking == "unset" else {"enable_thinking": args.thinking == "on"}

  qs = load_q(args.n)
  correct, results = 0, []
  t0 = time.time()
  for i, (q, gold) in enumerate(qs):
    if i < args.start:
      continue
    prompt = tok.apply_chat_template([{"role": "user", "content": q + COT}],
                                     tokenize=False, add_generation_prompt=True, **kw)
    if i == args.start:
      print("PROMPT TAIL:", repr(prompt[-80:]), flush=True)
    ids = tok(prompt, add_special_tokens=False, return_tensors="pt").to("mps")
    with torch.no_grad():
      out = model.generate(**ids, max_new_tokens=args.max_tokens,
                           do_sample=False, eos_token_id=eos)
    raw = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    del out, ids
    torch.mps.empty_cache()
    if "</think>" in raw:
      thought, content = raw.rsplit("</think>", 1)
    else:
      thought, content = ("", raw) if args.thinking == "off" else (raw, "")
    finished = bool(content.strip())
    pred = extract(content) or extract(thought)
    ok = norm(pred) == norm(gold)
    correct += ok
    row = {"i": i, "gold": gold, "pred": pred, "ok": ok, "finished": finished,
           "n_text": len(content), "n_think_chars": len(thought)}
    results.append(row)
    if args.jsonl:
      with open(args.jsonl, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"[{i}] {'OK ' if ok else 'NG '} gold={gold} pred={pred} finished={finished} "
          f"think_chars={len(thought)} ({correct}/{i - args.start + 1}) {time.time()-t0:.0f}s", flush=True)
  dt = time.time() - t0
  summary = {"tag": f"bf16-mps-thinking-{args.thinking}", "n": len(qs), "correct": correct,
             "acc": correct / len(qs), "seconds": round(dt, 1), "max_tokens": args.max_tokens}
  print(json.dumps(summary))
  if args.json_out:
    json.dump({"summary": summary, "results": results}, open(args.json_out, "w"), indent=1)


if __name__ == "__main__":
  main()
