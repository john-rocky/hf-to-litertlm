#!/usr/bin/env python3
"""GSM8K bf16 reference for Spark-X2.5 -- the parity baseline row.

Protocol matched to the engine rows (minicpm5_work/eval_gsm8k_api.py): same
questions (evaldata/gsm8k_test.jsonl, first n), same COT suffix, same extract()
and norm(), greedy, max-new-tokens 2048.
Template = the bundle's own templates/spark25_think.jinja (byte-identical to the
vendor template for this shape), rendered with jinja2.
Content vs thinking split at the LAST </think>; extraction falls back to the
thinking text exactly like the engine harness.

  ~/venvs/ltconv040dev/bin/python spark_work/gsm8k_bf16.py --hf src_models/Spark-X2.5-1.7B \
      --n 100 --json-out spark_work/gsm8k_bf16_1p7b.json [--resume]

Memory (measured 2026-09-07, user-reported twice): RSS lies on MPS — read
`vmmap --summary <pid>` "Physical footprint". (1) The MPS caching allocator keeps
every freed block, so a 3584-token generate per question grew the 1.7B run to
9.1 GB RSS / swap 8 of 9 GB and stalled the terminal → del + gc.collect() +
torch.mps.empty_cache() after every question. (2) That only bounds the STEADY
state: inside one question, generate's DynamicCache re-allocates the whole KV
cache with torch.cat on every step (O(T^2) churn) and the 4B run peaked at a
130.1 GB physical footprint (17.5 GB steady). Fixes: `cache_implementation=
"static"` (pre-allocated KV, no per-step cat) and PYTORCH_MPS_HIGH_WATERMARK_RATIO
(set before torch import; default here 0.3 ≈ 40 GB on a 137 GB machine) so the
allocator releases cached blocks instead of growing. The JSON is rewritten after
every question ("complete": false) and --resume continues from it.
"""
import argparse
import gc
import importlib.util
import json
import os
import sys
import time

# cap the MPS caching allocator BEFORE torch is imported (see the docstring)
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.3")
# torch refuses a low watermark above the high one ("invalid low watermark ratio 1.4")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.25")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

_spec = importlib.util.spec_from_file_location(
    "eval_gsm8k_api", os.path.join(ROOT, "minicpm5_work", "eval_gsm8k_api.py"))
_api = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_api)
extract, norm, load_q, COT = _api.extract, _api.norm, _api.load_q, _api.COT


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--hf", required=True)
  ap.add_argument("--n", type=int, default=100)
  ap.add_argument("--max-tokens", type=int, default=2048)
  ap.add_argument("--dtype", default="bfloat16")
  ap.add_argument("--device", default="mps")
  ap.add_argument("--attn", default="sdpa")
  ap.add_argument("--jinja", default=os.path.join(ROOT, "templates", "spark25_think.jinja"))
  ap.add_argument("--json-out", required=True)
  ap.add_argument("--resume", action="store_true")
  args = ap.parse_args()

  import torch
  import transformers
  from jinja2 import Environment

  tpl = Environment().from_string(open(args.jinja, encoding="utf-8").read())
  tok = transformers.AutoTokenizer.from_pretrained(args.hf)
  model = transformers.AutoModelForCausalLM.from_pretrained(
      args.hf, dtype=getattr(torch, args.dtype), trust_remote_code=True,
      attn_implementation=args.attn).to(args.device).eval()

  qs = load_q(args.n)
  rows, correct = [], 0
  if args.resume and os.path.exists(args.json_out):
    prev = json.load(open(args.json_out))
    rows = prev.get("rows", [])
    correct = sum(1 for r in rows if r.get("ok"))
    print(f"resuming from {len(rows)} rows ({correct} correct)", flush=True)

  def save(complete):
    summary = {"hf": args.hf, "n": len(qs), "done": len(rows), "correct": correct,
               "acc": correct / len(qs), "max_tokens": args.max_tokens, "dtype": args.dtype,
               "attn": args.attn, "jinja": args.jinja, "elapsed_s": round(time.time() - t0),
               "complete": complete, "rows": rows}
    tmp = args.json_out + ".tmp"
    json.dump(summary, open(tmp, "w"), indent=2)
    os.replace(tmp, args.json_out)

  t0 = time.time()
  for i, (q, gold) in enumerate(qs):
    if i < len(rows):
      continue
    text = tpl.render(messages=[{"role": "user", "content": q + COT}], add_generation_prompt=True)
    ids = tok(text, add_special_tokens=False, return_tensors="pt").to(args.device)
    with torch.no_grad():
      out = model.generate(**ids, max_new_tokens=args.max_tokens, do_sample=False,
                           cache_implementation="static")
    gen = out[0][ids["input_ids"].shape[1]:]
    raw = tok.decode(gen, skip_special_tokens=False).replace("<｜end▁of▁sentence｜>", "")
    if "</think>" in raw:
      think, content = raw.rsplit("</think>", 1)
    else:
      think, content = raw, ""
    pred = extract(content) or extract(think)
    ok = norm(pred) == norm(gold)
    correct += ok
    rows.append({"i": i, "gold": gold, "pred": pred, "ok": ok, "n_gen": int(gen.shape[0]),
                 "finished": "</think>" in raw, "n_think_chars": len(think),
                 "content_tail": content[-200:]})
    print(f"[{i}] {'ok' if ok else 'NG'} gold={gold} pred={pred} n_gen={gen.shape[0]} "
          f"finished={'</think>' in raw} acc={correct}/{i+1} ({time.time()-t0:.0f}s)", flush=True)
    # release the question's KV cache / activations back to the OS (MPS caching allocator)
    del out, gen, ids
    gc.collect()
    if args.device == "mps":
      torch.mps.synchronize()
      torch.mps.empty_cache()
    save(complete=False)
  save(complete=True)
  print(f"GSM8K bf16 {args.hf}: {correct}/{len(qs)} = {correct/len(qs):.3f} -> {args.json_out}")


if __name__ == "__main__":
  main()
