#!/usr/bin/env python3
"""GSM8K on a .litertlm through the litert_lm 0.17 python API, with a backend choice.

Same questions / COT suffix / extract() / norm() as minicpm5_work/eval_gsm8k_api.py
(imported from it) so the row is comparable with spark_work/gsm8k_bf16.py. Thinking on
(the bundle's template pre-opens <think>; `enable_thinking` is passed explicitly), greedy
(top_k=1), one conversation per question. Records whether the thought channel closed
(a cap hit mid-think is a truncation, not a wrong answer).

  ~/venvs/lt0170run/bin/python spark_work/eval_gsm8k_engine.py --model out/x/model.litertlm \
      --backend gpu --n 100 --max-tokens 3584 --json-out spark_work/gsm8k_x.json
"""
import argparse
import importlib.util
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
_spec = importlib.util.spec_from_file_location(
    "eval_gsm8k_api", os.path.join(ROOT, "minicpm5_work", "eval_gsm8k_api.py"))
_api = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_api)
extract, _norm, load_q, COT = _api.extract, _api.norm, _api.load_q, _api.COT


def norm(x):
  """eval_gsm8k_api.norm() does int(float(x)); a prediction such as '1e999' parses
  to inf and raised OverflowError, killing a 30-minute run with no JSON written
  (2026-09-07, 1.7B int4 row). Anything float() accepts but int() cannot is simply
  a wrong answer."""
  try:
    return _norm(x)
  except (OverflowError, ValueError):
    return str(x).strip()


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", required=True)
  ap.add_argument("--backend", default="cpu", choices=["cpu", "gpu"])
  ap.add_argument("--n", type=int, default=100)
  ap.add_argument("--max-tokens", type=int, default=3584)
  ap.add_argument("--max-num-tokens", type=int, default=4096)
  ap.add_argument("--json-out", required=True)
  ap.add_argument("--resume", action="store_true")
  ap.add_argument("--ids", default=None,
                  help="comma-separated question indices (subset run, e.g. the losses of another row)")
  args = ap.parse_args()

  from litert_lm import engine as engine_lib
  from litert_lm import interfaces

  be = {"cpu": interfaces.Backend.CPU(), "gpu": interfaces.Backend.GPU()}[args.backend]
  eng = engine_lib.Engine(args.model, backend=be, max_num_tokens=args.max_num_tokens)
  qs = load_q(args.n)
  ids = [int(x) for x in args.ids.split(',')] if args.ids else list(range(len(qs)))
  correct, unfinished, rows = 0, 0, []
  if args.resume and os.path.exists(args.json_out):
    prev = json.load(open(args.json_out))
    rows = prev.get("rows", [])
    correct = sum(1 for r in rows if r.get("ok"))
    unfinished = sum(1 for r in rows if r.get("finished") is False)
    print(f"resuming from {len(rows)} rows ({correct} correct)", flush=True)

  def save(complete):
    summary = {"model": args.model, "backend": args.backend, "n": len(ids), "ids": ids, "done": len(rows),
               "correct": correct, "acc": correct / len(ids), "unfinished": unfinished,
               "max_tokens": args.max_tokens, "elapsed_s": round(time.time() - t0),
               "runtime": "litert-lm 0.17.0 python API", "complete": complete, "rows": rows}
    tmp = args.json_out + ".tmp"
    json.dump(summary, open(tmp, "w"), indent=2)
    os.replace(tmp, args.json_out)

  t0 = time.time()
  for i in ids:
    q, gold = qs[i]
    if any(r.get("i") == i for r in rows):
      continue
    conv = eng.create_conversation(
        sampler_config=interfaces.SamplerConfig(top_k=1),
        thinking_config=interfaces.ThinkingConfig(enable_thinking=True),
    )
    try:
      resp = conv.send_message(q + COT, max_output_tokens=args.max_tokens)
    except Exception as e:  # noqa: BLE001
      print(f"[{i}] ERROR {str(e)[:200]}", flush=True)
      rows.append({"i": i, "gold": gold, "pred": None, "ok": False, "finished": True, "error": str(e)[:300]})
      conv.close()
      save(complete=False)
      continue
    content = ""
    parts = resp.get("content") if isinstance(resp, dict) else None
    if isinstance(parts, str):
      content = parts
    elif isinstance(parts, list):
      content = "".join(c.get("text", "") for c in parts if isinstance(c, dict) and c.get("type") == "text")
    chans = resp.get("channels", {}) if isinstance(resp, dict) else {}
    thought = chans.get("thought", "") if isinstance(chans, dict) else ""
    conv.close()
    finished = bool(content.strip())  # an unclosed think block yields no answer text
    pred = extract(content) or extract(thought)
    ok = norm(pred) == norm(gold)
    correct += ok
    unfinished += (not finished)
    rows.append({"i": i, "gold": gold, "pred": pred, "ok": ok, "finished": finished,
                 "n_text": len(content), "n_think_chars": len(thought), "content_tail": content[-160:]})
    print(f"[{i}] {'ok' if ok else 'NG'} gold={gold} pred={pred} finished={finished} "
          f"think_chars={len(thought)} acc={correct}/{len(rows)+1} ({time.time()-t0:.0f}s)", flush=True)
    save(complete=False)
  save(complete=True)
  print(f"GSM8K {args.model} [{args.backend}]: {correct}/{len(ids)} = {correct/len(ids):.3f} "
        f"unfinished={unfinished} -> {args.json_out}")


if __name__ == "__main__":
  main()
