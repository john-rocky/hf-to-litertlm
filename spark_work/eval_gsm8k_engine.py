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
extract, norm, load_q, COT = _api.extract, _api.norm, _api.load_q, _api.COT


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", required=True)
  ap.add_argument("--backend", default="cpu", choices=["cpu", "gpu"])
  ap.add_argument("--n", type=int, default=100)
  ap.add_argument("--max-tokens", type=int, default=3584)
  ap.add_argument("--max-num-tokens", type=int, default=4096)
  ap.add_argument("--json-out", required=True)
  args = ap.parse_args()

  from litert_lm import engine as engine_lib
  from litert_lm import interfaces

  be = {"cpu": interfaces.Backend.CPU(), "gpu": interfaces.Backend.GPU()}[args.backend]
  eng = engine_lib.Engine(args.model, backend=be, max_num_tokens=args.max_num_tokens)
  qs = load_q(args.n)
  correct, unfinished, rows = 0, 0, []
  t0 = time.time()
  for i, (q, gold) in enumerate(qs):
    conv = eng.create_conversation(
        sampler_config=interfaces.SamplerConfig(top_k=1),
        thinking_config=interfaces.ThinkingConfig(enable_thinking=True),
    )
    try:
      resp = conv.send_message(q + COT, max_output_tokens=args.max_tokens)
    except Exception as e:  # noqa: BLE001
      print(f"[{i}] ERROR {str(e)[:200]}", flush=True)
      rows.append({"i": i, "gold": gold, "pred": None, "ok": False, "error": str(e)[:300]})
      conv.close()
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
          f"think_chars={len(thought)} acc={correct}/{i+1} ({time.time()-t0:.0f}s)", flush=True)
  summary = {"model": args.model, "backend": args.backend, "n": len(qs), "correct": correct,
             "acc": correct / len(qs), "unfinished": unfinished, "max_tokens": args.max_tokens,
             "elapsed_s": round(time.time() - t0), "runtime": "litert-lm 0.17.0 python API", "rows": rows}
  json.dump(summary, open(args.json_out, "w"), indent=2)
  print(f"GSM8K {args.model} [{args.backend}]: {correct}/{len(qs)} = {correct/len(qs):.3f} "
        f"unfinished={unfinished} -> {args.json_out}")


if __name__ == "__main__":
  main()
