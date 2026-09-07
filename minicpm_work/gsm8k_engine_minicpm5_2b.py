#!/usr/bin/env python3
"""GSM8K on a MiniCPM5-2B .litertlm through the litert_lm 0.17 python API.

Same questions / COT suffix / extract() / norm() as minicpm_work/eval_gsm8k_api.py
(imported from it), greedy (top_k=1), one conversation per question, resumable JSON.
The bundle carries the official jinja verbatim, so the thinking mode is a TEMPLATE knob:
  --thinking off   -> enable_thinking=false -> '<think>\n\n</think>\n\n' prefill (official card protocol)
  --thinking on    -> enable_thinking=true  -> '<think>\n' prefill
  --thinking unset -> bare assistant prompt (the model's own default)
--toggle chooses how the knob reaches the template: thinking_config (ThinkingConfig; conversation.cc
injects `enable_thinking` into the render context) or extra_context (the 0.14-era path).

  ~/venvs/lt0170run/bin/python minicpm_work/gsm8k_engine.py --model X.litertlm --backend cpu \
      --thinking off --n 100 --max-tokens 2048 --json-out minicpm_work/gsm8k_int4_off_cpu.json
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


def conv_kwargs(interfaces, thinking, toggle):
  if thinking == "unset":
    return {}
  flag = thinking == "on"
  if toggle == "thinking_config":
    return {"thinking_config": interfaces.ThinkingConfig(enable_thinking=flag)}
  return {"extra_context": {"enable_thinking": flag}}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", required=True)
  ap.add_argument("--backend", default="cpu", choices=["cpu", "gpu"])
  ap.add_argument("--thinking", default="off", choices=["on", "off", "unset"])
  ap.add_argument("--toggle", default="thinking_config", choices=["thinking_config", "extra_context"])
  ap.add_argument("--n", type=int, default=100)
  ap.add_argument("--max-tokens", type=int, default=2048)
  ap.add_argument("--max-num-tokens", type=int, default=4096)
  ap.add_argument("--json-out", required=True)
  ap.add_argument("--resume", action="store_true")
  ap.add_argument("--ids", default=None)
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
    summary = {"model": args.model, "backend": args.backend, "thinking": args.thinking, "toggle": args.toggle,
               "n": len(ids), "ids": ids, "done": len(rows), "correct": correct, "acc": correct / len(ids),
               "unfinished": unfinished, "max_tokens": args.max_tokens, "elapsed_s": round(time.time() - t0),
               "runtime": "litert-lm 0.17.0 python API", "complete": complete, "rows": rows}
    tmp = args.json_out + ".tmp"
    json.dump(summary, open(tmp, "w"), indent=2)
    os.replace(tmp, args.json_out)

  t0 = time.time()
  for i in ids:
    q, gold = qs[i]
    if any(r.get("i") == i for r in rows):
      continue
    conv = eng.create_conversation(sampler_config=interfaces.SamplerConfig(top_k=1),
                                   **conv_kwargs(interfaces, args.thinking, args.toggle))
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
    finished = bool(content.strip())
    pred = extract(content) or extract(thought)
    ok = norm(pred) == norm(gold)
    correct += ok
    unfinished += (not finished)
    rows.append({"i": i, "gold": gold, "pred": pred, "ok": ok, "finished": finished,
                 "n_text": len(content), "n_think_chars": len(thought), "content_tail": content[-160:]})
    print(f"[{i}] {'ok' if ok else 'NG'} gold={gold} pred={pred} finished={finished} "
          f"think_chars={len(thought)} acc={correct}/{len(rows)} ({time.time()-t0:.0f}s)", flush=True)
    save(complete=False)
  save(complete=True)
  print(f"GSM8K {args.model} [{args.backend}, thinking={args.thinking}]: {correct}/{len(ids)} = {correct/len(ids):.3f} "
        f"unfinished={unfinished} -> {args.json_out}")


if __name__ == "__main__":
  main()
