#!/usr/bin/env python3
"""Digit test (hazard cpu-int8-runtime-activation-quant): ONE fresh process, one Engine (CPU), one Conversation, the
prompt "What is 17 times 23?" at the bundle's default rendering (no ThinkingConfig, no system prompt), greedy top_k=1,
max_output_tokens 400. Records the text, whether 391 appears, and the verify_quality.py degeneracy rule."""
import argparse, json, os, re, sys, time, traceback
from collections import Counter

import litert_lm
from litert_lm import interfaces


def degenerate(text):
  words = text.split()
  if len(words) >= 10:
    grams = [" ".join(words[i:i + 5]) for i in range(len(words) - 4)]
    if grams and Counter(grams).most_common(1)[0][1] >= 3:
      return True
    if len(set(words)) / len(words) < 0.30:
      return True
  if len(text) >= 40 and len(set(text)) < 15:
    return True
  if text.count("<|") >= 5 or text.count("<pad>") >= 5:
    return True
  return False


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", required=True)
  ap.add_argument("--label", required=True)
  ap.add_argument("--prompt", default="What is 17 times 23?")
  ap.add_argument("--max-output-tokens", type=int, default=400)
  ap.add_argument("--run", type=int, default=1)
  ap.add_argument("--out", required=True)
  a = ap.parse_args()
  rec = {"label": a.label, "model": a.model, "model_bytes": os.path.getsize(a.model), "prompt": a.prompt,
         "runtime": "litert-lm 0.18.0 (python, cpu, fresh process)", "sampler": "top_k=1", "thinking": "unset (bundle default)",
         "run": a.run, "pid": os.getpid(), "max_output_tokens": a.max_output_tokens}
  t0 = time.time()
  try:
    engine = litert_lm.Engine(a.model, backend=interfaces.Backend.CPU(), max_num_tokens=1024)
    rec["load_s"] = round(time.time() - t0, 1)
    conv = engine.create_conversation(sampler_config=interfaces.SamplerConfig(top_k=1))
    t1 = time.time()
    msg = conv.send_message(a.prompt, max_output_tokens=a.max_output_tokens)
    rec["gen_s"] = round(time.time() - t1, 1)
    text = "".join((p.to_json().get("text", "") if hasattr(p, "to_json") and isinstance(p.to_json(), dict) else "") for p in msg.contents)
    rec["text"] = text
    try:
      rec["channels"] = {k: v[:400] for k, v in msg.channels.items()}
    except Exception:  # noqa: BLE001
      rec["channels"] = {}
    rec["has_391"] = bool(re.search(r"\b391\b", text))
    rec["degenerate"] = degenerate(text)
    conv.close()
    engine.close()
  except Exception as e:  # noqa: BLE001
    rec["error"] = f"{type(e).__name__}: {e}"[:1500]
    rec["traceback"] = traceback.format_exc()[-2000:]
  rec["elapsed_s"] = round(time.time() - t0, 1)
  os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
  json.dump(rec, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
  print("RESULT " + json.dumps({k: rec.get(k) for k in ("label", "run", "has_391", "degenerate", "error")}), flush=True)
  print("TEXT " + json.dumps(rec.get("text", "")[:400], ensure_ascii=False), flush=True)
  return 0


if __name__ == "__main__":
  sys.exit(main())
