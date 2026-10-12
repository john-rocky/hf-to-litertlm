#!/usr/bin/env python3
"""Tool-call probe: ONE fresh process, one Engine, one Conversation, one prompt (litert-lm 0.18.0, CPU).

Arms (--arm):
  bundle      the bundle's own template and llm_model_type as published
  canonical   the bundle as published + the LiteRT-LM models/qwen3_5 chat template passed through the
              Python API's per-conversation `chat_template` override (model type stays whatever the bundle says)
  workaround  a re-packed copy (metadata only) whose llm_model_type / template were replaced; nothing
              is overridden at run time
Records the raw assistant text, the runtime-parsed tool_calls, whether the Python tool function actually ran
(automatic_tool_calling=True) and its arguments. Greedy (top_k=1).
"""
import argparse, json, os, sys, time, traceback

import litert_lm
from litert_lm import interfaces

TOOL_LOG = []


def get_weather(location: str) -> str:
  """Get the current weather for a city.

  Args:
    location: The city name, for example "Paris".
  """
  TOOL_LOG.append({"name": "get_weather", "args": {"location": location}, "t": time.time()})
  return json.dumps({"location": location, "temperature_c": 18, "condition": "cloudy"})


def part_to_json(p):
  if hasattr(p, "to_json"):
    try:
      return p.to_json()
    except Exception:  # noqa: BLE001
      pass
  return str(p)


def msg_to_dict(msg):
  d = {}
  try:
    d["contents"] = [part_to_json(p) for p in msg.contents]
  except Exception:  # noqa: BLE001
    d["contents"] = str(getattr(msg, "contents", msg))[:2000]
  try:
    d["tool_calls"] = [part_to_json(tc) for tc in msg.tool_calls]
  except Exception:  # noqa: BLE001
    d["tool_calls"] = str(getattr(msg, "tool_calls", None))[:600]
  try:
    d["channels"] = {k: v[:1200] for k, v in msg.channels.items()}
  except Exception:  # noqa: BLE001
    d["channels"] = str(getattr(msg, "channels", None))[:600]
  d["text"] = "".join(
      (c.get("text", "") if isinstance(c, dict) else "") for c in d["contents"] if isinstance(c, dict))
  return d


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", required=True)
  ap.add_argument("--label", required=True)
  ap.add_argument("--arm", choices=["bundle", "canonical", "workaround"], required=True)
  ap.add_argument("--template", default=None, help="jinja file for --arm canonical")
  ap.add_argument("--automatic", type=int, choices=[0, 1], default=1)
  ap.add_argument("--thinking", choices=["unset", "off", "on"], default="unset")
  ap.add_argument("--prompt", default="What is the weather in Paris right now?")
  ap.add_argument("--max-output-tokens", type=int, default=256)
  ap.add_argument("--max-num-tokens", type=int, default=1024)
  ap.add_argument("--run", type=int, default=1)
  ap.add_argument("--out", required=True)
  a = ap.parse_args()

  rec = {"label": a.label, "arm": a.arm, "model": a.model, "model_bytes": os.path.getsize(a.model),
         "runtime": "litert-lm 0.18.0 (python, cpu, fresh process)", "automatic_tool_calling": bool(a.automatic),
         "thinking": a.thinking, "prompt": a.prompt, "run": a.run, "pid": os.getpid(),
         "template_override": a.template, "sampler": "top_k=1"}
  t0 = time.time()
  try:
    engine = litert_lm.Engine(a.model, backend=interfaces.Backend.CPU(), max_num_tokens=a.max_num_tokens)
    rec["load_s"] = round(time.time() - t0, 1)
    cc = {"tools": [get_weather], "automatic_tool_calling": bool(a.automatic),
          "sampler_config": interfaces.SamplerConfig(top_k=1)}
    if a.arm == "canonical":
      cc["chat_template"] = open(a.template, encoding="utf-8").read()
    if a.thinking == "off":
      cc["thinking_config"] = interfaces.ThinkingConfig(enable_thinking=False)
    elif a.thinking == "on":
      cc["thinking_config"] = interfaces.ThinkingConfig(enable_thinking=True)
    conv = engine.create_conversation(**cc)
    t1 = time.time()
    msg = conv.send_message(a.prompt, max_output_tokens=a.max_output_tokens)
    rec["gen_s"] = round(time.time() - t1, 1)
    rec["message"] = msg_to_dict(msg)
    try:
      conv.close()
    except Exception:  # noqa: BLE001
      pass
    engine.close()
  except Exception as e:  # noqa: BLE001
    rec["error"] = f"{type(e).__name__}: {e}"[:1500]
    rec["traceback"] = traceback.format_exc()[-2000:]
  rec["tool_executed"] = bool(TOOL_LOG)
  rec["tool_log"] = TOOL_LOG
  rec["elapsed_s"] = round(time.time() - t0, 1)
  os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
  with open(a.out, "w", encoding="utf-8") as f:
    json.dump(rec, f, ensure_ascii=False, indent=1)
  print("RESULT " + json.dumps({k: rec.get(k) for k in ("label", "arm", "automatic_tool_calling", "thinking", "run",
                                                       "tool_executed", "error")}, ensure_ascii=False), flush=True)
  m = rec.get("message", {})
  print("TEXT " + json.dumps(m.get("text", "")[:600], ensure_ascii=False), flush=True)
  print("TOOL_CALLS " + json.dumps(m.get("tool_calls", []), ensure_ascii=False)[:600], flush=True)
  return 0


if __name__ == "__main__":
  sys.exit(main())
