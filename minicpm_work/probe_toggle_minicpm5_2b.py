#!/usr/bin/env python3
"""Behavioural probe of a MiniCPM5-2B bundle through the litert_lm 0.17 python API.

Checks, on one backend:
  1. thinking toggle: unset / on / off (via ThinkingConfig and via extra_context) — does the
     thought channel fill or stay empty, does the answer arrive, how many chars each;
  2. multi-turn (3 user turns in ONE conversation) under unset and off — the runtime's
     prefix contract (render(history) must string-extend the previous render) either
     holds, or send_message raises / the answers desync. Errors are recorded, not hidden.

  ~/venvs/lt0170run/bin/python minicpm_work/probe_toggle.py X.litertlm --backend cpu --out probe.json
"""
import argparse
import json
import time

Q1 = "What is 17 + 25? Answer briefly."
TURNS = ["My name is Ken and I live in Osaka. Reply with one short sentence.",
         "What is 8 times 7? Answer briefly.",
         "Which city do I live in? Answer briefly."]


def text_of(resp):
  parts = resp.get("content") if isinstance(resp, dict) else None
  if isinstance(parts, str):
    return parts
  if isinstance(parts, list):
    return "".join(c.get("text", "") for c in parts if isinstance(c, dict) and c.get("type") == "text")
  return ""


def thought_of(resp):
  ch = resp.get("channels", {}) if isinstance(resp, dict) else {}
  return ch.get("thought", "") if isinstance(ch, dict) else ""


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("model")
  ap.add_argument("--backend", default="cpu", choices=["cpu", "gpu"])
  ap.add_argument("--max-tokens", type=int, default=2048)
  ap.add_argument("--out", required=True)
  args = ap.parse_args()
  from litert_lm import engine as engine_lib
  from litert_lm import interfaces
  be = {"cpu": interfaces.Backend.CPU(), "gpu": interfaces.Backend.GPU()}[args.backend]
  eng = engine_lib.Engine(args.model, backend=be, max_num_tokens=4096)
  out = {"model": args.model, "backend": args.backend, "single": [], "multi": []}

  def mk(mode, toggle):
    kw = {"sampler_config": interfaces.SamplerConfig(top_k=1)}
    if mode != "unset":
      flag = mode == "on"
      if toggle == "thinking_config":
        kw["thinking_config"] = interfaces.ThinkingConfig(enable_thinking=flag)
      else:
        kw["extra_context"] = {"enable_thinking": flag}
    return eng.create_conversation(**kw)

  for mode, toggle in [("unset", "-"), ("on", "thinking_config"), ("off", "thinking_config"),
                       ("on", "extra_context"), ("off", "extra_context")]:
    conv = mk(mode, toggle)
    t0 = time.time()
    try:
      r = conv.send_message(Q1, max_output_tokens=args.max_tokens)
      row = {"mode": mode, "toggle": toggle, "text": text_of(r)[:300], "n_text": len(text_of(r)),
             "think_chars": len(thought_of(r)), "thought_head": thought_of(r)[:160], "s": round(time.time() - t0, 1)}
    except Exception as e:  # noqa: BLE001
      row = {"mode": mode, "toggle": toggle, "error": str(e)[:400]}
    conv.close()
    out["single"].append(row)
    print("SINGLE", json.dumps(row, ensure_ascii=False)[:400], flush=True)

  for mode in ["unset", "off", "on"]:
    conv = mk(mode, "thinking_config")
    rows = []
    for t in TURNS:
      t0 = time.time()
      try:
        r = conv.send_message(t, max_output_tokens=args.max_tokens)
        rows.append({"q": t, "text": text_of(r)[:200], "think_chars": len(thought_of(r)), "s": round(time.time() - t0, 1)})
      except Exception as e:  # noqa: BLE001
        rows.append({"q": t, "error": str(e)[:400]})
        break
    conv.close()
    out["multi"].append({"mode": mode, "turns": rows})
    print("MULTI", mode, json.dumps(rows, ensure_ascii=False)[:700], flush=True)
  json.dump(out, open(args.out, "w"), indent=1, ensure_ascii=False)
  print("PROBE_DONE", args.out)


if __name__ == "__main__":
  main()
