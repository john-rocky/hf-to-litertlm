#!/usr/bin/env python3
"""Multi-turn gate for a Spark-X2.5 bundle (thinking model, plain attention).

One conversation, three turns (plant a fact / arithmetic / recall the fact) on
the litert_lm 0.17 python API. Catches the prefix-contract failure modes
(conversation.cc: "does not start with the previous rendered template string"
hard error, or a silent rewind) and any cross-turn state damage. Greedy.
Each turn gets a budget large enough for the model to CLOSE its think block. Runs the conversation twice: with the default channel
handling and with filter_channel_content_from_kv_cache=True (the runtime
re-prefills history from the template's assistant branch in that mode).

    ~/venvs/lt0170run/bin/python spark_work/multiturn_gate.py <model.litertlm> [cpu|gpu] [out.json]
"""
import json
import re
import sys
import time

from litert_lm import engine as engine_lib
from litert_lm import interfaces

MODEL = sys.argv[1]
BACKEND = sys.argv[2] if len(sys.argv) > 2 else "cpu"
OUT = sys.argv[3] if len(sys.argv) > 3 else None
MAXTOK = 3072

TURNS = [
    ("My cat is named Alfred and he is orange. Please remember that. Just say OK.", None),
    ("What is 23 + 19? Answer briefly.", r"\b42\b"),
    ("What is the name and color of my cat? Answer briefly.", r"alfred.*orange|orange.*alfred"),
]


def reply_text(resp):
    parts = resp.get("content") if isinstance(resp, dict) else None
    if isinstance(parts, str):
        return parts.strip()
    if isinstance(parts, list):
        return "".join(c.get("text", "") for c in parts
                       if isinstance(c, dict) and c.get("type") == "text").strip()
    return str(resp).strip()


def run_conversation(eng, filter_channels):
    kwargs = dict(sampler_config=interfaces.SamplerConfig(top_k=1),
                  thinking_config=interfaces.ThinkingConfig(enable_thinking=True))
    if filter_channels is not None:
        kwargs["filter_channel_content_from_kv_cache"] = filter_channels
    conv = eng.create_conversation(**kwargs)
    rows, ok_all = [], True
    for i, (msg, pat) in enumerate(TURNS, 1):
        t0 = time.time()
        try:
            resp = conv.send_message(msg, max_output_tokens=MAXTOK)
        except Exception as e:  # noqa: BLE001
            rows.append({"turn": i, "error": str(e)[:500]})
            print(f"  turn {i}: ERROR {str(e)[:300]}", flush=True)
            ok_all = False
            break
        text = reply_text(resp)
        channels = resp.get("channels") if isinstance(resp, dict) else None
        thought = (channels or {}).get("thought", "") if isinstance(channels, dict) else ""
        ok = True if pat is None else bool(re.search(pat, text, re.I))
        leak = "<think>" in text or "</think>" in text
        empty = len(text) == 0
        ok_all &= ok and not leak and not empty
        rows.append({"turn": i, "ok": ok, "leak": leak, "empty": empty,
                     "seconds": round(time.time() - t0, 1),
                     "n_thought_chars": len(thought), "text": text[:300]})
        print(f"  turn {i}: {'ok' if ok else 'NG'}{'/LEAK' if leak else ''}{'/EMPTY' if empty else ''} "
              f"thought_chars={len(thought)} {text[:100]!r}", flush=True)
    return ok_all, rows


_be = {"cpu": interfaces.Backend.CPU(), "gpu": interfaces.Backend.GPU()}[BACKEND]
eng = engine_lib.Engine(MODEL, backend=_be, max_num_tokens=4096)
report = {"model": MODEL, "backend": BACKEND, "runs": {}}
verdict = True
for name, flt in (("default", None), ("filter_channels", True)):
    print(f"== {name}", flush=True)
    ok, rows = run_conversation(eng, flt)
    report["runs"][name] = {"pass": ok, "rows": rows}
    verdict &= ok
report["verdict"] = "PASS" if verdict else "FAIL"
print(f"multi-turn verdict={report['verdict']} backend={BACKEND}")
if OUT:
    json.dump(report, open(OUT, "w"), indent=2, ensure_ascii=False)
sys.exit(0 if verdict else 1)
