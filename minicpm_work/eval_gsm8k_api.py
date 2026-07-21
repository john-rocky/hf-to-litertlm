#!/usr/bin/env python3
"""GSM8K eval via the litert_lm python API (direct, no server).

Supports the thinking toggle through Conversation extra_context
(enable_thinking), matching the official MiniCPM5-1B card's eval setup
("Thinking mode has been turned off"). Greedy by default (top_k=1).

  python eval_gsm8k_api.py --model <path.litertlm> \
      --n 100 --max-tokens 1024 --thinking off --tag NAME --json-out out.json
"""
import argparse
import json
import os
import re
import time

DATA = os.environ.get("GSM8K_JSONL", "evaldata/gsm8k_test.jsonl")
COT = (
    "\n\nSolve this step by step. After your reasoning, write the final answer "
    "on its own line in the exact form:\n#### <number>"
)


def norm(x):
    if x is None:
        return None
    try:
        f = float(x)
        return str(int(f)) if f == int(f) else repr(f)
    except ValueError:
        return x.strip()


def extract(text):
    if not text:
        return None
    t = text.replace(",", "")
    m = re.findall(r"####\s*\$?(-?\d+(?:\.\d+)?)", t)
    if m:
        return m[-1]
    m = re.findall(r"\\boxed\{\s*\$?(-?\d+(?:\.\d+)?)", t)
    if m:
        return m[-1]
    m = re.findall(r"(-?\d+(?:\.\d+)?)", t)
    return m[-1] if m else None


def load_q(n):
    out = []
    for line in open(DATA):
        d = json.loads(line)
        out.append((d["question"],
                    d["answer"].split("####")[-1].strip().replace(",", "")))
        if len(out) >= n:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--max-num-tokens", type=int, default=4096)
    ap.add_argument("--thinking", choices=["on", "off", "unset"], default="off")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    from litert_lm import engine as engine_lib
    from litert_lm import interfaces

    eng = engine_lib.Engine(args.model, max_num_tokens=args.max_num_tokens)
    extra = {}
    if args.thinking != "unset":
        extra["enable_thinking"] = args.thinking == "on"

    qs = load_q(args.n)
    correct, results = 0, []
    t0 = time.time()
    for i, (q, gold) in enumerate(qs):
        conv = eng.create_conversation(
            extra_context=extra or None,
            sampler_config=interfaces.SamplerConfig(top_k=1),
        )
        try:
            resp = conv.send_message(q + COT, max_output_tokens=args.max_tokens)
        except Exception as e:
            print(f"[{i}] ERROR {e}", flush=True)
            results.append({"i": i, "gold": gold, "pred": None, "error": str(e)})
            conv.close()
            continue
        content = ""
        for c in resp.get("contents", []) if isinstance(resp, dict) else []:
            if isinstance(c, dict) and c.get("type") == "text":
                content += c.get("text", "")
        if not content and isinstance(resp, dict):
            content = resp.get("content") or resp.get("text") or ""
            if not isinstance(content, str):
                content = json.dumps(content)
        chans = resp.get("channels", {}) if isinstance(resp, dict) else {}
        thought = chans.get("thought", "")
        conv.close()
        pred = extract(content) or extract(thought)
        ok = norm(pred) == norm(gold)
        correct += ok
        results.append({"i": i, "gold": gold, "pred": pred, "ok": ok,
                        "n_text": len(content), "n_think": len(thought)})
        print(f"[{i}] {'OK ' if ok else 'NG '} gold={gold} pred={pred} "
              f"({correct}/{i + 1})", flush=True)
    dt = time.time() - t0
    tag = args.tag or os.path.basename(args.model)
    summary = {"tag": tag, "n": len(qs), "correct": correct,
               "acc": correct / len(qs), "seconds": round(dt, 1),
               "max_tokens": args.max_tokens, "thinking": args.thinking}
    print(json.dumps(summary))
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"summary": summary, "results": results}, f, indent=1)


if __name__ == "__main__":
    main()
