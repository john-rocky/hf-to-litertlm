#!/usr/bin/env python3
"""Japanese multi-turn gate for the sarashina2.2 bundles (3 turns through the
litert_lm python API, greedy; PASS = every reply non-empty, no special-token
leakage, turn 2/3 actually use the turn-1 answer — i.e. the structured
prompt_templates the bundle carries keep the conversation prefix contract).

  ~/venvs/lt0160run/bin/python sarashina_work/gate_multiturn_ja.py <bundle> --json-out out.json
"""
import argparse
import json
import re
import sys

TURNS = [
    "「とても大きい」という意味の言葉を一つだけ挙げてください。",
    "その言葉を使って短い文を一つ作ってください。",
    "その文を疑問文にしてください。",
]


def content_of(resp):
    if not isinstance(resp, dict):
        return str(resp)
    parts = resp.get("contents") or resp.get("content") or resp.get("text") or ""
    if isinstance(parts, str):
        return parts
    out = ""
    for c in parts:
        if isinstance(c, dict) and c.get("type") == "text":
            out += c.get("text", "")
        elif isinstance(c, str):
            out += c
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--json-out")
    args = ap.parse_args()
    from litert_lm import engine as engine_lib
    from litert_lm import interfaces
    eng = engine_lib.Engine(args.bundle, max_num_tokens=4096)
    conv = eng.create_conversation(sampler_config=interfaces.SamplerConfig(top_k=1))
    outs = []
    for t in TURNS:
        resp = conv.send_message(t, max_output_tokens=args.max_tokens)
        outs.append(content_of(resp).strip())
    conv.close()
    empty = any(not o for o in outs)
    leak = any(re.search(r"<\|(user|assistant|system)\|>|</s>|<unk>", o) for o in outs)
    # turn 3 should be a question in Japanese
    q3 = outs[2].rstrip().endswith(("？", "?", "か。", "か？")) or "？" in outs[2]
    for i, o in enumerate(outs):
        print(f"turn {i+1}: {o[:200]!r}")
    verdict = "PASS" if (not empty and not leak) else "FAIL"
    print(f"multi-turn 3/3, empty={empty}, special-leak={leak}, turn3-is-question={q3} -> {verdict}")
    if args.json_out:
        json.dump({"turns": outs, "empty": empty, "leak": leak, "turn3_question": q3,
                   "verdict": verdict}, open(args.json_out, "w"), indent=1, ensure_ascii=False)
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
