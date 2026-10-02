#!/usr/bin/env python3
"""Runtime worker for r3_stream.py (runs in ~/venvs/lt0171run = litert-lm-api 0.17.1, pure python).

One Engine for the process; per request (one JSON line on stdin: wav, prefix, max_output_tokens) a new Conversation
(greedy) and one user message [AudioFile(wav)] (+ [Text(prefix)] when the prefix is not empty). Replies on stdout as
'R3JSON {...}' lines (the runtime logs to stderr): reply text, the runtime's render of the message, seconds of
send_message, token_count.
"""
import argparse
import json
import sys
import time


def response_text(resp):
    # litert-lm-api 0.15-0.17.1 return a dict of content parts; later builds return a Message whose str() is its text
    if type(resp) is dict:
        return "".join(p.get("text", "") for p in resp.get("content", []) if isinstance(p, dict))
    return str(resp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    import os
    import litert_lm
    from litert_lm import interfaces
    os.makedirs(args.cache_dir, exist_ok=True)
    t0 = time.time()
    eng = litert_lm.Engine(args.bundle, backend=interfaces.CPU(thread_count=args.threads),
                           audio_backend=interfaces.CPU(thread_count=args.threads), cache_dir=args.cache_dir)
    sampler = interfaces.SamplerConfig(top_k=1, top_p=1.0, temperature=0.0)
    print("R3JSON " + json.dumps({"ready": True, "load_seconds": round(time.time() - t0, 3),
                                  "litert_lm": litert_lm.__file__}), flush=True)
    for line in sys.stdin:
        if not line.strip():
            continue
        req = json.loads(line)
        try:
            conv = eng.create_conversation(sampler_config=sampler, max_output_tokens=req["max_output_tokens"])
            items = [litert_lm.Content.AudioFile(req["wav"])]
            if req["prefix"]:
                items.append(litert_lm.Content.Text(req["prefix"]))
            msg = litert_lm.Message.user(litert_lm.Contents.of(*items))
            render = conv.render_message_to_string(msg)
            t1 = time.time()
            resp = conv.send_message(msg)
            dt = time.time() - t1
            out = {"text": response_text(resp), "render": render, "seconds": dt, "token_count": conv.token_count}
            conv.close()
        except Exception as e:  # noqa: BLE001
            out = {"error": f"{type(e).__name__}: {e}"}
        print("R3JSON " + json.dumps(out, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
