#!/usr/bin/env python3
"""Does speculative decoding commit the turn-final stop token?

Runs the same 5-turn greedy conversation with the flag off and on and prints,
after every turn, the engine's committed decode-token count, the size of that
turn's prefill, and the conversation's total token_count.
litert-lm 0.16.0 python API, CPU backend.  Usage: python3 probe.py <bundle>"""
import sys
import litert_lm

TURNS = [
    "Reply with only the single word MANGO.",
    "Reply with only the single word KIWI.",
    "What was the first word you replied with in this conversation? "
    "Reply with only that word.",
    "Repeat your previous reply exactly.",
    "Reply with only the single word LEMON.",
]

def text_of(stream):
    return "".join(i.get("text", "") for c in stream
                   for i in c.get("content", []) if i.get("type") == "text")

for spec in (False, True):
    print(f"--- enable_speculative_decoding={spec}", flush=True)
    eng = litert_lm.Engine(sys.argv[1], backend=litert_lm.Backend.CPU(),
                           enable_speculative_decoding=spec,
                           max_num_tokens=2048, enable_benchmark=True)
    sc = litert_lm.SamplerConfig(top_k=1, temperature=0.0, seed=0)
    with eng as engine, engine.create_conversation(sampler_config=sc) as conv:
        for prompt in TURNS:
            text = text_of(conv.send_message_async(prompt))
            info = conv.get_benchmark_info()
            print(f"{text[:20]!r:24} decode_committed={info.last_decode_token_count:<3}"
                  f" prefill={info.last_prefill_token_count:<3}"
                  f" token_count={conv.token_count}", flush=True)
