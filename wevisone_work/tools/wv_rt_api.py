#!/usr/bin/env python3
"""One engine, one conversation, the same calls the litert-lm 0.18.1 CLI makes (litert_lm_cli/commands/run.py:
Engine(model, backend, vision_backend, cache_dir), create_conversation(sampler_config=SamplerConfig(temperature=0,
seed=0)), user messages as content lists with the attachment first), for what the one-shot CLI cannot do:

  --turns-json '["My name is Ada.", "What is my name?"]'   several user turns on one conversation (answers -> stdout)
  --timing --image PNG --prompt Q [--repeat 3]            one image turn, timed: wall clock from send to the first
                                                         streamed chunk and to the end, chunk count, the engine's
                                                         benchmark info (prefill / decode turns as the C API reports)
                                                         and a second turn on the same image (state lever)
Prints one JSON line last.

    ~/venvs/lt0181run/bin/python3 -I tools/wv_rt_api.py --bundle B --backend cpu|gpu [--vision-backend cpu|gpu] ...
"""
import argparse
import json
import os
import sys
import time

import litert_lm


def backend_of(name, threads=None):
    if name == "gpu":
        return litert_lm.Backend.GPU()
    return litert_lm.Backend.CPU(thread_count=threads)


def message(prompt, image=None):
    content = []
    if image:
        content.append({"type": "image", "path": os.path.abspath(image)})
    content.append({"type": "text", "text": prompt})
    return {"role": "user", "content": content}


def text_of(msg):
    parts = msg.get("content") if isinstance(msg, dict) else None
    if isinstance(parts, list):
        return "".join(p.get("text", "") for p in parts if isinstance(p, dict))
    return str(msg)


def bench_dict(conv):
    """Every prefill / decode turn through the C API (the Python BenchmarkInfo keeps the last turn only: memory
    litert-lm-benchmark-info-last-turn), plus the first-token time the C API reports."""
    lib = conv._lib
    try:
        ptr = lib.litert_lm_conversation_get_benchmark_info(conv._ptr)
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}
    if not ptr:
        return {"error": "no benchmark info"}
    try:
        n_p = lib.litert_lm_benchmark_info_get_num_prefill_turns(ptr)
        n_d = lib.litert_lm_benchmark_info_get_num_decode_turns(ptr)
        return {"prefill": [[lib.litert_lm_benchmark_info_get_prefill_token_count_at(ptr, i),
                             lib.litert_lm_benchmark_info_get_prefill_tokens_per_sec_at(ptr, i)] for i in range(n_p)],
                "decode": [[lib.litert_lm_benchmark_info_get_decode_token_count_at(ptr, i),
                            lib.litert_lm_benchmark_info_get_decode_tokens_per_sec_at(ptr, i)] for i in range(n_d)],
                "time_to_first_token_s": lib.litert_lm_benchmark_info_get_time_to_first_token(ptr)}
    finally:
        lib.litert_lm_benchmark_info_delete(ptr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--vision-backend", default=None)
    ap.add_argument("--cpu-thread-count", type=int, default=None)
    ap.add_argument("--turns-json", default=None)
    ap.add_argument("--timing", action="store_true")
    ap.add_argument("--image", default=None)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--followup", default="Answer again in the same format.")
    ap.add_argument("--max-num-tokens", type=int, default=None)
    a = ap.parse_args()
    sampler = litert_lm.SamplerConfig(temperature=0.0, seed=0)
    t0 = time.perf_counter()
    eng = litert_lm.Engine(a.bundle, backend=backend_of(a.backend, a.cpu_thread_count),
                           vision_backend=backend_of(a.vision_backend) if a.vision_backend else None,
                           max_num_tokens=a.max_num_tokens, cache_dir=":nocache", enable_benchmark=a.timing)
    init_s = time.perf_counter() - t0
    out = {"bundle": a.bundle, "backend": a.backend, "vision_backend": a.vision_backend, "init_s": round(init_s, 3)}
    with eng:
        with eng.create_conversation(sampler_config=sampler) as conv:
            if a.turns_json:
                answers = []
                for i, turn in enumerate(json.loads(a.turns_json)):
                    msg = message(turn, a.image if i == 0 else None)
                    answers.append(text_of(conv.send_message(msg)).strip())
                out["answers"] = answers
            if a.timing:
                turns = []
                for i, (prompt, img) in enumerate(((a.prompt, a.image), (a.followup, None))):
                    t1 = time.perf_counter()
                    first, n_chunks, text = None, 0, []
                    for chunk in conv.send_message_async(message(prompt, img)):
                        s = str(chunk)
                        if s and first is None:
                            first = time.perf_counter() - t1
                        if s:
                            n_chunks += 1
                            text.append(s)
                    total = time.perf_counter() - t1
                    turns.append({"turn": i + 1, "first_chunk_s": round(first or total, 4), "total_s": round(total, 4),
                                  "n_chunks": n_chunks, "text": "".join(text).strip(), "bench": bench_dict(conv),
                                  "token_count_after": conv.token_count})
                out["turns"] = turns
    print(json.dumps(out, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
