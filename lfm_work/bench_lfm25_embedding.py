#!/usr/bin/env python3
"""Mac-CPU latency of the LFM2.5-Embedding-350M LiteRT exports (XNNPACK, all threads).

    .venv-092/bin/python scripts/bench_lfm25_embedding.py lfm25_embed_work/out_350m

Run on a QUIET machine — a background convert/verify job moves these numbers by
more than the difference between variants, and card numbers are quoted from
this output.
"""
import json
import os
import sys
import time

import numpy as np

OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "out_lfm25_embed_350m"
SEQS = (64, 128, 256, 512)


def bench(path, S, iters=20):
    from ai_edge_litert.interpreter import Interpreter

    it = Interpreter(model_path=path, num_threads=os.cpu_count())
    r = it.get_signature_runner(f"embed_{S}")
    ids = np.random.randint(10, 65000, (1, S)).astype(np.int32)
    ids[0, 0] = 1
    mask = np.ones((1, S), dtype=np.int32)
    for _ in range(3):
        r(input_ids=ids, attention_mask=mask)
    t0 = time.perf_counter()
    for _ in range(iters):
        r(input_ids=ids, attention_mask=mask)
    return (time.perf_counter() - t0) / iters * 1000


def main():
    out = {}
    for kind in ("wi8fc", "fp16", "fp32"):
        path = os.path.join(OUT_DIR, f"embed_{kind}.tflite")
        if not os.path.exists(path):
            continue
        out[kind] = {}
        for S in SEQS:
            ms = bench(path, S)
            out[kind][f"embed_{S}"] = round(ms, 1)
            print(f"{kind} embed_{S}: {ms:.1f} ms  ({S/ms*1000:.0f} tok/s)",
                  flush=True)
    dst = os.path.join(OUT_DIR, "bench_mac.json")
    with open(dst, "w") as f:
        json.dump({"threads": os.cpu_count(), "ms": out}, f, indent=2)
    print("wrote", dst)


if __name__ == "__main__":
    main()
