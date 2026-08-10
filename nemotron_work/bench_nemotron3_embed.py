#!/usr/bin/env python3
"""Mac CPU/XNNPACK latency for the Nemotron-3-Embed-1B LiteRT artifacts.

    python bench_nemotron3_embed.py <out_dir> [--variants wi8fc,fp16]

Serialized, one variant at a time — parallel runs contaminate the numbers.
"""
import argparse
import os
import statistics
import sys
import time

import numpy as np

PAD_ID = 11


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--variants", default="wi8fc,fp16")
    ap.add_argument("--seqs", default="64,128,256,512")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    args = ap.parse_args()

    from ai_edge_litert.interpreter import Interpreter

    print(f"threads={args.threads} reps={args.reps}")
    for v in args.variants.split(","):
        path = os.path.join(args.out_dir, f"embed_{v}.tflite")
        if not os.path.exists(path):
            print(f"(missing {path})")
            continue
        it = Interpreter(model_path=path, num_threads=args.threads)
        print(f"\n{v}  ({os.path.getsize(path)/1e6:.0f} MB)")
        for S in [int(s) for s in args.seqs.split(",")]:
            try:
                r = it.get_signature_runner(f"embed_{S}")
            except Exception:
                continue
            n = int(S * 0.75)
            x = np.full((1, S), PAD_ID, np.int32)
            m = np.zeros((1, S), np.int32)
            x[0, :n] = np.random.default_rng(0).integers(10, 120000, n)
            m[0, :n] = 1
            r(input_ids=x, attention_mask=m)  # warm up / pack
            ts = []
            for _ in range(args.reps):
                t0 = time.perf_counter()
                r(input_ids=x, attention_mask=m)
                ts.append((time.perf_counter() - t0) * 1000)
            med = statistics.median(ts)
            print(f"  embed_{S:<4d} {med:7.1f} ms  (min {min(ts):.1f})  "
                  f"{n} valid tokens -> {n/med*1000:.0f} tok/s")


if __name__ == "__main__":
    main()
