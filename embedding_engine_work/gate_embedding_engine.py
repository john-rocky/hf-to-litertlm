#!/usr/bin/env python3
"""Engine-side half of the EmbeddingEngine parity gate (run under the litert-lm >= 0.17.0 venv).

    ~/venvs/lt0170run/bin/python scripts/gate_embedding_engine.py <bundle.litertlm> <texts.json> <out.json>
        [--no-special-tokens] [--normalize/--no-normalize]

texts.json: {"texts": [...]}  ->  out.json: {"embeddings": [[...], ...], "dim": D, "options": {...}}
The tflite-side half (per lane) computes the same texts through the shipped .tflite with the HF
tokenizer, and a comparator reports per-text cosine. Kept model-agnostic on purpose.
"""
import argparse
import json
import sys
import time

import litert_lm
from litert_lm.embedding_engine import EmbeddingEngine, EmbeddingOptions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("texts")
    ap.add_argument("out")
    ap.add_argument("--no-special-tokens", action="store_true")
    ap.add_argument("--no-normalize", action="store_true")
    ap.add_argument("--backend", default="cpu", choices=["cpu", "gpu"])
    args = ap.parse_args()
    texts = json.load(open(args.texts))["texts"]
    backend = litert_lm.Backend.CPU() if args.backend == "cpu" else litert_lm.Backend.GPU()
    t0 = time.time()
    eng = EmbeddingEngine(args.bundle, backend=backend)
    print(f"LOAD_OK {time.time()-t0:.1f}s", flush=True)
    opts = EmbeddingOptions(normalize=not args.no_normalize,
                            insert_special_tokens=not args.no_special_tokens)
    embs, lat = [], []
    for t in texts:
        t1 = time.time()
        r = eng.compute_embedding(t, opts)
        lat.append(time.time() - t1)
        embs.append(list(r.embedding))
    json.dump({"embeddings": embs, "dim": len(embs[0]) if embs else 0,
               "options": {"normalize": not args.no_normalize,
                           "insert_special_tokens": not args.no_special_tokens},
               "latency_ms": [round(x * 1000, 1) for x in lat]}, open(args.out, "w"))
    print(f"EMBED_OK n={len(embs)} dim={len(embs[0]) if embs else 0} "
          f"median_ms={sorted(lat)[len(lat)//2]*1000:.1f}", flush=True)


if __name__ == "__main__":
    main()
