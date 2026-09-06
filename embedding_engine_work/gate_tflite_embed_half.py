#!/usr/bin/env python3
"""tflite half of an EmbeddingEngine parity gate, model-agnostic (run under .venv-092).

    .venv-092/bin/python scripts/gate_tflite_embed_half.py <shipped.tflite> <tokenizer.json> <texts.json> <out.json> [--pad-id N]

Assumes the shipped encoder .tflite exposes `embed_<S>` signatures with int32 `input_ids` +
`attention_mask` (right-padded) and one pooled output — the shape of every encoder we ship
(granite, LFM2.5-Embedding, Nemotron). Tokenizes with the HF tokenizer INCLUDING its
post-processor (bos/cls), picks the smallest signature that fits, writes the same JSON as
gate_embedding_engine.py.
"""
import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, '..', 'scripts')]
import _stub  # noqa: F401

import numpy as np
from ai_edge_litert.interpreter import Interpreter
from tokenizers import Tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tflite"); ap.add_argument("tokenizer"); ap.add_argument("texts"); ap.add_argument("out")
    ap.add_argument("--pad-id", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=512)
    args = ap.parse_args()
    tok = Tokenizer.from_file(args.tokenizer)
    texts = json.load(open(args.texts))["texts"]
    it = Interpreter(model_path=args.tflite, num_threads=os.cpu_count())
    sigs = sorted(int(k.split("_")[1]) for k in it.get_signature_list() if k.startswith("embed_"))
    runners = {S: it.get_signature_runner(f"embed_{S}") for S in sigs}
    embs, ids_out = [], []
    for t in texts:
        ids = tok.encode(t).ids[:args.max_tokens]
        S = next((s for s in sigs if len(ids) <= s), sigs[-1]); ids = ids[:S]
        a = np.full((1, S), args.pad_id, dtype=np.int32); m = np.zeros((1, S), dtype=np.int32)
        a[0, :len(ids)] = ids; m[0, :len(ids)] = 1
        v = list(runners[S](input_ids=a, attention_mask=m).values())[0][0]
        embs.append(v.astype(float).tolist()); ids_out.append(ids)
    json.dump({"embeddings": embs, "dim": len(embs[0]), "ids": ids_out}, open(args.out, "w"))
    print(f"TFLITE_OK n={len(embs)} dim={len(embs[0])} sigs={sigs} first_ids={ids_out[0][:6]}")


if __name__ == "__main__":
    main()
