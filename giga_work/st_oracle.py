#!/usr/bin/env python3
"""Independent oracle: the vendor's own documented path (sentence-transformers, remote code).

    python giga_work/st_oracle.py <texts.json> <out.json> [model_id]

Runs in a throwaway venv that has sentence-transformers (the conversion venv does not), and
writes the same JSON shape as embedding_engine_work/gate_embedding_engine.py so compare_embedding_gate.py can
score any of our artifacts against it. Also prints the README example (query vs the two
documents) so the card can quote a similarity the vendor's stack produces.
"""
import json
import sys
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

texts = json.load(open(sys.argv[1]))["texts"]
out = sys.argv[2]
model_id = sys.argv[3] if len(sys.argv) > 3 else "ai-sage/Giga-Embeddings-instruct-480M-0826"

torch.manual_seed(0)
m = SentenceTransformer(model_id, trust_remote_code=True, device="cpu",
                        model_kwargs={"torch_dtype": torch.float32})
print("modules:", [type(x).__name__ for x in m], "max_seq_length", m.max_seq_length, flush=True)
t0 = time.time()
E = m.encode(texts, normalize_embeddings=True, batch_size=1, convert_to_numpy=True)
print(f"encoded {len(texts)} texts in {time.time()-t0:.1f}s, dim {E.shape[1]}", flush=True)
json.dump({"embeddings": E.astype(float).tolist(), "dim": int(E.shape[1]),
           "options": {"path": "sentence-transformers", "model": model_id,
                       "dtype": "float32", "batch_size": 1}}, open(out, "w"))

instr = "Given a query, retrieve relevant passages"
q = m.encode([f"Instruct: {instr}\nQuery: Где столица России?"], normalize_embeddings=True)
d = m.encode(["Москва — столица Российской Федерации.", "Париж — столица Франции."],
             normalize_embeddings=True)
print("README example similarity (query vs [Moscow, Paris]):", np.round(q @ d.T, 4).tolist())
