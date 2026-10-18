"""The card's Python usage snippet, run as a script before the card is shown (rule: shipped code runs first).
The body between the markers is what the card shows verbatim; the tail checks the vector against the reference.

    python3 card/snippet_check.py [--model exports/vela_L64_ids_v3_fp16fc_fp16emb_shift_arith.tflite]
"""
import argparse
import json
import sys
from pathlib import Path

W = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="exports/vela_L64_ids_v3_fp16fc_fp16emb_shift_arith.tflite")
ap.add_argument("--tokenizer", default="hf/tokenizer.json")
a = ap.parse_args()
MODEL, TOKENIZER = str(W / a.model), str(W / a.tokenizer)

# --- card snippet begins -------------------------------------------------------------------------------------------------
import numpy as np
from ai_edge_litert.compiled_model import CompiledModel, CpuOptions, HardwareAccelerator, Options
from tokenizers import Tokenizer

L = 64                                                      # the file's window: vela_L64_fp16.tflite takes up to 64 tokens
tok = Tokenizer.from_file(TOKENIZER)                        # the repo's tokenizer.json (adds <bos> = 2 and <eos> = 1)
model = CompiledModel.from_file(MODEL, options=Options(HardwareAccelerator.CPU, cpu_options=CpuOptions(num_threads=4)))
sig = list(model.get_signature_list())[0]                  # "serving_default"
ins = {n: model.create_input_buffer_by_name(sig, n) for n in model.get_input_tensor_details(sig)}
outs = {n: model.create_output_buffer_by_name(sig, n) for n in model.get_output_tensor_details(sig)}


def embed(text: str) -> np.ndarray:
    ids = tok.encode(text.strip()).ids                      # strip first: sentence-transformers does, and a trailing space adds a token
    assert len(ids) <= L, f"{len(ids)} tokens > window {L}: use the next window file"
    input_ids = np.zeros((1, L), np.int32); input_ids[0, : len(ids)] = ids
    mask = np.zeros((1, L), np.float32); mask[0, : len(ids)] = 1.0
    ins["input_ids"].write(input_ids); ins["attention_mask"].write(mask)
    model.run_by_name(sig, ins, outs)
    return np.asarray(outs["embedding"].read(768, np.float32), dtype=np.float32).reshape(768)   # L2-normalized, 768-d


q, d1, d2 = "How do I reset my password?", "Steps to change a forgotten password", "Best pizza toppings for a party"
e = [embed(t) for t in (q, d1, d2)]
print("cos(q, d1) =", float(e[0] @ e[1]), " cos(q, d2) =", float(e[0] @ e[2]))
# --- card snippet ends ---------------------------------------------------------------------------------------------------

# Check against the reference (the publisher's stack): the fixture texts are in fixtures/texts.json, their vectors in oracle/
ref = np.load(W / "oracle" / "embeddings.npz")
fx = json.loads((W / "fixtures" / "texts.json").read_text())["rows"]
rows = [r for r in fx if r["group"] == "pair" and r["row_len"] <= L][:8]
worst = 1.0
for r in rows:
    v = embed(r["text"])
    c = float(v.astype(np.float64) @ ref[r["key"]].astype(np.float64))
    worst = min(worst, c)
print(f"snippet check: {len(rows)} fixture texts vs the reference, cos min {worst:.6f}", "PASS" if worst >= 0.999 and e[0] @ e[1] > e[0] @ e[2] else "FAIL")
sys.exit(0 if worst >= 0.999 and e[0] @ e[1] > e[0] @ e[2] else 1)
