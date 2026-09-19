#!/usr/bin/env python3
"""PrismML reference: run the Bonsai 2 MLX pack through its own bundled loader (runtime/vision_artifact.py,
mlx-vlm 0.6.3 + mlx 0.32.0) and greedy-decode the same prompts as hf_check.py. Text only.
Usage: mlx_ref.py <pack_dir> <out.json> [n_tokens]"""
import json, os, sys, time
pack, out = sys.argv[1], sys.argv[2]
n_tok = int(sys.argv[3]) if len(sys.argv) > 3 else 16
sys.path.insert(0, os.path.join(pack, "runtime"))
import mlx.core as mx
from vision_artifact import load_vl_model
from transformers import AutoTokenizer

PROMPTS = ["What is the capital of France? Answer in one word.",
           "Write one sentence about the ocean.",
           "What is 17 * 23? Show the calculation briefly."]
t0 = time.time()
model, _, config = load_vl_model(pack, load_processor=False)
tok = AutoTokenizer.from_pretrained(pack)
print(f"loaded pack in {time.time()-t0:.0f}s", flush=True)
lm = model.language_model

def logits_of(o):
    return o.logits if hasattr(o, "logits") else o

res = {}
for p in PROMPTS:
    ids = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, enable_thinking=False, tokenize=True, return_dict=False)
    ids = list(ids["input_ids"]) if hasattr(ids, "keys") else list(ids)
    cache = lm.make_cache()
    x = mx.array([ids])
    lg = logits_of(lm(x, cache=cache))[0, -1].astype(mx.float32)
    mx.eval(lg)
    top = mx.argpartition(-lg, 5)[:5]; top = sorted([(float(lg[i]), int(i)) for i in top.tolist()], reverse=True)
    gen = []
    nxt = int(mx.argmax(lg).item())
    for _ in range(n_tok):
        gen.append(nxt)
        if nxt in (248046, 248044): break
        lg = logits_of(lm(mx.array([[nxt]]), cache=cache))[0, -1]
        nxt = int(mx.argmax(lg).item())
    res[p] = {"prompt_ids": ids, "gen_ids": gen, "text": tok.decode(gen), "top5": [[i, round(v, 3)] for v, i in top]}
    print(repr(p), "->", repr(res[p]["text"]), "| top5", res[p]["top5"], flush=True)
json.dump(res, open(out, "w"), indent=1)
print("DONE", out, f"{time.time()-t0:.0f}s")
