#!/usr/bin/env python3
"""HF-side reference: load the dequantized bf16 checkpoint, install the Hadamard transform (the same module
the export uses), greedy-decode N tokens for fixed prompts, dump token ids + first-step top-5 logits.
Compared against mlx_ref.py (PrismML's own runtime on the same pack) -> token-identical = dequant + layout +
transform correct.  Usage: hf_check.py <ckpt_dir> <out.json> [n_tokens] [device]
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import hadamard_export

ckpt, out = sys.argv[1], sys.argv[2]
n_tok = int(sys.argv[3]) if len(sys.argv) > 3 else 16
device = sys.argv[4] if len(sys.argv) > 4 else "cpu"
PROMPTS = ["What is the capital of France? Answer in one word.",
           "Write one sentence about the ocean.",
           "What is 17 * 23? Show the calculation briefly."]
tok = AutoTokenizer.from_pretrained(ckpt)
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
model.eval()
n = hadamard_export.install_hadamard(model, ckpt)
print(f"loaded {type(model).__name__} in {time.time()-t0:.0f}s, hadamard modules {n}", flush=True)
model.to(device)
res = {}
for p in PROMPTS:
    ids = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True,
                                  enable_thinking=False, tokenize=True, return_dict=False)
    ids = torch.tensor([list(ids)], dtype=torch.long, device=device)
    with torch.no_grad():
        out_ids = model.generate(ids, max_new_tokens=n_tok, do_sample=False, num_beams=1)
        logits = model(ids).logits[0, -1].float()
    gen = out_ids[0, ids.shape[1]:].tolist()
    top = torch.topk(logits, 5)
    res[p] = {"prompt_ids": ids[0].tolist(), "gen_ids": gen, "text": tok.decode(gen),
              "top5": [[int(i), round(float(v), 3)] for v, i in zip(top.values, top.indices)]}
    print(repr(p), "->", repr(res[p]["text"]), "| top5", res[p]["top5"], flush=True)
json.dump(res, open(out, "w"), indent=1)
print("DONE", out, f"{time.time()-t0:.0f}s")
