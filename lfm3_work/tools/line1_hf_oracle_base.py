#!/usr/bin/env python3
"""line1_hf_oracle_base.py HF_DIR OUT_DIR - the HF side of the exporter check ("line 1") for LFM3-1.7B-Base.

Derived from tools/line1_hf_oracle.py (the LFM3-1.7B oracle): the same outputs, but raw-text prompts instead of
the chat template, because LFM3-1.7B-Base is a pre-trained checkpoint whose README feeds raw text.

Sequences (17), each prompt = [start token 124894] + tokenizer(text, add_special_tokens=False) - the ids LiteRT-LM's
Session sends for raw text (create_session(apply_prompt_template=False) adds the file's start token on the first turn,
runtime/core/session_utils.cc in v0.19.1):
  q1..q8        the 8 questions of `scripts/verify_quality.py` as "Question: " + q + SUFFIX + "\nAnswer:"
  c1..c8        8 text openings (c1 is the example of the source README)
  long_license  HF_DIR/LICENSE lines 1-58 + "\n\nQuestion: " + LONG_QUESTION + "\nAnswer:"
Each prompt is extended by the HF bf16 greedy continuation (64 new tokens at most, stops at EOS; repetition penalty
off).

Writes OUT_DIR/inputs.json (texts, ids, sha256 of each id list, continuation ids), OUT_DIR/hf_fp32/<name>.npy
(float32 logits [len, vocab] of one HF fp32 forward over prompt + continuation, CPU, eager attention) and
OUT_DIR/hf_bf16_vs_fp32.json (reference only, not a line: HF bf16 vs HF fp32 on the same sequences).
"""
import hashlib
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch
import transformers

REPO = os.path.expanduser("~/code/litertlm-convert")
_spec = importlib.util.spec_from_file_location("verify_quality", os.path.join(REPO, "scripts/verify_quality.py"))
verify_quality = importlib.util.module_from_spec(_spec)  # QUESTIONS, SUFFIX (loaded by path, no sys.path change)
_spec.loader.exec_module(verify_quality)

BOS_ID = 124894  # <|startoftext|>, the file's start token
LONG_QUESTION = (
    "According to section 5 of the license above, does the Threshold apply to a Qualified Non-Profit "
    "Organization's use of the Work for research purposes? Answer briefly."
)
OPENINGS = [
    "The nematode C. elegans is",
    "Photosynthesis is the process by which",
    'def fibonacci(n):\n    """Return the n-th Fibonacci number."""\n',
    "The Apollo 11 mission landed on the Moon in",
    "A simple recipe for tomato soup:\n1.",
    "La photosynthèse est le processus par lequel",
    "El sistema solar está formado por",
    "To convert a temperature from Celsius to Fahrenheit,",
]


def sha(ids):
  return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def qa(question):
  return "Question: " + question + "\nAnswer:"


def texts(hf_dir):
  out = []
  for i, (_, q, _) in enumerate(verify_quality.QUESTIONS):
    out.append((f"q{i + 1}", qa(q + verify_quality.SUFFIX)))
  for i, t in enumerate(OPENINGS):
    out.append((f"c{i + 1}", t))
  with open(os.path.join(hf_dir, "LICENSE")) as f:
    excerpt = "\n".join(f.read().split("\n")[:58]).strip()
  out.append(("long_license", excerpt + "\n\n" + qa(LONG_QUESTION)))
  return out


def main(hf_dir, out_dir):
  torch.set_grad_enabled(False)
  os.makedirs(os.path.join(out_dir, "hf_fp32"), exist_ok=True)
  tok = transformers.AutoTokenizer.from_pretrained(hf_dir, trust_remote_code=True)
  assert tok.convert_tokens_to_ids("<|startoftext|>") == BOS_ID
  seqs = []
  for name, text in texts(hf_dir):
    body = tok(text, add_special_tokens=False).input_ids
    plain = tok(text).input_ids
    assert plain == body, (name, "the tokenizer adds special tokens by itself")
    seqs.append({"name": name, "text": text, "prompt_ids": [BOS_ID] + body})

  t0 = time.time()
  bf16 = transformers.AutoModelForCausalLM.from_pretrained(
      hf_dir, dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="eager").eval()
  print(f"bf16 loaded {time.time() - t0:.1f}s", flush=True)
  eos = bf16.generation_config.eos_token_id
  bf16_logits = {}
  for s in seqs:
    t1 = time.time()
    gen = bf16.generate(
        torch.tensor([s["prompt_ids"]]), max_new_tokens=64, do_sample=False, temperature=None, top_k=None,
        top_p=None, repetition_penalty=1.0, eos_token_id=eos, pad_token_id=bf16.generation_config.pad_token_id)
    s["cont_ids"] = gen[0, len(s["prompt_ids"]):].tolist()
    s["ids"] = s["prompt_ids"] + s["cont_ids"]
    s["ids_sha256"] = sha(s["ids"])
    s["prompt_ids_sha256"] = sha(s["prompt_ids"])
    s["cont_text"] = tok.decode(s["cont_ids"])
    bf16_logits[s["name"]] = bf16(torch.tensor([s["ids"]])).logits[0].float().numpy()
    print(f"{s['name']}: prompt {len(s['prompt_ids'])} + cont {len(s['cont_ids'])} "
          f"({time.time() - t1:.1f}s) {s['cont_text'][:80]!r}", flush=True)
  del bf16

  t0 = time.time()
  fp32 = transformers.AutoModelForCausalLM.from_pretrained(
      hf_dir, dtype=torch.float32, trust_remote_code=True, attn_implementation="eager").eval()
  print(f"fp32 loaded {time.time() - t0:.1f}s", flush=True)
  ref = {}
  for s in seqs:
    t1 = time.time()
    logits = fp32(torch.tensor([s["ids"]])).logits[0].float().numpy()
    np.save(os.path.join(out_dir, "hf_fp32", s["name"] + ".npy"), logits)
    b = bf16_logits.pop(s["name"])
    top2 = np.sort(logits, axis=-1)[:, -2:]
    keep = (top2[:, 1] - top2[:, 0]) >= 1e-3
    ref[s["name"]] = {
        "positions": int(len(s["ids"])),
        "max_abs_dlogit": float(np.abs(b - logits).max()),
        "top1_agree": int((b.argmax(-1) == logits.argmax(-1))[keep].sum()),
        "top1_of": int(keep.sum()),
    }
    print(f"{s['name']}: fp32 forward {time.time() - t1:.1f}s; bf16 vs fp32 {ref[s['name']]}", flush=True)
  meta = {
      "hf_dir": os.path.abspath(hf_dir),
      "transformers": transformers.__version__,
      "torch": torch.__version__,
      "prompt_form": "raw text, [124894] + tokenizer(text, add_special_tokens=False); no chat template",
      "long_source": "HF_DIR/LICENSE lines 1-58 (LFM Open License v1.0, sections 1-5) + Question / Answer",
      "long_question": LONG_QUESTION,
      "sequences": seqs,
  }
  with open(os.path.join(out_dir, "inputs.json"), "w") as f:
    json.dump(meta, f, indent=1, ensure_ascii=False)
  with open(os.path.join(out_dir, "hf_bf16_vs_fp32.json"), "w") as f:
    json.dump(ref, f, indent=1)
  print("DONE line1_hf_oracle_base", flush=True)


if __name__ == "__main__":
  main(sys.argv[1], sys.argv[2])
