#!/usr/bin/env python3
"""line1_hf_oracle.py HF_DIR OUT_DIR - the HF side of the exporter check ("line 1").

Sequences (17): the 8 questions of `scripts/verify_quality.py` (QUESTIONS + SUFFIX) x thinking off / on,
rendered by the vendor template with the generation prompt (`apply_chat_template(..., add_generation_prompt=True,
enable_thinking=...)`), plus one long sequence = `HF_DIR/LICENSE` lines 1-58 (LFM Open License v1.0, sections 1-5,
the repo's own public file) + a question about its section 5, thinking off. Each prompt is extended by the HF bf16
greedy continuation (64 new tokens at most, stops at EOS; repetition penalty off).

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

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))  # repository root
_spec = importlib.util.spec_from_file_location("verify_quality", os.path.join(REPO, "scripts/verify_quality.py"))
verify_quality = importlib.util.module_from_spec(_spec)  # QUESTIONS, SUFFIX (loaded by path, no sys.path change)
_spec.loader.exec_module(verify_quality)

LONG_QUESTION = (
    "According to section 5 of the license above, does the Threshold apply to a Qualified Non-Profit "
    "Organization's use of the Work for research purposes? Answer briefly."
)


def sha(ids):
  return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def conversations(hf_dir):
  out = []
  for i, (label, q, _) in enumerate(verify_quality.QUESTIONS):
    for think in (False, True):
      out.append((f"q{i + 1}_{'on' if think else 'off'}", [{"role": "user", "content": q + verify_quality.SUFFIX}], think))
  with open(os.path.join(hf_dir, "LICENSE")) as f:
    excerpt = "\n".join(f.read().split("\n")[:58]).strip()
  out.append(("long_license", [{"role": "user", "content": excerpt + "\n\n" + LONG_QUESTION}], False))
  return out


def main(hf_dir, out_dir):
  torch.set_grad_enabled(False)
  os.makedirs(os.path.join(out_dir, "hf_fp32"), exist_ok=True)
  tok = transformers.AutoTokenizer.from_pretrained(hf_dir, trust_remote_code=True)
  seqs = []
  for name, messages, think in conversations(hf_dir):
    text = tok.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=think, tokenize=False)
    ids = tok(text, add_special_tokens=False).input_ids
    ids_t = tok.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=think, tokenize=True)
    if hasattr(ids_t, "input_ids") or isinstance(ids_t, dict):
      ids_t = ids_t["input_ids"]
    assert list(ids_t) == ids, (name, "apply_chat_template tokenize=True differs from tokenize(text)")
    seqs.append({"name": name, "thinking": think, "messages": messages, "text": text, "prompt_ids": ids})

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
    bf16_logits[s["name"]] = bf16(torch.tensor([s["ids"]])).logits[0].float().numpy()
    print(f"{s['name']}: prompt {len(s['prompt_ids'])} + cont {len(s['cont_ids'])} "
          f"({time.time() - t1:.1f}s) {tok.decode(s['cont_ids'])[:80]!r}", flush=True)
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
  for s in seqs:
    s.pop("messages")
  meta = {
      "hf_dir": os.path.abspath(hf_dir),
      "transformers": transformers.__version__,
      "torch": torch.__version__,
      "long_source": "HF_DIR/LICENSE lines 1-58 (LFM Open License v1.0, sections 1-5) + LONG_QUESTION",
      "long_question": LONG_QUESTION,
      "sequences": seqs,
  }
  with open(os.path.join(out_dir, "inputs.json"), "w") as f:
    json.dump(meta, f, indent=1, ensure_ascii=False)
  with open(os.path.join(out_dir, "hf_bf16_vs_fp32.json"), "w") as f:
    json.dump(ref, f, indent=1)
  print("DONE line1_hf_oracle", flush=True)


if __name__ == "__main__":
  main(sys.argv[1], sys.argv[2])
