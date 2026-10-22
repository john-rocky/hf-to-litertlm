"""calib_dory.py MODEL_DIR G9V3_CALIB THINK_IDS OUT_DIR - the 320 calibration samples of the GPTQ Hessians
(gptq_hessians.py), as token ids of NVIDIA-Nemotron-Labs-Dory1.0-4B.

  256 = the calibration set of g9v3_work (`g9v3_work/gptq_quantize.py OUT G9V3_CALIB --calib-only`: 128
        HuggingFaceH4/ultrachat_200k train_sft, 64 tatsu-lab/alpaca, 64 openai/gsm8k main train, at pinned revisions,
        checked by its sha256), stored there in G9v3's ChatML form. The user and assistant text of each row is taken
        back out of that text and rendered again with Dory's own template (transformers apply_chat_template on
        [user, assistant], add_generation_prompt=False, enable_thinking=False), tokenized with the repo's tokenizer.json
        (add_special_tokens=False), cut at 1,024 tokens.
   64 = thinking on: the same 64 GSM8K training questions + the instruction below, rendered as a thinking-on generation
        prompt (the text ends "<|im_start|>assistant\\n<think>\\n"), followed by THINK_IDS (calib_think_ids.jsonl): the
        model's own greedy continuation of each prompt, made once with llama.cpp (BF16, Metal) up to 1,024 tokens in all;
        a final stop id (2 or 11) would be left out; prompt + continuation cut at 1,024 tokens. Each row of THINK_IDS
        carries the prompt length it was made from, which must equal this render's.
Writes OUT_DIR/calib.jsonl (the 256 rows: k, source, n_full, n, ids, text), OUT_DIR/think_prompts.jsonl (the 64
prompts) and OUT_DIR/samples.jsonl (the 320 samples in Hessian order: name, source, ids) and prints the sha256 of
samples.jsonl.

Usage: python dory_work/calib_dory.py <model dir> <calib_g9v3_256.jsonl> dory_work/calib_think_ids.jsonl <out dir>
"""
import hashlib
import json
import os
import re
import sys

from tokenizers import Tokenizer
from transformers import AutoTokenizer

MAXLEN = 1024
STOP_IDS = (2, 11)
INSTRUCTION = ("\n\nSolve this step by step. After your reasoning, write the final answer on its own "
               "line in the exact form:\n#### <number>")
CHATML = re.compile(r"^<s><\|im_start\|>user\n(.*?)<\|im_end\|>\n<\|im_start\|>assistant\n<think>\n\n</think>\n\n(.*)"
                    r"<\|im_end\|>\n?$", re.S)


def main(snap, g9v3_calib, think_ids, out):
  os.makedirs(out, exist_ok=True)
  hf_tok = AutoTokenizer.from_pretrained(snap)
  tok = Tokenizer.from_file(os.path.join(snap, "tokenizer.json"))

  src = [json.loads(line) for line in open(g9v3_calib, encoding="utf-8")]
  assert len(src) == 256, len(src)
  rows, counts, cut = [], {}, 0
  for k, r in enumerate(src):
    m = CHATML.match(r["text"])
    assert m, (k, r["source"], r["text"][:120])
    user, assistant = m.group(1), m.group(2)
    text = hf_tok.apply_chat_template([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}],
                                      tokenize=False, add_generation_prompt=False, enable_thinking=False)
    ids = tok.encode(text, add_special_tokens=False).ids
    cut += len(ids) > MAXLEN
    rows.append(dict(k=k, source=r["source"], user=user, n_full=len(ids), n=min(len(ids), MAXLEN), ids=ids[:MAXLEN],
                     text=text))
    counts[r["source"]] = counts.get(r["source"], 0) + 1
  with open(os.path.join(out, "calib.jsonl"), "w", encoding="utf-8") as f:
    for r in rows:
      f.write(json.dumps({k: r[k] for k in ("k", "source", "n_full", "n", "ids", "text")}, ensure_ascii=False) + "\n")

  gsm = [r for r in rows if r["source"] == "gsm8k_train"]
  assert len(gsm) == 64, len(gsm)
  think = []
  for j, r in enumerate(gsm):
    text = hf_tok.apply_chat_template([{"role": "user", "content": r["user"] + INSTRUCTION}], tokenize=False,
                                      add_generation_prompt=True, enable_thinking=True)
    assert text.endswith("<|im_start|>assistant\n<think>\n"), text[-60:]
    ids = tok.encode(text, add_special_tokens=False).ids
    think.append(dict(name=f"t{j:02d}", k=r["k"], question=r["user"], text=text, ids=ids, n_prompt=len(ids),
                      maxnew=MAXLEN - len(ids)))
  with open(os.path.join(out, "think_prompts.jsonl"), "w", encoding="utf-8") as f:
    for t in think:
      f.write(json.dumps(t, ensure_ascii=False) + "\n")

  gen = {}
  for line in open(think_ids, encoding="utf-8"):
    d = json.loads(line)
    gen[d["name"]] = d
  assert sorted(gen) == [t["name"] for t in think], sorted(gen)[:5]
  samples = [dict(name=f"c{r['k']:03d}", source=r["source"], ids=r["ids"][:MAXLEN]) for r in rows]
  for t in think:
    g = gen[t["name"]]
    assert g["n_prompt"] == t["n_prompt"], (t["name"], g["n_prompt"], t["n_prompt"])
    body = g["ids"][:-1] if g["ids"] and g["ids"][-1] in STOP_IDS else g["ids"]
    samples.append(dict(name=t["name"], source="think_on", ids=(t["ids"] + body)[:MAXLEN]))
  path = os.path.join(out, "samples.jsonl")
  with open(path, "w", encoding="utf-8") as f:
    for s in samples:
      f.write(json.dumps(s) + "\n")
  digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
  ntok = sum(len(s["ids"]) for s in samples)
  print(f"CALIB 256 rows {counts} (cut at 1,024: {cut}) + 64 thinking-on rows; {len(samples)} samples, {ntok} tokens; "
        f"samples.jsonl sha256 {digest}", flush=True)
  return 0


if __name__ == "__main__":
  if len(sys.argv) != 5:
    raise SystemExit(__doc__)
  sys.exit(main(*sys.argv[1:5]))
