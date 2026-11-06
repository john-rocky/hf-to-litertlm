#!/usr/bin/env python3
"""make_calib.py MODEL_DIR TREE OUT_DIR - the 48 calibration prompts of npu_export's stage 2 for MNN-Pocket-0.3B, in
npu_export's form (OUT_DIR/ALL.jsonl, one {"text": ...} per line), and OUT_DIR/calib_manifest.json (sources, per-item
token counts and sha256).

Items, in this order:
  gsm8k_train 0..31  the questions of rows 0..31 of openai/gsm8k (config main, split train), read from the Hugging Face
                     datasets server; each prompt = cot4_prefix.txt (the first four of the GSM8K chain-of-thought
                     exemplars of Wei et al., 2022) + "Q: <question>\\nA:" as one user turn
  general 0..15      the first 16 prompts of npu_export's sample_calibration_prompts.json in TREE, as they are
Each prompt is one user message without a system message, rendered by transformers' apply_chat_template with the
repo's chat template (templates/mnn_pocket_0_3b.jinja) and add_generation_prompt=True: the text LiteRT-LM renders for a
string message. Stage 2 runs with enable_formatting=False, so this text is what it tokenizes.

The result is checked against the sha256 of the calibration file of litert-community/MNN-Pocket-0.3B's NPU file
(148f81d3...); a difference is printed and the exit code is 1."""
import hashlib
import json
import os
import sys
import urllib.request

from tokenizers import Tokenizer
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
TEMPLATE = os.path.join(REPO, "templates", "mnn_pocket_0_3b.jinja")
COT4 = os.path.join(HERE, "cot4_prefix.txt")
ROWS = ("https://datasets-server.huggingface.co/rows?dataset=openai%2Fgsm8k&config=main&split=train"
        "&offset=0&length=32")
DATASET_API = "https://huggingface.co/api/datasets/openai/gsm8k"
N_GSM, N_GEN = 32, 16
WANT = "148f81d31d8a40da0f8cc869ab45093a590b7da5c05f8909c85537287bdd48c9"


def sha(b):
  return hashlib.sha256(b).hexdigest()


def get(url):
  with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "hf-to-litertlm"}), timeout=60) as r:
    return r.read()


def main(model_dir, tree, out_dir):
  samples_path = os.path.join(tree, "litert_torch", "generative", "export_hf", "experimental", "npu_export",
                              "sample_calibration_prompts.json")
  rows_raw = get(ROWS)
  rows = json.loads(rows_raw)["rows"]
  assert [r["row_idx"] for r in rows] == list(range(N_GSM)), [r["row_idx"] for r in rows]
  revision = json.loads(get(DATASET_API)).get("sha")
  with open(COT4, encoding="utf-8") as f:
    prefix = f.read()
  with open(samples_path, encoding="utf-8") as f:
    samples = json.load(f)
  with open(TEMPLATE, encoding="utf-8") as f:
    template = f.read()
  tok = AutoTokenizer.from_pretrained(model_dir)
  ids_tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
  items = [("gsm8k_train", i, prefix + "Q: " + rows[i]["row"]["question"] + "\nA:") for i in range(N_GSM)]
  items += [("general", i, samples[i]) for i in range(N_GEN)]
  os.makedirs(out_dir, exist_ok=True)
  man = {"sources": {"gsm8k_rows_url": ROWS, "gsm8k_dataset_revision_now": revision,
                     "cot4_prefix_sha256": sha(prefix.encode()),
                     "samples": os.path.relpath(samples_path, tree),
                     "samples_sha256": sha(open(samples_path, "rb").read()),
                     "template_sha256": sha(template.encode()),
                     "tokenizer_json_sha256": sha(open(os.path.join(model_dir, "tokenizer.json"), "rb").read())},
         "items": []}
  lines = []
  for src, i, content in items:
    text = tok.apply_chat_template([{"role": "user", "content": content}], chat_template=template,
                                   add_generation_prompt=True, tokenize=False)
    assert text.startswith("<|im_start|>user\n") and text.endswith("<|im_end|>\n<|im_start|>assistant\n"), text[:80]
    n = len(ids_tok.encode(text, add_special_tokens=False).ids)
    lines.append(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    man["items"].append({"source": src, "index": i, "chars": len(text), "tokens": n,
                         "sha256": sha(text.encode())})
  data = "".join(lines).encode("utf-8")
  with open(os.path.join(out_dir, "ALL.jsonl"), "wb") as f:
    f.write(data)
  toks = [it["tokens"] for it in man["items"]]
  man["summary"] = {"n": len(toks), "tokens_min": min(toks), "tokens_max": max(toks), "tokens_total": sum(toks)}
  man["ALL_jsonl_sha256"] = sha(data)
  with open(os.path.join(out_dir, "calib_manifest.json"), "w") as f:
    json.dump(man, f, indent=1)
  print(json.dumps(man["summary"]), "ALL.jsonl", man["ALL_jsonl_sha256"], "dataset revision now", revision)
  if man["ALL_jsonl_sha256"] != WANT:
    print(f"DIFFERENT from the published file's calibration set ({WANT})")
    return 1
  print("calibration set = the published file's")
  return 0


if __name__ == "__main__":
  sys.exit(main(*sys.argv[1:4]))
