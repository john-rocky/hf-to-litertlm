#!/usr/bin/env python3
"""Upstream fp32's margin at the token where a bundle's reading leaves upstream's reading.

  python hf_margin.py --snap SNAPSHOT --fixtures FIX.json[:DIR] ... --name formula_2 --mode p448 \
      --oracle HF_RUNS.jsonl --other RUNS.jsonl Av-r_formula_2 [--other RUNS.jsonl TAG ...] --out margin.json

One model load and one greedy generation of the fixture, the same way as hf_oracle.py (fp32, fast_dwconv, the
checkpoint's chat template with enable_thinking=False, do_sample False), with the raw logits of every step kept.
The generated ids must equal hf_oracle.py's recorded run of the same fixture and mode (--oracle). Each other reading
(the "text" of the row whose "tag" is --tag in --other) is tokenized with the checkpoint's tokenizer; at the first
token where it leaves upstream's ids, the JSON gives upstream's top-5 and the logit and probability upstream gives
the other reading's token there (margin = upstream's top logit minus that logit).
"""
import argparse
import json
import os
import sys

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fast_dwconv  # noqa: E402
from hf_oracle import MAX_NEW, PAGE_MAX_NEW, load_items  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", required=True)
    ap.add_argument("--fixtures", nargs="+", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--mode", default="p448")
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--other", nargs=2, action="append", metavar=("RUNS", "TAG"), required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    fast_dwconv.install()

    from transformers import AutoModelForImageTextToText, AutoProcessor

    it = next(i for i in load_items(args.fixtures) if i["name"] == args.name)
    f = it["files"][args.mode]
    img = Image.open(os.path.join(it["_dir"], f["file"])).convert("RGB")
    processor = AutoProcessor.from_pretrained(args.snap)
    model = AutoModelForImageTextToText.from_pretrained(args.snap, dtype=torch.float32).eval()
    messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": it["prompt"]}]}]
    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True,
                                           return_tensors="pt", enable_thinking=False)
    n_in = inputs["input_ids"].shape[1]
    max_new = PAGE_MAX_NEW if it["name"].startswith("page") else MAX_NEW[it["kind"]]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False, output_logits=True,
                             return_dict_in_generate=True)
    up = out.sequences[0, n_in:].tolist()
    rec = next(json.loads(l) for l in open(args.oracle) if json.loads(l)["name"] == args.name and json.loads(l)["mode"] == args.mode)
    tok = processor.tokenizer
    result = {"name": args.name, "mode": args.mode, "upstream_ids_equal_oracle": up == rec["gen_ids"],
              "upstream_text": tok.decode(up, skip_special_tokens=True), "others": []}
    for runs, tag in args.other:
        row = next(json.loads(l) for l in open(runs) if json.loads(l).get("tag") == tag)
        other = tok.encode(row["text"], add_special_tokens=False)
        p = next((i for i, (a, b) in enumerate(zip(up, other)) if a != b), min(len(up), len(other)))
        e = {"runs": runs, "tag": tag, "text": row["text"], "first_diff_step": p,
             "prefix": tok.decode(up[:p], skip_special_tokens=False)}
        if p < len(out.logits) and p < len(other):
            lg = out.logits[p][0].float()
            pr = torch.softmax(lg, -1)
            top = torch.topk(lg, 5)
            e["top5"] = [{"id": int(i), "token": tok.decode([int(i)]), "logit": round(float(v), 4),
                          "prob": round(float(pr[i]), 6)} for v, i in zip(top.values, top.indices)]
            o = other[p]
            e["upstream_token"] = {"id": up[p], "token": tok.decode([up[p]]), "logit": round(float(lg[up[p]]), 4),
                                   "prob": round(float(pr[up[p]]), 6)}
            e["other_token"] = {"id": o, "token": tok.decode([o]), "logit": round(float(lg[o]), 4),
                                "prob": round(float(pr[o]), 6),
                                "rank": int((lg > lg[o]).sum()) + 1}
            e["margin_logit"] = round(float(lg[up[p]] - lg[o]), 4)
        result["others"].append(e)
    json.dump(result, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
