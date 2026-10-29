#!/usr/bin/env python3
"""Proof that fast_dwconv changes nothing that matters in the reference: the same image prompt through the full model,
prefill + 3 teacher-forced decode steps, with the original conv1d and with fast_dwconv; logits compared per position.

  python dwconv_proof.py SNAPSHOT IMAGE.png OUT.json
"""
import json
import os
import sys
import time

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fast_dwconv  # noqa: E402


def run(model, inputs, steps):
    logits, t = [], time.time()
    with torch.no_grad():
        out = model(**inputs, use_cache=True)
        logits.append(out.logits[0, -1].clone())
        pkv, nxt = out.past_key_values, out.logits[:, -1:].argmax(-1)
        for _ in range(steps):
            out = model(input_ids=nxt, past_key_values=pkv, use_cache=True)
            logits.append(out.logits[0, -1].clone())
            pkv, nxt = out.past_key_values, out.logits[:, -1:].argmax(-1)
    return torch.stack(logits), round(time.time() - t, 2)


def main():
    from transformers import AutoModelForImageTextToText, AutoProcessor
    snap, image, out_path = sys.argv[1:4]
    proc = AutoProcessor.from_pretrained(snap)
    model = AutoModelForImageTextToText.from_pretrained(snap, dtype=torch.float32).eval()
    img = Image.open(image).convert("RGB")
    msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": "Text Recognition:"}]}]
    inputs = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                      return_tensors="pt", enable_thinking=False)
    a, ta = run(model, inputs, 3)
    fast_dwconv.install()
    b, tb = run(model, inputs, 3)
    d = (a.double() - b.double()).abs()
    res = {"unit": fast_dwconv.check(), "positions": int(a.shape[0]),
           "max_abs_delta": float(d.max()), "max_abs_logit": float(a.abs().max()),
           "argmax_orig": a.argmax(-1).tolist(), "argmax_fast": b.argmax(-1).tolist(),
           "bit_equal": bool(torch.equal(a, b)), "seconds_orig": ta, "seconds_fast": tb}
    json.dump(res, open(out_path, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
