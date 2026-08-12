#!/usr/bin/env python3
"""HF torch reference on the same gate fixtures, for conversion-vs-capability
triage. Uses the vendor processor + bf16 weights on CPU, greedy decoding.

Usage: hf_reference.py <model_path> [n_tokens]
"""
import os
import sys

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image

MODEL = sys.argv[1]
N_TOK = int(sys.argv[2]) if len(sys.argv) > 2 else 32
FIXDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")

CASES = [
    ("color=red", "red.png", "What is the dominant color of this image? Answer briefly."),
    ("ocr=HELLO", "hello.png", "What does the text in this image say? Answer briefly."),
    ("shape=circle", "circle.png", "What shape is shown in this image? Answer briefly."),
    ("count=3", "count3.png", "How many black squares are in this image? Answer briefly."),
    ("big-word=CAT", "cat_dog.png", "What is the largest word written in this image? Answer briefly."),
]

processor = AutoProcessor.from_pretrained(MODEL)
model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16)
model.eval()

for label, img, q in CASES:
    messages = [{"role": "user", "content": [
        {"type": "image", "image": Image.open(os.path.join(FIXDIR, img))},
        {"type": "text", "text": q},
    ]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=N_TOK, do_sample=False)
    text = processor.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True)
    print(f"[{label}] {text!r}")
