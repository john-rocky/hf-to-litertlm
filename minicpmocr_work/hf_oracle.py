#!/usr/bin/env python3
"""Upstream reference runs of MiniCPM-OCR-1.0 (transformers main @2df500f3) on the r1 fixtures.

  python hf_oracle.py --snap SNAPSHOT --fixtures FIX.json[:DIR] ... --modes native,s448,p448 \
      --out runs.jsonl [--only name,name] [--dtype fp32] [--threads N]

One model load, then for every fixture and mode one greedy generation (do_sample False, no repetition
penalty, eos = generation_config's [248044, 248046]) on the CPU, with the depthwise conv1d of the GatedDeltaNet layers
computed by fast_dwconv (torch's CPU kernel takes ~10 s a decode step on macOS arm64; --orig-conv keeps it). The prompt is the checkpoint's chat template
with enable_thinking=False (its default) on one user turn: [image, text]. A finished (name, mode) pair already
in --out is skipped, so the run can be resumed. Each JSONL row: name, mode, image file + sha256, the processor's
target_sizes / grids, the number of input and image tokens, the generated ids (eos included when emitted), the
text with and without special tokens, the stop reason, and the seconds.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fast_dwconv  # noqa: E402

MAX_NEW = {"text": 512, "text_ref": 768, "table": 768, "formula": 256, "kie": 256}
PAGE_MAX_NEW = 1536
IMAGE_PAD = 248056


def load_items(specs):
    items = []
    for spec in specs:
        path, _, d = spec.partition(":")
        d = d or os.path.dirname(path)
        for it in json.load(open(path))["items"]:
            it["_dir"] = d
            items.append(it)
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", required=True)
    ap.add_argument("--fixtures", nargs="+", required=True)
    ap.add_argument("--modes", default="native,s448,p448")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=None)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--orig-conv", action="store_true",
                    help="keep torch's conv1d (default: fast_dwconv, logits within 6.3e-5 of it on an image prompt, dwconv_proof.py)")
    args = ap.parse_args()
    if not args.orig_conv:
        fast_dwconv.install()
    if args.threads:
        torch.set_num_threads(args.threads)

    from transformers import AutoModelForImageTextToText, AutoProcessor
    import transformers

    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            r = json.loads(line)
            done.add((r["name"], r["mode"]))

    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(args.snap)
    model = AutoModelForImageTextToText.from_pretrained(args.snap, dtype=dtype).eval()
    load_s = time.time() - t0
    print(f"loaded {type(model).__name__} {args.dtype} in {load_s:.1f}s, transformers {transformers.__version__}, "
          f"torch {torch.__version__}, threads {torch.get_num_threads()}", flush=True)
    tied = model.lm_head.weight.data_ptr() == model.model.language_model.embed_tokens.weight.data_ptr()
    print(f"lm_head tied to embed_tokens: {tied}", flush=True)

    only = set(args.only.split(",")) if args.only else None
    modes = args.modes.split(",")
    eos = set(model.generation_config.eos_token_id if isinstance(model.generation_config.eos_token_id, list)
              else [model.generation_config.eos_token_id])
    for it in load_items(args.fixtures):
        if only and it["name"] not in only:
            continue
        for mode in modes:
            if (it["name"], mode) in done or mode not in it["files"]:
                continue
            f = it["files"][mode]
            path = os.path.join(it["_dir"], f["file"])
            sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
            assert sha == f["sha256"], (path, sha)
            img = Image.open(path).convert("RGB")
            messages = [{"role": "user", "content": [{"type": "image", "image": img},
                                                      {"type": "text", "text": it["prompt"]}]}]
            # A GxG mode above 448 is fed as ONE image of that size (no slices): scale_resolution G, slice_mode off.
            g = int(mode[1:]) if mode[:1] in "sp" and mode[1:].isdigit() else 448
            extra = {"scale_resolution": g, "slice_mode": False} if g > 448 else {}
            inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                                   return_dict=True, return_tensors="pt", enable_thinking=False,
                                                   **extra)
            ids = inputs["input_ids"]
            n_img = int((ids == IMAGE_PAD).sum())
            max_new = PAGE_MAX_NEW if it["name"].startswith("page") else MAX_NEW[it["kind"]]
            t1 = time.time()
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
            gen_s = time.time() - t1
            new = out[0, ids.shape[1]:].tolist()
            stop = "eos" if new and new[-1] in eos else ("max_new_tokens" if len(new) >= max_new else "other")
            row = {
                "name": it["name"], "mode": mode, "file": f["file"], "sha256": sha, "image_size": list(img.size),
                "target_sizes": inputs["target_sizes"].tolist() if "target_sizes" in inputs else None,
                "n_input": int(ids.shape[1]), "n_image_tokens": n_img, "max_new_tokens": max_new,
                "gen_ids": new, "n_gen": len(new), "stop": stop,
                "text": processor.decode(new, skip_special_tokens=True),
                "text_raw": processor.decode(new, skip_special_tokens=False),
                "prompt_ids_head": ids[0, :12].tolist(), "prompt_ids_tail": ids[0, -24:].tolist(),
                "seconds": round(gen_s, 2), "dtype": args.dtype, "fast_dwconv": not args.orig_conv,
            }
            with open(args.out, "a") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"{it['name']:14s} {mode:6s} in {row['n_input']:5d} img {n_img:5d} gen {len(new):4d} "
                  f"{stop:14s} {gen_s:7.1f}s | {row['text'][:70]!r}", flush=True)


if __name__ == "__main__":
    main()
