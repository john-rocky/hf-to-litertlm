#!/usr/bin/env python3
"""Upstream runs of MiniCPM-OCR-1.0 on an app-side tiling that a 448x448 fast_vlm bundle could take: the processor's own
slice grid (get_sliced_grid) cut from the original picture, every tile resized to GxG, the thumbnail padded to a
white square and resized to GxG, and the processor's text layout around them:

  <image_id>0</image_id><image>[thumb]</image><slice>[t]</slice><slice>[t]</slice>\\n<slice>[t]</slice>...\\n{prompt}

([x] = (G/28)^2 image pads; rows of slices joined by '\\n'; MiniCPMOcrProcessor.replace_image_token). Mode name
"t448". Greedy, the same generation settings and row format as hf_oracle.py (fast_dwconv on).

  python hf_tiles.py --snap SNAPSHOT --fixtures FIX.json[:DIR] ... --out runs.jsonl [--only a,b] [--g 448]
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
from hf_oracle import IMAGE_PAD, MAX_NEW, PAGE_MAX_NEW, load_items  # noqa: E402


def pad_square(img, g):
    s = max(img.size)
    sq = Image.new("RGB", (s, s), "white")
    sq.paste(img, ((s - img.width) // 2, (s - img.height) // 2))
    return sq.resize((g, g), Image.BICUBIC)


def tiles(img, grid, g):
    rows, cols = grid
    W, H = img.size
    out = []
    for r in range(rows):
        for c in range(cols):
            box = (round(W * c / cols), round(H * r / rows), round(W * (c + 1) / cols), round(H * (r + 1) / rows))
            out.append(img.crop(box).resize((g, g), Image.BICUBIC))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", required=True)
    ap.add_argument("--fixtures", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=None)
    ap.add_argument("--g", type=int, default=448)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    fast_dwconv.install()
    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(args.snap)
    ip, tok = proc.image_processor, proc.tokenizer
    model = AutoModelForImageTextToText.from_pretrained(args.snap, dtype=torch.float32).eval()
    n_tok = (args.g // 28) ** 2
    mode = f"t{args.g}"
    only = set(args.only.split(",")) if args.only else None
    for it in load_items(args.fixtures):
        if only and it["name"] not in only:
            continue
        f = it["files"]["native"]
        path = os.path.join(it["_dir"], f["file"])
        img = Image.open(path).convert("RGB")
        grid = ip.get_sliced_grid((img.height, img.width), ip.max_slice_nums, ip.scale_resolution)
        if grid is None:
            print(f"{it['name']}: the processor does not slice it ({img.size}); skipped", flush=True)
            continue
        subs = [pad_square(img, args.g)] + tiles(img, grid, args.g)
        strips = []
        for s in subs:
            feats = ip(images=[s], return_tensors="pt")
            assert feats["target_sizes"].tolist() == [[args.g // 14, args.g // 14]], feats["target_sizes"]
            strips.append(feats["pixel_values"])
        pixel_values = torch.cat(strips, dim=-1)
        target_sizes = torch.tensor([[args.g // 14, args.g // 14]] * len(subs), dtype=torch.int32)
        pad = "<|image_pad|>" * n_tok
        rows = ["".join(f"<slice>{pad}</slice>" for _ in range(grid[1])) for _ in range(grid[0])]
        image_text = f"<image_id>0</image_id><image>{pad}</image>" + "\n".join(rows)
        prompt = (f"<|im_start|>user\n{image_text}\n{it['prompt']}<|im_end|>\n"
                  f"<|im_start|>assistant\n<think>\n\n</think>\n\n")
        ids = tok(prompt, add_special_tokens=False, return_tensors="pt").input_ids
        n_img = int((ids == IMAGE_PAD).sum())
        assert n_img == n_tok * len(subs), (n_img, len(subs))
        max_new = PAGE_MAX_NEW if it["name"].startswith("page") else MAX_NEW[it["kind"]]
        t1 = time.time()
        with torch.no_grad():
            out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=pixel_values,
                                 target_sizes=target_sizes, max_new_tokens=max_new, do_sample=False)
        gen_s = time.time() - t1
        new = out[0, ids.shape[1]:].tolist()
        eos = set(model.generation_config.eos_token_id)
        stop = "eos" if new and new[-1] in eos else ("max_new_tokens" if len(new) >= max_new else "other")
        row = {"name": it["name"], "mode": mode, "file": f["file"],
               "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(), "image_size": list(img.size),
               "grid": grid, "n_subimages": len(subs), "n_input": int(ids.shape[1]), "n_image_tokens": n_img,
               "max_new_tokens": max_new, "gen_ids": new, "n_gen": len(new), "stop": stop,
               "text": tok.decode(new, skip_special_tokens=True), "text_raw": tok.decode(new),
               "seconds": round(gen_s, 2), "dtype": "fp32", "fast_dwconv": True}
        with open(args.out, "a") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{it['name']:14s} {mode} grid {grid} subs {len(subs)} img {n_img} gen {len(new)} {stop} "
              f"{gen_s:.1f}s | {row['text'][:70]!r}", flush=True)


if __name__ == "__main__":
    main()
