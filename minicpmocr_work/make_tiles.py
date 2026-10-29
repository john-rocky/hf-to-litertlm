#!/usr/bin/env python3
"""Write the t448 tiling of hf_tiles.py as PNGs (thumbnail + slices, every one GxG) and a manifest, for the runtime.

  python make_tiles.py --fixtures FIX.json[:DIR] ... --out DIR [--only a,b] [--g 448]

Uses hf_tiles.py's pad_square / tiles and the processor's get_sliced_grid (max_slice_nums 9, scale 448), so the
pixels are the ones the upstream t448 runs saw. Manifest: DIR/tiles.json = name, grid, the PNG list in prompt order
(thumbnail first, then the slices row by row) with sha256, and the prompt.
"""
import argparse
import hashlib
import json
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hf_oracle import load_items  # noqa: E402
from hf_tiles import pad_square, tiles  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", required=True)
    ap.add_argument("--fixtures", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=None)
    ap.add_argument("--g", type=int, default=448)
    a = ap.parse_args()
    from transformers import AutoProcessor
    ip = AutoProcessor.from_pretrained(a.snap).image_processor
    os.makedirs(a.out, exist_ok=True)
    only = set(a.only.split(",")) if a.only else None
    man = []
    for it in load_items(a.fixtures):
        if only and it["name"] not in only:
            continue
        img = Image.open(os.path.join(it["_dir"], it["files"]["native"]["file"])).convert("RGB")
        grid = ip.get_sliced_grid((img.height, img.width), ip.max_slice_nums, ip.scale_resolution)
        if grid is None:
            continue
        subs = [pad_square(img, a.g)] + tiles(img, grid, a.g)
        files = []
        for k, s in enumerate(subs):
            p = os.path.join(a.out, f"{it['name']}_t{a.g}_{k:02d}.png")
            s.save(p)
            files.append({"file": os.path.basename(p), "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest()})
        man.append({"name": it["name"], "kind": it["kind"], "prompt": it["prompt"], "grid": grid, "files": files})
        print(it["name"], grid, len(files))
    json.dump({"g": a.g, "items": man}, open(os.path.join(a.out, "tiles.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
