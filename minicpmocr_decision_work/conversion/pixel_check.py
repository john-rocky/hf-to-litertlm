#!/usr/bin/env python3
"""The host's picture path against the author's processor path (transformers 5.10.4), no model. For each picture in bench/ and the three official examples:
 (a) decode: torchvision decode_image(path, RGB) against PIL Image.open(path).convert("RGB")
 (b) resize: tvF.resize as the processor calls it, against resize_uint8 of host/decide_host/picture.py (the numpy port) and against PIL BICUBIC
 (c) pixel_values: the processor's own output against to_pixel_values of the port's bytes, on one picture of 40
Shape classes: up = both axes grow, down = both shrink, mixed, same = no resize. Needs fetch_inputs.py first; it reads the pictures that are in bench/.

    $REF -I conversion/pixel_check.py <work> [--limit N]      -> results/pixel_check.json
"""
import argparse
import collections
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision.io import ImageReadMode, decode_image
from torchvision.transforms.v2 import functional as tvF
from transformers import AutoProcessor

ap = argparse.ArgumentParser()
ap.add_argument("work", type=Path, help="the work directory")
ap.add_argument("--limit", type=int, help="an even sample of this many pictures of bench/")
a = ap.parse_args()
WORK = a.work.resolve()
spec = importlib.util.spec_from_file_location("picture", WORK / "host/decide_host/picture.py")
pic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pic)
BENCH = WORK / "bench"
rows = [json.loads(x) for x in (BENCH / "data/jevocrbench.jsonl").read_text(encoding="utf-8").splitlines()]
paths = sorted({r["image_path"] for r in rows} | {r["render_image_path"] for r in rows})
paths = [p for p in paths if os.path.exists(BENCH / p)]
if a.limit:
    paths = paths[:: max(1, len(paths) // a.limit)]
examples = sorted((WORK / "author/scripts/minicpm-ocr-jev/examples").glob("*/*.png"))
processor = AutoProcessor.from_pretrained(WORK / "snapshot").image_processor


def shape_class(h, w, H, W):
    if (h, w) == (H, W):
        return "same"
    if H >= h and W >= w:
        return "up"
    if H <= h and W <= w:
        return "down"
    return "mixed"


stats = collections.defaultdict(collections.Counter)
decode_diff, port_bad = [], []
t0 = time.time()
for i, p in enumerate([BENCH / x for x in paths] + examples):
    t = decode_image(str(p), mode=ImageReadMode.RGB)
    with Image.open(p) as im:
        pil = np.asarray(im.convert("RGB"))
    tn = t.permute(1, 2, 0).numpy()
    if tn.shape != pil.shape or not np.array_equal(tn, pil):
        decode_diff.append({"file": p.name, "shape": list(pil.shape), "max": int(np.abs(tn.astype(int) - pil.astype(int)).max()) if tn.shape == pil.shape else None})
    h, w = tn.shape[:2]
    H, W = pic.smart_resize(h, w)
    s = stats[shape_class(h, w, H, W)]
    s["files"] += 1
    author = tvF.resize(t, [H, W], interpolation=tvF.InterpolationMode.BICUBIC, antialias=True).permute(1, 2, 0).numpy()
    port = pic.resize_uint8(tn, H, W)
    d = np.abs(author.astype(int) - port.astype(int))
    s["px"] += d.size
    s["port_px_diff"] += int((d > 0).sum())
    s["port_max"] = max(s["port_max"], int(d.max()))
    if d.max() > 0:
        s["port_files_diff"] += 1
        if len(port_bad) < 20:
            port_bad.append({"file": p.name, "wh": [w, h], "HW": [H, W], "max": int(d.max()), "n": int((d > 0).sum())})
    resized = np.asarray(Image.fromarray(tn).resize((W, H), Image.Resampling.BICUBIC))
    d2 = np.abs(author.astype(int) - resized.astype(int))
    s["pil_px_diff"] += int((d2 > 0).sum())
    s["pil_max"] = max(s["pil_max"], int(d2.max()))
    if d2.max() > 0:
        s["pil_files_diff"] += 1
    if i % 40 == 0:
        out = processor(images=[str(p)], return_tensors="pt")
        mine = pic.to_pixel_values(port)
        s["pv_checked"] += 1
        s["pv_bit_equal"] += int(np.array_equal(out["pixel_values"].numpy(), mine))
        grid = out["image_grid_thw"].tolist()[0]
        if grid != [1, H // 16, W // 16]:
            print("GRID", p.name, grid, H, W, flush=True)
report = {"pictures": len(paths) + len(examples), "n_decode_diff": len(decode_diff), "decode_diff": decode_diff[:50], "classes": {k: dict(v) for k, v in stats.items()}, "port_bad": port_bad, "seconds": round(time.time() - t0, 1)}
(WORK / "results/pixel_check.json").write_text(json.dumps(report, indent=1))
print(json.dumps({k: v for k, v in report.items() if k != "port_bad"}))
n_port = sum(c["port_px_diff"] for c in report["classes"].values())
n_pil = sum(c.get("pil_files_diff", 0) for c in report["classes"].values())
print(f"PIXEL_CHECK {report['pictures']} pictures: the port differs from the author's path on {n_port} values; Pillow differs on {n_pil} files, by up to {max(c['pil_max'] for c in report['classes'].values())} levels; decode differs on {len(decode_diff)} files")
