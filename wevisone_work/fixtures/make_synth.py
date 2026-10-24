#!/usr/bin/env python3
"""Draw the two synthetic fixture images deterministically and write their ground truth.

syn_a = 640x480 (landscape), syn_b = 480x640 (portrait): coloured rectangles and circles with
different counts per colour, plus large black text "EXIT 12". Placement comes from a seeded
random.Random, the text font is Pillow's bundled default font (ImageFont.load_default), so the
pixels depend only on the Pillow version (pinned 12.2.0 in the reference venv).

Usage: python -I make_synth.py <out_dir>   -> <out_dir>/syn_a.png, syn_b.png, synth_gt.json
"""
import json
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BACKGROUND = (236, 236, 232)
COLORS = {
    "red": (220, 40, 40),
    "blue": (40, 80, 220),
    "green": (40, 170, 70),
    "yellow": (240, 200, 30),
}
TEXT = "EXIT 12"

SPECS = {
    # name: (width, height, seed, shapes [(colour, kind, count)], text band (y0, y1), shape band (y0, y1))
    "syn_a": (640, 480, 1201, [("red", "rectangle", 3), ("blue", "circle", 4), ("green", "circle", 1)],
              (370, 470), (10, 360)),
    "syn_b": (480, 640, 1202, [("red", "rectangle", 5), ("blue", "circle", 2), ("yellow", "rectangle", 1)],
              (10, 110), (130, 630)),
}


def overlaps(a, b, margin=12):
    return not (a[2] + margin <= b[0] or b[2] + margin <= a[0] or a[3] + margin <= b[1] or b[3] + margin <= a[1])


def draw_one(name, width, height, seed, shapes, text_band, shape_band):
    rng = random.Random(seed)
    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    placed = []
    objects = []
    for colour, kind, count in shapes:
        for _ in range(count):
            for _attempt in range(10000):
                if kind == "rectangle":
                    w, h = rng.randint(50, 110), rng.randint(40, 90)
                else:
                    r = rng.randint(25, 45)
                    w = h = 2 * r
                x0 = rng.randint(10, width - 10 - w)
                y0 = rng.randint(shape_band[0], shape_band[1] - h)
                box = (x0, y0, x0 + w, y0 + h)
                if not any(overlaps(box, other) for other in placed):
                    break
            else:
                raise RuntimeError(f"{name}: no room for {colour} {kind}")
            placed.append(box)
            if kind == "rectangle":
                draw.rectangle([box[0], box[1], box[2] - 1, box[3] - 1], fill=COLORS[colour])
            else:
                draw.ellipse([box[0], box[1], box[2] - 1, box[3] - 1], fill=COLORS[colour])
            objects.append({"label": f"{colour} {kind}", "box_xyxy": list(box)})
    font = ImageFont.load_default(size=72)
    left, top, right, bottom = draw.textbbox((0, 0), TEXT, font=font)
    tx = (width - (right - left)) // 2 - left
    ty = text_band[0] + ((text_band[1] - text_band[0]) - (bottom - top)) // 2 - top
    draw.text((tx, ty), TEXT, fill=(0, 0, 0), font=font)
    words = []
    cursor = 0
    for word in TEXT.split(" "):
        start = TEXT.index(word, cursor)
        cursor = start + len(word)
        prefix_w = draw.textlength(TEXT[:start], font=font)
        wl, wt, wr, wb = draw.textbbox((tx + prefix_w, ty), word, font=font)
        words.append({"text": word, "box_xyxy": [int(wl), int(wt), int(wr), int(wb)]})
    return image, {"size": [width, height], "seed": seed, "objects": objects, "text": TEXT, "words": words}


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "img")
    out.mkdir(parents=True, exist_ok=True)
    gt = {}
    for name, spec in SPECS.items():
        image, truth = draw_one(name, *spec)
        image.save(out / f"{name}.png", optimize=False)
        gt[name] = truth
    (out / "synth_gt.json").write_text(json.dumps(gt, indent=1) + "\n")
    print(json.dumps({k: {"objects": len(v["objects"]), "words": v["words"]} for k, v in gt.items()}))


if __name__ == "__main__":
    main()
