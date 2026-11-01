#!/usr/bin/env python3
"""Draw the synthetic LFM3-VL fixture images deterministically (English text only).

Placement that varies comes from a seeded random.Random; text uses Pillow's bundled default font
(ImageFont.load_default(size=N)), so the pixels depend only on the Pillow version (13.0.0, recorded in
fixtures.json).

  python -I make_synth.py <out_dir>   -> <out_dir>/syn_*.png + <out_dir>/synth_gt.json
"""
import json
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BG = (236, 236, 232)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (220, 40, 40)
BLUE = (40, 80, 220)
GREEN = (30, 150, 60)
YELLOW = (240, 200, 30)
PURPLE = (130, 50, 170)
ORANGE = (240, 130, 20)


def font(size):
  return ImageFont.load_default(size=size)


def centered_text(draw, box, text, fnt, fill):
  left, top, right, bottom = draw.textbbox((0, 0), text, font=fnt)
  x = box[0] + ((box[2] - box[0]) - (right - left)) // 2 - left
  y = box[1] + ((box[3] - box[1]) - (bottom - top)) // 2 - top
  draw.text((x, y), text, fill=fill, font=fnt)
  return [int(v) for v in draw.textbbox((x, y), text, font=fnt)]


def overlaps(a, b, margin=14):
  return not (a[2] + margin <= b[0] or b[2] + margin <= a[0] or a[3] + margin <= b[1] or b[3] + margin <= a[1])


def syn_shapes():
  """640x480: 5 blue circles, 2 red squares at seeded places."""
  w, h, seed = 640, 480, 3101
  rng = random.Random(seed)
  im = Image.new("RGB", (w, h), BG)
  d = ImageDraw.Draw(im)
  placed, objs = [], []
  for colour, rgb, kind, count in (("red", RED, "square", 2), ("blue", BLUE, "circle", 5)):
    for _ in range(count):
      for _try in range(10000):
        s = rng.randint(60, 100)
        x0, y0 = rng.randint(10, w - 10 - s), rng.randint(10, h - 10 - s)
        box = (x0, y0, x0 + s, y0 + s)
        if not any(overlaps(box, o) for o in placed):
          break
      else:
        raise RuntimeError("no room")
      placed.append(box)
      if kind == "square":
        d.rectangle([box[0], box[1], box[2] - 1, box[3] - 1], fill=rgb)
      else:
        d.ellipse([box[0], box[1], box[2] - 1, box[3] - 1], fill=rgb)
      objs.append({"label": f"{colour} {kind}", "box_xyxy": list(box)})
  return im, {"size": [w, h], "seed": seed, "objects": objs}


def syn_exit():
  """640x480: a green sign with white text EXIT 12."""
  w, h = 640, 480
  im = Image.new("RGB", (w, h), BG)
  d = ImageDraw.Draw(im)
  sign = (120, 150, 520, 330)
  d.rectangle(sign, fill=GREEN)
  tb = centered_text(d, sign, "EXIT 12", font(88), WHITE)
  return im, {"size": [w, h], "sign_box_xyxy": list(sign), "text": "EXIT 12", "text_box_xyxy": tb}


def syn_gate():
  """480x640 portrait: black text GATE 47 on white, an arrow under it."""
  w, h = 480, 640
  im = Image.new("RGB", (w, h), WHITE)
  d = ImageDraw.Draw(im)
  tb = centered_text(d, (0, 120, w, 300), "GATE 47", font(84), BLACK)
  d.polygon([(240, 560), (160, 440), (210, 440), (210, 360), (270, 360), (270, 440), (320, 440)], fill=BLACK)
  return im, {"size": [w, h], "text": "GATE 47", "text_box_xyxy": tb}


def syn_bar():
  """640x480: a bar chart titled Visitors, bars Mon 3 / Tue 5 / Wed 8 / Thu 2 on a 0-10 axis."""
  w, h = 640, 480
  im = Image.new("RGB", (w, h), WHITE)
  d = ImageDraw.Draw(im)
  centered_text(d, (0, 10, w, 60), "Visitors", font(36), BLACK)
  x_axis_y, top_y, left_x = 410, 80, 80
  d.line([(left_x, top_y), (left_x, x_axis_y)], fill=BLACK, width=3)
  d.line([(left_x, x_axis_y), (w - 30, x_axis_y)], fill=BLACK, width=3)
  unit = (x_axis_y - top_y) / 10.0
  for v in (0, 5, 10):
    y = int(round(x_axis_y - v * unit))
    d.line([(left_x - 8, y), (left_x, y)], fill=BLACK, width=2)
    centered_text(d, (10, y - 15, left_x - 12, y + 15), str(v), font(24), BLACK)
  bars = [("Mon", 3), ("Tue", 5), ("Wed", 8), ("Thu", 2)]
  out = []
  for i, (lab, v) in enumerate(bars):
    x0 = left_x + 40 + i * 125
    box = (x0, int(round(x_axis_y - v * unit)), x0 + 80, x_axis_y)
    d.rectangle(box, fill=BLUE)
    centered_text(d, (x0 - 20, x_axis_y + 8, x0 + 100, x_axis_y + 50), lab, font(28), BLACK)
    out.append({"label": lab, "value": v, "box_xyxy": list(box)})
  return im, {"size": [w, h], "title": "Visitors", "bars": out}


def syn_table():
  """640x480: a 2-column table Item / Price with Apple 3, Bread 5, Milk 2."""
  w, h = 640, 480
  im = Image.new("RGB", (w, h), WHITE)
  d = ImageDraw.Draw(im)
  rows = [("Item", "Price"), ("Apple", "3"), ("Bread", "5"), ("Milk", "2")]
  x0, x1, x2, y0, rh = 120, 340, 520, 60, 90
  for r, (a, b) in enumerate(rows):
    y = y0 + r * rh
    if r == 0:
      d.rectangle([x0, y, x2, y + rh], fill=(210, 210, 210))
    centered_text(d, (x0, y, x1, y + rh), a, font(40), BLACK)
    centered_text(d, (x1, y, x2, y + rh), b, font(40), BLACK)
  for r in range(len(rows) + 1):
    d.line([(x0, y0 + r * rh), (x2, y0 + r * rh)], fill=BLACK, width=3)
  for x in (x0, x1, x2):
    d.line([(x, y0), (x, y0 + len(rows) * rh)], fill=BLACK, width=3)
  return im, {"size": [w, h], "rows": rows}


def syn_lr():
  """640x480: a red circle on the left half, a blue square on the right half."""
  w, h = 640, 480
  im = Image.new("RGB", (w, h), BG)
  d = ImageDraw.Draw(im)
  circ = (70, 170, 210, 310)
  sq = (440, 180, 560, 300)
  d.ellipse([circ[0], circ[1], circ[2] - 1, circ[3] - 1], fill=RED)
  d.rectangle([sq[0], sq[1], sq[2] - 1, sq[3] - 1], fill=BLUE)
  return im, {"size": [w, h], "objects": [{"label": "red circle", "box_xyxy": list(circ)},
                                          {"label": "blue square", "box_xyxy": list(sq)}]}


def syn_tb():
  """480x640 portrait: a green triangle near the top, an orange square near the bottom."""
  w, h = 480, 640
  im = Image.new("RGB", (w, h), BG)
  d = ImageDraw.Draw(im)
  tri = [(240, 50), (150, 200), (330, 200)]
  sq = (180, 450, 300, 570)
  d.polygon(tri, fill=GREEN)
  d.rectangle([sq[0], sq[1], sq[2] - 1, sq[3] - 1], fill=ORANGE)
  return im, {"size": [w, h], "objects": [{"label": "green triangle", "box_xyxy": [150, 50, 330, 200]},
                                          {"label": "orange square", "box_xyxy": list(sq)}]}


def syn_big():
  """512x512 (already a G=512 square: the native arm equals B512 here): one purple circle on white."""
  w, h = 512, 512
  im = Image.new("RGB", (w, h), WHITE)
  d = ImageDraw.Draw(im)
  c = (106, 106, 406, 406)
  d.ellipse([c[0], c[1], c[2] - 1, c[3] - 1], fill=PURPLE)
  return im, {"size": [w, h], "objects": [{"label": "purple circle", "box_xyxy": list(c)}]}


DRAW = {"syn_shapes": syn_shapes, "syn_exit": syn_exit, "syn_gate": syn_gate, "syn_bar": syn_bar,
        "syn_table": syn_table, "syn_lr": syn_lr, "syn_tb": syn_tb, "syn_big": syn_big}


def main():
  out = Path(sys.argv[1] if len(sys.argv) > 1 else "img")
  out.mkdir(parents=True, exist_ok=True)
  gt = {}
  for name, fn in DRAW.items():
    im, truth = fn()
    im.save(out / f"{name}.png", optimize=False)
    gt[name] = truth
  (out / "synth_gt.json").write_text(json.dumps(gt, indent=1) + "\n")
  print(json.dumps({k: v["size"] for k, v in gt.items()}))
  return 0


if __name__ == "__main__":
  sys.exit(main())
