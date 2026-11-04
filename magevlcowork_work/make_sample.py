#!/usr/bin/env python3
"""Draw the sample screenshot of the Mage-VL-Cowork card: a 1920 x 1080 settings window written for this conversion
(no third-party screen). DejaVu Sans from matplotlib's bundled copy (Bitstream Vera / DejaVu license).

  python make_sample.py OUT.png [--boxes BOXES.json]

The window: a title bar, a sidebar (General, Appearance, Downloads, Privacy, Notifications, About), a main panel with
two text fields and two toggles, and the buttons Cancel and Save changes. --boxes writes each clickable element's box
[x, y, w, h] in pixels, so a reply's point can be checked (cowork_step.py prints the point in these pixels).

The PNG's bytes depend on the font and the rasterizer: matplotlib 3.11.2 with Pillow 13.0.0 write the model card's
samples/settings.png (54,259 B, sha256 f9995ded6132ebcfc1a1430524ad8524e6c6f68d1c398200dc7b41be6ed8cc33); matplotlib
3.10.9 with Pillow 12.3.0 write other bytes.
"""
import argparse
import json
import os

import matplotlib
from PIL import Image, ImageDraw, ImageFont

TTF = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
W, H = 1920, 1080


def font(size, bold=False):
  return ImageFont.truetype(os.path.join(TTF, "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"), size)


def main():
  ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  ap.add_argument("out")
  ap.add_argument("--boxes", default=None)
  a = ap.parse_args()
  im = Image.new("RGB", (W, H), (236, 239, 244))
  d = ImageDraw.Draw(im)
  boxes = {}
  # desktop top bar
  d.rectangle([0, 0, W, 36], fill=(40, 44, 52))
  d.text((24, 8), "Files    Edit    View    Window    Help", fill=(230, 230, 230), font=font(18))
  d.text((W - 190, 8), "Tue 14:05", fill=(230, 230, 230), font=font(18))
  # window
  x0, y0, x1, y1 = 260, 110, 1660, 980
  d.rectangle([x0, y0, x1, y1], fill=(255, 255, 255), outline=(200, 204, 212), width=2)
  d.rectangle([x0, y0, x1, y0 + 56], fill=(246, 247, 249), outline=(200, 204, 212), width=2)
  for i, c in enumerate(((236, 95, 90), (245, 190, 79), (98, 197, 84))):
    d.ellipse([x0 + 20 + 30 * i, y0 + 20, x0 + 36 + 30 * i, y0 + 36], fill=c)
  d.text((x0 + 600, y0 + 16), "Settings", fill=(30, 30, 30), font=font(22, bold=True))
  # sidebar
  sx1 = x0 + 330
  d.rectangle([x0 + 2, y0 + 58, sx1, y1 - 2], fill=(244, 245, 248))
  items = ["General", "Appearance", "Downloads", "Privacy", "Notifications", "About"]
  for i, name in enumerate(items):
    ty = y0 + 90 + 64 * i
    box = [x0 + 18, ty - 8, sx1 - x0 - 36, 52]
    if name == "General":
      d.rounded_rectangle([box[0], box[1], box[0] + box[2], box[1] + box[3]], radius=10, fill=(214, 228, 252))
    d.text((x0 + 44, ty + 4), name, fill=(25, 25, 25), font=font(24))
    boxes[f"sidebar:{name}"] = box
  # main panel: General
  mx = sx1 + 50
  d.text((mx, y0 + 90), "General", fill=(20, 20, 20), font=font(34, bold=True))
  d.text((mx, y0 + 170), "Device name", fill=(60, 60, 60), font=font(22))
  d.rounded_rectangle([mx, y0 + 205, mx + 620, y0 + 255], radius=8, outline=(180, 186, 196), width=2)
  d.text((mx + 16, y0 + 216), "Studio Mac", fill=(30, 30, 30), font=font(22))
  boxes["field:Device name"] = [mx, y0 + 205, 620, 50]
  d.text((mx, y0 + 290), "Email address", fill=(60, 60, 60), font=font(22))
  d.rounded_rectangle([mx, y0 + 325, mx + 620, y0 + 375], radius=8, outline=(180, 186, 196), width=2)
  d.text((mx + 16, y0 + 336), "name@example.com", fill=(150, 150, 150), font=font(22))
  boxes["field:Email address"] = [mx, y0 + 325, 620, 50]
  for i, (label, on) in enumerate((("Open files after download", True), ("Check for updates automatically", False))):
    ty = y0 + 430 + 80 * i
    d.text((mx, ty + 6), label, fill=(30, 30, 30), font=font(22))
    tx = mx + 760
    d.rounded_rectangle([tx, ty, tx + 76, ty + 40], radius=20, fill=(52, 120, 246) if on else (205, 208, 214))
    kx = tx + 40 if on else tx + 4
    d.ellipse([kx, ty + 4, kx + 32, ty + 36], fill=(255, 255, 255))
    boxes[f"toggle:{label}"] = [tx, ty, 76, 40]
  # buttons
  by = y1 - 90
  d.rounded_rectangle([x1 - 420, by, x1 - 250, by + 52], radius=10, outline=(180, 186, 196), width=2, fill=(255, 255, 255))
  d.text((x1 - 375, by + 12), "Cancel", fill=(30, 30, 30), font=font(22))
  boxes["button:Cancel"] = [x1 - 420, by, 170, 52]
  d.rounded_rectangle([x1 - 230, by, x1 - 40, by + 52], radius=10, fill=(52, 120, 246))
  d.text((x1 - 208, by + 12), "Save changes", fill=(255, 255, 255), font=font(22))
  boxes["button:Save changes"] = [x1 - 230, by, 190, 52]
  im.save(a.out, format="PNG", optimize=False)
  if a.boxes:
    json.dump({"size": [W, H], "boxes": boxes}, open(a.boxes, "w"), indent=1)


if __name__ == "__main__":
  main()
