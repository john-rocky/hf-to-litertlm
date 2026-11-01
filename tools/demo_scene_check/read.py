#!/usr/bin/env python3
"""Read every answer: per request, every label's verdict, point and answer as text, and the points of chosen labels
drawn on the 512 x 512 picture over the truth region. The verdicts come from the score file; this is for the eye.

  python tools/demo_scene_check/read.py DIR/score/all.json --fixtures DIR/fixtures.json \
      --labels "<label>,<label>,..." --out DIR/read

Writes <out>/answers.md (per request: every label's verdict, point, how it was read and the answer's last 200
characters; a line is marked "?" when the reader found several points, read no point from a non-empty answer, or read
a point outside 0-1000: the answers the rule cannot settle, to list with the reason) and <out>/<request>.png (the
picture, the truth mask filled, each chosen label's point as a numbered dot, the legend under the picture). Judge right
and wrong from the score file, not from a scaled-down picture: the dots are for where the points land.
"""
import argparse
import json
import os
import re
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

COLS = [(230, 30, 30), (30, 90, 230), (20, 160, 40), (230, 140, 0), (150, 40, 200), (0, 170, 170), (200, 0, 120),
        (90, 90, 90)]
NUM = r"-?\d+(?:\.\d+)?"


def font(n):
  for p in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
    try:
      return ImageFont.truetype(p, n)
    except OSError:
      pass
  return ImageFont.load_default(size=n)


def n_points(ans):
  return len(re.findall(r"[\[(]\s*" + NUM + r"\s*,\s*" + NUM + r"\s*(?:,\s*" + NUM + r"\s*,\s*" + NUM + r"\s*)?[\])]",
                        ans or ""))


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("score")
  ap.add_argument("--fixtures", required=True)
  ap.add_argument("--labels", required=True)
  ap.add_argument("--out", required=True)
  a = ap.parse_args()
  d = json.load(open(a.score))
  fx = json.load(open(a.fixtures))
  fx_dir = os.path.dirname(os.path.abspath(a.fixtures))
  items = {it["id"]: it for it in fx["items"]}
  rows = [r for r in d["rows"] if r["q_type"] == "place"]
  chosen = [x.strip() for x in a.labels.split(",") if x.strip()]
  os.makedirs(a.out, exist_ok=True)
  md = ["# Answers per request (o right, x wrong, (o) right only in the favourable reading; ? = several points, no "
        "point in a non-empty answer, or a point outside 0-1000)", ""]
  for t, it in items.items():
    rs = [r for r in rows if r["id"] == t]
    md += [f"## {t} — {it['request']} ({it['rel_kind']}, area {it['truth']['area_pct']} %)", ""]
    for r in rs:
      v = "o" if r["ok"] else ("(o)" if r["ok_fav"] else "x")
      tail = (r.get("answer_tail") or "").replace("\n", " ").replace("|", "/")[-200:]
      flag = ""
      if r["point"] is None and tail.strip() and not (r["how"] or "").startswith("think on"):
        flag = " ?no point"
      elif r["point"] is not None and not (0 <= r["point"][0] <= 1000 and 0 <= r["point"][1] <= 1000):
        flag = " ?outside"
      elif n_points(tail) > 1:
        flag = f" ?{n_points(tail)} points"
      md.append(f"- {v}{flag} | {r['label']} | {r['point']} | {r['how']}{(' / ' + r['fav_how']) if r.get('fav_how') else ''}"
                f" | `{tail}`")
    md.append("")
    ph = fx["photos"][it["photo"]]
    im = Image.open(os.path.join(fx_dir, ph["s512"])).convert("RGBA")
    W, H = im.size
    mask = np.array(Image.open(os.path.join(fx_dir, it["truth"]["mask"])).convert("L")) > 127
    m = Image.fromarray(mask.astype(np.uint8) * 255).resize((W, H), Image.NEAREST)
    over = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    over.paste(Image.new("RGBA", (W, H), (255, 255, 0, 110)), (0, 0), m)
    im = Image.alpha_composite(im, over)
    canvas = Image.new("RGB", (W, H + 22 * (len(chosen) + 1)), (255, 255, 255))
    canvas.paste(im.convert("RGB"), (0, 0))
    dr = ImageDraw.Draw(canvas)
    f = font(14)
    dr.text((4, H + 3), f"{t}: {it['request']}", font=f, fill=(0, 0, 0))
    for i, lab in enumerate(chosen):
      r = next((x for x in rs if x["label"] == lab), None)
      col = COLS[i % len(COLS)]
      y = H + 22 * (i + 1) + 3
      v = "-" if r is None else ("o" if r["ok"] else ("(o)" if r["ok_fav"] else "x"))
      dr.text((4, y), f"{i + 1} {lab}: {v} {None if r is None else r['point']}", font=f, fill=col)
      if r is not None and r["point"] is not None:
        x, yy = r["point"][0] / 1000 * W, r["point"][1] / 1000 * H
        if 0 <= x <= W and 0 <= yy <= H:
          dr.ellipse([x - 7, yy - 7, x + 7, yy + 7], outline=(255, 255, 255), width=3, fill=col)
          dr.text((x + 8, yy - 9), str(i + 1), font=font(15), fill=col, stroke_width=2, stroke_fill=(255, 255, 255))
    canvas.save(os.path.join(a.out, t + ".png"))
  open(os.path.join(a.out, "answers.md"), "w").write("\n".join(md) + "\n")
  print("wrote", a.out, len(items), "requests")
  return 0


if __name__ == "__main__":
  sys.exit(main())
