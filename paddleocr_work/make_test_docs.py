"""Render synthetic document test images for the PaddleOCR-VL gate."""
import os
from PIL import Image, ImageDraw, ImageFont

OUT = os.path.join(os.path.dirname(__file__), "testdocs")
os.makedirs(OUT, exist_ok=True)

FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_B = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"


def para():
  img = Image.new("RGB", (960, 540), "white")
  d = ImageDraw.Draw(img)
  f_h = ImageFont.truetype(FONT_B, 34)
  f = ImageFont.truetype(FONT, 26)
  d.text((40, 30), "Quarterly Report 2026", font=f_h, fill="black")
  lines = [
      "Revenue increased by 18.4% to $2,315 million in Q2 2026,",
      "driven by strong demand for on-device AI accelerators.",
      "Operating margin improved from 21.3% to 24.7%, while",
      "R&D spending reached $412 million, up 9% year over year.",
      "The board approved a dividend of $0.85 per share, payable",
      "on August 15, 2026 to shareholders of record.",
  ]
  y = 110
  for ln in lines:
    d.text((40, y), ln, font=f, fill="black")
    y += 44
  d.text((40, y + 30), "Contact: ir@example.com  |  +1 (555) 010-2345", font=f, fill="black")
  img.save(os.path.join(OUT, "para.png"))


def table():
  img = Image.new("RGB", (760, 360), "white")
  d = ImageDraw.Draw(img)
  f_h = ImageFont.truetype(FONT_B, 24)
  f = ImageFont.truetype(FONT, 24)
  cols = [30, 280, 480, 730]
  rows = [30, 80, 130, 180, 230, 280]
  head = ["Product", "Units", "Revenue"]
  data = [
      ["Sensor A1", "12,400", "$310,000"],
      ["Module B2", "8,150", "$652,000"],
      ["Kit C3", "3,020", "$151,000"],
      ["Probe D4", "940", "$94,000"],
      ["Total", "24,510", "$1,207,000"],
  ]
  for y in rows:
    d.line([(cols[0], y), (cols[-1], y)], fill="black", width=2)
  d.line([(cols[0], rows[-1]), (cols[-1], rows[-1])], fill="black", width=2)
  for x in cols:
    d.line([(x, rows[0]), (x, rows[-1])], fill="black", width=2)
  for j, h in enumerate(head):
    d.text((cols[j] + 12, rows[0] + 12), h, font=f_h, fill="black")
  for i, row in enumerate(data):
    for j, cell in enumerate(row):
      d.text((cols[j] + 12, rows[i + 1] + 12), cell, font=f, fill="black")
  img.save(os.path.join(OUT, "table.png"))


para()
table()
print("wrote", OUT)
