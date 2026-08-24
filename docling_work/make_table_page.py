"""Render a synthetic document page with a known-ground-truth table.

The table is the structural gate for the .litertlm conversion: 5 cols x 6 rows
(header + 4 products + total). Judged on OTSL grid shape + cell values, not
token match. Page is square (1024x1024) so the runtime's stretch-to-512 does
not distort it — this isolates conversion damage from aspect distortion.

    .venv/bin/python docling_work/make_table_page.py
"""
from PIL import Image, ImageDraw, ImageFont

OUT = "docling_work/table_page.png"
W = H = 1024
FONT = "/System/Library/Fonts/Helvetica.ttc"

TITLE = "Quarterly Sales Report 2025"
PARA = ("This report summarizes unit sales for the first three quarters. "
        "Phones remain the strongest product line across all quarters.")
HEADERS = ["Product", "Q1", "Q2", "Q3", "Total"]
ROWS = [
    ["Laptops", "120", "135", "142", "397"],
    ["Tablets", "80", "95", "88", "263"],
    ["Phones", "210", "198", "225", "633"],
    ["Monitors", "45", "52", "61", "158"],
    ["Total", "455", "480", "516", "1451"],
]

img = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(img)
f_title = ImageFont.truetype(FONT, 44)
f_body = ImageFont.truetype(FONT, 30)
f_cell = ImageFont.truetype(FONT, 32)

d.text((80, 70), TITLE, font=f_title, fill="black")
# wrap paragraph at ~55 chars
words, lines, cur = PARA.split(), [], ""
for w_ in words:
    if len(cur) + len(w_) + 1 > 60:
        lines.append(cur); cur = w_
    else:
        cur = (cur + " " + w_).strip()
lines.append(cur)
y = 170
for ln in lines:
    d.text((80, y), ln, font=f_body, fill="black"); y += 44

# table grid
x0, y0 = 80, y + 60
col_w = [260, 130, 130, 130, 170]
row_h = 64
n_rows = len(ROWS) + 1
x_edges = [x0]
for cw in col_w:
    x_edges.append(x_edges[-1] + cw)
y_edges = [y0 + i * row_h for i in range(n_rows + 1)]
for xe in x_edges:
    d.line([(xe, y_edges[0]), (xe, y_edges[-1])], fill="black", width=3)
for ye in y_edges:
    d.line([(x_edges[0], ye), (x_edges[-1], ye)], fill="black", width=3)
# header shading
d.rectangle([x_edges[0] + 2, y_edges[0] + 2, x_edges[-1] - 2, y_edges[1] - 2], fill=(230, 230, 230))
for xe in x_edges:
    d.line([(xe, y_edges[0]), (xe, y_edges[-1])], fill="black", width=3)
d.line([(x_edges[0], y_edges[0]), (x_edges[-1], y_edges[0])], fill="black", width=3)
d.line([(x_edges[0], y_edges[1]), (x_edges[-1], y_edges[1])], fill="black", width=3)


def cell(r, c, text, bold=False):
    cx = x_edges[c] + 16
    cy = y_edges[r] + (row_h - 32) // 2
    d.text((cx, cy), text, font=f_cell, fill="black")


for c, h_ in enumerate(HEADERS):
    cell(0, c, h_)
for r, row in enumerate(ROWS, start=1):
    for c, v in enumerate(row):
        cell(r, c, v)

d.text((80, y_edges[-1] + 50),
       "Table 1: Unit sales by product and quarter.", font=f_body, fill="black")
img.save(OUT)
print("wrote", OUT, img.size)
