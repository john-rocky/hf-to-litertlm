#!/usr/bin/env python3
"""Render the r1 OCR fixtures with known ground truth (and their fixed-size variants).

  python make_fixtures.py OUT_DIR

Writes OUT_DIR/<name>.png (the region as rendered, native size), OUT_DIR/<name>_s448.png (stretched to
448x448, PIL BICUBIC), OUT_DIR/<name>_p448.png (padded to a white square, centred, then 448x448 BICUBIC) and
OUT_DIR/fixtures.json (name, kind, prompt, ground truth, native size, sha256 of every PNG).

Every text is written for this conversion (no third-party text). Fonts: DejaVu (matplotlib's bundled copy, Bitstream
Vera / DejaVu license) for Latin script and the matplotlib mathtext formulas; Hiragino Sans GB (macOS system font)
for the Chinese paragraph. The two photographs of the transformers model doc are fetched separately
(fetch_doc_images.py) and have no ground truth here.
"""
import hashlib
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image, ImageDraw, ImageFilter, ImageFont  # noqa: E402

MPL_TTF = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
SERIF = os.path.join(MPL_TTF, "DejaVuSerif.ttf")
SANS = os.path.join(MPL_TTF, "DejaVuSans.ttf")
SANS_BOLD = os.path.join(MPL_TTF, "DejaVuSans-Bold.ttf")
MONO = os.path.join(MPL_TTF, "DejaVuSansMono.ttf")
ZH = "/System/Library/Fonts/Hiragino Sans GB.ttc"
G = 448

TEXT_PROMPT = "Text Recognition:"
TABLE_PROMPT = "Table Recognition:"
FORMULA_PROMPT = "Formula Recognition:"

PARA_EN = ("The survey collected 1,284 samples from 37 rivers between April and September. Nitrate levels "
           "fell in 22 of the rivers compared with the previous year, while phosphate levels rose in 9. The "
           "largest change was recorded at the Kessel station, where the average nitrate concentration dropped "
           "from 4.8 mg/L to 3.1 mg/L. Samples were analysed within 48 hours of collection, and every result was "
           "checked by a second laboratory.")
PARA_EN_2 = ("Field teams used the same sampling bottles at every station. Each bottle was rinsed three times "
             "with river water before it was filled, sealed and stored at 4 degrees Celsius. Stations that could "
             "not be reached after heavy rain were visited again within one week.")
PARA_EN_3 = ("The full data set, including the 61 samples that failed the second check, is available on request. "
             "The next survey will start in April 2026 and will add 12 stations on the upper Arden.")
PARA_ZH = ("本次调查于2025年4月至9月进行，共从37条河流采集了1284份水样。与上一年相比，22条河流的硝酸盐含量下降，"
           "9条河流的磷酸盐含量上升。所有样品均在采集后48小时内完成分析，并由第二家实验室复核。")
TABLE = [["Station", "Samples", "Nitrate (mg/L)", "Phosphate (mg/L)"],
         ["Kessel", "212", "3.1", "0.42"],
         ["Arden", "185", "2.7", "0.38"],
         ["Morrow", "164", "4.0", "0.51"],
         ["Lisle", "143", "1.9", "0.29"]]
FORMULAS = {
    "formula_1": r"E = \frac{1}{2} m v^{2} + \int_{0}^{h} m g \, dz",
    "formula_2": r"\sum_{k=1}^{n} k^{2} = \frac{n(n+1)(2n+1)}{6}",
}
RECEIPT_LINES = ["GREENLEAF MARKET", "214 Harbor Road, Kessel", "Date: 2025-03-14   Time: 18:42", "",
                 "Apples 1kg              3.20", "Bread                   2.45", "Milk 2L                 1.90",
                 "Coffee beans            8.75", "", "Subtotal               16.30", "Tax                     1.30",
                 "TOTAL                  17.60", "", "Thank you for shopping!"]
RECEIPT_FIELDS = {"store_name": "GREENLEAF MARKET", "date": "2025-03-14", "total": "17.60"}


def kie_prompt(fields):
    q = ", ".join(f'"{f}"' for f in fields)
    return (f"Extract the {q}. Format the output as a JSON object with keys: {q}. "
            f'If an item is not in the image, use "###".')


def wrap(draw, text, font, width, by_char=False):
    units = list(text) if by_char else text.split(" ")
    sep = "" if by_char else " "
    lines, cur = [], ""
    for u in units:
        trial = (cur + sep + u) if cur else u
        if draw.textlength(trial, font=font) <= width:
            cur = trial
        else:
            lines.append(cur)
            cur = u
    if cur:
        lines.append(cur)
    return lines


def render_lines(lines, font, width=None, margin=16, spacing=1.35):
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    asc, desc = font.getmetrics()
    lh = int(round((asc + desc) * spacing))
    w = width if width is not None else int(max(probe.textlength(s, font=font) for s in lines))
    img = Image.new("RGB", (w + 2 * margin, lh * len(lines) + 2 * margin - (lh - asc - desc)), "white")
    d = ImageDraw.Draw(img)
    for i, s in enumerate(lines):
        d.text((margin, margin + i * lh), s, font=font, fill="black")
    return img


def render_para(text, font_path, size, width, by_char=False):
    font = ImageFont.truetype(font_path, size)
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    return render_lines(wrap(probe, text, font, width, by_char), font, width)


def render_table(rows, size=22, pad=12):
    font = ImageFont.truetype(SANS, size)
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    ncol = len(rows[0])
    colw = [int(max(probe.textlength(r[c], font=font) for r in rows)) + 2 * pad for c in range(ncol)]
    asc, desc = font.getmetrics()
    rowh = asc + desc + 2 * pad
    m = 12
    img = Image.new("RGB", (sum(colw) + 2 * m + 1, rowh * len(rows) + 2 * m + 1), "white")
    d = ImageDraw.Draw(img)
    y = m
    for r, row in enumerate(rows):
        x = m
        for c, cell in enumerate(row):
            d.rectangle([x, y, x + colw[c], y + rowh], outline="black", width=1)
            f = ImageFont.truetype(SANS_BOLD, size) if r == 0 else font
            d.text((x + pad, y + pad), cell, font=f, fill="black")
            x += colw[c]
        y += rowh
    return img


def render_formula(tex, fontsize=30, dpi=100, margin=16):
    fig = plt.figure(figsize=(0.01, 0.01), dpi=dpi)
    fig.text(0, 0, f"${tex}$", fontsize=fontsize)
    path = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"_formula_{os.getpid()}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.0, facecolor="white")
    plt.close(fig)
    img = Image.open(path).convert("RGB")
    os.remove(path)
    out = Image.new("RGB", (img.width + 2 * margin, img.height + 2 * margin), "white")
    out.paste(img, (margin, margin))
    return out


def render_receipt():
    font = ImageFont.truetype(MONO, 22)
    bold = ImageFont.truetype(SANS_BOLD, 28)
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    w = int(max(probe.textlength(s, font=font) for s in RECEIPT_LINES)) + 48
    asc, desc = font.getmetrics()
    lh = int((asc + desc) * 1.5)
    img = Image.new("RGB", (w, lh * (len(RECEIPT_LINES) + 1) + 48), "white")
    d = ImageDraw.Draw(img)
    y = 24
    for i, s in enumerate(RECEIPT_LINES):
        f = bold if i == 0 else font
        tw = probe.textlength(s, font=f)
        x = (w - tw) / 2 if i in (0, 1, 2, len(RECEIPT_LINES) - 1) else 24
        d.text((x, y), s, font=f, fill="black")
        y += lh + (8 if i == 0 else 0)
    return img


def render_page():
    """A4 at 150 dpi (1240x1754): title + three paragraphs, as one page."""
    W, H, m = 1240, 1754, 110
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    title = "Regional Water Quality Survey 2025"
    tf = ImageFont.truetype(SANS_BOLD, 40)
    d.text((m, m), title, font=tf, fill="black")
    font = ImageFont.truetype(SERIF, 22)
    asc, desc = font.getmetrics()
    lh = int((asc + desc) * 1.45)
    y = m + 90
    for para in (PARA_EN, PARA_EN_2, PARA_EN_3):
        for s in wrap(d, para, font, W - 2 * m):
            d.text((m, y), s, font=font, fill="black")
            y += lh
        y += lh // 2
    return img, "\n".join([title, PARA_EN, PARA_EN_2, PARA_EN_3])


def photo_like(img):
    """Mild capture damage: 1.5 degree rotation on white, blur, a lighting gradient, JPEG 60."""
    import io
    r = img.rotate(1.5, resample=Image.BICUBIC, expand=True, fillcolor="white")
    r = r.filter(ImageFilter.GaussianBlur(0.8))
    grad = Image.linear_gradient("L").resize(r.size).point(lambda v: 255 - v // 5)
    r = Image.composite(r, Image.new("RGB", r.size, (200, 196, 188)), grad)
    buf = io.BytesIO()
    r.save(buf, format="JPEG", quality=60)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def stretch(img):
    return img.resize((G, G), Image.BICUBIC)


def pad_square(img):
    s = max(img.size)
    sq = Image.new("RGB", (s, s), "white")
    sq.paste(img, ((s - img.width) // 2, (s - img.height) // 2))
    return sq.resize((G, G), Image.BICUBIC)


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    items = []

    def add(name, img, kind, prompt, gt):
        items.append((name, img, kind, prompt, gt))

    font = ImageFont.truetype(SERIF, 30)
    line = "Invoice 48213 was issued on 12 March 2025 and is due within 30 days."
    add("line_en", render_lines([line], font), "text", TEXT_PROMPT, line)
    title = "Regional Water Quality Survey 2025"
    add("title_en", render_lines([title], ImageFont.truetype(SANS_BOLD, 44)), "text", TEXT_PROMPT, title)
    para = render_para(PARA_EN, SERIF, 22, 880)
    add("para_en", para, "text", TEXT_PROMPT, PARA_EN)
    add("para_en_small", render_para(PARA_EN, SERIF, 13, 560), "text", TEXT_PROMPT, PARA_EN)
    add("para_en_photo", photo_like(para), "text", TEXT_PROMPT, PARA_EN)
    add("para_zh", render_para(PARA_ZH, ZH, 26, 640, by_char=True), "text", TEXT_PROMPT, PARA_ZH)
    add("table", render_table(TABLE), "table", TABLE_PROMPT, TABLE)
    for name, tex in FORMULAS.items():
        add(name, render_formula(tex), "formula", FORMULA_PROMPT, tex)
    add("receipt", render_receipt(), "kie", kie_prompt(list(RECEIPT_FIELDS)), RECEIPT_FIELDS)
    page, page_gt = render_page()
    add("page_en", page, "text", TEXT_PROMPT, page_gt)

    records = []
    for name, img, kind, prompt, gt in items:
        files = {}
        for suffix, im in (("", img), ("_s448", stretch(img)), ("_p448", pad_square(img))):
            p = os.path.join(out, f"{name}{suffix}.png")
            im.save(p, format="PNG")
            files[suffix.lstrip("_") or "native"] = {"file": os.path.basename(p), "size": list(im.size),
                                                     "sha256": sha256(p)}
        records.append({"name": name, "kind": kind, "prompt": prompt, "gt": gt, "files": files})
        print(f"{name:14s} {kind:8s} native {img.size[0]}x{img.size[1]}")
    meta = {"generator": "make_fixtures.py", "g": G, "resample": "PIL BICUBIC",
            "fonts": {"serif": SERIF, "sans": SANS, "sans_bold": SANS_BOLD, "mono": MONO, "zh": ZH},
            "matplotlib": matplotlib.__version__, "items": records}
    with open(os.path.join(out, "fixtures.json"), "w") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1)
        fh.write("\n")


if __name__ == "__main__":
    main()
