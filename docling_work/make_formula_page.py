"""Render a synthetic page with one LARGE display formula, legible at 512x512.

Isolates conversion damage from resolution loss: the real arxiv page's body
formulas are unreadable after the fast_vlm stretch-to-512 (eager rail baseline
already collapses), so the formula gate needs a page where the math IS legible
at the rail's input size. Formula = Planck brightness-temperature inversion
(eq. 16 of the WASP-121b page) so the full-mode reference LaTeX is comparable.

    .venv/bin/python docling_work/make_formula_page.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

OUT = "docling_work/formula_page.png"
W = H = 1024
FONT = "/System/Library/Fonts/Helvetica.ttc"

TITLE = "Brightness Temperature"
PARA = ("We convert the fitted emission spectra to brightness "
        "temperature by wavelength, using the inverted Planck law:")
TEX = r"$T_{\mathrm{bright}} = \frac{hc}{k\lambda} \cdot "\
      r"\left[ \ln\left( \frac{2hc^2}{\lambda^5 B_{\lambda}} + 1 \right) \right]^{-1}$"
TAIL = "where B is the thermal emission of the planet."

fig = plt.figure(figsize=(8, 2.2), dpi=160)
fig.text(0.5, 0.5, TEX, ha="center", va="center", fontsize=30)
fig.savefig("docling_work/_formula_frag.png", bbox_inches="tight", pad_inches=0.1)
plt.close(fig)

img = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(img)
f_title = ImageFont.truetype(FONT, 46)
f_body = ImageFont.truetype(FONT, 32)
d.text((80, 90), TITLE, font=f_title, fill="black")
words, lines, cur = PARA.split(), [], ""
for w_ in words:
    if len(cur) + len(w_) + 1 > 55:
        lines.append(cur); cur = w_
    else:
        cur = (cur + " " + w_).strip()
lines.append(cur)
y = 210
for ln in lines:
    d.text((80, y), ln, font=f_body, fill="black"); y += 48

frag = Image.open("docling_work/_formula_frag.png").convert("RGB")
fw = 760
fh = int(frag.height * fw / frag.width)
frag = frag.resize((fw, fh), Image.LANCZOS)
img.paste(frag, ((W - fw) // 2, y + 80))
d.text((80, y + 80 + fh + 90), TAIL, font=f_body, fill="black")
img.save(OUT)
print("wrote", OUT, img.size, "formula frag", (fw, fh))
