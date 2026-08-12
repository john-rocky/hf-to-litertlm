#!/usr/bin/env python3
"""Image-understanding gate for LFM2.5-VL .litertlm bundles via `litert-lm run`.

Deterministic synthetic fixtures (generated on first run into fixtures/):
  red.png     solid red                      -> mentions "red"
  hello.png   black "HELLO" on white         -> mentions "hello"
  circle.png  blue circle on yellow          -> mentions "circle"/"round"
  count3.png  three black squares in a row   -> mentions "three"/"3"
  cat_dog.png the word CAT big, DOG small    -> reads "cat" (sanity: layout)

PASS = >=4/5 correct and no degenerate output (repetition collapse / empty).

Usage: gate_image.py <model.litertlm> [backend] [out.json] [--vision-backend cpu|gpu]
"""
import json
import os
import re
import subprocess
import sys

LITERT_LM = os.environ.get("LITERT_LM", "litert-lm")
FIXDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")

MODEL = sys.argv[1]
BACKEND = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else "cpu"
OUT = sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else None
VISION_BACKEND = None
if "--vision-backend" in sys.argv:
    VISION_BACKEND = sys.argv[sys.argv.index("--vision-backend") + 1]


def make_fixtures():
    from PIL import Image, ImageDraw, ImageFont
    os.makedirs(FIXDIR, exist_ok=True)
    try:
        font_big = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 120)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 40)
    except OSError:
        font_big = font_small = None

    p = os.path.join(FIXDIR, "red.png")
    if not os.path.exists(p):
        Image.new("RGB", (512, 512), (220, 20, 20)).save(p)

    p = os.path.join(FIXDIR, "hello.png")
    if not os.path.exists(p):
        img = Image.new("RGB", (512, 512), "white")
        d = ImageDraw.Draw(img)
        d.text((60, 190), "HELLO", fill="black", font=font_big)
        img.save(p)

    p = os.path.join(FIXDIR, "circle.png")
    if not os.path.exists(p):
        img = Image.new("RGB", (512, 512), (250, 220, 40))
        d = ImageDraw.Draw(img)
        d.ellipse((128, 128, 384, 384), fill=(30, 60, 220))
        img.save(p)

    p = os.path.join(FIXDIR, "count3.png")
    if not os.path.exists(p):
        img = Image.new("RGB", (512, 512), "white")
        d = ImageDraw.Draw(img)
        for i in range(3):
            x = 60 + i * 140
            d.rectangle((x, 200, x + 100, 300), fill="black")
        img.save(p)

    p = os.path.join(FIXDIR, "cat_dog.png")
    if not os.path.exists(p):
        img = Image.new("RGB", (512, 512), "white")
        d = ImageDraw.Draw(img)
        d.text((100, 150), "CAT", fill="black", font=font_big)
        d.text((200, 350), "dog", fill="gray", font=font_small)
        img.save(p)


CASES = [
    ("color=red", "red.png",
     "What is the dominant color of this image? Answer briefly.", r"\bred\b"),
    ("ocr=HELLO", "hello.png",
     "What does the text in this image say? Answer briefly.", r"hello"),
    ("shape=circle", "circle.png",
     "What shape is shown in this image? Answer briefly.", r"circle|round|ellip"),
    ("count=3", "count3.png",
     "How many black squares are in this image? Answer briefly.", r"\bthree\b|\b3\b"),
    ("big-word=CAT", "cat_dog.png",
     "What is the largest word written in this image? Answer briefly.", r"\bcat\b"),
]


def degenerate(text):
    words = re.findall(r"\w+", text.lower())
    if len(words) >= 12:
        from collections import Counter
        top = Counter(words).most_common(1)[0][1]
        if top / len(words) > 0.5:
            return True
    return len(text.strip()) == 0


def main():
    make_fixtures()
    results, correct, degen = [], 0, 0
    for label, img, q, pat in CASES:
        cmd = [LITERT_LM, "run", MODEL, "--prompt", q,
               "--attachment", os.path.join(FIXDIR, img),
               "--backend", BACKEND, "--cache", "no",
               "--temperature", "0", "--seed", "0"]
        if VISION_BACKEND:
            cmd += ["--vision-backend", VISION_BACKEND]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
        text = p.stdout.strip()
        err = p.stderr.strip()[-400:] if p.returncode != 0 else ""
        ok = bool(re.search(pat, text, re.IGNORECASE))
        dg = degenerate(text)
        correct += ok
        degen += dg
        results.append({"label": label, "ok": ok, "degenerate": dg,
                        "rc": p.returncode, "text": text[:300], "err": err})
        print(f"[{'ok' if ok else 'NG'}{'/DEGEN' if dg else ''}] {label}: {text[:100]!r}"
              + (f" RC={p.returncode} {err[:150]}" if p.returncode else ""))

    verdict = "PASS" if correct >= 4 and degen == 0 else "FAIL"
    print(f"correct={correct}/5 degenerate={degen} verdict={verdict}")
    if OUT:
        json.dump({"model": os.path.basename(MODEL), "backend": BACKEND,
                   "vision_backend": VISION_BACKEND, "correct": correct,
                   "degenerate": degen, "verdict": verdict, "results": results},
                  open(OUT, "w"), indent=2)
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
