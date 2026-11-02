#!/usr/bin/env python3
"""Build the Xiaomi-OCR-0 test set (set.json) and the fixed-size pictures a fast_vlm bundle is given.

  python minicpmocr_work/make_fixtures.py FIXTURES          # the 11 synthetic pictures with their ground truth
  python xiaomiocr_work/make_set.py SNAPSHOT FIXTURES OUT [--forms p1344x768,p512,s512,...]
      [--extra doc_receipt=ocr_receipt.jpeg --extra doc_form=ocr_suggestion_form.jpg]

Cases (one row per picture x prompt; prompts are the model card's "Task prompts" verbatim):
  - the 11 synthetic pictures of minicpmocr_work/make_fixtures.py, with their ground truth;
  - the source repository's example_pics/ (receipt.jpg twice: the Document prompt and the README's KIE prompt), with
    the author's .md (and the README's KIE JSON) as the reference column;
  - optional (--extra): two photos of the transformers model doc, not redistributed here (dataset
    huggingface/documentation-images, cc-by-nc-sa-4.0; DOC_PHOTOS below gives the URL at the revision used and the
    sha256 each file must have).
Every picture gets EXIF applied first (ImageOps.exif_transpose), then one PNG per form:
  s{G}      stretched to G x G (PIL BICUBIC)
  p{G}      pasted centred on a white square of its long side, then G x G (PIL BICUBIC)
  p{H}x{W}  pasted centred on a white canvas of aspect H:W that just holds it, then W x H (PIL BICUBIC) - the shipped
            bundle's picture is p1344x768 (768 wide, 1344 tall; ocr_xiaomi.py prepares a picture the same way)
max_new_tokens = 2 x the reference's token count (the repository's tokenizer.json, no specials), at least 512, at most
4,096; a case without a reference gets 1,024. Writes OUT/set.json (cases, files relative to OUT with size and sha256,
the grid the processor picks for the original picture, token counts).
"""
import argparse
import hashlib
import json
import os
import re
import shutil

from PIL import Image, ImageOps

DOC = ("Extract all information from the main body of the document image and represent it in markdown format, "
       "ignoring headers and footers. Tables should be expressed in OTSL format, formulas in the document should be "
       "represented using LATEX format, and the parsing should be organized according to the reading order.")
TEXT = "Extract the text in the image."
TABLE = "Parse the table in the image into OTSL."
FORMULA = "Identify the formula in the image and represent it using LATEX format."
KIE_HEAD = "Extract key information in the image"

# synthetic picture -> (class, prompt); the classes pick score_xocr.py's line
SYNTH = {
    "line_en": ("text", TEXT), "title_en": ("text", TEXT), "para_en": ("text", TEXT),
    "para_en_small": ("text", TEXT), "para_en_photo": ("photo", TEXT), "para_zh": ("text", TEXT),
    "table": ("table", TABLE), "formula_1": ("formula", FORMULA), "formula_2": ("formula", FORMULA),
    "receipt": ("kie", None), "page_en": ("page", DOC),
}
REPO = [  # name, file in example_pics/, class, prompt, the author's reference file
    ("receipt_jpg", "receipt.jpg", "photo", DOC, "receipt.md"),
    ("receipt_jpg_kie", "receipt.jpg", "kie", None, None),
    ("dense_equations", "dense-equations.jpg", "page", DOC, "dense-equations.md"),
    ("handwrite_formula", "handwrite-formula.jpg", "page", DOC, "handwrite-formula.md"),
    ("qianziwen", "qianziwen.jpeg", "photo", DOC, "qianziwen.md"),
]
DOC_PHOTOS = {  # name -> (class, prompt, URL, sha256); huggingface/documentation-images at 8d04c5b3, cc-by-nc-sa-4.0
    "doc_receipt": ("photo", TEXT, "https://huggingface.co/datasets/huggingface/documentation-images/resolve/"
                    "8d04c5b37984c76e594293aab2d610490dae8ec4/transformers/model_doc/ocr_receipt.jpeg",
                    "18263c474c033ea1c15532c8060d0fc27ebd670947b50a726e9de151e6dbc077"),
    "doc_form": ("page", DOC, "https://huggingface.co/datasets/huggingface/documentation-images/resolve/"
                 "8d04c5b37984c76e594293aab2d610490dae8ec4/transformers/model_doc/ocr_suggestion_form.jpg",
                 "6aa381f179baf840dfa5cf56d1cee2addb58ee370cb8d991dfa97a22b89c8758"),
}


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def kie_prompt(keys):
    body = "{\n" + ",\n".join(f'    "{k}": ""' for k in keys) + "\n}"
    return (f"{KIE_HEAD}\n\nPlease output the key information in JSON format according to the following schema:\n"
            f"{body}")


def readme_kie(readme):
    """The README's KIE example: the ```text prompt block and the ```json answer after '### Key information extraction'."""
    sec = readme.split("### Key information extraction", 1)[1]
    prompt = re.search(r"```text\n(.*?)\n```", sec, flags=re.S).group(1)
    answer = json.loads(re.search(r"```json\n(.*?)\n```", sec, flags=re.S).group(1))
    return prompt, answer


def pad_to(img, h, w):
    """Centre img on a white canvas of aspect h:w that just holds it, then resize to (w, h)."""
    if img.height * w > img.width * h:              # taller than the canvas aspect: full height
        ch, cw = img.height, round(img.height * w / h)
    else:
        cw, ch = img.width, round(img.width * h / w)
    canvas = Image.new("RGB", (cw, ch), "white")
    canvas.paste(img, ((cw - img.width) // 2, (ch - img.height) // 2))
    return canvas.resize((w, h), Image.BICUBIC)


def form_picture(img, form):
    body = form[1:]
    if form[0] == "s":
        g = int(body)
        return img.resize((g, g), Image.BICUBIC)
    if form[0] == "p" and "x" in body:
        h, w = (int(x) for x in body.split("x"))
        return pad_to(img, h, w)
    if form[0] == "p":
        g = int(body)
        side = max(img.size)
        sq = Image.new("RGB", (side, side), "white")
        sq.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
        return sq.resize((g, g), Image.BICUBIC)
    raise ValueError(form)


def variants(src_img, out, stem, forms):
    files = {}
    img = ImageOps.exif_transpose(src_img).convert("RGB")
    os.makedirs(os.path.join(out, "forms"), exist_ok=True)
    for form in forms:
        path = os.path.join(out, "forms", f"{stem}_{form}.png")
        im = form_picture(img, form)
        im.save(path, format="PNG")
        files[form] = {"file": os.path.relpath(path, out), "size": list(im.size), "sha256": sha(path)}
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("snapshot", help="a local copy of xiaomi-research/Xiaomi-OCR-0 (README.md, example_pics/, tokenizer)")
    ap.add_argument("fixtures", help="the folder minicpmocr_work/make_fixtures.py wrote (fixtures.json + PNGs)")
    ap.add_argument("out")
    ap.add_argument("--forms", default="p1344x768")
    ap.add_argument("--extra", action="append", default=[], help="NAME=PATH for a picture of DOC_PHOTOS")
    a = ap.parse_args()
    forms = [f for f in a.forms.split(",") if f]
    os.makedirs(a.out, exist_ok=True)
    from tokenizers import Tokenizer
    from transformers import AutoProcessor
    tok = Tokenizer.from_file(os.path.join(a.snapshot, "tokenizer.json"))
    ip = AutoProcessor.from_pretrained(a.snapshot).image_processor
    ntok = lambda s: len(tok.encode(s, add_special_tokens=False).ids)  # noqa: E731

    def grid(path):
        im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        g = ip(images=[im], return_tensors="pt")["image_grid_thw"][0].tolist()
        return {"grid_thw": g, "soft_tokens": g[1] * g[2] // 4, "resized_hw": [g[1] * 16, g[2] * 16]}

    def cap(n):
        return 1024 if n is None else max(512, min(4096, 2 * n))

    def copy_in(src, sub):
        dst = os.path.join(a.out, sub, os.path.basename(src))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.exists(dst):
            shutil.copyfile(src, dst)
        assert sha(dst) == sha(src), dst
        return dst

    cases = []
    fx = json.load(open(os.path.join(a.fixtures, "fixtures.json")))
    for it in fx["items"]:
        cls, prompt = SYNTH[it["name"]]
        gt = it["gt"]
        if cls == "kie":
            prompt = kie_prompt(list(gt))
            ref_text = json.dumps(gt, ensure_ascii=False, indent=2)
        elif cls == "table":
            ref_text = "".join("".join(f"<fcel>{c}" for c in row) + "<nl>" for row in gt)
        else:
            ref_text = gt
        src = os.path.join(a.fixtures, it["files"]["native"]["file"])
        assert sha(src) == it["files"]["native"]["sha256"], src
        nat = copy_in(src, "fixtures")
        files = {"native": {"file": os.path.relpath(nat, a.out), "size": it["files"]["native"]["size"],
                            "sha256": sha(nat)}}
        files.update(variants(Image.open(nat), a.out, it["name"], forms))
        n = ntok(ref_text)
        cases.append({"name": it["name"], "source": "synthetic (make_fixtures.py)", "class": cls, "prompt": prompt,
                      "gt": gt, "ref": None, "ref_kind": "truth", "ref_tokens": n, "max_new_tokens": cap(n),
                      "native": grid(nat), "files": files})

    pics = os.path.join(a.snapshot, "example_pics")
    readme = open(os.path.join(a.snapshot, "README.md"), encoding="utf-8").read()
    kie_p, kie_a = readme_kie(readme)
    for name, f, cls, prompt, ref in REPO:
        nat = copy_in(os.path.join(pics, f), "example_pics")
        if cls == "kie":
            prompt, gt, ref_text, ref_kind = kie_p, kie_a, json.dumps(kie_a, ensure_ascii=False, indent=2), "author_readme_json"
        else:
            gt, ref_kind = None, "author_md"
            ref_text = open(os.path.join(pics, ref), encoding="utf-8").read()
        n = ntok(ref_text)
        files = {"native": {"file": os.path.relpath(nat, a.out), "size": list(Image.open(nat).size), "sha256": sha(nat)}}
        files.update(variants(Image.open(nat), a.out, name, forms))
        cases.append({"name": name, "source": f"repo example_pics/{f}", "class": cls, "prompt": prompt, "gt": gt,
                      "ref": ref_text, "ref_file": ref and f"example_pics/{ref}", "ref_kind": ref_kind,
                      "ref_tokens": n, "max_new_tokens": cap(n), "native": grid(nat), "files": files})

    for spec in a.extra:
        name, path = spec.split("=", 1)
        cls, prompt, url, want = DOC_PHOTOS[name]
        assert sha(path) == want, f"{path}: not the file of {url}"
        nat = copy_in(path, "extra")
        files = {"native": {"file": os.path.relpath(nat, a.out), "size": list(Image.open(nat).size), "sha256": sha(nat)}}
        files.update(variants(Image.open(nat), a.out, name, forms))
        cases.append({"name": name, "source": url, "license": "cc-by-nc-sa-4.0 (dataset card); not redistributed",
                      "class": cls, "prompt": prompt, "gt": None, "ref": None, "ref_kind": "none", "ref_tokens": None,
                      "max_new_tokens": cap(None), "native": grid(nat), "files": files})

    meta = {"generator": "xiaomiocr_work/make_set.py", "forms": forms, "resample": "PIL BICUBIC",
            "exif": "ImageOps.exif_transpose before every form", "cases": cases}
    with open(os.path.join(a.out, "set.json"), "w") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    for c in cases:
        print(f"{c['name']:18s} {c['class']:7s} native {c['files']['native']['size']} -> grid {c['native']['grid_thw']} "
              f"soft {c['native']['soft_tokens']:5d} | ref {c['ref_kind']:18s} {c['ref_tokens']} tok -> max_new {c['max_new_tokens']}")


if __name__ == "__main__":
    main()
