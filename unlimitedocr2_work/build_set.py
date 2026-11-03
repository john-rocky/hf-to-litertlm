#!/usr/bin/env python3
"""Build the Unlimited-OCR-2 test set (set.json) and the 1,024 x 1,024 pictures of every page.

  python xiaomiocr_work/make_set.py SNAPSHOT FIXTURES XSET --extra ...   # the Xiaomi-OCR-0 set (18 cases)
  python unlimitedocr2_work/build_set.py XSET/set.json MDB_DIR TOKENIZER.json OUT [--fetch-mdb]

Cases (19) and run pictures (18: 17 single pages + one 3-page document):
  - the 18 cases of the Xiaomi-OCR-0 set (xiaomiocr_work/make_set.py: 11 synthetic pictures with their ground truth,
    the 4 example pictures of xiaomi-research/Xiaomi-OCR-0 with receipt.jpg used twice, 2 photos of the transformers
    model doc). Every picture is read by its path in that set and checked against its sha256. With one prompt for
    every case ("Multi page parsing.", the author's default), receipt_jpg and receipt_jpg_kie are one run scored twice
    (as a photo and by its printed fields).
  - multi_mdb3: baidu/MarathonDocBench (Apache-2.0) at revision fa05fccb, document mp3lan_0543 pages 16-18 (prose with
    cross-page continuations). MDB_DIR holds MarathonDocBench.json and mp3lan_0543/page_{16,17,18}.png of that
    revision (--fetch-mdb downloads the missing ones from the Hub and checks their sha256); the per-page truth is the
    dataset's layout_dets text (title + text_block, reading order), header / footer / page number left out.
Pictures of every page (EXIF orientation applied first with ImageOps.exif_transpose; the set lists any tag):
  s1024 = stretched to 1024 x 1024 (PIL BICUBIC) = what LiteRT-LM does to a picture of another size;
  w1024 = ImageOps.pad(img, (1024, 1024), color=white);
  a1024 = ImageOps.pad(img, (1024, 1024), color=(127, 127, 127)) = the author's base-mode pad done by the app (the fill
          is tuple(int(x * 255) for x in mean 0.5) in _uocr_build_inputs); given to the author's processor it is
          padded again to the same size, which PIL returns as a copy = the reference's pixels. ocr_unlimited.py
          prepares a page the same way.
max_new_tokens of the reference run R: 6,000 for every picture (R's own EOS sets its length); the arms' caps are 2 x
R's generated length, at least 512, at most 6,000 (rt_uocr.py). truth_tokens (the repository's tokenizer.json, no
special tokens) is an instrument column.
Writes OUT/set.json (cases; images with every page's original and variant pictures: file relative to OUT, size,
sha256), OUT/pictures/ (the originals, copied) and OUT/fixtures/ (the variants).
"""
import argparse
import hashlib
import json
import os
import shutil
import urllib.request

from PIL import Image, ImageOps

PROMPT = "Multi page parsing."
GRAY = (127, 127, 127)
MDB_REPO = "https://huggingface.co/datasets/baidu/MarathonDocBench/resolve/"
MDB_REV = "fa05fccbe82ee6ec628006a1d58bb153954c3ee3"
MDB_DOC, MDB_PAGES = "mp3lan_0543", (16, 17, 18)
MDB_SHA = {"MarathonDocBench.json": "7db0e659c76661ce87d4e1b3d47e010edbd48adfb446e3e2ee37260c60adf51c",
           f"{MDB_DOC}/page_16.png": "aba7a67b6cf309c5c76453f9607349ae4842a0fa744060e975b69cb5c1402774",
           f"{MDB_DOC}/page_17.png": "6487f563d95db2f59ef0b104f1aab7ca5164bd5ca95e50f05eada549b087946e",
           f"{MDB_DOC}/page_18.png": "f5249286893c7df83a2b60d4959b2cde049834ac12bb610f29d23526d3a91b3d"}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def mdb_url(rel):
    return MDB_REPO + MDB_REV + "/" + (rel if rel.endswith(".json") else "images/" + rel)


def mdb_files(mdb, fetch):
    """Checks (and with fetch, downloads) the four MarathonDocBench files of the pinned revision."""
    for rel, want in MDB_SHA.items():
        p = os.path.join(mdb, rel)
        if not os.path.exists(p):
            if not fetch:
                raise SystemExit(f"missing {p}: download {mdb_url(rel)} (or pass --fetch-mdb)")
            os.makedirs(os.path.dirname(p), exist_ok=True)
            urllib.request.urlretrieve(mdb_url(rel), p + ".part")
            os.replace(p + ".part", p)
        assert sha(p) == want, f"{p}: sha256 is not the one of {mdb_url(rel)}"


def tok_count(tok, s):
    if s is None:
        return None
    if isinstance(s, (list, dict)):
        s = json.dumps(s, ensure_ascii=False)
    return len(tok.encode(s, add_special_tokens=False).ids)


def variants(src, stem, out):
    im = Image.open(src)
    orient = im.getexif().get(0x0112)
    im = ImageOps.exif_transpose(im).convert("RGB")
    res = {}
    v = {"s1024": im.resize((1024, 1024), Image.BICUBIC),
         "w1024": ImageOps.pad(im, (1024, 1024), color=(255, 255, 255)),
         "a1024": ImageOps.pad(im, (1024, 1024), color=GRAY)}
    for k, img in v.items():
        p = os.path.join(out, "fixtures", f"{stem}_{k}.png")
        img.save(p)
        res[k] = {"file": os.path.relpath(p, out), "size": list(img.size), "sha256": sha(p)}
    return res, orient


def copy_in(src, name, out):
    dst = os.path.join(out, "pictures", name + os.path.splitext(src)[1])
    shutil.copyfile(src, dst)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xiaomi_set", help="set.json of xiaomiocr_work/make_set.py")
    ap.add_argument("mdb", help="directory with MarathonDocBench.json and mp3lan_0543/page_{16,17,18}.png")
    ap.add_argument("tokenizer", help="tokenizer.json of baidu/Unlimited-OCR-2")
    ap.add_argument("out")
    ap.add_argument("--fetch-mdb", action="store_true", help="download the missing MarathonDocBench files")
    a = ap.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(a.tokenizer)
    for d in ("pictures", "fixtures"):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)
    mdb_files(a.mdb, a.fetch_mdb)
    xdir = os.path.dirname(os.path.abspath(a.xiaomi_set))
    xs = json.load(open(a.xiaomi_set))
    cases, images = [], {}
    for c in xs["cases"]:
        f = c["files"]["native"]
        src = os.path.join(xdir, f["file"])
        h = sha(src)
        assert h == f["sha256"], (c["name"], h, f["sha256"])
        img_name = "receipt_jpg" if c["name"] == "receipt_jpg_kie" else c["name"]
        truth = c["gt"] if c["gt"] is not None else c.get("ref")
        truth_kind = c["ref_kind"] if c["gt"] is None else "truth"
        n_truth = tok_count(tok, c.get("ref") if c["name"] == "receipt_jpg_kie" else truth)
        if img_name not in images:
            nat = copy_in(src, img_name, a.out)
            vs, orient = variants(nat, img_name, a.out)
            images[img_name] = {"name": img_name, "pages": [{"file": os.path.relpath(nat, a.out), "size": f["size"],
                                                            "sha256": h, "exif_orientation": orient,
                                                            "variants": vs}],
                                "source": c["source"], "license": c.get("license"),
                                "max_new_tokens_R": 6000}
        cases.append({"name": c["name"], "image": img_name, "class": c["class"], "prompt": PROMPT,
                      "truth": truth, "truth_kind": truth_kind, "truth_tokens": n_truth,
                      "xiaomi_prompt": c["prompt"]})
    gt = json.load(open(os.path.join(a.mdb, "MarathonDocBench.json")))
    doc = next(e for e in gt if (e["page_info"].get("images_list") or [""])[0].split("/")[1] == MDB_DOC)
    pages, ptext = [], []
    for p in MDB_PAGES:
        src = os.path.join(a.mdb, MDB_DOC, f"page_{p}.png")
        nat = copy_in(src, f"multi_mdb3_p{p}", a.out)
        vs, orient = variants(nat, f"multi_mdb3_p{p}", a.out)
        w, hgt = Image.open(src).size
        pages.append({"file": os.path.relpath(nat, a.out), "size": [w, hgt], "sha256": sha(src),
                      "exif_orientation": orient, "variants": vs, "url": mdb_url(f"{MDB_DOC}/page_{p}.png")})
        blocks = sorted([b for b in doc["layout_dets"] if b["page_id"] == p and not b.get("ignore")
                         and b["category_type"] in ("title", "text_block")], key=lambda b: b.get("order", 0))
        ptext.append("\n\n".join((b.get("text") or "").strip() for b in blocks))
    n_truth = tok_count(tok, "\n".join(ptext))
    images["multi_mdb3"] = {"name": "multi_mdb3", "pages": pages,
                            "source": f"baidu/MarathonDocBench@{MDB_REV} {MDB_DOC} pages {list(MDB_PAGES)}",
                            "license": "apache-2.0 (dataset card YAML)", "max_new_tokens_R": 6000}
    cases.append({"name": "multi_mdb3", "image": "multi_mdb3", "class": "multi", "prompt": PROMPT,
                  "truth": ptext, "truth_kind": "truth (dataset layout_dets text, per page)",
                  "truth_tokens": n_truth})
    out = {"generator": "unlimitedocr2_work/build_set.py", "prompt": PROMPT, "xiaomi_set_sha256": sha(a.xiaomi_set),
           "mdb_revision": MDB_REV, "mdb_gt_sha256": MDB_SHA["MarathonDocBench.json"],
           "variants": {"s1024": "PIL BICUBIC stretch", "w1024": "ImageOps.pad white", "a1024": "ImageOps.pad (127,127,127)"},
           "exif": "ImageOps.exif_transpose before every variant (the author's infer() does not apply EXIF)",
           "cases": cases, "images": list(images.values())}
    json.dump(out, open(os.path.join(a.out, "set.json"), "w"), ensure_ascii=False, indent=1)
    for im in images.values():
        print(im["name"], len(im["pages"]), [p["size"] for p in im["pages"]], im["max_new_tokens_R"],
              [p["exif_orientation"] for p in im["pages"]])
    print("cases", len(cases), "images", len(images))


if __name__ == "__main__":
    main()
