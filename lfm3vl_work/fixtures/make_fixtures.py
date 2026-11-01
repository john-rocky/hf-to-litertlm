#!/usr/bin/env python3
"""LFM3-VL-2B fixtures: 12 pictures and 23 English questions about them, reproducible from scripts.

Images (none in git; `fetch.sh` rebuilds img/ and checks every hash against fixtures.json):
  img/orig/<file>     the four NASA originals (public domain, pinned URLs + sha256)
  img/src/<image>.png the SOURCE image of each row: synthetic = make_synth.py output as drawn; NASA =
                      ImageOps.exif_transpose(original).convert("RGB") resized to longest side 1024 with
                      LANCZOS (the originals are 3000-6662 px = 4.6k-9k image tokens at the processor's
                      native smart_resize; 1024 px stands for a phone upload and is the size the native-resolution arm reads)
  img/g<G>/<image>.png ImageOps.exif_transpose(Image.open(src)).convert("RGB").resize((G, G), BICUBIC),
                      G = 512 and 768 (the bundle's fixed grid; the runtime's own resize is then the identity)

  python -I make_fixtures.py write   -> img/src, img/g512, img/g768 from img/orig + make_synth.py, fixtures.json
  python -I make_fixtures.py check   -> recompute every hash, exit 1 on any difference

Row fields: id, image, kind, question, expect (short answer for the scored rows, None otherwise),
public_ok (true = the picture may be redistributed: a public-domain photo or a drawing made here), plus per image: source (kind, url / seed, licence), src size, sha256 of every PNG file and
of the decoded RGB bytes (pixel_sha256).
"""
import hashlib
import json
import os
import sys

import PIL
from PIL import Image, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "img")
GS = (512, 768)
SRC_LONG = 1024

sys.path.insert(0, HERE)
import make_synth  # noqa: E402

NASA = {
    # image id: (file, nasa_id, title, url, sha256 of the original)
    "nasa_jungle": ("nasa_S67-34267.jpg", "S67-34267", "ASTRONAUT GROUP - TRAINING - JUNGLE - PANAMA (1967)",
                    "https://images-assets.nasa.gov/image/S67-34267/S67-34267~orig.jpg",
                    "7e748be3b43b91a25652329f4026b1c19c96ae60decc2ea99cd71a538754db21"),
    "nasa_guppy": ("nasa_KSC-20170424-PH_SWW01_0007.jpg", "KSC-20170424-PH_SWW01_0007",
                   "Guppy Arrival for Orion EM-1 Crew Module Structural Test Article (2017)",
                   "https://images-assets.nasa.gov/image/KSC-20170424-PH_SWW01_0007/KSC-20170424-PH_SWW01_0007~orig.JPG",
                   "497419455401d41fd0042b2af1d07944d4013a8362d549f6787f4cc6e66110a2"),
    "nasa_truck": ("nasa_KSC-08pd3809.jpg", "KSC-08pd3809", "Crane segments arrive by truck at Launch Pad 39B (2008)",
                   "https://images-assets.nasa.gov/image/KSC-08pd3809/KSC-08pd3809~orig.jpg",
                   "4b187db43aaea1720286e4fc35989e020a170510d3d11fa3d21dfefb07fabfd4"),
    "nasa_radar": ("nasa_PIA12503.jpg", "PIA12503", "NASA Airborne Radar Captures Its First Images of Post-Quake Haiti (2010)",
                   "https://images-assets.nasa.gov/image/PIA12503/PIA12503~orig.jpg",
                   "7b3a301daab2254f311913d173e10f5ff2c6a10a589f8d36918f8f3ecc6cb008"),
}

# (row id, image id, kind, question, expect). expect = None: not scored (descriptions and boxes,
# the output form is recorded).
ROWS = [
    ("syn_shapes_count", "syn_shapes", "count", "How many blue circles are in this image? Answer with a number.", "5"),
    ("syn_shapes_color", "syn_shapes", "color", "What color are the squares? Answer with one word.", "red"),
    ("syn_shapes_loc", "syn_shapes", "position", "Locate the red squares in this image and give their bounding boxes.", None),
    ("syn_exit_text", "syn_exit", "text", "What text is written on the sign? Answer with the text only.", "exit 12"),
    ("syn_exit_color", "syn_exit", "color", "What color is the sign? Answer with one word.", "green"),
    ("syn_gate_num", "syn_gate", "text", "What is the gate number? Answer with the number only.", "47"),
    ("syn_bar_max", "syn_bar", "chart", "Which day has the tallest bar in this chart? Answer with the day label only.", "wed"),
    ("syn_bar_count", "syn_bar", "chart", "How many bars are in this chart? Answer with a number.", "4"),
    ("syn_table_bread", "syn_table", "table", "In this table, what is the price of Bread? Answer with the number only.", "5"),
    ("syn_lr_side", "syn_lr", "side", "Is the red circle on the left side or the right side of the image? Answer with one word.", "left"),
    ("syn_lr_loc", "syn_lr", "position", "Locate the blue square in this image and give its bounding box.", None),
    ("syn_tb_side", "syn_tb", "side", "Is the green triangle at the top or at the bottom of the image? Answer with one word.", "top"),
    ("syn_big_color", "syn_big", "color", "What color is the circle? Answer with one word.", "purple"),
    ("nasa_jungle_count", "nasa_jungle", "count", "How many people are in this image? Answer with a number.", "3"),
    ("nasa_jungle_color", "nasa_jungle", "color", "What color are the people's clothes? Answer with one word.", "orange"),
    ("nasa_jungle_desc", "nasa_jungle", "describe", "Describe this image in one sentence.", None),
    ("nasa_guppy_yes", "nasa_guppy", "yesno", "Is there an airplane in this image? Answer yes or no.", "yes"),
    ("nasa_guppy_sky", "nasa_guppy", "word", "Is the sky clear or cloudy? Answer with one word.", "cloudy"),
    ("nasa_truck_sign", "nasa_truck", "text", "What words are written on the yellow sign? Answer with the words only.", "oversize load"),
    ("nasa_truck_color", "nasa_truck", "color", "What color is the truck? Answer with one word.", "red"),
    ("nasa_truck_loc", "nasa_truck", "position", "Locate the yellow sign in this image and give its bounding box.", None),
    ("nasa_radar_color", "nasa_radar", "color", "What is the main color of the land in this image? Answer with one word.", "green"),
    ("nasa_radar_desc", "nasa_radar", "describe", "Describe this image in one sentence.", None),
]


def sha256_file(p):
  h = hashlib.sha256()
  with open(p, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 22), b""):
      h.update(chunk)
  return h.hexdigest()


def pixel_sha(p):
  im = Image.open(p).convert("RGB")
  return hashlib.sha256(im.tobytes()).hexdigest()


def g_image(src_path, g):
  return ImageOps.exif_transpose(Image.open(src_path)).convert("RGB").resize((g, g), Image.BICUBIC)


def write():
  for sub in ("src",) + tuple(f"g{g}" for g in GS):
    os.makedirs(os.path.join(IMG, sub), exist_ok=True)
  synth_dir = os.path.join(IMG, "synth")
  os.makedirs(synth_dir, exist_ok=True)
  gt = {}
  for name, fn in make_synth.DRAW.items():
    im, truth = fn()
    im.save(os.path.join(IMG, "src", f"{name}.png"), optimize=False)
    gt[name] = truth
  for iid, (fname, *_rest, sha) in NASA.items():
    op = os.path.join(IMG, "orig", fname)
    got = sha256_file(op)
    assert got == sha, (fname, got)
    im = ImageOps.exif_transpose(Image.open(op)).convert("RGB")
    w, h = im.size
    s = SRC_LONG / max(w, h)
    im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    im.save(os.path.join(IMG, "src", f"{iid}.png"), optimize=False)
  images = {}
  for iid in list(make_synth.DRAW) + list(NASA):
    sp = os.path.join(IMG, "src", f"{iid}.png")
    rec = {"src": f"img/src/{iid}.png", "src_size_wh": list(Image.open(sp).size),
           "src_sha256": sha256_file(sp), "src_pixel_sha256": pixel_sha(sp)}
    for g in GS:
      gp = os.path.join(IMG, f"g{g}", f"{iid}.png")
      g_image(sp, g).save(gp, optimize=False)
      rec[f"g{g}"] = f"img/g{g}/{iid}.png"
      rec[f"g{g}_sha256"] = sha256_file(gp)
      rec[f"g{g}_pixel_sha256"] = pixel_sha(gp)
    if iid in NASA:
      fname, nid, title, url, sha = NASA[iid]
      rec["source"] = {"kind": "nasa", "nasa_id": nid, "title": title, "url": url, "orig_file": f"img/orig/{fname}",
                       "orig_sha256": sha, "licence": "NASA / public domain", "credit": "NASA",
                       "src_rule": f"exif_transpose -> RGB -> longest side {SRC_LONG} LANCZOS"}
    else:
      rec["source"] = {"kind": "synthetic", "script": "make_synth.py", "fn": iid, "truth": gt[iid],
                       "licence": "own drawing (public_ok)"}
    rec["public_ok"] = True
    images[iid] = rec
  rows = [{"id": rid, "image": iid, "kind": kind, "question": q, "expect": exp, "public_ok": True}
          for rid, iid, kind, q, exp in ROWS]
  out = {"format": "lfm3vl-fixtures-1", "pillow": PIL.__version__,
         "g_rule": "ImageOps.exif_transpose(Image.open(src)).convert('RGB').resize((G, G), Image.BICUBIC)",
         "grids": list(GS), "images": images, "rows": rows}
  with open(os.path.join(HERE, "fixtures.json"), "w") as f:
    json.dump(out, f, indent=1)
    f.write("\n")
  n_exp = sum(1 for r in rows if r["expect"] is not None)
  print(f"FIXTURES_WRITTEN rows={len(rows)} scored={n_exp} images={len(images)} pillow={PIL.__version__}")
  return 0


def check():
  fx = json.load(open(os.path.join(HERE, "fixtures.json")))
  bad = []
  for iid, rec in fx["images"].items():
    for key in ("src",) + tuple(f"g{g}" for g in fx["grids"]):
      p = os.path.join(HERE, rec[key])
      if not os.path.exists(p):
        bad.append((iid, key, "missing"))
        continue
      if sha256_file(p) != rec[f"{key}_sha256"]:
        bad.append((iid, key, "sha256"))
      if pixel_sha(p) != rec[f"{key}_pixel_sha256"]:
        bad.append((iid, key, "pixel_sha256"))
  print("FIXTURES_CHECK", "OK" if not bad else f"BAD {bad}", f"pillow={PIL.__version__} (written with {fx['pillow']})")
  return 0 if not bad else 1


if __name__ == "__main__":
  mode = sys.argv[1] if len(sys.argv) > 1 else "check"
  sys.exit(write() if mode == "write" else check())
