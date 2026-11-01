#!/usr/bin/env python3
"""Build a scene's fixtures: the photos (downloaded from Wikimedia Commons, sha1 checked), the source picture and its
512 x 512 stretch, the truth mask of every request and reference object, overlays to look at before any model runs,
and fixtures.json with every hash.

  python tools/demo_scene_check/fixtures.py write --out out/demo_scene_check/where_to_put [--scene <scene.py>]
  python tools/demo_scene_check/fixtures.py check --out out/demo_scene_check/where_to_put   # exit 1 on any difference

Needs numpy and Pillow. Paths in fixtures.json are relative to --out.

src  = ImageOps.exif_transpose(original).convert("RGB"), LANCZOS to long side 1024 when larger (a phone-upload size;
       the picture given to a model's own processor).
s512 = ImageOps.exif_transpose(Image.open(src)).convert("RGB").resize((512, 512), Image.BICUBIC), saved as PNG: the
       upright picture stretched to 512 x 512, as an app hands it to a vision encoder with a fixed 512 x 512 grid.
Masks (regions.py) are in 0-1000 of the picture, so the stretch keeps them. Pictures to look at before any model run:
  gt/<photo>.png      placement regions on the source picture
  gt512/<photo>.png   placement regions on the 512 x 512 picture (what the model sees)
  obj/<photo>.png     reference-object regions on the source picture
  geom/<photo>.png    the polygons the masks come from (surfaces green, objects magenta, zones in the region colour)
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
import urllib.error
import urllib.request

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name):
  spec = importlib.util.spec_from_file_location("demo_scene_" + name, os.path.join(HERE, name + ".py"))
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


REG = _load("regions")
UA = {"User-Agent": "hf-to-litertlm-demo-scene-check/1.0 (https://github.com/john-rocky/hf-to-litertlm; "
                    "one request every 2 s)"}
SRC_LONG = 1024
COLS = [(230, 40, 40), (40, 120, 230), (30, 170, 60), (200, 120, 0), (150, 60, 200), (0, 160, 170)]


def sha256(p):
  return hashlib.sha256(open(p, "rb").read()).hexdigest()


def sha1(p):
  return hashlib.sha1(open(p, "rb").read()).hexdigest()


def fetch(url, dst, want_sha1):
  if os.path.exists(dst) and sha1(dst) == want_sha1:
    return
  for _ in range(8):
    try:
      data = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=180).read()
      open(dst, "wb").write(data)
      break
    except urllib.error.HTTPError as e:
      if e.code == 429:
        wait = int(e.headers.get("Retry-After") or 60)
        print(f"429 on {url}, waiting {wait} s", flush=True)
        time.sleep(wait)
        continue
      raise
  time.sleep(2.0)
  got = sha1(dst)
  if got != want_sha1:
    raise SystemExit(f"sha1 mismatch for {url}: {got} != {want_sha1}")


def font(size):
  for p in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
    try:
      return ImageFont.truetype(p, size)
    except OSError:
      pass
  return ImageFont.load_default(size=size)


def overlay(src_img, dst, regions, label_size):
  """regions = [(label, bool mask 1000 x 1000)]: a translucent fill, a 1-px edge and the label at the centroid."""
  im = src_img.convert("RGBA")
  W, H = im.size
  over = Image.new("RGBA", (W, H), (0, 0, 0, 0))
  marks = []
  for i, (name, mask) in enumerate(regions):
    col = COLS[i % len(COLS)]
    m = Image.fromarray(mask.astype(np.uint8) * 255).resize((W, H), Image.NEAREST)
    over.paste(Image.new("RGBA", (W, H), col + (95,)), (0, 0), m)
    edge = ImageChops.difference(m, m.filter(ImageFilter.MinFilter(3)))
    over.paste(Image.new("RGBA", (W, H), col + (255,)), (0, 0), edge)
    ys, xs = np.nonzero(mask)
    marks.append((name, col, xs.mean() / 1000 * W, ys.mean() / 1000 * H))
  out = Image.alpha_composite(im, over)
  d = ImageDraw.Draw(out)
  f = font(label_size)
  for name, col, cx, cy in marks:
    d.text((cx, cy), name, font=f, fill=col + (255,), stroke_width=3, stroke_fill=(255, 255, 255, 255), anchor="mm")
  out.convert("RGB").save(dst)


def geom(scene, src_img, dst, pid, items):
  im = src_img.convert("RGB")
  W, H = im.size
  d = ImageDraw.Draw(im, "RGBA")

  def P(poly):
    return [(x / 1000 * W, y / 1000 * H) for x, y in poly]
  for poly in scene.SURF[pid].values():
    d.line(P(poly + [poly[0]]), fill=(0, 200, 60, 255), width=3)
  for o in scene.OBJ[pid].values():
    for poly in o["polys"]:
      d.line(P(poly + [poly[0]]), fill=(255, 0, 255, 255), width=2)
    for poly in o.get("base", []):
      d.line(P(poly + [poly[0]]), fill=(255, 0, 255, 160), width=1)
  for i, it in enumerate(items):
    if it.get("zone"):
      col = COLS[i % len(COLS)]
      d.line(P(it["zone"] + [it["zone"][0]]), fill=col + (255,), width=2)
  im.save(dst)


def build(scene, out, write):
  rel = lambda p: os.path.relpath(p, out)  # noqa: E731
  for sub in ("orig", "src", "s512", "gt", "gt512", "obj", "geom", "mask", "omask"):
    os.makedirs(os.path.join(out, sub), exist_ok=True)
  photos = {}
  for pid, m in scene.PHOTOS.items():
    orig = os.path.join(out, "orig", pid + os.path.splitext(m["url"])[1].lower())
    if write:
      fetch(m["url"], orig, m["sha1"])
    src = os.path.join(out, "src", pid + ".png")
    dst = os.path.join(out, "s512", pid + ".png")
    if write:
      im = ImageOps.exif_transpose(Image.open(orig)).convert("RGB")
      if max(im.size) > SRC_LONG:
        s = SRC_LONG / max(im.size)
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
      im.save(src)
      ImageOps.exif_transpose(Image.open(src)).convert("RGB").resize((512, 512), Image.BICUBIC).save(dst)
    sw, sh = Image.open(src).size
    photos[pid] = {"title": m["title"], "page": m["page"], "url": m["url"], "author": m["author"],
                   "license": m["license"], "orig": rel(orig), "orig_sha1": m["sha1"], "orig_sha256": sha256(orig),
                   "orig_wh": m["orig_wh"], "src": rel(src), "src_sha256": sha256(src), "src_wh": [sw, sh],
                   "orientation": "portrait" if sh > sw else "landscape", "s512": rel(dst), "s512_sha256": sha256(dst),
                   "scene": m["scene"]}
  items, objects = [], []
  for it in scene.ITEMS:
    mask = REG.item_mask(scene, it)
    if not mask.any():
      raise SystemExit(f"empty truth region: {it['id']}")
    p = os.path.join(out, "mask", it["id"] + ".png")
    if write:
      REG.save(mask, p)
    items.append(dict(it, truth={"mask": rel(p), "mask_sha256": sha256(p), "area_pct": round(mask.mean() * 100, 2)}))
  for ph, key, name, minus in scene.OBJECTS:
    oid = REG.object_id(ph, key)
    mask = REG.object_mask(scene, ph, key, minus)
    p = os.path.join(out, "omask", oid + ".png")
    if write:
      REG.save(mask, p)
    objects.append({"id": oid, "photo": ph, "name": name, "key": key, "minus": minus,
                    "truth": {"mask": rel(p), "mask_sha256": sha256(p), "area_pct": round(mask.mean() * 100, 2)}})
  for pid in photos:
    src_img = Image.open(os.path.join(out, photos[pid]["src"]))
    s512_img = Image.open(os.path.join(out, photos[pid]["s512"]))
    its = [it for it in items if it["photo"] == pid]
    regs = [(it["id"].split("_", 1)[1], REG.load(os.path.join(out, it["truth"]["mask"]))) for it in its]
    oregs = [(ob["id"].split("_obj_", 1)[1], REG.load(os.path.join(out, ob["truth"]["mask"])))
             for ob in objects if ob["photo"] == pid]
    for key, img, sub, rr, ls in (("gt_png", src_img, "gt", regs, 22), ("gt512_png", s512_img, "gt512", regs, 13),
                                  ("obj_png", src_img, "obj", oregs, 18)):
      dst = os.path.join(out, sub, pid + ".png")
      if write:
        overlay(img, dst, rr, ls)
      photos[pid][key] = rel(dst)
    dst = os.path.join(out, "geom", pid + ".png")
    if write:
      geom(scene, src_img, dst, pid, its)
    photos[pid]["geom_png"] = rel(dst)
  counts = {"photos": len(photos), "items": len(items), "objects": len(objects),
            "rel_kind": {k: sum(1 for i in items if i["rel_kind"] == k) for k in ("lr", "between", "fb", "surface")},
            "rule_solvable_yes": sum(1 for i in items if i["rule_solvable"]),
            "orientation": {k: sum(1 for p in photos.values() if p["orientation"] == k) for k in ("landscape", "portrait")}}
  scripts = {os.path.basename(p): sha256(p) for p in (scene.__path_on_disk__, os.path.join(HERE, "regions.py"),
                                                     os.path.abspath(__file__))}
  return {"scene": getattr(scene, "NAME", os.path.basename(scene.__path_on_disk__)), "rules": __doc__,
          "scene_file": os.path.abspath(scene.__path_on_disk__), "photos": photos, "items": items, "objects": objects,
          "counts": counts, "scripts": scripts}


def check(out, scene_path):
  old = json.load(open(os.path.join(out, "fixtures.json")))
  scene = REG.load_scene(scene_path or old.get("scene_file"))
  bad = 0
  for pid, m in old["photos"].items():
    for key in ("orig", "src", "s512"):
      if sha256(os.path.join(out, m[key])) != m[key + "_sha256"]:
        print("DIFF", pid, key)
        bad += 1
    if sha1(os.path.join(out, m["orig"])) != m["orig_sha1"]:
      print("DIFF sha1", pid)
      bad += 1
  now = {it["id"]: it for it in scene.ITEMS}
  for it in old["items"]:
    p = os.path.join(out, it["truth"]["mask"])
    if sha256(p) != it["truth"]["mask_sha256"]:
      print("DIFF mask file", it["id"])
      bad += 1
    if not (REG.load(p) == REG.item_mask(scene, now[it["id"]])).all():
      print("DIFF mask vs scene file", it["id"])
      bad += 1
  for ob in old["objects"]:
    if sha256(os.path.join(out, ob["truth"]["mask"])) != ob["truth"]["mask_sha256"]:
      print("DIFF omask file", ob["id"])
      bad += 1
  print("check:", "OK" if not bad else f"{bad} differences", old["counts"])
  return 1 if bad else 0


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("mode", choices=("write", "check"))
  ap.add_argument("--out", required=True, help="the fixtures directory (fixtures.json and the pictures go here)")
  ap.add_argument("--scene", default=None, help="scene file (default: scene_where_to_put.py; check: the one recorded)")
  a = ap.parse_args()
  out = os.path.abspath(a.out)
  if a.mode == "write":
    scene = REG.load_scene(a.scene)
    fx = build(scene, out, write=True)
    json.dump(fx, open(os.path.join(out, "fixtures.json"), "w"), indent=1, ensure_ascii=False)
    print("wrote", os.path.join(out, "fixtures.json"), fx["counts"])
    return 0
  return check(out, a.scene)


if __name__ == "__main__":
  sys.exit(main())
