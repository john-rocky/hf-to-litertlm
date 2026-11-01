#!/usr/bin/env python3
"""Truth regions as masks, built from the polygons of a scene file.

A mask is 1000 x 1000 pixels over 0-1000 of the picture (pixel (i, j) = the unit square x in [i, i + 1),
y in [j, j + 1)); a point (x, y) is right when the pixel (min(999, int(x)), min(999, int(y))) is set. Because the masks
are in picture fractions, the same mask scores a point read on the source picture and on its 512 x 512 stretch.

  placement request: zone AND (its surface shrunk by MARGIN) AND NOT (every other object of the photo grown by MARGIN)
                     (zone = the item's zone polygon, or the surface itself for a surface request; "other object" =
                     every OBJ entry of the photo except the one named like the item's surface)
  reference object:  the object's outline (or the surface polygon for "surf:<name>") minus the listed objects

Shrinking and growing are Pillow's MinFilter / MaxFilter with a (2 MARGIN + 1)-pixel square.

  python tools/demo_scene_check/regions.py [--scene tools/demo_scene_check/scene_where_to_put.py]   # area per request
"""
import argparse
import importlib.util
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SCENE = os.path.join(HERE, "scene_where_to_put.py")
N = 1000
MARGIN = 4


def load_scene(path=None):
  path = os.path.abspath(path or DEFAULT_SCENE)
  spec = importlib.util.spec_from_file_location("demo_scene_" + os.path.splitext(os.path.basename(path))[0], path)
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  mod.__path_on_disk__ = path
  return mod


def poly_img(polys):
  im = Image.new("L", (N, N), 0)
  d = ImageDraw.Draw(im)
  for p in polys:
    d.polygon([(float(x), float(y)) for x, y in p], fill=255)
  return im


def grow(im, r=MARGIN):
  return im.filter(ImageFilter.MaxFilter(2 * r + 1))


def shrink(im, r=MARGIN):
  return im.filter(ImageFilter.MinFilter(2 * r + 1))


def as_bool(im):
  return np.array(im) > 127


def item_mask(scene, it):
  ph = it["photo"]
  surf = as_bool(shrink(poly_img([scene.SURF[ph][it["surface"]]])))
  others = [p for k, o in scene.OBJ[ph].items() if k != it["surface"] for p in o["polys"]]
  obst = as_bool(grow(poly_img(others))) if others else np.zeros((N, N), bool)
  zone = as_bool(poly_img([it["zone"]])) if it.get("zone") else np.ones((N, N), bool)
  return zone & surf & ~obst


def object_mask(scene, photo, key, minus):
  if key.startswith("surf:"):
    m = as_bool(poly_img([scene.SURF[photo][key[5:]]]))
  else:
    m = as_bool(poly_img(scene.OBJ[photo][key]["polys"]))
  if minus:
    m &= ~as_bool(poly_img([p for k in minus for p in scene.OBJ[photo][k]["polys"]]))
  return m


def object_id(photo, key):
  return f"{photo}_obj_" + key.replace("surf:", "").replace(" ", "-")


def save(mask, path):
  os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
  Image.fromarray(mask.astype(np.uint8) * 255).convert("1").save(path)


def load(path):
  return np.array(Image.open(path).convert("L")) > 127


def hit(mask, pt):
  x, y = pt
  return bool(mask[min(N - 1, max(0, int(y))), min(N - 1, max(0, int(x)))])


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--scene", default=DEFAULT_SCENE)
  a = ap.parse_args()
  scene = load_scene(a.scene)
  for it in scene.ITEMS:
    m = item_mask(scene, it)
    print(f"{it['id']:32s} {it['rel_kind']:8s} area {m.mean() * 100:6.2f} %")
  return 0


if __name__ == "__main__":
  sys.exit(main())
