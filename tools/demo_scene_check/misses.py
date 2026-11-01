#!/usr/bin/env python3
"""Where the wrong placement points landed, by kind (a reading aid; the verdicts stay those of score.py).

  python tools/demo_scene_check/misses.py DIR/score/all.json [--scene <scene.py>] [--labels "<label>,..."] \
      [--json out.json] [--x1000]

For every placement row the point is put in the first class it fits:
  right        inside the truth mask (the verdict of score.py; a mismatch stops the script)
  no point     the reader found no point (or no answer)
  on ref       inside a reference object's outline grown by MARGIN (the object the request names)
  on other     inside another object's outline grown by MARGIN (not a reference, not the item's surface)
  near edge    inside the zone and the surface, but in the MARGIN band the truth mask leaves out (surface edge or an
               object's grown border)
  wrong side   on the item's surface (not shrunk) but outside the zone (free surface on the wrong side / too far)
  off surface  anywhere else (wall, floor, the front face of a piece of furniture, outside the picture)
For a surface request ("on the counter") the reference is the surface itself, so "on ref" does not occur there.
"dist" (per wrong point) = the distance in 0-1000 units from the point to the nearest pixel centre of the truth mask;
"inside" (per right point) = the distance to the nearest pixel centre outside it (how far the point is from the edge).
--x1000: only the labels that answer on a 0-1 scale (most of their points have both coordinates <= 1.0, so their
registered points all land in the top-left corner), every such point read x 1000 (favourable reading (a), not
swapped); counts only, no verdict check.
"""
import argparse
import collections
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name):
  spec = importlib.util.spec_from_file_location("demo_scene_" + name, os.path.join(HERE, name + ".py"))
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


R = _load("regions")
ORDER = ["right", "no point", "on ref", "on other", "near edge", "wrong side", "off surface"]


def at(mask, pt):
  x, y = pt
  if not (0 <= x <= 1000 and 0 <= y <= 1000):
    return False
  return bool(mask[min(999, int(y)), min(999, int(x))])


def dist_to(mask, pt):
  ys, xs = np.nonzero(mask)
  if pt is None or not len(xs):
    return None
  return round(float(np.sqrt(((xs + 0.5 - pt[0]) ** 2 + (ys + 0.5 - pt[1]) ** 2).min())), 1)


def classify(scene, it, pt, truth):
  if pt is None:
    return "no point"
  if at(truth, pt):
    return "right"
  ph = it["photo"]
  objs = scene.OBJ[ph]
  refs = [r["name"] for r in it["refs"] if r["name"] in objs and r["name"] != it["surface"]]
  for name in refs:
    if at(R.as_bool(R.grow(R.poly_img(objs[name]["polys"]))), pt):
      return "on ref"
  for name, o in objs.items():
    if name in refs or name == it["surface"]:
      continue
    if at(R.as_bool(R.grow(R.poly_img(o["polys"]))), pt):
      return "on other"
  surf = R.as_bool(R.poly_img([scene.SURF[ph][it["surface"]]]))
  zone = R.as_bool(R.poly_img([it["zone"]])) if it.get("zone") else np.ones((R.N, R.N), bool)
  if at(surf & zone, pt):
    return "near edge"
  if at(surf, pt):
    return "wrong side"
  return "off surface"


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("score")
  ap.add_argument("--scene", default=None)
  ap.add_argument("--labels", default="")
  ap.add_argument("--json", default="")
  ap.add_argument("--x1000", action="store_true")
  a = ap.parse_args()
  d = json.load(open(a.score))
  scene = R.load_scene(a.scene)
  items = {it["id"]: it for it in scene.ITEMS}
  want = [x.strip() for x in a.labels.split(",") if x.strip()]
  truth = {t: R.item_mask(scene, it) for t, it in items.items()}
  rows = [r for r in d["rows"] if r["q_type"] == "place" and (not want or r["label"] in want)]
  if a.x1000:   # a label answers on a 0-1 scale when most of its points have both coordinates <= 1.0 ([0, 0] alone is not)
    pts = collections.defaultdict(list)
    for x in rows:
      if x["point"] is not None:
        pts[x["label"]].append(max(x["point"]) <= 1.0)
    scaled = {lab for lab, v in pts.items() if sum(v) * 2 > len(v)}
    rows = [r for r in rows if r["label"] in scaled]
  out = collections.OrderedDict()
  per = collections.defaultdict(dict)
  dist = collections.defaultdict(dict)
  inside = collections.defaultdict(dict)
  for r in rows:
    p = r["point"]
    if a.x1000 and p is not None and max(p) <= 1.0:
      p = [p[0] * 1000, p[1] * 1000]
    c = classify(scene, items[r["id"]], p, truth[r["id"]])
    if not a.x1000 and (c == "right") != bool(r["ok"]):
      raise SystemExit(f"verdict mismatch {r['label']} {r['id']} {r['point']} {c} ok={r['ok']}")
    out.setdefault(r["label"], collections.Counter())[c] += 1
    per[r["label"]][r["id"]] = c
    if c not in ("right", "no point"):
      dist[r["label"]][r["id"]] = dist_to(truth[r["id"]], p)
    elif c == "right":
      inside[r["label"]][r["id"]] = dist_to(~truth[r["id"]], p)
  for lab, cnt in out.items():
    near = sum(1 for x in dist[lab].values() if x is not None and x <= 20)
    print(f"{lab:34s}{' (x 1000 reading)' if a.x1000 else ''} " + "  ".join(f"{k} {cnt.get(k, 0)}" for k in ORDER) +
          f"  (n {sum(cnt.values())})  | wrong within 20 units: {near}, right within 20 units of the edge: " +
          str(sum(1 for x in inside[lab].values() if x is not None and x <= 20)))
  if a.json:
    json.dump({"score": os.path.abspath(a.score), "x1000": a.x1000, "counts": {k: dict(v) for k, v in out.items()},
               "per_target": per, "dist": dist, "inside": inside}, open(a.json, "w"), indent=1)
  return 0


if __name__ == "__main__":
  sys.exit(main())
