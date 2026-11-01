#!/usr/bin/env python3
"""The detector + fixed-rule arm: one point per request from the detector's boxes and a rule written before any run.

  python tools/demo_scene_check/crule.py --fixtures DIR/fixtures.json --detections DIR/c/detections.json \
      --out DIR/c/crule.json

Per request: the box of each reference = the detector's best-scoring box for its det_query, or, when the reference
carries a pick ("left" / "right" / "front" / "back" / "top" / "bottom", for a name with two of a kind), the box at
that side among the phrase's boxes (centre x smallest / largest, centre y largest (front, bottom) / smallest (back,
top)). Then, in 0-1000 of the photo:
  left / right of A   -> A's box centre moved by A's box width to that side
  in front of A       -> A's box centre moved down by A's box height (nearer the camera = lower in the picture)
  behind A            -> A's box centre moved up by A's box height
  between A and B     -> the midpoint of the two box centres
  on S (a surface)    -> the centre of S's box
and the point clipped to 0-1000. A reference the detector did not find = no point = wrong.
"""
import argparse
import json
import os
import sys


def centre(b):
  return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def pick_box(boxes, pick):
  if not boxes:
    return None
  if not pick:
    return boxes[0]["box"]
  key = {"left": lambda b: centre(b)[0], "right": lambda b: -centre(b)[0], "front": lambda b: -centre(b)[1],
         "bottom": lambda b: -centre(b)[1], "back": lambda b: centre(b)[1], "top": lambda b: centre(b)[1]}[pick]
  return sorted((x["box"] for x in boxes), key=key)[0]


def clip(p):
  return [min(1000.0, max(0.0, p[0])), min(1000.0, max(0.0, p[1]))]


def rule_point(it, det):
  phr = det["photos"][it["photo"]]["phrases"]
  boxes = []
  for r in it["refs"]:
    b = pick_box(phr[r["det_query"]]["boxes"], r.get("pick"))
    if b is None:
      return None, f"not found: {r['det_query']}"
    boxes.append(b)
  rel = it["rel"]
  if rel in ("left", "right", "front", "behind", "on"):
    b = boxes[0]
    cx, cy = centre(b)
    w, h = b[2] - b[0], b[3] - b[1]
    p = {"left": (cx - w, cy), "right": (cx + w, cy), "front": (cx, cy + h), "behind": (cx, cy - h),
         "on": (cx, cy)}[rel]
  elif rel == "between":
    (x1, y1), (x2, y2) = centre(boxes[0]), centre(boxes[1])
    p = ((x1 + x2) / 2, (y1 + y2) / 2)
  else:
    raise ValueError(rel)
  return clip(p), f"{rel} of {[r['det_query'] for r in it['refs']]} boxes {boxes}"


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--fixtures", required=True)
  ap.add_argument("--detections", required=True)
  ap.add_argument("--out", required=True)
  a = ap.parse_args()
  fx = json.load(open(a.fixtures))
  det = json.load(open(a.detections))
  cases = []
  for it in fx["items"]:
    p, how = rule_point(it, det)
    cases.append({"key": f"{it['id']}|CRULE|off|rule", "id": it["id"], "photo": it["photo"], "point": p, "how": how})
    print(it["id"], p, how[:120])
  os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
  json.dump({"form": "rule", "rules": __doc__, "detections": os.path.abspath(a.detections), "cases": cases},
            open(a.out, "w"), indent=1, ensure_ascii=False)
  print("CRULE_DONE", a.out)
  return 0


if __name__ == "__main__":
  sys.exit(main())
