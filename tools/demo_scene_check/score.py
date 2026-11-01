#!/usr/bin/env python3
"""Score every answer against the truth masks, with the rules fixed before any model answered.

  python tools/demo_scene_check/score.py --fixtures DIR/fixtures.json --subject lfm3 \
      [--raw DIR/raw/<run>.jsonl ...] [--bundle DIR/litertlm/<label>.json ...] [--crule DIR/c/crule.json] \
      --out DIR/score/all.json
  python tools/demo_scene_check/score.py --selftest

--subject = the model the demo is about. Its picture arms and every .litertlm run are scored by the registered reading
only; every other arm (the rivals, and the detector-to-text arm even when its text LLM is the subject) also gets the
favourable readings. Give the --raw files in run order: a label's first appearance is the plan order that breaks ties
in table.py.

answer part: the generated text (special tokens kept) after the last thinking close (the record's think_close; else
  "</think>", "<channel|>" for gemma4), end-of-turn tokens removed; thinking on without the close = no answer = wrong
  (a model that answers without opening a thought counts here too: read those apart if they matter). A .litertlm
  reply with thinking on: the runtime splits the thought off; an empty answer = no answer. A .litertlm case whose
  child returned no reply (rc != 0) = no answer = wrong.
point: the first of
  (1) "point_2d": [x, y]
  (2) "bbox_2d": [x1, y1, x2, y2] -> centre
  (3) Google's key order: "box_2d": [y1, x1, y2, x2] -> centre; "point": [y, x]
  (4) the first bracket / parenthesis group of only numbers: 2 numbers = a point, 4 = a box -> centre, in [x, y]
      order (the record's coord_order "yx", or model gemma4: [y, x] / [y1, x1, y2, x2]).
  Coordinates are 0-1000 of the picture. Right = the truth mask is set at the pixel (min(999, int(x)),
  min(999, int(y))); no readable point or a coordinate outside 0-1000 = wrong. Several points: the first.
detector + rule: the point the rule computed (crule.json); no point (a phrase not found) = wrong.
Favourable readings (rivals only): an answer is also right when it lands with (a) every number <= 1.0 read x 1000,
  (b) x and y swapped (of the registered and of the x 1000 reading), (c) only when a number of the answer is > 1000:
  pixels of the image given to the model, or of the processor's resized image, divided by its size x 1000 (both
  orders). Both verdicts are kept per answer.
"""
import argparse
import importlib.util
import json
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name):
  spec = importlib.util.spec_from_file_location("demo_scene_" + name, os.path.join(HERE, name + ".py"))
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


REG = _load("regions")
_MASKS = {}
END_TOKENS = ("<|im_end|>", "<|endoftext|>", "<|pad|>", "</s>", "<turn|>", "<eos>", "<|tool_response>")
NUM = r"-?\d+(?:\.\d+)?"
READERS = {"gemma4": ("<channel|>", "yx")}   # model -> (thinking close, bare-number order) for records without the fields
SUBJECT = None


def reader(rec):
  close, order = READERS.get(rec.get("model"), ("</think>", "xy"))
  return rec.get("think_close") or close, rec.get("coord_order") or order


def answer_part(rec):
  raw = rec.get("text_raw") or rec.get("content") or ""
  for t in END_TOKENS:
    raw = raw.replace(t, "")
  close = reader(rec)[0]
  if close in raw:
    return raw.rsplit(close, 1)[1].strip(), None
  if rec.get("think") == "on" and rec.get("thought") is not None:   # a .litertlm reply: the runtime split the thought off
    return (raw.strip(), None) if raw.strip() else (None, "think on: thought channel only, no answer")
  if rec.get("think") == "on":
    return None, f"think on: no {close} (cut at the cap or never closed)"
  return raw.strip(), None


def parse_point(ans, order):
  """-> ((x, y), how, raw_numbers, order) with order 'xy' or 'yx' (the order the raw numbers were read in)."""
  m = re.search(r'"point_2d"\s*:\s*\[\s*(' + NUM + r')\s*,\s*(' + NUM + r')\s*\]', ans)
  if m:
    a, b = float(m.group(1)), float(m.group(2))
    return (a, b), "point_2d", [a, b], "xy"
  m = re.search(r'"bbox_2d"\s*:\s*\[\s*(' + NUM + r')\s*,\s*(' + NUM + r')\s*,\s*(' + NUM + r')\s*,\s*(' + NUM + r')\s*\]', ans)
  if m:
    a, b, c, d = map(float, m.groups())
    return ((a + c) / 2, (b + d) / 2), "bbox_2d centre", [a, b, c, d], "xy"
  m = re.search(r'"box_2d"\s*:\s*\[\s*(' + NUM + r')\s*,\s*(' + NUM + r')\s*,\s*(' + NUM + r')\s*,\s*(' + NUM + r')\s*\]', ans)
  if m:
    y1, x1, y2, x2 = map(float, m.groups())
    return ((x1 + x2) / 2, (y1 + y2) / 2), "box_2d [y1,x1,y2,x2] centre", [y1, x1, y2, x2], "yx"
  m = re.search(r'"point"\s*:\s*\[\s*(' + NUM + r')\s*,\s*(' + NUM + r')\s*\]', ans)
  if m:
    y, x = float(m.group(1)), float(m.group(2))
    return (x, y), "point [y,x]", [y, x], "yx"
  yx = order == "yx"
  for g in re.finditer(r"[\[(]([^\[\]()]*)[\])]", ans):
    parts = [p.strip() for p in g.group(1).split(",")]
    if parts and all(re.fullmatch(NUM, p) for p in parts):
      v = list(map(float, parts))
      if len(v) == 2:
        return ((v[1], v[0]) if yx else (v[0], v[1])), "bare pair", v, "yx" if yx else "xy"
      if len(v) == 4:
        if yx:
          return ((v[1] + v[3]) / 2, (v[0] + v[2]) / 2), "bare box centre [y1,x1,y2,x2]", v, "yx"
        return ((v[0] + v[2]) / 2, (v[1] + v[3]) / 2), "bare box centre", v, "xy"
  return None, "no point", None, None


def in_poly(pt, poly):
  x, y = pt
  inside = False
  n = len(poly)
  for i in range(n):
    x1, y1 = poly[i]
    x2, y2 = poly[(i + 1) % n]
    cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
    if abs(cross) < 1e-9 and min(x1, x2) - 1e-9 <= x <= max(x1, x2) + 1e-9 and min(y1, y2) - 1e-9 <= y <= max(y1, y2) + 1e-9:
      return True
    if (y1 > y) != (y2 > y):
      xi = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
      if x < xi:
        inside = not inside
  return inside


def in_truth(pt, truth):
  if "mask_path" in truth:
    if truth["mask_path"] not in _MASKS:
      _MASKS[truth["mask_path"]] = REG.load(truth["mask_path"])
    return REG.hit(_MASKS[truth["mask_path"]], pt)
  if "mask_array" in truth:
    return REG.hit(truth["mask_array"], pt)
  return any(in_poly(pt, p) for p in truth["polys"])


def valid(pt):
  return pt is not None and 0 <= pt[0] <= 1000 and 0 <= pt[1] <= 1000


def favourable_points(pt, raw, rec):
  """Every favourable reading of one answer's point (list of (name, (x, y)))."""
  out = []
  if pt is None:
    return out
  x, y = pt
  out.append(("swap", (y, x)))
  if raw and all(abs(v) <= 1.0 for v in raw):
    out += [("x1000", (x * 1000, y * 1000)), ("x1000 swap", (y * 1000, x * 1000))]
  if raw and any(abs(v) > 1000 for v in raw):
    sizes = []
    if rec.get("image_size_wh"):
      sizes.append(("px input", rec["image_size_wh"]))
    if rec.get("resized_wh"):
      sizes.append(("px processed", rec["resized_wh"]))
    for nm, (w, h) in sizes:
      out += [(nm, (x / w * 1000, y / h * 1000)), (nm + " swap", (y / w * 1000, x / h * 1000))]
  return out


def is_registered_only(rec):
  return rec.get("model") == SUBJECT and rec.get("arm") != "CLLM"


def score_one(rec, truth):
  res = {"ok": False, "ok_fav": False, "how": None, "point": None, "fav_how": None}
  if rec.get("arm") == "CRULE":
    pt = rec.get("point")
    res["how"] = rec.get("how")
    if pt is not None:
      res["point"] = [round(pt[0], 1), round(pt[1], 1)]
      res["ok"] = res["ok_fav"] = valid(pt) and in_truth(pt, truth)
    return res
  ans, why = answer_part(rec)
  res["how"] = why
  if ans is None:
    return res
  pt, how, raw, order = parse_point(ans, reader(rec)[1])
  res["how"] = how
  if pt is not None:
    res["point"] = [round(pt[0], 1), round(pt[1], 1)]
  res["ok"] = valid(pt) and in_truth(pt, truth)
  res["ok_fav"] = res["ok"]
  if not res["ok"] and not is_registered_only(rec) and not rec.get("bundle") and pt is not None:
    for nm, p in favourable_points(pt, raw, rec):
      if valid(p) and in_truth(p, truth):
        res["ok_fav"], res["fav_how"] = True, nm
        break
  return res


def load_raw(p):
  for line in open(p):
    if line.strip():
      yield json.loads(line)


def resized_wh(rec):
  if rec.get("image_grid_thw"):          # Qwen: (t, h, w) patches of 16 px
    t, h, w = rec["image_grid_thw"][0]
    return [w * 16, h * 16]
  if rec.get("spatial_shapes"):          # LFM2-VL family: (h, w) patches of 16 px
    h, w = rec["spatial_shapes"][0]
    return [w * 16, h * 16]
  return None


def lookup(fixtures):
  fx = json.load(open(fixtures))
  d = os.path.dirname(os.path.abspath(fixtures))
  items = {}
  for it in fx["items"]:
    items[it["id"]] = dict(it, q_type="place", truth=dict(it["truth"], mask_path=os.path.join(d, it["truth"]["mask"])))
  for ob in fx["objects"]:
    items[ob["id"]] = dict(ob, q_type="obj", truth=dict(ob["truth"], mask_path=os.path.join(d, ob["truth"]["mask"])))
  return items


def row_of(src, rec, it, s, label, model, arm, think, form):
  return {"src": os.path.relpath(src), "key": rec.get("key"), "label": label, "model": model, "arm": arm,
          "think": think, "form": form, "id": it["id"], "photo": it["photo"], "q_type": it["q_type"],
          "rel_kind": it.get("rel_kind"), "rule_solvable": it.get("rule_solvable"),
          "n_input_ids": rec.get("n_input_ids") if "prefill_tokens" not in rec else rec.get("prefill_tokens"),
          "n_new": rec.get("n_new") if "decode_tokens" not in rec else rec.get("decode_tokens"),
          "think_tokens": rec.get("think_tokens"), "ended_on_eos": rec.get("ended_on_eos"), **s,
          "answer_tail": ((answer_part(rec)[0] or "") if arm != "CRULE" else str(rec.get("point")))[-300:]}


def main():
  global SUBJECT
  ap = argparse.ArgumentParser()
  ap.add_argument("--fixtures")
  ap.add_argument("--subject")
  ap.add_argument("--raw", action="append", default=[])
  ap.add_argument("--bundle", action="append", default=[], help="a run_litertlm.py output file")
  ap.add_argument("--crule", action="append", default=[])
  ap.add_argument("--out")
  ap.add_argument("--selftest", action="store_true")
  a = ap.parse_args()
  if a.selftest:
    return selftest()
  if not (a.fixtures and a.subject and a.out):
    ap.error("--fixtures, --subject and --out are required")
  SUBJECT = a.subject
  items = lookup(a.fixtures)
  rows = []
  for p in a.raw:
    for rec in load_raw(p):
      rec["resized_wh"] = resized_wh(rec)
      it = items[rec["id"]]
      s = score_one(rec, it["truth"])
      label = f"{rec['model']} {rec['arm']} {rec['think']} {rec.get('form', 'std')}"
      rows.append(row_of(p, rec, it, s, label, rec["model"], rec["arm"], rec["think"], rec.get("form")))
  for p in a.bundle:
    b = json.load(open(p))
    for c in b["cases"]:
      if c.get("void"):
        continue
      it = items[c["id"]]
      th = c.get("think", b.get("think", "off"))
      arm = c["key"].split("|")[1] if c.get("key", "").count("|") == 3 else "S512"
      rec = {"model": SUBJECT, "arm": arm, "think": th, "content": c.get("content", ""), "bundle": b["label"],
             "thought": (c.get("channels") or {}).get("thought") if th == "on" else None, "key": c["key"],
             "prefill_tokens": c.get("prefill_tokens"), "decode_tokens": c.get("decode_tokens")}
      if "content" not in c:   # the child returned no reply (rc != 0): no answer = wrong
        s = {"ok": False, "ok_fav": False, "how": f"bundle: no reply (rc {c.get('rc')})", "point": None,
             "fav_how": None}
      else:
        s = score_one(rec, it["truth"])
      rows.append(row_of(p, rec, it, s, f"bundle {b['label']}", "bundle " + b["label"], arm, th, b["label"]))
  for p in a.crule:
    cr = json.load(open(p))
    for c in cr["cases"]:
      it = items[c["id"]]
      rec = dict(c, model="crule", arm="CRULE", think="off")
      s = score_one(rec, it["truth"])
      rows.append(row_of(p, rec, it, s, f"crule CRULE off {cr.get('form', 'rule')}", "crule", "CRULE", "off",
                         cr.get("form", "rule")))
  summary = {}
  for r in rows:
    k = (r["label"], r["q_type"])
    e = summary.setdefault(k, {"label": r["label"], "q_type": r["q_type"], "n": 0, "right": 0, "right_fav": 0,
                               "no_point": 0, "no_answer": 0, "n_new": [], "by_rel": {}})
    e["n"] += 1
    e["right"] += r["ok"]
    e["right_fav"] += r["ok_fav"]
    e["no_point"] += r["point"] is None
    e["no_answer"] += (r["how"] or "").startswith(("think on", "bundle: no reply"))
    if r["rel_kind"]:
      br = e["by_rel"].setdefault(r["rel_kind"], [0, 0, 0])
      br[0] += r["ok"]
      br[1] += r["ok_fav"]
      br[2] += 1
    if r["n_new"] is not None:
      e["n_new"].append(r["n_new"])
  summ = []
  for e in summary.values():
    nn = sorted(e.pop("n_new"))
    e["new_tokens_median"] = statistics.median(nn) if nn else None
    e["new_tokens_max"] = nn[-1] if nn else None
    summ.append(e)
  summ.sort(key=lambda e: (e["label"], e["q_type"]))
  os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
  json.dump({"rules": __doc__, "subject": SUBJECT, "fixtures": os.path.abspath(a.fixtures), "summary": summ,
             "rows": rows}, open(a.out, "w"), indent=1, ensure_ascii=False)
  for e in summ:
    rel = " ".join(f"{k} {v[0]}/{v[2]}(fav {v[1]})" for k, v in sorted(e["by_rel"].items()))
    print(f"{e['label']:34s} {e['q_type']:5s} {e['right']:2d}/{e['n']} (fav {e['right_fav']}) | {rel} | no-point "
          f"{e['no_point']} no-answer {e['no_answer']} new med {e['new_tokens_median']} max {e['new_tokens_max']}")
  return 0


def selftest():
  """Synthetic answers against a square truth [400..600] x [400..600]; every reader and every favourable reading."""
  global SUBJECT
  SUBJECT = "lfm3"
  truth = {"polys": [[[400, 400], [600, 400], [600, 600], [400, 600]]]}
  cases = [
      # (rec, ok, ok_fav)
      ({"model": "lfm3", "arm": "S512", "think": "off", "text_raw": '[{"point_2d": [500, 450], "label": "spot"}]<|im_end|>'}, True, True),
      ({"model": "lfm3", "arm": "S512", "think": "off", "text_raw": '[{"point_2d": [700, 450], "label": "spot"}]<|im_end|>'}, False, False),
      ({"model": "lfm3", "arm": "S512", "think": "off", "text_raw": '[{"point_2d": [0.5, 0.45]}]'}, False, False),   # subject: registered only
      ({"model": "lfm3", "arm": "S512", "think": "on", "text_raw": 'think think [{"point_2d": [500, 450]}]'}, False, False),  # no </think>
      ({"model": "lfm3", "arm": "S512", "think": "on", "text_raw": 'a [1, 2] </think>[{"point_2d": [500, 450]}]<|im_end|>'}, True, True),
      ({"model": "qwen35", "arm": "N", "think": "off", "text_raw": '{"point_2d": [0.5, 0.45], "label": "x"}'}, False, True),  # x1000
      ({"model": "qwen35", "arm": "N", "think": "off", "text_raw": '{"point_2d": [450, 700], "label": "x"}'}, False, False),
      ({"model": "qwen35", "arm": "N", "think": "off", "text_raw": '{"point_2d": [700, 450], "label": "x"}'}, False, False),
      ({"model": "qwen35", "arm": "N", "think": "off", "text_raw": '{"point_2d": [450, 550], "label": "x"}'}, True, True),
      ({"model": "qwen35", "arm": "N", "think": "off", "text_raw": '{"point_2d": [300, 500]}'}, False, False),
      ({"model": "qwen35", "arm": "N", "think": "off", "text_raw": '{"point_2d": [500, 300]}'}, False, False),
      ({"model": "qwen35", "arm": "N", "think": "off", "text_raw": '{"point_2d": [450, 300]}'}, False, False),
      ({"model": "qwen35", "arm": "N", "think": "off", "text_raw": '{"point_2d": [300, 450]}', "image_size_wh": [1000, 1000]}, False, False),
      ({"model": "qwen35", "arm": "N", "think": "off", "text_raw": '{"point_2d": [1500, 1200]}', "image_size_wh": [3000, 2400]}, False, True),  # px input
      ({"model": "lfm25_3", "arm": "N", "think": "off", "text_raw": '[{"image_id": 0, "bbox_2d": [420, 410, 580, 590], "label": "s"}]'}, True, True),
      ({"model": "lfm25_16", "arm": "N", "think": "off", "text_raw": '[0.42, 0.41, 0.58, 0.59]'}, False, True),
      ({"model": "gemma4", "arm": "N", "think": "off", "text_raw": '```json\n[{"box_2d": [410, 420, 590, 580], "label": "s"}]\n```<turn|>'}, True, True),
      ({"model": "gemma4", "arm": "N", "think": "off", "text_raw": '[{"point": [450, 500], "label": "s"}]'}, True, True),
      ({"model": "gemma4", "arm": "N", "think": "off", "text_raw": '[{"point": [300, 500], "label": "s"}]'}, False, False),  # x=500,y=300; swap (300,500) outside
      ({"model": "gemma4", "arm": "N", "think": "on", "text_raw": '<|channel>thought\nhmm<channel|>[{"point": [450, 500]}]<turn|>'}, True, True),
      ({"model": "gemma4", "arm": "N", "think": "on", "text_raw": '<|channel>thought\nhmm hmm'}, False, False),
      ({"model": "rival", "arm": "N", "think": "on", "coord_order": "yx", "think_close": "<channel|>", "text_raw": 'x<channel|>[450, 500]'}, True, True),  # record fields
      ({"model": "lfm3", "arm": "CLLM", "think": "off", "text_raw": '[0.5, 0.45]'}, False, True),   # detector-to-text with the subject's LLM = rival
      ({"model": "qwen35", "arm": "CLLM", "think": "off", "text_raw": 'The point is [510, 470].'}, True, True),
      ({"model": "crule", "arm": "CRULE", "think": "off", "point": [500, 500], "how": "right of laptop"}, True, True),
      ({"model": "crule", "arm": "CRULE", "think": "off", "point": None, "how": "laptop not found"}, False, False),
      ({"model": "lfm3", "arm": "S512", "think": "off", "content": '[{"point_2d": [500, 450]}]', "bundle": "i4_gpu"}, True, True),
      ({"model": "lfm3", "arm": "S512", "think": "off", "content": '[{"point_2d": [0.5, 0.45]}]', "bundle": "i4_gpu"}, False, False),
      ({"model": "lfm3", "arm": "S512", "think": "on", "content": '', "thought": "x", "bundle": "i4_gpu"}, False, False),
      ({"model": "lfm3", "arm": "S512", "think": "on", "content": '[{"point_2d": [500, 450]}]', "thought": "x", "bundle": "i4_gpu"}, True, True),
      ({"model": "qwen35", "arm": "N", "think": "off", "text_raw": 'nothing here'}, False, False),
      ({"model": "qwen35", "arm": "N", "think": "off", "text_raw": '{"point_2d": [600, 600]}'}, True, True),   # boundary in
      ({"model": "qwen35", "arm": "N", "think": "off", "text_raw": '{"point_2d": [-5, 500]}'}, False, False),
  ]
  bad = 0
  for i, (rec, ok, okf) in enumerate(cases):
    s = score_one(rec, truth)
    flag = "ok " if (s["ok"], s["ok_fav"]) == (ok, okf) else "BAD"
    bad += flag == "BAD"
    print(f"{flag} {i:2d} {rec['model']:8s} {rec['arm']:5s} {rec['think']:3s} -> ok {s['ok']} fav {s['ok_fav']} "
          f"({s['how']}; {s['fav_how']}) want {ok}/{okf}")
  L = {"polys": [[[0, 0], [100, 0], [100, 50], [50, 50], [50, 100], [0, 100]]]}
  for pt, want in (((25, 75), True), ((75, 75), False), ((75, 25), True), ((50, 50), True), ((101, 10), False)):
    got = in_truth(pt, L)
    bad += got != want
    print(("ok " if got == want else "BAD"), "poly", pt, got)
  sq = {"mask_array": REG.as_bool(REG.poly_img([[[400, 400], [600, 400], [600, 600], [400, 600]]]))}
  for pt, want in (((500, 450), True), ((400, 400), True), ((399.9, 500), False), ((599.5, 599.5), True),
                   ((601, 500), False), ((700, 450), False)):
    got = in_truth(pt, sq)
    bad += got != want
    print(("ok " if got == want else "BAD"), "mask", pt, got)
  edge = {"mask_array": REG.as_bool(REG.poly_img([[[900, 900], [1000, 900], [1000, 1000], [900, 1000]]]))}
  for pt, want in (((1000, 1000), True), ((999.6, 950), True), ((899, 950), False)):
    got = in_truth(pt, edge)
    bad += got != want
    print(("ok " if got == want else "BAD"), "mask edge", pt, got)
  for rec, want in (({"model": "lfm3", "arm": "S512", "think": "off", "text_raw": '[{"point_2d": [500, 450]}]'}, True),
                    ({"model": "qwen35", "arm": "N", "think": "off", "text_raw": '{"point_2d": [0.5, 0.45]}'}, True)):
    s = score_one(rec, sq)
    bad += s["ok_fav"] != want
    print(("ok " if s["ok_fav"] == want else "BAD"), "mask score_one", rec["model"], s["ok"], s["ok_fav"])
  print("SELFTEST", "PASS" if not bad else f"FAIL {bad}")
  return 1 if bad else 0


if __name__ == "__main__":
  sys.exit(main())
