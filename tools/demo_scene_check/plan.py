#!/usr/bin/env python3
"""Plans: the cases one model runs, with every prompt text fixed in this file before any model answers.

  python tools/demo_scene_check/plan.py --fixtures DIR/fixtures.json --model lfm3 --out DIR/plan/lfm3.json \
      [--prompt lfm3] [--conditions S512:off,N:off,S512:on,N:on] [--forms box,point] [--objects]
  python tools/demo_scene_check/plan.py --fixtures DIR/fixtures.json --model qwen35 --cllm DIR/c/detections.json \
      --out DIR/plan/cllm_qwen35.json [--conditions off,on]

--model = the name written into every case and record (a label is "<model> <arm> <think> <form>"); --prompt = the
prompt family (default: the model name when it is one): each family uses its own author's pointing or grounding
request, with the request put in as the object phrase, so every model is asked in the form its documentation shows:
  lfm3    the pointing request of the Liquid AI docs page "Vision Capabilities", section "Pointing"
          (https://docs.liquid.ai/lfm/key-concepts/vision-capabilities): a JSON list of point_2d in 0-1000
  qwen35  the point request of the Qwen3-VL cookbook (https://github.com/QwenLM/Qwen3-VL, cookbooks/2d_grounding.ipynb),
          its extra keys dropped: point_2d in JSON
  lfm25   the same Liquid AI page, section "LFM2.5-VL object detection and grounding": its system prompt and its
          "Provide bounding boxes for ..." user line (bbox_2d; the box centre is read as the point)
  gemma4  the Gemma docs show no pointing form: box = the detection request of
          https://ai.google.dev/gemma/docs/capabilities/vision/image (box_2d read [y1, x1, y2, x2]), point = the
          pointing request of https://ai.google.dev/gemini-api/docs/robotics-overview (its Python example, "no more
          than 10 items" replaced by the phrase; [y, x] in 0-1000); --forms box,point runs both
Arms: S512 = the 512 x 512 stretch (the app's pixels), N = the source picture (long side 1024) through the model's own
processor. --conditions = arm:think pairs in run order (think on = the chat template's enable_thinking). max_new_tokens:
think off 512, think on 4096 (--max-new-off, --max-new-on).
--objects: the reference objects asked alone ("Point to <name>.") instead of the placement requests, think off.
--cllm: the detector-to-text arm: no image; a text LLM gets the detector's boxes of the objects the request names (then
of the photo's other listed objects, up to 3 boxes per phrase, best first) and the request; conditions = think off, on.
"""
import argparse
import json
import os
import sys

TARGET = 'the free spot where the item should be placed for the request "{q}"'
LFM3_POINT = ('Point to {target}. Answer in JSON as a list of points: [{{"point_2d": [x, y], "label": "<short label>"}}, '
              '...], with x and y as integers normalized to 0-1000.')
QWEN_POINT = ('Locate {target} with points, report their point coordinates in JSON format like this: '
              '{{"point_2d": [x, y], "label": "object"}}')
LFM25_BOX_SYSTEM = """When asked for bounding boxes for objects, return a valid JSON array.
Each array item must be an object with:
- image_id: the 0-based index of the image
- bbox_2d: [xmin, ymin, xmax, ymax] normalized integer coordinates in [0, 1000]
- label: a concise label you choose for the predicted object or region

Return one item per visible matching object or region. Return [] if none are visible."""
LFM25_BOX_USER = "Provide bounding boxes for {target}."
GEMMA_BOX = "detect {target}, output only ```json"
GEMMA_POINT = ('Point to {target}. The label returned should be an identifying name for the object detected. The answer '
               'should follow the json format: [{{"point": <point>, "label": <label1>}}, ...]. The points are in [y, x] '
               'format normalized to 0-1000.')
CLLM_TEMPLATE = ("Below is the list of the objects that a detector found in a photo. Each line is one object: its number, "
                 "its name, and its box [left, top, right, bottom] in coordinates normalized to 0-1000 of the photo (x "
                 "grows to the right, y grows downward).\n\n{items}\n\n"
                 "A person asks a robot: \"{q}\"\n"
                 "At which point of the photo should the robot put the item down? Answer with one point [x, y] in the "
                 "same 0-1000 coordinates, and nothing else.")
CLLM_MAX_BOXES = 3
# family -> (system, {form: user text}, thinking close, coordinate order of bare numbers)
FAMILIES = {
    "lfm3": (None, {"std": LFM3_POINT}, "</think>", "xy"),
    "qwen35": (None, {"std": QWEN_POINT}, "</think>", "xy"),
    "lfm25": (LFM25_BOX_SYSTEM, {"std": LFM25_BOX_USER}, "</think>", "xy"),
    "gemma4": (None, {"box": GEMMA_BOX, "point": GEMMA_POINT}, "<channel|>", "yx"),
}


def case(it, ph, model, fam, arm, think, form, kind, max_new):
  system, forms, close, order = FAMILIES[fam]
  target = TARGET.format(q=it["request"]) if kind == "place" else it["name"]
  return {"key": f"{it['id']}|{arm}|{think}|{form}", "id": it["id"], "photo": it["photo"], "q_type": kind,
          "arm": arm, "think": think, "form": form, "image": ph["src"] if arm == "N" else ph["s512"],
          "system": system, "text": forms[form].format(target=target), "tools": None, "max_new": max_new[think],
          "think_close": close, "coord_order": order}


def cllm_items(det, it):
  lines, k = [], 1
  for r in it["refs"] + it.get("llm_refs", []):
    boxes = det["photos"][it["photo"]]["phrases"][r["det_query"]]["boxes"][:CLLM_MAX_BOXES]
    for b in boxes:
      lines.append(f"[{k}] {r['det_query']} [{', '.join(str(round(v)) for v in b['box'])}]")
      k += 1
  return "\n".join(lines) if lines else "(the detector found none of the named objects)"


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--fixtures", required=True)
  ap.add_argument("--model", required=True)
  ap.add_argument("--prompt", default=None, choices=sorted(FAMILIES))
  ap.add_argument("--out", required=True)
  ap.add_argument("--conditions", default=None)
  ap.add_argument("--forms", default=None)
  ap.add_argument("--objects", action="store_true")
  ap.add_argument("--cllm", default=None, help="detections.json of detect.py: write the detector-to-text plan")
  ap.add_argument("--max-new-off", type=int, default=512)
  ap.add_argument("--max-new-on", type=int, default=4096)
  a = ap.parse_args()
  fx = json.load(open(a.fixtures))
  ph = fx["photos"]
  max_new = {"off": a.max_new_off, "on": a.max_new_on}
  meta = {"model": a.model, "rules": __doc__, "fixtures": os.path.abspath(a.fixtures)}
  if a.cllm:
    det = json.load(open(a.cllm))
    thinks = (a.conditions or "off,on").split(",")
    cs = []
    for th in thinks:
      for it in fx["items"]:
        cs.append({"key": f"{it['id']}|CLLM|{th}|text", "id": it["id"], "photo": it["photo"], "q_type": "place",
                   "arm": "CLLM", "think": th, "form": "text", "image": None, "system": None,
                   "text": CLLM_TEMPLATE.format(items=cllm_items(det, it), q=it["request"]), "tools": None,
                   "max_new": max_new[th], "think_close": "</think>", "coord_order": "xy"})
    meta["detections"] = os.path.abspath(a.cllm)
  else:
    fam = a.prompt or a.model
    if fam not in FAMILIES:
      raise SystemExit(f"--prompt: {a.model} is not a prompt family; pass one of {sorted(FAMILIES)}")
    forms = (a.forms or ",".join(FAMILIES[fam][1])).split(",")
    default = "S512:off,N:off" if a.objects else "S512:off,N:off,S512:on,N:on"
    conds = [c.split(":") for c in (a.conditions or default).split(",")]
    cs = []
    if a.objects:
      for arm, th in conds:
        for form in forms:
          cs += [case(ob, ph[ob["photo"]], a.model, fam, arm, th, form, "obj", max_new) for ob in fx["objects"]]
    else:
      for form in forms:
        for arm, th in conds:
          cs += [case(it, ph[it["photo"]], a.model, fam, arm, th, form, "place", max_new) for it in fx["items"]]
    meta["prompt_family"] = fam
  meta["cases"] = cs
  os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
  json.dump(meta, open(a.out, "w"), indent=1, ensure_ascii=False)
  print(a.out, len(cs), "cases")
  return 0


if __name__ == "__main__":
  sys.exit(main())
