"""Choose the red arms on the reference alone (line (e) of score.py, both rows, before any candidate number): for each
frame of an arms file, the candidates whose watched value moves by more than 0.02 against their base on the provider's
float32 reference of d1-Omni-2B; the chosen arm of a frame is the one that moves most. The d1-2B conversion's rule
with the files as arguments. Written once; never changed after a candidate is scored.

  $REF conversion/pick_arms.py --arms set/red_arms.json --prefix conversion/red_arms_chosen
  $REF conversion/pick_arms.py --arms set/red_arms_audio.json --prefix conversion/red_arms_audio_chosen
  -> <prefix>.json (rule, chosen ids, every candidate's move) and <prefix>_arms.json (the chosen arms, score.py's
     --red-arms)
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from score import watched  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="out/ref/ref_fp32.json")
    ap.add_argument("--arms", required=True)
    ap.add_argument("--prefix", required=True)
    a = ap.parse_args()
    ref = json.load(open(os.path.join(WORK, a.ref)))["records"]
    arms = json.load(open(os.path.join(WORK, a.arms)))
    for p in (a.prefix + ".json", a.prefix + "_arms.json"):
        assert not os.path.exists(os.path.join(WORK, p)), f"written once: {p} exists"
    rows, by_frame = [], {}
    for arm in arms:
        w = arm["watch"]
        x, b = ref.get(arm["id"]), ref.get(arm["base_id"])
        va = watched(x["answers"].get(w["qid"]), w["field"]) if x and not x.get("error") else None
        vb = watched(b["answers"].get(w["qid"]), w["field"]) if b and not b.get("error") else None
        move = None if va is None or vb is None else va - vb
        row = {"id": arm["id"], "base_id": arm["base_id"], "frame": arm.get("frame"), "watch": w, "base": vb,
               "arm": va, "move": move, "qualifies": move is not None and abs(move) > 0.02}
        rows.append(row)
        if row["qualifies"]:
            by_frame.setdefault(arm.get("frame"), []).append(row)
    chosen = [max(v, key=lambda r: abs(r["move"])) for _, v in sorted(by_frame.items())]
    chosen_ids = {r["id"] for r in chosen}
    out = {"rule": "per frame, the candidate moving its watched value most on the float32 reference, among those "
                   "moving it by > 0.02", "ref": a.ref, "arms": a.arms, "chosen": [r["id"] for r in chosen],
           "candidates": rows, "frames_without_arm": sorted({a_.get("frame") for a_ in arms} - set(by_frame))}
    json.dump(out, open(os.path.join(WORK, a.prefix + ".json"), "w"), indent=1)
    json.dump([x for x in arms if x["id"] in chosen_ids], open(os.path.join(WORK, a.prefix + "_arms.json"), "w"), indent=1)
    for r in rows:
        mark = "CHOSEN" if r["id"] in chosen_ids else ("ok" if r["qualifies"] else "--")
        print(f"{mark:6s} {str(r['frame']):14s} {r['id']:40s} {r['watch']['qid']}.{r['watch']['field']:28s} "
              f"base {r['base']} arm {r['arm']} move {r['move']}")
    print("frames without an arm:", out["frames_without_arm"])


if __name__ == "__main__":
    main()
