"""Choose the red arms on the reference alone (line (e), registered before any candidate number): for each frame of
set/red_arms.json, the candidates whose watched value moves by more than 0.02 against their base on the provider's
float32 reference; the chosen arm of a frame is the one that moves most. Written once; never changed after a
candidate is scored.

  python3 conversion/pick_arms.py  ->  conversion/red_arms_chosen.json (+ every candidate's move)
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from score import watched  # noqa: E402


def main():
    ref = json.load(open(os.path.join(WORK, "out/ref/ref_fp32.json")))["records"]
    arms = json.load(open(os.path.join(WORK, "set/red_arms.json")))
    rows, by_frame = [], {}
    for arm in arms:
        w = arm["watch"]
        a, b = ref.get(arm["id"]), ref.get(arm["base_id"])
        va = watched(a["answers"].get(w["qid"]), w["field"]) if a and not a.get("error") else None
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
                   "moving it by > 0.02", "chosen": [r["id"] for r in chosen], "candidates": rows,
           "frames_without_arm": sorted({a.get("frame") for a in arms} - set(by_frame))}
    json.dump(out, open(os.path.join(HERE, "red_arms_chosen.json"), "w"), indent=1)
    json.dump([a for a in arms if a["id"] in chosen_ids], open(os.path.join(HERE, "red_arms_chosen_arms.json"), "w"),
              indent=1)
    for r in rows:
        mark = "CHOSEN" if r["id"] in chosen_ids else ("ok" if r["qualifies"] else "--")
        print(f"{mark:6s} {r['frame']:14s} {r['id']:40s} {r['watch']['qid']}.{r['watch']['field']:28s} "
              f"base {r['base']} arm {r['arm']} move {r['move']}")
    print("frames without an arm:", out["frames_without_arm"])


if __name__ == "__main__":
    main()
