"""The record sets of the eight-sequence graph's checks, chosen before any number: the reference-answered records whose
every sequence fits a bucket (set/token_counts.json `longest`), and the chosen control arms
(conversion/red_arms_chosen_arms.json, pick_arms.py) whose arm and base both fit it.

  python3 conversion/make_subsets.py    -> conversion/ids_L128.json, conversion/red_arms_L128.json (and the same two
                                           files for L256, which no number on the card uses)
"""
import json
import os

WORK = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def main():
    tc = json.load(open(os.path.join(WORK, "set/token_counts.json")))
    longest = {**{k: v.get("longest") for k, v in tc["records"].items()},
               **{k: v.get("longest") for k, v in tc["arms"].items()}}
    ref = json.load(open(os.path.join(WORK, "out/ref/ref_fp32.json")))["records"]
    arms = json.load(open(os.path.join(WORK, "conversion/red_arms_chosen_arms.json")))
    for L, d in ((256, "quant"), (128, "batch")):
        ids = sorted(k for k, r in ref.items() if not r.get("error") and (longest.get(k) or 10 ** 9) <= L)
        keep = [a for a in arms if (longest.get(a["id"]) or 10 ** 9) <= L and (longest.get(a["base_id"]) or 10 ** 9) <= L]
        json.dump({"what": f"reference-answered records whose every sequence is <= {L} tokens (chosen before any number)",
                   "bucket": L, "count": len(ids), "ids": ids}, open(os.path.join(WORK, "conversion", f"ids_L{L}.json"), "w"),
                  indent=0)
        json.dump(keep, open(os.path.join(WORK, "conversion", f"red_arms_L{L}.json"), "w"), indent=1)
        print(L, len(ids), [a["id"] for a in keep])


if __name__ == "__main__":
    main()
