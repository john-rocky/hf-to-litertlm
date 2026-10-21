#!/usr/bin/env python3
"""P1 scoring of LANE §1b for the r4 runs (venv .venv-ref; r2_score.py's rules). For every finished LiteRT run
results/r4/p1_<form>_<backend>_<bkt|w30>_<set>[.s<k>of<n>].jsonl (r4_drive.py; a sharded run is scored when every
shard has its .DONE): window token ids decoded one window at a time with the TDT decode (tokenizers
Tokenizer.decode(ids, skip_special_tokens=True), LANE §0b-8), joined with a space, scored with wer.py (L = leaderboard
formula, J = score.py formula) against the manifest text and compared with B_ii (results/ref_bii_score.json):
P1 = WER <= B_ii + 0.15 (LS) / + 0.30 (FLEURS) for L and J, no empty hypothesis, no window with 10 symbols on one
frame. Also the utterances whose text differs from B_ii and the windows per bucket.
-> results/r4/p1_<tag>.txt.jsonl (id, text) and results/r4/p1_score.json (all runs found)
"""
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import redux_ref as R  # noqa: E402
import wer as W  # noqa: E402

SETS = {"ls": ("data/sets/librispeech-test-clean/manifest.jsonl", "en", 0.15),
        "de": ("data/sets/fleurs-de/manifest.jsonl", "de", 0.30),
        "fr": ("data/sets/fleurs-fr/manifest.jsonl", "fr", 0.30),
        "ru": ("data/sets/fleurs-ru/manifest.jsonl", "ru", 0.30)}
PAT = re.compile(r"^p1_(?P<tag>.+_(?P<set>ls|de|fr|ru))(?:\.s(?P<k>\d+)of(?P<n>\d+))?\.jsonl$")


def runs():
    groups = {}
    for path in sorted(glob.glob("results/r4/p1_*.jsonl")):
        base = os.path.basename(path)
        if base.endswith(".txt.jsonl"):
            continue
        m = PAT.match(base)
        if not m:
            continue
        n = int(m.group("n") or 1)
        g = groups.setdefault(m.group("tag"), {"set": m.group("set"), "n": n, "paths": []})
        g["paths"].append(path)
    for tag, g in groups.items():
        if len(g["paths"]) == g["n"] and all(os.path.exists(p + ".DONE") for p in g["paths"]):
            yield tag, g


def main():
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file("hf/redux/tokenizer.json")
    bii = json.load(open("results/ref_bii_score.json"))["sets"]
    out = {}
    for tag, g in runs():
        s = g["set"]
        man, lang, line = SETS[s]
        rows = R.read_jsonl(man)
        pred = {}
        for p in g["paths"]:
            pred.update({x["id"]: x for x in R.read_jsonl(p)})
        if len(pred) != len(rows):
            print(tag, "incomplete", len(pred), len(rows), flush=True)
            continue
        ref_ii = {x["id"]: x for x in R.read_jsonl(f"results/ref_bii_{s}.jsonl")}
        hyps, recs = [], []
        for r in rows:
            x = pred[r["id"]]
            texts = [tk.decode([i for i in t if i != R.BLANK], skip_special_tokens=True) for t in x["win_tokens"]]
            text = " ".join(t for t in texts if t)
            hyps.append(text)
            recs.append({"id": r["id"], "text": text})
        with open(f"results/r4/p1_{tag}.txt.jsonl", "w") as f:
            for x in recs:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        res, _ = W.score([r["text"] for r in rows], hyps, lang, [r["id"] for r in rows])
        b = bii[s]
        lim_l, lim_j = round(b["L"] + line, 3), round(b["J"] + line, 3)
        cap_win = sum(1 for x in pred.values() if x["cap_hits"] > 0 or x["max_run_same_frame"] >= 10)
        diff = [r["id"] for r, h in zip(rows, hyps) if h != ref_ii[r["id"]]["text"]]
        ok = (res["L"]["wer"] <= lim_l and res["J"]["wer"] <= lim_j and res["empty_hyp"] == 0 and cap_win == 0)
        hist = {}
        for x in pred.values():
            for bk in x.get("buckets", []):
                hist[str(bk)] = hist.get(str(bk), 0) + 1
        out[tag] = {"set": s, "shards": g["n"], "utterances": res["utterances"], "L": res["L"]["wer"],
                    "J": res["J"]["wer"], "J_SDI": [res["J"]["S"], res["J"]["D"], res["J"]["I"]],
                    "B_ii": {"L": b["L"], "J": b["J"]}, "delta": {"L": round(res["L"]["wer"] - b["L"], 3),
                                                                  "J": round(res["J"]["wer"] - b["J"], 3)},
                    "line": {"L": lim_l, "J": lim_j}, "empty_hyp": res["empty_hyp"], "cap_or_run10_utts": cap_win,
                    "max_run_same_frame": max(x["max_run_same_frame"] for x in pred.values()),
                    "eos_utts": sum(1 for x in pred.values() if x["eos"]),
                    "frame_mask_bad": sum(1 for x in pred.values() if not x["frame_mask_ok"]),
                    "windows_per_bucket": hist, "text_ne_bii": len(diff), "text_ne_bii_ids": diff[:40],
                    "P1": "PASS" if ok else "FAIL", "audio_s": round(sum(x["audio_s"] for x in pred.values()), 2),
                    "proc_s": round(sum(x["proc_s"] for x in pred.values()), 2)}
        print(tag, json.dumps({k: out[tag][k] for k in ("L", "J", "delta", "line", "empty_hyp", "cap_or_run10_utts",
                                                         "text_ne_bii", "windows_per_bucket", "P1")}), flush=True)
    json.dump(out, open("results/r4/p1_score.json", "w"), indent=1, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
