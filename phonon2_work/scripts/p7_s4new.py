#!/usr/bin/env python3
"""The token check's second part for the new buckets: every S4 window a new bucket takes under the 7-bucket rule (7.5 s: 80,640 < n <= 120,320 samples, 15 s:
160,000 < n <= 240,640, 25 s: 320,000 < n <= 400,640), through the 7-bucket fp32 route (mel<b> + enc<b>_fp32 + dec, Mac CPU), against P_ii's tokens
(results/r1/ref_pii_s4.jsonl win_tokens, the reference's rows of the S4 utterances). A differing window is listed with its steps; the noise rule (P's top-2 gap < 0.01 at the first differing step) is read by a
separate torch run when one shows up. Venv .venv-q, cwd = phonon2_work. -> results/r7/p0_s4new_<form>_<dec>.json. CLI: p7_s4new.py [--form fp32] [--dec dec] [--threads 4]"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p7_common  # noqa: E402

DR = p7_common.patch_drive()
NEW = tuple(p7_common.T_NEW)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--form", default="fp32")
    ap.add_argument("--dec", default="dec", choices=sorted(DR.DECS))
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    ref = {x["id"]: x for x in DR.read_jsonl("results/r1/ref_pii_s4.jsonl")}
    man = DR.read_jsonl(DR.SETS["s4"][0])
    p = DR.Pipeline(a.form, "cpu", a.threads, p7_common.B7, dec=a.dec)
    rows, differ = [], []
    t0 = time.time()
    n_all = 0
    for r in man:
        audio = DR.load_wav(r["path"])
        wins = DR.windows(len(audio))
        n_all += len(wins)
        for k, (s0, n) in enumerate(wins):
            if DR.bucket_of(n, p7_common.B7) not in NEW:
                continue
            out = p.window(audio[s0:s0 + n], keep_steps=True)
            want = [t for t in ref[r["id"]]["win_tokens"][k] if t != DR.BLANK]
            eq = out["tokens"] == want
            rows.append({"id": r["id"], "w": k, "samples": n, "bucket": out["bucket"], "tokens_eq": eq, "n_tok": len(out["tokens"]), "n_tok_ref": len(want),
                         "frame_mask_ok": out["frame_mask_sum"] == out["valid_enc"], "cap_hits": out["cap_hits"]})
            if not eq:
                differ.append({"id": r["id"], "w": k, "samples": n, "bucket": out["bucket"], "tokens": out["tokens"], "ref": want, "steps": out["steps"]})
    per = {str(b): {"windows": sum(1 for x in rows if x["bucket"] == b), "tokens_equal": sum(1 for x in rows if x["bucket"] == b and x["tokens_eq"])} for b in NEW}
    rep = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "form": a.form, "dec": a.dec, "threads": a.threads, "s4_utterances": len(man), "s4_windows": n_all,
           "windows_in_new_buckets": len(rows), "per_bucket": per, "tokens_equal": sum(x["tokens_eq"] for x in rows), "frame_mask_ok": all(x["frame_mask_ok"] for x in rows),
           "cap_hits": sum(x["cap_hits"] for x in rows), "differ": differ, "wall_s": round(time.time() - t0, 1), "pipeline": p.info(), "rows": rows}
    json.dump(rep, open(f"results/r7/p0_s4new_{a.form}_{a.dec}.json", "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("rows", "pipeline", "differ")}), "differ", [(d["id"], d["w"]) for d in differ], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
