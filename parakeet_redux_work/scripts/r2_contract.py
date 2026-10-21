#!/usr/bin/env python3
"""Signature check of the built r2 files against CONTRACT.md (names, dtypes, shapes), from the census JSONs written
by r2_fb.py census (no model execution). -> results/r2/contract_check.json; exit 1 on a mismatch."""
import json
import os
import sys

WANT = {
    "mel30": {"serving_default": {
        "inputs": {"audio": ("FLOAT32", [1, 480000]), "n_valid": ("INT32", [1])},
        "outputs": {"features": ("FLOAT32", [1, 128, 3000]), "mel_mask": ("FLOAT32", [1, 3000]),
                    "frame_mask": ("FLOAT32", [1, 375])}}},
    "enc30": {"serving_default": {
        "inputs": {"features": ("FLOAT32", [1, 128, 3000]), "mel_mask": ("FLOAT32", [1, 3000])},
        "outputs": {"enc": ("FLOAT32", [1, 375, 640])}}},
    "dec": {"predict": {
        "inputs": {"token": ("INT32", [1, 1]), "h": ("FLOAT32", [2, 1, 640]), "c": ("FLOAT32", [2, 1, 640])},
        "outputs": {"dec_out": ("FLOAT32", [1, 640]), "h_out": ("FLOAT32", [2, 1, 640]),
                    "c_out": ("FLOAT32", [2, 1, 640])}},
        "joint": {"inputs": {"enc_frame": ("FLOAT32", [1, 640]), "dec_out": ("FLOAT32", [1, 640])},
                  "outputs": {"logits": ("FLOAT32", [1, 8198])}}},
}
FILES = {"mel30": "mel30", "enc30_fp32": "enc30", "enc30_int4x": "enc30", "enc30_fp16w": "enc30", "dec": "dec"}


def main():
    rep, bad = {}, []
    for f, kind in FILES.items():
        p = f"results/r2/census_{f}.json"
        if not os.path.exists(p):
            rep[f] = "no census"
            continue
        c = json.load(open(p))
        got = {s["key"]: s for s in c["signatures"]}
        rows = {}
        for key, want in WANT[kind].items():
            s = got.get(key)
            if s is None:
                bad.append((f, key, "signature missing", sorted(got)))
                continue
            for io in ("inputs", "outputs"):
                g = {x["name"]: (x["dtype"], x["shape"]) for x in s[io]}
                for name, (dt, shp) in want[io].items():
                    if g.get(name) != (dt, shp):
                        bad.append((f, key, io, name, "want", dt, shp, "got", g.get(name)))
                extra = sorted(set(g) - set(want[io]))
                if extra:
                    bad.append((f, key, io, "extra", extra))
                rows[f"{key}.{io}"] = g
        rep[f] = rows
    rep["mismatches"] = bad
    json.dump(rep, open("results/r2/contract_check.json", "w"), indent=1)
    print("contract check:", "OK" if not bad else bad, flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
