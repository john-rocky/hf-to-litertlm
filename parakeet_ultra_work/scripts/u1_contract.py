#!/usr/bin/env python3
"""Signature check of the built ultra r1 files against CONTRACT.md (redux v3's names, dtypes, shapes + the decoder of
redux v2), from the census JSONs written by r2_fb.py census (results/r1/census_<file>.json), plus bytes and sha256 of
every file present (no model execution). Copied from parakeet_redux_work/scripts/r4_contract.py: out/r1, this lane's
forms, mel30 and the decoder files added.
-> results/r1/contract_check.json, results/r1/files.sha256; exit 1 on a mismatch.
CLI: u1_contract.py [--no-sha]
"""
import hashlib
import json
import os
import sys

T_MEL = {10: 1000, 20: 2000, 30: 3000}


def mel_want(b):
    tm = T_MEL[b]
    return {"serving_default": {
        "inputs": {"audio": ("FLOAT32", [1, 160 * tm]), "n_valid": ("INT32", [1])},
        "outputs": {"features": ("FLOAT32", [1, 128, tm]), "mel_mask": ("FLOAT32", [1, tm]),
                    "frame_mask": ("FLOAT32", [1, tm // 8])}}}


def enc_want(b):
    tm = T_MEL[b]
    return {"serving_default": {
        "inputs": {"features": ("FLOAT32", [1, 128, tm]), "mel_mask": ("FLOAT32", [1, tm])},
        "outputs": {"enc": ("FLOAT32", [1, tm // 8, 640])}}}


DEC = {"predict": {"inputs": {"token": ("INT32", [1, 1]), "h": ("FLOAT32", [2, 1, 640]), "c": ("FLOAT32", [2, 1, 640])},
                    "outputs": {"dec_out": ("FLOAT32", [1, 640]), "h_out": ("FLOAT32", [2, 1, 640]),
                                "c_out": ("FLOAT32", [2, 1, 640])}},
       "joint": {"inputs": {"enc_frame": ("FLOAT32", [1, 640]), "dec_out": ("FLOAT32", [1, 640])},
                 "outputs": {"logits": ("FLOAT32", [1, 8198])}}}
FILES = {"mel10": mel_want(10), "mel20": mel_want(20), "mel30": mel_want(30), "dec": DEC, "dec_int8": DEC}
for _b in (10, 20, 30):
    for _f in ("fp32", "fp32s", "fp16ws", "int8", "i4b32", "i4b128"):
        FILES[f"enc{_b}_{_f}"] = enc_want(_b)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


def main(argv):
    rep, bad, sums = {}, [], []
    for f, want in FILES.items():
        p = f"results/r1/census_{f}.json"
        path = f"out/r1/{f}.tflite"
        if not os.path.exists(p) or not os.path.exists(path):
            rep[f] = "not built"
            continue
        c = json.load(open(p))
        got = {s["key"]: s for s in c["signatures"]}
        rows = {}
        for key, w in want.items():
            s = got.get(key)
            if s is None:
                bad.append((f, key, "signature missing", sorted(got)))
                continue
            for io in ("inputs", "outputs"):
                g = {x["name"]: (x["dtype"], x["shape"]) for x in s[io]}
                for name, (dt, shp) in w[io].items():
                    if g.get(name) != (dt, shp):
                        bad.append((f, key, io, name, "want", dt, shp, "got", g.get(name)))
                extra = sorted(set(g) - set(w[io]))
                if extra:
                    bad.append((f, key, io, "extra", extra))
                rows[f"{key}.{io}"] = g
        extra_sigs = sorted(set(got) - set(want))
        if extra_sigs:
            bad.append((f, "extra signatures", extra_sigs))
        rep[f] = rows
        if "--no-sha" not in argv:
            sums.append((f, os.path.getsize(path), sha256(path)))
    rep["mismatches"] = bad
    rep["files"] = [{"file": f"{f}.tflite", "bytes": n, "sha256": h} for f, n, h in sums]
    json.dump(rep, open("results/r1/contract_check.json", "w"), indent=1)
    if sums:
        with open("results/r1/files.sha256", "w") as fo:
            for f, n, h in sums:
                fo.write(f"{h}  {n}  out/r1/{f}.tflite\n")
    print("contract check:", "OK" if not bad else bad, "| files:", len(sums), flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
