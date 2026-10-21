#!/usr/bin/env python3
"""Signature check of the built r4 files against CONTRACT.md v3 (names, dtypes, shapes), from the census JSONs written
by r2_fb.py census (results/r4/census_<file>.json), plus bytes and sha256 of every r4 file present (no model execution).
-> results/r4/contract_check.json, results/r4/files.sha256; exit 1 on a mismatch.
CLI: r4_contract.py [--no-sha]
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


FILES = {"mel10": mel_want(10), "mel20": mel_want(20)}
for _b in (10, 20):
    for _f in ("fp32", "int4x"):
        FILES[f"enc{_b}_{_f}"] = enc_want(_b)
for _b in (10, 20, 30):
    for _f in ("fp32s", "fp16ws"):
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
        p = f"results/r4/census_{f}.json"
        path = f"out/r4/{f}.tflite"
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
    json.dump(rep, open("results/r4/contract_check.json", "w"), indent=1)
    if sums:
        with open("results/r4/files.sha256", "w") as fo:
            for f, n, h in sums:
                fo.write(f"{h}  {n}  out/r4/{f}.tflite\n")
    print("contract check:", "OK" if not bad else bad, "| files:", len(sums), flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
