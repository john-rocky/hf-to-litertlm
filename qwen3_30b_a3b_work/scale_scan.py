#!/usr/bin/env python3
"""scale_scan.py PACK_DIR OUT.json

Scale check of a tapi_pack_moe.py pack: after quantization, count the zero and the non-finite scales (0 of each is
the condition). Reads index.tsv and weights.bin (memmap, one tensor at a time):
- expert scales = the fp32 tensors <name>_scales of xb4 / xb8 / xa8 experts ([out, E, 1, blocks]);
- FC scales = the per-row (pc8) or per-block (bw4) fp32 scales stored at scale_off of quantized attention / lm_head.
Per tensor: count, zeros, non-finite, min, max, and how many sit exactly at the 2^-14 floor (blockwise only; the pack
raises block scales to it). Also scans every fp32 weight for non-finite values. Exit 0 = no zero and no non-finite
scale anywhere; 3 otherwise (the JSON says where).
"""
import json
import os
import sys

import numpy as np

FLOOR = np.float32(2.0 ** -14)


def main():
    pack, out = sys.argv[1], sys.argv[2]
    rows = []
    for line in open(os.path.join(pack, "index.tsv")):
        f = line.rstrip("\n").split("\t")
        rows.append({"name": f[0], "kind": f[1], "shape": [int(x) for x in f[2].split(",")], "off": int(f[3]),
                     "bytes": int(f[4]), "scale_off": int(f[5]), "scale_count": int(f[6]), "block": int(f[7])})
    mm = np.memmap(os.path.join(pack, "weights.bin"), dtype=np.uint8, mode="r")
    per = []
    tot = {"scale_tensors": 0, "scales": 0, "zeros": 0, "nonfinite": 0, "at_floor": 0, "fp32_tensors": 0,
           "fp32_nonfinite": 0, "min_scale": None}
    for r in rows:
        if r["kind"] == "fp32":
            a = np.frombuffer(mm, dtype=np.float32, count=r["bytes"] // 4, offset=r["off"])
            nf = int((~np.isfinite(a)).sum())
            tot["fp32_tensors"] += 1
            tot["fp32_nonfinite"] += nf
            if not r["name"].endswith("_scales"):
                if nf:
                    per.append({"name": r["name"], "what": "fp32 weight", "nonfinite": nf})
                continue
            s, kind, blockwise = a, "expert scales", True
        elif r["kind"] in ("pc8", "bw4"):
            s = np.frombuffer(mm, dtype=np.float32, count=r["scale_count"], offset=r["scale_off"])
            kind, blockwise = f"{r['kind']} scales", r["kind"] == "bw4"
        else:
            continue  # the expert weight itself (its scales are the <name>_scales row)
        z = int((s == 0).sum())
        nf = int((~np.isfinite(s)).sum())
        fl = int((s == FLOOR).sum()) if blockwise else 0
        fin = s[np.isfinite(s)]
        mn = float(fin.min()) if fin.size else None
        tot["scale_tensors"] += 1
        tot["scales"] += int(s.size)
        tot["zeros"] += z
        tot["nonfinite"] += nf
        tot["at_floor"] += fl
        if mn is not None and (tot["min_scale"] is None or mn < tot["min_scale"]):
            tot["min_scale"] = mn
        per.append({"name": r["name"], "what": kind, "count": int(s.size), "zeros": z, "nonfinite": nf,
                    "at_floor": fl, "min": mn, "max": float(fin.max()) if fin.size else None})
    res = {"pack": os.path.abspath(pack), "total": tot,
           "floored_tensors": [p["name"] for p in per if p.get("at_floor")],
           "bad": [p for p in per if p.get("zeros") or p.get("nonfinite")],
           "per_tensor": per}
    json.dump(res, open(out, "w"), indent=1)
    ok = tot["zeros"] == 0 and tot["nonfinite"] == 0 and tot["fp32_nonfinite"] == 0
    print(f"SCALE_SCAN {'OK' if ok else 'BAD'}: {tot['scale_tensors']} scale tensors, {tot['scales']} scales, "
          f"zeros {tot['zeros']}, non-finite {tot['nonfinite']}, at the 2^-14 floor {tot['at_floor']} "
          f"({len(res['floored_tensors'])} tensors), min {tot['min_scale']}; fp32 tensors {tot['fp32_tensors']} "
          f"non-finite {tot['fp32_nonfinite']}")
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
