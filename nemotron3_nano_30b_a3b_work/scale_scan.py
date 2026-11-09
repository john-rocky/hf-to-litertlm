#!/usr/bin/env python3
"""scale_scan.py PACK_DIR - the scale scan of an nh_pack.py output (the packer floors every block scale at 2^-14;
this scan is the check).
Reads index.tsv + weights.bin (mmap): every blockwise scale (the xb4 / xb8 experts' "<name>_scales" fp32 tensors and the
bw4 FCs' inline scales) must be finite, > 0 and >= 2^-14; every per-channel int8 scale (pc8) finite and > 0 (no floor rule
for per-channel); every fp32 tensor finite. Prints one line per problem, then a summary; writes PACK_DIR/scale_scan.json.
Exit 0 = no problem, 1 = a problem.
"""
import json
import os
import sys

import numpy as np

FLOOR = 2.0 ** -14


def main():
    d = sys.argv[1]
    mm = np.memmap(os.path.join(d, "weights.bin"), dtype=np.uint8, mode="r")
    rows = [l.rstrip("\n").split("\t") for l in open(os.path.join(d, "index.tsv")) if l.strip()]
    names = {r[0] for r in rows}
    bad, stats = [], {"blockwise_tensors": 0, "blockwise_scales": 0, "pc8_tensors": 0, "pc8_scales": 0,
                      "fp32_tensors": 0, "min_blockwise": None, "min_pc8": None, "at_floor": 0}

    def f32(off, count):
        return np.frombuffer(mm[off:off + 4 * count], dtype=np.float32)

    for name, kind, shape, off, nbytes, soff, scount, block in rows:
        off, nbytes, soff, scount = int(off), int(nbytes), int(soff), int(scount)
        if kind == "fp32":
            v = f32(off, nbytes // 4)
            stats["fp32_tensors"] += 1
            if name.endswith("_scales") and name[: -len("_scales")] in names:
                stats["blockwise_tensors"] += 1
                stats["blockwise_scales"] += v.size
                m = float(v.min()) if v.size else None
                stats["min_blockwise"] = m if stats["min_blockwise"] is None else min(stats["min_blockwise"], m)
                stats["at_floor"] += int((v == np.float32(FLOOR)).sum())
                if not np.isfinite(v).all() or (v <= 0).any() or (v < np.float32(FLOOR)).any():
                    bad.append(f"{name}: blockwise scale out of range (min {m}, non-finite {int((~np.isfinite(v)).sum())})")
            elif not np.isfinite(v).all():
                bad.append(f"{name}: non-finite fp32 values ({int((~np.isfinite(v)).sum())})")
        elif kind == "bw4":
            v = f32(soff, scount)
            stats["blockwise_tensors"] += 1
            stats["blockwise_scales"] += v.size
            m = float(v.min())
            stats["min_blockwise"] = m if stats["min_blockwise"] is None else min(stats["min_blockwise"], m)
            stats["at_floor"] += int((v == np.float32(FLOOR)).sum())
            if not np.isfinite(v).all() or (v < np.float32(FLOOR)).any():
                bad.append(f"{name}: bw4 scale out of range (min {m})")
        elif kind == "pc8":
            v = f32(soff, scount)
            stats["pc8_tensors"] += 1
            stats["pc8_scales"] += v.size
            m = float(v.min())
            stats["min_pc8"] = m if stats["min_pc8"] is None else min(stats["min_pc8"], m)
            if not np.isfinite(v).all() or (v <= 0).any():
                bad.append(f"{name}: pc8 scale not finite / not positive (min {m})")
    for b in bad:
        print(b)
    res = {"pack": os.path.abspath(d), "problems": len(bad), **stats}
    json.dump(res, open(os.path.join(d, "scale_scan.json"), "w"), indent=1)
    print("SCAN", "PASS" if not bad else "FAIL", json.dumps(res))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
