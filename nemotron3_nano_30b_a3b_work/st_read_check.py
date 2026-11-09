#!/usr/bin/env python3
"""st_read_check.py DIR [--prefix P] — every tensor (or those under --prefix) read by st_read.py equals
safe_open(...).get_tensor(name).float() bit for bit (the torch tensors compared with torch.equal and as uint32 bits).
Exit 0 = all equal."""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from st_read import StReader  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--prefix", default="")
    a = ap.parse_args()
    from safetensors import safe_open
    r = StReader(a.dir)
    idx = os.path.join(a.dir, "model.safetensors.index.json")
    wm = json.load(open(idx))["weight_map"] if os.path.exists(idx) else None
    hs, n, bad = {}, 0, []
    for k in r.keys():
        if not k.startswith(a.prefix):
            continue
        fn = wm[k] if wm else "model.safetensors"
        if fn not in hs:
            hs[fn] = safe_open(os.path.join(a.dir, fn), framework="pt")
        want = hs[fn].get_tensor(k).float()
        got = r.get_torch(k)
        if want.shape != got.shape or not torch.equal(want.view(torch.int32), got.view(torch.int32)):
            bad.append(k)
        n += 1
    print(json.dumps({"tensors": n, "unequal": len(bad), "first": bad[:5]}))
    return 1 if bad or not n else 0


if __name__ == "__main__":
    sys.exit(main())
