#!/usr/bin/env python3
"""tfl_unique_names.py IN.tflite OUT.tflite [--report R.json]

Gives every tensor of a .tflite a unique name and changes nothing else. ai-edge-quantizer 0.10.0 refuses a graph with
repeated tensor names ("Tensor name ... is not unique in the model ... ParamsGenerator assumes tensor names are
unique"), and tfl_rank4.py's rewrite repeats the names of some tensors it adds (`r4_begin` / `r4_end` / `r4_strides` at
every axis-0 STRIDED_SLICE site, and some `<name>/r4` tensors): the second and later holders of a name get "#2", "#3",
... appended. The file is read and written with tfl_rank4.py's own Graph / write (loaded from this directory by path:
the same external-buffer layout, constants copied byte for byte); op order, tensor indices, shapes, types and buffers
stay as they are. tfl_same_fn.py checks the result against IN.
"""
import argparse
import collections
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def load_rank4():
    spec = importlib.util.spec_from_file_location("tfl_rank4_by_path", os.path.join(HERE, "tfl_rank4.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--report")
    a = ap.parse_args()
    if os.path.exists(a.dst):
        raise SystemExit(f"refusing to overwrite {a.dst}")
    tr = load_rank4()
    g = tr.Graph(a.src)
    count = collections.Counter(t.name for t in g.sg.tensors)
    taken = set(count)
    seen = collections.Counter()
    renamed = []
    for i, t in enumerate(g.sg.tensors):
        if count[t.name] < 2:
            continue
        seen[t.name] += 1
        if seen[t.name] == 1:
            continue
        k = seen[t.name]
        new = t.name + b"#" + str(k).encode()
        while new in taken:
            k += 1
            new = t.name + b"#" + str(k).encode()
        taken.add(new)
        renamed.append({"tensor": i, "from": t.name.decode(errors="replace"), "to": new.decode(errors="replace")})
        t.name = new
    left = [n for n, c in collections.Counter(t.name for t in g.sg.tensors).items() if c > 1]
    if left:
        raise SystemExit(f"names still repeated: {left[:5]}")
    hlen, size = tr.write(g, a.dst)
    print(f"wrote {a.dst}: tensors {len(g.sg.tensors)} renamed {len(renamed)} header {hlen} B file {size} B", flush=True)
    if a.report:
        with open(a.report, "w") as f:
            json.dump({"src": a.src, "dst": a.dst, "renamed": renamed}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
