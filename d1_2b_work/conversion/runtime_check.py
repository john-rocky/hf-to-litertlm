"""Does another ai-edge-litert release compute the shipped graphs the same way? (The phones' test app is built on
LiteRT 2.2.0; the Mac checks ran ai-edge-litert 2.3.0.) Runs one fixed input per graph (real table rows, half state,
three questions; for a tower the largest saved picture that fits the canvas) on the CPU (XNNPACK) and Metal float32
with the ai-edge-litert of the running Python, saves each output under out/rtcheck/<version>/, and diffs against the
same graph's output of every other version already saved there. numpy only (no torch). multi8_check.py imports its
decide_inputs(L).

  <python with ai-edge-litert 2.2.0> conversion/runtime_check.py out/graphs/d1-2b_decide_L128_fp16fc.tflite ...
  $EXPORT conversion/runtime_check.py out/graphs/d1-2b_decide_L128_fp16fc.tflite ...                 # 2.3.0
"""
from __future__ import annotations

import glob
import importlib.metadata as md
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)


def decide_inputs(L):
    import numpy as np

    from d1_layout import make_sees

    table = np.load(os.path.join(WORK, "tables/embed_table.npy"), mmap_mode="r")
    T = L - 5
    rng = np.random.default_rng(0)
    ids = rng.integers(0, 124000, T)
    rows = np.asarray(table[ids])
    emb = np.zeros((1, L, 2048), np.float32)
    emb[0, :T] = (rows.astype(np.uint32) << 16).view(np.float32) if rows.dtype == np.uint16 else rows
    state = T // 2
    tags = [-1] * state + [1 + (k * 3) // (T - state) for k in range(T - state)]
    q, k = make_sees(state, tags, L)
    return {"embeds": emb, "sees_q": q, "sees_k": k}, T


def tower_inputs(GH, GW):
    import numpy as np

    from d1_layout import to_canvas

    best = None
    for p in glob.glob(os.path.join(WORK, "out/pixels/*.npz")):
        z = np.load(p)
        gh, gw = (int(v) for v in z["grid"][-2:])
        if gh <= GH and gw <= GW and (best is None or gh * gw > best[0]):
            best = (gh * gw, p, z["pixel_values"], gh, gw)
    if best is None:
        raise SystemExit(f"no saved picture fits {GH}x{GW}")
    _, p, pv, gh, gw = best
    patches, valid = to_canvas(pv, gh, gw, GH, GW)
    return {"patches": patches, "valid": valid}, (os.path.basename(p), gh, gw)


def main():
    import numpy as np

    from lrt_backend import Graph

    ver = md.version("ai-edge-litert")
    out_dir = os.path.join(WORK, "out/rtcheck", ver)
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for arg in sys.argv[1:]:
        path = arg if os.path.isabs(arg) else os.path.join(WORK, arg)
        name = os.path.basename(path)[:-len(".tflite")]
        tower = "_tower_" in name
        if tower:
            GH, GW = (int(v) for v in re.search(r"_G(\d+)x(\d+)", name).groups())
            feeds, what = tower_inputs(GH, GW)
            sig, out_name = "tower", "features"
        else:
            L = int(re.search(r"_L(\d+)", name).group(1))
            feeds, what = decide_inputs(L)
            sig, out_name = "decide", "hidden"
        row = {"graph": os.path.relpath(path, WORK), "ai_edge_litert": ver, "input": what}
        outs = {}
        for acc in ("cpu", "gpu"):
            try:
                t0 = time.time()
                g = Graph(path, sig, acc, "fp32")
                load = time.time() - t0
                y = g(**feeds)[out_name]
                g.close()
                y = y[0, :what] if not tower else y
                outs[acc] = y
                np.save(os.path.join(out_dir, f"{name}__{acc}.npy"), y)
                row[acc] = {"load_s": round(load, 2), "fully_accelerated": g.fully_accelerated,
                            "finite": bool(np.isfinite(y).all()), "max_abs": float(np.abs(y).max())}
            except Exception as e:
                row[acc] = {"error": f"{type(e).__name__}: {e}"}
        if "cpu" in outs and "gpu" in outs:
            row["gpu_vs_cpu_max_abs"] = float(np.abs(outs["gpu"] - outs["cpu"]).max())
        for other in sorted(glob.glob(os.path.join(WORK, "out/rtcheck", "*"))):
            ov = os.path.basename(other)
            if ov == ver:
                continue
            for acc, y in outs.items():
                f = os.path.join(other, f"{name}__{acc}.npy")
                if os.path.exists(f):
                    row[f"{acc}_vs_{ov}_{acc}_max_abs"] = float(np.abs(y - np.load(f)).max())
        rows.append(row)
        print(json.dumps(row), flush=True)
    res = os.path.join(WORK, f"results/runtime_check_{ver}.json")
    old = json.load(open(res)) if os.path.exists(res) else []
    json.dump(old + rows, open(res, "w"), indent=1)


if __name__ == "__main__":
    main()
