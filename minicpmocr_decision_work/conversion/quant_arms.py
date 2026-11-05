#!/usr/bin/env python3
"""The integer storage arms, from a float32 file. The decision graph's by default; --what tower for the tower's (its float32 files come from
export_tower.py --variant xb --keep-fp32: the plain module).

    $EXPORT conversion/quant_arms.py [--what tower] --tag L256 --form dyni8fc|wi8fc|i4dyn32

  dyni8fc  dynamic int8 on every FULLY_CONNECTED (the FC reads an INT8 constant; activations float).
  wi8fc    weight-only int8 CHANNELWISE on every FULLY_CONNECTED (DEQUANTIZE -> FC). Metal 2.3.0 refused this encoding.
  i4dyn32  dynamic int4 BLOCKWISE_32 on every FULLY_CONNECTED, with the clamp-before-quantize floor at 2^-14 and the scale scan.
Writes out/graphs/<what>_<tag>_<form>.tflite, results/quant_<what>_<tag>_<form>.json. Never overwrites."""
from __future__ import annotations

import argparse
import json
import mmap
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(os.path.abspath(__file__)).parent
WORK = HERE.parent        # the work directory: snapshot/, out/, results/
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "host"))
import fp16fc as Q  # noqa: E402
import lrt  # noqa: E402
from export_decide import sha256_file  # noqa: E402

FLOOR = np.float32(2.0 ** -14)
STATS = {"tensors": 0, "scales": 0}


def install_floor() -> None:
    from ai_edge_quantizer.algorithms.uniform_quantize import uniform_quantize_tensor as u

    orig = u.tensor_zp_scale_from_min_max

    def clamped(min_value, max_value, num_bits, symmetric, granularity, *rest, **kw):
        zp, scale = orig(min_value, max_value, num_bits, symmetric, granularity, *rest, **kw)
        if u.is_blockwise(granularity) and num_bits == 4:
            if not symmetric:
                raise ValueError("asymmetric blockwise int4 is not covered by this floor")
            low = scale < FLOOR
            n = int(np.count_nonzero(low))
            if n:
                scale = np.where(low, FLOOR, scale).astype(scale.dtype)
                STATS["tensors"] += 1
                STATS["scales"] += n
        return zp, scale

    u.tensor_zp_scale_from_min_max = clamped


def scale_scan(path) -> dict:
    """Every blockwise tensor's fp16 scales: zero, subnormal (0 < s < 2^-14), at exactly 2^-14, the minimum. XNNPACK's check is std::isnormal."""
    from ai_edge_litert import schema_py_generated as schema

    tiny = np.float16(2.0 ** -14)
    f = open(path, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    fb = schema.Model.GetRootAs(mm, 0)
    seen = set()
    n_t = n_zero = n_sub = n_tiny = n_scales = 0
    gmin = None
    for si in range(fb.SubgraphsLength()):
        sg = fb.Subgraphs(si)
        for i in range(sg.TensorsLength()):
            t = sg.Tensors(i)
            q = t.Quantization()
            if q is None or q.DetailsType() != schema.QuantizationDetails.BlockwiseQuantization:
                continue
            bq = schema.BlockwiseQuantization()
            tab = q.Details()
            bq.Init(tab.Bytes, tab.Pos)
            bi = sg.Tensors(bq.Scales()).Buffer()
            if bi in seen:
                continue
            seen.add(bi)
            buf = fb.Buffers(bi)
            if buf.DataLength() > 0:
                v = np.frombuffer(buf.DataAsNumpy().tobytes(), dtype=np.float16)
            else:
                v = np.frombuffer(mm, dtype=np.float16, count=buf.Size() // 2, offset=buf.Offset())
            n_t += 1
            n_scales += v.size
            n_zero += int((v == 0).sum())
            n_sub += int(((v > 0) & (v < tiny)).sum())
            n_tiny += int((v == tiny).sum())
            m = float(v.min())
            gmin = m if gmin is None else min(gmin, m)
            del v
    del fb
    return {"blockwise_tensors": n_t, "scales": n_scales, "zero": n_zero, "subnormal": n_sub, "at_floor": n_tiny, "min": gmin}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="L256, L4096_qb1024, ...")
    ap.add_argument("--form", required=True)
    ap.add_argument("--what", default="decide", choices=("decide", "tower"))
    a = ap.parse_args()
    src = WORK / f"out/graphs/{a.what}_{a.tag}_fp32.tflite"
    if a.what == "tower" and not src.exists():
        src = WORK / f"out/graphs/tower_{a.tag}_xb32.tflite"
    dst = WORK / f"out/graphs/{a.what}_{a.tag}_{a.form}.tflite"
    rec_path = WORK / f"results/quant_{a.what}_{a.tag}_{a.form}.json"
    for p in (dst, rec_path):
        assert not p.exists(), f"never overwrite {p}"
    assert src.exists(), src
    if a.form == "i4dyn32":
        install_floor()
    t0 = time.time()
    recipe = Q.quantize(src, dst, a.form)
    rec = {"tag": a.tag, "form": a.form, "source": src.name, "source_bytes": src.stat().st_size, "recipe": recipe, "quant_s": round(time.time() - t0, 1),
           "file": str(dst.relative_to(WORK)), "bytes": dst.stat().st_size, "sha256": sha256_file(dst), "scan": Q.fc_scan(dst)}
    if a.form == "i4dyn32":
        rec["clamp"] = dict(STATS)
        rec["scale_scan"] = scale_scan(dst)
    rec["memory_end"] = lrt.memory()
    rec["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rec["wall_s"] = round(time.time() - t0, 1)
    rec_path.write_text(json.dumps(rec, indent=1, default=str) + "\n")
    print("QUANT_DONE " + json.dumps({k: rec.get(k) for k in ("tag", "form", "bytes", "quant_s", "clamp", "scale_scan")} | {"fc_src": rec["scan"]["fc_weight_source_dtype"], "fc_direct": rec["scan"]["fc_weight_tensor_dtype_direct"],
                                                                                        "footprint_gib": round(rec["memory_end"].get("lifetime_max_phys_footprint", 0) / 2**30, 2)}), flush=True)


if __name__ == "__main__":
    main()
