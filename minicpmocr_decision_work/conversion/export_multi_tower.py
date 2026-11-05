#!/usr/bin/env python3
"""The tower's buckets up to 4,096 patches as signatures of ONE .tflite over one set of weights, then the fp16 FC cast, then the count of how the weights are stored (`buffers`). Every
bucket's parameters are the first bucket's objects, so every signature's FC weight is the same torch storage and litert-torch's constant cache lowers it once
Same module as the single files (tower_graph.TowerRow).

    PYTHONHASHSEED=0 $EXPORT conversion/export_multi_tower.py [--keep-fp32] [--buckets 400,576,784,1024,1296,1600,1936,2304,3136,4096:1024 --name half]

Writes out/graphs/multi_tower_N4096_{fp32,fp16fc}.tflite, results/export_multi_tower_N4096.json (with --name: multi_tower_<name>_*, export_multi_tower_<name>.json). `size:qblock` in --buckets. Never overwrites."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import resource
import sys
import threading
import time
from pathlib import Path

HERE = Path(os.path.abspath(__file__)).parent
WORK = HERE.parent        # the work directory: snapshot/, out/, results/
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "host"))
import fp16fc as Q  # noqa: E402
import lrt  # noqa: E402
from export_decide import footprint_sampler, sha256_file  # noqa: E402
from export_tower import pictures  # noqa: E402
from fc_buffers import buffers  # noqa: E402

BUCKETS = ((400, 0), (576, 0), (1024, 0), (1600, 0), (2304, 0), (4096, 1024))


def share_weights(first, other) -> int:
    """`other`'s parameters become `first`'s objects (the same names: the buckets differ in the attention's blocks, which hold no parameter)."""
    params = dict(first.named_parameters())
    names = [n for n, _ in other.named_parameters()]
    assert sorted(names) == sorted(params), "the buckets do not hold the same parameters"
    for name in names:
        parent = other
        *path, leaf = name.split(".")
        for p in path:
            parent = getattr(parent, p)
        setattr(parent, leaf, params[name])
    return len(names)


def main():
    import torch

    import litert_torch
    import tower_graph as T

    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-fp32", action="store_true")
    ap.add_argument("--buckets", default="")
    ap.add_argument("--name", default="N4096")
    a = ap.parse_args()
    keep = a.keep_fp32
    buckets = BUCKETS if not a.buckets else tuple((int(x.split(":")[0]), int(x.split(":")[1]) if ":" in x else 0) for x in a.buckets.split(","))
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    gdir = WORK / "out/graphs"
    p32, p16 = gdir / f"multi_tower_{a.name}_fp32.tflite", gdir / f"multi_tower_{a.name}_fp16fc.tflite"
    rec_path = WORK / f"results/export_multi_tower_{a.name}.json"
    for p in (p16, rec_path):
        assert not p.exists(), f"never overwrite {p}"
    t0 = time.time()
    stop = threading.Event()
    watch = {}
    threading.Thread(target=footprint_sampler, args=(stop, watch), daemon=True).start()
    sd = T.load_visual_state(WORK / "snapshot")
    sd.pop("pos_embed.weight")
    rec = {"buckets": buckets, "pythonhashseed": os.environ.get("PYTHONHASHSEED"), "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "versions": {p: importlib.metadata.version(p) for p in ("torch", "litert-torch", "litert-converter", "ai-edge-litert", "ai-edge-quantizer", "numpy")},
           "scripts": {f.name: sha256_file(f)[:16] for f in sorted(HERE.glob("*.py"))}}
    conv = first = None
    sigs = []
    for N, qb in buckets:
        m = T.graph(sd, N, qb)
        if first is None:
            first = m
        else:
            share_weights(first, m)
        name, n, z = pictures(N)[0]
        sample = {k: torch.from_numpy(v) for k, v in T.pad_inputs(z["pixel_values"], z["pos"], z["cos"], z["sin"], N).items()}
        sig = f"tower_{T.tag(N, qb)}"
        conv = litert_torch.signature(sig, m, sample_kwargs=sample) if conv is None else conv.signature(sig, m, sample_kwargs=sample)
        sigs.append(sig)
    rec["signatures"] = sigs
    t1 = time.time()
    out = conv.convert()
    rec["convert_s"] = round(time.time() - t1, 1)
    t2 = time.time()
    out.export(str(p32))
    rec["write_s"] = round(time.time() - t2, 1)
    del out, conv
    rec["fp32"] = {"file": str(p32.relative_to(WORK)), "bytes": p32.stat().st_size, "sha256": sha256_file(p32)}
    rec["buffers_fp32"] = buffers(str(p32))
    print(json.dumps({"fp32_bytes": rec["fp32"]["bytes"], **{k: rec["buffers_fp32"][k] for k in ("subgraphs", "fc_weight_buffers_distinct", "fc_weight_bytes_distinct", "fc_weight_buffers_read_by_every_subgraph")}}), flush=True)
    t3 = time.time()
    Q.fp16fc(p32, p16)
    rec["quant_s"] = round(time.time() - t3, 1)
    rec["fp16fc"] = {"file": str(p16.relative_to(WORK)), "bytes": p16.stat().st_size, "sha256": sha256_file(p16)}
    rec["buffers_fp16fc"] = buffers(str(p16))
    rec["scan_fp16fc"] = {**Q.fc_scan(p16), **Q.general_scan(p16)}
    if not keep:
        p32.unlink()
        rec["fp32"]["deleted"] = True
    stop.set()
    rec.update(watch)
    rec["memory_end"] = lrt.memory()
    rec["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rec["wall_s"] = round(time.time() - t0, 1)
    rec_path.write_text(json.dumps(rec, indent=1, default=str) + "\n")
    b = rec["buffers_fp16fc"]
    print("EXPORT_DONE " + json.dumps({"fp16_bytes": rec["fp16fc"]["bytes"], "signatures": b["signatures"], "fc_weight_buffers_distinct": b["fc_weight_buffers_distinct"], "fc_weight_bytes_distinct": b["fc_weight_bytes_distinct"],
                                       "read_by_every_subgraph": b["fc_weight_buffers_read_by_every_subgraph"], "footprint_max_gib": round(rec.get("footprint_max", 0) / 2**30, 1), "wall_s": rec["wall_s"]}), flush=True)


if __name__ == "__main__":
    main()
