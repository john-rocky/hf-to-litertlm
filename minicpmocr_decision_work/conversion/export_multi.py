#!/usr/bin/env python3
"""The decision graph's phone-tier buckets as signatures of ONE .tflite over one set of weights, then the fp16 FC cast, then the count of how the
weights are stored (`buffers`). The modules share one text model, so every signature's FC weight is the same torch storage
and litert-torch's constant cache lowers it once.

    PYTHONHASHSEED=0 $EXPORT conversion/export_multi.py [--keep-fp32] [--buckets 256,384,512,768,1024,1536,2048,3072,4096:1024 --name half]

Writes out/graphs/multi_decide_phone_{fp32,fp16fc}.tflite, results/export_multi_phone.json (with --name: multi_decide_<name>_*, export_multi_<name>.json). `size:qblock` in --buckets. Never overwrites."""
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
from export_decide import footprint_sampler, padded_inputs, sample_record, sha256_file  # noqa: E402
from fc_buffers import buffers  # noqa: E402

BUCKETS = ((256, 0), (512, 0), (1024, 0), (2048, 0), (4096, 1024))


def main():
    import torch

    import decide_graph as G

    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-fp32", action="store_true")
    ap.add_argument("--buckets", default="")
    ap.add_argument("--name", default="phone")
    a = ap.parse_args()
    keep = a.keep_fp32
    buckets = BUCKETS if not a.buckets else tuple((int(x.split(":")[0]), int(x.split(":")[1]) if ":" in x else 0) for x in a.buckets.split(","))
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    gdir = WORK / "out/graphs"
    p32, p16 = gdir / f"multi_decide_{a.name}_fp32.tflite", gdir / f"multi_decide_{a.name}_fp16fc.tflite"
    rec_path = WORK / f"results/export_multi_{a.name}.json"
    for p in (p16, rec_path):
        assert not p.exists(), f"never overwrite {p}"
    t0 = time.time()
    stop = threading.Event()
    watch = {}
    threading.Thread(target=footprint_sampler, args=(stop, watch), daemon=True).start()
    model, info = G.load_text_model()
    rec = {"buckets": buckets, "load_info": info, "form_apply": G.apply_form(model), "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
           "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "litert-torch", "litert-converter", "ai-edge-litert", "ai-edge-quantizer", "numpy")}}

    class SigRow(G.DecideRow):
        """Sets the attention form of its own signature at every trace: the text model is shared."""

        def forward(self, embeds, valid, cos, sin):
            G.set_attention(self.text_model, G.ATTN_QB if self.qblock and self.qblock < self.L else G.ATTN)
            return super().forward(embeds, valid, cos, sin)

    import litert_torch

    conv = None
    sigs = []
    for L, qb in buckets:
        m = SigRow(model, L, qb).eval().requires_grad_(False)
        name, z = sample_record(L)
        sample = {k: torch.from_numpy(v) for k, v in padded_inputs(z, L).items()}
        sig = f"decide_{G.tag(L, qb)}"
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
    print(json.dumps({"fp32_bytes": rec["fp32"]["bytes"], **{k: rec["buffers_fp32"][k] for k in ("subgraphs", "fc_weight_buffers_distinct", "fc_weight_bytes_distinct", "fc_weight_buffers_read_by_every_subgraph")}, "memory": lrt.memory()}), flush=True)
    t3 = time.time()
    Q.fp16fc(p32, p16)
    rec["quant_s"] = round(time.time() - t3, 1)
    rec["fp16fc"] = {"file": str(p16.relative_to(WORK)), "bytes": p16.stat().st_size, "sha256": sha256_file(p16)}
    rec["buffers_fp16fc"] = buffers(str(p16))
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
