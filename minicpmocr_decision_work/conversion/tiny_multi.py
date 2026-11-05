#!/usr/bin/env python3
"""A tiny rehearsal of the multi-signature files before the heavy exports: random tiny decide and tower, three buckets each (a query-block one included), the single files and the one
multi file, fp16 FC; how the FC weights are stored, and the instrument of conversion/multi_check.py (bit-equal against the single files, the same option on both sides) on Metal fp32 +
constant tensor sharing and CPU + weight cache. No checkpoint read. About a minute.

    PYTHONHASHSEED=0 $EXPORT conversion/tiny_multi.py -> results/tiny_multi.json, out/tiny/m_*.tflite"""
import json
import shutil
import sys
import os
from pathlib import Path

import numpy as np

HERE = Path(os.path.abspath(__file__)).parent
WORK = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(WORK / "conversion"))
sys.path.insert(0, str(WORK / "host"))
import fc_buffers  # noqa: E402
import multi_check as MC  # noqa: E402
import tiny_check as TC  # noqa: E402

DECIDE = ((64, 0), (256, 0), (512, 128))
TOWER = ((64, 0), (256, 0), (1024, 256))
TMP = WORK / "out/tiny"


def run_decide(res):
    import litert_torch
    import torch

    import decide_graph as G

    torch.set_num_threads(4)
    model, stock, cfg = TC.build_decide()
    res["form_apply"] = G.apply_form(model)

    class SigRow(G.DecideRow):
        def forward(self, embeds, valid, cos, sin):
            G.set_attention(self.text_model, G.ATTN_QB if self.qblock and self.qblock < self.L else G.ATTN)
            return super().forward(embeds, valid, cos, sin)

    singles, feeds = {}, {}
    conv = None
    for L, qb in DECIDE:
        sig = "decide_" + G.tag(L, qb)
        feed_a, _ = TC.decide_feed(cfg, L, TC.decide_T(L)[0])
        feed_b, _ = TC.decide_feed(cfg, L, TC.decide_T(L)[1])
        feeds[sig] = [("T%d" % TC.decide_T(L)[0], TC.decide_T(L)[0], feed_a), ("T%d" % TC.decide_T(L)[1], TC.decide_T(L)[1], feed_b)]
        sample = {k: torch.from_numpy(x) for k, x in feed_a.items()}
        f32, f16 = TMP / f"m_{sig}_fp32.tflite", TMP / f"m_{sig}_fp16fc.tflite"
        gr = G.graph(model, L, qb)
        litert_torch.signature("decide", gr, sample_kwargs=sample).convert().export(str(f32))
        TC.fp16fc(f32, f16)
        singles[sig] = (f16, "decide")
        m = SigRow(model, L, qb).eval().requires_grad_(False)
        conv = litert_torch.signature(sig, m, sample_kwargs=sample) if conv is None else conv.signature(sig, m, sample_kwargs=sample)
    m32, m16 = TMP / "m_multi_decide_fp32.tflite", TMP / "m_multi_decide_fp16fc.tflite"
    conv.convert().export(str(m32))
    TC.fp16fc(m32, m16)
    return m16, singles, feeds, "hidden"


def run_tower(res):
    import litert_torch
    import torch

    import tower_graph as T
    from export_multi_tower import share_weights

    torch.set_num_threads(4)
    hidden, heads = 64, 4
    d = hidden // heads
    sd = T.tiny_state(hidden=hidden, heads=heads)
    g = torch.Generator().manual_seed(2)
    singles, feeds = {}, {}
    conv = first = None
    for N, qb in TOWER:
        sig = "tower_" + T.tag(N, qb)
        rows = []
        for n in (N, max(4, N // 2 - (N // 2) % 4)):
            pv = torch.randn(n, T.PATCH_DIM, generator=g).numpy()
            pos = (torch.randn(n, hidden, generator=g) * 0.1).numpy()
            ang = (torch.rand(n, d, generator=g) * 6.0).numpy()
            rows.append((f"n{n}", n // 4, T.pad_inputs(pv, pos, np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32), N)))
        feeds[sig] = rows
        sample = {k: torch.from_numpy(x) for k, x in rows[0][2].items()}
        gr = T.TowerRow(sd, N, heads=heads, qblock=qb).eval().requires_grad_(False)
        f32, f16 = TMP / f"m_{sig}_fp32.tflite", TMP / f"m_{sig}_fp16fc.tflite"
        litert_torch.signature("tower", gr, sample_kwargs=sample).convert().export(str(f32))
        TC.fp16fc(f32, f16)
        singles[sig] = (f16, "tower")
        m = T.TowerRow(sd, N, heads=heads, qblock=qb).eval().requires_grad_(False)
        if first is None:
            first = m
        else:
            share_weights(first, m)
        conv = litert_torch.signature(sig, m, sample_kwargs=sample) if conv is None else conv.signature(sig, m, sample_kwargs=sample)
    m32, m16 = TMP / "m_multi_tower_fp32.tflite", TMP / "m_multi_tower_fp16fc.tflite"
    conv.convert().export(str(m32))
    TC.fp16fc(m32, m16)
    return m16, singles, feeds, "rows"


def main():
    TMP.mkdir(parents=True, exist_ok=True)
    res = {}
    for what, fn in (("decide", run_decide), ("tower", run_tower)):
        doc = res[what] = {}
        m16, singles, feeds, out_name = fn(doc)
        b = fc_buffers.buffers(str(m16))
        doc["buffers"] = {k: b[k] for k in ("subgraphs", "signatures", "fc_weight_buffers_distinct", "fc_weight_bytes_distinct", "fc_weight_buffers_read_by_every_subgraph", "file_bytes")}
        doc["single_bytes"] = {k: Path(v[0]).stat().st_size for k, v in singles.items()}
        doc["single_buffers"] = {k: fc_buffers.buffers(str(v[0]))["fc_weight_buffers_distinct"] for k, v in singles.items()}
        print(what, json.dumps(doc["buffers"]), doc["single_buffers"], flush=True)
        for acc, sharing, cache in (("gpu_f32", True, False), ("cpu", False, True), ("gpu_f32", False, False)):
            cdir = ""
            if cache:
                cdir = str(TMP / "m_wcache")
                shutil.rmtree(cdir, ignore_errors=True)
                Path(cdir).mkdir(parents=True)
            r = MC.compare(m16, singles, feeds, out_name, acc, 4, sharing, cdir)
            doc[f"{acc}{'_sharing' if sharing else ''}{'_wcache' if cache else ''}"] = r
            print(what, acc, sharing, cache, r["summary"], {k: v["again_in_reverse"]["bit_equal"] for k, v in r["signatures"].items()}, flush=True)
            if cdir:
                shutil.rmtree(cdir, ignore_errors=True)
    (WORK / "results/tiny_multi.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    print("TINY_MULTI_DONE", flush=True)


if __name__ == "__main__":
    main()
