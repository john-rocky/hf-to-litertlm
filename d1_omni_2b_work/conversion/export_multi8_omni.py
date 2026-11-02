"""Export ONE .tflite holding the seven length buckets of the d1-Omni-2B decision graph AND the batch graph B = 8 of
the L128 bucket as eight signatures over one set of weights ("multi8"): decide_L128, decide_L256, decide_L512,
decide_L1024, decide_L2048, decide_L4096_qb1024 and decide_L6144_qb1024 (each d1_graph.D1Decide with export_multi.py's
sample) plus D1DecideB (d1_graph_b.py, B = 8, L = 128, causal_runtime=True, export_batch.py's sample) as the eighth
signature `decide_b8_L128`. The eight modules are built over ONE weight dict (d1_graph.linear keeps the tensor itself,
so every FC weight of every signature is the same torch storage and litert-torch's constant cache lowers it once), one
convert, then export_decide.fp16fc (FULLY_CONNECTED weights float16 by float_casting, channelwise; activations
float32).

Measured into the record: the FC weight buffers each subgraph reads and how many are distinct (export_multi.buffers:
one copy = every FC weight buffer read by all eight subgraphs), bytes, sha256, signatures and their inputs, the
operand-rank dump of every subgraph (rank_dump8.py: runtime rank 3 + constant rank 4 pairs = 0 wanted, and the wider
constant mixed-rank class), the largest runtime buffers (the phone's 1 GiB cap), each subgraph's op sequence against
the seven-signature file (export_multi.py) and the B = 8 file (export_b8_omni.py) when they are in out/graphs/ (same ops and
shapes = the same graph), and the process's peak RSS.

  PYTHONHASHSEED=0 $EXPORT conversion/export_multi8_omni.py [--keep-fp32]

Writes out/graphs/d1-omni-2b_decide_multi8_fp16fc.tflite (+ the float32 file, deleted after the fp16 one unless
--keep-fp32) and results/export_decide_multi8_omni.json. Never overwrites a finished output; a float32 file is reused only when its `.done`
mark exists (a run stopped mid-write leaves none).

The graph code is the d1-2B code with the d1-Omni-2B snapshot (d1_graph.py's default snapshot is this work directory's
snapshot/); the op sequences are also compared with the d1-2B files of the same graph code when those are in
out/graphs/ (an instrument). Helpers from omni_common.py.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import resource
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.normpath(os.path.join(HERE, ".."))
R3X = HERE
sys.path.insert(0, R3X)
SNAP = os.path.join(WORK, "snapshot")
B8, L8 = 8, 128
MULTI7 = os.path.join(WORK, "out/graphs/d1-omni-2b_decide_multi7_fp16fc.tflite")
B8FILE = os.path.join(WORK, "out/graphs/d1-omni-2b_decide_b8_L128_fp16fc.tflite")
MULTI7_D12B = os.path.join(WORK, "out/graphs/d1-2b_decide_multi7_fp16fc.tflite")
B8FILE_D12B = os.path.join(WORK, "out/graphs/d1-2b_decide_b8_L128_fp16fc.tflite")


def _common():
    spec = importlib.util.spec_from_file_location("omni_common", os.path.join(HERE, "omni_common.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["omni_common"] = mod
    spec.loader.exec_module(mod)
    return mod


C = _common()
sha256_file, load_by_path, sample_rows, op_sequences = C.sha256_file, C.load_by_path, C.sample_rows, C.op_sequences


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-fp32", action="store_true")
    a = ap.parse_args()
    assert os.environ.get("PYTHONHASHSEED") == "0", "run with PYTHONHASHSEED=0 (the other exports did)"
    import numpy as np
    import torch

    # this folder's modules by name first (on sys.path[0]), then the batch module by path: d1_graph_b imports
    # `d1_graph` by name and must get the same module (asserted below)
    import d1_graph as G
    from export_decide import fp16fc
    from export_multi import BUCKETS, buffers, sig_name
    from tflite_scan_d1 import scan
    rd = load_by_path("rank_dump8", os.path.join(HERE, "rank_dump8.py"))
    GB = load_by_path("d1_graph_b", os.path.join(HERE, "d1_graph_b.py"))
    from d1_layout_b import make_sees_batch
    assert GB.G is G, "d1_graph_b must use this d1_graph module"
    assert os.path.samefile(G.__file__, os.path.join(R3X, "d1_graph.py")), G.__file__
    assert os.path.samefile(G.SNAP_DEFAULT, SNAP), G.SNAP_DEFAULT

    gdir = os.path.join(WORK, "out/graphs")
    os.makedirs(gdir, exist_ok=True)
    os.makedirs(os.path.join(WORK, "results"), exist_ok=True)
    p32 = os.path.join(gdir, "d1-omni-2b_decide_multi8_fp32.tflite")
    p16 = os.path.join(gdir, "d1-omni-2b_decide_multi8_fp16fc.tflite")
    rec_path = os.path.join(WORK, "results/export_decide_multi8_omni.json")
    for p in (p16, rec_path):
        assert not os.path.exists(p), f"never overwrite {p}"
    torch.set_num_threads(8)
    t0 = time.time()
    cfg = G.Cfg(SNAP)
    W = G.load_text_weights(SNAP, dtype=torch.float32)
    W.pop("embed_tokens.weight", None)      # the graph takes embeddings; the table stays on the host
    import litert_torch

    conv, sigs, mods = None, [], {}
    for L, qb in BUCKETS:                   # the seven buckets and export_multi.py's sample
        model = G.D1Decide(W, cfg, L, qb or None, torch.float32).eval().requires_grad_(False)
        T = L // 2 + 1
        tags = [-1] * (T // 2) + [1] * (T - T // 2 - 3) + [2] * 3
        q, k = G.make_sees(T // 2, tags, L)
        g = torch.Generator().manual_seed(0)
        emb = torch.zeros(1, L, cfg.hidden)
        emb[0, :T] = torch.randn(T, cfg.hidden, generator=g) * 0.02
        sample = {"embeds": emb, "sees_q": torch.from_numpy(q), "sees_k": torch.from_numpy(k)}
        name = sig_name(L, qb)
        conv = (litert_torch.signature(name, model, sample_kwargs=sample) if conv is None
                else conv.signature(name, model, sample_kwargs=sample))
        sigs.append(name)
        mods[name] = model
    mb = GB.build(B8, L8, W, cfg, causal_runtime=True)   # the form of the shipped B = 8 file
    rows = sample_rows(B8, L8, cfg.hidden)
    emb = np.zeros((B8, L8, cfg.hidden), np.float32)
    for b, (T, st, tags, e) in enumerate(rows):
        emb[b, :T] = e
    q, k = make_sees_batch([(st, tags) for T, st, tags, e in rows], B8, L8)
    sb = f"decide_b{B8}_L{L8}"
    conv = conv.signature(sb, mb, sample_kwargs={"embeds": torch.from_numpy(emb), "sees_q": torch.from_numpy(q),
                                                  "sees_k": torch.from_numpy(k)})
    sigs.append(sb)
    # the weights are shared in torch: every nn.Linear weight of the batch module is the L128 module's storage
    lin = lambda m: [x.weight for x in m.modules() if isinstance(x, torch.nn.Linear)]
    w128, wb = lin(mods["decide_L128"]), lin(mb)
    shared = sum(1 for x, y in zip(w128, wb) if x.data_ptr() == y.data_ptr() and x.shape == y.shape)
    rec = {"model": "LiquidAI/d1-Omni-2B@431f1d667de5afcc9b0c58b2d9c170a7b38b358b", "signatures": sigs,
           "graph": "d1_graph.D1Decide x 7 (the d1-2B code, Omni snapshot) + d1_graph_b.D1DecideB B=8 L=128 "
           "causal_runtime, one weight dict", "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
           "snapshot": os.path.relpath(SNAP, WORK),
           "modules": {"d1_graph": os.path.relpath(G.__file__, WORK), "d1_graph_sha256": sha256_file(G.__file__),
                       "d1_graph_b": os.path.relpath(GB.__file__, WORK), "d1_graph_b_sha256": sha256_file(GB.__file__),
                       "export_multi_sha256": sha256_file(os.path.join(R3X, "export_multi.py")),
                       "export_decide_sha256": sha256_file(os.path.join(R3X, "export_decide.py")),
                       "rank_dump8_sha256": sha256_file(rd.__file__), "omni_common_sha256": sha256_file(C.__file__),
                       "this_sha256": sha256_file(__file__)},
           "torch_linear_weights": {"decide_L128": len(w128), sb: len(wb), "same_storage": shared},
           "b8_sample_rows_T": [r[0] for r in rows],
           "versions": {p: importlib.metadata.version(p) for p in
                        ("torch", "litert-torch", "litert-converter", "ai-edge-litert", "ai-edge-quantizer", "numpy")},
           "load_build_s": round(time.time() - t0, 1)}
    print(json.dumps({"signatures": sigs, "torch_linear_weights": rec["torch_linear_weights"]}), flush=True)
    t1 = time.time()
    done32 = p32 + ".done"
    if not (os.path.exists(p32) and os.path.exists(done32)):
        lrt = conv.convert()
        rec["convert_s"] = round(time.time() - t1, 1)
        t2 = time.time()
        lrt.export(p32)
        open(done32, "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
        rec["write_s"] = round(time.time() - t2, 1)
        del lrt
    else:
        rec["fp32_reused"] = True
    del conv, W, mods, mb, model
    rec["fp32"] = {"file": os.path.relpath(p32, WORK), "bytes": os.path.getsize(p32), "sha256": sha256_file(p32)}
    rec["buffers_fp32"] = buffers(p32)
    print(json.dumps({"fp32_bytes": rec["fp32"]["bytes"], **{k_: rec["buffers_fp32"][k_] for k_ in (
        "subgraphs", "fc_weight_buffers_distinct", "fc_weight_bytes_distinct",
        "fc_weight_buffers_read_by_every_subgraph")}}), flush=True)
    t4 = time.time()
    rec["recipe"] = fp16fc(p32, p16)
    rec["quant_s"] = round(time.time() - t4, 1)
    rec["fp16fc"] = {"file": os.path.relpath(p16, WORK), "bytes": os.path.getsize(p16), "sha256": sha256_file(p16)}
    rec["buffers_fp16fc"] = buffers(p16)
    s16 = scan(p16, with_sha=False)
    rec["scan_fp16fc"] = {k_: s16[k_] for k_ in ("subgraphs", "signatures", "operator_count", "op_histogram",
                                                 "custom_ops", "rank_gt4_tensor_count", "int64_tensor_count",
                                                 "broadcast_to_count", "fc_weight_dtypes", "dequantize_outputs")}
    seen, top = set(), []
    for t in s16["largest_activations"]:
        if tuple(t["shape"]) in seen:
            continue
        seen.add(tuple(t["shape"]))
        top.append({"shape": t["shape"], "MiB": round(t["bytes"] / 2 ** 20, 1), "name": t["name"][:90]})
        if len(top) == 6:
            break
    rec["largest_runtime_buffers"] = top
    rec["largest_runtime_buffer_under_1GiB"] = all(t["MiB"] < 1024 for t in top)
    rdump = rd.dump(p16, quiet=True)
    rec["rank_dump_fp16fc"] = rdump
    rec["risk_pairs_per_signature"] = {s["signature"]: s["risk_pairs"] for s in rdump}
    rec["const_mixed_rank_ops_per_signature"] = {s["signature"]: s["const_mixed_rank_ops"] for s in rdump}
    rec["risk_pairs"] = sum(s["risk_pairs"] for s in rdump)
    # the same graph as the comparators? op sequences with their shapes, per signature
    mine = op_sequences(p16)
    rec["op_sequences"] = mine
    cmp, cmp_d12b = {}, {}
    if os.path.exists(MULTI7):
        o7 = op_sequences(MULTI7)
        for s in sigs[:-1]:
            cmp[s] = {"vs": "omni multi7", "ops": [mine[s]["ops"], o7.get(s, {}).get("ops")],
                      "same_sequence": mine[s]["seq_sha256"] == o7.get(s, {}).get("seq_sha256")}
    if os.path.exists(B8FILE):
        ob = op_sequences(B8FILE)
        cmp[sb] = {"vs": "omni b8 single file", "ops": [mine[sb]["ops"], ob.get(sb, {}).get("ops")],
                   "same_sequence": mine[sb]["seq_sha256"] == ob.get(sb, {}).get("seq_sha256")}
    if os.path.exists(MULTI7_D12B):                      # instrument: the d1-2B files of the same graph code
        o7 = op_sequences(MULTI7_D12B)
        for s in sigs[:-1]:
            cmp_d12b[s] = {"vs": "d1-2b multi7", "same_sequence": mine[s]["seq_sha256"] == o7.get(s, {}).get("seq_sha256")}
    if os.path.exists(B8FILE_D12B):
        ob = op_sequences(B8FILE_D12B)
        cmp_d12b[sb] = {"vs": "d1-2b b8 file", "same_sequence": mine[sb]["seq_sha256"] == ob.get(sb, {}).get("seq_sha256")}
    rec["op_sequence_compare"] = cmp
    rec["op_sequence_compare_d1-2b_instrument"] = cmp_d12b
    if not a.keep_fp32:
        os.remove(p32)
        os.remove(done32)
        rec["fp32"]["deleted"] = True
    rec["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rec["wall_s"] = round(time.time() - t0, 1)
    json.dump(rec, open(rec_path, "w"), indent=1)
    b = rec["buffers_fp16fc"]
    print(json.dumps({"fp16fc": rec["fp16fc"], "signatures": b["signatures"],
                      "fc_weight_buffers_distinct": b["fc_weight_buffers_distinct"],
                      "fc_weight_bytes_distinct": b["fc_weight_bytes_distinct"],
                      "read_by_every_subgraph": b["fc_weight_buffers_read_by_every_subgraph"],
                      "read_by_one_subgraph": b["fc_weight_buffers_read_by_one_subgraph"],
                      "per_subgraph": [(s["name"], s["fc_weight_buffers"]) for s in b["per_subgraph"]],
                      "risk_pairs": rec["risk_pairs"], "const_mixed": rec["const_mixed_rank_ops_per_signature"],
                      "largest": top[:3], "op_sequence_compare": cmp,
                      "peak_rss_gib": round(rec["peak_rss_bytes"] / 2 ** 30, 1), "wall_s": rec["wall_s"]}))


if __name__ == "__main__":
    main()
