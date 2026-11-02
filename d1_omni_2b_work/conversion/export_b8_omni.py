"""Export the d1-Omni-2B batch decide graph B = 8 of the L128 bucket as ONE single-signature file: the d1-2B
batch export (D1DecideB of d1_graph_b.py, causal_runtime=True) with the Omni weights (d1_graph.py's default snapshot
is this work directory's snapshot/), export_decide.fp16fc (FULLY_CONNECTED weights float16 by float_casting,
channelwise; activations float32), PYTHONHASHSEED=0.

Import order matters: this folder goes first on sys.path and d1_graph is imported by name BEFORE d1_graph_b is loaded
by path (d1_graph_b imports `d1_graph` by name: it must get the module already in sys.modules); asserted below.

Measured into the record: torch rows of the batch module vs the batch-1 D1Decide (4 rows), the float32 file on the
CPU vs the module, bytes, sha256, the FC weight buffers (export_multi.buffers), the scan, the largest runtime buffers
(Pixel 11 Pro: 1 GiB per buffer), the operand-rank dump of the subgraph (rank_dump8.py: RISK pairs and constant
mixed-rank ops), the op sequence against the d1-2B B = 8 file when it is in out/graphs/ (the same graph code = the same
ops and shapes is the expectation), peak RSS.

  PYTHONHASHSEED=0 $EXPORT conversion/export_b8_omni.py [--keep-fp32]

Writes out/graphs/d1-omni-2b_decide_b8_L128_fp16fc.tflite (+ the float32 file, deleted unless --keep-fp32) and
results/export_b8_L128_omni.json. Never overwrites.
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
R4B8 = os.path.join(WORK, "out/graphs/d1-2b_decide_b8_L128_fp16fc.tflite")
B, L = 8, 128


def _common():
    spec = importlib.util.spec_from_file_location("omni_common", os.path.join(HERE, "omni_common.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["omni_common"] = mod
    spec.loader.exec_module(mod)
    return mod


C = _common()
sha256_file, load_by_path = C.sha256_file, C.load_by_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-fp32", action="store_true")
    a = ap.parse_args()
    assert os.environ.get("PYTHONHASHSEED") == "0", "run with PYTHONHASHSEED=0"
    import numpy as np
    import torch
    import litert_torch

    import d1_graph as G                                   # the Omni snapshot as default
    from export_decide import fp16fc
    from export_multi import buffers
    from tflite_scan_d1 import scan
    rd = load_by_path("rank_dump8", os.path.join(HERE, "rank_dump8.py"))
    GB = load_by_path("d1_graph_b", os.path.join(HERE, "d1_graph_b.py"))
    from d1_layout_b import make_sees_batch
    assert GB.G is G, "d1_graph_b must use this d1_graph module"
    assert os.path.samefile(G.__file__, os.path.join(R3X, "d1_graph.py")), G.__file__
    assert os.path.samefile(G.SNAP_DEFAULT, SNAP), G.SNAP_DEFAULT

    sig = f"decide_b{B}_L{L}"
    gdir = os.path.join(WORK, "out/graphs")
    os.makedirs(gdir, exist_ok=True)
    os.makedirs(os.path.join(WORK, "results"), exist_ok=True)
    p32 = os.path.join(gdir, f"d1-omni-2b_decide_b{B}_L{L}_fp32.tflite")
    p16 = os.path.join(gdir, f"d1-omni-2b_decide_b{B}_L{L}_fp16fc.tflite")
    rec_path = os.path.join(WORK, f"results/export_b{B}_L{L}_omni.json")
    for p in (p32, p16, rec_path):
        assert not os.path.exists(p), f"never overwrite {p}"
    torch.set_num_threads(8)
    t0 = time.time()
    cfg = G.Cfg(SNAP)
    W = G.load_text_weights(SNAP, dtype=torch.float32)
    W.pop("embed_tokens.weight", None)
    model = GB.build(B, L, W, cfg, causal_runtime=True)
    rows = C.sample_rows(B, L, cfg.hidden)                  # the batch export's sample (seed 0)
    emb = np.zeros((B, L, cfg.hidden), np.float32)
    for b, (T, st, tags, e) in enumerate(rows):
        emb[b, :T] = e
    q, k = make_sees_batch([(st, tags) for T, st, tags, e in rows], B, L)
    sample = {"embeds": torch.from_numpy(emb), "sees_q": torch.from_numpy(q), "sees_k": torch.from_numpy(k)}
    with torch.inference_mode():
        ref = model(**sample)["hidden"].numpy()
        single = G.D1Decide(W, cfg, L).eval().requires_grad_(False)
        rows_vs_single = []
        for b in (0, 1, B // 2, B - 1):
            T, st, tags, e = rows[b]
            q1, k1 = G.make_sees(st, tags, L)
            e1 = torch.zeros(1, L, cfg.hidden)
            e1[0, :T] = torch.from_numpy(e)
            h1 = single(embeds=e1, sees_q=torch.from_numpy(q1), sees_k=torch.from_numpy(k1))["hidden"][0, :T].numpy()
            rows_vs_single.append({"row": b, "T": T, "max_abs": float(np.abs(ref[b, :T] - h1).max())})
        del single
    rec = {"model": "LiquidAI/d1-Omni-2B@431f1d667de5afcc9b0c58b2d9c170a7b38b358b", "B": B, "L": L, "signature": sig,
           "graph": "d1_graph_b.D1DecideB over d1_graph", "causal_runtime": True,
           "snapshot": os.path.relpath(SNAP, WORK),
           "safetensors_bytes": os.path.getsize(os.path.join(SNAP, "model.safetensors")),
           "modules": {"d1_graph": os.path.relpath(G.__file__, WORK), "d1_graph_sha256": sha256_file(G.__file__),
                       "d1_graph_b": os.path.relpath(GB.__file__, WORK), "d1_graph_b_sha256": sha256_file(GB.__file__),
                       "export_decide_sha256": sha256_file(os.path.join(R3X, "export_decide.py")),
                       "rank_dump8_sha256": sha256_file(rd.__file__), "omni_common_sha256": sha256_file(C.__file__),
                       "this_sha256": sha256_file(__file__)},
           "inputs": {n: list(v.shape) for n, v in sample.items()}, "rows_T": [r[0] for r in rows],
           "torch_rows_vs_single_module": rows_vs_single, "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
           "versions": {p: importlib.metadata.version(p) for p in
                        ("torch", "litert-torch", "litert-converter", "ai-edge-litert", "ai-edge-quantizer", "numpy")},
           "load_build_s": round(time.time() - t0, 1)}
    print(json.dumps({"rows_vs_single": rows_vs_single}), flush=True)
    t1 = time.time()
    lrt = litert_torch.signature(sig, model, sample_kwargs=sample).convert()
    rec["convert_s"] = round(time.time() - t1, 1)
    t2 = time.time()
    lrt.export(p32)
    rec["write_s"] = round(time.time() - t2, 1)
    del lrt
    rec["fp32"] = {"file": os.path.relpath(p32, WORK), "bytes": os.path.getsize(p32), "sha256": sha256_file(p32)}
    from ai_edge_litert.interpreter import Interpreter

    t3 = time.time()
    it = Interpreter(model_path=p32, num_threads=8)
    got = it.get_signature_runner(sig)(**{n: v.numpy() for n, v in sample.items()})["hidden"]
    rec["fp32_cpu_vs_module_real_max_abs"] = max(float(np.abs(got[b, :T] - ref[b, :T]).max())
                                                 for b, (T, *_) in enumerate(rows))
    rec["fp32_cpu_s"] = round(time.time() - t3, 1)
    del it, model, W
    t4 = time.time()
    rec["recipe"] = fp16fc(p32, p16)
    rec["quant_s"] = round(time.time() - t4, 1)
    rec["fp16fc"] = {"file": os.path.relpath(p16, WORK), "bytes": os.path.getsize(p16), "sha256": sha256_file(p16)}
    rec["buffers_fp16fc"] = buffers(p16)
    s32, s16 = scan(p32, with_sha=False), scan(p16, with_sha=False)
    keys = ("operator_count", "op_histogram", "custom_ops", "rank_gt4_tensor_count", "int64_tensor_count",
            "broadcast_to_count", "fc_weight_dtypes")
    rec["scan_fp32"] = {k_: s32[k_] for k_ in keys}
    rec["scan_fp16fc"] = {k_: s16[k_] for k_ in keys}
    seen, top = set(), []
    for t in s16["largest_activations"]:
        if tuple(t["shape"]) in seen:
            continue
        seen.add(tuple(t["shape"]))
        top.append({"shape": t["shape"], "MiB": round(t["bytes"] / 2 ** 20, 1), "name": t["name"][:90]})
        if len(top) == 5:
            break
    rec["largest_runtime_buffers"] = top
    rec["largest_runtime_buffer_under_1GiB"] = all(t["MiB"] < 1024 for t in top)
    rdump = rd.dump(p16, quiet=True)
    rec["rank_dump_fp16fc"] = rdump
    rec["risk_pairs"] = sum(s["risk_pairs"] for s in rdump)
    rec["const_mixed_rank_ops"] = sum(s["const_mixed_rank_ops"] for s in rdump)
    mine = C.op_sequences(p16)
    rec["op_sequences"] = mine
    if os.path.exists(R4B8):
        ob = C.op_sequences(R4B8)
        rec["op_sequence_vs_d1-2b_b8"] = {"ops": [mine[sig]["ops"], ob.get(sig, {}).get("ops")],
                                             "same_sequence": mine[sig]["seq_sha256"] == ob.get(sig, {}).get("seq_sha256")}
    if not a.keep_fp32:
        os.remove(p32)
        rec["fp32"]["deleted"] = True
    rec["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rec["wall_s"] = round(time.time() - t0, 1)
    json.dump(rec, open(rec_path, "w"), indent=1)
    bf = rec["buffers_fp16fc"]
    print(json.dumps({k_: rec[k_] for k_ in ("signature", "fp32", "fp16fc", "fp32_cpu_vs_module_real_max_abs",
                                             "torch_rows_vs_single_module", "largest_runtime_buffers", "risk_pairs",
                                             "const_mixed_rank_ops", "wall_s")}
                     | {"fc_weight_buffers_distinct": bf["fc_weight_buffers_distinct"],
                        "op_sequence_vs_d1-2b_b8": rec.get("op_sequence_vs_d1-2b_b8")}))
    print("peak_rss_gib", round(rec["peak_rss_bytes"] / 2 ** 30, 2))


if __name__ == "__main__":
    main()
