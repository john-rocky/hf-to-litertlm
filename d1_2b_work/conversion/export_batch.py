"""Export the batch decide graph (d1_graph_b.D1DecideB, B sequences of the L128 bucket in one call) with
litert-torch, cast its FULLY_CONNECTED weights to float16 (export_decide.fp16fc), check the float32 file on the CPU
against the module and four of its rows against the batch-1 D1Decide (the module form's own identity), scan both
files, dump the operand ranks (rank_dump: runtime rank 3 + constant rank 4 pairs = 0 wanted), list the largest runtime
buffers (DEQUANTIZE outputs excluded: the GPU takes the float16 constants as weights; Pixel 11 Pro caps one buffer at
1 GiB).

  PYTHONHASHSEED=0 $EXPORT conversion/export_batch.py --B 8 --causal-runtime [--L 128] [--keep-fp32]

--causal-runtime = the sliding layers' causal mask as the batch-1 graph's runtime rank-4 tensor (constant x the row's
first key flag): the shipped form; without it the constant [1, 1, L, L] is added to the rank-4 scores.

Writes out/graphs/d1-2b_decide_b<B>_L<L>_{fp32,fp16fc}.tflite (the float32 file deleted after the fp16 one unless
--keep-fp32) and results/export_batch_b<B>_L<L>.json. Never overwrites.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
import os
import resource
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, HERE)
SNAP = os.path.join(WORK, "snapshot")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(1 << 24):
            h.update(b)
    return h.hexdigest()


def sample_rows(B, L, hidden, seed=0):
    """B rows of different lengths (one full-length row, the rest 43..L), state halves, 1-4 questions; embeds of the
    export sample = small random values (as the batch-1 export's sample)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    rows = []
    for j in range(B):
        T = L if j == 0 else int(rng.integers(43, L + 1))
        st = int(rng.integers(5, T // 2))
        nq = int(rng.integers(1, 5))
        tags = [-1] * st + [1 + (i * nq) // (T - st) for i in range(T - st)]
        rows.append((T, st, tags, (rng.standard_normal((T, hidden)) * 0.02).astype(np.float32)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, required=True)
    ap.add_argument("--L", type=int, default=128)
    ap.add_argument("--keep-fp32", action="store_true")
    ap.add_argument("--causal-runtime", action="store_true")
    a = ap.parse_args()
    import numpy as np
    import torch
    import litert_torch

    import d1_graph as G
    import d1_graph_b as GB
    import rank_dump
    from d1_layout_b import make_sees_batch
    from export_decide import fp16fc
    from tflite_scan_d1 import scan

    assert os.environ.get("PYTHONHASHSEED") == "0"
    B, L = a.B, a.L
    sig = f"decide_b{B}_L{L}"
    gdir = os.path.join(WORK, "out/graphs")
    os.makedirs(gdir, exist_ok=True)
    p32 = os.path.join(gdir, f"d1-2b_decide_b{B}_L{L}_fp32.tflite")
    p16 = os.path.join(gdir, f"d1-2b_decide_b{B}_L{L}_fp16fc.tflite")
    rec_path = os.path.join(WORK, f"results/export_batch_b{B}_L{L}.json")
    for p in (p32, p16, rec_path):
        assert not os.path.exists(p), f"never overwrite {p}"
    torch.set_num_threads(8)
    t0 = time.time()
    cfg = G.Cfg(SNAP)
    W = G.load_text_weights(SNAP, dtype=torch.float32)
    W.pop("embed_tokens.weight", None)
    model = GB.build(B, L, W, cfg, causal_runtime=a.causal_runtime)
    rows = sample_rows(B, L, cfg.hidden)
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
    rec = {"B": B, "L": L, "signature": sig, "graph": "conversion/d1_graph_b.D1DecideB", "causal_runtime": a.causal_runtime,
           "graph_module_sha256": sha256_file(os.path.join(HERE, "d1_graph_b.py")),
           "inputs": {n: list(v.shape) for n, v in sample.items()}, "rows_T": [r[0] for r in rows],
           "torch_rows_vs_single_module": rows_vs_single, "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
           "versions": {p: importlib.metadata.version(p) for p in
                        ("torch", "litert-torch", "litert-converter", "ai-edge-litert", "ai-edge-quantizer", "numpy")},
           "load_build_s": round(time.time() - t0, 1)}
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
                                                 for b, (T, *_ ) in enumerate(rows))
    rec["fp32_cpu_s"] = round(time.time() - t3, 1)
    del it
    t4 = time.time()
    rec["recipe"] = fp16fc(p32, p16)
    rec["quant_s"] = round(time.time() - t4, 1)
    rec["fp16fc"] = {"file": os.path.relpath(p16, WORK), "bytes": os.path.getsize(p16), "sha256": sha256_file(p16)}
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
    sys.argv = ["rank_dump.py", p16]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        pats = rank_dump.main()
    rec["rank_dump_fp16fc"] = buf.getvalue().splitlines()
    rec["risk_pairs"] = sum(p["count"] for p in pats if p["risk"].startswith("RISK"))
    if not a.keep_fp32:
        os.remove(p32)
        rec["fp32"]["deleted"] = True
    rec["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rec["wall_s"] = round(time.time() - t0, 1)
    json.dump(rec, open(rec_path, "w"), indent=1)
    print(json.dumps({k_: rec[k_] for k_ in ("signature", "fp32", "fp16fc", "fp32_cpu_vs_module_real_max_abs",
                                             "torch_rows_vs_single_module", "largest_runtime_buffers", "risk_pairs",
                                             "wall_s")}))
    print("peak_rss_gb", round(rec["peak_rss_bytes"] / 2 ** 30, 1))


if __name__ == "__main__":
    main()
