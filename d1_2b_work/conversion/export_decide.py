"""Export one bucket of the d1-2B decision graph (d1_graph.D1Decide) with litert-torch, then cast its
FULLY_CONNECTED weights to float16 with ai-edge-quantizer (float_casting; activations stay float32), then scan both
files. One bucket a process, one process at a time (the export peaks near 3x the float32 weights).

  PYTHONHASHSEED=0 $EXPORT conversion/export_decide.py --L 1024 [--qblock 1024] [--keep-fp32]

Writes out/graphs/d1-2b_decide_L{L}{_qb<Q>}_fp32.tflite (deleted after the fp16 file unless --keep-fp32) and
..._fp16fc.tflite, and results/export_decide_L{L}{_qb<Q>}.json (seconds, bytes, sha256, peak RSS, versions, op and
dtype histograms, FC weight dtypes, the largest tensors). Existing outputs are never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import resource
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(1 << 24):
            h.update(b)
    return h.hexdigest()


def fp16fc(src, dst):
    """The d1-3B v2e recipe: FULLY_CONNECTED weights float16 (float_casting, channelwise), nothing else."""
    from ai_edge_quantizer import qtyping, quantizer, recipe_manager
    from ai_edge_quantizer.algorithm_manager import AlgorithmName

    rm = recipe_manager.RecipeManager()
    rm.add_weight_only_config(regex=".*", operation_name=qtyping.TFLOperationName.FULLY_CONNECTED, num_bits=16,
                              granularity=qtyping.QuantGranularity.CHANNELWISE,
                              algorithm_key=AlgorithmName.FLOAT_CASTING)
    recipe = rm.get_quantization_recipe()
    qt = quantizer.Quantizer(src, recipe)
    qt.quantize().export_model(dst)
    return json.loads(json.dumps(recipe, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--qblock", type=int, default=0)
    ap.add_argument("--keep-fp32", action="store_true")
    ap.add_argument("--lightweight", action="store_true")
    ap.add_argument("--norm-k", default="", help="JSON {module or site: k} (d1_graph.apply_norm_k)")
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    import numpy as np
    import torch

    import d1_graph as G
    from tflite_scan_d1 import scan

    tag = f"L{a.L}" + (f"_qb{a.qblock}" if a.qblock else "") + a.suffix
    gdir = os.path.join(WORK, "out/graphs")
    os.makedirs(gdir, exist_ok=True)
    p32 = os.path.join(gdir, f"d1-2b_decide_{tag}_fp32.tflite")
    p16 = os.path.join(gdir, f"d1-2b_decide_{tag}_fp16fc.tflite")
    rec_path = os.path.join(WORK, f"results/export_decide_{tag}.json")
    for p in (p16, rec_path):
        assert not os.path.exists(p), f"never overwrite {p}"
    torch.set_num_threads(8)
    t0 = time.time()
    cfg = G.Cfg()
    W = G.load_text_weights(dtype=torch.float32)
    W.pop("embed_tokens.weight", None)      # the graph takes embeddings; the table stays on the host
    model = G.D1Decide(W, cfg, a.L, a.qblock or None, torch.float32).eval().requires_grad_(False)
    norm_scaled = G.apply_norm_k(model, json.loads(a.norm_k)) if a.norm_k else []
    T = a.L // 2 + 1
    tags = [-1] * (T // 2) + [1] * (T - T // 2 - 3) + [2] * 3
    q, k = G.make_sees(T // 2, tags, a.L)
    g = torch.Generator().manual_seed(0)
    emb = torch.zeros(1, a.L, cfg.hidden)
    emb[0, :T] = torch.randn(T, cfg.hidden, generator=g) * 0.02
    sample = {"embeds": emb, "sees_q": torch.from_numpy(q), "sees_k": torch.from_numpy(k)}
    with torch.inference_mode():
        ref = model(**sample)["hidden"]
    rec = {"L": a.L, "qblock": a.qblock or None, "tag": tag, "graph": "d1_graph.D1Decide", "norm_scaled": norm_scaled,
           "signature": "decide", "inputs": {k_: list(v.shape) for k_, v in sample.items()},
           "pythonhashseed": os.environ.get("PYTHONHASHSEED"), "lightweight": a.lightweight,
           "versions": {p: importlib.metadata.version(p) for p in
                        ("torch", "litert-torch", "litert-converter", "ai-edge-litert", "ai-edge-quantizer", "numpy")},
           "load_build_s": round(time.time() - t0, 1)}
    import litert_torch

    t1 = time.time()
    if not os.path.exists(p32):
        lrt = litert_torch.signature("decide", model, sample_kwargs=sample).convert(lightweight_conversion=a.lightweight)
        rec["convert_s"] = round(time.time() - t1, 1)
        t2 = time.time()
        lrt.export(p32)
        rec["write_s"] = round(time.time() - t2, 1)
        del lrt
    else:
        rec["fp32_reused"] = True
    rec["fp32"] = {"file": os.path.relpath(p32, WORK), "bytes": os.path.getsize(p32), "sha256": sha256_file(p32)}
    # the float32 file on the CPU against the module (one sample): conversion error alone
    from ai_edge_litert.interpreter import Interpreter

    t3 = time.time()
    it = Interpreter(model_path=p32, num_threads=8)
    run = it.get_signature_runner("decide")
    got = run(**{k_: v.numpy() for k_, v in sample.items()})["hidden"]
    rec["fp32_cpu_vs_module_max_abs"] = float(np.abs(got - ref.numpy()).max())
    rec["fp32_cpu_vs_module_real_max_abs"] = float(np.abs(got[0, :T] - ref.numpy()[0, :T]).max())
    rec["fp32_cpu_s"] = round(time.time() - t3, 1)
    del it, run
    t4 = time.time()
    rec["recipe"] = fp16fc(p32, p16)
    rec["quant_s"] = round(time.time() - t4, 1)
    rec["fp16fc"] = {"file": os.path.relpath(p16, WORK), "bytes": os.path.getsize(p16), "sha256": sha256_file(p16)}
    rec["scan_fp32"] = scan(p32, with_sha=False)
    rec["scan_fp16fc"] = scan(p16, with_sha=False)
    if not a.keep_fp32:
        os.remove(p32)
        rec["fp32"]["deleted"] = True
    rec["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rec["wall_s"] = round(time.time() - t0, 1)
    json.dump(rec, open(rec_path, "w"), indent=1)
    s16 = rec["scan_fp16fc"]
    print(json.dumps({"tag": tag, "fp16fc_bytes": rec["fp16fc"]["bytes"], "fp32_cpu_vs_module_real_max_abs":
                      rec["fp32_cpu_vs_module_real_max_abs"], "ops": s16["operator_count"],
                      "fc_weight_dtypes": s16["fc_weight_dtypes"], "custom": s16["custom_ops"],
                      "rank_gt4": s16["rank_gt4_tensor_count"], "int64": s16["int64_tensor_count"],
                      "largest": s16["largest_tensors"][:3], "peak_rss_gb": round(rec["peak_rss_bytes"] / 2**30, 1),
                      "wall_s": rec["wall_s"]}))


if __name__ == "__main__":
    main()
