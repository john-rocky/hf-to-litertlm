"""Export the d1-2B decision graph as ONE .tflite holding all seven buckets as signatures (decide_L128, decide_L256,
decide_L512, decide_L1024, decide_L2048, decide_L4096_qb1024, decide_L6144_qb1024), then cast its FULLY_CONNECTED
weights to float16 with the single-bucket recipe (export_decide.fp16fc), then count how the weights are stored.

The seven D1Decide modules are built over ONE weight dict: d1_graph.linear keeps the tensors themselves, so every
signature's FC weight is the same torch storage, and litert-torch's constant cache (backend/inline_consts.py, keyed by
storage pointer, shape, stride and offset) can lower it once. Whether the converter and the quantizer keep one copy is
what this script measures (`buffers` in the record): the FC weight buffers each subgraph reads, how many are distinct
across subgraphs, and their bytes.

  PYTHONHASHSEED=0 $EXPORT conversion/export_multi.py [--keep-fp32]

Writes out/graphs/d1-2b_decide_multi7_fp16fc.tflite (+ the float32 file, deleted after the fp16 one unless
--keep-fp32) and results/export_decide_multi7.json. Existing outputs are never overwritten.
"""
from __future__ import annotations

import argparse
import collections
import importlib.metadata
import json
import mmap
import os
import resource
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
BUCKETS = ((128, 0), (256, 0), (512, 0), (1024, 0), (2048, 0), (4096, 1024), (6144, 1024))


def sig_name(L, qb):
    return f"decide_L{L}" + (f"_qb{qb}" if qb else "")


def buffers(path):
    """FC weight storage across subgraphs: per subgraph the buffer each FULLY_CONNECTED reads (through a DEQUANTIZE
    when the weight is float16), the distinct ones over the file, their bytes; and every buffer's bytes."""
    from ai_edge_litert import schema_py_generated as schema

    f = open(path, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    model = schema.Model.GetRootAs(mm, 0)
    op_names = {v: k for k, v in vars(schema.BuiltinOperator).items() if isinstance(v, int)}
    codes = [op_names.get(max(model.OperatorCodes(i).BuiltinCode(), model.OperatorCodes(i).DeprecatedBuiltinCode()))
             for i in range(model.OperatorCodesLength())]

    def nbytes(bi):
        b = model.Buffers(bi)
        if b is None:
            return 0
        return b.DataLength() if b.DataLength() > 0 else (b.Size() if b.Offset() > 1 else 0)

    per_sg, users = [], collections.defaultdict(set)
    for si in range(model.SubgraphsLength()):
        sg = model.Subgraphs(si)
        producer = {}
        for oi in range(sg.OperatorsLength()):
            op = sg.Operators(oi)
            for k in range(op.OutputsLength()):
                producer[op.Outputs(k)] = op
        fc_bufs = []
        for oi in range(sg.OperatorsLength()):
            op = sg.Operators(oi)
            if codes[op.OpcodeIndex()] != "FULLY_CONNECTED" or op.InputsLength() < 2:
                continue
            ti = op.Inputs(1)
            t = sg.Tensors(ti)
            bi = t.Buffer()
            if nbytes(bi) == 0 and ti in producer and codes[producer[ti].OpcodeIndex()] == "DEQUANTIZE":
                bi = sg.Tensors(producer[ti].Inputs(0)).Buffer()
            if nbytes(bi) == 0:
                continue                      # a runtime operand (the mask's sees_q . sees_k)
            fc_bufs.append(bi)
            users[bi].add(si)
        name = sg.Name().decode() if sg.Name() else str(si)
        per_sg.append({"subgraph": si, "name": name, "fc_weight_buffers": len(fc_bufs),
                       "fc_weight_bytes": sum(nbytes(b) for b in fc_bufs)})
    distinct = sorted(users)
    all_bytes = sum(nbytes(i) for i in range(model.BuffersLength()))
    return {"file_bytes": os.path.getsize(path), "buffers": model.BuffersLength(), "all_buffer_bytes": all_bytes,
            "subgraphs": model.SubgraphsLength(),
            "signatures": [model.SignatureDefs(i).SignatureKey().decode() for i in range(model.SignatureDefsLength())],
            "per_subgraph": per_sg, "fc_weight_buffers_distinct": len(distinct),
            "fc_weight_bytes_distinct": sum(nbytes(b) for b in distinct),
            "fc_weight_buffers_read_by_every_subgraph": sum(1 for b in distinct if len(users[b]) == model.SubgraphsLength()),
            "fc_weight_buffers_read_by_one_subgraph": sum(1 for b in distinct if len(users[b]) == 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-fp32", action="store_true")
    a = ap.parse_args()
    import torch

    import d1_graph as G
    from export_decide import fp16fc, sha256_file

    gdir = os.path.join(WORK, "out/graphs")
    p32 = os.path.join(gdir, "d1-2b_decide_multi7_fp32.tflite")
    p16 = os.path.join(gdir, "d1-2b_decide_multi7_fp16fc.tflite")
    rec_path = os.path.join(WORK, "results/export_decide_multi7.json")
    for p in (p16, rec_path):
        assert not os.path.exists(p), f"never overwrite {p}"
    torch.set_num_threads(8)
    t0 = time.time()
    cfg = G.Cfg()
    W = G.load_text_weights(dtype=torch.float32)
    W.pop("embed_tokens.weight", None)      # the graph takes embeddings; the table stays on the host
    import litert_torch

    conv, sigs = None, []
    for L, qb in BUCKETS:
        model = G.D1Decide(W, cfg, L, qb or None, torch.float32).eval().requires_grad_(False)
        T = L // 2 + 1                          # the single-bucket export's sample (export_decide.py)
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
    rec = {"signatures": sigs, "graph": "d1_graph.D1Decide x 7 over one weight dict",
           "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
           "versions": {p: importlib.metadata.version(p) for p in
                        ("torch", "litert-torch", "litert-converter", "ai-edge-litert", "ai-edge-quantizer", "numpy")},
           "load_build_s": round(time.time() - t0, 1)}
    t1 = time.time()
    if not os.path.exists(p32):
        lrt = conv.convert()
        rec["convert_s"] = round(time.time() - t1, 1)
        t2 = time.time()
        lrt.export(p32)
        rec["write_s"] = round(time.time() - t2, 1)
        del lrt
    else:
        rec["fp32_reused"] = True
    del conv, W, model
    rec["fp32"] = {"file": os.path.relpath(p32, WORK), "bytes": os.path.getsize(p32), "sha256": sha256_file(p32)}
    rec["buffers_fp32"] = buffers(p32)
    print(json.dumps({"fp32_bytes": rec["fp32"]["bytes"], **{k: rec["buffers_fp32"][k] for k in (
        "subgraphs", "fc_weight_buffers_distinct", "fc_weight_bytes_distinct",
        "fc_weight_buffers_read_by_every_subgraph")}}), flush=True)
    t4 = time.time()
    rec["recipe"] = fp16fc(p32, p16)
    rec["quant_s"] = round(time.time() - t4, 1)
    rec["fp16fc"] = {"file": os.path.relpath(p16, WORK), "bytes": os.path.getsize(p16), "sha256": sha256_file(p16)}
    rec["buffers_fp16fc"] = buffers(p16)
    if not a.keep_fp32:
        os.remove(p32)
        rec["fp32"]["deleted"] = True
    rec["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rec["wall_s"] = round(time.time() - t0, 1)
    json.dump(rec, open(rec_path, "w"), indent=1)
    b = rec["buffers_fp16fc"]
    print(json.dumps({"fp16fc_bytes": rec["fp16fc"]["bytes"], "signatures": b["signatures"],
                      "fc_weight_buffers_distinct": b["fc_weight_buffers_distinct"],
                      "fc_weight_bytes_distinct": b["fc_weight_bytes_distinct"],
                      "read_by_every_subgraph": b["fc_weight_buffers_read_by_every_subgraph"],
                      "per_subgraph": [(s["name"], s["fc_weight_buffers"]) for s in b["per_subgraph"]],
                      "peak_rss_gb": round(rec["peak_rss_bytes"] / 2**30, 1), "wall_s": rec["wall_s"]}))


if __name__ == "__main__":
    main()
