"""Export ONE .tflite holding the seven length buckets of the d1-2B decision graph AND the batch graph B = 8 of
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
the seven-signature file (export_multi.py) and the B = 8 file (export_batch.py) when they are in out/graphs/ (same ops and
shapes = the same graph), and the process's peak RSS.

  PYTHONHASHSEED=0 $EXPORT conversion/export_multi8.py [--keep-fp32]

Writes out/graphs/d1-2b_decide_multi8_fp16fc.tflite (+ the float32 file, deleted after the fp16 one unless
--keep-fp32) and results/export_decide_multi8.json. Never overwrites a finished output; a float32 file is reused only when its `.done`
mark exists (a run stopped mid-write leaves none).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import mmap
import os
import resource
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.normpath(os.path.join(HERE, ".."))
R1X = HERE
sys.path.insert(0, R1X)
B8, L8 = 8, 128
MULTI7 = os.path.join(WORK, "out/graphs/d1-2b_decide_multi7_fp16fc.tflite")
B8FILE = os.path.join(WORK, "out/graphs/d1-2b_decide_b8_L128_fp16fc.tflite")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(1 << 24):
            h.update(b)
    return h.hexdigest()


def load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def sample_rows(B, L, hidden, seed=0):
    """export_batch.sample_rows, copied: B rows of different lengths (one full-length row, the rest 43..L),
    state halves, 1-4 questions; embeds = small random values (the batch-1 export sample's scale)."""
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


def op_sequences(path):
    """Per signature: the subgraph's ops as (op name, input shapes, output shapes), and a sha256 of that list."""
    from ai_edge_litert import schema_py_generated as schema

    f = open(path, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    m = schema.Model.GetRootAs(mm, 0)
    bo = {v: k for k, v in vars(schema.BuiltinOperator).items() if isinstance(v, int) and not k.startswith("_")}
    out = {}
    for i in range(m.SignatureDefsLength()):
        sd = m.SignatureDefs(i)
        sg = m.Subgraphs(sd.SubgraphIndex())

        def shp(t):
            if t < 0:
                return None
            x = sg.Tensors(t)
            return [int(x.Shape(k)) for k in range(x.ShapeLength())]
        seq = []
        for o in range(sg.OperatorsLength()):
            op = sg.Operators(o)
            oc = m.OperatorCodes(op.OpcodeIndex())
            name = bo.get(max(oc.BuiltinCode(), oc.DeprecatedBuiltinCode()), "?")
            seq.append([name, [shp(int(t)) for t in op.InputsAsNumpy()], [shp(int(t)) for t in op.OutputsAsNumpy()]])
        blob = json.dumps(seq, separators=(",", ":")).encode()
        out[sd.SignatureKey().decode()] = {"ops": len(seq), "seq_sha256": hashlib.sha256(blob).hexdigest()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-fp32", action="store_true")
    a = ap.parse_args()
    assert os.environ.get("PYTHONHASHSEED") == "0", "run with PYTHONHASHSEED=0 (the other exports did)"
    import numpy as np
    import torch

    # this folder's modules by name first (on sys.path[0]), then the batch module by path: d1_graph_b imports
    # `d1_graph` by name and gets the same module
    import d1_graph as G
    from export_decide import fp16fc
    from export_multi import BUCKETS, buffers, sig_name
    from tflite_scan_d1 import scan
    rd = load_by_path("rank_dump8", os.path.join(HERE, "rank_dump8.py"))
    GB = load_by_path("d1_graph_b", os.path.join(HERE, "d1_graph_b.py"))
    from d1_layout_b import make_sees_batch
    assert GB.G is G, "d1_graph_b must use this d1_graph module"

    gdir = os.path.join(WORK, "out/graphs")
    os.makedirs(gdir, exist_ok=True)
    os.makedirs(os.path.join(WORK, "results"), exist_ok=True)
    p32 = os.path.join(gdir, "d1-2b_decide_multi8_fp32.tflite")
    p16 = os.path.join(gdir, "d1-2b_decide_multi8_fp16fc.tflite")
    rec_path = os.path.join(WORK, "results/export_decide_multi8.json")
    for p in (p16, rec_path):
        assert not os.path.exists(p), f"never overwrite {p}"
    torch.set_num_threads(8)
    t0 = time.time()
    cfg = G.Cfg()
    W = G.load_text_weights(dtype=torch.float32)
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
    rec = {"signatures": sigs, "graph": "d1_graph.D1Decide x 7 + d1_graph_b.D1DecideB B=8 L=128 causal_runtime, "
           "one weight dict", "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
           "modules": {"d1_graph": os.path.relpath(G.__file__, WORK), "d1_graph_sha256": sha256_file(G.__file__),
                       "d1_graph_b": os.path.relpath(GB.__file__, WORK), "d1_graph_b_sha256": sha256_file(GB.__file__),
                       "export_multi_sha256": sha256_file(os.path.join(R1X, "export_multi.py")),
                       "export_decide_sha256": sha256_file(os.path.join(R1X, "export_decide.py"))},
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
    cmp = {}
    if os.path.exists(MULTI7):
        o7 = op_sequences(MULTI7)
        for s in sigs[:-1]:
            cmp[s] = {"vs": "multi7", "ops": [mine[s]["ops"], o7.get(s, {}).get("ops")],
                      "same_sequence": mine[s]["seq_sha256"] == o7.get(s, {}).get("seq_sha256")}
    if os.path.exists(B8FILE):
        ob = op_sequences(B8FILE)
        cmp[sb] = {"vs": "b8 file", "ops": [mine[sb]["ops"], ob.get(sb, {}).get("ops")],
                   "same_sequence": mine[sb]["seq_sha256"] == ob.get(sb, {}).get("seq_sha256")}
    rec["op_sequence_compare"] = cmp
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
