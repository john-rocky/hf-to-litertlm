#!/usr/bin/env python3
"""Export one bucket of the decision graph (decide_graph.DecideRow), then cast its FULLY_CONNECTED weights to float16, then read
both files. One bucket a process, one process at a time.

    PYTHONHASHSEED=0 $EXPORT conversion/export_decide.py --L 256 [--qblock 1024] [--keep-fp32] [--path default|lightweight]

Writes out/graphs/decide_<tag>_{fp32,fp16fc}.tflite and results/export_decide_<tag>.json (seconds, bytes, sha256, memory,
versions, op histograms, FC weight dtypes, the acceptance checks). Existing outputs are never overwritten; an existing fp32 file
is reused (the fp16 step alone).
Acceptance (form by form):
  fp32:   COS / SIN / GATHER / GATHER_ND / BROADCAST_TO / SELECT_V2 / EMBEDDING_LOOKUP 0, INT64 0, CUSTOM 0, rank > 4 0,
          all-constant-input ops 0, every FC weight a constant with data, signature = the contract, embed_tokens / score not in
          the file (no tensor with 248320 rows, no constant of shape [2, 2048]).
  fp16fc: FC count = the input's, every FC weight FLOAT16 through DEQUANTIZE, DEQUANTIZE = the FC weight buffers of the input, op
          histogram = the input's + DEQUANTIZE; the shared 0 / 1 kernel matrices read back bit-equal.
  sample: the module on a fixture record (the author's inputs_embeds and rotary tables, out/fix/rec_<id>.npz) against the
          author's hidden: rel_rms recorded, the fp32 file on the CPU against the module: rel_rms recorded. With no fixture
          (fixtures_author.py not run) the sample is a blank record of L tokens: it only carries the shapes through the export, and
          the file on the CPU is compared with the module alone."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import resource
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(os.path.abspath(__file__)).parent
WORK = HERE.parent        # the work directory: snapshot/, out/, results/
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "host"))
import fp16fc as Q  # noqa: E402
import lrt  # noqa: E402

THRESHOLD = 25_000_000
NEG_LIST = ("COS", "SIN", "GATHER", "GATHER_ND", "BROADCAST_TO", "SELECT_V2", "EMBEDDING_LOOKUP")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(1 << 24):
            h.update(b)
    return h.hexdigest()


def sample_record(L: int):
    """The largest fixture record with T <= L: (tag, npz)."""
    best = None
    for p in (WORK / "out/fix").glob("rec_*.npz"):
        with np.load(p) as z:
            T = int(z["ids"].shape[0])
        if T <= L and (best is None or T > best[0]):
            best = (T, p)
    if best is None:
        return "blank", blank_record(L)
    return best[1].stem[4:], np.load(best[1])


def blank_record(L: int, hidden: int = 2048) -> dict:
    """No fixture: L tokens of seeded random embeddings, the planes and tables of a text record. No author's numbers in it."""
    from decide_host import formulas as F

    pos3 = np.tile(np.arange(L, dtype=np.int64), (3, 1))
    c, s = F.mrope_tables(pos3)
    return {"ids": np.zeros(L, np.int32), "inputs_embeds": np.random.default_rng(0).normal(0.0, 0.02, (L, hidden)).astype(np.float32), "pos3": pos3, "cos": c[0], "sin": s[0]}


def padded_inputs(z, L: int, hidden: int = 2048) -> dict:
    """The contract's padding on the author's own tensors. The pad rows of cos / sin = the author's rotary at the continuing
    positions (host_formulas.padded_planes), computed here from the planes."""
    from decide_host import formulas as F

    T = int(z["ids"].shape[0])
    e = np.zeros((1, L, hidden), np.float32)
    e[0, :T] = z["inputs_embeds"]
    v = np.zeros((1, L), np.float32)
    v[0, :T] = 1.0
    c, s = F.mrope_tables(F.padded_planes(z["pos3"].astype(np.int64), L))
    c[0, :T] = z["cos"]
    s[0, :T] = z["sin"]
    return {"embeds": e, "valid": v, "cos": c.astype(np.float32), "sin": s.astype(np.float32)}


def footprint_sampler(stop, rec, every=5.0):
    while not stop.wait(every):
        m = lrt.memory()
        rec["footprint_max"] = max(rec.get("footprint_max", 0), m.get("phys_footprint", 0))
        rec["rss_max"] = max(rec.get("rss_max", 0), m.get("resident_size", 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--qblock", type=int, default=0)
    ap.add_argument("--keep-fp32", action="store_true")
    ap.add_argument("--path", choices=("default", "lightweight"), default="default")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--suffix", default="", help="a variant's file names (the variant's module is set by --form)")
    ap.add_argument("--form", default="", help="another kernel form than decide_graph.FORM")
    ap.add_argument("--variant", default="", choices=("", "f16s", "zba", "head", "xb"), help="other forms of the graph: files decide_<tag>_<variant>32 / <variant>fc, record export_decide_<tag>_<variant>.json; xb = the plain module, another bucket set kept apart")
    a = ap.parse_args()
    if a.path == "lightweight":
        os.environ["RESOURCE_CONSTANT_NUMEL_THRESHOLD"] = str(THRESHOLD)
    import torch

    import decide_graph as G

    torch.set_num_threads(a.threads)
    torch.set_grad_enabled(False)
    tag = G.tag(a.L, a.qblock) + a.suffix
    gdir = WORK / "out/graphs"
    gdir.mkdir(parents=True, exist_ok=True)
    v = a.variant
    p32 = gdir / (f"decide_{tag}_{v}32.tflite" if v else f"decide_{tag}_fp32.tflite")
    p16 = gdir / (f"decide_{tag}_{v}fc.tflite" if v else f"decide_{tag}_fp16fc.tflite")
    rec_path = WORK / (f"results/export_decide_{tag}_{v}.json" if v else f"results/export_decide_{tag}.json")
    for p in (p16, rec_path):
        assert not p.exists(), f"never overwrite {p}"
    t0 = time.time()
    rec = {"L": a.L, "qblock": a.qblock, "tag": tag, "signature": "decide", "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
           "path": a.path, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "versions": {p: importlib.metadata.version(p) for p in
                        ("torch", "transformers", "litert-torch", "litert-converter", "ai-edge-litert", "ai-edge-quantizer", "numpy")},
           "scripts": {f.name: sha256_file(f)[:16] for f in sorted(HERE.glob("*.py"))}}
    stop = threading.Event()
    watch = {}
    threading.Thread(target=footprint_sampler, args=(stop, watch), daemon=True).start()
    model, info = G.load_text_model()
    rec["load_info"] = info
    form = a.form or G.FORM
    rec["form"] = form
    rec["form_apply"] = G.apply_form(model, form)
    rec["variant"] = v
    if v == "f16s":
        rec["variant_apply"] = G.apply_f16s(model)
    elif v == "zba":
        rec["variant_apply"] = G.apply_zba(model)
    name, z = sample_record(a.L)
    assert v != "head" or "logits" in z, "--variant head compares with the author's logits: it needs fixtures"
    T = int(z["ids"].shape[0])
    feed = padded_inputs(z, a.L)
    if v == "head":
        graph = G.graph_head(model, a.L, a.qblock)
        feed["pick"] = np.zeros((1, a.L), np.float32)
        feed["pick"][0, T - 1] = 1.0
    else:
        graph = G.graph(model, a.L, a.qblock)
    sample = {k: torch.from_numpy(x) for k, x in feed.items()}
    ts = time.time()
    if v == "head":
        l_mod = graph(**sample)["logits"][0].numpy()
        rec["sample"] = {"record": name, "T": T, "module_forward_s": round(time.time() - ts, 1), "logits_module": l_mod.tolist(), "logits_author": z["logits"].tolist(),
                         "d_logit": float(np.abs(l_mod - z["logits"]).max())}
    else:
        h_mod = graph(**sample)["hidden"][0].numpy()
        want = z["hidden"] if "hidden" in z else None
        rec["sample"] = {"record": name, "T": T, "module_forward_s": round(time.time() - ts, 1)}
        if want is not None:
            rec["sample"].update(module_vs_author=lrt.stats(h_mod[:T], want), pooled_cos=lrt.stats(h_mod[T - 1], want[T - 1])["cos"])
    print(json.dumps({"tag": tag, "sample": rec["sample"], "memory": lrt.memory()}), flush=True)
    rec["load_and_check_s"] = round(time.time() - t0, 1)
    rec["memory_after_load"] = lrt.memory()
    if not p32.exists():
        import litert_torch

        t1 = time.time()
        conv = litert_torch.signature("decide", graph, sample_kwargs=sample)
        kw = dict(lightweight_conversion=True, runtime_constant_folding=False) if a.path == "lightweight" else {}
        lrt_model = conv.convert(**kw)
        rec["convert_s"] = round(time.time() - t1, 1)
        rec["memory_after_convert"] = lrt.memory()
        t2 = time.time()
        lrt_model.export(str(p32))
        rec["write_s"] = round(time.time() - t2, 1)
        del lrt_model, conv
    else:
        rec["fp32_reused"] = True
    rec["fp32"] = {"file": str(p32.relative_to(WORK)), "bytes": p32.stat().st_size, "sha256": sha256_file(p32)}
    rec["memory_after_fp32"] = lrt.memory()
    t3 = time.time()
    s32 = Q.fc_scan(p32)
    gen = Q.general_scan(p32)
    rec["scan_fp32"] = {**s32, **gen}
    rec["scan_fp32_s"] = round(time.time() - t3, 1)
    print(json.dumps({"tag": tag, "fp32_bytes": rec["fp32"]["bytes"], "ops": s32["operator_count"], "fc": s32["fully_connected"],
                      "convert_s": rec.get("convert_s"), "memory": rec["memory_after_fp32"]}), flush=True)
    t4 = time.time()
    g = lrt.Graph(p32, "decide", "cpu", threads=a.threads)
    if v == "head":
        out = g(**feed)["logits"][0]
        rec["fp32_cpu"] = {"load_s": round(g.load_s, 1), "call_ms": round(g.last_ms, 1), "logits": out.tolist(), "vs_author": {"rel_rms": float(np.abs(out - z["logits"]).max())}}
    else:
        out = g(**feed)["hidden"][0]
        rec["fp32_cpu"] = {"load_s": round(g.load_s, 1), "call_ms": round(g.last_ms, 1), "vs_module": lrt.stats(out[:T], h_mod[:T])}
        if want is not None:
            rec["fp32_cpu"].update(vs_author=lrt.stats(out[:T], want), pooled_vs_author=lrt.stats(out[T - 1], want[T - 1]))
    g.close()
    rec["fp32_cpu_check_s"] = round(time.time() - t4, 1)
    print(json.dumps({"tag": tag, "fp32_cpu": rec["fp32_cpu"]}), flush=True)
    t5 = time.time()
    q = Q.fp16fc(p32, p16)
    rec["fp16fc_recipe"] = q
    rec["quant_s"] = round(time.time() - t5, 1)
    rec["fp16fc"] = {"file": str(p16.relative_to(WORK)), "bytes": p16.stat().st_size, "sha256": sha256_file(p16)}
    s16 = Q.fc_scan(p16)
    gen16 = Q.general_scan(p16)
    rec["scan_fp16fc"] = {**s16, **gen16}
    checks = {
        "fp32_forbidden_0": all(gen["op_histogram"].get(k, 0) == 0 for k in NEG_LIST),
        "fp32_int64_0": gen["int64_tensor_count"] == 0, "fp32_custom_0": s32["custom_op_count"] == 0, "fp32_rank_gt4_0": gen["rank_gt4_tensor_count"] == 0,
        "fp32_all_constant_input_ops_0": gen["all_constant_input_ops"] == 0,
        "fp32_fc_weights_constant": s32["fc_weight_tensor_dtype_direct"] == {"FLOAT32": s32["fully_connected"]},
        "fp32_no_table_no_head": gen["tensors_with_vocab_rows"] == 0 and gen["constants_shape_2x2048"] == (1 if v == "head" else 0),
        "signature": Q.signature_ok(gen["signatures"], G.contract(a.L, a.qblock, head=v == "head")),
        "fp16_fc_count": s16["fully_connected"] == s32["fully_connected"],
        "fp16_fc_weight_all_FLOAT16_via_DEQUANTIZE": s16["fc_weight_source_dtype"] == {"FLOAT16": s32["fully_connected"]},
        "fp16_dequantize_one_per_input_fc_buffer": s16["dequantize_in_out"] == {"FLOAT16->FLOAT32": s32["fc_weight_buffers"]},
        "fp16_op_histogram_input_plus_dequantize": {k: v for k, v in s16["op_histogram"].items() if k != "DEQUANTIZE"} == s32["op_histogram"],
        "fp16_int8_int4_0": s16["tensor_dtype_histogram"].get("INT8", 0) == 0 and s16["tensor_dtype_histogram"].get("INT4", 0) == 0,
    }
    rec["checks"] = checks
    rec["checks_pass"] = all(checks.values())
    rec["failing"] = [k for k, v in checks.items() if not v]
    if not a.keep_fp32:
        p32.unlink()
        rec["fp32"]["deleted"] = True
    stop.set()
    rec.update(watch)
    rec["memory_end"] = lrt.memory()
    rec["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rec["wall_s"] = round(time.time() - t0, 1)
    rec["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    rec_path.parent.mkdir(parents=True, exist_ok=True)
    rec_path.write_text(json.dumps(rec, indent=1, default=str) + "\n")
    print("EXPORT_DONE " + json.dumps({"tag": tag, "checks_pass": rec["checks_pass"], "failing": rec["failing"], "fp16_bytes": rec["fp16fc"]["bytes"],
                                       "ops": s16["operator_count"], "fc": s16["fully_connected"], "dq": s16["dequantize_count"],
                                       "fp32_cpu_vs_author_rel": rec["fp32_cpu"].get("vs_author", {}).get("rel_rms"), "fp32_cpu_vs_module_rel": rec["fp32_cpu"].get("vs_module", {}).get("rel_rms"),
                                       "footprint_max_gib": round(rec.get("footprint_max", 0) / 2**30, 1), "peak_rss_gib": round(rec["peak_rss_bytes"] / 2**30, 1),
                                       "wall_s": rec["wall_s"]}), flush=True)
    return 0 if rec["checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
