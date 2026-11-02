"""Is the eight-signature file the same computation as the files it merges, on one accelerator under one option set:
Metal float32 with GpuOptions.constant_tensor_sharing (every file of the comparison) or the CPU (XNNPACK 8 threads)
with the XNNPACK weight cache for the multi files (a fresh cache file each) and without one for the single B = 8 file
(the form of its own gate; the cache changes no bit there).

  A  the seven-signature file (out/graphs/d1-2b_decide_multi7_fp16fc.tflite, export_multi.py): every bucket signature
     on runtime_check.decide_inputs(L) (table rows, half state, three questions)
  B  the multi8 file: the same seven inputs (1: vs A), then decide_b8_L128 on the host's own sequences of
     check_batch.py (states64 = the 64-states workload's 64 sequences in 8 calls, mixed64 = 64 records' first sequences
     of 43-128 tokens in 8 calls, partial5 = states64[:5] alone = 3 padding rows), and decide_L128 of the same file on
     each of those 128 sequences (3: batch row vs single call, an instrument)
  C  the B = 8 single file (out/graphs/d1-2b_decide_b8_L128_fp16fc.tflite, export_batch.py) on the same b8 calls
     (2: vs B)
Per comparison: max |d| over the real rows and bit equality; per stage load seconds, is_fully_accelerated and the
process's phys_footprint (on the CPU also the size of the weight cache file). A difference above 1e-3 in 1 or 2
(`stop_condition_1e-3`) stops the checks.

  $EXPORT conversion/multi8_check.py --accel gpu --out results/multi8_check_gpu.json
  $EXPORT conversion/multi8_check.py --accel cpu --out results/multi8_check_cpu.json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
BUCKETS = ((128, 0), (256, 0), (512, 0), (1024, 0), (2048, 0), (4096, 1024), (6144, 1024))
MULTI7 = "out/graphs/d1-2b_decide_multi7_fp16fc.tflite"
MULTI8 = "out/graphs/d1-2b_decide_multi8_fp16fc.tflite"
B8FILE = "out/graphs/d1-2b_decide_b8_L128_fp16fc.tflite"
BSIG = "decide_b8_L128"


def load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def tag(L, qb):
    return f"L{L}" + (f"_qb{qb}" if qb else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--accel", choices=("cpu", "gpu"), required=True)
    ap.add_argument("--cache-dir", default="out/xnn_cache")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import numpy as np

    from lrt_backend import MultiGraph
    fp = load_by_path("r4_fp", os.path.join(WORK, "conversion/fp.py")).footprint
    rc = load_by_path("r1_runtime_check", os.path.join(WORK, "conversion/runtime_check.py"))
    cb = load_by_path("r4_check_batch", os.path.join(WORK, "conversion/check_batch.py"))
    from d1_layout import SEES, make_sees

    out_path = os.path.join(WORK, a.out)
    gpu = a.accel == "gpu"
    doc = {"accelerator": a.accel, "precision": "fp32", "time": time.strftime("%Y-%m-%d %H:%M:%S"),
           "options": ("Metal float32 + constant_tensor_sharing on every file" if gpu else
                       "XNNPACK 8 threads; weight cache: a fresh file for multi7 and multi8, none for the B=8 file"),
           "files": {"multi7": MULTI7, "multi8": MULTI8, "b8_single": B8FILE}, "stages": {}, "compare": {}}

    def dump():
        json.dump(doc, open(out_path + ".tmp", "w"), indent=1)
        os.replace(out_path + ".tmp", out_path)

    def cache(name):
        if gpu:
            return ""
        d = os.path.join(WORK, a.cache_dir)
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, name)
        return p

    def load(key, rel, wc=""):
        t0 = time.time()
        fresh = bool(wc) and not os.path.exists(wc)
        m = MultiGraph(os.path.join(WORK, rel), a.accel, "fp32", 8, sharing=gpu, weight_cache=wc)
        doc["stages"][key] = {"file": rel, "load_s": round(time.time() - t0, 2), "fully_accelerated": m.fully_accelerated,
                              "signatures": sorted(m.signatures), "weight_cache": os.path.relpath(wc, WORK) if wc else None,
                              "weight_cache_written_by_this_load": fresh, "footprint_after_load": fp()}
        if wc:
            doc["stages"][key]["weight_cache_bytes"] = os.path.getsize(wc) if os.path.exists(wc) else None
        dump()
        return m

    feeds = {}
    for L, qb in BUCKETS:
        feeds[(L, qb)] = rc.decide_inputs(L)
    states64, mixed64, pick = cb.sequences()
    sets = {"states64": states64, "mixed64": mixed64, "partial5": states64[:5]}
    doc["b8_sets"] = {n: {"sequences": len(s), "T_min": min(len(q) for _, _, q in s),
                          "T_max": max(len(q) for _, _, q in s),
                          "embeds_sha256": hashlib.sha256(b"".join(np.ascontiguousarray(e).tobytes()
                                                                     for e, _, _ in s)).hexdigest()}
                      for n, s in sets.items()}
    doc["mixed64_records"] = pick

    def b8_calls(m, seqs):
        outs = []
        for i in range(0, len(seqs), 8):
            chunk = seqs[i:i + 8]
            e = np.zeros((8, 128, 2048), np.float32)
            q = np.zeros((8, 128, SEES), np.float32)
            k = np.zeros((8, 128, SEES), np.float32)
            q[:, :, 1] = 1.0
            for b, (emb, st, tags) in enumerate(chunk):
                e[b, : emb.shape[0]] = emb
                qb_, kb_ = make_sees(st, tags, 128)
                q[b], k[b] = qb_[0], kb_[0]
            h = m(BSIG, embeds=e, sees_q=q, sees_k=k)["hidden"]
            outs.extend(h[b, : chunk[b][0].shape[0]].copy() for b in range(len(chunk)))
        return outs

    def single_calls(m, seqs):
        outs = []
        for emb, st, tags in seqs:
            T = emb.shape[0]
            e = np.zeros((1, 128, 2048), np.float32)
            e[0, :T] = emb
            q, k = make_sees(st, tags, 128)
            outs.append(m("decide_L128", embeds=e, sees_q=q, sees_k=k)["hidden"][0, :T].copy())
        return outs

    def cmp(name, xs, ys):
        d = [float(np.abs(x - y).max()) for x, y in zip(xs, ys)]
        doc["compare"][name] = {"max_abs": max(d), "bit_equal": all(np.array_equal(x, y) for x, y in zip(xs, ys)),
                                "n": len(d), "finite": all(bool(np.isfinite(x).all()) for x in xs)}
        print(json.dumps({name: doc["compare"][name]}), flush=True)
        dump()

    # A: the seven-signature file
    wc7 = cache("multi7_check.xnncache")
    m = load("A_multi7", MULTI7, wc7)
    out7 = {}
    for L, qb in BUCKETS:
        f, T = feeds[(L, qb)]
        out7[(L, qb)] = m("decide_" + tag(L, qb), **f)["hidden"][0, :T].copy()
    doc["stages"]["A_multi7"]["footprint_after_runs"] = fp()
    m.close()
    del m
    if wc7 and os.path.exists(wc7):
        doc["stages"]["A_multi7"]["weight_cache_deleted"] = os.path.getsize(wc7)
        os.remove(wc7)
    dump()
    # B: multi8
    m = load("B_multi8", MULTI8, cache("multi8_fp16fc.xnncache"))
    out8 = {}
    t_first = {}
    for L, qb in BUCKETS:
        f, T = feeds[(L, qb)]
        t1 = time.perf_counter()
        out8[(L, qb)] = m("decide_" + tag(L, qb), **f)["hidden"][0, :T].copy()
        t_first["decide_" + tag(L, qb)] = round(time.perf_counter() - t1, 3)
    for L, qb in BUCKETS:
        cmp(f"1_multi8_vs_multi7_decide_{tag(L, qb)}", [out8[(L, qb)]], [out7[(L, qb)]])
    hb8 = {}
    for n, s in sets.items():
        t1 = time.perf_counter()
        hb8[n] = b8_calls(m, s)
        t_first[f"{BSIG}:{n}"] = round(time.perf_counter() - t1, 3)
    s128 = {n: single_calls(m, sets[n]) for n in ("states64", "mixed64")}
    doc["stages"]["B_multi8"]["first_runs_s"] = t_first
    doc["stages"]["B_multi8"]["footprint_after_runs"] = fp()
    m.close()
    del m
    for n in ("states64", "mixed64"):
        cmp(f"3_multi8_b8_vs_multi8_L128_{n}", hb8[n], s128[n])
    cmp("3_multi8_b8_partial5_vs_full_batch", hb8["partial5"], hb8["states64"][:5])
    # C: the B = 8 single file
    m = load("C_b8_single", B8FILE, "")
    hr4 = {n: b8_calls(m, s) for n, s in sets.items()}
    doc["stages"]["C_b8_single"]["footprint_after_runs"] = fp()
    m.close()
    del m
    for n in sets:
        cmp(f"2_multi8_b8_vs_r4_b8_{n}", hb8[n], hr4[n])
    worst = max(v["max_abs"] for k, v in doc["compare"].items() if k[0] in "12")
    doc["max_abs_1_2"] = worst
    doc["all_bit_equal_1_2"] = all(v["bit_equal"] for k, v in doc["compare"].items() if k[0] in "12")
    doc["stop_condition_1e-3"] = worst > 1e-3
    doc["footprint_end"] = fp()
    doc["torch_imported"] = "torch" in sys.modules
    dump()
    print("DONE", json.dumps({k: doc[k] for k in ("accelerator", "max_abs_1_2", "all_bit_equal_1_2",
                                                  "stop_condition_1e-3", "torch_imported")}))


if __name__ == "__main__":
    main()
