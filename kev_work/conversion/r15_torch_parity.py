"""Proof that a kernel form (r15_form.py) does not change the fp32 result of Kev-4B (torch, all 402 questions):
r14_torch_parity.py with --model 4b; the math and the gate are the same.

    python r15_torch_parity.py --model 4b --form R64+sp+ec+dd+vs6+in1+fn5 [--shards 8] [--concurrency 4]
        -> results/r15_torch_parity_4b_<R64-sp-ec-dd-vs6-in1-fn5>.json   (never overwritten)

Per question: the patched torch graph (kev_graph.KevPrefill, fp32, torch 1 thread per shard) on the row padded to the
smallest bucket L in (64, 128, 256, 512, 1024, 2048) that fits it, with the loop kernel and no rewrite ("stock") and
with the form applied; h_sel max |diff|, hidden max |diff| at the real positions, probs max |diff| (host head =
r2_common.Head of the 4B checkpoint), argmax equality, both vs the 4B oracle. Gate: h_sel <= 1e-4, probs <= 1e-5,
argmax 402/402, all finite (the hidden number is recorded, not gated). The stock hidden rows are computed once per
shard layout and cached (cache/r15/stock_rows_<model>_<i>of<n>.npz), so a second form runs only its own pass.
The 4B checkpoint is fp32 safetensors (mmapped by transformers: the shards share the page cache)."""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

from r2_common import MODEL, RESULT_SUFFIX, K, Head, add_model_arg, dump_json, load_oracle, qkey, select

BUCKETS = (64, 128, 256, 512, 1024, 2048)
GATE = {"h_sel_max_abs": 1e-4, "probs_max_abs": 1e-5}
CACHE = K / "cache/r15"


def ftag(form):
    return form.replace("+", "-")


def worker(form, i, n):
    import torch
    torch.set_num_threads(1)
    import r15_form as R   # r13_kernel's tokens + fn<k> / in<k> (norm pre-scales)
    from kev_graph import KevPrefill, load_text_model, row_inputs
    out = CACHE / f"parity_{MODEL}_{ftag(form)}_{i}of{n}.json"
    model, load_info = load_text_model()
    oracle = load_oracle()
    qs = oracle["questions"][i::n]
    head = Head()
    stock_path = CACHE / f"stock_rows_{MODEL}_{i}of{n}.npz"
    t0 = time.perf_counter()

    def run_all():
        graphs, res = {}, {}
        for q in qs:
            L = next(b for b in BUCKETS if b >= q["row_len"])
            if L not in graphs:
                graphs[L] = KevPrefill(model, L).eval().requires_grad_(False)
            with torch.no_grad():
                h = graphs[L](*row_inputs(q["row_ids"], L))["hidden"][0].numpy()
            res[qkey(q)] = h[: q["row_len"]].copy()
        return res

    if stock_path.exists():
        z = np.load(stock_path)
        stock = {k: z[k] for k in z.files}
        stock_cached = True
    else:
        stock = run_all()
        np.savez(stock_path, **stock)
        stock_cached = False
    t1 = time.perf_counter()
    info = R.apply(model, form)
    new = run_all()
    R.reset(model)
    rows = []
    for q in qs:
        k = qkey(q)
        hs, hr = stock[k].astype(np.float64), new[k]
        ss, sr = select(hs, q), select(hr, q)
        _, _, ps = head(ss.astype(np.float32))
        _, _, pr = head(sr)
        po = np.asarray(q["probs"], np.float64)
        rows.append({"key": k, "row_len": q["row_len"], "bucket": next(b for b in BUCKETS if b >= q["row_len"]),
                     "finite": bool(np.isfinite(hr).all()),
                     "hidden_max_abs": float(np.abs(hr - hs).max()), "h_sel_max_abs": float(np.abs(sr - ss).max()),
                     "probs_max_abs": float(np.abs(pr.astype(np.float64) - ps).max()),
                     "argmax_equal": int(np.argmax(pr)) == int(np.argmax(ps)),
                     "stock_vs_oracle_max_abs_dp": float(np.abs(ps - po).max()),
                     "form_vs_oracle_max_abs_dp": float(np.abs(pr - po).max())})
    dump_json(out, {"form": form, "model": MODEL, "shard": f"{i}/{n}",
                    "apply": {k: v for k, v in info.items() if k != "tokens"}, "load_info": load_info,
                    "stock_from_cache": stock_cached, "seconds_stock": round(t1 - t0, 1),
                    "seconds_form": round(time.perf_counter() - t1, 1), "rows": rows}, overwrite=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--form", required=True)
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=0,
                    help="run the shards at most this many at a time (0 = all at once); the shard layout, and so the "
                         "stock-row cache of that layout, stays the --shards one (timing-window rule: 4 workers)")
    ap.add_argument("--worker", default="", help="internal: i/n")
    add_model_arg(ap)
    a = ap.parse_args()
    assert a.model == MODEL
    if a.worker:
        i, n = (int(x) for x in a.worker.split("/"))
        worker(a.form, i, n)
        return
    out = K / f"results/r15_torch_parity{RESULT_SUFFIX}_{ftag(a.form)}.json"
    assert not out.exists(), out
    CACHE.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    env = dict(os.environ, OMP_NUM_THREADS="1")
    def start(i):
        return subprocess.Popen([sys.executable, __file__, "--model", MODEL, "--form", a.form,
                                 "--worker", f"{i}/{a.shards}"],
                                cwd=str(K / "scripts"), env=env,
                                stdout=open(K / f"logs/r15_parity_{MODEL}_{ftag(a.form)}_{i}.log", "w"),
                                stderr=subprocess.STDOUT)

    conc = a.concurrency or a.shards
    pending, running, rcs = list(range(a.shards)), [], []
    while pending or running:
        while pending and len(running) < conc:
            running.append(start(pending.pop(0)))
        time.sleep(1)
        for p in [p for p in running if p.poll() is not None]:
            rcs.append(p.returncode)
            running.remove(p)
    assert all(rc == 0 for rc in rcs), rcs
    parts = [json.loads((CACHE / f"parity_{MODEL}_{ftag(a.form)}_{i}of{a.shards}.json").read_text())
             for i in range(a.shards)]
    rows = sorted((r for p in parts for r in p["rows"]), key=lambda r: r["key"])
    hs = max(r["h_sel_max_abs"] for r in rows)
    pm = max(r["probs_max_abs"] for r in rows)
    by_bucket = {}
    for r in rows:
        b = by_bucket.setdefault(str(r["bucket"]), {"questions": 0, "h_sel_max_abs": 0.0, "probs_max_abs": 0.0})
        b["questions"] += 1
        b["h_sel_max_abs"] = max(b["h_sel_max_abs"], r["h_sel_max_abs"])
        b["probs_max_abs"] = max(b["probs_max_abs"], r["probs_max_abs"])
    doc = {"step": "form vs the loop-kernel graph in torch fp32 (all oracle questions, each at its smallest bucket)",
           "model": MODEL, "form": a.form, "apply": parts[0]["apply"], "load_info": parts[0]["load_info"],
           "questions": len(rows), "gate": GATE,
           "h_sel_max_abs": hs, "probs_max_abs": pm,
           "hidden_max_abs_real_positions": max(r["hidden_max_abs"] for r in rows),
           "argmax_equal": sum(r["argmax_equal"] for r in rows), "all_finite": all(r["finite"] for r in rows),
           "stock_vs_oracle_max_abs_dp": max(r["stock_vs_oracle_max_abs_dp"] for r in rows),
           "form_vs_oracle_max_abs_dp": max(r["form_vs_oracle_max_abs_dp"] for r in rows),
           "pass": bool(hs <= GATE["h_sel_max_abs"] and pm <= GATE["probs_max_abs"] and all(r["argmax_equal"] for r in rows)
                        and all(r["finite"] for r in rows)),
           "by_bucket": by_bucket,
           "seconds_wall": round(time.time() - t0, 1),
           "shard_seconds": [(p["seconds_stock"], p["seconds_form"], p["stock_from_cache"]) for p in parts],
           "torch_threads": 1, "shards": a.shards, "concurrency": conc, "buckets": BUCKETS, "stock": "kev_qwen35_patch._rank4_chunk_gated_delta_rule, no rewrite",
           "top10_by_probs": sorted(rows, key=lambda r: -r["probs_max_abs"])[:10],
           "top10_by_h_sel": sorted(rows, key=lambda r: -r["h_sel_max_abs"])[:10], "rows": rows}
    dump_json(out, doc)
    print(json.dumps({k: v for k, v in doc.items() if k not in ("rows", "top10_by_probs", "top10_by_h_sel", "apply",
                                                                "load_info")}, indent=1))


if __name__ == "__main__":
    main()
