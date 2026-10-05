"""Proof that an r17_kernel form does not change the fp32 result of Kev-0.8B (torch, all 402 questions): r14_torch_parity.py
with the forms through r17_kernel.apply.

    python r17_torch_parity.py --form R64+sp+ec+dd+vs6+bk1024@in_proj_z.q_proj [--shards 8]
        -> results/r17_torch_parity_R64-sp-ec-dd-vs6-bk1024_in_proj_z.q_proj.json   (never overwritten)

Per question: the patched torch graph (kev_graph.KevPrefill, fp32, torch 1 thread per shard) on the row padded to the
smallest bucket L in (64, 128, 256, 512, 1024, 2048) that fits it, as with the loop kernel (no rewrite) and with
r17_kernel.apply(model, form); h_sel max |diff|, hidden max |diff| at the real positions, probs max |diff| (host head =
r2_common.Head), argmax equality. Gate: h_sel <= 1e-4, probs <= 1e-5, argmax 402/402, all finite ("pass"); a second
reading allows h_sel up to 1.3e-4 for forms that change the summation order ("pass_r17"). The loop-kernel rows are
computed once per shard layout and cached (cache/r14/stock_rows_<i>of<n>.npz when present, else cache/r17). At most 4
shards start while the file named by the environment variable KEV_GPU_LOCK (another job timing the GPU, when set) holds
an owner line with "timing"."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from r2_common import K, Head, dump_json, load_oracle, qkey, select

BUCKETS = (64, 128, 256, 512, 1024, 2048)
GATE = {"h_sel_max_abs": 1e-4, "probs_max_abs": 1e-5}
# a form that changes the summation order may move h_sel up to 1.3e-4 (a BATCH_MATMUL chain form reached 1.27e-4)
GATE_R17 = {"h_sel_max_abs": 1.3e-4, "probs_max_abs": 1e-5}


def ftag(form):
    return form.replace("+", "-").replace("@", "_").replace(",", ".")


def worker(form, i, n):
    import torch
    torch.set_num_threads(1)
    import r17_kernel as R
    from kev_graph import KevPrefill, load_text_model, row_inputs
    out = K / f"cache/r17/parity_{ftag(form)}_{i}of{n}.json"
    model, _ = load_text_model()
    oracle = load_oracle()
    qs = oracle["questions"][i::n]
    head = Head()
    stock_path = K / f"cache/r14/stock_rows_{i}of{n}.npz"          # r14_torch_parity.py's cache, read only
    if not stock_path.exists():
        stock_path = K / f"cache/r17/stock_rows_{i}of{n}.npz"
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
    else:
        stock = run_all()
        np.savez(stock_path, **stock)
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
        rows.append({"key": k, "row_len": q["row_len"], "finite": bool(np.isfinite(hr).all()),
                     "hidden_max_abs": float(np.abs(hr - hs).max()), "h_sel_max_abs": float(np.abs(sr - ss).max()),
                     "probs_max_abs": float(np.abs(pr.astype(np.float64) - ps).max()),
                     "argmax_equal": int(np.argmax(pr)) == int(np.argmax(ps)),
                     "stock_vs_oracle_max_abs_dp": float(np.abs(ps - po).max()),
                     "form_vs_oracle_max_abs_dp": float(np.abs(pr - po).max())})
    dump_json(out, {"form": form, "shard": f"{i}/{n}", "apply": {k: v for k, v in info.items() if k != "tokens"},
                    "seconds_stock": round(t1 - t0, 1), "seconds_form": round(time.perf_counter() - t1, 1),
                    "rows": rows}, overwrite=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--form", required=True)
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--worker", default="", help="internal: i/n")
    a = ap.parse_args()
    if a.worker:
        i, n = (int(x) for x in a.worker.split("/"))
        worker(a.form, i, n)
        return
    out = K / f"results/r17_torch_parity_{ftag(a.form)}.json"
    assert not out.exists(), out
    (K / "cache/r17").mkdir(parents=True, exist_ok=True)
    try:
        gl = Path(os.environ["KEV_GPU_LOCK"]).read_text().strip() if os.environ.get("KEV_GPU_LOCK") else ""
    except OSError:
        gl = ""
    if "timing" in gl:
        a.shards = min(a.shards, 4)
    t0 = time.time()
    env = dict(os.environ, OMP_NUM_THREADS="1")
    procs = [subprocess.Popen([sys.executable, __file__, "--form", a.form, "--worker", f"{i}/{a.shards}"],
                              cwd=str(K / "scripts"), env=env,
                              stdout=open(K / f"logs/r17_parity_{ftag(a.form)}_{i}.log", "w"), stderr=subprocess.STDOUT)
             for i in range(a.shards)]
    rcs = [p.wait() for p in procs]
    assert all(rc == 0 for rc in rcs), rcs
    parts = [json.loads((K / f"cache/r17/parity_{ftag(a.form)}_{i}of{a.shards}.json").read_text())
             for i in range(a.shards)]
    rows = sorted((r for p in parts for r in p["rows"]), key=lambda r: r["key"])
    hs = max(r["h_sel_max_abs"] for r in rows)
    pm = max(r["probs_max_abs"] for r in rows)
    doc = {"step": "form vs the loop-kernel graph in torch fp32 (all oracle questions, each at its smallest bucket)",
           "gpu_lock_at_launch": gl, "shards": a.shards,
           "form": a.form, "apply": parts[0]["apply"], "questions": len(rows), "gate": GATE,
           "h_sel_max_abs": hs, "probs_max_abs": pm,
           "hidden_max_abs_real_positions": max(r["hidden_max_abs"] for r in rows),
           "argmax_equal": sum(r["argmax_equal"] for r in rows), "all_finite": all(r["finite"] for r in rows),
           "stock_vs_oracle_max_abs_dp": max(r["stock_vs_oracle_max_abs_dp"] for r in rows),
           "form_vs_oracle_max_abs_dp": max(r["form_vs_oracle_max_abs_dp"] for r in rows),
           "pass": bool(hs <= GATE["h_sel_max_abs"] and pm <= GATE["probs_max_abs"] and all(r["argmax_equal"] for r in rows)
                        and all(r["finite"] for r in rows)),
           "gate_r17": GATE_R17,
           "pass_r17": bool(hs <= GATE_R17["h_sel_max_abs"] and pm <= GATE_R17["probs_max_abs"]
                            and all(r["argmax_equal"] for r in rows) and all(r["finite"] for r in rows)),
           "h_sel_over_1e-4": sum(r["h_sel_max_abs"] > 1e-4 for r in rows),
           "seconds_wall": round(time.time() - t0, 1), "shard_seconds": [(p["seconds_stock"], p["seconds_form"]) for p in parts],
           "torch_threads": 1, "buckets": BUCKETS, "stock": "kev_qwen35_patch._rank4_chunk_gated_delta_rule, no rewrite",
           "top10_by_probs": sorted(rows, key=lambda r: -r["probs_max_abs"])[:10], "rows": rows}
    dump_json(out, doc)
    print(json.dumps({k: v for k, v in doc.items() if k not in ("rows", "top10_by_probs", "apply")}, indent=1))


if __name__ == "__main__":
    main()
