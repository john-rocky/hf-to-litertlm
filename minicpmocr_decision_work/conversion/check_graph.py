#!/usr/bin/env python3
"""A graph file against the author's numbers (out/fix/, conversion/fixtures_author.py), the host not in the path: the author's own
inputs go in, padded the contract's way. One file a process; one accelerator a line.

    <python> conversion/check_graph.py <work> decide --file out/graphs/decide_L256_fp16fc.tflite --size 256 [--acc cpu,gpu_f32,gpu_default]
    <python> conversion/check_graph.py <work> tower  --file out/graphs/tower_N400_fp16fc.tflite --size 400

decide rows: rel_rms over the T rows, hidden[T - 1] cos and max|d|, |d logits|, |d prob_1|. tower rows: rel_rms, cos row by row (min), max|d|.
Records: every fixture that fits. Writes results/check_<file stem>.json, runtime logs in logs/."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("work", type=Path, help="the work directory")
ap.add_argument("what", choices=("decide", "tower"))
ap.add_argument("--file", required=True)
ap.add_argument("--size", type=int, required=True)
ap.add_argument("--acc", default="cpu,gpu_f32")
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--records", default="", help="comma list of record tags")
ap.add_argument("--limit", type=int, default=0, help="at most this many fixtures, largest first")
ap.add_argument("--suffix", default="")
ap.add_argument("--signature", default="", help="one signature of a multi-signature file; default: the what")
a = ap.parse_args()
WORK = a.work.resolve()
sys.path.insert(0, str(WORK / "host"))
sys.path.insert(0, str(WORK / "conversion"))
import lrt  # noqa: E402
from decide_host import formulas as F  # noqa: E402
from rt_log import PARTITIONED, REPLACING, runtime_log_verbose  # noqa: E402

path = WORK / a.file if not Path(a.file).is_absolute() else Path(a.file)
fix = []
if a.what == "decide":
    W = F.Bf16Table(WORK / "snapshot/model.safetensors", "score.weight").rows()
    for p in sorted((WORK / "out/fix").glob("rec_*.npz")):
        z = np.load(p)
        if z["ids"].shape[0] <= a.size:
            fix.append((p.stem[4:], z["ids"].shape[0], z))
else:
    for p in sorted((WORK / "out/fix").glob("pic_*.npz")):
        z = np.load(p)
        if z["pixel_values"].shape[0] <= a.size:
            fix.append((p.stem[4:], z["pixel_values"].shape[0], z))
fix.sort(key=lambda x: -x[1])
if a.records:
    want = set(a.records.split(","))
    fix = [f for f in fix if f[0] in want]
if a.limit:
    fix = fix[:a.limit]
assert fix, "no fixture fits"


def decide_feed(z, L, head=False):
    T = z["ids"].shape[0]
    e = np.zeros((1, L, 2048), np.float32)
    e[0, :T] = z["inputs_embeds"]
    v = np.zeros((1, L), np.float32)
    v[0, :T] = 1.0
    c, s = F.mrope_tables(F.padded_planes(z["pos3"].astype(np.int64), L))
    c[0, :T], s[0, :T] = z["cos"], z["sin"]
    f = {"embeds": e, "valid": v, "cos": c, "sin": s}
    if head:
        f["pick"] = np.zeros((1, L), np.float32)
        f["pick"][0, T - 1] = 1.0
    return f


def tower_feed(z, N):
    n = z["pixel_values"].shape[0]
    f = {"patches": np.zeros((1, N, 1536), np.float32), "valid": np.zeros((1, N), np.float32), "pos": np.zeros((1, N, 1024), np.float32),
         "cos": np.ones((1, N, 64), np.float32), "sin": np.zeros((1, N, 64), np.float32)}
    f["patches"][0, :n] = z["pixel_values"]
    f["valid"][0, :n] = 1.0
    f["pos"][0, :n] = z["pos"]
    f["cos"][0, :n], f["sin"][0, :n] = z["cos"], z["sin"]
    return f


doc = {"file": str(path.relative_to(WORK)), "bytes": path.stat().st_size, "what": a.what, "size": a.size, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "runs": {}}
for acc in a.acc.split(","):
    log = WORK / f"logs/check_{path.stem}{a.suffix}_{acc}.runtime.log"
    lf = open(log, "w")
    saved = os.dup(2)
    os.dup2(lf.fileno(), 2)
    item = {"records": []}
    try:
        runtime_log_verbose()
        g = lrt.Graph(path, a.signature or a.what, acc, threads=a.threads)
        item.update(load_s=round(g.load_s, 2), buffer_s=round(g.buffer_s, 2), fully_accelerated=g.fully_accelerated, memory_after_load=lrt.memory())
        for name, n, z in fix:
            t0 = time.perf_counter()
            if a.what == "decide" and "logits" in g.out_names:
                lg = g(**decide_feed(z, a.size, head=True))["logits"][0]
                e_ = np.exp(lg - lg.max()).astype(np.float32)
                p1 = float((e_ / e_.sum()).astype(np.float32)[1])
                rec = {"record": name, "T": n, "L": a.size, "run_ms": round(g.last_ms, 1), "prob_1": p1, "prob_author": float(z["prob_1"]), "d_prob": abs(p1 - float(z["prob_1"])),
                       "d_logit": float(np.abs(lg - z["logits"]).max()), "read_ms": round(g.read_ms, 2), "write_ms": round(g.write_ms, 2)}
            elif a.what == "decide":
                h = g(**decide_feed(z, a.size))["hidden"][0]
                want = z["hidden"]
                st = lrt.stats(h[:n], want)
                pool = lrt.stats(h[n - 1], want[n - 1])
                logits, p1 = F.softmax_prob1(h[n - 1], W)
                rec = {"record": name, "T": n, "L": a.size, "run_ms": round(g.last_ms, 1), "rel_rms": st["rel_rms"], "max_abs": st["max_abs"], "finite": st["finite"],
                       "pooled_cos": pool["cos"], "pooled_max_abs": pool["max_abs"], "prob_1": p1, "prob_author": float(z["prob_1"]),
                       "d_prob": abs(p1 - float(z["prob_1"])), "d_logit": float(np.abs(logits - z["logits"]).max()),
                       "nonfinite_pad_rows": int((~np.isfinite(h[n:])).sum()), "read_ms": round(g.read_ms, 2), "write_ms": round(g.write_ms, 2)}
            else:
                r = g(**tower_feed(z, a.size))["rows"][0]
                want = z["rows"]
                m = n // 4
                st = lrt.stats(r[:m], want)
                gg, ww = r[:m].astype(np.float64), want.astype(np.float64)
                row_cos = (gg * ww).sum(1) / np.maximum(np.linalg.norm(gg, axis=1) * np.linalg.norm(ww, axis=1), 1e-300)
                rec = {"record": name, "n": n, "N": a.size, "run_ms": round(g.last_ms, 1), "rel_rms": st["rel_rms"], "max_abs": st["max_abs"], "finite": st["finite"],
                       "cos": st["cos"], "row_cos_min": float(row_cos.min()), "padded": n < a.size, "nonfinite_pad_rows": int((~np.isfinite(r[m:])).sum())}
            rec["seconds"] = round(time.perf_counter() - t0, 2)
            item["records"].append(rec)
            print(json.dumps({"acc": acc, **{k: (float(f"{v:.4g}") if isinstance(v, float) else v) for k, v in rec.items()}}), file=sys.stdout, flush=True)
        item["memory_end"] = lrt.memory()
        g.close()
        item["status"] = "OK"
    except BaseException as e:   # a refused compile is a result
        item.update(status="FAIL", error=f"{type(e).__name__}: {e}"[:600])
        print(json.dumps({"acc": acc, "status": "FAIL", "error": item["error"]}), flush=True)
    finally:
        sys.stderr.flush()
        os.dup2(saved, 2)
        lf.close()
    txt = log.read_text(errors="replace")
    item["replacing"] = [{"delegated": int(m[0]), "total": int(m[1]), "partitions": int(m[3])} for m in REPLACING.findall(txt)]
    item["partitioned"] = [{"selected": int(m[1]), "total": int(m[2]), "partitions": int(m[3])} for m in PARTITIONED.findall(txt)]
    item["errors"] = [ln for ln in txt.splitlines() if ("ERROR" in ln or "No support" in ln or "not supported" in ln.lower())][:8]
    doc["runs"][acc] = item
    print("REPLACING", acc, item["replacing"][:2], item["errors"][:2], flush=True)
out = WORK / f"results/check_{path.stem}{a.suffix}.json"
out.write_text(json.dumps(doc, indent=1) + "\n")
print("CHECK_DONE", out.name, flush=True)
