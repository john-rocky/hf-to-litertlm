"""A phone run's work on the Mac (ai-edge-litert CompiledModel, the options of host/decide_host/litert_backends.py): rows files through one graph, one compile, every output's whole tensor
appended per row to <out-dir>/out_<tag>.f32, little-endian float32, and a report <out-dir>/<tag>.json with the keys that the replay and make_rows.py read (status, rows_file, signature,
summary.count, out_bytes, per-row times).

  $EXPORT conversion/mac_generic.py --graph out/graphs/<file> --leg <tag>=<rows.json> [--leg ...] --out-dir out/mac --acc gpu_f32 [--sharing] [--sig decide] [--threads 4]

It stands in for the phone in the replay's own check (Metal float32 through the replay = the Metal float32 rows of run_set.py), so the replay's path and a phone's own error are told apart.
A run whose report says DONE is skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "host"))


def run_leg(g, sig_used, tag, rows_path, out_dir, rep_base):
    doc = json.load(open(rows_path))
    rows_dir = os.path.dirname(os.path.abspath(rows_path))
    declared_in = [i["name"] for i in doc["inputs"]]
    declared_out = [o["name"] for o in doc["outputs"]]
    assert sorted(declared_in) == sorted(g.in_names), (declared_in, g.in_names)
    assert sorted(declared_out) == sorted(g.out_names), (declared_out, g.out_names)
    for spec in doc["inputs"]:
        assert spec["shape"] == g.in_shapes[spec["name"]], (spec, g.in_shapes)
    mms = {s["name"]: (np.memmap(os.path.join(rows_dir, s["file"]), dtype="<f4", mode="r"), int(np.prod(s["shape"])), s["shape"]) for s in doc["inputs"]}
    out_path = os.path.join(out_dir, f"out_{tag}.f32")
    recs, nonfinite_rows, nbytes = [], 0, 0
    t0 = time.time()
    with open(out_path + ".tmp", "wb") as fh:
        for r in doc["rows"]:
            arrays = {n: np.asarray(mm[r["index"] * n_:(r["index"] + 1) * n_]).reshape(shape) for n, (mm, n_, shape) in mms.items()}
            t1 = time.perf_counter()
            out = g(**arrays)
            dt = (time.perf_counter() - t1) * 1000
            bad = 0
            for o in doc["outputs"]:
                v = np.ascontiguousarray(out[o["name"]], dtype="<f4")
                bad += int((~np.isfinite(v)).sum())
                b = v.tobytes()
                fh.write(b)
                nbytes += len(b)
            nonfinite_rows += int(bad > 0)
            recs.append({"key": r["key"], "index": r["index"], "write_run_read_ms": round(dt, 3), "nonfinite": bad})
    os.replace(out_path + ".tmp", out_path)
    rep = dict(rep_base, rows_file=os.path.basename(rows_path), signature=sig_used, status="DONE", out_file=os.path.basename(out_path), out_bytes=nbytes,
               summary={"count": len(recs), "finite_rows": len(recs) - nonfinite_rows, "nonfinite_rows": nonfinite_rows}, rows=recs, wall_s=round(time.time() - t0, 2))
    with open(os.path.join(out_dir, f"{tag}.json.tmp"), "w") as fh:
        json.dump(rep, fh, indent=1)
    os.replace(os.path.join(out_dir, f"{tag}.json.tmp"), os.path.join(out_dir, f"{tag}.json"))
    print(json.dumps({"tag": tag, "graph": rep["graph"], "acc": rep["acc"], "signature": sig_used, "compile_ms": rep["compile_ms"], "fully_accelerated": rep["fully_accelerated"],
                      "out_bytes": nbytes, "summary": rep["summary"], "wall_s": rep["wall_s"]}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--leg", action="append", required=True, help="<tag>=<rows.json>")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--acc", choices=("gpu_f32", "gpu_default", "cpu"), default="gpu_f32")
    ap.add_argument("--sharing", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--sig", default=None, help="the signature (default: each rows file's)")
    a = ap.parse_args()
    from decide_host.litert_backends import Graph

    os.makedirs(a.out_dir, exist_ok=True)
    legs = [x.split("=", 1) for x in a.leg]
    todo = []
    for tag, rows in legs:
        p = os.path.join(a.out_dir, f"{tag}.json")
        if os.path.exists(p) and json.load(open(p)).get("status") == "DONE":
            print(f"{tag}: already DONE", flush=True)
            continue
        todo.append((tag, rows))
    if not todo:
        return
    import ai_edge_litert

    base = {"graph": os.path.basename(a.graph), "acc": a.acc, "sharing": a.sharing, "threads": a.threads, "mode": "generic",
            "stand_in": "Mac CompiledModel (conversion/mac_generic.py)", "litert": getattr(ai_edge_litert, "__version__", None)}
    # one compile per graph file; a multi-signature file gets a Graph per signature on the same options (one process, one signature after another)
    graphs = {}
    for tag, rows in todo:
        sig = a.sig or json.load(open(rows))["signature"]
        if sig not in graphs:
            for old in graphs.values():
                old.close()
            graphs = {sig: Graph(a.graph, sig, a.acc, a.threads, a.sharing)}
            base["compile_ms"] = round(graphs[sig].load_s * 1000, 1)
            base["fully_accelerated"] = graphs[sig].fully_accelerated
        run_leg(graphs[sig], sig, tag, rows, a.out_dir, base)
    for g in graphs.values():
        g.close()


if __name__ == "__main__":
    main()
