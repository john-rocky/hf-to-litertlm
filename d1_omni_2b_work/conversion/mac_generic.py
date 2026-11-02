"""The test app's generic mode on the Mac (ai-edge-litert CompiledModel, lrt_backend.py's options): one generic rows
file through one signature of one graph, every output's whole tensor appended per row to <out-dir>/out_<tag>.f32 in the
app's byte layout, and a report <out-dir>/<tag>.json with the app's keys that the replay and make_rows.py prep read
(status, rows_file, signature, summary.count, out_bytes). `--sharing` = GpuOptions.constant_tensor_sharing (the app's
gpu_sharing=1) through lrt_backend.MultiGraph (one CompiledModel, the rows file's signature; the seven-signature decide
file is called by the bucket's signature, as on the phone).

  $EXPORT conversion/mac_generic.py --graph out/graphs/<file> --rows out/rows/<rows>.json \
      --tag <tag> --out-dir out/mac/<run> --accel gpu --precision fp32 [--sig audio] [--sharing]

It stands in for the phone in the replay's own check (Metal fp32 through the replay = the Mac gate's Metal fp32
numbers), so the replay's path and the phone's own error are told apart, and leg_check_omni.py compares each phone
leg with it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--accel", choices=("gpu", "cpu"), default="gpu")
    ap.add_argument("--precision", choices=("fp32", "default"), default="fp32")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--sig", default=None, help="the signature (default: the rows file's)")
    ap.add_argument("--sharing", action="store_true", help="GpuOptions.constant_tensor_sharing (the app's gpu_sharing=1)")
    a = ap.parse_args()
    from lrt_backend import MultiGraph

    doc = json.load(open(a.rows))
    rows_dir = os.path.dirname(os.path.abspath(a.rows))
    sig = a.sig or doc["signature"]
    os.makedirs(a.out_dir, exist_ok=True)
    rep = {"graph": os.path.basename(a.graph), "accel": a.accel, "precision": a.precision,
           "rows_file": os.path.basename(a.rows), "mode": "generic", "signature": sig, "sharing": a.sharing,
           "stand_in": "Mac CompiledModel (conversion/mac_generic.py)", "litert": None}
    try:
        import ai_edge_litert

        rep["litert"] = getattr(ai_edge_litert, "__version__", None)
    except Exception:
        pass
    t0 = time.time()
    g = MultiGraph(a.graph, a.accel, a.precision, a.threads, a.sharing)
    rep["compile_ms"] = round(g.load_s * 1000, 1)
    rep["fully_accelerated"] = g.fully_accelerated
    if sig not in g.signatures:
        raise SystemExit(f"{a.graph}: no signature {sig} (the file has {sorted(g.signatures)})")
    s = g.signatures[sig]
    in_names, out_names = list(s["inputs"]), list(s["outputs"])
    in_shapes = {n: list(g.model.get_input_tensor_details(sig)[n]["shape"]) for n in in_names}
    declared_in = [i["name"] for i in doc["inputs"]]
    declared_out = [o["name"] for o in doc["outputs"]]
    assert sorted(declared_in) == sorted(in_names), (declared_in, in_names)
    assert sorted(declared_out) == sorted(out_names), (declared_out, out_names)
    for spec in doc["inputs"]:
        assert spec["shape"] == in_shapes[spec["name"]], (spec, in_shapes)
    mms = {sp["name"]: (np.memmap(os.path.join(rows_dir, sp["file"]), dtype="<f4", mode="r"), int(np.prod(sp["shape"])),
                        sp["shape"]) for sp in doc["inputs"]}
    out_path = os.path.join(a.out_dir, f"out_{a.tag}.f32")
    recs, nonfinite_rows, nbytes = [], 0, 0
    with open(out_path + ".tmp", "wb") as fh:
        for r in doc["rows"]:
            arrays = {n: np.asarray(mm[r["index"] * n_:(r["index"] + 1) * n_]).reshape(shape)
                      for n, (mm, n_, shape) in mms.items()}
            t1 = time.perf_counter()
            out = g(sig, **arrays)
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
    g.close()
    rep.update(status="DONE", out_file=os.path.basename(out_path), out_bytes=nbytes,
               summary={"count": len(recs), "finite_rows": len(recs) - nonfinite_rows, "nonfinite_rows": nonfinite_rows},
               rows=recs, wall_s=round(time.time() - t0, 2))
    json.dump(rep, open(os.path.join(a.out_dir, f"{a.tag}.json"), "w"), indent=1)
    print(json.dumps({k: rep[k] for k in ("graph", "accel", "precision", "signature", "sharing", "compile_ms",
                                           "fully_accelerated", "out_bytes", "summary", "wall_s")}))


if __name__ == "__main__":
    main()
