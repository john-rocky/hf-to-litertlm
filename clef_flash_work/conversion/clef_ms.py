#!/usr/bin/env python3
"""Round 1 step 10: the ladder row = one CPU call of the fp32 row graph (a full 1,024-token prefill) on the Mac,
inside a measurement window of its own (nothing else heavy runs beside it).

    ~/venvs/lt094dev/bin/python scripts/clef_ms.py --tflite out/r1/clef_rowprefill_L1024_fp32.tflite [--threads 8]
        -> results/ms_cpu_L1024_fp32.json (refuses to overwrite)

CompiledModel.from_file (load + compile, one wall) with CpuOptions(num_threads=8), no weight cache path; the input =
own_t12's ids cycled to 1,024 real tokens (valid all 1: the call computes every position whatever the content), embeds
/ cos / sin from the NumPy host; the inputs are written once. 5 warm-up calls, then 20 timed calls of run_by_name only
(time.perf_counter around the call); median, min, max, every value. Also: the first call's wall, the read-back of
`hidden` after the last call (finite), phys_footprint after compile and its lifetime maximum at the end, the load
average before and after, and whether the measurement lock held this run's label for the whole run."""
import argparse
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

K = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(K / "scripts"))
import clef_host as H  # noqa: E402
from clef_export import memory  # noqa: E402

SNAP = K / "hf/hub/models--Cloudflare--clef-flash/snapshots/f0b7d3cd65296abfe459e6c228b618f5d8b26d2f"
SIG = "serving_default"
LOCK = Path(os.environ.get("MEASUREMENT_LOCK_FILE", "/dev/null"))   # the machine-wide measurement lock (its line is recorded)


def lock_line() -> str:
    try:
        return LOCK.read_text(errors="replace").strip()
    except OSError:
        return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tflite", required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--timed", type=int, default=20)
    a = ap.parse_args()
    out = K / "results/ms_cpu_L1024_fp32.json"
    assert not out.exists(), f"refusing to overwrite {out}"
    path = K / a.tflite
    from ai_edge_litert.compiled_model import CompiledModel, CpuOptions, HardwareAccelerator, Options
    doc = {"what": "round 1 step 10: one CPU call of the fp32 L1024 row graph (1,024-token prefill)", "tflite": a.tflite,
           "tflite_bytes": path.stat().st_size, "ai_edge_litert": importlib.metadata.version("ai-edge-litert"),
           "threads": a.threads, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "loadavg_before": os.getloadavg(),
           "lock_before": lock_line(), "memory_before": memory()}
    cpu = CpuOptions(num_threads=a.threads)
    doc["cpu_options"] = cpu._as_flat_kwargs()
    t0 = time.perf_counter()
    model = CompiledModel.from_file(str(path), options=Options(hardware_accelerators=HardwareAccelerator.CPU, cpu_options=cpu))
    doc["load_and_compile_seconds"] = round(time.perf_counter() - t0, 2)
    doc["memory_after_compile"] = memory()
    in_det, out_det = model.get_input_tensor_details(SIG), model.get_output_tensor_details(SIG)
    L, D = (int(x) for x in list(out_det["hidden"]["shape"])[1:])
    ref = json.loads((K / "ref/records_ref.json").read_text())
    r = next(x for x in ref["rows"] if x["id"] == "own_t12" and x["arm"] == "text")
    ids = (r["ids"] * (L // len(r["ids"]) + 1))[:L]
    embeds, valid, cos, sin = H.row_inputs(ids, L, H.EmbedTable(SNAP))
    ins = {n: model.create_input_buffer_by_name(SIG, n) for n in in_det}
    outs = {n: model.create_output_buffer_by_name(SIG, n) for n in out_det}
    for n, v in (("embeds", embeds), ("valid", valid), ("cos", cos), ("sin", sin)):
        ins[n].write(v)
    doc["input"] = {"record": "own_t12", "record_tokens": r["tokens"], "row_tokens": L, "valid_ones": int(valid.sum())}
    first = None
    warm = []
    for i in range(a.warmup):
        t = time.perf_counter()
        model.run_by_name(SIG, ins, outs)
        warm.append((time.perf_counter() - t) * 1000)
    first = warm[0] if warm else None
    lock_ok = True
    timed = []
    for i in range(a.timed):
        lock_ok &= "clef-r1" in lock_line()
        t = time.perf_counter()
        model.run_by_name(SIG, ins, outs)
        timed.append((time.perf_counter() - t) * 1000)
    h = np.asarray(outs["hidden"].read(L * D, np.float32), dtype=np.float32)
    doc.update(first_call_ms=first, warmup_ms=warm, timed_ms=timed,
               ms={"median": float(np.median(timed)), "min": float(np.min(timed)), "max": float(np.max(timed)),
                   "n": len(timed)},
               hidden_finite=bool(np.isfinite(h).all()), memory_end=memory(), loadavg_after=os.getloadavg(),
               lock_held_by_this_run_every_timed_call=bool(lock_ok), lock_after=lock_line(),
               finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    for b in list(ins.values()) + list(outs.values()):
        try:
            b.destroy()
        except Exception:
            pass
    model.close()
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({k: doc[k] for k in ("load_and_compile_seconds", "first_call_ms", "ms", "hidden_finite",
                                          "lock_held_by_this_run_every_timed_call")}, indent=1))
    print(json.dumps({"phys_footprint_after_compile_gb": round(doc["memory_after_compile"].get("phys_footprint", 0) / 1e9, 2),
                      "lifetime_max_phys_footprint_gb": round(doc["memory_end"].get("lifetime_max_phys_footprint", 0) / 1e9, 2)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
