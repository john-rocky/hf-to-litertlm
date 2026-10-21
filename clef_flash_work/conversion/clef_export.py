#!/usr/bin/env python3
"""Round 1 step 8: export ClefRowPrefill (one L, float32) by litert-torch's lightweight path, then the static scan.

    TMPDIR=<K>/tmp ~/venvs/lt094dev/bin/python scripts/clef_export.py --L 1024 [--threshold 100000000]
        -> out/r1/clef_rowprefill_L1024_fp32.tflite, results/opscan_L1024_fp32.json, results/export_L1024_fp32.json
           (refuses to overwrite any of them)

1. Guard: results/score_torch_L1024.json must say line_pass (the torch module on the reference runs, the author's head).
2. RESOURCE_CONSTANT_NUMEL_THRESHOLD = --threshold (bytes: litert-torch 0.9.4 compares numel * element_size; the 9B
   text precedent's 100000000, convert_qwen35_hybrid_lw.py) is set before litert_torch is imported.
3. The model (clef_graph.load_text_model + the form R64+sp+ec+dd+vs6), the graph at L, the sample = own_t01's row
   (embeds / valid / cos / sin from the NumPy host); the graph object is run once on it and compared with the torch
   check's hidden of the same run (cache/r1/torch/own_t01__text.npy).
4. litert_torch.convert(graph, sample_kwargs={embeds, valid, cos, sin}, lightweight_conversion=True,
   runtime_constant_folding=False) -> .export(path). runtime_constant_folding defaults to True with the lightweight
   path, and that stage fails in every local litert-converter (0.3.1 .. 0.6.0.dev20261016): `ModuleNotFoundError: No
   module named 'litert_converter.mlir.dialects.tfl'` (the wheel has no tfl dialect module; scripts/tiny_lw.py
   --mode lw, logs/tiny_lw_L128_lw.log). On the 4-layer rehearsal the lightweight file without it has 0
   all-constant-input ops, every FC weight a constant, the same 781 ops as the default path and the same CPU
   numbers (results/tiny_lw_L128_lw_norcf_t30000.json vs ..._plain_t30000.json); the acceptance below is what would
   show a fold left undone. --runtime-constant-folding on restores the default (it fails here).
   Memory (ru_maxrss, phys_footprint and its lifetime maximum from proc_pid_rusage) after the load, the convert and the
   write; RSS sampled every 20 s; seconds of each part; TMPDIR's content listed after the write.
5. Scan (scripts/clef_scan.py): the acceptance = all-constant-input ops 0, every FULLY_CONNECTED weight a constant with
   data, the file size within [fp32 bytes of the text stack, + 1 % + 64 MB] (pre-registered: the 27.7 GB of float32
   weights + the mask and kernel constants), BROADCAST_TO / INT64 / CUSTOM / rank > 4 = 0, signature = the contract.
   status = EXPORTED only when all hold; otherwise the failing checks are listed (the round's stop rule decides)."""
import argparse
import ctypes
import importlib.metadata
import json
import os
import resource
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

K = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(K / "scripts"))

_RU_FIELDS = (
    "ri_user_time", "ri_system_time", "ri_pkg_idle_wkups", "ri_interrupt_wkups", "ri_pageins", "ri_wired_size",
    "ri_resident_size", "ri_phys_footprint", "ri_proc_start_abstime", "ri_proc_exit_abstime", "ri_child_user_time",
    "ri_child_system_time", "ri_child_pkg_idle_wkups", "ri_child_interrupt_wkups", "ri_child_pageins",
    "ri_child_elapsed_abstime", "ri_diskio_bytesread", "ri_diskio_byteswritten", "ri_cpu_time_qos_default",
    "ri_cpu_time_qos_maintenance", "ri_cpu_time_qos_background", "ri_cpu_time_qos_utility", "ri_cpu_time_qos_legacy",
    "ri_cpu_time_qos_user_initiated", "ri_cpu_time_qos_user_interactive", "ri_billed_system_time",
    "ri_serviced_system_time", "ri_logical_writes", "ri_lifetime_max_phys_footprint", "ri_instructions", "ri_cycles",
    "ri_billed_energy", "ri_serviced_energy", "ri_interval_max_phys_footprint", "ri_runnable_time")


class _RUsageInfoV4(ctypes.Structure):     # every field of v4 (a shorter struct lets the kernel write past it)
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [(n, ctypes.c_uint64) for n in _RU_FIELDS]


def memory() -> dict:
    """kev_work/scripts/r3_common.memory: ru_maxrss, phys_footprint, its lifetime max, resident size (bytes)."""
    doc = {"ru_maxrss": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)}
    info = _RUsageInfoV4()
    if ctypes.CDLL("/usr/lib/libproc.dylib").proc_pid_rusage(os.getpid(), 4, ctypes.byref(info)) == 0:
        doc.update(phys_footprint=int(info.ri_phys_footprint),
                   lifetime_max_phys_footprint=int(info.ri_lifetime_max_phys_footprint),
                   resident_size=int(info.ri_resident_size))
    return doc


def rss_sampler(stop, samples, every=20):
    pid = os.getpid()
    while not stop.wait(every):
        try:
            kb = int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)], text=True).strip())
        except Exception:
            continue
        samples.append((round(time.time(), 1), kb * 1024))
        print(json.dumps({"rss_gib": round(kb / 2 ** 20, 2), "t": time.strftime("%H:%M:%S")}), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=1024)
    ap.add_argument("--threshold", type=int, default=100_000_000)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--runtime-constant-folding", choices=["off", "on"], default="off")
    a = ap.parse_args()
    L = a.L
    rcf = a.runtime_constant_folding == "on"
    path = K / f"out/r1/clef_rowprefill_L{L}_fp32.tflite"
    rec_path = K / f"results/export_L{L}_fp32.json"
    scan_path = K / f"results/opscan_L{L}_fp32.json"
    for p in (path, rec_path, scan_path):
        assert not p.exists(), f"never overwrite {p}"
    guard_path = K / f"results/score_torch_L{L}.json"
    guard = json.loads(guard_path.read_text())
    assert guard["summary"].get("line_pass") is True, f"torch check did not pass: {guard_path.name}"
    os.environ["RESOURCE_CONSTANT_NUMEL_THRESHOLD"] = str(a.threshold)
    path.parent.mkdir(parents=True, exist_ok=True)
    t_wall = time.perf_counter()
    import torch
    torch.set_num_threads(a.threads)
    torch.set_grad_enabled(False)
    import clef_graph as G
    import clef_host as H
    record = {"L": L, "form": G.FORM, "status": "RUNNING", "file": str(path.relative_to(K)),
              "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "tmpdir": os.environ.get("TMPDIR"),
              "lightweight_conversion": True, "runtime_constant_folding": rcf,
              "resource_constant_numel_threshold_bytes": a.threshold,
              "guard": {"file": str(guard_path.relative_to(K)), "summary": guard["summary"]},
              "memory_before_load": memory(),
              "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "litert-torch",
                                                                      "litert-converter", "ai-edge-litert")}}
    model, info = G.load_text_model()
    record["load_info"] = info
    record["form_apply"] = G.apply_form(model)
    graph = G.graph(model, L)
    ref = json.loads((K / "ref/records_ref.json").read_text())
    r0 = next(r for r in ref["rows"] if r["id"] == "own_t01" and r["arm"] == "text")
    table = H.EmbedTable(G.SNAP)
    embeds, valid, cos, sin = H.row_inputs(r0["ids"], L, table)
    sample = {"embeds": torch.from_numpy(embeds), "valid": torch.from_numpy(valid), "cos": torch.from_numpy(cos),
              "sin": torch.from_numpy(sin)}
    h = graph(**sample)["hidden"][0, :r0["tokens"]].numpy()
    prev = np.load(K / "cache/r1/torch/own_t01__text.npy")
    record["sample"] = {"id": "own_t01", "tokens": r0["tokens"], "finite": bool(np.isfinite(h).all()),
                        "max_abs_vs_torch_check": float(np.abs(h - prev).max()), "bit_equal_torch_check": bool(np.array_equal(h, prev)),
                        "shapes": {k: list(v.shape) for k, v in sample.items()}}
    record["memory_after_load"] = memory()
    record["load_and_check_seconds"] = round(time.perf_counter() - t_wall, 1)
    print(json.dumps({"sample": record["sample"], "memory_after_load": record["memory_after_load"]}), flush=True)
    stop, samples = threading.Event(), []
    sampler = threading.Thread(target=rss_sampler, args=(stop, samples), daemon=True)
    sampler.start()
    import litert_torch
    t0 = time.perf_counter()
    lrt = litert_torch.convert(graph, sample_kwargs=sample, lightweight_conversion=True, runtime_constant_folding=rcf)
    record["convert_seconds"] = round(time.perf_counter() - t0, 1)
    record["memory_after_convert"] = memory()
    print(json.dumps({"convert_seconds": record["convert_seconds"], "memory": record["memory_after_convert"]}), flush=True)
    t1 = time.perf_counter()
    lrt.export(str(path))
    record["write_seconds"] = round(time.perf_counter() - t1, 1)
    record["memory_after_write"] = memory()
    stop.set()
    record["rss_samples_bytes"] = samples
    record["peak_rss_bytes_getrusage"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    tmp = Path(os.environ.get("TMPDIR", "/nonexistent"))
    record["tmpdir_after_write"] = ([{"name": p.name, "bytes": p.stat().st_size if p.is_file() else None}
                                     for p in sorted(tmp.iterdir())][:50] if tmp.is_dir() else None)
    del lrt
    from clef_scan import full_scan
    t2 = time.perf_counter()
    s = full_scan(path)
    record["scan_seconds"] = round(time.perf_counter() - t2, 1)
    scan_path.write_text(json.dumps(s, indent=1) + "\n")
    lw = s["lightweight"]
    sig = s["signatures"]
    want_in = sorted([("embeds", [1, L, G.HIDDEN], "FLOAT32"), ("valid", [1, L], "FLOAT32"), ("cos", [1, L, 64], "FLOAT32"),
                      ("sin", [1, L, 64], "FLOAT32")])
    sig_ok = (len(sig) == 1 and sorted((i["name"], i["shape"], i["dtype"]) for i in sig[0]["inputs"]) == want_in
              and [(o["name"], o["shape"], o["dtype"]) for o in sig[0]["outputs"]] == [("hidden", [1, L, G.HIDDEN], "FLOAT32")])
    lo, hi = info["fp32_bytes"], int(info["fp32_bytes"] * 1.01) + 64 * 2 ** 20
    checks = {"all_constant_input_ops_0": lw["all_constant_input_ops"] == 0,
              "fc_weights_all_constant_with_data": lw["fc_weight_not_constant"] == 0 and lw["fc_weight_constant_without_data"] == 0,
              "size_as_expected": lo <= s["bytes"] <= hi,
              "broadcast_to_0": s["forbidden_counts"]["BROADCAST_TO"] == 0, "int64_0": s["int64_tensor_count"] == 0,
              "custom_0": s["custom_op_count"] == 0, "rank_gt4_0": s["rank_gt4_tensor_count"] == 0,
              "stablehlo_0": not s["stablehlo_ops"], "signature_matches_contract": sig_ok}
    record.update(status="EXPORTED" if all(checks.values()) else "ACCEPTANCE_FAIL", checks=checks,
                  failing=[k for k, v in checks.items() if not v], expected_bytes_range=[lo, hi],
                  bytes=s["bytes"], sha256=s["sha256"], signatures=sig, operator_count=s["operator_count"],
                  op_histogram=s["op_histogram"], fully_connected_count=s["fully_connected_count"],
                  fc_weight_dtypes=lw["fc_weight_dtypes"], constant_bytes=lw["constant_bytes"],
                  constant_bytes_by_dtype=lw["constant_bytes_by_dtype"], all_constant_input_ops=lw["all_constant_input_ops"],
                  all_constant_input_op_names=lw["all_constant_input_op_names"], forbidden_counts=s["forbidden_counts"],
                  int64_tensor_count=s["int64_tensor_count"], rank_gt4_tensor_count=s["rank_gt4_tensor_count"],
                  custom_op_count=s["custom_op_count"], stablehlo_ops=s["stablehlo_ops"],
                  tensor_rank_histogram=s["tensor_rank_histogram"], pad_count=s["pad_count"],
                  batch_matmul_count=s["batch_matmul_count"], batch_matmul_all_rank4=s["batch_matmul_all_rank4"],
                  gather_count=s["gather_count"], embedding_lookup_count=s["embedding_lookup_count"],
                  opscan=str(scan_path.relative_to(K)), contract=G.contract(L),
                  seconds_wall=round(time.perf_counter() - t_wall, 1), finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    rec_path.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({k: record[k] for k in ("status", "failing", "bytes", "sha256", "operator_count", "fully_connected_count",
                                             "all_constant_input_ops", "convert_seconds", "write_seconds")}), flush=True)
    print(json.dumps({"peak_rss_gb": round(record["peak_rss_bytes_getrusage"] / 1e9, 2),
                      "lifetime_max_phys_footprint_gb": round(record["memory_after_write"].get("lifetime_max_phys_footprint", 0) / 1e9, 2)}),
          flush=True)
    return 0 if record["status"] == "EXPORTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
