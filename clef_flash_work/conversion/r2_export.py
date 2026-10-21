#!/usr/bin/env python3
"""Round 2: export ClefRowPrefill (one L, float32) by r1's lightweight recipe (scripts/clef_export.py, unchanged in
every setting), then the static scan and r1's acceptance.

    TMPDIR=<K>/tmp ~/venvs/lt094dev/bin/python scripts/r2_export.py --L 256 [--threshold 100000000] [--nscale]
        -> out/r2/clef_rowprefill_L256[_ns]_fp32.tflite, results/r2_opscan_L256[_ns]_fp32.json,
           results/r2_export_L256[_ns]_fp32.json (refuses to overwrite any of them)

Same as clef_export.py: RESOURCE_CONSTANT_NUMEL_THRESHOLD = --threshold bytes before litert_torch is imported;
clef_graph.load_text_model + the form R64+sp+ec+dd+vs6; litert_torch.convert(graph, sample_kwargs={embeds, valid, cos,
sin}, lightweight_conversion=True, runtime_constant_folding=False) -> .export(path); memory (ru_maxrss, phys_footprint and
its lifetime maximum) after the load, the convert and the write; RSS every 20 s; TMPDIR listed after the write; the scan
(scripts/clef_scan.py) and the 9 checks (all-constant-input ops 0, every FC weight a constant with data, size within
[fp32 bytes of the text stack, + 1 % + 64 MiB], BROADCAST_TO / INT64 / CUSTOM / rank > 4 / StableHLO 0, signature =
the contract) -> status EXPORTED or ACCEPTANCE_FAIL.
Differences (round 2 only): the guard is r1's torch check of the same module and form (results/score_torch_L1024.json
line_pass; the graph is L-independent code), plus with --nscale results/r2_nscale_check_L256.json bit_equal_all; the
sample = the longest fixture text run that fits L (r1 used own_t01 at L1024), whose graph output is compared with the
reference's last_hidden at the real positions (recorded, not asserted); --nscale applies scripts/r2_nscale.py's plan
(results/r2_nscale_plan.json) after the form and names the file `_ns_fp32`."""
import argparse
import importlib.metadata
import json
import os
import resource
import sys
import threading
import time
from pathlib import Path

import numpy as np

K = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(K / "scripts"))
from clef_export import memory, rss_sampler  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--threshold", type=int, default=100_000_000)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--nscale", action="store_true")
    a = ap.parse_args()
    L = a.L
    sfx = "_ns" if a.nscale else ""
    path = K / f"out/r2/clef_rowprefill_L{L}{sfx}_fp32.tflite"
    rec_path = K / f"results/r2_export_L{L}{sfx}_fp32.json"
    scan_path = K / f"results/r2_opscan_L{L}{sfx}_fp32.json"
    for p in (path, rec_path, scan_path):
        assert not p.exists(), f"never overwrite {p}"
    guard_path = K / "results/score_torch_L1024.json"
    guard = json.loads(guard_path.read_text())
    assert guard["summary"].get("line_pass") is True, f"torch check did not pass: {guard_path.name}"
    ns_guard = None
    if a.nscale:
        ns_guard = json.loads((K / "results/r2_nscale_check_L256.json").read_text())
        assert ns_guard["bit_equal_all"] is True, "the pre-scaled graph is not bit-equal in fp32"
    os.environ["RESOURCE_CONSTANT_NUMEL_THRESHOLD"] = str(a.threshold)
    path.parent.mkdir(parents=True, exist_ok=True)
    t_wall = time.perf_counter()
    import torch
    torch.set_num_threads(a.threads)
    torch.set_grad_enabled(False)
    import clef_graph as G
    import clef_host as H
    record = {"L": L, "form": G.FORM + ("+nscale(results/r2_nscale_plan.json)" if a.nscale else ""), "status": "RUNNING",
              "file": str(path.relative_to(K)), "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "tmpdir": os.environ.get("TMPDIR"), "lightweight_conversion": True, "runtime_constant_folding": False,
              "resource_constant_numel_threshold_bytes": a.threshold,
              "guard": {"file": str(guard_path.relative_to(K)), "summary": guard["summary"]},
              "nscale_guard": ({"file": "results/r2_nscale_check_L256.json", "bit_equal_all": ns_guard["bit_equal_all"]}
                               if ns_guard else None),
              "memory_before_load": memory(),
              "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "litert-torch",
                                                                      "litert-converter", "ai-edge-litert")}}
    model, info = G.load_text_model()
    record["load_info"] = info
    record["form_apply"] = G.apply_form(model)
    if a.nscale:
        import r2_nscale as NS
        record["nscale_apply"] = NS.apply(model, NS.load_plan())
    graph = G.graph(model, L)
    ref = json.loads((K / "ref/records_ref.json").read_text())
    fits = [r for r in ref["rows"] if r["arm"] == "text" and r["tokens"] <= L]
    r0 = max(fits, key=lambda r: r["tokens"])
    table = H.EmbedTable(G.SNAP)
    embeds, valid, cos, sin = H.row_inputs(r0["ids"], L, table)
    sample = {"embeds": torch.from_numpy(embeds), "valid": torch.from_numpy(valid), "cos": torch.from_numpy(cos),
              "sin": torch.from_numpy(sin)}
    t_s = time.perf_counter()
    h = graph(**sample)["hidden"][0, :r0["tokens"]].numpy()
    z = np.load(K / f"ref/npz/{r0['id']}__text.npz")
    record["sample"] = {"id": r0["id"], "tokens": r0["tokens"], "finite": bool(np.isfinite(h).all()),
                        "max_abs_vs_reference_last_hidden": float(np.abs(h - z["last_hidden"]).max()),
                        "forward_seconds": round(time.perf_counter() - t_s, 1),
                        "shapes": {k: list(v.shape) for k, v in sample.items()}}
    record["memory_after_load"] = memory()
    record["load_and_check_seconds"] = round(time.perf_counter() - t_wall, 1)
    print(json.dumps({"sample": record["sample"], "memory_after_load": record["memory_after_load"]}), flush=True)
    stop, samples = threading.Event(), []
    sampler = threading.Thread(target=rss_sampler, args=(stop, samples), daemon=True)
    sampler.start()
    import litert_torch
    t0 = time.perf_counter()
    lrt = litert_torch.convert(graph, sample_kwargs=sample, lightweight_conversion=True, runtime_constant_folding=False)
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
