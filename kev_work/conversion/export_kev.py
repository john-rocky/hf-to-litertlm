"""Export KevPrefill (fp32, one window L) with litert_torch 0.9.4.

    python export_kev.py --L 1024 [--model 4b] [--guard-L 1024]

1. Guard: results/torch_graph_parity_L{L}[_4b].json must say pass (bar vs the reference, hidden vs stock 5.14.1,
   kev_eager bit-equal). The graph object being exported is re-checked on two rows against the reference (argmax,
   |dp| <= 1e-4): tv4_000 and the longest row that fits L.
2. litert_torch.convert(graph, sample_kwargs={"ids": int32 [1, L], "valid": float32 [1, L]}).export(...) ->
   exports/kev08b_rowprefill_L{L}_fp32.tflite (kev4b_... for 4B; input names = the kwargs keys, output name = the
   returned dict key).
3. Op-table scan (tflite_scan.py) -> results/opscan_L{L}[_4b].json; the summary goes into results/export_L{L}[_4b].json
   with seconds, bytes, sha256, versions, peak RSS and the stop checks (a CUSTOM op or a rank > 4 tensor = stop).
Existing artifacts are never overwritten. On an exception: logs/export_L{L}.traceback.txt + results/export_attempt_L{L}.json.

--guard-L <L0> takes the torch-graph guard from another window of the same model when this window has no
torch_graph_parity json of its own (the 4B graph was checked in PyTorch at L=1024 only): that json must say pass (same
graph code and weights, only the constant L differs), and the two probe rows of this window (the longest is the
1,805-token row at L=2048) must match the reference as above. The json records which guard was used.
--pre-oracle lets a 4B export start before the 4B reference exists: the guard is then the torch-graph precheck
(torch_graph_parity.py --phase precheck) and the probe rows are compared with the unpatched baseline's hidden states
(max |diff| <= 1e-3); --phase compare of torch_graph_parity.py follows.
The calls of the patch's head interleave during tracing are counted (4B: ratio 2)."""
import argparse
import importlib.metadata
import json
import os
import resource
import subprocess
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from r2_common import (FILE_PREFIX, HIDDEN, MODEL, RESULT_SUFFIX, K, Clock, Head, add_model_arg, cache_dir, dump_json,
                       load_oracle, qkey, select)


def rss_sampler(stop, samples, every=20):
    pid = os.getpid()
    while not stop.wait(every):
        try:
            kb = int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)], text=True).strip())
        except Exception:
            continue
        samples.append((round(time.time(), 1), kb * 1024))
        print(json.dumps({"rss_gib": round(kb / 2 ** 20, 2), "t": time.strftime("%H:%M:%S")}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--pre-oracle", action="store_true", help="guard = torch-graph precheck (see docstring)")
    ap.add_argument("--rows-from", default="", help="--pre-oracle: oracle json whose questions give the probe rows")
    ap.add_argument("--guard-L", type=int, default=0,
                    help="torch_graph_parity of window L0 as the guard when this L has none (see docstring)")
    add_model_arg(ap)
    a = ap.parse_args()
    assert a.model == MODEL
    L = a.L
    path = K / f"exports/{FILE_PREFIX}_rowprefill_L{L}_fp32.tflite"
    out = K / f"results/export_L{L}{RESULT_SUFFIX}.json"
    scan_out = K / f"results/opscan_L{L}{RESULT_SUFFIX}.json"
    for p in (path, out, scan_out):
        assert not p.exists(), f"never overwrite {p}"
    if a.pre_oracle:
        pre_path = cache_dir("cache/r2") / f"torch_graph_precheck_L{L}{RESULT_SUFFIX}.json"
        pre = json.loads(pre_path.read_text())
        assert pre["pass"] and pre["attention_all_bit_equal"] and pre["all_positions_finite"], pre
        guard = {"kind": "pre-oracle precheck (oracle comparison pending: torch_graph_parity --phase compare)",
                 "file": str(pre_path.relative_to(K)), "max_abs_hidden_vs_unpatched": pre["max_abs"], "bar": pre["bar"],
                 "within_1e-4": pre["within_1e-4"], "interleave": pre.get("interleave")}
    elif a.guard_L:
        own = K / f"results/torch_graph_parity_L{L}{RESULT_SUFFIX}.json"
        assert not own.exists(), f"{own.name} exists: use it (drop --guard-L)"
        src = K / f"results/torch_graph_parity_L{a.guard_L}{RESULT_SUFFIX}.json"
        tp = json.loads(src.read_text())
        assert tp["pass"] and tp["L"] == a.guard_L and tp.get("model", MODEL) == MODEL, "guard window did not pass"
        guard = {"kind": f"torch_graph_parity pass at L={a.guard_L} (same graph code and weights) + this window's "
                         "probe rows vs the oracle (argmax equal, |dp| <= 1e-4)",
                 "file": str(src.relative_to(K)), "guard_L": a.guard_L,
                 "guard_overall": tp["summary"]["overall"],
                 "guard_hidden_real_max_abs_vs_unpatched": tp.get("hidden_real_max_abs_vs_unpatched")}
    else:
        tp = json.loads((K / f"results/torch_graph_parity_L{L}{RESULT_SUFFIX}.json").read_text())
        assert tp["pass"], "step 2 (torch graph parity) did not pass for this L: do not export"
        guard = {"kind": "torch_graph_parity pass", "file": f"results/torch_graph_parity_L{L}{RESULT_SUFFIX}.json"}
    clock = Clock()
    started_at = clock.stamp()
    torch.set_num_threads(4)
    from kev_graph import KevPrefill, contract, count_interleave, load_text_model, row_inputs
    model, load_info = load_text_model()
    counter = count_interleave()
    graph = KevPrefill(model, L).eval().requires_grad_(False)
    head = Head()
    checks = []
    if a.pre_oracle:
        questions = json.loads(Path(a.rows_from).read_text())["questions"]
        step1 = np.load(cache_dir("cache/r2") / f"tf514_hidden_full{RESULT_SUFFIX}.npz")
        probes = [next(q for q in questions if q["id"] == "tv4_000"),
                  max((q for q in questions if q["row_len"] <= L), key=lambda q: q["row_len"])]
        with torch.no_grad():
            for q in probes:
                h = graph(*row_inputs(q["row_ids"], L))["hidden"][0].numpy()
                d = float(np.abs(h[: q["row_len"]].astype(np.float64) - step1[qkey(q)]).max())
                checks.append({"key": qkey(q), "row_len": q["row_len"], "max_abs_hidden_vs_unpatched": d})
        assert all(c["max_abs_hidden_vs_unpatched"] <= 1e-3 for c in checks), checks
    else:
        oracle = load_oracle()
        probes = [next(q for q in oracle["questions"] if q["id"] == "tv4_000"),
                  max((q for q in oracle["questions"] if q["row_len"] <= L), key=lambda q: q["row_len"])]
        with torch.no_grad():
            for q in probes:
                h = graph(*row_inputs(q["row_ids"], L))["hidden"][0].numpy()
                _, _, p = head(select(h, q))
                dp = float(np.abs(p - np.asarray(q["probs"])).max())
                checks.append({"key": qkey(q), "row_len": q["row_len"], "max_abs_dp": dp,
                               "argmax_equal": q["keys"][int(np.argmax(p))] == q["argmax_key"]})
        assert all(c["argmax_equal"] and c["max_abs_dp"] <= 1e-4 for c in checks), checks
    calls_probe = counter["calls"]
    sample_ids, sample_valid = row_inputs(probes[0]["row_ids"], L)
    record = {"L": L, "model": MODEL, "status": "RUNNING", "started_at": started_at, "file": str(path.relative_to(K)),
              "guard": guard, "probe_checks": checks, "load_info": load_info,
              "torch_graph_parity": guard["file"] if a.guard_L else f"results/torch_graph_parity_L{L}{RESULT_SUFFIX}.json",
              "sample_kwargs": {"ids": {"shape": list(sample_ids.shape), "dtype": "int32", "row": qkey(probes[0])},
                                "valid": {"shape": list(sample_valid.shape), "dtype": "float32"}},
              "convert_call": "litert_torch.convert(graph, sample_kwargs={'ids': ids, 'valid': valid}).export(path)",
              "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "litert-torch",
                                                                      "litert-converter", "ai-edge-litert")},
              "contract": contract(L, head)}
    stop, samples = threading.Event(), []
    sampler = threading.Thread(target=rss_sampler, args=(stop, samples), daemon=True)
    sampler.start()
    t0 = time.perf_counter()
    try:
        import litert_torch
        lrt = litert_torch.convert(graph, sample_kwargs={"ids": sample_ids, "valid": sample_valid})
        record["convert_seconds"] = round(time.perf_counter() - t0, 1)
        record["interleave_calls"] = {"probe_forwards": calls_probe, "during_convert": counter["calls"] - calls_probe,
                                      "per_forward_expected": 2 * load_info["gdn_layers"] if load_info["gdn_heads"]["ratio"] > 1 else 0}
        t1 = time.perf_counter()
        lrt.export(str(path))
        record["write_seconds"] = round(time.perf_counter() - t1, 1)
        record["export_seconds_total"] = round(time.perf_counter() - t0, 1)
        stop.set()
        from tflite_scan import scan
        t2 = time.perf_counter()
        s = scan(path)
        record["scan_seconds"] = round(time.perf_counter() - t2, 1)
        dump_json(scan_out, s)
        sig = s["signatures"]
        sig_ok = (len(sig) == 1 and [(i["name"], i["shape"], i["dtype"]) for i in sig[0]["inputs"]] ==
                  [("ids", [1, L], "INT32"), ("valid", [1, L], "FLOAT32")] and
                  [(o["name"], o["shape"], o["dtype"]) for o in sig[0]["outputs"]] == [("hidden", [1, L, HIDDEN], "FLOAT32")])
        record.update(
            status="CUSTOM_OP_STOP" if s["custom_op_count"] else ("RANK5_STOP" if s["rank_gt4_tensor_count"] else "EXPORTED"),
            bytes=s["bytes"], sha256=s["sha256"], signatures=sig, signature_matches_contract=sig_ok,
            operator_count=s["operator_count"], op_histogram=s["op_histogram"],
            tensor_rank_histogram=s["tensor_rank_histogram"], tensor_dtype_histogram=s["tensor_dtype_histogram"],
            int64_tensor_count=s["int64_tensor_count"], rank_gt4_tensor_count=s["rank_gt4_tensor_count"],
            forbidden_counts=s["forbidden_counts"], forbidden_total=s["forbidden_total"],
            pad_count=s["pad_count"], pad_summary=s["pad_summary"],
            batch_matmul_count=s["batch_matmul_count"], batch_matmul_all_rank4=s["batch_matmul_all_rank4"],
            batch_matmul_shape_groups=s["batch_matmul_shape_groups"],
            fully_connected_count=s["fully_connected_count"], embedding_lookup_count=s["embedding_lookup_count"],
            gather_count=s["gather_count"], custom_ops=s["custom_ops"], custom_op_count=s["custom_op_count"],
            stablehlo_ops=s["stablehlo_ops"], opscan=str(scan_out.relative_to(K)),
        )
    except BaseException:
        stop.set()
        (K / f"logs/export_L{L}{RESULT_SUFFIX}.traceback.txt").write_text(traceback.format_exc())
        record.update(status="FAIL", seconds=round(time.perf_counter() - t0, 1))
        dump_json(K / f"results/export_attempt_L{L}{RESULT_SUFFIX}.json", record)
        raise
    record["peak_rss_bytes_getrusage"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss   # bytes on macOS
    try:
        from r3_common import memory
        record["memory_end"] = memory()
    except Exception as e:   # informational
        record["memory_end"] = repr(e)
    record["rss_samples_bytes"] = samples
    record["seconds_wall"] = clock.seconds()
    dump_json(out, record)
    print(json.dumps({k: v for k, v in record.items() if k not in ("op_histogram", "batch_matmul_shape_groups", "contract",
                                                                    "rss_samples_bytes", "load_info")}, indent=1))


if __name__ == "__main__":
    main()
