"""Export KevPrefill with an r15_form form (fp32 weights) and its V2 storage form (the published recipe), for Kev-4B
(--model 4b: exports/kev4b_..., results/..._4b...). r13_export.py with the torch-invariance guard required, the export
memory recorded, and the fp32 file deleted right after the V2 file is checked.

    python r15_export.py --model 4b --L 128 --form R64+sp+ec+dd+vs6+in1+fn5 [--tag <t>] [--keep-fp32]

tag defaults to "r15" + the form with '+' -> '-' (exports/kev4b_rowprefill_L{L}_v2_fp16fc_i8emb_<tag>.tflite). The
published Kev-4B files are this script's output with --form R64+sp+ec+dd+vs6+in1+fn5, renamed (conversion/README.md).
1. Guard (required): results/r15_torch_parity<_4b>_<form with ->.json must say pass (402 questions, h_sel <= 1e-4,
   probs <= 1e-5, argmax 402/402, finite: r15_torch_parity.py); the graph object is re-checked on two rows against the
   oracle (argmax equal, |dp| <= 1e-4), as export_kev.py does (tv4_000 or the first row that fits, and the longest row
   that fits L).
2. r15_form.apply(model, form); litert_torch.convert(graph, sample_kwargs={"ids", "valid"}).export(...) ->
   exports/<prefix>_rowprefill_L{L}_fp32_<tag>.tflite; static scan -> results/opscan_L{L}<_4b>_<tag>.json; record ->
   results/export_L{L}<_4b>_<tag>.json: convert / write / wall seconds, the process memory (ru_maxrss and
   phys_footprint / lifetime max from proc_pid_rusage, r3_common.memory) at the end of the convert and of the write, RSS
   samples every 20 s, op deltas vs the loop kernel's export (results/opscan_L{L}<_4b>.json, when present).
   Stops on CUSTOM / rank > 4 / STABLEHLO / INT64 / a forbidden op other than the exp clamp's MAXIMUM.
3. V2 (quantize_kev.build_recipe("v2")) -> exports/<prefix>_rowprefill_L{L}_v2_fp16fc_i8emb_<tag>.tflite +
   results/quant_L{L}<_4b>_v2_fp16fc_i8emb_<tag>.json: every FC weight fp16 (the count is the fp32 file's: 248 + the dd
   constant 0/1 FCs), one int8 table, op histogram = the fp32 file's except DEQUANTIZE; quantizer peak RSS recorded.
   The fp32 file is deleted unless --keep-fp32 (a 4B fp32 file is about 17 GB).
Nothing existing is overwritten."""
import argparse
import importlib.metadata
import json
import os
import resource
import subprocess
import threading
import time

import numpy as np
import torch

from r2_common import (FILE_PREFIX, HIDDEN, MODEL, RESULT_SUFFIX, K, Head, add_model_arg, dump_json, load_oracle, qkey,
                       select, sha256_file)
from r3_common import memory


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
    ap.add_argument("--form", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--keep-fp32", action="store_true")
    add_model_arg(ap)
    a = ap.parse_args()
    assert a.model == MODEL
    L = a.L
    ftag = a.form.replace("+", "-")
    tag = a.tag or "r15" + ftag
    fp32 = K / f"exports/{FILE_PREFIX}_rowprefill_L{L}_fp32_{tag}.tflite"
    v2 = K / f"exports/{FILE_PREFIX}_rowprefill_L{L}_v2_fp16fc_i8emb_{tag}.tflite"
    rec_path = K / f"results/export_L{L}{RESULT_SUFFIX}_{tag}.json"
    scan_path = K / f"results/opscan_L{L}{RESULT_SUFFIX}_{tag}.json"
    q_path = K / f"results/quant_L{L}{RESULT_SUFFIX}_v2_fp16fc_i8emb_{tag}.json"
    for p in (fp32, v2, rec_path, scan_path, q_path):
        assert not p.exists(), f"never overwrite {p}"
    guard_file = K / f"results/r15_torch_parity{RESULT_SUFFIX}_{ftag}.json"
    guard = json.loads(guard_file.read_text())
    assert guard["pass"] and guard["questions"] == 402 and guard["form"] == a.form, \
        f"torch invariance guard did not pass: {guard_file.name}"
    t_wall = time.perf_counter()
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    torch.set_num_threads(4)
    import r15_form as R   # r13_kernel's tokens + fn<k> / in<k> (norm pre-scales)
    from kev_graph import KevPrefill, load_text_model, row_inputs
    model, load_info = load_text_model()
    applied = R.apply(model, a.form)
    graph = KevPrefill(model, L).eval().requires_grad_(False)
    head = Head()
    oracle = load_oracle()
    first = next(q for q in oracle["questions"] if q["id"] == "tv4_000")
    if first["row_len"] > L:
        first = next(q for q in oracle["questions"] if q["row_len"] <= L)
    probes = [first, max((q for q in oracle["questions"] if q["row_len"] <= L), key=lambda q: q["row_len"])]
    checks = []
    with torch.no_grad():
        for q in probes:
            h = graph(*row_inputs(q["row_ids"], L))["hidden"][0].numpy()
            _, _, p = head(select(h, q))
            dp = float(np.abs(p - np.asarray(q["probs"])).max())
            checks.append({"key": qkey(q), "row_len": q["row_len"], "max_abs_dp": dp,
                           "argmax_equal": q["keys"][int(np.argmax(p))] == q["argmax_key"]})
    assert all(c["argmax_equal"] and c["max_abs_dp"] <= 1e-4 for c in checks), checks
    sample_ids, sample_valid = row_inputs(probes[0]["row_ids"], L)
    record = {"L": L, "model": MODEL, "form": a.form, "tag": tag,
              "applied": {k: v for k, v in applied.items() if k != "tokens"},
              "status": "RUNNING", "started_at": started, "file": str(fp32.relative_to(K)),
              "guard": {"file": str(guard_file.relative_to(K)), "pass": guard["pass"],
                        "h_sel_max_abs": guard["h_sel_max_abs"], "probs_max_abs": guard["probs_max_abs"],
                        "argmax_equal": guard["argmax_equal"], "questions": guard["questions"]},
              "probe_checks": checks, "load_info": load_info, "memory_before_convert": memory(),
              "tmpdir": os.environ.get("TMPDIR"),
              "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "litert-torch",
                                                                      "litert-converter", "ai-edge-litert",
                                                                      "ai-edge-quantizer")}}
    stop, samples = threading.Event(), []
    sampler = threading.Thread(target=rss_sampler, args=(stop, samples), daemon=True)
    sampler.start()
    t0 = time.perf_counter()
    import litert_torch
    lrt = litert_torch.convert(graph, sample_kwargs={"ids": sample_ids, "valid": sample_valid})
    record["convert_seconds"] = round(time.perf_counter() - t0, 1)
    record["memory_after_convert"] = memory()
    t1 = time.perf_counter()
    lrt.export(str(fp32))
    record["write_seconds"] = round(time.perf_counter() - t1, 1)
    record["export_seconds_total"] = round(time.perf_counter() - t0, 1)
    record["memory_after_write"] = memory()
    stop.set()
    record["rss_samples_bytes"] = samples
    record["peak_rss_bytes_getrusage"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    del lrt
    from tflite_scan import scan
    s = scan(fp32)
    dump_json(scan_path, s)
    sig = s["signatures"]
    sig_ok = (len(sig) == 1 and [(i["name"], i["shape"], i["dtype"]) for i in sig[0]["inputs"]] ==
              [("ids", [1, L], "INT32"), ("valid", [1, L], "FLOAT32")] and
              [(o["name"], o["shape"], o["dtype"]) for o in sig[0]["outputs"]] == [("hidden", [1, L, HIDDEN], "FLOAT32")])
    hist = s["op_histogram"]

    def delta(path):
        p = K / path
        if not p.exists():
            return None
        other = json.loads(p.read_text())
        oh = other["operator_count"], other["op_histogram"]
        return {"vs": path, "operator_count": s["operator_count"] - oh[0],
                "op_histogram": {k: hist.get(k, 0) - oh[1].get(k, 0) for k in sorted(set(hist) | set(oh[1]))
                                 if hist.get(k, 0) != oh[1].get(k, 0)}}

    expclamp = applied["expclamp"]
    forbidden_stop = sum(v for k, v in s["forbidden_counts"].items() if not (expclamp and k == "MAXIMUM"))
    stop_reason = ("CUSTOM_OP_STOP" if s["custom_op_count"] else "RANK5_STOP" if s["rank_gt4_tensor_count"] else
                   "STABLEHLO_STOP" if s["stablehlo_ops"] else "FORBIDDEN_STOP" if forbidden_stop else
                   "INT64_STOP" if s["int64_tensor_count"] else "EXPORTED")
    record.update(status=stop_reason, bytes=s["bytes"], sha256=s["sha256"], signature_matches_contract=sig_ok,
                  signatures=sig, operator_count=s["operator_count"], op_histogram=hist,
                  forbidden_counts=s["forbidden_counts"], int64_tensor_count=s["int64_tensor_count"],
                  rank_gt4_tensor_count=s["rank_gt4_tensor_count"], tensor_rank_histogram=s["tensor_rank_histogram"],
                  custom_op_count=s["custom_op_count"], stablehlo_ops=s["stablehlo_ops"],
                  batch_matmul_count=s["batch_matmul_count"],
                  batch_matmul_constant_left=s.get("batch_matmul_constant_left"),
                  batch_matmul_shape_groups=s.get("batch_matmul_shape_groups"), pad_count=s.get("pad_count"),
                  fully_connected_count=s["fully_connected_count"], embedding_lookup_count=s["embedding_lookup_count"],
                  gather_count=s["gather_count"], opscan=str(scan_path.relative_to(K)),
                  delta_vs_shipped_kernel=delta(f"results/opscan_L{L}{RESULT_SUFFIX}.json"))
    dump_json(rec_path, record)
    print(json.dumps({k: record[k] for k in ("status", "bytes", "operator_count", "signature_matches_contract",
                                             "fully_connected_count", "convert_seconds", "write_seconds")}), flush=True)
    print(json.dumps({"lifetime_max_phys_footprint_gb": round(record["memory_after_write"].get(
        "lifetime_max_phys_footprint", 0) / 1e9, 2), "ru_maxrss_gb": round(record["peak_rss_bytes_getrusage"] / 1e9, 2)}),
        flush=True)
    assert record["status"] == "EXPORTED" and sig_ok, record["status"]
    del graph, model
    import gc
    gc.collect()
    from quantize_kev import EXPECT, build_recipe, quant_scan
    from ai_edge_quantizer import quantizer
    recipe, need_cal = build_recipe("v2")
    t2 = time.perf_counter()
    qt = quantizer.Quantizer(str(fp32), recipe)
    assert not (qt.need_calibration or need_cal)
    qt.quantize().export_model(str(v2))
    q_s = time.perf_counter() - t2
    del qt
    scan_q, scan_src = quant_scan(v2), quant_scan(fp32)
    emb = scan_q["embedding_lookup"]
    exp = EXPECT["v2"]
    nfc = scan_src["fully_connected"]
    checks_q = {
        f"fc_count_{nfc}_as_fp32_file": scan_q["fully_connected"] == nfc,
        "fc_weight_all_" + exp["fc_weight"]: scan_q["fc_weight_source_dtype"] == {exp["fc_weight"]: nfc},
        "embedding_lookup_1_table_" + exp["table"]: len(emb) == 1 and emb[0]["table_source_dtype"] == exp["table"],
        "batch_matmul_constants_FLOAT32_only": set(scan_q["batch_matmul_constant_operand_dtype"]) <= {"FLOAT32"},
        "custom_0": scan_q["custom_op_count"] == 0,
        "fp32_constants_under_64MB": scan_q["constant_bytes_by_dtype"].get("FLOAT32", 0) < 64 * 2 ** 20,
        "op_histogram_unchanged_except_DEQUANTIZE": {k: v for k, v in scan_q["op_histogram"].items() if k != "DEQUANTIZE"}
        == {k: v for k, v in scan_src["op_histogram"].items() if k != "DEQUANTIZE"},
    }
    qdoc = {"step": f"V2 of the {a.form} graph ({tag}), L={L}, model {MODEL}", "input": str(fp32.relative_to(K)),
            "input_bytes": fp32.stat().st_size,
            "output": {"file": str(v2.relative_to(K)), "bytes": v2.stat().st_size, "sha256": sha256_file(v2)},
            "recipe": recipe, "seconds": round(q_s, 1), "memory_after_quantize": memory(),
            "scan": {k: scan_q[k] for k in ("operator_count", "op_histogram", "fully_connected", "fc_weight_source_dtype",
                                            "dequantize_count", "dequantize_in_out", "embedding_lookup",
                                            "batch_matmul_constant_operand_dtype", "constant_bytes_by_dtype")},
            "checks": checks_q, "checks_pass": all(checks_q.values())}
    dump_json(q_path, qdoc)
    print(json.dumps({k: qdoc[k] for k in ("output", "checks_pass", "seconds")}), flush=True)
    assert qdoc["checks_pass"], checks_q
    if not a.keep_fp32:
        fp32.unlink()
        print(f"deleted {fp32.name} (rebuild: this script)", flush=True)
    print(json.dumps({"wall_seconds": round(time.perf_counter() - t_wall, 1)}), flush=True)


if __name__ == "__main__":
    main()
