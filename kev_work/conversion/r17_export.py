"""Export KevPrefill with an r17_kernel form (fp32 weights) and its V2 storage form (the published recipe): r13_export.py
with the form through r17_kernel.apply (r13_kernel's forms + the FC tokens kc<kmax> / bk<kmax> / os<k>), the guard read
from results/r17_torch_parity_<form>.json, and the BmmLinear BATCH_MATMULs' weights cast to fp16 like the FCs'.

    python r17_export.py --L 128 --form R64+sp+ec+dd+vs6+bk1024@in_proj_z.q_proj --tag r17C7-bkzq [--keep-fp32]

The tag defaults to "r17" + the form with '+' -> '-', '@' -> '_' and ',' -> '.'
(exports/kev08b_rowprefill_L{L}_v2_fp16fc_i8emb_<tag>.tflite); the NPU-ready files were made with --tag r17C7-bkzq.
1. Guard (recorded, not required): results/r17_torch_parity_<form>.json when present; the graph object is re-checked on
   two rows against the oracle (argmax equal, |dp| <= 1e-4), as export_kev.py does.
2. r17_kernel.apply(model, form); litert_torch.convert(graph, sample_kwargs={"ids", "valid"}).export(...) ->
   exports/kev08b_rowprefill_L{L}_fp32_<tag>.tflite; static scan -> results/opscan_L{L}_<tag>.json; record ->
   results/export_L{L}_<tag>.json with the op deltas against the earlier release's graph (results/opscan_L{L}.json) and
   the R64 export (results/opscan_L{L}_r12R64.json) when those exist. Stops on CUSTOM / rank > 4 / STABLEHLO / INT64 / a
   forbidden op other than the exp clamp's MAXIMUM.
3. V2 (quantize_kev.build_recipe("v2"), plus FLOAT_CASTING for the BmmLinear BATCH_MATMULs) ->
   exports/kev08b_rowprefill_L{L}_v2_fp16fc_i8emb_<tag>.tflite + results/quant_L{L}_v2_fp16fc_i8emb_<tag>.json: every FC
   weight fp16, one int8 table, the BmmLinear BATCH_MATMULs that stay BATCH_MATMUL (m > 1) with an fp16 right operand,
   op histogram = the fp32 file's except DEQUANTIZE. The fp32 file is deleted unless --keep-fp32.
Nothing existing is overwritten."""
import argparse
import importlib.metadata
import json
import resource
import time

import numpy as np
import torch

from r2_common import K, Head, dump_json, load_oracle, qkey, select, sha256_file


def bmm_rhs_fp16(path):
    """Count the BmmLinear BATCH_MATMULs whose right operand is DEQUANTIZE(FLOAT16 constant) (identical constants may
    share one DEQUANTIZE, so DEQUANTIZE ops are not counted)."""
    from ai_edge_litert import schema_py_generated as schema
    buf = open(path, "rb").read()
    m = schema.Model.GetRootAs(buf, 0)
    names = {v: k for k, v in vars(schema.BuiltinOperator).items() if isinstance(v, int)}
    g = m.Subgraphs(0)
    prod = {}
    ops = [g.Operators(i) for i in range(g.OperatorsLength())]
    code = lambda op: names.get(max(m.OperatorCodes(op.OpcodeIndex()).BuiltinCode(),
                                    m.OperatorCodes(op.OpcodeIndex()).DeprecatedBuiltinCode()))
    for op in ops:
        for t in op.OutputsAsNumpy():
            prod[int(t)] = op
    n = 0
    for op in ops:
        if code(op) != "BATCH_MATMUL":
            continue
        if b"BmmLinear" not in g.Tensors(int(op.OutputsAsNumpy()[0])).Name():
            continue
        rhs = int(op.InputsAsNumpy()[1])
        p = prod.get(rhs)
        if p is not None and code(p) == "DEQUANTIZE" and g.Tensors(int(p.InputsAsNumpy()[0])).Type() == schema.TensorType.FLOAT16:
            n += 1
    return n


def build_recipe_v2_bmm():
    """quantize_kev.build_recipe("v2") + FLOAT_CASTING for the BmmLinear BATCH_MATMULs (their constant right operand
    -> fp16 + DEQUANTIZE, as the FCs). ai-edge-quantizer 0.9.0 registers FLOAT_CASTING for FC / CONV / EMBEDDING only;
    this process adds BATCH_MATMUL with the same materialize function (parse_fc_bmm_conv_tensors already handles
    BATCH_MATMUL). The scope regex "BmmLinear" leaves every other BATCH_MATMUL (kernel, attention) as it is."""
    from ai_edge_quantizer import algorithm_manager as am, qtyping, recipe_manager
    from ai_edge_quantizer.algorithms.nonlinear_quantize import float_casting
    OP = qtyping.TFLOperationName
    G = qtyping.QuantGranularity
    float_casting.SUPPORTED_WEIGHT_QUANT_OPS = float_casting.SUPPORTED_WEIGHT_QUANT_OPS | {OP.BATCH_MATMUL}
    am.register_quantized_op(am.AlgorithmName.FLOAT_CASTING, OP.BATCH_MATMUL, float_casting.init_qsvs,
                             calibration_func=float_casting.calibrate, materialize_func=float_casting.materialize_fc_conv)
    rm = recipe_manager.RecipeManager()
    rm.add_weight_only_config(regex=".*", operation_name=OP.FULLY_CONNECTED, num_bits=16, granularity=G.CHANNELWISE,
                              algorithm_key=am.AlgorithmName.FLOAT_CASTING)
    rm.add_dynamic_config(regex=".*", operation_name=OP.EMBEDDING_LOOKUP, num_bits=8, granularity=G.CHANNELWISE)
    rm.add_weight_only_config(regex="BmmLinear", operation_name=OP.BATCH_MATMUL, num_bits=16, granularity=G.CHANNELWISE,
                              algorithm_key=am.AlgorithmName.FLOAT_CASTING)
    return rm.get_quantization_recipe(), rm.need_calibration()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--form", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--keep-fp32", action="store_true")
    a = ap.parse_args()
    L = a.L
    tag = a.tag or "r17" + a.form.replace("+", "-").replace("@", "_").replace(",", ".")
    fp32 = K / f"exports/kev08b_rowprefill_L{L}_fp32_{tag}.tflite"
    v2 = K / f"exports/kev08b_rowprefill_L{L}_v2_fp16fc_i8emb_{tag}.tflite"
    rec_path = K / f"results/export_L{L}_{tag}.json"
    scan_path = K / f"results/opscan_L{L}_{tag}.json"
    q_path = K / f"results/quant_L{L}_v2_fp16fc_i8emb_{tag}.json"
    for p in (fp32, v2, rec_path, scan_path, q_path):
        assert not p.exists(), f"never overwrite {p}"
    guard_file = K / f"results/r17_torch_parity_{a.form.replace('+', '-').replace('@', '_').replace(',', '.')}.json"
    guard = json.loads(guard_file.read_text()) if guard_file.exists() else None
    torch.set_num_threads(4)
    import r17_kernel as R
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
    record = {"L": L, "form": a.form, "tag": tag, "applied": {k: v for k, v in applied.items() if k != "tokens"},
              "status": "RUNNING", "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "file": str(fp32.relative_to(K)),
              "guard": ({"file": str(guard_file.relative_to(K)), "pass": guard["pass"], "h_sel_max_abs": guard["h_sel_max_abs"],
                         "probs_max_abs": guard["probs_max_abs"], "argmax_equal": guard["argmax_equal"]}
                        if guard else "not present at export time (screen export; see the parity json later)"),
              "probe_checks": checks, "load_info": load_info,
              "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "litert-torch",
                                                                      "litert-converter", "ai-edge-litert",
                                                                      "ai-edge-quantizer")}}
    t0 = time.perf_counter()
    import litert_torch
    lrt = litert_torch.convert(graph, sample_kwargs={"ids": sample_ids, "valid": sample_valid})
    record["convert_seconds"] = round(time.perf_counter() - t0, 1)
    lrt.export(str(fp32))
    record["export_seconds_total"] = round(time.perf_counter() - t0, 1)
    record["peak_rss_bytes_getrusage"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    from tflite_scan import scan
    s = scan(fp32)
    dump_json(scan_path, s)
    sig = s["signatures"]
    sig_ok = (len(sig) == 1 and [(i["name"], i["shape"], i["dtype"]) for i in sig[0]["inputs"]] ==
              [("ids", [1, L], "INT32"), ("valid", [1, L], "FLOAT32")] and
              [(o["name"], o["shape"], o["dtype"]) for o in sig[0]["outputs"]] == [("hidden", [1, L, 1024], "FLOAT32")])
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
                  operator_count=s["operator_count"], op_histogram=hist, forbidden_counts=s["forbidden_counts"],
                  int64_tensor_count=s["int64_tensor_count"], rank_gt4_tensor_count=s["rank_gt4_tensor_count"],
                  tensor_rank_histogram=s["tensor_rank_histogram"], custom_op_count=s["custom_op_count"],
                  stablehlo_ops=s["stablehlo_ops"], batch_matmul_count=s["batch_matmul_count"],
                  batch_matmul_constant_left=s.get("batch_matmul_constant_left"),
                  fully_connected_count=s["fully_connected_count"], embedding_lookup_count=s["embedding_lookup_count"],
                  gather_count=s["gather_count"], opscan=str(scan_path.relative_to(K)),
                  delta_vs_shipped_kernel=delta(f"results/opscan_L{L}.json"),
                  delta_vs_r12R64=delta(f"results/opscan_L{L}_r12R64.json"))
    dump_json(rec_path, record)
    print(json.dumps({k: record[k] for k in ("status", "bytes", "operator_count", "signature_matches_contract",
                                             "fully_connected_count", "convert_seconds")}), flush=True)
    assert record["status"] == "EXPORTED" and sig_ok, record["status"]
    from quantize_kev import EXPECT, build_recipe, quant_scan
    from ai_edge_quantizer import quantizer
    nbmm = applied["split_linears"] if applied.get("bmm_chain") else 0
    recipe, need_cal = build_recipe_v2_bmm() if nbmm else build_recipe("v2")
    # the converter turns a batch-1 BmmLinear (m = 1) back into FULLY_CONNECTED:
    # only the m > 1 ones stay BATCH_MATMUL with an fp16 right operand
    import re as _re
    nbmm = sum(v for k, v in (applied.get("r17_swap_shapes") or {}).items() if not _re.search(r"-> m1 x", k)) if nbmm else 0
    t1 = time.perf_counter()
    qt = quantizer.Quantizer(str(fp32), recipe)
    assert not (qt.need_calibration or need_cal)
    qt.quantize().export_model(str(v2))
    q_s = time.perf_counter() - t1
    scan_q, scan_src = quant_scan(v2), quant_scan(fp32)
    emb = scan_q["embedding_lookup"]
    exp = EXPECT["v2"]
    nfc = scan_src["fully_connected"]
    checks_q = {
        f"fc_count_{nfc}_as_fp32_file": scan_q["fully_connected"] == nfc,
        "fc_weight_all_" + exp["fc_weight"]: scan_q["fc_weight_source_dtype"] == {exp["fc_weight"]: nfc},
        "embedding_lookup_1_table_" + exp["table"]: len(emb) == 1 and emb[0]["table_source_dtype"] == exp["table"],
        "custom_0": scan_q["custom_op_count"] == 0,
        "fp32_constants_under_64MB": scan_q["constant_bytes_by_dtype"].get("FLOAT32", 0) < 64 * 2 ** 20,
        f"bmmlinear_rhs_fp16_dequant_{nbmm}": bmm_rhs_fp16(v2) == nbmm,
        "op_histogram_unchanged_except_DEQUANTIZE": {k: v for k, v in scan_q["op_histogram"].items() if k != "DEQUANTIZE"}
        == {k: v for k, v in scan_src["op_histogram"].items() if k != "DEQUANTIZE"},
    }
    qdoc = {"step": f"V2 of the {a.form} graph ({tag}), L={L}", "input": str(fp32.relative_to(K)),
            "output": {"file": str(v2.relative_to(K)), "bytes": v2.stat().st_size, "sha256": sha256_file(v2)},
            "recipe": recipe, "seconds": round(q_s, 1), "scan": {k: scan_q[k] for k in (
                "operator_count", "op_histogram", "fc_weight_source_dtype", "dequantize_count", "dequantize_in_out",
                "embedding_lookup", "constant_bytes_by_dtype")},
            "checks": checks_q, "checks_pass": all(checks_q.values())}
    dump_json(q_path, qdoc)
    print(json.dumps({k: qdoc[k] for k in ("output", "checks_pass")}), flush=True)
    assert qdoc["checks_pass"], checks_q
    if not a.keep_fp32:
        fp32.unlink()
        print(f"deleted {fp32.name} (rebuild: this script)")


if __name__ == "__main__":
    main()
