"""Round 3 step 2: weight forms of the fp32 D1Decision graph with ai-edge-quantizer (exporter venv lt094dev, run only).

    cd d1_omni_work
    ~/code/standup/tools/quiet/quiet_wait.py -- ~/venvs/lt094dev/bin/python scripts/quant_forms.py --L 256
        [--forms fp16,fp16fc_i8emb,wi8fc]

Input  = out/d1omni_decide_L{L}_fp32.tflite (step 1, never touched).
Output = out/d1omni_decide_L{L}_{form}.tflite (never overwritten; the quantizer refuses to overwrite anyway) and
results/quant_L{L}.json (one entry per form; a later call adds forms it does not hold yet and never rewrites an entry).
Forms (no calibration; the ShortConv taps, norms and the attention matmuls are not weights of a quantizable op):
  fp16          FLOAT_CASTING fp16 weights on every op the algorithm supports EXCEPT EMBEDDING_LOOKUP: the graph's
                FULLY_CONNECTED ops (it has no CONV_2D / DEPTHWISE_CONV_2D / CONV_2D_TRANSPOSE, asserted). Config of
                scripts/quantize_lfm25_fp16.py (add_quantization_config, float_casting, weight 16-bit FLOAT, compute
                FLOAT) with the operation narrowed from ALL_SUPPORTED: ALL_SUPPORTED would also cast the embedding table
                to fp16, and an fp16 table is not made in this lane (launch 2026-10-08 r3; agreed 11:0x). The table
                stays FLOAT32 (asserted: one table, FLOAT32 [65536, 1024]).
  fp16fc_i8emb  kev_work/scripts/quantize_kev.py `v2`: add_weight_only_config(FULLY_CONNECTED, 16, CHANNELWISE,
                FLOAT_CASTING) + add_dynamic_config(EMBEDDING_LOOKUP, 8, CHANNELWISE) = fp16 FC weights through
                DEQUANTIZE + an int8 table read by the lookup itself (no DEQUANTIZE; asserted: one INT8 table, its bytes
                = 65536 x 1024, i.e. not duplicated).
  wi8fc         kev_work/scripts/quantize_kev.py `v1` / scripts/convert_lfm25_encoder.py: add_dynamic_config(
                FULLY_CONNECTED, 8) + add_dynamic_config(EMBEDDING_LOOKUP, 8, CHANNELWISE) = dynamic-range int8 FC
                (int8 weights consumed directly, activations quantized at run time) + int8 table.
Per form: bytes, sha256, op histogram (asserted unchanged except DEQUANTIZE), tensor dtype histogram, constant bytes by
dtype, FC weight dtype (through a DEQUANTIZE producer), the table, and the checks.

Round 4 (addition): --multi <fp32 multi-signature file> --forms fp16 quantizes graph_build.py --multi's file with the
same fp16 recipe -> out/d1omni_decide_<tag>_fp16.tflite + results/multisig_quant.json; checks: every FC reads FLOAT16
weights, one FLOAT32 table buffer read by every signature's lookup, FLOAT16 bytes counted once per buffer equal to the
single-signature L256 fp16 file's (= the weights are held once), size 1.0-1.1 x the single-signature fp16 file.

Round 8 (addition): --variant f16safe reads out/d1omni_decide_L{L}_f16safe_fp32.tflite (graph_build.py --f16safe) and
writes out/d1omni_decide_L{L}_f16safe_{form}.tflite + results/quant_L{L}_f16safe.json, same recipes and checks.
"""
import argparse
import importlib.metadata as md
import json
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import litert_run as R  # noqa: E402

K = R.K
FORMS = ("fp16", "fp16fc_i8emb", "wi8fc")
TABLE = [65536, 1024]


def recipe_for(form):
    from ai_edge_quantizer import qtyping, recipe_manager
    from ai_edge_quantizer.algorithm_manager import AlgorithmName

    G, OP = qtyping.QuantGranularity, qtyping.TFLOperationName
    rm = recipe_manager.RecipeManager()
    if form == "fp16":
        rm.add_quantization_config(
            regex=".*", operation_name=OP.FULLY_CONNECTED, algorithm_key="float_casting",
            op_config=qtyping.OpQuantizationConfig(
                weight_tensor_config=qtyping.TensorQuantizationConfig(num_bits=16, dtype=qtyping.TensorDataType.FLOAT),
                compute_precision=qtyping.ComputePrecision.FLOAT))
    elif form == "fp16fc_i8emb":
        rm.add_weight_only_config(regex=".*", operation_name=OP.FULLY_CONNECTED, num_bits=16,
                                  granularity=G.CHANNELWISE, algorithm_key=AlgorithmName.FLOAT_CASTING)
        rm.add_dynamic_config(regex=".*", operation_name=OP.EMBEDDING_LOOKUP, num_bits=8, granularity=G.CHANNELWISE)
    elif form == "wi8fc":
        rm.add_dynamic_config(regex=".*", operation_name=OP.FULLY_CONNECTED, num_bits=8)
        rm.add_dynamic_config(regex=".*", operation_name=OP.EMBEDDING_LOOKUP, num_bits=8, granularity=G.CHANNELWISE)
    else:
        raise ValueError(form)
    return rm.get_quantization_recipe(), rm.need_calibration()


EXPECT = {  # FC weight source dtype for every FC; table dtype at the lookup's input / after a DEQUANTIZE producer
    "fp16": {"fc": "FLOAT16", "fc_direct": "None", "table": "FLOAT32", "table_via": "constant"},
    "fp16fc_i8emb": {"fc": "FLOAT16", "fc_direct": "None", "table": "INT8", "table_via": "constant"},
    "wi8fc": {"fc": "INT8", "fc_direct": "INT8", "table": "INT8", "table_via": "constant"},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int)
    ap.add_argument("--forms", default=",".join(FORMS))
    ap.add_argument("--table-ab", action="store_true", help="diagnostic: int8 table / fp16 FC weights in torch fp32")
    ap.add_argument("--multi", metavar="FP32_FILE", help="round 4: quantize the multi-signature fp32 file")
    ap.add_argument("--variant", choices=("f16safe", "f16safe_l0sum"), help="round 8: the graph variant's fp32 file")
    a = ap.parse_args()
    if a.multi:
        return quantize_multi(K / a.multi, a.forms.split(","))
    if a.table_ab:
        table_ab(a.L)
        return 0
    vt = f"{a.variant}_" if a.variant else ""
    src = K / f"out/d1omni_decide_L{a.L}_{vt}fp32.tflite"
    res = K / (f"results/quant_L{a.L}_{a.variant}.json" if a.variant else f"results/quant_L{a.L}.json")
    assert src.exists(), src
    rnd = 8 if a.variant else (3 if a.L in (256, 512) else 4)
    doc = json.loads(res.read_text()) if res.exists() else {
        "step": f"round {rnd} step {3 if a.variant else 2}: weight forms of {src.relative_to(K)}", "forms": {}}
    scan_src = R.scan(src, with_sha=False)
    assert not any(k in scan_src["op_histogram"] for k in ("CONV_2D", "DEPTHWISE_CONV_2D", "TRANSPOSE_CONV")), \
        scan_src["op_histogram"]
    n_fc = scan_src["fully_connected_count"]
    doc["input"] = {"file": str(src.relative_to(K)), "bytes": src.stat().st_size,
                    "operator_count": scan_src["operator_count"], "fully_connected": n_fc,
                    "tensor_dtype_histogram": scan_src["tensor_dtype_histogram"],
                    "constant_bytes_by_dtype": scan_src["constant_bytes_by_dtype"],
                    "embedding_lookup": scan_src["embedding_lookup"]}
    doc["versions"] = {p: md.version(p) for p in ("ai-edge-quantizer", "ai-edge-litert")}
    from ai_edge_quantizer import quantizer

    for form in a.forms.split(","):
        if form in doc["forms"]:
            print(f"{form}: already in {res.name}, skipped")
            continue
        out = K / f"out/d1omni_decide_L{a.L}_{vt}{form}.tflite"
        assert not out.exists(), f"refusing to overwrite {out}"
        entry = {"output": str(out.relative_to(K)), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        t0 = time.time()
        try:
            recipe, need_cal = recipe_for(form)
            entry["recipe"] = recipe
            qt = quantizer.Quantizer(str(src), recipe)
            if qt.need_calibration or need_cal:
                raise SystemExit(f"STOP: recipe {form} needs calibration")
            t1 = time.time()
            result = qt.quantize()
            entry["quantize_seconds"] = round(time.time() - t1, 1)
            result.export_model(str(out))
            del result, qt
            entry["seconds_wall"] = round(time.time() - t0, 1)
            sc = R.scan(out)
            emb = sc["embedding_lookup"]
            exp = EXPECT[form]
            const_tbl = TABLE[0] * TABLE[1] * {"FLOAT32": 4, "INT8": 1}[exp["table"]]
            checks = {
                f"fc_count_{n_fc}": sc["fully_connected_count"] == n_fc,
                f"fc_weight_all_{exp['fc']}": sc["fc_weight_source_dtype"] == {exp["fc"]: n_fc},
                f"fc_weight_direct_{exp['fc_direct']}": sc["fc_weight_tensor_dtype_direct"] == {exp["fc_direct"]: n_fc},
                f"embedding_lookup_1_table_{exp['table']}_{TABLE[0]}x{TABLE[1]}": len(emb) == 1
                and emb[0]["table_source_dtype"] == exp["table"] and emb[0]["table_shape"] == TABLE
                and emb[0]["table_via"] == exp["table_via"],
                "table_not_duplicated_bytes": sc["constant_bytes_by_dtype"].get(exp["table"], 0) >= const_tbl
                and (exp["table"] != "INT8" or sc["constant_bytes_by_dtype"].get("INT8", 0) - const_tbl
                     == (n_fc_int8_bytes(sc) if form == "wi8fc" else 0)),
                "custom_0": sc["custom_op_count"] == 0,
                "op_histogram_unchanged_except_DEQUANTIZE":
                    {k: v for k, v in sc["op_histogram"].items() if k != "DEQUANTIZE"}
                    == {k: v for k, v in scan_src["op_histogram"].items() if k != "DEQUANTIZE"},
                "no_fp16_table": not any(e["table_source_dtype"] == "FLOAT16" for e in emb),
            }
            entry.update(bytes=sc["bytes"], sha256=sc["sha256"], operator_count=sc["operator_count"],
                         op_histogram=sc["op_histogram"], tensor_dtype_histogram=sc["tensor_dtype_histogram"],
                         constant_bytes_by_dtype=sc["constant_bytes_by_dtype"],
                         constant_count_by_dtype=sc["constant_count_by_dtype"],
                         fc_weight_source_dtype=sc["fc_weight_source_dtype"],
                         fc_weight_tensor_dtype_direct=sc["fc_weight_tensor_dtype_direct"],
                         fc_bias_dtype=sc["fc_bias_dtype"], fc_input_rank=sc["fc_input_rank"],
                         dequantize_count=sc["dequantize_count"], dequantize_in_out=sc["dequantize_in_out"],
                         embedding_lookup=emb, fc_rows_first3=sc["fc_rows_first3"],
                         peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                         checks=checks, checks_pass=all(checks.values()), status="OK")
        except BaseException as e:  # recorded; the next form still runs
            import traceback
            entry.update(status="FAIL", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-4000:],
                         seconds_wall=round(time.time() - t0, 1))
        doc["forms"][form] = entry
        doc["written"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        R.dump_json(res, doc, overwrite=True)
        print(json.dumps({"form": form, "status": entry["status"], "bytes": entry.get("bytes"),
                          "checks": entry.get("checks"), "error": entry.get("error"),
                          "tensor_dtype_histogram": entry.get("tensor_dtype_histogram"),
                          "constant_bytes_by_dtype": entry.get("constant_bytes_by_dtype"),
                          "dequantize_in_out": entry.get("dequantize_in_out"),
                          "table": [(e["table_source_dtype"], e["table_shape"], e["table_via"], e["table_bytes"])
                                    for e in entry.get("embedding_lookup", [])],
                          "seconds": entry.get("seconds_wall")}, indent=1), flush=True)
    return 0 if all(e["status"] == "OK" and e.get("checks_pass") for e in doc["forms"].values()) else 1


def quantize_multi(src, forms):
    """Round 4 step 4: the fp16 form of the multi-signature fp32 file (one output per form, never overwritten)."""
    from ai_edge_quantizer import quantizer

    assert forms == ["fp16"], forms
    assert src.exists(), src
    tag = src.stem.replace("d1omni_decide_", "").replace("_fp32", "")
    res = K / "results/multisig_quant.json"
    assert not res.exists(), f"refusing to overwrite {res}"
    scan_src = R.scan(src, with_sha=False)
    n_fc, n_sig = scan_src["fully_connected_count"], len(scan_src["signatures"])
    single = K / "out/d1omni_decide_L256_fp16.tflite"
    single_q = json.loads((K / "results/quant_L256.json").read_text())["forms"]["fp16"]
    doc = {"step": f"round 4 step 4: fp16 form of {src.name}", "written": None,
           "versions": {p: md.version(p) for p in ("ai-edge-quantizer", "ai-edge-litert")},
           "input": {"file": str(src.relative_to(K)), "bytes": src.stat().st_size, "signatures": n_sig,
                     "operator_count": scan_src["operator_count"], "fully_connected": n_fc,
                     "constant_unique_buffer_bytes_by_dtype": scan_src["constant_unique_buffer_bytes_by_dtype"]},
           "single_signature_fp16_L256": {"file": str(single.relative_to(K)), "bytes": single.stat().st_size,
                                          "float16_constant_bytes": single_q["constant_bytes_by_dtype"]["FLOAT16"]},
           "forms": {}}
    for form in forms:
        out = K / f"out/d1omni_decide_{tag}_{form}.tflite"
        assert not out.exists(), f"refusing to overwrite {out}"
        entry = {"output": str(out.relative_to(K)), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        t0 = time.time()
        recipe, need_cal = recipe_for(form)
        entry["recipe"] = recipe
        qt = quantizer.Quantizer(str(src), recipe)
        if qt.need_calibration or need_cal:
            raise SystemExit(f"STOP: recipe {form} needs calibration")
        result = qt.quantize()
        result.export_model(str(out))
        del result, qt
        entry["seconds_wall"] = round(time.time() - t0, 1)
        sc = R.scan(out)
        emb = sc["embedding_lookup"]
        f16_unique = sc["constant_unique_buffer_bytes_by_dtype"].get("FLOAT16", 0)
        ratio = sc["bytes"] / single.stat().st_size
        checks = {
            f"fc_count_{n_fc}": sc["fully_connected_count"] == n_fc,
            "fc_weight_all_FLOAT16": sc["fc_weight_source_dtype"] == {"FLOAT16": n_fc},
            f"embedding_lookup_{n_sig}_one_FLOAT32_table_buffer": len(emb) == n_sig
            and all(e["table_source_dtype"] == "FLOAT32" and e["table_shape"] == TABLE for e in emb)
            and len({e["table_buffer"] for e in emb}) == 1,
            "float16_unique_bytes_eq_single_L256": f16_unique == single_q["constant_bytes_by_dtype"]["FLOAT16"],
            "size_1.0_to_1.1_x_single_fp16_L256": 1.0 <= ratio <= 1.1,
            "custom_0": sc["custom_op_count"] == 0,
            "op_histogram_unchanged_except_DEQUANTIZE":
                {k: v for k, v in sc["op_histogram"].items() if k != "DEQUANTIZE"}
                == {k: v for k, v in scan_src["op_histogram"].items() if k != "DEQUANTIZE"},
            "no_fp16_table": not any(e["table_source_dtype"] == "FLOAT16" for e in emb),
        }
        entry.update(bytes=sc["bytes"], sha256=sc["sha256"], bytes_over_single_fp16_L256=ratio,
                     operator_count=sc["operator_count"], op_histogram=sc["op_histogram"],
                     tensor_dtype_histogram=sc["tensor_dtype_histogram"],
                     constant_bytes_by_dtype=sc["constant_bytes_by_dtype"],
                     constant_unique_buffer_bytes_by_dtype=sc["constant_unique_buffer_bytes_by_dtype"],
                     constant_unique_buffer_count_by_dtype=sc["constant_unique_buffer_count_by_dtype"],
                     fc_weight_source_dtype=sc["fc_weight_source_dtype"], dequantize_count=sc["dequantize_count"],
                     dequantize_in_out=sc["dequantize_in_out"], embedding_lookup=emb, signatures=sc["signatures"],
                     peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                     checks=checks, checks_pass=all(checks.values()), status="OK")
        doc["forms"][form] = entry
    doc["written"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    R.dump_json(res, doc)
    print(json.dumps({f: {k: e[k] for k in ("bytes", "bytes_over_single_fp16_L256", "checks", "seconds_wall",
                                             "constant_unique_buffer_bytes_by_dtype")}
                      for f, e in doc["forms"].items()}, indent=1))
    return 0 if all(e["checks_pass"] for e in doc["forms"].values()) else 1


def table_from_file(path):
    """(int8 table [65536, 1024], per-row scales, zero points) of the file's one EMBEDDING_LOOKUP (flatbuffers only)."""
    import mmap

    import numpy as np
    from ai_edge_litert import schema_py_generated as schema

    f = open(path, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    model = schema.Model.GetRootAs(mm, 0)
    emb_code = [i for i in range(model.OperatorCodesLength())
                if max(model.OperatorCodes(i).BuiltinCode(), model.OperatorCodes(i).DeprecatedBuiltinCode())
                == schema.BuiltinOperator.EMBEDDING_LOOKUP]
    g = model.Subgraphs(0)
    op = next(g.Operators(o) for o in range(g.OperatorsLength()) if g.Operators(o).OpcodeIndex() in emb_code)
    t = g.Tensors(int(op.InputsAsNumpy()[1]))
    b = model.Buffers(t.Buffer())
    raw = b.DataAsNumpy().tobytes() if b.DataLength() > 0 else mm[b.Offset(): b.Offset() + b.Size()]
    q = t.Quantization()
    table = np.frombuffer(raw, np.int8).reshape(t.ShapeAsNumpy().tolist()).copy()
    scales = q.ScaleAsNumpy().astype(np.float32).copy()
    zps = q.ZeroPointAsNumpy().copy() if q.ZeroPointLength() else np.zeros_like(scales, np.int64)
    qdim = q.QuantizedDimension()
    mm.close()
    f.close()
    return table, scales, zps, qdim


def table_ab(L):
    """Diagnostic: how much of the fp16fc_i8emb residual is the int8 table, in torch fp32 (no LiteRT run).
    base = eager D1Decision fp32 (checkpoint); `table_int8` = the same with the embedding table replaced by the file's
    int8 table dequantized (int8 x row scale); `fc_fp16` = fp32 table, every Linear weight rounded to fp16 and back
    (the weights the fp16 / fp16fc_i8emb files carry). Rows = the L bucket's rows (litert_gate.load_rows); statistics =
    litert_gate.compare (variant vs base). Also the table's own error per row."""
    import numpy as np
    import torch

    import graph_build as B
    import litert_gate as LG

    t0 = time.time()
    torch.set_num_threads(8)
    path = K / f"out/d1omni_decide_L{L}_fp16fc_i8emb.tflite"
    tq, scales, zps, qdim = table_from_file(path)
    assert qdim == 0 and tq.shape == (65536, 1024) and scales.shape == (65536,) and not np.any(zps), (qdim, zps[:4])
    deq = tq.astype(np.float32) * scales[:, None]
    base, wrep = B.build(L, "checkpoint", B.S.config())
    W = base.encoder.embed_tokens.weight.detach().numpy().astype(np.float32)
    err = deq - W
    row_rms = np.sqrt((W.astype(np.float64) ** 2).mean(1))
    rel = np.sqrt((err.astype(np.float64) ** 2).mean(1)) / np.maximum(row_rms, 1e-30)
    peak_ratio = np.abs(W).max(1) / np.maximum(row_rms, 1e-30)
    oracle = LG.oracle_doc()
    rows, source, _ = LG.load_rows(L, oracle)
    used = sorted({i for r in rows for i in r["ids"]})
    stats_table = {"rows": int(W.shape[0]), "scale_formula_check_max_abs": float(np.abs(
        scales - np.abs(W).max(1) / 127.0).max()),
        "rel_rms_error": {"median": float(np.median(rel)), "p99": float(np.percentile(rel, 99)), "max": float(rel.max())},
        "rel_rms_error_ids_used_by_rows": {"n_ids": len(used), "median": float(np.median(rel[used])),
                                           "p99": float(np.percentile(rel[used], 99)), "max": float(rel[used].max())},
        "row_peak_over_rms": {"median": float(np.median(peak_ratio)), "p99": float(np.percentile(peak_ratio, 99)),
                              "max": float(peak_ratio.max())},
        "max_abs_error": float(np.abs(err).max())}

    def scores_of(model):
        out = {}
        for r in rows:
            x = LG.row_inputs(r, L)
            with torch.no_grad():
                s = model(**{k: torch.from_numpy(v) for k, v in x.items()})["scores"].numpy().reshape(-1)
            out[r["key"]] = LG.readout(s[:r["P"] + r["n"]].copy(), r)
        return out

    ref = scores_of(base)
    sources = LG.record_sources()
    doc = {"step": f"round 3 diagnostic: int8 table vs fp16 FC weights in torch fp32, L={L} (no LiteRT run)",
           "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "row_source": source, "rows": len(rows),
           "torch": torch.__version__, "table_file": str(path.relative_to(K)), "table": stats_table, "variants": {}}
    with torch.no_grad():
        base.encoder.embed_tokens.weight.copy_(torch.from_numpy(deq))
    v = scores_of(base)
    doc["variants"]["table_int8"] = LG.compare(rows, v, ref, sources)
    doc["variants"]["table_int8"]["red_arm"] = LG.red_arm(v, ref)
    with torch.no_grad():
        base.encoder.embed_tokens.weight.copy_(torch.from_numpy(W))
        n_lin = 0
        for m in base.modules():
            if isinstance(m, torch.nn.Linear):
                m.weight.copy_(m.weight.half().float())
                n_lin += 1
        base.head.type_emb.weight.copy_(base.head.type_emb.weight.half().float())
    v = scores_of(base)
    doc["variants"]["fc_fp16"] = LG.compare(rows, v, ref, sources)
    doc["variants"]["fc_fp16"]["linear_weights_rounded"] = n_lin + 1
    doc["seconds_wall"] = round(time.time() - t0, 1)
    for k in ("table_int8", "fc_fp16"):
        for drop in ("cutoff_crossing_rows",):
            doc["variants"][k][drop] = {c: v[:20] for c, v in doc["variants"][k][drop].items()}
    R.dump_json(K / f"results/quant_table_ab_L{L}.json", doc)
    print(json.dumps({"table": stats_table, **{k: {x: doc["variants"][k][x] for x in (
        "max_abs_dp", "p95_abs_dp", "mean_abs_dp", "max_abs_dlogit", "cutoff_crossings", "bar_pass")}
        for k in doc["variants"]}, "seconds": doc["seconds_wall"]}, indent=1))


def n_fc_int8_bytes(sc):
    """INT8 constant bytes that belong to FC weights (wi8fc: int8 FC weights sit next to the int8 table)."""
    total = 0
    for shape_json, count in sc["fc_weight_shapes"].items():
        shape = json.loads(shape_json)
        total += count * shape[0] * shape[1]
    return total


if __name__ == "__main__":
    sys.exit(main())
