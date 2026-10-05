"""Storage variants of the fp32 row-prefill graph with ai-edge-quantizer (applied to the exported file).

    python quantize_kev.py --L 1024 --variant v1|v2|v3|v4 [--model 4b]

Input  = exports/kev08b_rowprefill_L{L}_fp32.tflite (kev4b_... for 4B; never changed).
Output = exports/<prefix>_rowprefill_L{L}_{v1_wi8fc|v2_fp16fc_i8emb|v3_i8emb|v4_fp16fc_fp16emb}.tflite +
results/quant_L{L}[_4b]_{variant}.json (never overwritten). Conv, delta rule and norms stay float (no ALL_SUPPORTED):
  v1 wi8fc           dynamic int8 FULLY_CONNECTED + dynamic int8 EMBEDDING_LOOKUP (channelwise)
  v2 fp16fc_i8emb    weight-only 16-bit FLOAT_CASTING FULLY_CONNECTED (channelwise) + dynamic int8 EMBEDDING_LOOKUP
                     = the shipped files
  v3 i8emb           dynamic int8 EMBEDDING_LOOKUP only (FC stay fp32)
  v4 fp16fc_fp16emb  weight-only 16-bit FLOAT_CASTING on FULLY_CONNECTED and EMBEDDING_LOOKUP (the Metal GPU delegate
                     refuses the float16 table)
The FC count is the model's: 186 on 0.8B, 248 on 4B. If the recipe needs calibration the script stops before
quantize(). After export the file is scanned statically (flatbuffers only): FC weight dtype per FC (through a
DEQUANTIZE producer when there is one), DEQUANTIZE count, EMBEDDING_LOOKUP table dtype, BATCH_MATMUL constant operand
dtypes, CUSTOM count, op histogram vs the fp32 input; the expected dtypes are asserted (the json is written first,
then the asserts run)."""
import argparse
import collections
import importlib.metadata
import json
import mmap
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r2_common import FILE_PREFIX, MODEL, RESULT_SUFFIX, K, Clock, add_model_arg, dump_json, sha256_file  # noqa: E402

VARIANTS = {"v1": "v1_wi8fc", "v2": "v2_fp16fc_i8emb", "v3": "v3_i8emb", "v4": "v4_fp16fc_fp16emb"}
FC_COUNT = {"0.8b": 186, "4b": 248}   # 18 x 8 + 6 x 7 and 24 x 8 + 8 x 7 (linear-attention / full-attention layers)
EXPECT = {  # FC weight dtype (after following a DEQUANTIZE producer) for all FCs; table dtype (same rule) of the lookup
    "v1": {"fc_weight": "INT8", "fc_weight_direct": "INT8", "table": "INT8"},
    "v2": {"fc_weight": "FLOAT16", "fc_weight_direct": None, "table": "INT8"},
    "v3": {"fc_weight": "FLOAT32", "fc_weight_direct": "FLOAT32", "table": "INT8"},
    "v4": {"fc_weight": "FLOAT16", "fc_weight_direct": None, "table": "FLOAT16"},
}


def build_recipe(variant):
    """The ai-edge-quantizer recipe of one storage variant (v1 to v4, described at the top of this file)."""
    from ai_edge_quantizer import qtyping, recipe_manager
    from ai_edge_quantizer.algorithm_manager import AlgorithmName

    G = qtyping.QuantGranularity
    OP = qtyping.TFLOperationName
    rm = recipe_manager.RecipeManager()
    if variant == "v1":    # = build_recipe("wi8fc")
        rm.add_dynamic_config(regex=".*", operation_name=OP.FULLY_CONNECTED, num_bits=8)
        rm.add_dynamic_config(regex=".*", operation_name=OP.EMBEDDING_LOOKUP, num_bits=8, granularity=G.CHANNELWISE)
    elif variant == "v2":  # = build_recipe("fp16fc_i8emb")
        rm.add_weight_only_config(regex=".*", operation_name=OP.FULLY_CONNECTED, num_bits=16,
                                  granularity=G.CHANNELWISE, algorithm_key=AlgorithmName.FLOAT_CASTING)
        rm.add_dynamic_config(regex=".*", operation_name=OP.EMBEDDING_LOOKUP, num_bits=8, granularity=G.CHANNELWISE)
    elif variant == "v3":  # = build_recipe("embed_only")
        rm.add_dynamic_config(regex=".*", operation_name=OP.EMBEDDING_LOOKUP, num_bits=8, granularity=G.CHANNELWISE)
    elif variant == "v4":  # = build_recipe("wfp16")
        for op in (OP.FULLY_CONNECTED, OP.EMBEDDING_LOOKUP):
            rm.add_weight_only_config(regex=".*", operation_name=op, num_bits=16, granularity=G.CHANNELWISE,
                                      algorithm_key=AlgorithmName.FLOAT_CASTING)
    else:
        raise ValueError(variant)
    return rm.get_quantization_recipe(), rm.need_calibration()


def quant_scan(path):
    """Static scan of a (quantized) tflite: weight dtypes of FC / EMBEDDING_LOOKUP / BATCH_MATMUL constants."""
    from ai_edge_litert import schema_py_generated as schema
    f = open(path, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    model = schema.Model.GetRootAs(mm, 0)
    op_names = {v: k for k, v in vars(schema.BuiltinOperator).items() if isinstance(v, int)}
    type_names = {v: k for k, v in vars(schema.TensorType).items() if isinstance(v, int)}
    codes = []
    for i in range(model.OperatorCodesLength()):
        c = model.OperatorCodes(i)
        name = op_names.get(max(c.BuiltinCode(), c.DeprecatedBuiltinCode()), "UNKNOWN")
        if name == "CUSTOM":
            name = "CUSTOM:" + (c.CustomCode().decode() if c.CustomCode() else "")
        codes.append(name)
    hist = collections.Counter()
    fc_rows, emb_rows = [], []
    fc_weight = collections.Counter()
    fc_weight_direct = collections.Counter()
    fc_bias = collections.Counter()
    fc_quant_axis = collections.Counter()
    deq_in_out = collections.Counter()
    bmm_const = collections.Counter()
    tensor_dtypes = collections.Counter()
    const_bytes_by_dtype = collections.Counter()
    for gi in range(model.SubgraphsLength()):
        g = model.Subgraphs(gi)
        tensors = [g.Tensors(t) for t in range(g.TensorsLength())]
        ttype = lambda t: type_names.get(tensors[t].Type(), str(tensors[t].Type()))
        shape = lambda t: tensors[t].ShapeAsNumpy().tolist() if tensors[t].ShapeLength() else []
        ops = [g.Operators(o) for o in range(g.OperatorsLength())]
        producer = {}
        for oi, op in enumerate(ops):
            for t in (op.OutputsAsNumpy().tolist() if op.OutputsLength() else []):
                producer[int(t)] = oi
        graph_inputs = set(g.InputsAsNumpy().tolist()) if g.InputsLength() else set()
        is_const = lambda t: t >= 0 and t not in producer and t not in graph_inputs

        def nbytes(t):
            b = model.Buffers(tensors[t].Buffer())
            return b.DataLength() if b.DataLength() > 0 else (b.Size() if b.Offset() > 1 else 0)

        def quant(t):
            q = tensors[t].Quantization()
            if q is None or q.ScaleLength() == 0:
                return None
            return {"scales": q.ScaleLength(), "quantized_dimension": q.QuantizedDimension(),
                    "zero_points_nonzero": int((q.ZeroPointAsNumpy() != 0).sum()) if q.ZeroPointLength() else 0}

        for ti in range(len(tensors)):
            tensor_dtypes[ttype(ti)] += 1
            if is_const(ti):
                const_bytes_by_dtype[ttype(ti)] += nbytes(ti)
        for oi, op in enumerate(ops):
            name = codes[op.OpcodeIndex()]
            hist[name] += 1
            ins = op.InputsAsNumpy().tolist() if op.InputsLength() else []
            if name == "DEQUANTIZE":
                deq_in_out[f"{ttype(ins[0])}->{ttype(op.OutputsAsNumpy()[0])}"] += 1
            if name == "FULLY_CONNECTED":
                w = ins[1]
                direct = ttype(w) if is_const(w) else None
                src = direct
                via = "constant" if is_const(w) else "activation"
                if not is_const(w) and w in producer and codes[ops[producer[w]].OpcodeIndex()] == "DEQUANTIZE":
                    dq_in = ops[producer[w]].InputsAsNumpy()[0]
                    src, via = ttype(dq_in), "DEQUANTIZE(" + ttype(dq_in) + ")"
                fc_weight[src] += 1
                fc_weight_direct[direct] += 1
                q = quant(w) if is_const(w) else None
                if q:
                    fc_quant_axis[f"scales={q['scales']} axis={q['quantized_dimension']}"] += 1
                b = ins[2] if len(ins) > 2 else -1
                fc_bias[ttype(b) if b >= 0 else "none"] += 1
                fc_rows.append({"op_index": oi, "out": tensors[op.OutputsAsNumpy()[0]].Name().decode(),
                                "weight_shape": shape(w), "weight_source_dtype": src, "weight_via": via,
                                "input_dtype": ttype(ins[0]), "weight_quant": q})
            if name == "EMBEDDING_LOOKUP":
                tbl = ins[1]
                src_t, via = tbl, "constant" if is_const(tbl) else "activation"
                if not is_const(tbl) and tbl in producer and codes[ops[producer[tbl]].OpcodeIndex()] == "DEQUANTIZE":
                    src_t = int(ops[producer[tbl]].InputsAsNumpy()[0])
                    via = "DEQUANTIZE(" + ttype(src_t) + ")"
                emb_rows.append({"op_index": oi, "table_dtype": ttype(tbl), "table_shape": shape(tbl),
                                 "table_constant": is_const(tbl), "table_quant": quant(tbl), "table_bytes": nbytes(tbl),
                                 "table_source_dtype": ttype(src_t), "table_via": via,
                                 "table_source_bytes": nbytes(src_t) if is_const(src_t) else None,
                                 "ids_dtype": ttype(ins[0]), "out_dtype": ttype(op.OutputsAsNumpy()[0])})
            if name == "BATCH_MATMUL":
                for t in ins:
                    if is_const(t):
                        bmm_const[ttype(t)] += 1
    mm.close()
    f.close()
    return {
        "operator_count": sum(hist.values()), "op_histogram": dict(sorted(hist.items())),
        "fully_connected": len(fc_rows), "fc_weight_source_dtype": dict(fc_weight),
        "fc_weight_tensor_dtype_direct": {str(k): v for k, v in fc_weight_direct.items()},
        "fc_weight_quant_layout": dict(fc_quant_axis), "fc_bias_dtype": dict(fc_bias),
        "dequantize_count": hist.get("DEQUANTIZE", 0), "dequantize_in_out": dict(deq_in_out),
        "quantize_count": hist.get("QUANTIZE", 0),
        "embedding_lookup": emb_rows,
        "batch_matmul_count": hist.get("BATCH_MATMUL", 0), "batch_matmul_constant_operand_dtype": dict(bmm_const),
        "custom_op_count": sum(v for k, v in hist.items() if k.startswith("CUSTOM")),
        "tensor_dtype_histogram": dict(sorted(tensor_dtypes.items())),
        "constant_bytes_by_dtype": dict(sorted(const_bytes_by_dtype.items())),
        "fc_rows_first5": fc_rows[:5],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    add_model_arg(ap)
    a = ap.parse_args()
    assert a.model == MODEL
    name = VARIANTS[a.variant]
    src = K / f"exports/{FILE_PREFIX}_rowprefill_L{a.L}_fp32.tflite"
    out = K / f"exports/{FILE_PREFIX}_rowprefill_L{a.L}_{name}.tflite"
    res = K / f"results/quant_L{a.L}{RESULT_SUFFIX}_{name}.json"
    assert src.exists(), src
    assert not out.exists() and not res.exists(), f"refusing to overwrite {out.name} / {res.name}"
    clock = Clock()
    started = clock.stamp()
    recipe, need_cal = build_recipe(a.variant)
    print(json.dumps(recipe, indent=1, default=str), flush=True)
    from ai_edge_quantizer import quantizer
    t0 = time.perf_counter()
    qt = quantizer.Quantizer(str(src), recipe)
    load_s = time.perf_counter() - t0
    if qt.need_calibration or need_cal:
        raise SystemExit(f"STOP: recipe {name} needs calibration (need_calibration={qt.need_calibration})")
    t1 = time.perf_counter()
    result = qt.quantize()
    quant_s = time.perf_counter() - t1
    t2 = time.perf_counter()
    result.export_model(str(out))
    export_s = time.perf_counter() - t2
    import resource
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on macOS
    del result, qt
    print(f"quantized {name}: load {load_s:.1f}s quantize {quant_s:.1f}s export {export_s:.1f}s", flush=True)
    t3 = time.perf_counter()
    sha = sha256_file(out)
    scan_q = quant_scan(out)
    scan_src = quant_scan(src)
    scan_s = time.perf_counter() - t3
    exp = EXPECT[a.variant]
    emb = scan_q["embedding_lookup"]
    n_fc = FC_COUNT[MODEL]
    checks = {
        f"fc_count_{n_fc}": scan_q["fully_connected"] == n_fc,
        "fc_weight_all_" + exp["fc_weight"]: scan_q["fc_weight_source_dtype"] == {exp["fc_weight"]: n_fc},
        "embedding_lookup_1_table_" + exp["table"]: len(emb) == 1 and emb[0]["table_source_dtype"] == exp["table"],
        "batch_matmul_constants_FLOAT32_only": set(scan_q["batch_matmul_constant_operand_dtype"]) <= {"FLOAT32"},
        "custom_0": scan_q["custom_op_count"] == 0,
        "op_histogram_unchanged_except_DEQUANTIZE": {k: v for k, v in scan_q["op_histogram"].items() if k != "DEQUANTIZE"}
        == {k: v for k, v in scan_src["op_histogram"].items() if k != "DEQUANTIZE"},
    }
    doc = {
        "step": f"quantize L={a.L} {name}" + (f" (model {MODEL})" if RESULT_SUFFIX else ""),
        "model": MODEL, "started_at": started, "seconds_wall": clock.seconds(),
        "input": {"file": str(src.relative_to(K)), "bytes": src.stat().st_size},
        "output": {"file": str(out.relative_to(K)), "bytes": out.stat().st_size, "sha256": sha},
        "ai_edge_quantizer": importlib.metadata.version("ai-edge-quantizer"),
        "ai_edge_litert": importlib.metadata.version("ai-edge-litert"),
        "recipe": recipe, "need_calibration": bool(need_cal),
        "seconds": {"load": round(load_s, 1), "quantize": round(quant_s, 1), "export": round(export_s, 1),
                    "scan_and_sha": round(scan_s, 1)},
        "peak_rss_bytes_getrusage": int(peak_rss),
        "scan": scan_q,
        "scan_fp32_input": {k: scan_src[k] for k in ("operator_count", "op_histogram", "fully_connected",
                                                     "fc_weight_source_dtype", "dequantize_count", "embedding_lookup",
                                                     "batch_matmul_constant_operand_dtype", "constant_bytes_by_dtype")},
        "checks": checks, "checks_pass": all(checks.values()),
    }
    dump_json(res, doc)
    print(json.dumps({k: doc[k] for k in ("output", "seconds", "checks", "checks_pass")}, indent=1))
    print(json.dumps({k: scan_q[k] for k in ("fc_weight_source_dtype", "fc_weight_tensor_dtype_direct",
                                             "fc_weight_quant_layout", "dequantize_count", "dequantize_in_out",
                                             "batch_matmul_constant_operand_dtype", "constant_bytes_by_dtype")}, indent=1))
    print(json.dumps(emb, indent=1))
    assert doc["checks_pass"], checks


if __name__ == "__main__":
    main()
