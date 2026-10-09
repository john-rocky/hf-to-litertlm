"""Round 6d copy of d1_storage.py (round 6b's file, copied at sha256 917a7fcd279728cd42de13deabc7056eb7bb278c33560e4ac18b32e095a8a139,
HEAD 67d05491): the fp16-FC storage variant of the picture graphs (vision tower, projector) with ai-edge-quantizer (no
calibration).

    $EXPORT scripts/d1v_storage.py --tflite exports/real_vision_tower_fp32.tflite --variant v2 \
        --rtag realv --log-prefix r6d_

The only recipe here is v2 of d1_storage.py without its table part (the picture graphs have no EMBEDDING_LOOKUP):
  v2 fp16fc   weight-only 16-bit FLOAT_CASTING on FULLY_CONNECTED (channelwise): the FC weights are stored FLOAT16 and
              read through a DEQUANTIZE (FLOAT16 -> FLOAT32) in front of each FC; layer norms, attention, softmax and
              GELU stay float32.
No int8 form is made (round 6d: weight-only int8 / int4 are out of scope; dynamic int8 moves calibrated
outputs).
Output = <input dir>/<input stem with _fp32 replaced by _v2_fp16fc>.tflite + results/<rname>_quant.json (Out names of
d1_vision_graph: rname = the output stem with {tag} replaced by {rtag}; never overwritten). After export the file is
scanned statically (flatbuffers only, `quant_scan` = quantize_kev's + the FC weights' channel layout and options): FC
weight dtype per FC (through a DEQUANTIZE producer when there is one), DEQUANTIZE in -> out dtypes, BATCH_MATMUL
constant operand dtypes, CUSTOM count, op histogram vs the float32 input. The json is written first, then the checks
assert.
"""
from __future__ import annotations

import argparse
import collections
import importlib.metadata
import json
import mmap
import sys
import time
from pathlib import Path

K = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from d1_common import sha256_file  # noqa: E402

VARIANTS = {"v2": "v2_fp16fc"}
# FC weight dtype after following a DEQUANTIZE producer / the FC weight tensor's own dtype (None = an activation, i.e.
# a DEQUANTIZE output) / lookup tables expected / DEQUANTIZE ops expected ("fc" = one per FC, FLOAT16 -> FLOAT32)
EXPECT = {"v2": {"fc_weight": "FLOAT16", "fc_weight_direct": None, "tables": 0, "dequantize": "fc"}}


def build_recipe(variant):
    """d1_storage.build_recipe's v2 without the EMBEDDING_LOOKUP line (no table in the picture graphs)."""
    from ai_edge_quantizer import qtyping, recipe_manager
    from ai_edge_quantizer.algorithm_manager import AlgorithmName

    G = qtyping.QuantGranularity
    OP = qtyping.TFLOperationName
    rm = recipe_manager.RecipeManager()
    if variant == "v2":
        rm.add_weight_only_config(regex=".*", operation_name=OP.FULLY_CONNECTED, num_bits=16,
                                  granularity=G.CHANNELWISE, algorithm_key=AlgorithmName.FLOAT_CASTING)
    else:
        raise ValueError(variant)
    return rm.get_quantization_recipe(), rm.need_calibration()


def quant_scan(path):
    """quantize_kev.quant_scan: weight dtypes of FC / EMBEDDING_LOOKUP / BATCH_MATMUL constants (flatbuffers only)."""
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
    fc_weight, fc_weight_direct, fc_quant_axis = collections.Counter(), collections.Counter(), collections.Counter()
    fc_options, fc_channelwise_symmetric = collections.Counter(), 0
    deq_in_out, bmm_const, tensor_dtypes, const_bytes = (collections.Counter(), collections.Counter(),
                                                         collections.Counter(), collections.Counter())
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
                const_bytes[ttype(ti)] += nbytes(ti)
        for oi, op in enumerate(ops):
            name = codes[op.OpcodeIndex()]
            hist[name] += 1
            ins = op.InputsAsNumpy().tolist() if op.InputsLength() else []
            if name == "DEQUANTIZE":
                deq_in_out[f"{ttype(ins[0])}->{ttype(op.OutputsAsNumpy()[0])}"] += 1
            if name == "FULLY_CONNECTED":
                w = ins[1]
                direct = ttype(w) if is_const(w) else None
                src, via = direct, ("constant" if is_const(w) else "activation")
                if not is_const(w) and w in producer and codes[ops[producer[w]].OpcodeIndex()] == "DEQUANTIZE":
                    dq_in = ops[producer[w]].InputsAsNumpy()[0]
                    src, via = ttype(dq_in), "DEQUANTIZE(" + ttype(dq_in) + ")"
                fc_weight[src] += 1
                fc_weight_direct[direct] += 1
                q = quant(w) if is_const(w) else None
                if q:
                    fc_quant_axis[f"scales={q['scales']} axis={q['quantized_dimension']}"] += 1
                    # one scale per output channel (rows of the [out, in] weight), symmetric (all zero points 0)
                    fc_channelwise_symmetric += (q["scales"] == shape(w)[0] and q["quantized_dimension"] == 0
                                                 and q["zero_points_nonzero"] == 0)
                if op.BuiltinOptionsType() == schema.BuiltinOptions.FullyConnectedOptions:
                    o = schema.FullyConnectedOptions()
                    o.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)
                    fc_options[f"asymmetric_quantize_inputs={bool(o.AsymmetricQuantizeInputs())} "
                               f"keep_num_dims={bool(o.KeepNumDims())} output={ttype(op.OutputsAsNumpy()[0])}"] += 1
                fc_rows.append({"op_index": oi, "weight_shape": shape(w), "weight_source_dtype": src, "weight_via": via,
                                "input_dtype": ttype(ins[0]), "input_rank": len(shape(ins[0])), "weight_quant": q})
            if name == "EMBEDDING_LOOKUP":
                tbl = ins[1]
                src_t, via = tbl, ("constant" if is_const(tbl) else "activation")
                if not is_const(tbl) and tbl in producer and codes[ops[producer[tbl]].OpcodeIndex()] == "DEQUANTIZE":
                    src_t = int(ops[producer[tbl]].InputsAsNumpy()[0])
                    via = "DEQUANTIZE(" + ttype(src_t) + ")"
                emb_rows.append({"op_index": oi, "table_dtype": ttype(tbl), "table_shape": shape(tbl),
                                 "table_constant": is_const(tbl), "table_quant": quant(tbl), "table_bytes": nbytes(tbl),
                                 "table_source_dtype": ttype(src_t), "table_via": via,
                                 "ids_dtype": ttype(ins[0]), "out_dtype": ttype(op.OutputsAsNumpy()[0])})
            if name == "BATCH_MATMUL":
                for t in ins:
                    if is_const(t):
                        bmm_const[ttype(t)] += 1
    mm.close()
    f.close()
    return {"operator_count": sum(hist.values()), "op_histogram": dict(sorted(hist.items())),
            "fully_connected": len(fc_rows), "fc_weight_source_dtype": dict(fc_weight),
            "fc_weight_tensor_dtype_direct": {str(k): v for k, v in fc_weight_direct.items()},
            "fc_weight_quant_layout": dict(fc_quant_axis), "fc_weight_channelwise_symmetric": fc_channelwise_symmetric,
            "fc_options": dict(fc_options), "dequantize_count": hist.get("DEQUANTIZE", 0),
            "dequantize_in_out": dict(deq_in_out), "quantize_count": hist.get("QUANTIZE", 0),
            "embedding_lookup": emb_rows, "batch_matmul_count": hist.get("BATCH_MATMUL", 0),
            "batch_matmul_constant_operand_dtype": dict(bmm_const),
            "custom_op_count": sum(v for k, v in hist.items() if k.startswith("CUSTOM")),
            "tensor_dtype_histogram": dict(sorted(tensor_dtypes.items())),
            "constant_bytes_by_dtype": dict(sorted(const_bytes.items())), "fc_rows_first5": fc_rows[:5],
            "fc_input_ranks": dict(collections.Counter(r["input_rank"] for r in fc_rows))}


def main() -> int:
    import d1_vision_graph as VG

    ap = argparse.ArgumentParser()
    ap.add_argument("--tflite", required=True, help="K-relative float32 picture graph (name ends in _fp32.tflite)")
    ap.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    VG.Out.add_args(ap, tag_default=None)
    a = ap.parse_args()
    src = K / a.tflite
    assert src.name.endswith("_fp32.tflite"), src.name
    name = VARIANTS[a.variant]
    out = src.with_name(src.name.replace("_fp32.tflite", f"_{name}.tflite"))
    o = VG.Out.from_args(a, tag=a.tag or src.stem.split("_")[0])
    res = o.result(f"{o.rname(out.stem)}_quant.json")
    assert not out.exists() and not res.exists(), f"refusing to overwrite {out.name} / {res.name}"
    t_all = time.time()
    recipe, need_cal = build_recipe(a.variant)
    from ai_edge_quantizer import quantizer

    t0 = time.perf_counter()
    qt = quantizer.Quantizer(str(src), recipe)
    if qt.need_calibration or need_cal:
        raise SystemExit(f"STOP: recipe {name} needs calibration (need_calibration={qt.need_calibration})")
    result = qt.quantize()
    result.export_model(str(out))
    quant_s = time.perf_counter() - t0
    scan_q, scan_src = quant_scan(out), quant_scan(src)
    exp = EXPECT[a.variant]
    n_fc = scan_src["fully_connected"]
    checks = {
        "fc_count_same_as_input": scan_q["fully_connected"] == n_fc,
        "fc_weight_all_" + exp["fc_weight"]: scan_q["fc_weight_source_dtype"] == {exp["fc_weight"]: n_fc},
        f"embedding_lookup_{exp['tables']}": len(scan_q["embedding_lookup"]) == exp["tables"],
        "batch_matmul_constants_FLOAT32_only": set(scan_q["batch_matmul_constant_operand_dtype"]) <= {"FLOAT32"},
        "custom_0": scan_q["custom_op_count"] == 0,
        "op_histogram_unchanged_except_DEQUANTIZE": {k: v for k, v in scan_q["op_histogram"].items() if k != "DEQUANTIZE"}
        == {k: v for k, v in scan_src["op_histogram"].items() if k != "DEQUANTIZE"},
    }
    if exp["dequantize"] == "fc":   # one DEQUANTIZE (FLOAT16 -> FLOAT32) per FC weight
        checks["dequantize_one_per_fc_FLOAT16_to_FLOAT32"] = (
            scan_q["dequantize_count"] == n_fc and scan_q["dequantize_in_out"] == {"FLOAT16->FLOAT32": n_fc})
    doc = {"what": f"storage variant {name} of {src.name} (round 6d acceptance 1-3; copy of d1_storage.py v2 without "
                   f"the table)", "variant": name, "out_names": o.as_dict(),
           "input": {"file": str(src.relative_to(K)), "bytes": src.stat().st_size, "sha256": sha256_file(src)},
           "output": {"file": str(out.relative_to(K)), "bytes": out.stat().st_size, "sha256": sha256_file(out)},
           "versions": {p: importlib.metadata.version(p) for p in ("ai-edge-quantizer", "ai-edge-litert")},
           "recipe": recipe, "need_calibration": bool(need_cal), "quantize_export_seconds": round(quant_s, 2),
           "scan": scan_q, "scan_fp32_input": {k: scan_src[k] for k in ("operator_count", "op_histogram", "fully_connected",
                                                                         "fc_weight_source_dtype", "dequantize_count",
                                                                         "embedding_lookup", "constant_bytes_by_dtype")},
           "checks": checks, "checks_pass": all(checks.values()), "seconds_wall": round(time.time() - t_all, 1)}
    res.write_text(json.dumps(doc, indent=1, default=str) + "\n")
    print(json.dumps({k: doc[k] for k in ("output", "checks", "checks_pass")}, indent=1))
    print(json.dumps({k: scan_q[k] for k in ("fc_weight_source_dtype", "fc_weight_tensor_dtype_direct", "fc_weight_quant_layout",
                                             "fc_weight_channelwise_symmetric", "fc_options", "dequantize_count",
                                             "dequantize_in_out", "constant_bytes_by_dtype", "fc_input_ranks")}, indent=1))
    assert doc["checks_pass"], checks
    return 0


if __name__ == "__main__":
    sys.exit(main())
