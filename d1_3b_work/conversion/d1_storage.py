"""Round 2 acceptance 6 / round 5 acceptance 1: storage variants of a float32 row graph with ai-edge-quantizer (no
calibration).

    $EXPORT scripts/d1_storage.py --tflite exports/tiny_rowprefill_L64_fp32.tflite --variant v2
    $EXPORT scripts/d1_storage.py --tflite exports/tiny_rowprefill_L64_fp32.tflite --variant v3
    $EXPORT scripts/d1_storage.py --tflite exports/tiny_rowprefill_L64_fp32.tflite --variant v1

The recipes are Kev-0.8B LiteRT's quantize_kev.py's (from an earlier build_recipe); norms, conv, softmax and
attention stay float32:
  v1 wi8fc         dynamic int8 FULLY_CONNECTED (channelwise, symmetric, integer compute: the FC weights are stored int8
                   and read by the FC itself, no DEQUANTIZE) + dynamic int8 EMBEDDING_LOOKUP (channelwise) = Kev's
                   `v1_wi8fc` = decider's `wi8fc`; the one form the round-4 estimate fits on the S26 CPU. Its option
                   probabilities move (activation quantization moves calibrated probabilities), so it
                   goes to a phone only after the |dp| gate on the real weights
  v2 fp16fc_i8emb  weight-only 16-bit FLOAT_CASTING on FULLY_CONNECTED (channelwise) + dynamic int8 EMBEDDING_LOOKUP
                   (channelwise) = Kev's shipped form, this conversion's initial candidate
  v3 i8emb         dynamic int8 EMBEDDING_LOOKUP only (FC weights stay float32) = the float32 reference a GPU accepts
  v2e fp16fc       (round 6c) weight-only 16-bit FLOAT_CASTING on FULLY_CONNECTED only, for the embeds graph
                   (D1PrefillEmbeds: no table in the graph, the host writes float32 rows of the bfloat16 table)
  v2 --pair        (round 9) the shared-state pair file (scripts/d1_shared_state.py: state_prefill + question_step in
                   one file): the v2 rules + one later rule (later rules win in ai-edge-quantizer) that leaves the
                   FULLY_CONNECTED ops whose scope matches `RopeSelect` unquantized (NO_QUANTIZE), so the question
                   step's RoPE table stays float32 and its one-hot selection exact (Kev r10's quantize_shared_state.py).
                   Checks: every FC reads FLOAT16 weights except the RopeSelect ones (FLOAT32, at least one), every
                   EMBEDDING_LOOKUP table INT8, the rest as v2; the size beside the row form's v2 file
                   (exports/real_rowprefill_L256_v2_fp16fc_i8emb.tflite) shows whether the two signatures keep one
                   copy of the weights.
  v2e --pair       (round 10) the embeds pair file (scripts/d1_shared_state.py --embeds: no table in the graph): the
                   v2e rule + the same NO_QUANTIZE rule on `RopeSelect`. Checks: every FC FLOAT16 except the RopeSelect
                   ones (FLOAT32), no EMBEDDING_LOOKUP; the size beside the row form's v2e L256 file
                   (exports/real_rowprefill_embeds_L256_v2e_fp16fc.tflite).
Weight-only int8 (DEQUANTIZE -> float FC) is not made here: CPU-only and refused by the GPU, not a shipping form.
Output = exports/<input stem with _fp32 replaced by _{v1_wi8fc|v2_fp16fc_i8emb|v3_i8emb}>.tflite + results/<same
stem>_quant.json (never overwritten). After export the file is scanned statically (flatbuffers only, `quant_scan` =
quantize_kev's + the FC weights' channel layout and options): FC weight dtype per FC (through a DEQUANTIZE producer when
there is one), DEQUANTIZE in -> out dtypes, EMBEDDING_LOOKUP table dtype, BATCH_MATMUL constant operand dtypes, CUSTOM
count, op histogram vs the float32 input. The FC count is the input file's (34 on the tiny model, 22 x 5 + 8 x 7 = 166
on d1-3B). The json is written first, then the checks assert.
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

VARIANTS = {"v1": "v1_wi8fc", "v2": "v2_fp16fc_i8emb", "v3": "v3_i8emb", "v2e": "v2e_fp16fc"}
# FC weight dtype after following a DEQUANTIZE producer / the FC weight tensor's own dtype (None = an activation, i.e.
# a DEQUANTIZE output) / the lookup table's dtype (None = the graph has no EMBEDDING_LOOKUP) / DEQUANTIZE ops expected
# (None = not checked)
EXPECT = {"v1": {"fc_weight": "INT8", "fc_weight_direct": "INT8", "table": "INT8", "dequantize": 0},
          "v2": {"fc_weight": "FLOAT16", "fc_weight_direct": None, "table": "INT8", "dequantize": None},
          "v3": {"fc_weight": "FLOAT32", "fc_weight_direct": "FLOAT32", "table": "INT8", "dequantize": 0},
          "v2e": {"fc_weight": "FLOAT16", "fc_weight_direct": None, "table": None, "dequantize": None}}
KEV_V1_RESULT = K / "external/kev/quant_L1024_v1_wi8fc.json"   # read-only: the recipe Kev's v1 recorded
ROPE_SCOPE = "RopeSelect"                    # round 9: the pair's RoPE table FC (d1_shared_state.RopeSelect)
ROW_V2_L256 = K / "exports/real_rowprefill_L256_v2_fp16fc_i8emb.tflite"
ROW_V2E_L256 = K / "exports/real_rowprefill_embeds_L256_v2e_fp16fc.tflite"     # round 10: the embeds pair's twin


def fc_weights(path):
    """round 9: [(output tensor name, weight source dtype)] of every FULLY_CONNECTED, through a DEQUANTIZE producer
    (Kev r10's quantize_shared_state.fc_weights)."""
    from ai_edge_litert import schema_py_generated as schema
    f = open(path, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    model = schema.Model.GetRootAs(mm, 0)
    names = {v: k for k, v in vars(schema.BuiltinOperator).items() if isinstance(v, int)}
    types = {v: k for k, v in vars(schema.TensorType).items() if isinstance(v, int)}
    codes = [names.get(max(model.OperatorCodes(i).BuiltinCode(), model.OperatorCodes(i).DeprecatedBuiltinCode()))
             for i in range(model.OperatorCodesLength())]
    out = []
    for gi in range(model.SubgraphsLength()):
        g = model.Subgraphs(gi)
        ops = [g.Operators(o) for o in range(g.OperatorsLength())]
        producer = {int(t): op for op in ops for t in (op.OutputsAsNumpy() if op.OutputsLength() else [])}
        for op in ops:
            if codes[op.OpcodeIndex()] != "FULLY_CONNECTED":
                continue
            w = int(op.InputsAsNumpy()[1])
            src = w
            if w in producer and codes[producer[w].OpcodeIndex()] == "DEQUANTIZE":
                src = int(producer[w].InputsAsNumpy()[0])
            out.append((g.Tensors(int(op.OutputsAsNumpy()[0])).Name().decode(), types.get(g.Tensors(src).Type())))
    mm.close()
    f.close()
    return out


def build_recipe(variant, keep_rope=False):
    """quantize_kev.build_recipe for v1 / v2 / v3 (v1's FC granularity is written out: CHANNELWISE is the default of
    add_dynamic_config in 0.9.0, which Kev's call relied on). keep_rope (round 9, v2 only): + NO_QUANTIZE for the
    FULLY_CONNECTED ops in the RopeSelect scope, after the v2 rules (the later rule wins)."""
    from ai_edge_quantizer import qtyping, recipe_manager
    from ai_edge_quantizer.algorithm_manager import AlgorithmName

    G = qtyping.QuantGranularity
    OP = qtyping.TFLOperationName
    rm = recipe_manager.RecipeManager()
    if variant == "v1":
        rm.add_dynamic_config(regex=".*", operation_name=OP.FULLY_CONNECTED, num_bits=8, granularity=G.CHANNELWISE)
        rm.add_dynamic_config(regex=".*", operation_name=OP.EMBEDDING_LOOKUP, num_bits=8, granularity=G.CHANNELWISE)
    elif variant == "v2":
        rm.add_weight_only_config(regex=".*", operation_name=OP.FULLY_CONNECTED, num_bits=16,
                                  granularity=G.CHANNELWISE, algorithm_key=AlgorithmName.FLOAT_CASTING)
        rm.add_dynamic_config(regex=".*", operation_name=OP.EMBEDDING_LOOKUP, num_bits=8, granularity=G.CHANNELWISE)
    elif variant == "v3":
        rm.add_dynamic_config(regex=".*", operation_name=OP.EMBEDDING_LOOKUP, num_bits=8, granularity=G.CHANNELWISE)
    elif variant == "v2e":
        rm.add_weight_only_config(regex=".*", operation_name=OP.FULLY_CONNECTED, num_bits=16,
                                  granularity=G.CHANNELWISE, algorithm_key=AlgorithmName.FLOAT_CASTING)
    else:
        raise ValueError(variant)
    if keep_rope:
        assert variant in ("v2", "v2e"), "--pair is the v2 (ids pair) or v2e (embeds pair, round 10) recipe"
        rm.add_quantization_config(regex=ROPE_SCOPE, operation_name=OP.FULLY_CONNECTED,
                                   algorithm_key=AlgorithmName.NO_QUANTIZE)
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--tflite", required=True, help="K-relative float32 row graph (name ends in _fp32.tflite)")
    ap.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    ap.add_argument("--pair", action="store_true", help="round 9: the shared-state pair file (v2 + RopeSelect kept "
                                                         "float32; module docstring)")
    a = ap.parse_args()
    src = K / a.tflite
    assert src.name.endswith("_fp32.tflite"), src.name
    assert not a.pair or (a.variant == "v2" and "sharedstate" in src.name and "sharedstate_embeds" not in src.name) or \
        (a.variant == "v2e" and "sharedstate_embeds" in src.name), \
        "--pair = v2 of a real_sharedstate_* file or v2e of a real_sharedstate_embeds_* file (round 10)"
    name = VARIANTS[a.variant]
    out = src.with_name(src.name.replace("_fp32.tflite", f"_{name}.tflite"))
    res = K / f"results/{out.stem}_quant.json"
    assert not out.exists() and not res.exists(), f"refusing to overwrite {out.name} / {res.name}"
    t_all = time.time()
    recipe, need_cal = build_recipe(a.variant, keep_rope=a.pair)
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
    emb = scan_q["embedding_lookup"]
    n_fc = scan_src["fully_connected"]
    checks = {
        "fc_count_same_as_input": scan_q["fully_connected"] == n_fc,
        "fc_weight_all_" + exp["fc_weight"]: scan_q["fc_weight_source_dtype"] == {exp["fc_weight"]: n_fc},
        ("embedding_lookup_1_table_" + exp["table"] if exp["table"] else "embedding_lookup_0"):
            (len(emb) == 1 and emb[0]["table_source_dtype"] == exp["table"]) if exp["table"] else len(emb) == 0,
        "batch_matmul_constants_FLOAT32_only": set(scan_q["batch_matmul_constant_operand_dtype"]) <= {"FLOAT32"},
        "custom_0": scan_q["custom_op_count"] == 0,
        "op_histogram_unchanged_except_DEQUANTIZE": {k: v for k, v in scan_q["op_histogram"].items() if k != "DEQUANTIZE"}
        == {k: v for k, v in scan_src["op_histogram"].items() if k != "DEQUANTIZE"},
    }
    if exp["fc_weight_direct"]:   # the FC reads its weight tensor itself (no DEQUANTIZE in between)
        checks["fc_weight_direct_all_" + exp["fc_weight_direct"]] = (
            scan_q["fc_weight_tensor_dtype_direct"] == {exp["fc_weight_direct"]: n_fc})
    if exp["dequantize"] is not None:
        checks[f"dequantize_{exp['dequantize']}"] = scan_q["dequantize_count"] == exp["dequantize"]
    if a.variant == "v1":         # int8 FC weights: one scale per output channel, zero points 0
        checks["fc_weight_channelwise_symmetric_all"] = scan_q["fc_weight_channelwise_symmetric"] == n_fc
    pair = None
    if a.pair:                    # round 9: the RopeSelect FCs stay float32; one table per signature, all INT8
        fcs = fc_weights(out)
        rope = [(n, d) for n, d in fcs if ROPE_SCOPE in n]
        other = [(n, d) for n, d in fcs if ROPE_SCOPE not in n]
        del checks["fc_weight_all_" + exp["fc_weight"]]
        checks.update({
            "fc_weight_all_FLOAT16_except_rope": bool(other) and all(d == "FLOAT16" for _, d in other),
            "rope_fc_FLOAT32": bool(rope) and all(d == "FLOAT32" for _, d in rope)})
        if exp["table"]:
            del checks["embedding_lookup_1_table_" + exp["table"]]
            checks["embedding_lookup_tables_INT8"] = bool(emb) and all(e["table_source_dtype"] == "INT8" for e in emb)
        twin = ROW_V2_L256 if exp["table"] else ROW_V2E_L256
        pair = {"rope_fully_connected": [{"output": n, "weight_dtype": d} for n, d in rope],
                "other_fc_weight_dtypes": dict(collections.Counter(d for _, d in other)),
                "embedding_lookups": len(emb),
                "row_form_twin_L256": {"file": str(twin.relative_to(K)), "bytes": twin.stat().st_size}
                if twin.exists() else None,
                "bytes_minus_row_form_twin_L256": out.stat().st_size - twin.stat().st_size if twin.exists() else None}
        if exp["table"]:      # round 9's key names, kept for the ids pair
            pair["row_form_v2_L256"] = pair.pop("row_form_twin_L256")
            pair["bytes_minus_row_form_v2_L256"] = pair.pop("bytes_minus_row_form_twin_L256")
    reference = None
    if a.variant == "v1" and KEV_V1_RESULT.exists():
        kev_recipe = json.loads(KEV_V1_RESULT.read_text())["recipe"]
        reference = {"file": str(KEV_V1_RESULT.relative_to(K.parent)),
                     "recipe_equal": json.loads(json.dumps(recipe, default=str)) == kev_recipe}
    doc = {"what": f"storage variant {name} of {src.name} (round 2 acceptance 6; v1 = round 5 acceptance 1)",
           "variant": name, "reference_recipe": reference,
           "input": {"file": str(src.relative_to(K)), "bytes": src.stat().st_size, "sha256": sha256_file(src)},
           "output": {"file": str(out.relative_to(K)), "bytes": out.stat().st_size, "sha256": sha256_file(out)},
           "versions": {p: importlib.metadata.version(p) for p in ("ai-edge-quantizer", "ai-edge-litert")},
           "recipe": recipe, "need_calibration": bool(need_cal), "quantize_export_seconds": round(quant_s, 2),
           "scan": scan_q, "scan_fp32_input": {k: scan_src[k] for k in ("operator_count", "op_histogram", "fully_connected",
                                                                         "fc_weight_source_dtype", "dequantize_count",
                                                                         "embedding_lookup", "constant_bytes_by_dtype")},
           "checks": checks, "checks_pass": all(checks.values()), "seconds_wall": round(time.time() - t_all, 1)}
    if pair is not None:
        doc["pair"] = pair
    res.write_text(json.dumps(doc, indent=1, default=str) + "\n")
    print(json.dumps({k: doc[k] for k in ("output", "checks", "checks_pass")}, indent=1))
    print(json.dumps({k: scan_q[k] for k in ("fc_weight_source_dtype", "fc_weight_tensor_dtype_direct", "fc_weight_quant_layout",
                                             "fc_weight_channelwise_symmetric", "fc_options", "dequantize_count",
                                             "dequantize_in_out", "constant_bytes_by_dtype", "fc_input_ranks")}, indent=1))
    if reference:
        print(json.dumps({"reference_recipe": reference}))
    print(json.dumps(emb, indent=1))
    assert doc["checks_pass"], checks
    return 0


if __name__ == "__main__":
    sys.exit(main())
