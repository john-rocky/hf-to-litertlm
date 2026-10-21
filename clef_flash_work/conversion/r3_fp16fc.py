"""Round 3: the fp16-FC storage form of the tower graphs (ai-edge-quantizer, no calibration) = the row graph's V2 recipe.

build_recipe() and quant_scan() are copied from d1_3b_work/scripts/d1v_storage.py (sha256 1d0687c4fb88b01d..., the
picture graphs' v2 of d1-3B) unchanged except the names:
  v2 fp16fc   weight-only 16-bit FLOAT_CASTING on FULLY_CONNECTED (channelwise): the FC weights are stored FLOAT16 and
              read through a DEQUANTIZE (FLOAT16 -> FLOAT32) in front of each FC; layer norms, attention, softmax and
              GELU stay float32.
fp16fc(src, out) writes the file and returns the record with the checks (FC count unchanged, every FC weight FLOAT16
through one DEQUANTIZE, no table, BATCH_MATMUL constants float32 only, CUSTOM 0, op histogram unchanged but DEQUANTIZE).
The LayerNorm gamma folded into the q / k / v / fc1 weights by the converter is not a bf16 value, so the FLOAT16 cast
can move those weights (the converter folds per-channel scales into FC weights): fold_census() counts, per FC, the weights
whose float16 cast is exact."""
from __future__ import annotations

import collections
import importlib.metadata
import mmap
import time
from pathlib import Path

import numpy as np


def build_recipe():
    from ai_edge_quantizer import qtyping, recipe_manager
    from ai_edge_quantizer.algorithm_manager import AlgorithmName

    G = qtyping.QuantGranularity
    OP = qtyping.TFLOperationName
    rm = recipe_manager.RecipeManager()
    rm.add_weight_only_config(regex=".*", operation_name=OP.FULLY_CONNECTED, num_bits=16,
                              granularity=G.CHANNELWISE, algorithm_key=AlgorithmName.FLOAT_CASTING)
    return rm.get_quantization_recipe(), rm.need_calibration()


def quant_scan(path):
    """d1v_storage.quant_scan (the FC / DEQUANTIZE / BATCH_MATMUL constant dtypes, flatbuffers only)."""
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
    fc_rows = []
    fc_buffers = set()                     # distinct weight buffers of the FCs (FCs that read one constant share it)
    fc_weight, fc_weight_direct = collections.Counter(), collections.Counter()
    deq_in_out, bmm_const, tensor_dtypes, const_bytes = (collections.Counter(), collections.Counter(),
                                                         collections.Counter(), collections.Counter())
    emb = 0
    for gi in range(model.SubgraphsLength()):
        g = model.Subgraphs(gi)
        tensors = [g.Tensors(t) for t in range(g.TensorsLength())]
        ttype = lambda t: type_names.get(tensors[t].Type(), str(tensors[t].Type()))   # noqa: E731
        shape = lambda t: tensors[t].ShapeAsNumpy().tolist() if tensors[t].ShapeLength() else []   # noqa: E731
        ops = [g.Operators(o) for o in range(g.OperatorsLength())]
        producer = {}
        for oi, op in enumerate(ops):
            for t in (op.OutputsAsNumpy().tolist() if op.OutputsLength() else []):
                producer[int(t)] = oi
        graph_inputs = set(g.InputsAsNumpy().tolist()) if g.InputsLength() else set()
        is_const = lambda t: t >= 0 and t not in producer and t not in graph_inputs   # noqa: E731

        def nbytes(t):
            b = model.Buffers(tensors[t].Buffer())
            return b.DataLength() if b.DataLength() > 0 else (b.Size() if b.Offset() > 1 else 0)

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
                src_t = w
                if not is_const(w) and w in producer and codes[ops[producer[w]].OpcodeIndex()] == "DEQUANTIZE":
                    dq_in = ops[producer[w]].InputsAsNumpy()[0]
                    src, via = ttype(dq_in), "DEQUANTIZE(" + ttype(dq_in) + ")"
                    src_t = int(dq_in)
                if src_t >= 0:
                    fc_buffers.add((gi, int(tensors[src_t].Buffer())))
                fc_weight[src] += 1
                fc_weight_direct[direct] += 1
                fc_rows.append({"op_index": oi, "weight_shape": shape(w), "weight_source_dtype": src, "weight_via": via,
                                "input_dtype": ttype(ins[0]), "input_rank": len(shape(ins[0]))})
            if name == "EMBEDDING_LOOKUP":
                emb += 1
            if name == "BATCH_MATMUL":
                for t in ins:
                    if is_const(t):
                        bmm_const[ttype(t)] += 1
    mm.close()
    f.close()
    return {"operator_count": sum(hist.values()), "op_histogram": dict(sorted(hist.items())),
            "fully_connected": len(fc_rows), "fc_weight_buffers": len(fc_buffers), "fc_weight_source_dtype": dict(fc_weight),
            "fc_weight_tensor_dtype_direct": {str(k): v for k, v in fc_weight_direct.items()},
            "dequantize_count": hist.get("DEQUANTIZE", 0), "dequantize_in_out": dict(deq_in_out),
            "embedding_lookup": emb, "batch_matmul_count": hist.get("BATCH_MATMUL", 0),
            "batch_matmul_constant_operand_dtype": dict(bmm_const),
            "custom_op_count": sum(v for k, v in hist.items() if k.startswith("CUSTOM")),
            "tensor_dtype_histogram": dict(sorted(tensor_dtypes.items())),
            "constant_bytes_by_dtype": dict(sorted(const_bytes.items())), "fc_rows_first5": fc_rows[:5],
            "fc_input_ranks": dict(collections.Counter(r["input_rank"] for r in fc_rows))}


def fc_weights(path) -> list[np.ndarray]:
    """The FULLY_CONNECTED weight arrays of a float32 file in op order (constants read from the flatbuffer)."""
    from ai_edge_litert import schema_py_generated as schema
    data = Path(path).read_bytes()
    model = schema.Model.GetRootAs(data, 0)
    op_names = {v: k for k, v in vars(schema.BuiltinOperator).items() if isinstance(v, int)}
    out = []
    g = model.Subgraphs(0)
    for oi in range(g.OperatorsLength()):
        op = g.Operators(oi)
        c = model.OperatorCodes(op.OpcodeIndex())
        if op_names.get(max(c.BuiltinCode(), c.DeprecatedBuiltinCode())) != "FULLY_CONNECTED":
            continue
        t = g.Tensors(op.Inputs(1))
        b = model.Buffers(t.Buffer())
        if b.DataLength() > 0:
            raw = b.DataAsNumpy().tobytes()
        else:
            raw = data[b.Offset(): b.Offset() + b.Size()]
        out.append(np.frombuffer(raw, dtype=np.float32).reshape(t.ShapeAsNumpy().tolist()))
    return out


def fold_census(path) -> dict:
    """Per FC of the float32 file: the share of weights whose float16 cast is exact (a bf16 value always is)."""
    rows = []
    for i, w in enumerate(fc_weights(path)):
        exact = np.asarray(w.astype(np.float16).astype(np.float32) == w)
        rows.append({"fc": i, "shape": list(w.shape), "exact_share": float(exact.mean())})
    tot = sum(int(np.prod(r["shape"])) for r in rows)
    ex = sum(r["exact_share"] * int(np.prod(r["shape"])) for r in rows)
    return {"fc": len(rows), "weights": tot, "fp16_exact": int(round(ex)), "fp16_exact_share": ex / tot if tot else None,
            "fully_exact_fc": sum(r["exact_share"] == 1.0 for r in rows), "rows": rows}


def fp16fc(src: Path, out: Path) -> dict:
    assert not out.exists(), f"refusing to overwrite {out}"
    recipe, need_cal = build_recipe()
    from ai_edge_quantizer import quantizer

    t0 = time.perf_counter()
    qt = quantizer.Quantizer(str(src), recipe)
    if qt.need_calibration or need_cal:
        raise SystemExit(f"STOP: the fp16fc recipe needs calibration (need_calibration={qt.need_calibration})")
    result = qt.quantize()
    result.export_model(str(out))
    quant_s = time.perf_counter() - t0
    scan_q, scan_src = quant_scan(out), quant_scan(src)
    n_fc = scan_src["fully_connected"]
    checks = {
        "fc_count_same_as_input": scan_q["fully_connected"] == n_fc,
        "fc_weight_all_FLOAT16": scan_q["fc_weight_source_dtype"] == {"FLOAT16": n_fc},
        "embedding_lookup_0": scan_q["embedding_lookup"] == 0,
        "batch_matmul_constants_FLOAT32_only": set(scan_q["batch_matmul_constant_operand_dtype"]) <= {"FLOAT32"},
        "custom_0": scan_q["custom_op_count"] == 0,
        "op_histogram_unchanged_except_DEQUANTIZE": {k: v for k, v in scan_q["op_histogram"].items() if k != "DEQUANTIZE"}
        == {k: v for k, v in scan_src["op_histogram"].items() if k != "DEQUANTIZE"},
        # one DEQUANTIZE (FLOAT16 -> FLOAT32) per distinct FC weight buffer (FCs reading one constant share it)
        "dequantize_one_per_fc_weight_buffer_FLOAT16_to_FLOAT32": (
            scan_q["dequantize_count"] == scan_src["fc_weight_buffers"]
            and scan_q["dequantize_in_out"] == {"FLOAT16->FLOAT32": scan_src["fc_weight_buffers"]}),
    }
    return {"variant": "v2_fp16fc", "recipe": recipe, "need_calibration": bool(need_cal),
            "versions": {p: importlib.metadata.version(p) for p in ("ai-edge-quantizer", "ai-edge-litert")},
            "quantize_export_seconds": round(quant_s, 2), "scan": scan_q,
            "scan_fp32_input": {k: scan_src[k] for k in ("operator_count", "op_histogram", "fully_connected",
                                                          "fc_weight_buffers", "fc_weight_source_dtype", "dequantize_count",
                                                          "constant_bytes_by_dtype")},
            "checks": checks, "checks_pass": all(checks.values())}
