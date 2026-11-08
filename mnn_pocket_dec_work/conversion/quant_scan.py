"""Static scan of a (quantized) tflite: weight dtypes of FULLY_CONNECTED / EMBEDDING_LOOKUP / BATCH_MATMUL constants.

Used by quantize_dec.py for its checks (every FULLY_CONNECTED weight of the expected type)."""
import collections
import mmap


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
